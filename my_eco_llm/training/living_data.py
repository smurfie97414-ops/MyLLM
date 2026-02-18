from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import Iterator

import torch


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab
        self.eos_token_id = int(getattr(self._enc, "eot_token", 100257))

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self._enc.encode_ordinary_batch(texts)

    def decode(self, token_ids: list[int]) -> str:
        return self._enc.decode(token_ids)


@dataclass
class LivingSeedDataConfig:
    fineweb_name: str = "HuggingFaceFW/fineweb-edu"
    fineweb_subset: str = "sample-10BT"
    fineweb_split: str = "train"
    # Practical proxy for top-1% quality in fineweb-edu.
    min_score: float = 4.8
    seq_len: int = 512
    batch_size: int = 4
    seed: int = 1337
    min_text_chars: int = 64
    max_text_chars: int = 8192
    shuffle_buffer: int = 20000
    max_stream_retries: int = 8
    retry_backoff_sec: float = 2.0
    hf_token: str | None = None
    hf_token_env_var: str = "HF_TOKEN"
    disable_hf_cache: bool = True


def build_living_tokenizer(name: str = "cl100k_base") -> TiktokenTokenizer:
    return TiktokenTokenizer(name)


def _resolve_hf_token(config: LivingSeedDataConfig) -> str | None:
    if config.hf_token:
        return config.hf_token
    env_key = config.hf_token_env_var.strip()
    if env_key:
        token = os.environ.get(env_key)
        if token:
            return token
    return os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _extract_text(sample: dict) -> str | None:
    for key in ("text", "content", "markdown", "raw_content", "document"):
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def stream_fineweb_seed_batches(
    tokenizer: TiktokenTokenizer,
    config: LivingSeedDataConfig,
) -> Iterator[dict[str, torch.Tensor]]:
    from datasets import disable_caching, load_dataset

    if config.disable_hf_cache:
        try:
            disable_caching()
        except Exception:
            pass
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    seq_plus_one = config.seq_len + 1
    rng = random.Random(config.seed)
    retries = 0

    while True:
        try:
            token = _resolve_hf_token(config)
            ds = load_dataset(
                config.fineweb_name,
                name=config.fineweb_subset,
                split=config.fineweb_split,
                streaming=True,
                token=token,
            )
            ds = ds.filter(lambda x: float(x.get("score", 0.0)) >= config.min_score)
            if config.shuffle_buffer > 0:
                ds = ds.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer)

            batch_inputs: list[torch.Tensor] = []
            batch_labels: list[torch.Tensor] = []
            tok_buffer: list[int] = []
            cursor = 0

            text_batch: list[str] = []
            tokenize_batch_size = 64

            def flush_texts(texts: list[str]) -> Iterator[dict[str, torch.Tensor]]:
                nonlocal tok_buffer, cursor, batch_inputs, batch_labels
                if not texts:
                    return
                tokenized = tokenizer.encode_batch(texts)
                for ids in tokenized:
                    if not ids:
                        continue
                    tok_buffer.extend(ids)
                    tok_buffer.append(tokenizer.eos_token_id)

                    while (len(tok_buffer) - cursor) >= seq_plus_one:
                        chunk = tok_buffer[cursor : cursor + seq_plus_one]
                        cursor += seq_plus_one
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)
                        batch_inputs.append(input_ids)
                        batch_labels.append(labels)

                        if len(batch_inputs) == config.batch_size:
                            yield {
                                "input_ids": torch.stack(batch_inputs, dim=0),
                                "labels": torch.stack(batch_labels, dim=0),
                            }
                            batch_inputs.clear()
                            batch_labels.clear()

                    if cursor >= 4096:
                        tok_buffer = tok_buffer[cursor:]
                        cursor = 0

            for sample in ds:
                text = _extract_text(sample)
                if text is None:
                    continue
                if len(text) < config.min_text_chars:
                    continue
                if len(text) > config.max_text_chars:
                    span = len(text) - config.max_text_chars
                    start = rng.randint(0, span) if span > 0 else 0
                    text = text[start : start + config.max_text_chars]
                text_batch.append(text)
                if len(text_batch) >= tokenize_batch_size:
                    yield from flush_texts(text_batch)
                    text_batch.clear()

            if text_batch:
                yield from flush_texts(text_batch)
                text_batch.clear()

            retries = 0
        except Exception as exc:
            retries += 1
            if retries > config.max_stream_retries:
                raise RuntimeError(
                    f"FineWeb stream failed after {config.max_stream_retries} retries."
                ) from exc
            wait_s = config.retry_backoff_sec * retries
            print(
                f"[living-data] seed stream error retry={retries}/{config.max_stream_retries} "
                f"wait={wait_s:.1f}s error={exc}"
            )
            time.sleep(wait_s)
