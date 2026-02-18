from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import gzip
import hashlib
import json
import os
import queue
import random
import threading
from typing import Any, Iterable
import time
from urllib import error as urlerror
from urllib import request as urlrequest
import warnings

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


@dataclass
class StreamingDataConfig:
    data_backend: str = "hybrid"  # hf | commoncrawl | hybrid
    fineweb_name: str = "HuggingFaceFW/fineweb-edu"
    fineweb_subset: str = "sample-10BT"
    fineweb_split: str = "train"
    cosmopedia_name: str = "HuggingFaceTB/cosmopedia"
    cosmopedia_config: str = "web_samples_v2"
    cosmopedia_split: str = "train"
    fineweb_ratio: float = 0.7
    min_score: float = 3.0
    seed: int = 42
    max_stream_retries: int = 5
    retry_backoff_sec: float = 2.0
    allow_synthetic_fallback: bool = False
    min_text_chars: int = 64
    max_text_chars: int = 8192
    hf_token: str | None = None
    hf_token_env_var: str = "HF_TOKEN"
    sample_prefetch: int = 1024
    source_prefetch: int = 128
    source_read_timeout_sec: float = 20.0
    tokenize_batch_size: int = 64
    stream_shuffle_buffer: int = 10000
    disable_hf_cache: bool = True
    use_hf_transfer: bool = True
    synthetic_backfill_on_error: bool = False
    synthetic_backfill_batches: int = 32
    synthetic_mix_ratio: float = 0.0
    bootstrap_seed_batches: int = 64
    max_source_failovers: int = 64
    commoncrawl_ratio: float = 0.5
    commoncrawl_index_url: str = "https://index.commoncrawl.org/collinfo.json"
    commoncrawl_data_url_prefix: str = "https://data.commoncrawl.org"
    commoncrawl_crawl_ids: str = ""
    commoncrawl_latest_crawls: int = 2
    commoncrawl_wet_paths_per_crawl: int = 256
    commoncrawl_records_per_file: int = 256
    commoncrawl_max_files_per_worker_cycle: int = 64
    commoncrawl_parallel_files: int = 4
    commoncrawl_file_retries: int = 2
    commoncrawl_http_timeout_sec: int = 45
    commoncrawl_user_agent: str = "EcoReasoningGPT/0.1 (+https://commoncrawl.org)"
    commoncrawl_min_alpha_ratio: float = 0.20
    commoncrawl_max_newline_ratio: float = 0.35
    commoncrawl_min_unique_char_ratio: float = 0.06
    dedup_window_size: int = 65536
    dedup_normalize_chars: int = 4096
    hybrid_warmstart_hf_samples: int = 1024


class _AsyncIterator:
    def __init__(self, source: Iterable[dict[str, Any]], prefetch: int) -> None:
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1, int(prefetch)))
        self._sentinel = object()
        self._thread = threading.Thread(target=self._producer, args=(source,), daemon=True)
        self._thread.start()

    def _producer(self, source: Iterable[dict[str, Any]]) -> None:
        try:
            for item in source:
                self._queue.put(item)
        except Exception as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(self._sentinel)

    def next(self, timeout_sec: float) -> dict[str, Any]:
        timeout = max(0.01, float(timeout_sec))
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"source read timeout ({timeout:.2f}s)") from exc
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, dict):
            raise RuntimeError(f"Unexpected source item type: {type(item)!r}")
        return item


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab
        self.eos_token_id = getattr(self._enc, "eot_token", 100257)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self._enc.encode_ordinary_batch(texts)

    def decode(self, token_ids: list[int]) -> str:
        return self._enc.decode(token_ids)


class HFTokenizer:
    def __init__(self, tokenizer_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self._tok.eos_token_id is None:
            raise ValueError(f"Tokenizer '{tokenizer_name}' has no eos_token_id.")
        self.vocab_size = int(self._tok.vocab_size)
        self.eos_token_id = int(self._tok.eos_token_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        encoded = self._tok(texts, add_special_tokens=False, return_attention_mask=False)
        return encoded["input_ids"]

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)


def build_tokenizer(tokenizer_backend: str = "tiktoken", tokenizer_name: str = "cl100k_base") -> Any:
    if tokenizer_backend == "tiktoken":
        return TiktokenTokenizer(tokenizer_name)
    if tokenizer_backend == "hf":
        return HFTokenizer(tokenizer_name)
    raise ValueError(f"Unsupported tokenizer backend: {tokenizer_backend}")


class InterleavedTokenStreamDataset(IterableDataset):
    SOURCE_ID_MAP = {
        "unknown": 0,
        "fineweb": 1,
        "cosmopedia": 2,
        "commoncrawl": 3,
        "synthetic": 4,
    }

    def __init__(
        self,
        tokenizer: Any,
        seq_len: int,
        data_config: StreamingDataConfig,
        force_synthetic: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_config = data_config
        self.force_synthetic = force_synthetic
        self._cached_cc_ids: list[str] | None = None
        self._recent_fingerprints: OrderedDict[bytes, None] = OrderedDict()

    def _extract_text(self, sample: dict[str, Any]) -> str | None:
        for key in ("text", "content", "markdown", "raw_content", "document"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _sample_source_id(self, sample: dict[str, Any]) -> int:
        src = sample.get("_eco_source")
        if not isinstance(src, str):
            return self.SOURCE_ID_MAP["unknown"]
        return self.SOURCE_ID_MAP.get(src.strip().lower(), self.SOURCE_ID_MAP["unknown"])

    def _passes_commoncrawl_quality_gate(self, text: str) -> bool:
        cfg = self.data_config
        if not text:
            return False
        length = len(text)
        if length < cfg.min_text_chars:
            return False

        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / max(length, 1)
        if alpha_ratio < float(cfg.commoncrawl_min_alpha_ratio):
            return False

        newline_ratio = text.count("\n") / max(length, 1)
        if newline_ratio > float(cfg.commoncrawl_max_newline_ratio):
            return False

        unique_window = min(length, 1024)
        unique_char_ratio = len(set(text[:unique_window])) / max(unique_window, 1)
        if unique_char_ratio < float(cfg.commoncrawl_min_unique_char_ratio):
            return False
        return True

    def _dedup_accept(self, text: str, source_id: int) -> bool:
        cfg = self.data_config
        if source_id == self.SOURCE_ID_MAP["synthetic"]:
            return True
        window = max(0, int(cfg.dedup_window_size))
        if window <= 0:
            return True
        take = max(128, int(cfg.dedup_normalize_chars))
        normalized = " ".join(text[:take].lower().split())
        fp = hashlib.blake2b(normalized.encode("utf-8", errors="ignore"), digest_size=8).digest()
        if fp in self._recent_fingerprints:
            self._recent_fingerprints.move_to_end(fp, last=True)
            return False
        self._recent_fingerprints[fp] = None
        if len(self._recent_fingerprints) > window:
            self._recent_fingerprints.popitem(last=False)
        return True

    def _tag_stream(self, samples: Iterable[dict[str, Any]], source_name: str) -> Iterable[dict[str, Any]]:
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            tagged = dict(sample)
            tagged["_eco_source"] = source_name
            yield tagged

    def _build_cosmopedia_stream(self):
        from datasets import load_dataset

        cfg = self.data_config
        token = self._resolve_hf_token()
        return load_dataset(
            cfg.cosmopedia_name,
            cfg.cosmopedia_config,
            split=cfg.cosmopedia_split,
            streaming=True,
            token=token,
        )

    def _resolve_hf_token(self) -> str | None:
        cfg = self.data_config
        if cfg.hf_token:
            return cfg.hf_token
        env_key = cfg.hf_token_env_var.strip()
        if env_key:
            token = os.environ.get(env_key)
            if token:
                return token
        return os.environ.get("HUGGINGFACE_HUB_TOKEN")

    def _maybe_configure_hf_runtime(self) -> None:
        cfg = self.data_config
        if cfg.use_hf_transfer:
            # HF Hub now routes large-file acceleration through Xet; keep the legacy
            # flag too for older environments.
            os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        if cfg.disable_hf_cache:
            try:
                from datasets import disable_caching

                disable_caching()
            except Exception:
                pass

    def _prefetch_samples(self, samples: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        buffer_size = max(0, int(self.data_config.sample_prefetch))
        if buffer_size <= 0:
            yield from samples
            return

        q: queue.Queue[object] = queue.Queue(maxsize=buffer_size)
        sentinel = object()

        def producer() -> None:
            try:
                for item in samples:
                    q.put(item)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]

    def _probabilistic_interleave_two(
        self,
        first: Iterable[dict[str, Any]],
        second: Iterable[dict[str, Any]],
        first_ratio: float,
        *,
        first_name: str,
        second_name: str,
        non_blocking: bool = False,
    ) -> Iterable[dict[str, Any]]:
        cfg = self.data_config
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(cfg.seed + (worker_id * 1009))
        first_ratio = min(max(float(first_ratio), 0.0), 1.0)
        max_failovers = max(0, int(cfg.max_source_failovers))
        failovers = 0
        if not non_blocking:
            first_iter = iter(first)
            second_iter = iter(second)
            while True:
                use_first = rng.random() < first_ratio
                primary_iter = first_iter if use_first else second_iter
                secondary_iter = second_iter if use_first else first_iter
                try:
                    yield next(primary_iter)
                    continue
                except StopIteration:
                    return
                except Exception as primary_exc:
                    try:
                        yield next(secondary_iter)
                        failovers += 1
                        if failovers > max_failovers:
                            raise RuntimeError(
                                f"Exceeded max_source_failovers={max_failovers} while interleaving "
                                f"{first_name} and {second_name} streams."
                            ) from primary_exc
                    except StopIteration:
                        return
                    except Exception as secondary_exc:
                        raise RuntimeError(
                            f"Both interleaved streams failed on sample fetch: {first_name}, {second_name}."
                        ) from secondary_exc
            return
        timeout_sec = max(0.05, float(cfg.source_read_timeout_sec))
        prefetch = max(1, int(cfg.source_prefetch))
        first_iter = _AsyncIterator(first, prefetch=prefetch)
        second_iter = _AsyncIterator(second, prefetch=prefetch)
        warmstart = max(0, int(cfg.hybrid_warmstart_hf_samples)) if first_name == "hf" else 0
        warm_emitted = 0
        while True:
            use_first = (warm_emitted < warmstart) or (rng.random() < first_ratio)
            primary_iter = first_iter if use_first else second_iter
            secondary_iter = second_iter if use_first else first_iter
            from_first = bool(use_first)

            try:
                item = primary_iter.next(timeout_sec=timeout_sec)
                if from_first:
                    warm_emitted += 1
                yield item
                continue
            except StopIteration:
                try:
                    yield secondary_iter.next(timeout_sec=timeout_sec)
                    continue
                except StopIteration:
                    return
                except Exception as secondary_exc:
                    raise RuntimeError(
                        f"Primary source exhausted and secondary source failed for {first_name}/{second_name}."
                    ) from secondary_exc
            except TimeoutError:
                try:
                    yield secondary_iter.next(timeout_sec=timeout_sec)
                    continue
                except StopIteration:
                    continue
                except TimeoutError:
                    continue
                except Exception:
                    continue
            except Exception as primary_exc:
                try:
                    yield secondary_iter.next(timeout_sec=timeout_sec)
                    failovers += 1
                    if failovers > max_failovers:
                        raise RuntimeError(
                            f"Exceeded max_source_failovers={max_failovers} while interleaving "
                            f"{first_name} and {second_name} streams."
                        ) from primary_exc
                except StopIteration:
                    return
                except TimeoutError:
                    failovers += 1
                    if failovers > max_failovers:
                        raise RuntimeError(
                            f"Exceeded max_source_failovers={max_failovers} while interleaving "
                            f"{first_name} and {second_name} streams."
                        ) from primary_exc
                    continue
                except Exception as secondary_exc:
                    failovers += 1
                    if failovers > max_failovers:
                        raise RuntimeError(
                            f"Both interleaved streams failed on sample fetch: {first_name}, {second_name}."
                        ) from secondary_exc
                    continue

    def _build_hf_stream(self) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        cfg = self.data_config
        token = self._resolve_hf_token()
        fineweb = load_dataset(
            cfg.fineweb_name,
            name=cfg.fineweb_subset,
            split=cfg.fineweb_split,
            streaming=True,
            token=token,
        )
        fineweb = fineweb.filter(lambda x: float(x.get("score", 0.0)) >= cfg.min_score)
        cosmopedia = self._build_cosmopedia_stream()

        if cfg.stream_shuffle_buffer > 0:
            fineweb = fineweb.shuffle(seed=cfg.seed, buffer_size=cfg.stream_shuffle_buffer)
            cosmopedia = cosmopedia.shuffle(seed=cfg.seed + 1, buffer_size=cfg.stream_shuffle_buffer)

        worker = get_worker_info()
        if worker is not None:
            fineweb = fineweb.shard(num_shards=worker.num_workers, index=worker.id)
            cosmopedia = cosmopedia.shard(num_shards=worker.num_workers, index=worker.id)

        return self._probabilistic_interleave_two(
            self._tag_stream(fineweb, "fineweb"),
            self._tag_stream(cosmopedia, "cosmopedia"),
            cfg.fineweb_ratio,
            first_name="fineweb",
            second_name="cosmopedia",
            non_blocking=False,
        )

    def _open_url(self, url: str):
        cfg = self.data_config
        request = urlrequest.Request(
            url,
            headers={
                "User-Agent": cfg.commoncrawl_user_agent,
                "Accept-Encoding": "identity",
            },
        )
        return urlrequest.urlopen(request, timeout=max(5, int(cfg.commoncrawl_http_timeout_sec)))

    def _resolve_commoncrawl_ids(self) -> list[str]:
        if self._cached_cc_ids is not None:
            return self._cached_cc_ids

        cfg = self.data_config
        override_ids = [s.strip() for s in cfg.commoncrawl_crawl_ids.split(",") if s.strip()]
        if override_ids:
            self._cached_cc_ids = override_ids
            return self._cached_cc_ids

        with self._open_url(cfg.commoncrawl_index_url) as response:
            payload = response.read()
        rows = json.loads(payload.decode("utf-8", errors="replace"))
        if not isinstance(rows, list):
            raise RuntimeError("Common Crawl index metadata is not a list.")
        ids: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                cid = row.get("id")
                if isinstance(cid, str) and cid.strip():
                    ids.append(cid.strip())
        ids = sorted(set(ids), reverse=True)
        keep = max(1, int(cfg.commoncrawl_latest_crawls))
        if not ids:
            raise RuntimeError("No crawl IDs returned from Common Crawl index metadata.")
        self._cached_cc_ids = ids[:keep]
        return self._cached_cc_ids

    @staticmethod
    def _iter_wet_plaintexts_from_stream(gz_stream, min_text_chars: int) -> Iterable[str]:
        min_chars = max(1, int(min_text_chars))
        while True:
            line = gz_stream.readline()
            if not line:
                return
            if not line.startswith(b"WARC/"):
                continue

            headers: dict[bytes, bytes] = {}
            while True:
                raw = gz_stream.readline()
                if not raw:
                    return
                stripped = raw.strip()
                if not stripped:
                    break
                if b":" in stripped:
                    key, value = stripped.split(b":", 1)
                    headers[key.strip().lower()] = value.strip()

            warc_type = headers.get(b"warc-type", b"").lower()
            content_type = headers.get(b"content-type", b"").lower()
            content_length_raw = headers.get(b"content-length", b"0")
            try:
                content_length = int(content_length_raw)
            except Exception:
                content_length = 0
            if content_length < 0:
                content_length = 0
            payload = gz_stream.read(content_length) if content_length > 0 else b""

            # Skip trailing separators between records.
            _ = gz_stream.readline()

            if warc_type != b"conversion":
                continue
            if b"text/plain" not in content_type:
                continue

            text = payload.decode("utf-8", errors="ignore").strip()
            if len(text) >= min_chars:
                yield text

    def _iter_commoncrawl_wet_urls(self) -> Iterable[str]:
        cfg = self.data_config
        ids = self._resolve_commoncrawl_ids()
        prefix = cfg.commoncrawl_data_url_prefix.rstrip("/")
        urls: list[str] = []
        paths_per_crawl = max(1, int(cfg.commoncrawl_wet_paths_per_crawl))
        for crawl_id in ids:
            list_url = f"{prefix}/crawl-data/{crawl_id}/wet.paths.gz"
            with self._open_url(list_url) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    count = 0
                    for line in gz:
                        rel = line.decode("utf-8", errors="ignore").strip()
                        if not rel:
                            continue
                        urls.append(f"{prefix}/{rel.lstrip('/')}")
                        count += 1
                        if count >= paths_per_crawl:
                            break

        if not urls:
            raise RuntimeError("No Common Crawl WET file URLs were resolved.")

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(cfg.seed + (worker_id * 911))
        rng.shuffle(urls)
        if worker is not None:
            urls = [u for i, u in enumerate(urls) if (i % worker.num_workers) == worker.id]
        for url in urls:
            yield url

    def _fetch_commoncrawl_records(self, wet_url: str, records_per_file: int) -> list[dict[str, Any]]:
        cfg = self.data_config
        retries = max(0, int(cfg.commoncrawl_file_retries))
        for attempt in range(retries + 1):
            try:
                out: list[dict[str, Any]] = []
                with self._open_url(wet_url) as response:
                    with gzip.GzipFile(fileobj=response) as gz:
                        for text in self._iter_wet_plaintexts_from_stream(gz, cfg.min_text_chars):
                            if not self._passes_commoncrawl_quality_gate(text):
                                continue
                            out.append({"text": text, "_eco_source": "commoncrawl"})
                            if len(out) >= records_per_file:
                                break
                return out
            except (urlerror.URLError, OSError, TimeoutError) as exc:
                if attempt >= retries:
                    warnings.warn(f"Common Crawl file fetch failed: {wet_url} error={exc}", RuntimeWarning)
                    return []
                wait_s = 0.5 * float(attempt + 1)
                time.sleep(wait_s)
        return []

    def _build_commoncrawl_stream(self) -> Iterable[dict[str, Any]]:
        cfg = self.data_config
        files_limit = max(1, int(cfg.commoncrawl_max_files_per_worker_cycle))
        records_per_file = max(1, int(cfg.commoncrawl_records_per_file))
        parallel_files = max(1, int(cfg.commoncrawl_parallel_files))
        records_emitted = 0
        urls = iter(self._iter_commoncrawl_wet_urls())
        submitted = 0
        in_flight: dict[Future[list[dict[str, Any]]], str] = {}

        with ThreadPoolExecutor(max_workers=parallel_files, thread_name_prefix="cc_fetch") as pool:
            def _submit() -> bool:
                nonlocal submitted
                if submitted >= files_limit:
                    return False
                try:
                    wet_url = next(urls)
                except StopIteration:
                    return False
                fut = pool.submit(self._fetch_commoncrawl_records, wet_url, records_per_file)
                in_flight[fut] = wet_url
                submitted += 1
                return True

            while len(in_flight) < parallel_files and _submit():
                pass

            while in_flight:
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    _ = in_flight.pop(fut, None)
                    try:
                        records = fut.result()
                    except Exception as exc:
                        warnings.warn(f"Common Crawl worker failed: {exc}", RuntimeWarning)
                        records = []
                    for rec in records:
                        yield rec
                        records_emitted += 1
                    _submit()

        if records_emitted == 0:
            raise RuntimeError("Common Crawl stream produced zero records.")

    def _build_stream(self) -> Iterable[dict[str, Any]]:
        backend = self.data_config.data_backend.strip().lower()
        if backend == "hf":
            return self._build_hf_stream()
        if backend == "commoncrawl":
            return self._build_commoncrawl_stream()
        if backend == "hybrid":
            cfg = self.data_config
            hf_ratio = min(max(1.0 - float(cfg.commoncrawl_ratio), 0.0), 1.0)
            return self._probabilistic_interleave_two(
                self._build_hf_stream(),
                self._build_commoncrawl_stream(),
                hf_ratio,
                first_name="hf",
                second_name="commoncrawl",
                non_blocking=True,
            )
        raise ValueError(f"Unsupported data backend: {self.data_config.data_backend}")

    def _mix_remote_and_synthetic(self, remote_stream: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        ratio = float(self.data_config.synthetic_mix_ratio)
        if ratio <= 0.0:
            yield from remote_stream
            return
        ratio = min(max(ratio, 0.0), 0.95)

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.data_config.seed + (worker_id * 1337))
        remote_iter = iter(remote_stream)
        synth_iter = self._synthetic_samples()

        while True:
            if rng.random() < ratio:
                yield next(synth_iter)
                continue
            try:
                yield next(remote_iter)
            except StopIteration:
                return
            except Exception as exc:
                if self.data_config.allow_synthetic_fallback:
                    warnings.warn(
                        f"Remote stream sample fetch failed, substituting synthetic sample. error={exc}",
                        RuntimeWarning,
                    )
                    yield next(synth_iter)
                else:
                    raise RuntimeError(f"Remote stream sample fetch failed: {exc}") from exc

    def _tokenize_to_chunks(self, samples: Iterable[dict[str, Any]]):
        buffer: list[int] = []
        cursor = 0
        seq_plus_one = self.seq_len + 1
        tokenize_batch_size = max(1, int(self.data_config.tokenize_batch_size))

        def flush_text_batch(text_batch: list[tuple[str, int]]):
            nonlocal buffer, cursor
            if not text_batch:
                return

            texts = [t for t, _ in text_batch]
            source_ids = [sid for _, sid in text_batch]
            if hasattr(self.tokenizer, "encode_batch"):
                token_batches = self.tokenizer.encode_batch(texts)
            else:
                token_batches = [self.tokenizer.encode(text) for text in texts]

            for tokens, source_id in zip(token_batches, source_ids):
                if not tokens:
                    continue
                buffer.extend(tokens)
                buffer.append(self.tokenizer.eos_token_id)

                while (len(buffer) - cursor) >= seq_plus_one:
                    chunk = buffer[cursor : cursor + seq_plus_one]
                    cursor += seq_plus_one
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    src = torch.tensor(source_id, dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": labels, "source_id": src}

                # Avoid repeated O(n) list front-deletions.
                if cursor >= 4096:
                    buffer = buffer[cursor:]
                    cursor = 0

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.data_config.seed + (worker_id * 2281))
        max_text_chars = max(0, int(self.data_config.max_text_chars))
        text_batch: list[tuple[str, int]] = []
        for sample in samples:
            text = self._extract_text(sample)
            if text is None:
                continue
            if len(text) < self.data_config.min_text_chars:
                continue
            if max_text_chars > 0 and len(text) > max_text_chars:
                span = len(text) - max_text_chars
                start = rng.randint(0, span) if span > 0 else 0
                text = text[start : start + max_text_chars]
            source_id = self._sample_source_id(sample)
            if not self._dedup_accept(text, source_id):
                continue
            text_batch.append((text, source_id))
            if len(text_batch) >= tokenize_batch_size:
                yield from flush_text_batch(text_batch)
                text_batch.clear()

        if text_batch:
            yield from flush_text_batch(text_batch)

    def _synthetic_samples(self):
        corpus = [
            "A compact reasoning model improves quality per FLOP by combining quantization, sparsity, and recurrence.",
            "Given x and y, compute z carefully and justify each intermediate step before giving the final answer.",
            "Mixture of experts routes tokens dynamically to specialized subnetworks while preserving shared capabilities.",
            "Latent attention compresses key-value states to reduce memory pressure in long-context decoding.",
            "BitNet-style ternary weights can improve efficiency when training is stabilized with RMS normalization.",
        ]
        while True:
            for text in corpus:
                yield {"text": text, "_eco_source": "synthetic"}

    def __iter__(self):
        self._maybe_configure_hf_runtime()
        bootstrap_batches = max(0, int(self.data_config.bootstrap_seed_batches))
        if bootstrap_batches > 0:
            synthetic_chunks = self._tokenize_to_chunks(self._synthetic_samples())
            emitted = 0
            while emitted < bootstrap_batches:
                try:
                    yield next(synthetic_chunks)
                except StopIteration:
                    synthetic_chunks = self._tokenize_to_chunks(self._synthetic_samples())
                    continue
                emitted += 1
        if self.force_synthetic:
            yield from self._tokenize_to_chunks(self._synthetic_samples())
            return

        cfg = self.data_config
        retries = 0
        while True:
            try:
                stream = self._build_stream()
                mixed_stream = self._mix_remote_and_synthetic(stream)
                yield from self._tokenize_to_chunks(self._prefetch_samples(mixed_stream))
                # Iterable stream should not usually end; restart defensively.
                warnings.warn("Streaming dataset exhausted; restarting stream.", RuntimeWarning)
                retries = 0
            except Exception as exc:
                retries += 1
                if retries <= cfg.max_stream_retries:
                    wait_s = cfg.retry_backoff_sec * retries
                    if cfg.allow_synthetic_fallback and cfg.synthetic_backfill_on_error:
                        warnings.warn(
                            f"Streaming transient error; injecting {cfg.synthetic_backfill_batches} synthetic batches "
                            f"while retrying. Error: {exc}",
                            RuntimeWarning,
                        )
                        synthetic_chunks = self._tokenize_to_chunks(self._synthetic_samples())
                        for _ in range(max(0, cfg.synthetic_backfill_batches)):
                            try:
                                yield next(synthetic_chunks)
                            except StopIteration:
                                break
                    warnings.warn(
                        f"Streaming data error (attempt {retries}/{cfg.max_stream_retries}): {exc}. "
                        f"Retrying in {wait_s:.1f}s.",
                        RuntimeWarning,
                    )
                    time.sleep(wait_s)
                    continue

                if cfg.allow_synthetic_fallback:
                    warnings.warn(
                        "Streaming source unavailable after retries. Falling back to synthetic stream.",
                        RuntimeWarning,
                    )
                    yield from self._tokenize_to_chunks(self._synthetic_samples())
                    return
                raise RuntimeError(
                    f"Streaming source unavailable after {cfg.max_stream_retries} retries with no synthetic fallback."
                ) from exc


def _collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([sample["input_ids"] for sample in samples], dim=0)
    labels = torch.stack([sample["labels"] for sample in samples], dim=0)
    out = {"input_ids": input_ids, "labels": labels}
    if samples and "source_id" in samples[0]:
        out["source_id"] = torch.stack([sample["source_id"] for sample in samples], dim=0)
    return out


def build_streaming_dataloader(
    tokenizer: Any,
    seq_len: int,
    batch_size: int,
    data_config: StreamingDataConfig | None = None,
    force_synthetic: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = InterleavedTokenStreamDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        data_config=data_config or StreamingDataConfig(),
        force_synthetic=force_synthetic,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
    return DataLoader(**loader_kwargs)
