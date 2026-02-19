from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from model.sigma_llm import SigmaConfig, SigmaLLM
from training.data import build_tokenizer
from training.memory_hack import apply_instant_memory_hack

_CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference script for SigmaLLM checkpoints.")
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint_step_*.pt (or directory containing checkpoints).")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory searched for latest checkpoint if --checkpoint is omitted.")
    p.add_argument("--config", type=Path, required=True, help="JSON config compatible with SigmaConfig.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--tokenizer-backend", choices=["tiktoken", "hf"], default="tiktoken")
    p.add_argument("--tokenizer-name", type=str, default="cl100k_base")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    return p.parse_args()


def _resolve_checkpoint_path(checkpoint: Path | None, checkpoint_dir: Path) -> Path:
    if checkpoint is not None:
        if checkpoint.is_dir():
            checkpoint_dir = checkpoint
        else:
            if not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
            return checkpoint

    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_step_*.pt"):
        m = _CHECKPOINT_RE.match(path.name)
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_step_*.pt found in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _load_sigma_config(config_path: Path, vocab_size: int) -> SigmaConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    cfg_data: dict[str, Any] = dict(payload)
    cfg_data["vocab_size"] = int(vocab_size)
    cfg = SigmaConfig(**cfg_data)
    cfg.validate()
    return cfg


def _resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    tokenizer = build_tokenizer(args.tokenizer_backend, args.tokenizer_name)
    config = _load_sigma_config(args.config, vocab_size=int(tokenizer.vocab_size))

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    model = SigmaLLM(config)
    apply_instant_memory_hack(model, comp_dim=config.instant_comp_dim, error_threshold=config.instant_error_threshold)
    model_dtype = _resolve_dtype(args.dtype, device)
    model.to(device=device, dtype=model_dtype)

    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"Invalid checkpoint format in {checkpoint_path}: expected dict payload")
    model_state = checkpoint_payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint payload is missing a valid 'model' state_dict key")
    model.load_state_dict(model_state, strict=True)
    model.eval()

    input_ids = tokenizer.encode(args.prompt)
    if not input_ids:
        raise ValueError("Prompt tokenization produced zero tokens. Provide a non-empty prompt.")

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
        )

    text = tokenizer.decode(output_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
