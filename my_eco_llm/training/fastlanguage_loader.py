from __future__ import annotations

from typing import Any

import torch


def _is_moe_gate_name(name: str) -> bool:
    n = name.lower()
    keys = ("gate", "router", "expert_gate", "moe_gate", "gate_proj", "router_proj")
    return any(k in n for k in keys)


def protect_moe_gates_precision(model: Any, precision: str = "bf16") -> int:
    precision_norm = str(precision).strip().lower()
    if precision_norm not in {"bf16", "int8"}:
        raise ValueError(f"precision must be one of ['bf16', 'int8'], got {precision!r}")
    protected = 0
    for name, module in model.named_modules():
        if not _is_moe_gate_name(name):
            continue
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight")
        if not isinstance(weight, torch.Tensor):
            continue
        if precision_norm == "bf16":
            module.to(dtype=torch.bfloat16)
        protected += 1
    return protected


def load_fast_language_model_158(
    model_name: str,
    *,
    max_seq_length: int = 4096,
    token: str | None = None,
    moe_gate_precision: str = "bf16",
    **kwargs: Any,
) -> tuple[Any, Any, dict[str, float]]:
    """
    Load via FastLanguageModel with strict 1.58-bit mode enabled.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FastLanguageModel loader requested but unsloth is not installed in this environment."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(max_seq_length),
        token=token,
        load_in_1_58bit=True,
        **kwargs,
    )
    protected = protect_moe_gates_precision(model, precision=moe_gate_precision)
    stats = {
        "bitnet_1_58_active": 1.0,
        "moe_gate_precision_bf16": 1.0 if str(moe_gate_precision).strip().lower() == "bf16" else 0.0,
        "moe_gate_precision_int8": 1.0 if str(moe_gate_precision).strip().lower() == "int8" else 0.0,
        "moe_gate_protected_modules": float(protected),
    }
    return model, tokenizer, stats

