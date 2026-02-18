from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import sys
from typing import Any


def _import_or_fail(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - hard fail path
        raise RuntimeError(f"Missing or broken dependency '{name}': {exc}") from exc


def _check_trl_symbols() -> None:
    trl = _import_or_fail("trl")
    missing: list[str] = []
    for symbol in ("GRPOConfig", "GRPOTrainer"):
        if not hasattr(trl, symbol):
            missing.append(symbol)
    if missing:
        raise RuntimeError(f"trl is installed but missing symbols: {missing}")


def _check_unsloth_presence() -> None:
    if importlib.util.find_spec("unsloth") is None:
        raise RuntimeError("Missing unsloth package.")
    if importlib.util.find_spec("unsloth_zoo") is None:
        raise RuntimeError("Missing unsloth_zoo package.")
    try:
        _ = importlib.metadata.version("unsloth")
        _ = importlib.metadata.version("unsloth-zoo")
    except Exception as exc:
        raise RuntimeError(f"Unsloth package metadata lookup failed: {exc}") from exc


def main() -> int:
    mods = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "wandb",
        "wandb_workspaces",
        "tiktoken",
        "pynvml",
        "triton",
    ]
    loaded = {name: 0 for name in mods}
    for m in mods:
        _import_or_fail(m)
        loaded[m] = 1
    _check_trl_symbols()
    _check_unsloth_presence()
    loaded["unsloth"] = 1
    loaded["unsloth_zoo"] = 1

    import torch  # noqa: WPS433

    payload = {
        "python": sys.version.split()[0],
        "cuda_available": int(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "loaded": loaded,
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
