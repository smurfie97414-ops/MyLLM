from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

TRITON_AVAILABLE = False
TRITON_IMPORT_ERROR: str | None = None
_mamba3_complex_scan_kernel = None

try:
    import triton

    TRITON_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    triton = None  # type: ignore[assignment]
    TRITON_IMPORT_ERROR = repr(exc)


def _candidate_kernel_paths() -> list[Path]:
    candidates: list[Path] = []
    module_dir = Path(__file__).resolve().parent
    candidates.append(module_dir / "sigma_kernel_src.py")

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            base = Path(meipass)
            candidates.append(base / "model" / "sigma_kernel_src.py")
            candidates.append(base / "_internal" / "model" / "sigma_kernel_src.py")
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "model" / "sigma_kernel_src.py")
        candidates.append(exe_dir / "_internal" / "model" / "sigma_kernel_src.py")
    return candidates


def _load_kernel_from_source():
    last_error: Exception | None = None
    for candidate in _candidate_kernel_paths():
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("sigma_triton_kernel_src", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kernel = getattr(module, "mamba3_complex_scan_interleaved_kernel", None)
            if kernel is not None:
                return kernel
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"Failed to load sigma kernel source. last_error={last_error}")
    raise RuntimeError("Failed to locate sigma_kernel_src.py for Triton JIT source loading.")


if TRITON_AVAILABLE:
    try:
        _mamba3_complex_scan_kernel = _load_kernel_from_source()
    except Exception as exc:  # pragma: no cover - environment dependent
        TRITON_AVAILABLE = False
        TRITON_IMPORT_ERROR = repr(exc)


def mamba3_complex_scan_interleaved(
    x_interleaved: torch.Tensor,
    a_interleaved: torch.Tensor,
    b_interleaved: torch.Tensor,
    state_interleaved: torch.Tensor,
    *,
    block_t: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      x_interleaved: [N, T, 2], contiguous, CUDA tensor.
      a_interleaved: [N, 2], contiguous, CUDA tensor.
      b_interleaved: [N, 2], contiguous, CUDA tensor.
      state_interleaved: [N, 2], contiguous, CUDA tensor (updated in-place copy).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"sigma kernels require Triton; import failed: {TRITON_IMPORT_ERROR}")
    if _mamba3_complex_scan_kernel is None:
        raise RuntimeError("sigma Triton kernel symbol is unavailable.")
    if x_interleaved.ndim != 3 or x_interleaved.size(-1) != 2:
        raise ValueError(f"x_interleaved must be [N,T,2], got {tuple(x_interleaved.shape)}")
    if a_interleaved.shape != (x_interleaved.size(0), 2):
        raise ValueError(
            f"a_interleaved must be [N,2] with N={x_interleaved.size(0)}, got {tuple(a_interleaved.shape)}"
        )
    if b_interleaved.shape != (x_interleaved.size(0), 2):
        raise ValueError(
            f"b_interleaved must be [N,2] with N={x_interleaved.size(0)}, got {tuple(b_interleaved.shape)}"
        )
    if state_interleaved.shape != (x_interleaved.size(0), 2):
        raise ValueError(
            f"state_interleaved must be [N,2] with N={x_interleaved.size(0)}, got {tuple(state_interleaved.shape)}"
        )
    if not x_interleaved.is_cuda:
        raise RuntimeError("sigma Triton kernel requires CUDA tensors.")
    if not x_interleaved.is_contiguous():
        raise ValueError("x_interleaved must be contiguous.")
    if not a_interleaved.is_contiguous() or not b_interleaved.is_contiguous() or not state_interleaved.is_contiguous():
        raise ValueError("a_interleaved, b_interleaved and state_interleaved must be contiguous.")
    if int(x_interleaved.size(1)) != int(block_t):
        raise ValueError(
            f"x_interleaved second dimension must match block_t for static scan. got T={x_interleaved.size(1)} "
            f"block_t={block_t}"
        )

    n_lanes = int(x_interleaved.size(0))
    out = torch.empty_like(x_interleaved)
    next_state = state_interleaved
    grid = lambda meta: (triton.cdiv(n_lanes, meta["BLOCK_N"]),)
    _mamba3_complex_scan_kernel[grid](
        x_interleaved,
        out,
        a_interleaved,
        b_interleaved,
        next_state,
        n_lanes,
        BLOCK_T=int(block_t),
        BLOCK_N=max(32, int(block_n)),
        num_warps=4,
    )
    return out, next_state
