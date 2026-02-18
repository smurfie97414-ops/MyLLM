from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

TRITON_AVAILABLE = False
TRITON_IMPORT_ERROR: str | None = None
_bitlinear_mm_kernel = None

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_IMPORT_ERROR = repr(exc)


def _candidate_kernel_paths() -> list[Path]:
    candidates: list[Path] = []
    module_dir = Path(__file__).resolve().parent
    candidates.append(module_dir / "triton_kernel_src.py")

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            base = Path(meipass)
            candidates.append(base / "model" / "triton_kernel_src.py")
            candidates.append(base / "_internal" / "model" / "triton_kernel_src.py")
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "model" / "triton_kernel_src.py")
        candidates.append(exe_dir / "_internal" / "model" / "triton_kernel_src.py")
    return candidates


def _load_triton_kernel_from_source():
    last_error: Exception | None = None
    for candidate in _candidate_kernel_paths():
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("eco_triton_kernel_src", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kernel = getattr(module, "bitlinear_mm_kernel", None)
            if kernel is not None:
                return kernel
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"Failed to load Triton kernel source. last_error={last_error}")
    raise RuntimeError("Failed to locate triton_kernel_src.py for Triton JIT source loading.")


if TRITON_AVAILABLE:
    try:
        _bitlinear_mm_kernel = _load_triton_kernel_from_source()
    except Exception as exc:  # pragma: no cover - environment dependent
        TRITON_AVAILABLE = False
        TRITON_IMPORT_ERROR = repr(exc)


def _triton_linear_2d(x_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
    if _bitlinear_mm_kernel is None:
        raise RuntimeError("Triton kernel symbol is unavailable.")

    if x_2d.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"Expected rank-2 tensors, got x={tuple(x_2d.shape)} weight={tuple(weight.shape)}")
    if x_2d.size(1) != weight.size(1):
        raise ValueError(
            f"Incompatible matmul dimensions: x={tuple(x_2d.shape)} weight={tuple(weight.shape)} "
            "(expect x.shape[1] == weight.shape[1])."
        )

    x_2d = x_2d.contiguous()
    weight = weight.contiguous()

    M, K = x_2d.shape
    N = weight.size(0)

    # OOM-safe static launch selection (avoids Triton autotune benchmark allocations on full VRAM workloads).
    if M >= 128 and N >= 128:
        block_m, block_n, warps, stages = 128, 128, 8, 4
    elif M >= 128:
        block_m, block_n, warps, stages = 128, 64, 8, 3
    elif N >= 128:
        block_m, block_n, warps, stages = 64, 128, 8, 3
    else:
        block_m, block_n, warps, stages = 64, 64, 4, 3
    block_k = 32
    group_m = 8

    out = torch.empty((M, N), device=x_2d.device, dtype=x_2d.dtype)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    _bitlinear_mm_kernel[grid](
        x_2d,
        weight,
        out,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=warps,
        num_stages=stages,
    )
    if bias is not None:
        out = out + bias
    return out


class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):  # type: ignore[override]
        x_2d = x.reshape(-1, x.size(-1))
        out_2d = _triton_linear_2d(x_2d, weight, bias)
        ctx.save_for_backward(x_2d, weight)
        ctx.input_shape = x.shape
        ctx.has_bias = bias is not None
        return out_2d.view(*x.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x_2d, weight = ctx.saved_tensors
        grad_2d = grad_out.reshape(-1, grad_out.size(-1))

        grad_x = grad_2d @ weight
        grad_w = grad_2d.transpose(0, 1) @ x_2d
        grad_b = grad_2d.sum(dim=0) if ctx.has_bias else None

        grad_x = grad_x.view(ctx.input_shape)
        return grad_x, grad_w, grad_b


def can_use_triton_linear(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if not x.is_cuda or not weight.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dtype != x.dtype:
        return False
    if x.size(-1) != weight.size(1):
        return False
    if x.numel() == 0:
        return False
    # Keep compatibility with torch.compile; eager path still uses Triton.
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling():
        return False
    return True


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return TritonLinearFunction.apply(x, weight, bias)
