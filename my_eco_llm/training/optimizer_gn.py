from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch.optim import Optimizer

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

try:
    import triton

    TRITON_AVAILABLE = True
    TRITON_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - import guard
    triton = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False
    TRITON_IMPORT_ERROR = repr(exc)


def _candidate_kernel_paths() -> list[Path]:
    candidates: list[Path] = []
    module_dir = Path(__file__).resolve().parent
    candidates.append(module_dir / "optimizer_gn_kernel_src.py")
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            base = Path(meipass)
            candidates.append(base / "training" / "optimizer_gn_kernel_src.py")
            candidates.append(base / "_internal" / "training" / "optimizer_gn_kernel_src.py")
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "training" / "optimizer_gn_kernel_src.py")
        candidates.append(exe_dir / "_internal" / "training" / "optimizer_gn_kernel_src.py")
    return candidates


def _load_gnprox_kernel():
    last_error: Exception | None = None
    for candidate in _candidate_kernel_paths():
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("eco_gn_kernel_src", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kernel = getattr(module, "gnprox_update_kernel", None)
            if kernel is not None:
                return kernel
        except Exception as exc:  # pragma: no cover
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"Failed to load gnprox kernel source. last_error={last_error}")
    raise RuntimeError("Failed to locate optimizer_gn_kernel_src.py for Triton JIT loading.")


_gnprox_update_kernel = None
if TRITON_AVAILABLE:
    try:
        _gnprox_update_kernel = _load_gnprox_kernel()
    except Exception as exc:  # pragma: no cover
        TRITON_AVAILABLE = False
        TRITON_IMPORT_ERROR = repr(exc)


@dataclass
class GNProxConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    damping: float = 0.01
    damping_min: float = 1e-4
    damping_max: float = 1.0
    damping_up: float = 1.08
    damping_down: float = 0.97
    damping_gain: float = 0.20
    ns_steps: int = 5
    block_size: int = 1024
    clip_grad: float = 0.0


class GNProx(Optimizer):
    """
    GN-Prox optimizer:
    - Diagonal Gauss-Newton/Hessian approximation from EMA of squared gradients.
    - Fused Triton kernel update in-place on GPU tensors.
    - Newton-Schulz reciprocal iteration inside the kernel.
    - Levenberg-Marquardt adaptive damping.
    """

    def __init__(self, params, config: GNProxConfig) -> None:
        if not TRITON_AVAILABLE:
            raise RuntimeError(f"GNProx requires Triton, import failed: {TRITON_IMPORT_ERROR}")
        defaults = dict(
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            beta1=float(config.beta1),
            beta2=float(config.beta2),
            eps=float(config.eps),
            damping=float(config.damping),
            damping_min=float(config.damping_min),
            damping_max=float(config.damping_max),
            damping_up=float(config.damping_up),
            damping_down=float(config.damping_down),
            damping_gain=float(config.damping_gain),
            ns_steps=max(1, int(config.ns_steps)),
            block_size=max(128, int(config.block_size)),
            clip_grad=float(config.clip_grad),
        )
        super().__init__(params, defaults)
        self.supports_internal_grad_clip = True
        self._last_metrics: dict[str, float] = {
            "gnprox_damping": float(config.damping),
            "gnprox_curvature_ratio": 0.0,
            "gnprox_step_elements": 0.0,
        }

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_elements = 0
        ratio_acc = 0.0
        ratio_count = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            damping = float(group["damping"])
            damping_min = float(group["damping_min"])
            damping_max = float(group["damping_max"])
            damping_up = float(group["damping_up"])
            damping_down = float(group["damping_down"])
            damping_gain = float(group["damping_gain"])
            ns_steps = int(group["ns_steps"])
            block_size = int(group["block_size"])
            clip_grad = float(group["clip_grad"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("GNProx does not support sparse gradients.")
                if p.device.type != "cuda":
                    raise RuntimeError("GNProx requires CUDA tensors.")

                g = p.grad.detach()
                if not g.is_contiguous():
                    g = g.contiguous()
                if not p.data.is_contiguous():
                    p.data = p.data.contiguous()

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["h"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] = 0

                m = state["m"]
                h = state["h"]
                state["step"] = int(state["step"]) + 1
                step_idx = int(state["step"])
                bc1_inv = 1.0 / max(1.0 - (beta1**step_idx), 1e-8)
                bc2_inv = 1.0 / max(1.0 - (beta2**step_idx), 1e-8)

                p_flat = p.data.view(-1)
                g_flat = g.view(-1)
                m_flat = m.view(-1)
                h_flat = h.view(-1)
                n = int(p_flat.numel())
                if n == 0:
                    continue

                grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
                if _gnprox_update_kernel is None:
                    raise RuntimeError("GNProx Triton kernel symbol is unavailable.")
                _gnprox_update_kernel[grid](
                    p_flat,
                    g_flat,
                    m_flat,
                    h_flat,
                    n,
                    lr,
                    beta1,
                    beta2,
                    wd,
                    bc1_inv,
                    bc2_inv,
                    eps,
                    damping,
                    damping_gain,
                    clip_grad,
                    BLOCK_SIZE=block_size,
                    NS_STEPS=ns_steps,
                    num_warps=4,
                )

                # Global LM damping control from running gradient/curvature ratio.
                ratio = float((g_flat.float().abs().mean() / (h_flat.float().mean().sqrt().item() + eps)))
                ratio_acc += ratio
                ratio_count += 1
                total_elements += n

            if ratio_count > 0:
                ratio_mean = ratio_acc / ratio_count
                if ratio_mean > 1.25:
                    damping = min(damping_max, damping * damping_up)
                else:
                    damping = max(damping_min, damping * damping_down)
                group["damping"] = damping

        self._last_metrics = {
            "gnprox_damping": float(self.param_groups[0].get("damping", 0.0)),
            "gnprox_curvature_ratio": float(ratio_acc / max(ratio_count, 1)),
            "gnprox_step_elements": float(total_elements),
        }
        return loss
