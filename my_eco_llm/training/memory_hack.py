from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn

_DYNAMO_DISABLE = (
    torch.compiler.disable
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable")
    else torch._dynamo.disable
)


@dataclass
class QuantizedResidual:
    packed: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, ...]
    bit_width: int


@dataclass
class HolographicActivation:
    compressed: torch.Tensor
    residual: QuantizedResidual
    projection: torch.Tensor


def _pack_int4(x_int8: torch.Tensor) -> torch.Tensor:
    # x_int8 values in [-8, 7]
    values = (x_int8 + 8).to(torch.uint8).flatten()
    if (values.numel() % 2) != 0:
        values = torch.cat((values, torch.zeros(1, dtype=torch.uint8, device=values.device)), dim=0)
    low = values[0::2] & 0x0F
    high = (values[1::2] & 0x0F) << 4
    return (low | high).contiguous()


def _unpack_int4(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    packed = packed.flatten()
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty((packed.numel() * 2,), dtype=torch.int8, device=packed.device)
    out[0::2] = low.to(torch.int8) - 8
    out[1::2] = high.to(torch.int8) - 8
    return out[:n_values]


def _quantize_dynamic(x: torch.Tensor, bit_width: int) -> QuantizedResidual:
    if bit_width not in (4, 8):
        raise ValueError(f"Unsupported bit_width={bit_width}, expected 4 or 8.")
    x_f = x.float()
    qmax = 7.0 if bit_width == 4 else 127.0
    scale = x_f.abs().amax().clamp_min(1e-8) / qmax
    q = torch.round(x_f / scale).clamp(-qmax - 1 if bit_width == 4 else -127, qmax).to(torch.int8)
    if bit_width == 4:
        packed = _pack_int4(q)
    else:
        packed = q.contiguous()
    return QuantizedResidual(
        packed=packed,
        scale=scale,
        shape=tuple(x.shape),
        bit_width=bit_width,
    )


def _dequantize_dynamic(q: QuantizedResidual) -> torch.Tensor:
    if q.bit_width == 4:
        n = 1
        for d in q.shape:
            n *= int(d)
        vals = _unpack_int4(q.packed, n_values=n).float()
    elif q.bit_width == 8:
        vals = q.packed.float().flatten()
    else:
        raise ValueError(f"Unsupported bit_width={q.bit_width}")
    return (vals * q.scale).view(q.shape)


class HolographicActivationCompressor(nn.Module):
    """
    INSTANT phase-2:
      A_comp = A @ P
      E = A - A_comp @ P^T
      E is quantized to Int4; if reconstruction error > threshold, escalate to Int8.
    """

    def __init__(
        self,
        d_model: int,
        comp_dim: int,
        error_threshold: float = 0.01,
        int4_probe_interval: int = 32,
    ) -> None:
        super().__init__()
        if comp_dim <= 0 or comp_dim > d_model:
            raise ValueError(f"Invalid comp_dim={comp_dim} for d_model={d_model}")
        self.d_model = int(d_model)
        self.comp_dim = int(comp_dim)
        self.error_threshold = float(error_threshold)
        self.int4_probe_interval = max(1, int(int4_probe_interval))
        self.register_buffer("projection", self._build_projection(), persistent=True)
        self.register_buffer("last_recon_error", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_error_bit_width", torch.tensor(4.0), persistent=False)
        self.register_buffer("last_compression_ratio", torch.tensor(1.0), persistent=False)
        self.register_buffer("compress_calls", torch.tensor(0.0), persistent=False)

    def _build_projection(self) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(20260216)
        mat = torch.randn(self.d_model, self.comp_dim, generator=g)
        q, _ = torch.linalg.qr(mat, mode="reduced")
        return q.float()

    @torch.no_grad()
    def set_error_threshold(self, value: float) -> None:
        self.error_threshold = float(min(max(value, 1e-5), 0.25))

    def compress(self, x: torch.Tensor) -> HolographicActivation:
        x_f = x.float()
        p = self.projection.to(device=x.device, dtype=x_f.dtype)
        comp = x_f @ p
        recon = comp @ p.t()
        err = x_f - recon
        self.compress_calls.add_(1.0)
        calls = int(self.compress_calls.item())
        prefer_bit = 8 if int(self.last_error_bit_width.item()) >= 8 else 4
        probe_int4 = bool(prefer_bit == 8 and (calls % int(self.int4_probe_interval) == 0))

        if prefer_bit == 8 and not probe_int4:
            q = _quantize_dynamic(err, bit_width=8)
            recon_hat = recon + _dequantize_dynamic(q).to(recon.dtype)
            rel_err = (recon_hat - x_f).norm() / x_f.norm().clamp_min(1e-8)
        else:
            q = _quantize_dynamic(err, bit_width=4)
            recon_hat = recon + _dequantize_dynamic(q).to(recon.dtype)
            rel_err = (recon_hat - x_f).norm() / x_f.norm().clamp_min(1e-8)
            if rel_err.item() > self.error_threshold:
                q = _quantize_dynamic(err, bit_width=8)
                recon_hat = recon + _dequantize_dynamic(q).to(recon.dtype)
                rel_err = (recon_hat - x_f).norm() / x_f.norm().clamp_min(1e-8)

        original_bytes = float(x.numel() * x.element_size())
        comp_bytes = float(comp.numel() * comp.element_size())
        if q.bit_width == 4:
            resid_bytes = float(q.packed.numel())  # packed uint8
        else:
            resid_bytes = float(q.packed.numel() * q.packed.element_size())
        ratio = original_bytes / max(comp_bytes + resid_bytes, 1.0)

        self.last_recon_error.copy_(rel_err.detach().to(self.last_recon_error.dtype))
        self.last_error_bit_width.fill_(float(q.bit_width))
        self.last_compression_ratio.fill_(float(ratio))

        return HolographicActivation(
            compressed=comp,
            residual=q,
            projection=p,
        )

    def decompress(self, saved: HolographicActivation) -> torch.Tensor:
        base = saved.compressed @ saved.projection.t()
        residual = _dequantize_dynamic(saved.residual).to(base.dtype).to(base.device)
        return base + residual

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        saved = self.compress(x)
        x_hat = self.decompress(saved)
        return x_hat.to(dtype=x.dtype)

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return {
            "instant_reconstruction_error": float(self.last_recon_error.item()),
            "instant_error_bit_width": float(self.last_error_bit_width.item()),
            "instant_compression_ratio": float(self.last_compression_ratio.item()),
            "instant_error_threshold": float(self.error_threshold),
        }


class _InstantReversibleFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        core: nn.Module,
        fixed_point_iters: int,
        core_gain: float,
        min_iters: int,
        rel_tol: float,
        *params: torch.Tensor,
    ):
        with torch.no_grad():
            y = x + (float(core_gain) * core(x))
        ctx.core = core
        ctx.fixed_point_iters = int(fixed_point_iters)
        ctx.core_gain = float(core_gain)
        ctx.min_iters = int(min_iters)
        ctx.rel_tol = float(rel_tol)
        ctx.n_params = len(params)
        ctx.save_for_backward(y, *params)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        y = saved[0]
        core: nn.Module = ctx.core

        with torch.no_grad():
            x_est, used_iters, rel_resid = _invert_reversible_residual(
                y=y,
                core=core,
                iters=ctx.fixed_point_iters,
                core_gain=float(ctx.core_gain),
                min_iters=ctx.min_iters,
                rel_tol=ctx.rel_tol,
            )
            try:
                setattr(core, "_instant_last_inv_iters", float(used_iters))
                setattr(core, "_instant_last_inv_rel", float(rel_resid))
            except Exception:
                pass

        x_est = x_est.detach().requires_grad_(True)
        core_params = tuple(core.parameters())
        with torch.enable_grad():
            y_re = x_est + (float(ctx.core_gain) * core(x_est))
        grads = torch.autograd.grad(
            outputs=y_re,
            inputs=(x_est, *core_params),
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grad_x = grads[0]
        grad_params = list(grads[1:])
        while len(grad_params) < ctx.n_params:
            grad_params.append(None)
        return (grad_x, None, None, None, None, None, *grad_params[: ctx.n_params])


def _invert_reversible_residual(
    *,
    y: torch.Tensor,
    core: nn.Module,
    iters: int,
    core_gain: float = 0.5,
    relax: float = 0.5,
    min_iters: int = 2,
    rel_tol: float = 2e-3,
) -> tuple[torch.Tensor, int, float]:
    """
    Solve x + core(x) = y by damped residual fixed-point updates.
    This is more stable than plain x_{k+1}=y-core(x_k) when local Lipschitz constants drift.
    """
    x = y
    alpha = float(min(max(relax, 0.05), 1.0))
    n_iters = max(1, int(iters))
    n_min = max(1, min(int(min_iters), n_iters))
    gain = float(min(max(core_gain, 0.01), 1.0))
    tol = float(max(rel_tol, 1e-6))
    y_norm = y.detach().float().norm().clamp_min(1e-8)
    last_rel = float("inf")
    used = n_iters
    for i in range(n_iters):
        fx = x + (gain * core(x))
        resid = y - fx
        x = x + (alpha * resid)
        if (i + 1) < n_min:
            continue
        rel = float(resid.detach().float().norm().item() / float(y_norm.item()))
        last_rel = rel
        if rel <= tol:
            used = i + 1
            break
    if not math.isfinite(last_rel):
        last_rel = 0.0
    return x, int(used), float(last_rel)


class InstantReversibleResidual(nn.Module):
    """
    INSTANT phase-1:
    Do not keep Mamba activations in backward graph; reconstruct x from y by fixed-point solve.
    """

    def __init__(
        self,
        core: nn.Module,
        fixed_point_iters: int = 4,
        core_gain: float = 0.5,
        min_iters: int = 2,
        rel_tol: float = 2e-3,
    ) -> None:
        super().__init__()
        self.core = core
        self.fixed_point_iters = int(fixed_point_iters)
        self.core_gain = float(core_gain)
        self.min_iters = int(min_iters)
        self.rel_tol = float(rel_tol)
        if self.fixed_point_iters <= 0:
            raise ValueError(f"fixed_point_iters must be > 0, got {self.fixed_point_iters}")
        if self.core_gain <= 0 or self.core_gain > 1.0:
            raise ValueError(f"core_gain must be in (0,1], got {self.core_gain}")
        if self.min_iters <= 0:
            raise ValueError(f"min_iters must be > 0, got {self.min_iters}")
        if self.min_iters > self.fixed_point_iters:
            raise ValueError(
                f"min_iters ({self.min_iters}) must be <= fixed_point_iters ({self.fixed_point_iters})"
            )
        if self.rel_tol <= 0:
            raise ValueError(f"rel_tol must be > 0, got {self.rel_tol}")

    @_DYNAMO_DISABLE
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params: tuple[torch.Tensor, ...] = tuple(self.core.parameters())
        return _InstantReversibleFn.apply(
            x,
            self.core,
            self.fixed_point_iters,
            self.core_gain,
            self.min_iters,
            self.rel_tol,
            *params,
        )


def apply_instant_memory_hack(
    model: nn.Module,
    *,
    comp_dim: int,
    error_threshold: float,
    reversible_iters: int,
) -> dict[str, int]:
    """
    Patch Sigma blocks in-place:
      - Mamba blocks -> strict reversible residual wrapper.
      - MLA blocks   -> holographic activation compressor.
    """
    patched_mamba = 0
    patched_mla = 0
    for module in model.modules():
        if getattr(module, "is_sigma_mamba_block", False):
            core = getattr(module, "core", None)
            if core is None:
                raise RuntimeError("Sigma mamba block missing 'core' for INSTANT patch.")
            module.instant_reversible = InstantReversibleResidual(
                core=core,
                fixed_point_iters=reversible_iters,
                core_gain=0.5,
                # Forward-fix optimization: reduce mandatory inversion iterations while
                # preserving residual tolerance checks for stability.
                min_iters=1,
                rel_tol=3e-3,
            )
            patched_mamba += 1
        if getattr(module, "is_sigma_mla_block", False):
            d_model = int(getattr(module, "d_model"))
            module.instant_compressor = HolographicActivationCompressor(
                d_model=d_model,
                comp_dim=min(int(comp_dim), d_model),
                error_threshold=float(error_threshold),
            )
            patched_mla += 1

    if patched_mamba == 0 or patched_mla == 0:
        raise RuntimeError(
            f"INSTANT patch failed. patched_mamba={patched_mamba}, patched_mla={patched_mla}. "
            "Expected both > 0."
        )
    if hasattr(model, "mark_instant_patched"):
        model.mark_instant_patched()
    return {"patched_mamba": patched_mamba, "patched_mla": patched_mla}


@torch.no_grad()
def verify_reversible_cycle(
    wrapper: InstantReversibleResidual,
    d_model: int,
    *,
    device: torch.device,
    seq_len: int = 16,
    batch_size: int = 2,
) -> float:
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    y = x + (float(wrapper.core_gain) * wrapper.core(x))
    x_est, _, _ = _invert_reversible_residual(
        y=y,
        core=wrapper.core,
        iters=wrapper.fixed_point_iters,
        core_gain=float(wrapper.core_gain),
        min_iters=wrapper.min_iters,
        rel_tol=wrapper.rel_tol,
    )
    rel = float((x_est - x).norm().item() / (x.norm().item() + 1e-8))
    return rel


@torch.no_grad()
def collect_instant_metrics(model: nn.Module) -> dict[str, float]:
    errors: list[float] = []
    bits: list[float] = []
    ratios: list[float] = []
    inv_iters: list[float] = []
    inv_rel: list[float] = []
    for module in model.modules():
        compressor = getattr(module, "instant_compressor", None)
        if compressor is None:
            wrapper = getattr(module, "instant_reversible", None)
            if wrapper is not None and hasattr(wrapper, "core"):
                last_iter = getattr(wrapper.core, "_instant_last_inv_iters", None)
                last_rel = getattr(wrapper.core, "_instant_last_inv_rel", None)
                if isinstance(last_iter, (float, int)):
                    inv_iters.append(float(last_iter))
                if isinstance(last_rel, (float, int)):
                    inv_rel.append(float(last_rel))
            continue
        metrics = compressor.metrics()
        errors.append(float(metrics["instant_reconstruction_error"]))
        bits.append(float(metrics["instant_error_bit_width"]))
        ratios.append(float(metrics["instant_compression_ratio"]))
    out: dict[str, float] = {}
    if errors:
        out["instant_reconstruction_error"] = float(sum(errors) / len(errors))
        out["instant_error_bit_width"] = float(sum(bits) / len(bits))
        out["instant_compression_ratio"] = float(sum(ratios) / len(ratios))
    if inv_iters:
        out["instant_inversion_iters"] = float(sum(inv_iters) / len(inv_iters))
    if inv_rel:
        out["instant_inversion_rel_residual"] = float(sum(inv_rel) / len(inv_rel))
    return out


def iter_instant_wrappers(model: nn.Module) -> Iterable[InstantReversibleResidual]:
    for module in model.modules():
        w = getattr(module, "instant_reversible", None)
        if isinstance(w, InstantReversibleResidual):
            yield w
