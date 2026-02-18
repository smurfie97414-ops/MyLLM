from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_kernels import can_use_triton_linear, triton_linear


class _BitLinearSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        row_scale: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        threshold_factor: float,
        use_triton_kernel: bool,
        matmul_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        w = _rms_norm_rows(weight, eps)
        w_q, abs_mean = _ternary_quantize(w, threshold_factor)
        w_base = w_q * abs_mean
        w_eff = w_base * row_scale

        # Align matmul dtype to activation dtype to keep tensor-core paths hot.
        if matmul_dtype is not None and w_eff.dtype != matmul_dtype:
            w_eff = w_eff.to(matmul_dtype)

        if use_triton_kernel and can_use_triton_linear(x, w_eff):
            out = triton_linear(x, w_eff, bias)
        else:
            out = F.linear(x, w_eff, bias)

        ctx.save_for_backward(x, w_eff, w_base, row_scale)
        ctx.input_shape = x.shape
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return _bitlinear_ste_backward(ctx, grad_out, n_extra_nones=4)


class _BitLinearCachedSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        row_scale: torch.Tensor,
        bias: torch.Tensor | None,
        w_eff: torch.Tensor,
        w_base: torch.Tensor,
        use_triton_kernel: bool,
    ) -> torch.Tensor:
        if use_triton_kernel and can_use_triton_linear(x, w_eff):
            out = triton_linear(x, w_eff, bias)
        else:
            out = F.linear(x, w_eff, bias)

        ctx.save_for_backward(x, w_eff, w_base, row_scale)
        ctx.input_shape = x.shape
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return _bitlinear_ste_backward(ctx, grad_out, n_extra_nones=3)


def _bitlinear_ste_backward(
    ctx,
    grad_out: torch.Tensor,
    *,
    n_extra_nones: int,
):
        x, w_eff, w_base, row_scale = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.size(-1))
        x_2d = x.reshape(-1, x.size(-1))

        grad_out_mat = grad_out_2d.to(w_eff.dtype)
        grad_x_2d = grad_out_mat @ w_eff
        grad_x = grad_x_2d.view(ctx.input_shape)

        grad_w_eff = grad_out_2d.float().transpose(0, 1) @ x_2d.float()
        grad_w_eff = grad_w_eff.to(row_scale.dtype)

        # STE approximation: treat quantized transform as identity wrt full-precision weights.
        grad_weight = grad_w_eff * row_scale
        grad_row_scale = (grad_w_eff * w_base.to(grad_w_eff.dtype)).sum(dim=1, keepdim=True)
        grad_bias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        return (grad_x, grad_weight, grad_row_scale, grad_bias, *([None] * n_extra_nones))

class RMSNorm(nn.Module):
    """RMSNorm with optional affine scaling."""

    def __init__(self, dim: int, eps: float = 1e-8, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prefer fused kernel path when available.
        if hasattr(F, "rms_norm"):
            if self.weight is None:
                return F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps)
            if self.weight.dtype == x.dtype:
                return F.rms_norm(x, (x.size(-1),), weight=self.weight, eps=self.eps)
            # Keep fast RMS reduction path while avoiding dtype-mismatch slow path.
            return F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps) * self.weight.to(dtype=x.dtype)

        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


def _rms_norm_rows(weight: torch.Tensor, eps: float) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(weight, (weight.size(-1),), weight=None, eps=eps)
    row_rms = weight.pow(2).mean(dim=1, keepdim=True)
    return weight * torch.rsqrt(row_rms + eps)


def _ternary_quantize(weight: torch.Tensor, threshold_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    abs_mean = weight.abs().mean(dim=1, keepdim=True)
    threshold = threshold_factor * abs_mean
    mask = (weight.abs() > threshold).to(weight.dtype)
    return weight.sign() * mask, abs_mean


class BitLinear(nn.Module):
    """
    BitNet-style ternary linear layer.

    - Weights are projected to {-1, 0, 1} at forward time.
    - Straight-through estimator (STE) keeps gradients flowing to fp weights.
    - RMSNorm is applied to inputs and weights before quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-8,
        threshold_factor: float = 0.5,
        input_rmsnorm: bool = True,
        use_triton_kernel: bool = True,
        fast_ste_backward: bool = True,
        matmul_dtype_align: bool = True,
        quant_cache_training: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.threshold_factor = threshold_factor
        self.use_triton_kernel = use_triton_kernel
        self.fast_ste_backward = fast_ste_backward
        self.matmul_dtype_align = matmul_dtype_align
        self.quant_cache_training = quant_cache_training

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.row_scale = nn.Parameter(torch.ones(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.input_norm = RMSNorm(in_features, eps=eps) if input_rmsnorm else nn.Identity()
        self._cache_key: tuple[int, int, torch.dtype] | None = None
        self._cache_w_eff: torch.Tensor | None = None
        self._cache_w_base: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.ones_(self.row_scale)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache_key = None
        self._cache_w_eff = None
        self._cache_w_base = None

    def _cached_quantized_weights(self, matmul_dtype: torch.dtype | None) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = matmul_dtype if matmul_dtype is not None else self.weight.dtype
        key = (int(self.weight._version), int(self.row_scale._version), dtype)
        if self._cache_key != key or self._cache_w_eff is None or self._cache_w_base is None:
            with torch.no_grad():
                w = _rms_norm_rows(self.weight, self.eps)
                w_q, abs_mean = _ternary_quantize(w, self.threshold_factor)
                w_base = w_q * abs_mean
                w_eff = w_base * self.row_scale
                if w_eff.dtype != dtype:
                    w_eff = w_eff.to(dtype)
            self._cache_key = key
            self._cache_w_eff = w_eff
            self._cache_w_base = w_base
        return self._cache_w_eff, self._cache_w_base

    def _quantized_weight(self) -> torch.Tensor:
        w = _rms_norm_rows(self.weight, self.eps)
        w_q, abs_mean = _ternary_quantize(w, self.threshold_factor)

        # STE: forward uses ternary values, backward uses continuous weights.
        w_ste = w + (w_q - w).detach()

        # Per-row scaling maintains dynamic range while keeping ternary codes.
        return w_ste * abs_mean.detach() * self.row_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        matmul_dtype = x.dtype if self.matmul_dtype_align and x.is_floating_point() else None
        if self.fast_ste_backward and self.training:
            use_cache = self.quant_cache_training
            if use_cache and hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
                use_cache = not torch.compiler.is_compiling()
            if use_cache:
                w_eff, w_base = self._cached_quantized_weights(matmul_dtype)
                return _BitLinearCachedSTEFunction.apply(
                    x,
                    self.weight,
                    self.row_scale,
                    self.bias,
                    w_eff,
                    w_base,
                    self.use_triton_kernel,
                )
            return _BitLinearSTEFunction.apply(
                x,
                self.weight,
                self.row_scale,
                self.bias,
                self.eps,
                self.threshold_factor,
                self.use_triton_kernel,
                matmul_dtype,
            )

        w_q = self._quantized_weight()
        if matmul_dtype is not None and w_q.dtype != matmul_dtype:
            w_q = w_q.to(matmul_dtype)
        if self.use_triton_kernel and can_use_triton_linear(x, w_q):
            return triton_linear(x, w_q, self.bias)
        return F.linear(x, w_q, self.bias)

    @torch.no_grad()
    def scale_output_groups_(
        self,
        scales: torch.Tensor,
        group_size: int,
        min_row_scale: float = 1e-4,
        max_row_scale: float = 1e4,
    ) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}.")
        if scales.ndim != 1:
            raise ValueError(f"scales must be rank-1, got shape {tuple(scales.shape)}.")
        expected_out = int(scales.numel()) * int(group_size)
        if expected_out != self.out_features:
            raise ValueError(
                f"Incompatible scale layout: scales={scales.numel()} group_size={group_size} "
                f"does not match out_features={self.out_features}."
            )
        expanded = scales.to(device=self.row_scale.device, dtype=self.row_scale.dtype).repeat_interleave(group_size)
        self.row_scale.mul_(expanded.view(-1, 1))
        self.row_scale.clamp_(min_row_scale, max_row_scale)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, threshold_factor={self.threshold_factor}, "
            f"use_triton_kernel={self.use_triton_kernel}, "
            f"fast_ste_backward={self.fast_ste_backward}, matmul_dtype_align={self.matmul_dtype_align}, "
            f"quant_cache_training={self.quant_cache_training}"
        )
