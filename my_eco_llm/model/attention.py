from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear, RMSNorm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}.")
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self.max_seq_len_cached and self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        # [1, seq, 1, dim]
        self.cos_cached = cos.unsqueeze(0).unsqueeze(2)
        self.sin_cached = sin.unsqueeze(0).unsqueeze(2)
        self.max_seq_len_cached = seq_len

    def apply(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        x: [B, T, H, D]
        start_pos: absolute position index for the first token in x.
        """
        seq_len = x.size(1)
        end_pos = start_pos + seq_len
        self._build_cache(end_pos, x.device, x.dtype)
        cos = self.cos_cached[:, start_pos:end_pos, :, :]
        sin = self.sin_cached[:, start_pos:end_pos, :, :]
        return (x * cos) + (_rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA):
    - KV states are compressed into a latent vector for reduced cache memory.
    - RoPE is only applied to decoupled Q/K rotary paths.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_latent_dim: int,
        rope_dim: int | None = None,
        rope_base: int = 10000,
        dropout: float = 0.0,
        qk_norm: bool = True,
        use_triton_bitlinear: bool = True,
        use_bitlinear_quant_cache_training: bool = False,
        qk_clip_ema_beta: float = 0.0,
        qk_clip_metric_interval: int = 10,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.dropout = dropout
        self.rope_dim = self.head_dim if rope_dim is None else rope_dim
        self.qk_norm = qk_norm
        self.qk_clip_ema_beta = qk_clip_ema_beta
        self.qk_clip_metric_interval = max(1, int(qk_clip_metric_interval))

        if self.rope_dim > self.head_dim:
            raise ValueError(f"rope_dim ({self.rope_dim}) cannot be larger than head_dim ({self.head_dim}).")
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {self.rope_dim}.")
        if self.qk_clip_ema_beta < 0 or self.qk_clip_ema_beta > 1:
            raise ValueError(f"qk_clip_ema_beta must be in [0, 1], got {self.qk_clip_ema_beta}.")

        # Decoupled Q/K paths: content + rotary
        self.q_content = BitLinear(
            d_model,
            d_model,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.q_rope = BitLinear(
            d_model,
            n_heads * self.rope_dim,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )

        # Latent KV compression and decompression
        self.kv_down = BitLinear(
            d_model,
            kv_latent_dim,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.k_content_up = BitLinear(
            kv_latent_dim,
            d_model,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.v_up = BitLinear(
            kv_latent_dim,
            d_model,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.k_rope_up = BitLinear(
            kv_latent_dim,
            n_heads * self.rope_dim,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )

        self.out_proj = BitLinear(
            d_model,
            d_model,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.rope = RotaryEmbedding(self.rope_dim, base=rope_base)
        self.attn_dropout = nn.Dropout(dropout)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.register_buffer("running_qk_max", torch.zeros(self.n_heads), persistent=False)
        self.register_buffer("last_qk_max", torch.zeros(self.n_heads), persistent=False)
        self.register_buffer("last_qk_clip_scale_mean", torch.tensor(1.0), persistent=False)
        self._qk_metric_step = 0
        self._qk_stats_initialized = False

    def _reshape_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        return x.view(bsz, seqlen, self.n_heads, head_dim)

    def _build_attn_bias(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        device: torch.device,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        causal = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]

        mask_value = -1e9
        bias = torch.zeros((batch_size, 1, q_len, kv_len), device=device, dtype=torch.float32)
        bias = bias.masked_fill(~causal, mask_value)

        if attention_mask is None:
            return bias

        keep_mask = attention_mask.bool()
        if keep_mask.ndim == 2:
            keep_mask = keep_mask[:, None, None, :]  # [B,1,1,K]
        elif keep_mask.ndim == 3:
            keep_mask = keep_mask[:, None, :, :]  # [B,1,Q,K]
        elif keep_mask.ndim != 4:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

        if keep_mask.size(-1) != kv_len:
            if keep_mask.size(-1) < kv_len:
                pad_len = kv_len - keep_mask.size(-1)
                pad_shape = (*keep_mask.shape[:-1], pad_len)
                keep_mask = torch.cat((torch.ones(pad_shape, dtype=torch.bool, device=device), keep_mask), dim=-1)
            else:
                keep_mask = keep_mask[..., -kv_len:]

        return bias.masked_fill(~keep_mask, mask_value)

    @torch.no_grad()
    def _update_qk_stats(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if self.qk_clip_ema_beta <= 0:
            return
        self._qk_metric_step += 1
        if (self._qk_metric_step % self.qk_clip_metric_interval) != 0:
            return

        q_norm_max = q.detach().float().norm(dim=-1).amax(dim=(0, 2))
        k_norm_max = k.detach().float().norm(dim=-1).amax(dim=(0, 2))
        qk_bound = (q_norm_max * k_norm_max) / (self.head_dim**0.5)
        qk_bound = qk_bound.to(self.running_qk_max.dtype)
        self.last_qk_max.copy_(qk_bound)

        if not self._qk_stats_initialized:
            self.running_qk_max.copy_(qk_bound)
            self._qk_stats_initialized = True
            return

        beta = float(self.qk_clip_ema_beta)
        self.running_qk_max.mul_(1.0 - beta).add_(qk_bound, alpha=beta)

    @torch.no_grad()
    def apply_qk_clip(
        self,
        threshold: float,
        strength: float = 1.0,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
        eps: float = 1e-6,
    ) -> float:
        if threshold <= 0:
            self.last_qk_clip_scale_mean.fill_(1.0)
            return 1.0
        if not self._qk_stats_initialized:
            self.last_qk_clip_scale_mean.fill_(1.0)
            return 1.0

        head_max = torch.maximum(self.running_qk_max.float(), self.last_qk_max.float())
        over = head_max > threshold
        if not over.any():
            self.last_qk_clip_scale_mean.fill_(1.0)
            return 1.0

        target = torch.sqrt((threshold / head_max.clamp_min(eps)).clamp(max=1.0))
        if strength > 0 and strength != 1.0:
            target = target.pow(strength)
        scales = torch.where(over, target, torch.ones_like(target))
        scales = scales.clamp(min=min_scale, max=max_scale)

        self.q_content.scale_output_groups_(scales, self.head_dim)
        self.q_rope.scale_output_groups_(scales, self.rope_dim)
        self.k_content_up.scale_output_groups_(scales, self.head_dim)
        self.k_rope_up.scale_output_groups_(scales, self.rope_dim)

        scale_mean = float(scales.mean().item())
        self.last_qk_clip_scale_mean.fill_(scale_mean)
        return scale_mean

    @torch.no_grad()
    def qk_metrics(self) -> dict[str, float]:
        if not self._qk_stats_initialized:
            return {
                "qk_max_mean": 0.0,
                "qk_max_peak": 0.0,
                "qk_clip_scale_mean": float(self.last_qk_clip_scale_mean.item()),
            }
        running = self.running_qk_max.float()
        return {
            "qk_max_mean": float(running.mean().item()),
            "qk_max_peak": float(running.max().item()),
            "qk_clip_scale_mean": float(self.last_qk_clip_scale_mean.item()),
        }

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = x.shape

        q_content = self._reshape_heads(self.q_content(x), self.head_dim)
        q_rope = self._reshape_heads(self.q_rope(x), self.rope_dim)

        latent_new = self.kv_down(x)  # [B, Q, L]
        if past_kv is not None:
            latent = torch.cat((past_kv[0], latent_new), dim=1)
            past_len = past_kv[0].size(1)
        else:
            latent = latent_new
            past_len = 0
        kv_len = latent.size(1)

        k_content = self._reshape_heads(self.k_content_up(latent), self.head_dim)
        v = self._reshape_heads(self.v_up(latent), self.head_dim)
        k_rope = self._reshape_heads(self.k_rope_up(latent), self.rope_dim)

        q_content = self.q_norm(q_content)
        k_content = self.k_norm(k_content)

        q_rope = self.rope.apply(q_rope, start_pos=past_len)
        k_rope = self.rope.apply(k_rope, start_pos=0)

        if self.rope_dim < self.head_dim:
            pad = self.head_dim - self.rope_dim
            q_rope = F.pad(q_rope, (0, pad))
            k_rope = F.pad(k_rope, (0, pad))

        q = (q_content + q_rope).transpose(1, 2)  # [B, H, Q, D]
        k = (k_content + k_rope).transpose(1, 2)  # [B, H, K, D]
        v = v.transpose(1, 2)  # [B, H, K, D]
        self._update_qk_stats(q, k)

        is_causal = False
        attn_bias: torch.Tensor | None = None
        if attention_mask is None and past_len == 0:
            # Fast-path: lets PyTorch route to flash/mem-efficient kernels.
            is_causal = True
        elif attention_mask is None and q_len == 1:
            # Single-token decode can attend all current KV without an explicit mask.
            attn_bias = None
        else:
            attn_bias = self._build_attn_bias(
                batch_size=bsz,
                q_len=q_len,
                kv_len=kv_len,
                past_len=past_len,
                device=x.device,
                attention_mask=attention_mask,
            )

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        attn_out = self.attn_dropout(self.out_proj(attn_out))

        next_cache = (latent.detach(),) if use_cache else None
        return attn_out, next_cache
