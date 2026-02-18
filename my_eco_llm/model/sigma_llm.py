from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadLatentAttention
from .bitlinear import BitLinear, RMSNorm
from .moe import DeepSeekMoE
from .sigma_kernels import TRITON_AVAILABLE as SIGMA_TRITON_AVAILABLE
from .sigma_kernels import TRITON_IMPORT_ERROR as SIGMA_TRITON_IMPORT_ERROR
from .sigma_kernels import mamba3_complex_scan_interleaved


@dataclass
class SigmaConfig:
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 18
    max_seq_len: int = 4096
    dropout: float = 0.0
    kv_latent_dim: int = 128
    d_ff: int = 3072
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    moe_top_k: int = 2
    moe_dispatch_mode: str = "grouped"
    router_balance_lr: float = 1e-3
    router_metrics_interval: int = 10
    router_jitter_noise: float = 0.01
    router_precision: str = "bf16"
    rib_router_enabled: bool = True
    rib_info_weight: float = 0.03
    rib_collapse_penalty: float = 0.08
    rib_confidence_weight: float = 0.03
    rib_temp_gain: float = 0.55
    use_triton_bitlinear: bool = True
    bitlinear_quant_cache_training: bool = True

    # Mamba-3 MIMO
    mamba_ratio: int = 5
    attention_ratio: int = 1
    mimo_rank: int = 4
    mamba_block_t: int = 32
    mamba_block_n: int = 128
    mamba_step_scale: float = 0.05

    # INSTANT protocol defaults (must be applied by trainer)
    instant_enabled: bool = True
    instant_comp_dim: int = 64
    instant_error_threshold: float = 0.01
    instant_reversible_iters: int = 8

    # TTRL defaults
    ttrl_group_size: int = 5
    ttrl_interval: int = 8
    ttrl_refine_iters: int = 3

    # Differential MLA defaults
    diff_attention_lambda_init: float = 0.35
    diff_attention_lambda_min: float = 0.05
    diff_attention_lambda_max: float = 1.20

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.mamba_ratio <= 0 or self.attention_ratio <= 0:
            raise ValueError("mamba_ratio and attention_ratio must both be > 0.")
        period = self.mamba_ratio + self.attention_ratio
        full_cycles = self.n_layers // period
        rem = self.n_layers % period
        mamba_layers = (full_cycles * self.mamba_ratio) + min(rem, self.mamba_ratio)
        mla_layers = self.n_layers - mamba_layers
        if mamba_layers <= 0 or mla_layers <= 0:
            raise ValueError(
                "SIGMA requires at least one Mamba block and one MLA block. "
                f"got n_layers={self.n_layers}, mamba_ratio={self.mamba_ratio}, attention_ratio={self.attention_ratio} "
                f"-> mamba_layers={mamba_layers}, mla_layers={mla_layers}. "
                f"Use n_layers >= {period} for a full ratio cycle."
            )
        if self.mimo_rank <= 0:
            raise ValueError("mimo_rank must be > 0.")
        if self.mamba_block_t <= 0:
            raise ValueError("mamba_block_t must be > 0.")
        if self.mamba_block_n <= 0:
            raise ValueError("mamba_block_n must be > 0.")
        if self.mamba_step_scale <= 0 or self.mamba_step_scale >= 1:
            raise ValueError("mamba_step_scale must be in (0, 1) for reversible contraction.")
        if self.instant_comp_dim <= 0:
            raise ValueError("instant_comp_dim must be > 0.")
        if self.instant_reversible_iters <= 0:
            raise ValueError("instant_reversible_iters must be > 0.")
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be > 0 for SIGMA.")
        if str(self.router_precision).strip().lower() not in {"bf16", "int8"}:
            raise ValueError(f"router_precision must be one of ['bf16', 'int8'], got {self.router_precision!r}")
        if str(self.moe_dispatch_mode).strip().lower() not in {"legacy", "packed", "grouped"}:
            raise ValueError(f"Invalid moe_dispatch_mode: {self.moe_dispatch_mode!r}")
        if self.ttrl_group_size <= 1:
            raise ValueError("ttrl_group_size must be > 1.")
        if self.ttrl_refine_iters <= 0:
            raise ValueError("ttrl_refine_iters must be > 0.")
        if self.rib_info_weight < 0 or self.rib_collapse_penalty < 0 or self.rib_confidence_weight < 0:
            raise ValueError("RIB weights must be >= 0.")
        if self.diff_attention_lambda_min < 0:
            raise ValueError("diff_attention_lambda_min must be >= 0.")
        if self.diff_attention_lambda_max <= self.diff_attention_lambda_min:
            raise ValueError("diff_attention_lambda_max must be > diff_attention_lambda_min.")
        if not (self.diff_attention_lambda_min <= self.diff_attention_lambda_init <= self.diff_attention_lambda_max):
            raise ValueError(
                "diff_attention_lambda_init must be within "
                "[diff_attention_lambda_min, diff_attention_lambda_max]."
            )


class SigmaMambaCore(nn.Module):
    """
    Mamba-3 style complex SSM core with MIMO state.

    State layout is interleaved complex: [real, imag, real, imag, ...].
    """

    def __init__(self, config: SigmaConfig) -> None:
        super().__init__()
        self.d_model = int(config.d_model)
        self.mimo_rank = int(config.mimo_rank)
        self.block_t = int(config.mamba_block_t)
        self.block_n = int(config.mamba_block_n)
        self.step_scale = float(config.mamba_step_scale)

        self.norm = RMSNorm(self.d_model)
        self.in_proj = BitLinear(
            self.d_model,
            self.d_model * self.mimo_rank * 2,
            bias=False,
            use_triton_kernel=config.use_triton_bitlinear,
            quant_cache_training=config.bitlinear_quant_cache_training,
        )
        self.out_proj = BitLinear(
            self.d_model,
            self.d_model,
            bias=False,
            use_triton_kernel=config.use_triton_bitlinear,
            quant_cache_training=config.bitlinear_quant_cache_training,
        )

        # Complex state transition parameters per channel/rank.
        self.a_param = nn.Parameter(torch.empty(self.d_model, self.mimo_rank, 2))
        self.b_param = nn.Parameter(torch.empty(self.d_model, self.mimo_rank, 2))
        self.c_param = nn.Parameter(torch.empty(self.d_model, self.mimo_rank, 2))
        self._init_params()
        self._state_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._last_effective_block_t: float = float(self.block_t)
        self._last_scan_chunks: float = 0.0

    def _init_params(self) -> None:
        nn.init.normal_(self.a_param, mean=0.0, std=0.12)
        nn.init.normal_(self.b_param, mean=0.0, std=0.15)
        nn.init.normal_(self.c_param, mean=0.0, std=0.15)

    def set_kernel_tiling(self, block_t: int, block_n: int) -> None:
        if block_t <= 0 or block_n <= 0:
            raise ValueError(f"Invalid kernel tiling: block_t={block_t}, block_n={block_n}")
        self.block_t = int(block_t)
        self.block_n = int(block_n)

    def _complex_coefficients(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep |a| < 1 for stable oscillating dynamics.
        a = torch.tanh(self.a_param.float()) * 0.97
        b = torch.tanh(self.b_param.float())
        c = torch.tanh(self.c_param.float())
        a = a.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().to(device=device)
        b = b.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().to(device=device)
        c = c.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().to(device=device)
        return a, b, c

    def _effective_block_t(self, seq_len: int) -> int:
        base = max(1, int(self.block_t))
        # Prefer larger static scan tiles to reduce kernel launches.
        # Padding is already handled by forward(), so exact divisibility is not required.
        upper = max(base, min(int(seq_len), 256))
        if int(seq_len) >= 1024:
            ordered = (256, 128, 64, 32, 16)
        else:
            # On common short/medium contexts (e.g. 512), 128 is typically faster than 256.
            ordered = (128, 64, 32, 16)
        for candidate in ordered:
            if candidate < base or candidate > upper:
                continue
            return int(candidate)
        return int(min(max(base, 1), max(int(seq_len), 1)))

    def _zero_state(self, *, lanes: int, device: torch.device) -> torch.Tensor:
        key = (device, int(lanes))
        buf = self._state_cache.get(key)
        if buf is None or buf.device != device or buf.numel() != (lanes * 2):
            buf = torch.empty((lanes, 2), device=device, dtype=torch.float32)
            self._state_cache[key] = buf
            if len(self._state_cache) > 8:
                self._state_cache.pop(next(iter(self._state_cache)))
        buf.zero_()
        return buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SigmaMambaCore expects [B,T,D], got {tuple(x.shape)}")
        bsz, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")
        if not x.is_cuda:
            raise RuntimeError("SigmaMambaCore requires CUDA execution.")
        if not SIGMA_TRITON_AVAILABLE:
            raise RuntimeError(f"SigmaMambaCore requires Triton kernel. detail={SIGMA_TRITON_IMPORT_ERROR}")

        x_norm = self.norm(x)
        u = self.in_proj(x_norm).view(bsz, seq_len, d_model, self.mimo_rank, 2)
        u = u.to(torch.bfloat16)

        a, b, c = self._complex_coefficients(batch_size=bsz, device=x.device)
        lanes = bsz * d_model * self.mimo_rank
        a_i = a.view(lanes, 2).contiguous()
        b_i = b.view(lanes, 2).contiguous()
        u_i = u.view(lanes, seq_len, 2).contiguous()
        state = self._zero_state(lanes=lanes, device=x.device)
        chunk = self._effective_block_t(seq_len=seq_len)
        if (seq_len % chunk) != 0:
            pad_t = chunk - (seq_len % chunk)
            u_i = F.pad(u_i, (0, 0, 0, pad_t))
        else:
            pad_t = 0
        seq_work = int(u_i.size(1))
        out_i = torch.empty_like(u_i)
        scan_chunks = 0

        for start in range(0, seq_work, chunk):
            x_chunk = u_i[:, start : start + chunk, :].contiguous()
            y_chunk, state = mamba3_complex_scan_interleaved(
                x_chunk,
                a_i,
                b_i,
                state,
                block_t=chunk,
                block_n=self.block_n,
            )
            out_i[:, start : start + chunk, :] = y_chunk
            scan_chunks += 1

        if pad_t > 0:
            out_i = out_i[:, :seq_len, :]
        self._last_effective_block_t = float(chunk)
        self._last_scan_chunks = float(scan_chunks)

        h = out_i.reshape(bsz, seq_len, d_model, self.mimo_rank, 2).to(dtype=x.dtype)
        c = c.to(dtype=h.dtype).unsqueeze(1)
        y_real = (h[..., 0] * c[..., 0]) - (h[..., 1] * c[..., 1])
        y = y_real.sum(dim=3) / math.sqrt(float(self.mimo_rank))
        return self.step_scale * self.out_proj(y)

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return {
            "mamba_effective_block_t": float(self._last_effective_block_t),
            "mamba_scan_chunks": float(self._last_scan_chunks),
        }


class SigmaMambaBlock(nn.Module):
    is_sigma_mamba_block = True

    def __init__(self, config: SigmaConfig) -> None:
        super().__init__()
        self.core = SigmaMambaCore(config)
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = DeepSeekMoE(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            top_k=config.moe_top_k,
            dropout=config.dropout,
            router_jitter_noise=config.router_jitter_noise,
            router_balance_lr=config.router_balance_lr,
            router_metrics_interval=config.router_metrics_interval,
            router_precision=config.router_precision,
            dispatch_mode=config.moe_dispatch_mode,
            rib_enabled=config.rib_router_enabled,
            rib_info_weight=config.rib_info_weight,
            rib_collapse_penalty=config.rib_collapse_penalty,
            rib_confidence_weight=config.rib_confidence_weight,
            rib_temp_gain=config.rib_temp_gain,
            use_triton_bitlinear=config.use_triton_bitlinear,
            use_bitlinear_quant_cache_training=config.bitlinear_quant_cache_training,
        )
        self.moe_gate = nn.Parameter(torch.tensor(0.0))
        self.instant_reversible: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.instant_reversible is not None:
            x = self.instant_reversible(x)
        else:
            x = x + self.core(x)
        moe_scale = torch.sigmoid(self.moe_gate).to(dtype=x.dtype)
        x = x + (moe_scale * self.moe(self.moe_norm(x), disable_jitter=not self.training))
        return x


class SigmaMLABlock(nn.Module):
    is_sigma_mla_block = True

    def __init__(self, config: SigmaConfig) -> None:
        super().__init__()
        self.d_model = int(config.d_model)
        self._diff_lambda_min = float(config.diff_attention_lambda_min)
        self._diff_lambda_max = float(config.diff_attention_lambda_max)
        self.norm = RMSNorm(config.d_model)
        self.attn_exp = MultiHeadLatentAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            kv_latent_dim=config.kv_latent_dim,
            rope_dim=None,
            dropout=config.dropout,
            qk_norm=True,
            use_triton_bitlinear=config.use_triton_bitlinear,
            use_bitlinear_quant_cache_training=config.bitlinear_quant_cache_training,
            qk_clip_ema_beta=0.0,
            qk_clip_metric_interval=10,
        )
        self.attn_ref = MultiHeadLatentAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            kv_latent_dim=config.kv_latent_dim,
            rope_dim=None,
            dropout=config.dropout,
            qk_norm=True,
            use_triton_bitlinear=config.use_triton_bitlinear,
            use_bitlinear_quant_cache_training=config.bitlinear_quant_cache_training,
            qk_clip_ema_beta=0.0,
            qk_clip_metric_interval=10,
        )
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = DeepSeekMoE(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            top_k=config.moe_top_k,
            dropout=config.dropout,
            router_jitter_noise=config.router_jitter_noise,
            router_balance_lr=config.router_balance_lr,
            router_metrics_interval=config.router_metrics_interval,
            router_precision=config.router_precision,
            dispatch_mode=config.moe_dispatch_mode,
            rib_enabled=config.rib_router_enabled,
            rib_info_weight=config.rib_info_weight,
            rib_collapse_penalty=config.rib_collapse_penalty,
            rib_confidence_weight=config.rib_confidence_weight,
            rib_temp_gain=config.rib_temp_gain,
            use_triton_bitlinear=config.use_triton_bitlinear,
            use_bitlinear_quant_cache_training=config.bitlinear_quant_cache_training,
        )
        self.moe_gate = nn.Parameter(torch.tensor(0.0))
        self.instant_compressor: Any = None
        init = (float(config.diff_attention_lambda_init) - self._diff_lambda_min) / (
            self._diff_lambda_max - self._diff_lambda_min
        )
        init = min(max(init, 1e-4), 1.0 - 1e-4)
        self.diff_lambda_logit = nn.Parameter(torch.tensor(math.log(init / (1.0 - init))))

    def diff_lambda(self) -> torch.Tensor:
        span = self._diff_lambda_max - self._diff_lambda_min
        return self._diff_lambda_min + (span * torch.sigmoid(self.diff_lambda_logit))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.instant_compressor is not None and self.training:
            h = self.instant_compressor.roundtrip(h)
        attn_exp, _ = self.attn_exp(h, use_cache=False)
        attn_ref, _ = self.attn_ref(h, use_cache=False)
        lam = self.diff_lambda().to(dtype=h.dtype)
        x = x + (attn_exp - (lam * attn_ref))
        moe_scale = torch.sigmoid(self.moe_gate).to(dtype=x.dtype)
        x = x + (moe_scale * self.moe(self.moe_norm(x), disable_jitter=not self.training))
        return x


class SigmaLLM(nn.Module):
    """
    SIGMA stack model:
      - 5:1 Mamba-3 MIMO : MLA block ratio
      - BitLinear projections
      - DeepSeekMoE in all blocks
      - INSTANT hooks (strictly required when config.instant_enabled=True)
    """

    def __init__(self, config: SigmaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self._instant_patched = False

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList()
        period = config.mamba_ratio + config.attention_ratio
        for i in range(config.n_layers):
            if (i % period) < config.mamba_ratio:
                self.blocks.append(SigmaMambaBlock(config))
            else:
                self.blocks.append(SigmaMLABlock(config))

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = BitLinear(
            config.d_model,
            config.vocab_size,
            bias=False,
            input_rmsnorm=False,
            use_triton_kernel=config.use_triton_bitlinear,
            quant_cache_training=config.bitlinear_quant_cache_training,
        )

    def mark_instant_patched(self) -> None:
        self._instant_patched = True

    def algorithmic_features(self) -> dict[str, int]:
        bitlinear_layers = sum(1 for m in self.modules() if isinstance(m, BitLinear))
        mla_layers = sum(1 for m in self.modules() if isinstance(m, MultiHeadLatentAttention))
        moe_layers = sum(1 for m in self.modules() if isinstance(m, DeepSeekMoE))
        mamba_layers = sum(1 for m in self.modules() if isinstance(m, SigmaMambaBlock))
        return {
            "bitlinear_layers": int(bitlinear_layers),
            "bitnet_1_58_active": 1,
            "mla_layers": int(mla_layers),
            "moe_layers": int(moe_layers),
            "mamba3_layers": int(mamba_layers),
            "mamba_ratio": int(self.config.mamba_ratio),
            "attention_ratio": int(self.config.attention_ratio),
            "mimo_rank": int(self.config.mimo_rank),
            "instant_enabled": int(self.config.instant_enabled),
            "rib_router_enabled": int(self.config.rib_router_enabled),
            "moe_gate_precision_bf16": int(str(self.config.router_precision).strip().lower() == "bf16"),
            "moe_gate_precision_int8": int(str(self.config.router_precision).strip().lower() == "int8"),
            "differential_mla_enabled": 1,
            "triton_sigma_kernel": int(SIGMA_TRITON_AVAILABLE),
        }

    @torch.no_grad()
    def collect_moe_metrics(self) -> dict[str, float]:
        moe_modules = [m for m in self.modules() if isinstance(m, DeepSeekMoE)]
        if not moe_modules:
            return {}
        metrics = [m.router_metrics() for m in moe_modules]
        keys = metrics[0].keys()
        return {k: float(sum(d[k] for d in metrics) / len(metrics)) for k in keys}

    @torch.no_grad()
    def collect_instant_metrics(self) -> dict[str, float]:
        comp_errors: list[float] = []
        comp_bits: list[float] = []
        for block in self.blocks:
            compressor = getattr(block, "instant_compressor", None)
            if compressor is None:
                continue
            metrics = compressor.metrics()
            comp_errors.append(float(metrics.get("instant_reconstruction_error", 0.0)))
            comp_bits.append(float(metrics.get("instant_error_bit_width", 0.0)))
        if not comp_errors:
            return {}
        return {
            "instant_reconstruction_error": float(sum(comp_errors) / len(comp_errors)),
            "instant_error_bit_width": float(sum(comp_bits) / len(comp_bits)),
        }

    @torch.no_grad()
    def collect_attention_metrics(self) -> dict[str, float]:
        lambdas: list[float] = []
        for block in self.blocks:
            if isinstance(block, SigmaMLABlock):
                lambdas.append(float(block.diff_lambda().detach().item()))
        if not lambdas:
            return {}
        return {
            "diff_attn_lambda_mean": float(sum(lambdas) / len(lambdas)),
            "diff_attn_lambda_min": float(min(lambdas)),
            "diff_attn_lambda_max": float(max(lambdas)),
        }

    @torch.no_grad()
    def collect_mamba_metrics(self) -> dict[str, float]:
        vals: list[dict[str, float]] = []
        for block in self.blocks:
            if isinstance(block, SigmaMambaBlock):
                vals.append(block.core.metrics())
        if not vals:
            return {}
        keys = vals[0].keys()
        return {k: float(sum(v[k] for v in vals) / len(vals)) for k in keys}

    def collect_aux_loss(self) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for m in self.modules():
            if isinstance(m, DeepSeekMoE):
                losses.append(m.aux_loss())
        if not losses:
            return self.token_emb.weight.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.instant_enabled and not self._instant_patched:
            raise RuntimeError("INSTANT is required but SigmaLLM was not patched. Apply training.memory_hack first.")
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            if out.size(1) > self.config.max_seq_len:
                out = out[:, -self.config.max_seq_len :]
            logits = self(out)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                k = min(int(top_k), int(next_logits.size(-1)))
                v, _ = torch.topk(next_logits, k=k, dim=-1)
                cutoff = v[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probs = F.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, nxt), dim=1)
        self.train()
        return out
