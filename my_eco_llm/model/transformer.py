from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadLatentAttention
from .bitlinear import BitLinear, RMSNorm
from .memory import ConditionalNgramMemory
from .moe import DeepSeekMoE


@dataclass
class EcoConfig:
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_seq_len: int = 2048
    kv_latent_dim: int = 128
    rope_dim: int | None = None
    dropout: float = 0.0

    d_ff: int = 2048
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    moe_top_k: int = 2
    moe_dispatch_mode: str = "grouped"
    router_jitter_noise: float = 0.01
    router_balance_lr: float = 1e-3
    router_metrics_interval: int = 10
    router_temperature_adaptive: bool = True
    router_temp_min: float = 0.8
    router_temp_max: float = 1.35
    router_precision: str = "bf16"

    recursive_depth: int = 0
    tie_embeddings: bool = False
    activation_checkpointing: bool = False
    qk_norm: bool = True
    qk_clip_ema_beta: float = 0.0
    qk_clip_threshold: float = 0.0
    qk_clip_strength: float = 1.0
    qk_clip_metric_interval: int = 10
    use_triton_bitlinear: bool = True
    bitlinear_quant_cache_training: bool = False
    mtp_tokens: int = 0
    mtp_loss_weight: float = 0.15
    moe_z_loss_weight: float = 0.0
    memory_slots: int = 0
    memory_ngram: int = 3
    memory_dropout: float = 0.0
    memory_metrics_interval: int = 10

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).")
        head_dim = self.d_model // self.n_heads
        if self.rope_dim is not None:
            if self.rope_dim <= 0:
                raise ValueError(f"rope_dim must be positive when set, got {self.rope_dim}.")
            if self.rope_dim > head_dim:
                raise ValueError(f"rope_dim ({self.rope_dim}) cannot exceed head_dim ({head_dim}).")
            if self.rope_dim % 2 != 0:
                raise ValueError(f"rope_dim must be even, got {self.rope_dim}.")
        if self.moe_top_k > self.n_routed_experts and self.n_routed_experts > 0:
            raise ValueError(
                f"moe_top_k ({self.moe_top_k}) cannot exceed n_routed_experts ({self.n_routed_experts})."
            )
        if str(self.moe_dispatch_mode).strip().lower() not in {"legacy", "packed", "grouped"}:
            raise ValueError(f"Invalid moe_dispatch_mode: {self.moe_dispatch_mode!r}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}.")
        if self.recursive_depth < 0:
            raise ValueError(f"recursive_depth must be >= 0, got {self.recursive_depth}.")
        if self.router_balance_lr < 0:
            raise ValueError(f"router_balance_lr must be >= 0, got {self.router_balance_lr}.")
        if self.router_metrics_interval <= 0:
            raise ValueError(f"router_metrics_interval must be > 0, got {self.router_metrics_interval}.")
        if self.router_temp_min <= 0:
            raise ValueError(f"router_temp_min must be > 0, got {self.router_temp_min}.")
        if self.router_temp_max < self.router_temp_min:
            raise ValueError(
                f"router_temp_max ({self.router_temp_max}) must be >= router_temp_min ({self.router_temp_min})."
            )
        if str(self.router_precision).strip().lower() not in {"bf16", "int8"}:
            raise ValueError(f"router_precision must be one of ['bf16', 'int8'], got {self.router_precision!r}")
        if self.mtp_tokens < 0:
            raise ValueError(f"mtp_tokens must be >= 0, got {self.mtp_tokens}.")
        if self.mtp_loss_weight < 0:
            raise ValueError(f"mtp_loss_weight must be >= 0, got {self.mtp_loss_weight}.")
        if self.qk_clip_ema_beta < 0 or self.qk_clip_ema_beta > 1:
            raise ValueError(f"qk_clip_ema_beta must be in [0, 1], got {self.qk_clip_ema_beta}.")
        if self.qk_clip_threshold < 0:
            raise ValueError(f"qk_clip_threshold must be >= 0, got {self.qk_clip_threshold}.")
        if self.qk_clip_strength < 0:
            raise ValueError(f"qk_clip_strength must be >= 0, got {self.qk_clip_strength}.")
        if self.qk_clip_metric_interval <= 0:
            raise ValueError(f"qk_clip_metric_interval must be > 0, got {self.qk_clip_metric_interval}.")
        if self.moe_z_loss_weight < 0:
            raise ValueError(f"moe_z_loss_weight must be >= 0, got {self.moe_z_loss_weight}.")
        if self.memory_slots < 0:
            raise ValueError(f"memory_slots must be >= 0, got {self.memory_slots}.")
        if self.memory_ngram <= 0:
            raise ValueError(f"memory_ngram must be > 0, got {self.memory_ngram}.")
        if self.memory_metrics_interval <= 0:
            raise ValueError(f"memory_metrics_interval must be > 0, got {self.memory_metrics_interval}.")


class EcoTransformerBlock(nn.Module):
    def __init__(self, config: EcoConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MultiHeadLatentAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            kv_latent_dim=config.kv_latent_dim,
            rope_dim=config.rope_dim,
            dropout=config.dropout,
            qk_norm=config.qk_norm,
            use_triton_bitlinear=config.use_triton_bitlinear,
            use_bitlinear_quant_cache_training=config.bitlinear_quant_cache_training,
            qk_clip_ema_beta=config.qk_clip_ema_beta,
            qk_clip_metric_interval=config.qk_clip_metric_interval,
        )

        self.moe_norm = RMSNorm(config.d_model)
        self.moe = DeepSeekMoE(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            top_k=config.moe_top_k,
            dispatch_mode=config.moe_dispatch_mode,
            dropout=config.dropout,
            router_jitter_noise=config.router_jitter_noise,
            router_balance_lr=config.router_balance_lr,
            router_metrics_interval=config.router_metrics_interval,
            router_temperature_adaptive=config.router_temperature_adaptive,
            router_temp_min=config.router_temp_min,
            router_temp_max=config.router_temp_max,
            router_precision=config.router_precision,
            use_triton_bitlinear=config.use_triton_bitlinear,
            use_bitlinear_quant_cache_training=config.bitlinear_quant_cache_training,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor] | None = None,
        use_cache: bool = False,
        disable_router_jitter: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor] | None]:
        attn_out, next_cache = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.moe(self.moe_norm(x), disable_jitter=disable_router_jitter)
        return x, next_cache


class EcoReasoningGPT(nn.Module):
    """BitLinear + MLA + MoE language model with optional recursive depth."""

    def __init__(
        self,
        config: EcoConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.memory = (
            ConditionalNgramMemory(
                d_model=config.d_model,
                memory_slots=config.memory_slots,
                ngram=config.memory_ngram,
                dropout=config.memory_dropout,
                metrics_interval=config.memory_metrics_interval,
            )
            if config.memory_slots > 0
            else None
        )

        if config.recursive_depth > 0:
            self.depth = config.recursive_depth
            self.shared_block = EcoTransformerBlock(config)
            self.blocks = None
        else:
            self.depth = config.n_layers
            self.shared_block = None
            self.blocks = nn.ModuleList([EcoTransformerBlock(config) for _ in range(config.n_layers)])

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = BitLinear(
            config.d_model,
            config.vocab_size,
            bias=False,
            input_rmsnorm=False,
            use_triton_kernel=config.use_triton_bitlinear,
            quant_cache_training=config.bitlinear_quant_cache_training,
        )
        self.mtp_heads = nn.ModuleList(
            [
                BitLinear(
                    config.d_model,
                    config.vocab_size,
                    bias=False,
                    input_rmsnorm=False,
                    use_triton_kernel=config.use_triton_bitlinear,
                    quant_cache_training=config.bitlinear_quant_cache_training,
                )
                for _ in range(config.mtp_tokens)
            ]
        )

        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def _block_at(self, idx: int) -> EcoTransformerBlock:
        if self.shared_block is not None:
            return self.shared_block
        assert self.blocks is not None
        return self.blocks[idx]

    def algorithmic_features(self) -> dict[str, int]:
        bitlinear_layers = sum(1 for module in self.modules() if isinstance(module, BitLinear))
        mla_layers = sum(1 for module in self.modules() if isinstance(module, MultiHeadLatentAttention))
        moe_layers = sum(1 for module in self.modules() if isinstance(module, DeepSeekMoE))
        return {
            "bitlinear_layers": bitlinear_layers,
            "mla_layers": mla_layers,
            "moe_layers": moe_layers,
            "recursive_depth": self.config.recursive_depth,
            "n_routed_experts": self.config.n_routed_experts,
            "n_shared_experts": self.config.n_shared_experts,
            "qk_norm": int(self.config.qk_norm),
            "qk_clip_threshold": self.config.qk_clip_threshold,
            "use_triton_bitlinear": int(self.config.use_triton_bitlinear),
            "bitlinear_quant_cache_training": int(self.config.bitlinear_quant_cache_training),
            "mtp_tokens": self.config.mtp_tokens,
            "router_metrics_interval": self.config.router_metrics_interval,
            "router_temperature_adaptive": int(self.config.router_temperature_adaptive),
            "moe_gate_precision_bf16": int(str(self.config.router_precision).strip().lower() == "bf16"),
            "moe_gate_precision_int8": int(str(self.config.router_precision).strip().lower() == "int8"),
            "moe_z_loss_weight": self.config.moe_z_loss_weight,
            "memory_slots": self.config.memory_slots,
            "memory_ngram": self.config.memory_ngram,
        }

    @torch.no_grad()
    def collect_moe_metrics(self) -> dict[str, float]:
        moe_modules = [m for m in self.modules() if isinstance(m, DeepSeekMoE)]
        if not moe_modules:
            return {}
        metrics = [m.router_metrics() for m in moe_modules]
        keys = metrics[0].keys()
        return {key: float(sum(m[key] for m in metrics) / len(metrics)) for key in keys}

    @torch.no_grad()
    def collect_qk_metrics(self) -> dict[str, float]:
        attn_modules = [m for m in self.modules() if isinstance(m, MultiHeadLatentAttention)]
        if not attn_modules:
            return {}
        metrics = [m.qk_metrics() for m in attn_modules]
        keys = metrics[0].keys()
        return {key: float(sum(m[key] for m in metrics) / len(metrics)) for key in keys}

    @torch.no_grad()
    def collect_memory_metrics(self) -> dict[str, float]:
        if self.memory is None:
            return {}
        return self.memory.metrics()

    def collect_aux_loss(self) -> torch.Tensor:
        moe_modules = [m for m in self.modules() if isinstance(m, DeepSeekMoE)]
        if not moe_modules:
            return self.token_emb.weight.new_zeros(())
        losses = [m.aux_loss() for m in moe_modules]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def apply_qk_clip(self) -> float:
        if self.config.qk_clip_threshold <= 0:
            return 1.0
        attn_modules = [m for m in self.modules() if isinstance(m, MultiHeadLatentAttention)]
        if not attn_modules:
            return 1.0
        scales = [
            m.apply_qk_clip(
                threshold=self.config.qk_clip_threshold,
                strength=self.config.qk_clip_strength,
            )
            for m in attn_modules
        ]
        return float(sum(scales) / len(scales))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor] | None] | None = None,
        use_cache: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[tuple[torch.Tensor] | None]]:
        x = self.token_emb(input_ids)
        if self.memory is not None:
            x = self.memory(input_ids, x)
        x = self.dropout(x)

        if past_key_values is None:
            past_key_values = [None] * self.depth
        if len(past_key_values) != self.depth:
            raise ValueError(f"Expected {self.depth} cache entries, got {len(past_key_values)}.")

        next_key_values: list[tuple[torch.Tensor] | None] = []
        for i in range(self.depth):
            block = self._block_at(i)
            if self.config.activation_checkpointing and self.training and not use_cache:
                # Checkpoint attention only; MoE routing is dynamic and can invalidate
                # full-block checkpoint metadata due expert dispatch shape changes.
                def checkpointed_attn(t: torch.Tensor) -> torch.Tensor:
                    attn_out, _ = block.attn(
                        block.attn_norm(t),
                        attention_mask=attention_mask,
                        past_kv=None,
                        use_cache=False,
                    )
                    return attn_out

                x = x + checkpoint(checkpointed_attn, x, use_reentrant=False)
                x = x + block.moe(block.moe_norm(x), disable_jitter=False)
                next_cache = None
            else:
                x, next_cache = block(
                    x,
                    attention_mask=attention_mask,
                    past_kv=past_key_values[i],
                    use_cache=use_cache,
                    disable_router_jitter=False,
                )
            if use_cache:
                next_key_values.append(next_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if return_aux and use_cache:
            raise ValueError("return_aux=True is not supported with use_cache=True.")
        if return_aux and self.config.mtp_tokens > 0:
            aux_logits: list[torch.Tensor] = []
            for k, head in enumerate(self.mtp_heads, start=1):
                if x.size(1) <= k:
                    break
                aux_logits.append(head(x[:, :-k, :]))
            return logits, aux_logits
        if use_cache:
            return logits, next_key_values
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        cache: list[tuple[torch.Tensor] | None] | None = None

        for _ in range(max_new_tokens):
            model_input = generated if cache is None else generated[:, -1:]
            out = self(
                model_input,
                past_key_values=cache,
                use_cache=True,
            )
            logits, cache = out
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                cutoff = values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated
