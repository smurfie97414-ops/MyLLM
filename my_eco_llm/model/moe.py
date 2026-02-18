from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear, RMSNorm

_DYNAMO_DISABLE = (
    torch.compiler.disable
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable")
    else torch._dynamo.disable
)


def _is_compiling() -> bool:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        try:
            return bool(torch.compiler.is_compiling())
        except Exception:
            return False
    return False


class ExpertMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        use_triton_bitlinear: bool = True,
        use_bitlinear_quant_cache_training: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = BitLinear(
            d_model,
            d_ff,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.fc2 = BitLinear(
            d_ff,
            d_model,
            bias=False,
            use_triton_kernel=use_triton_bitlinear,
            quant_cache_training=use_bitlinear_quant_cache_training,
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style MoE:
    - Shared experts are always active.
    - Routed experts are sparsely activated via top-k routing.
    - Jitter noise is added to router logits during training to prevent collapse.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_shared_experts: int,
        n_routed_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
        router_jitter_noise: float = 0.01,
        router_balance_lr: float = 1e-3,
        router_metrics_interval: int = 10,
        router_temperature_adaptive: bool = True,
        router_temp_min: float = 0.8,
        router_temp_max: float = 1.35,
        dispatch_mode: str = "grouped",
        rib_enabled: bool = True,
        rib_info_weight: float = 0.03,
        rib_collapse_penalty: float = 0.08,
        rib_confidence_weight: float = 0.03,
        rib_temp_gain: float = 0.55,
        router_precision: str = "bf16",
        use_triton_bitlinear: bool = True,
        use_bitlinear_quant_cache_training: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.top_k = min(top_k, n_routed_experts) if n_routed_experts > 0 else 0
        self.router_jitter_noise = router_jitter_noise
        self.router_balance_lr = router_balance_lr
        self.router_metrics_interval = max(1, int(router_metrics_interval))
        self.router_temperature_adaptive = bool(router_temperature_adaptive)
        self.router_temp_min = float(router_temp_min)
        self.router_temp_max = float(router_temp_max)
        dispatch_mode_norm = str(dispatch_mode).strip().lower()
        valid_dispatch = {"legacy", "packed", "grouped"}
        if dispatch_mode_norm not in valid_dispatch:
            raise ValueError(f"dispatch_mode must be one of {sorted(valid_dispatch)}, got {dispatch_mode!r}")
        self.dispatch_mode = dispatch_mode_norm
        self.rib_enabled = bool(rib_enabled)
        self.rib_info_weight = float(rib_info_weight)
        self.rib_collapse_penalty = float(rib_collapse_penalty)
        self.rib_confidence_weight = float(rib_confidence_weight)
        self.rib_temp_gain = float(rib_temp_gain)
        router_precision_norm = str(router_precision).strip().lower()
        if router_precision_norm not in {"bf16", "int8"}:
            raise ValueError(f"router_precision must be one of ['bf16', 'int8'], got {router_precision!r}")
        self.router_precision = router_precision_norm
        if self.router_temp_min <= 0:
            raise ValueError(f"router_temp_min must be > 0, got {self.router_temp_min}")
        if self.router_temp_max < self.router_temp_min:
            raise ValueError(
                f"router_temp_max ({self.router_temp_max}) must be >= router_temp_min ({self.router_temp_min})"
            )

        self.pre_norm = RMSNorm(d_model)
        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    d_model,
                    d_ff,
                    dropout,
                    use_triton_bitlinear=use_triton_bitlinear,
                    use_bitlinear_quant_cache_training=use_bitlinear_quant_cache_training,
                )
                for _ in range(n_shared_experts)
            ]
        )
        self.routed_experts = nn.ModuleList(
            [
                ExpertMLP(
                    d_model,
                    d_ff,
                    dropout,
                    use_triton_bitlinear=use_triton_bitlinear,
                    use_bitlinear_quant_cache_training=use_bitlinear_quant_cache_training,
                )
                for _ in range(n_routed_experts)
            ]
        )
        self.router = nn.Linear(d_model, n_routed_experts, bias=False) if n_routed_experts > 0 else None
        if self.router is not None and self.router_precision == "bf16":
            self.router = self.router.to(dtype=torch.bfloat16)

        self.register_buffer("expert_usage_ema", torch.zeros(max(1, n_routed_experts)), persistent=False)
        self.register_buffer("router_bias", torch.zeros(max(1, n_routed_experts)), persistent=False)
        self.register_buffer("last_router_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_router_z_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_routed_fraction", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_router_temperature", torch.tensor(1.0), persistent=False)
        self.register_buffer("last_rib_retained_info", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_rib_compression", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_rib_collapse", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_rib_aux_loss", torch.tensor(0.0), persistent=False)
        self._router_metric_step = 0
        self._aux_z_loss: torch.Tensor | None = None
        self._last_stage_router_s = 0.0
        self._last_stage_topk_s = 0.0
        self._last_stage_dispatch_s = 0.0
        self._last_stage_expert_s = 0.0
        self._last_stage_scatter_s = 0.0
        self._stage_ema_beta = 0.9
        self._token_id_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}

    @contextmanager
    def _nvtx(self, name: str):
        pushed = False
        if torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(name)
                pushed = True
            except Exception:
                pushed = False
        try:
            yield
        finally:
            if pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    def _update_stage_ema(self, key: str, value: float) -> None:
        beta = float(self._stage_ema_beta)
        old = float(getattr(self, key))
        setattr(self, key, (beta * old) + ((1.0 - beta) * max(0.0, float(value))))

    def _flat_token_ids(self, *, flat_slots: int, device: torch.device) -> torch.Tensor:
        key = (device, int(flat_slots), int(self.top_k))
        cached = self._token_id_cache.get(key)
        if cached is not None:
            return cached
        slot_ids = torch.arange(flat_slots, device=device, dtype=torch.long)
        token_ids = torch.div(slot_ids, max(1, int(self.top_k)), rounding_mode="floor")
        self._token_id_cache[key] = token_ids
        if len(self._token_id_cache) > 32:
            self._token_id_cache.pop(next(iter(self._token_id_cache)))
        return token_ids

    def _dispatch_legacy(
        self,
        flat_x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_gates: torch.Tensor,
        routed_out: torch.Tensor,
    ) -> None:
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_gates = topk_gates.reshape(-1, self.top_k)
        for expert_id, expert in enumerate(self.routed_experts):
            selected = flat_indices == expert_id
            token_mask = selected.any(dim=-1)
            token_ids = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if token_ids.numel() == 0:
                continue
            expert_out = expert(flat_x[token_ids])
            selected_slots = selected[token_ids].to(flat_gates.dtype)
            gate = (flat_gates[token_ids] * selected_slots).sum(dim=-1, keepdim=True)
            routed_out[token_ids] = routed_out[token_ids] + (expert_out * gate)

    def _dispatch_packed(
        self,
        flat_x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_gates: torch.Tensor,
        routed_out: torch.Tensor,
    ) -> None:
        flat_indices = topk_indices.reshape(-1)
        flat_gates = topk_gates.reshape(-1, 1)
        n_tokens = topk_indices.size(0) * topk_indices.size(1)
        token_ids = torch.arange(n_tokens, device=flat_x.device).repeat_interleave(self.top_k)

        # Sort by expert id so each expert sees a compact token slice.
        order = torch.argsort(flat_indices)
        sorted_experts = flat_indices[order]
        sorted_tokens = token_ids[order]
        sorted_gates = flat_gates[order]

        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        start = 0
        for expert_id, count in zip(unique_experts.tolist(), counts.tolist()):
            stop = start + count
            expert_token_ids = sorted_tokens[start:stop]
            expert_gates = sorted_gates[start:stop]
            expert_out = self.routed_experts[expert_id](flat_x[expert_token_ids])
            routed_out.index_add_(0, expert_token_ids, expert_out * expert_gates)
            start = stop

    @_DYNAMO_DISABLE
    def _dispatch_grouped(
        self,
        flat_x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_gates: torch.Tensor,
        routed_out: torch.Tensor,
        *,
        collect_timing: bool,
    ) -> tuple[float, float]:
        """
        Behavior-preserving grouped dispatch:
        - same top-k assignments and gates
        - deterministic expert bucketing
        - reduced overhead from repeated allocations/shape transforms
        """
        t_expert = 0.0
        t_scatter = 0.0
        flat_indices = topk_indices.reshape(-1)
        flat_gates = topk_gates.reshape(-1, 1)
        flat_slots = int(flat_indices.numel())
        token_ids = self._flat_token_ids(flat_slots=flat_slots, device=flat_x.device)

        # Stable sort preserves deterministic behavior for equal expert IDs.
        sorted_experts, order = torch.sort(flat_indices, stable=True)
        sorted_tokens = token_ids.index_select(0, order)
        sorted_gates = flat_gates.index_select(0, order)

        counts = torch.bincount(sorted_experts, minlength=self.n_routed_experts)
        offsets = torch.cumsum(counts, dim=0)
        start = 0
        for expert_id in range(self.n_routed_experts):
            stop = int(offsets[expert_id].item())
            if stop <= start:
                continue
            expert_token_ids = sorted_tokens[start:stop]
            expert_gates = sorted_gates[start:stop]

            te0 = time.perf_counter() if collect_timing else 0.0
            expert_in = flat_x.index_select(0, expert_token_ids)
            expert_out = self.routed_experts[expert_id](expert_in)
            if collect_timing:
                t_expert += max(time.perf_counter() - te0, 0.0)

            ts0 = time.perf_counter() if collect_timing else 0.0
            routed_out.index_add_(0, expert_token_ids, expert_out * expert_gates)
            if collect_timing:
                t_scatter += max(time.perf_counter() - ts0, 0.0)
            start = stop
        return t_expert, t_scatter

    @_DYNAMO_DISABLE
    def forward(self, x: torch.Tensor, disable_jitter: bool = False) -> torch.Tensor:
        collect_timing = not _is_compiling()
        x = self.pre_norm(x)

        shared_out = torch.zeros_like(x)
        if self.n_shared_experts > 0:
            for expert in self.shared_experts:
                shared_out = shared_out + expert(x)
            shared_out = shared_out / float(self.n_shared_experts)

        if self.n_routed_experts == 0:
            self._aux_z_loss = x.new_zeros(())
            return shared_out
        if self.top_k == 0:
            self.last_routed_fraction.fill_(0.0)
            self._aux_z_loss = x.new_zeros(())
            return shared_out

        assert self.router is not None
        t0 = time.perf_counter() if collect_timing else 0.0
        router_ctx = self._nvtx("moe.router") if collect_timing else nullcontext()
        with router_ctx:
            if self.router_precision == "bf16":
                router_logits = self.router(x.to(dtype=torch.bfloat16)).float()
            else:
                # Keep MoE gate in INT8-safe path (fake-quantized matmul with fp master weights).
                router_weight = self.router.weight.float()
                scale = router_weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 127.0
                q = torch.round(router_weight / scale).clamp(-127, 127)
                deq = q * scale
                router_logits = F.linear(x.float(), deq, bias=None)
            router_logits = router_logits + self.router_bias[: self.n_routed_experts].view(1, 1, -1)
        if collect_timing:
            self._update_stage_ema("_last_stage_router_s", time.perf_counter() - t0)
        probs_for_temp = F.softmax(router_logits, dim=-1)
        if self.training and self.router_temperature_adaptive and self.n_routed_experts > 1:
            usage = self.expert_usage_ema[: self.n_routed_experts].float()
            usage = usage / usage.sum().clamp_min(1e-9)
            imbalance = (usage.max() - usage.min()).clamp(min=0.0, max=1.0)
            entropy = -(probs_for_temp * probs_for_temp.clamp_min(1e-9).log()).sum(dim=-1).mean()
            entropy_norm = entropy / max(math.log(max(self.n_routed_experts, 2)), 1e-6)
            retained_info = (1.0 - entropy_norm).clamp(min=0.0, max=1.0)
            if self.rib_enabled:
                # RIB-Router: reciprocal information bottleneck temperature control.
                temp_raw = 1.0 + imbalance - (self.rib_temp_gain * retained_info)
                target_temp = temp_raw.clamp(min=self.router_temp_min, max=self.router_temp_max)
            else:
                target_temp = (1.0 + imbalance).clamp(min=self.router_temp_min, max=self.router_temp_max)
            router_logits = router_logits / target_temp
            self.last_router_temperature.copy_(target_temp.detach().to(self.last_router_temperature.dtype))
        else:
            self.last_router_temperature.fill_(1.0)
        # Deterministic tie-break for top-k routing stability (important for checkpoint recompute).
        tie_break = torch.arange(self.n_routed_experts, device=router_logits.device, dtype=router_logits.dtype)
        router_logits = router_logits + tie_break.view(1, 1, -1) * 1e-4
        if self.training and self.router_jitter_noise > 0 and not disable_jitter:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise
        self._aux_z_loss = router_logits.pow(2).mean()
        if self.training and self.rib_enabled and self.n_routed_experts > 1:
            probs = F.softmax(router_logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            entropy_norm = entropy / max(math.log(max(self.n_routed_experts, 2)), 1e-6)
            retained = (1.0 - entropy_norm).clamp(min=0.0, max=1.0)
            usage_probs = probs.mean(dim=(0, 1))
            collapse = (usage_probs.max() - usage_probs.min()).clamp(min=0.0, max=1.0)
            confidence = probs.max(dim=-1).values.mean().clamp(min=0.0, max=1.0)
            rib_aux = (
                (self.rib_info_weight * (entropy_norm / retained.clamp_min(1e-6)))
                + (self.rib_collapse_penalty * collapse)
                - (self.rib_confidence_weight * confidence * retained)
            )
            self._aux_z_loss = self._aux_z_loss + rib_aux
            self.last_rib_retained_info.copy_(retained.detach().to(self.last_rib_retained_info.dtype))
            self.last_rib_compression.copy_(entropy_norm.detach().to(self.last_rib_compression.dtype))
            self.last_rib_collapse.copy_(collapse.detach().to(self.last_rib_collapse.dtype))
            self.last_rib_aux_loss.copy_(rib_aux.detach().to(self.last_rib_aux_loss.dtype))
        else:
            self.last_rib_retained_info.fill_(0.0)
            self.last_rib_compression.fill_(0.0)
            self.last_rib_collapse.fill_(0.0)
            self.last_rib_aux_loss.fill_(0.0)

        with torch.no_grad():
            self._router_metric_step += 1
            self.last_router_z_loss.copy_(self._aux_z_loss.detach().to(self.last_router_z_loss.dtype))
            if (self._router_metric_step % self.router_metrics_interval) == 0:
                probs = F.softmax(router_logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
                self.last_router_entropy.copy_(entropy.to(self.last_router_entropy.dtype))

        t0 = time.perf_counter() if collect_timing else 0.0
        topk_ctx = self._nvtx("moe.topk") if collect_timing else nullcontext()
        with topk_ctx:
            topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
            topk_gates = F.softmax(topk_logits, dim=-1).to(x.dtype)
        if collect_timing:
            self._update_stage_ema("_last_stage_topk_s", time.perf_counter() - t0)

        flat_x = x.reshape(-1, self.d_model)
        routed_out = torch.zeros_like(flat_x)
        t_dispatch = time.perf_counter() if collect_timing else 0.0
        dispatch_ctx = self._nvtx("moe.dispatch") if collect_timing else nullcontext()
        with dispatch_ctx:
            if self.dispatch_mode == "legacy":
                self._dispatch_legacy(flat_x, topk_indices, topk_gates, routed_out)
                t_expert = 0.0
                t_scatter = 0.0
            elif self.dispatch_mode == "packed":
                self._dispatch_packed(flat_x, topk_indices, topk_gates, routed_out)
                t_expert = 0.0
                t_scatter = 0.0
            else:
                t_expert, t_scatter = self._dispatch_grouped(
                    flat_x,
                    topk_indices,
                    topk_gates,
                    routed_out,
                    collect_timing=collect_timing,
                )
        if collect_timing:
            self._update_stage_ema("_last_stage_dispatch_s", time.perf_counter() - t_dispatch)
            if self.dispatch_mode == "grouped":
                self._update_stage_ema("_last_stage_expert_s", t_expert)
                self._update_stage_ema("_last_stage_scatter_s", t_scatter)

        with torch.no_grad():
            usage = torch.bincount(topk_indices.reshape(-1), minlength=self.n_routed_experts).to(self.expert_usage_ema.dtype)
            usage = usage / usage.sum().clamp_min(1.0)
            self.expert_usage_ema[: self.n_routed_experts].mul_(0.95).add_(usage, alpha=0.05)
            if self.training and self.router_balance_lr > 0:
                target = torch.full_like(usage, 1.0 / max(self.n_routed_experts, 1))
                self.router_bias[: self.n_routed_experts].add_((target - usage) * self.router_balance_lr)
                self.router_bias[: self.n_routed_experts].clamp_(-2.0, 2.0)
            routed_fraction = (topk_gates.reshape(-1) > 0).float().mean()
            self.last_routed_fraction.copy_(routed_fraction.detach().to(self.last_routed_fraction.dtype))

        routed_out = routed_out.view_as(x)
        return shared_out + routed_out

    def aux_loss(self) -> torch.Tensor:
        if self._aux_z_loss is None:
            return self.expert_usage_ema.new_zeros(())
        return self._aux_z_loss

    @torch.no_grad()
    def router_metrics(self) -> dict[str, float]:
        if self.n_routed_experts == 0:
            return {
                "moe_entropy": 0.0,
                "moe_z_loss": 0.0,
                "moe_usage_max": 0.0,
                "moe_usage_min": 0.0,
                "moe_usage_imbalance": 0.0,
                "moe_routed_fraction": 0.0,
                "moe_router_bias_std": 0.0,
                "moe_router_temp": 1.0,
                "moe_gate_precision_bf16": 1.0 if self.router_precision == "bf16" else 0.0,
                "moe_gate_precision_int8": 1.0 if self.router_precision == "int8" else 0.0,
                "moe_gate_quantized": 1.0 if self.router_precision == "int8" else 0.0,
            }

        usage = self.expert_usage_ema[: self.n_routed_experts].float()
        usage = usage / usage.sum().clamp_min(1e-9)
        usage_max = float(usage.max().item())
        usage_min = float(usage.min().item())
        imbalance = usage_max - usage_min

        return {
            "moe_entropy": float(self.last_router_entropy.item()),
            "moe_z_loss": float(self.last_router_z_loss.item()),
            "moe_usage_max": usage_max,
            "moe_usage_min": usage_min,
            "moe_usage_imbalance": imbalance,
            "moe_routed_fraction": float(self.last_routed_fraction.item()),
            "moe_router_bias_std": float(self.router_bias[: self.n_routed_experts].float().std(unbiased=False).item()),
            "moe_router_temp": float(self.last_router_temperature.item()),
            "rib_retained_info": float(self.last_rib_retained_info.item()),
            "rib_router_compression": float(self.last_rib_compression.item()),
            "rib_router_collapse": float(self.last_rib_collapse.item()),
            "rib_aux_loss": float(self.last_rib_aux_loss.item()),
            "moe_stage_router_s": float(self._last_stage_router_s),
            "moe_stage_topk_s": float(self._last_stage_topk_s),
            "moe_stage_dispatch_s": float(self._last_stage_dispatch_s),
            "moe_stage_expert_s": float(self._last_stage_expert_s),
            "moe_stage_scatter_s": float(self._last_stage_scatter_s),
            "moe_gate_precision_bf16": 1.0 if self.router_precision == "bf16" else 0.0,
            "moe_gate_precision_int8": 1.0 if self.router_precision == "int8" else 0.0,
            "moe_gate_quantized": 1.0 if self.router_precision == "int8" else 0.0,
        }
