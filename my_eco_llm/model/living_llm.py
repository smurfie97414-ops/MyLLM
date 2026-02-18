from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.moe import DeepSeekMoE


@dataclass
class TTTLinearState:
    fast_weight: torch.Tensor
    momentum: torch.Tensor
    input_ema: torch.Tensor
    lr_mult_ema: torch.Tensor
    steps: int = 0


class TTTLinear(nn.Module):
    """
    Test-Time Training linear memory.

    The layer keeps a transient weight matrix per batch element and updates it
    online with a local gradient step at each token. Memory is O(B * D^2), not
    O(context_length), so state size is constant with sequence growth.
    """

    def __init__(
        self,
        d_model: int,
        inner_lr: float = 0.05,
        momentum: float = 0.9,
        first_order: bool = True,
        max_fast_weight_norm: float = 32.0,
        adaptive_lr: bool = True,
        novelty_beta: float = 0.95,
        novelty_gain: float = 3.0,
        adaptive_lr_min: float = 0.5,
        adaptive_lr_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.first_order = bool(first_order)
        self.momentum_beta = float(momentum)
        self.max_fast_weight_norm = float(max_fast_weight_norm)
        self.adaptive_lr = bool(adaptive_lr)
        self.novelty_beta = float(novelty_beta)
        self.novelty_gain = float(novelty_gain)
        self.adaptive_lr_min = float(adaptive_lr_min)
        self.adaptive_lr_max = float(adaptive_lr_max)

        self.base_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.log_inner_lr = nn.Parameter(torch.tensor(float(inner_lr)).log())
        self.register_buffer("last_adaptive_lr_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("last_novelty", torch.tensor(0.0), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.bias)

    @property
    def inner_lr(self) -> torch.Tensor:
        return self.log_inner_lr.exp().clamp(min=1e-5, max=1.0)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TTTLinearState:
        w = self.base_weight
        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        fast_weight = w.unsqueeze(0).expand(batch_size, -1, -1).clone()
        momentum = torch.zeros_like(fast_weight)
        input_ema = torch.zeros(batch_size, self.d_model, device=fast_weight.device, dtype=fast_weight.dtype)
        lr_mult_ema = torch.ones(batch_size, 1, device=fast_weight.device, dtype=fast_weight.dtype)
        return TTTLinearState(
            fast_weight=fast_weight,
            momentum=momentum,
            input_ema=input_ema,
            lr_mult_ema=lr_mult_ema,
            steps=0,
        )

    def _online_update(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        target_t: torch.Tensor,
        state: TTTLinearState,
    ) -> TTTLinearState:
        # dL/dW for MSE(y_t, target_t), where y_t = W x_t + b
        err = (y_t - target_t) / float(self.d_model)
        grad_w = torch.einsum("bi,bj->bij", err, x_t)
        if self.first_order:
            grad_w = grad_w.detach()

        new_momentum = (self.momentum_beta * state.momentum) + ((1.0 - self.momentum_beta) * grad_w)

        if self.adaptive_lr:
            novelty = (x_t - state.input_ema).abs().mean(dim=-1, keepdim=True)
            lr_scale = (1.0 + (self.novelty_gain * novelty)).clamp(self.adaptive_lr_min, self.adaptive_lr_max)
            new_lr_mult_ema = (self.novelty_beta * state.lr_mult_ema) + ((1.0 - self.novelty_beta) * lr_scale)
            eff_lr = self.inner_lr.to(dtype=x_t.dtype) * new_lr_mult_ema.view(-1, 1, 1)
            new_input_ema = (self.novelty_beta * state.input_ema) + ((1.0 - self.novelty_beta) * x_t)
            self.last_adaptive_lr_scale.copy_(new_lr_mult_ema.mean().detach().to(self.last_adaptive_lr_scale.dtype))
            self.last_novelty.copy_(novelty.mean().detach().to(self.last_novelty.dtype))
        else:
            eff_lr = self.inner_lr.to(dtype=x_t.dtype)
            new_input_ema = state.input_ema
            new_lr_mult_ema = state.lr_mult_ema
            self.last_adaptive_lr_scale.fill_(1.0)
            self.last_novelty.fill_(0.0)

        new_fast_weight = state.fast_weight - (eff_lr * new_momentum)

        if self.max_fast_weight_norm > 0:
            flat_norm = new_fast_weight.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            max_norm = torch.tensor(
                self.max_fast_weight_norm,
                device=new_fast_weight.device,
                dtype=new_fast_weight.dtype,
            )
            scale = (max_norm / flat_norm).clamp(max=1.0).view(-1, 1, 1)
            new_fast_weight = new_fast_weight * scale

        return TTTLinearState(
            fast_weight=new_fast_weight,
            momentum=new_momentum,
            input_ema=new_input_ema,
            lr_mult_ema=new_lr_mult_ema,
            steps=state.steps + 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: TTTLinearState | None = None,
        targets: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, TTTLinearState]:
        if x.ndim != 3:
            raise ValueError(f"TTTLinear expects [B, T, D], got {tuple(x.shape)}")
        bsz, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        if state is None:
            state = self.init_state(batch_size=bsz, device=x.device, dtype=x.dtype)
        elif state.fast_weight.size(0) != bsz:
            raise ValueError(
                f"TTT state batch mismatch: state batch={state.fast_weight.size(0)} input batch={bsz}"
            )

        outputs: list[torch.Tensor] = []
        current = state
        for t in range(seq_len):
            x_t = x[:, t, :]
            y_t = torch.bmm(current.fast_weight, x_t.unsqueeze(-1)).squeeze(-1) + self.bias.to(dtype=x_t.dtype)
            outputs.append(y_t.unsqueeze(1))

            if not update_state:
                continue
            target_t = x_t if targets is None else targets[:, t, :]
            current = self._online_update(x_t=x_t, y_t=y_t, target_t=target_t, state=current)

        return torch.cat(outputs, dim=1), current

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return {
            "ttt_adaptive_lr_scale": float(self.last_adaptive_lr_scale.item()),
            "ttt_input_novelty": float(self.last_novelty.item()),
        }


class DiffAttention(nn.Module):
    """
    Differential attention:
      Softmax(QK1^T) - lambda * Softmax(QK2^T)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        lambda_init: float = 0.5,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not (0.0 <= lambda_init <= 1.0):
            raise ValueError(f"lambda_init must be in [0, 1], got {lambda_init}")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_exp_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_noise_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.lambda_logit = nn.Parameter(torch.full((n_heads,), torch.logit(torch.tensor(lambda_init))))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _causal_mask(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device).unsqueeze(-1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        return k_pos <= q_pos

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"DiffAttention expects [B, T, D], got {tuple(x.shape)}")
        bsz, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        q = self._shape(self.q_proj(x))
        k1 = self._shape(self.k_exp_proj(x))
        k2 = self._shape(self.k_noise_proj(x))
        v = self._shape(self.v_proj(x))

        score_1 = torch.matmul(q, k1.transpose(-2, -1)) * self.scale
        score_2 = torch.matmul(q, k2.transpose(-2, -1)) * self.scale

        if causal:
            mask = self._causal_mask(seq_len, seq_len, x.device).view(1, 1, seq_len, seq_len)
            min_val = torch.finfo(score_1.dtype).min
            score_1 = score_1.masked_fill(~mask, min_val)
            score_2 = score_2.masked_fill(~mask, min_val)
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                keep = attn_mask[:, None, None, :].bool()
            elif attn_mask.ndim == 4:
                keep = attn_mask.bool()
            else:
                raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")
            min_val = torch.finfo(score_1.dtype).min
            score_1 = score_1.masked_fill(~keep, min_val)
            score_2 = score_2.masked_fill(~keep, min_val)

        attn_exp = F.softmax(score_1, dim=-1)
        attn_noise = F.softmax(score_2, dim=-1)
        lambda_h = torch.sigmoid(self.lambda_logit).view(1, self.n_heads, 1, 1)
        attn = attn_exp - (lambda_h * attn_noise)

        # Renormalize signed map for numerical stability.
        denom = attn.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = self.dropout(attn / denom)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        out = self.out_proj(out)
        stats = {
            "diff_lambda_mean": lambda_h.mean().detach(),
            "diff_attn_abs_mean": attn.abs().mean().detach(),
        }
        return out, stats


class NestedLearningMemory(nn.Module):
    """
    Persistent session memory ("slow weights") updated across forward passes.
    """

    def __init__(self, d_model: int, slots: int = 64, update_rate: float = 0.01) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.slots = int(slots)
        self.update_rate = float(update_rate)
        self.register_buffer("memory", torch.zeros(slots, d_model), persistent=True)
        self.register_buffer("write_ptr", torch.zeros((), dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        if self.slots <= 0:
            return x
        # Use a detached snapshot for retrieval so memory writes in this forward
        # do not invalidate autograd version checks.
        mem_snapshot = self.memory.detach().clone()
        scores = torch.matmul(x, mem_snapshot.t()) / (self.d_model**0.5)
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, mem_snapshot)
        out = x + retrieved
        if update_memory:
            self.update_from_hidden(x.detach())
        return out

    @torch.no_grad()
    def update_from_hidden(self, hidden: torch.Tensor) -> None:
        if self.slots <= 0:
            return
        summary = hidden.mean(dim=(0, 1))
        idx = int(self.write_ptr.item()) % self.slots
        self.memory[idx].mul_(1.0 - self.update_rate).add_(summary, alpha=self.update_rate)
        self.write_ptr.add_(1)


class LivingBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        adaptive_ttt_lr: bool = True,
        use_moe: bool = True,
        n_shared_experts: int = 1,
        n_routed_experts: int = 8,
        moe_top_k: int = 2,
        router_balance_lr: float = 1e-3,
        router_jitter_noise: float = 0.01,
        router_metrics_interval: int = 10,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.RMSNorm(d_model)
        self.norm_2 = nn.RMSNorm(d_model)
        self.norm_3 = nn.RMSNorm(d_model)
        self.norm_4 = nn.RMSNorm(d_model)
        self.diff_attn = DiffAttention(d_model=d_model, n_heads=n_heads)
        self.ttt = TTTLinear(d_model=d_model, adaptive_lr=adaptive_ttt_lr)
        self.use_moe = bool(use_moe)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model, bias=False),
        )
        self.ff_gate = nn.Parameter(torch.tensor(0.0))
        self.moe_gate = nn.Parameter(torch.tensor(0.0))
        self.moe = (
            DeepSeekMoE(
                d_model=d_model,
                d_ff=d_model * ff_mult,
                n_shared_experts=n_shared_experts,
                n_routed_experts=n_routed_experts,
                top_k=moe_top_k,
                dropout=0.0,
                router_jitter_noise=router_jitter_noise,
                router_balance_lr=router_balance_lr,
                router_metrics_interval=router_metrics_interval,
                use_triton_bitlinear=False,
                use_bitlinear_quant_cache_training=True,
            )
            if self.use_moe
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        ttt_state: TTTLinearState | None = None,
        update_ttt: bool = True,
    ) -> tuple[torch.Tensor, TTTLinearState, dict[str, torch.Tensor]]:
        attn_out, attn_stats = self.diff_attn(self.norm_1(x), causal=True)
        x = x + attn_out
        ttt_out, next_state = self.ttt(self.norm_2(x), state=ttt_state, targets=None, update_state=update_ttt)
        x = x + ttt_out
        ff_out = self.ff(self.norm_3(x))
        ff_scale = torch.sigmoid(self.ff_gate).to(dtype=x.dtype)
        x = x + (ff_scale * ff_out)

        moe_stats: dict[str, torch.Tensor] = {}
        if self.moe is not None:
            moe_out = self.moe(self.norm_4(x), disable_jitter=not self.training)
            moe_scale = torch.sigmoid(self.moe_gate).to(dtype=x.dtype)
            x = x + (moe_scale * moe_out)
            rm = self.moe.router_metrics()
            for k, v in rm.items():
                moe_stats[k] = torch.tensor(float(v), device=x.device, dtype=x.dtype)
            moe_stats["moe_aux_z_loss"] = self.moe.aux_loss().detach().to(dtype=x.dtype)
            moe_stats["moe_gate_scale"] = moe_scale.detach()

        ttt_metrics = self.ttt.metrics()
        return x, next_state, {
            **attn_stats,
            **moe_stats,
            "ttt_adaptive_lr_scale": torch.tensor(ttt_metrics["ttt_adaptive_lr_scale"], device=x.device),
            "ttt_input_novelty": torch.tensor(ttt_metrics["ttt_input_novelty"], device=x.device),
        }


class LivingLLM(nn.Module):
    """
    2026 hybrid:
      - TTTLinear transient learning layers
      - Differential Attention
      - Nested Learning persistent memory
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        ff_mult: int = 4,
        memory_slots: int = 64,
        max_seq_len: int = 4096,
        adaptive_ttt_lr: bool = True,
        use_moe: bool = True,
        n_shared_experts: int = 1,
        n_routed_experts: int = 8,
        moe_top_k: int = 2,
        router_balance_lr: float = 1e-3,
        router_jitter_noise: float = 0.01,
        router_metrics_interval: int = 10,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList(
            [
                LivingBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    adaptive_ttt_lr=adaptive_ttt_lr,
                    use_moe=use_moe,
                    n_shared_experts=n_shared_experts,
                    n_routed_experts=n_routed_experts,
                    moe_top_k=moe_top_k,
                    router_balance_lr=router_balance_lr,
                    router_jitter_noise=router_jitter_noise,
                    router_metrics_interval=router_metrics_interval,
                )
                for _ in range(n_layers)
            ]
        )
        self.memory = NestedLearningMemory(d_model=d_model, slots=memory_slots, update_rate=0.01)
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @torch.no_grad()
    def reset_nested_memory(self) -> None:
        if self.memory.slots > 0:
            self.memory.memory.zero_()
            self.memory.write_ptr.zero_()

    def init_ttt_states(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[TTTLinearState]:
        states: list[TTTLinearState] = []
        for block in self.blocks:
            states.append(block.ttt.init_state(batch_size=batch_size, device=device, dtype=dtype))
        return states

    def forward(
        self,
        input_ids: torch.Tensor,
        ttt_states: list[TTTLinearState] | None = None,
        update_ttt: bool = True,
        update_nested_memory: bool = True,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, list[TTTLinearState], dict[str, torch.Tensor]] | torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"LivingLLM expects [B, T] token ids, got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        x = self.memory(x, update_memory=update_nested_memory)

        if ttt_states is None:
            ttt_states = self.init_ttt_states(batch_size=bsz, device=x.device, dtype=x.dtype)
        if len(ttt_states) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} TTT states, got {len(ttt_states)}")

        next_states: list[TTTLinearState] = []
        attn_metric_accum: dict[str, torch.Tensor] = {}
        for block, state in zip(self.blocks, ttt_states):
            x, new_state, attn_stats = block(x, ttt_state=state, update_ttt=update_ttt)
            next_states.append(new_state)
            for key, value in attn_stats.items():
                attn_metric_accum[key] = attn_metric_accum.get(key, torch.zeros_like(value)) + value

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if not return_state:
            return logits
        n = float(max(len(self.blocks), 1))
        avg_stats = {k: (v / n) for k, v in attn_metric_accum.items()}
        return logits, next_states, avg_stats

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
        ttt_states: list[TTTLinearState] | None = None,
        update_ttt: bool = True,
        update_nested_memory: bool = True,
    ) -> tuple[torch.Tensor, list[TTTLinearState], dict[str, torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError(f"generate expects [B, T], got {tuple(input_ids.shape)}")
        generated = input_ids

        logits, ttt_states, stats = self(
            generated[:, -self.max_seq_len :],
            ttt_states=ttt_states,
            update_ttt=update_ttt,
            update_nested_memory=update_nested_memory,
            return_state=True,
        )
        next_token_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                step_logits = next_token_logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), step_logits.size(-1))
                    topv, _ = torch.topk(step_logits, k=k, dim=-1)
                    cutoff = topv[:, [-1]]
                    step_logits = step_logits.masked_fill(step_logits < cutoff, float("-inf"))
                probs = F.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            logits, ttt_states, stats = self(
                next_token,
                ttt_states=ttt_states,
                update_ttt=update_ttt,
                update_nested_memory=update_nested_memory,
                return_state=True,
            )
            next_token_logits = logits[:, -1, :]

        return generated, ttt_states, stats


@torch.no_grad()
def verify_ttt_constant_memory_latency(
    model: LivingLLM,
    device: str = "cuda",
    lengths: tuple[int, int] = (10_000, 100_000),
    chunk_size: int = 128,
) -> dict[str, Any]:
    """
    Runs streaming inference with persistent TTT states and reports time/token.
    For a valid TTT setup, memory footprint does not grow with total length.
    """
    model = model.to(device).eval()
    results: dict[str, Any] = {}

    for total_len in lengths:
        states = model.init_ttt_states(batch_size=1, device=torch.device(device), dtype=torch.float32)
        start = time.perf_counter()
        peak_mem_gb = 0.0
        remaining = int(total_len)
        while remaining > 0:
            cur = min(chunk_size, remaining)
            ids = torch.randint(0, model.vocab_size, (1, cur), device=device)
            _, states, _ = model(
                ids,
                ttt_states=states,
                update_ttt=True,
                update_nested_memory=True,
                return_state=True,
            )
            if device.startswith("cuda"):
                peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated() / (1024**3))
            remaining -= cur
        elapsed = time.perf_counter() - start
        results[str(total_len)] = {
            "elapsed_s": elapsed,
            "tokens_per_s": total_len / max(elapsed, 1e-6),
            "peak_mem_gb": peak_mem_gb,
        }

    t10 = results[str(lengths[0])]["tokens_per_s"]
    t100 = results[str(lengths[1])]["tokens_per_s"]
    results["speed_ratio_100k_over_10k"] = float(t100 / max(t10, 1e-6))
    return results


__all__ = [
    "TTTLinear",
    "TTTLinearState",
    "DiffAttention",
    "NestedLearningMemory",
    "LivingLLM",
    "verify_ttt_constant_memory_latency",
]
