from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.optim import Optimizer


@dataclass
class C3OConfig:
    lr: float = 3e-4
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    damping: float = 1e-4
    grad_clip: float = 1.0
    block_size: int = 4096
    credit_ema: float = 0.92
    credit_gain: float = 0.40
    credit_skip_threshold: float = 1e-2
    block_scale_min: float = 0.60
    block_scale_max: float = 1.80
    trust_radius: float = 0.15
    trust_norm_refresh_steps: int = 16
    trust_norm_refresh_drift: float = 0.02
    foreach_fused: bool = True
    state_dtype: str = "auto"  # auto|fp32|bf16|fp16


class C3O(Optimizer):
    """
    Counterfactual Credit-Carrying Optimizer (C3O).

    C3O extends curvature-aware preconditioning with a credit signal that
    up/down-weights parameter blocks based on expected downstream utility.
    """

    def __init__(self, params, config: C3OConfig) -> None:
        defaults = dict(
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            beta1=float(config.beta1),
            beta2=float(config.beta2),
            eps=float(config.eps),
            damping=float(config.damping),
            grad_clip=float(config.grad_clip),
            block_size=max(64, int(config.block_size)),
            credit_ema=min(max(float(config.credit_ema), 0.0), 0.9999),
            credit_gain=float(config.credit_gain),
            credit_skip_threshold=float(config.credit_skip_threshold),
            block_scale_min=float(config.block_scale_min),
            block_scale_max=float(config.block_scale_max),
            trust_radius=float(config.trust_radius),
            trust_norm_refresh_steps=max(1, int(config.trust_norm_refresh_steps)),
            trust_norm_refresh_drift=float(config.trust_norm_refresh_drift),
            foreach_fused=bool(config.foreach_fused),
            state_dtype=str(config.state_dtype).strip().lower(),
        )
        super().__init__(params, defaults)
        # Trainer can skip extra global clip when optimizer already enforces clip internally.
        self.supports_internal_grad_clip = True
        self._credit_signal = 0.0
        self._last_metrics: dict[str, float] = {
            "c3o_credit_signal": 0.0,
            "c3o_block_scale_mean": 1.0,
            "c3o_block_scale_std": 0.0,
            "c3o_step_elements": 0.0,
            "c3o_update_norm": 0.0,
            "c3o_block_scaling_batches": 0.0,
            "c3o_block_scaling_skipped": 0.0,
            "c3o_trust_exact_recompute": 0.0,
            "c3o_trust_norm_skipped": 0.0,
            "c3o_foreach_groups": 0.0,
            "c3o_grad_clip_foreach_groups": 0.0,
        }

    @staticmethod
    def _resolve_state_dtype(param: torch.Tensor, state_dtype: str) -> torch.dtype:
        mode = str(state_dtype).strip().lower()
        if mode == "fp32":
            return torch.float32
        if mode == "bf16":
            return torch.bfloat16
        if mode == "fp16":
            return torch.float16
        if mode == "auto":
            if param.dtype in (torch.bfloat16, torch.float16):
                return param.dtype
            return torch.float32
        raise ValueError(f"Unsupported C3O state_dtype: {state_dtype!r}")

    @staticmethod
    def _apply_block_scaling_vectorized_(
        *,
        flat_upd: torch.Tensor,
        flat_grad: torch.Tensor,
        block_size: int,
        credit_gain: float,
        credit_value: float,
        s_min: float,
        s_max: float,
        chunk_blocks: int = 4096,
    ) -> tuple[float, float, int, int]:
        n = int(flat_upd.numel())
        if n <= 0:
            return 0.0, 0.0, 0, 0
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        scale_sum = 0.0
        scale_sq_sum = 0.0
        scale_count = 0
        batches = 0
        n_blocks = (n + block_size - 1) // block_size
        cb = max(1, int(chunk_blocks))
        alpha = float(credit_gain * credit_value)

        for blk0 in range(0, n_blocks, cb):
            blk1 = min(blk0 + cb, n_blocks)
            i0 = blk0 * block_size
            i1 = min(blk1 * block_size, n)
            if i1 <= i0:
                continue
            upd_chunk = flat_upd[i0:i1]
            grad_chunk = flat_grad[i0:i1]
            length = int(i1 - i0)

            full = length // block_size
            if full > 0:
                used = full * block_size
                upd2 = upd_chunk[:used].float().view(full, block_size)
                grd2 = grad_chunk[:used].float().view(full, block_size)
                align = (upd2 * grd2).mean(dim=1)
                blk_scale = 1.0 + (alpha * (1.0 + torch.tanh(10.0 * align)))
                blk_scale = torch.clamp(blk_scale, min=float(s_min), max=float(s_max))
                expanded = blk_scale.repeat_interleave(block_size).to(dtype=upd_chunk.dtype)
                upd_chunk[:used].mul_(expanded)
                scale_sum += float(blk_scale.sum().item())
                scale_sq_sum += float((blk_scale * blk_scale).sum().item())
                scale_count += int(blk_scale.numel())
                batches += 1

            tail = length - (full * block_size)
            if tail > 0:
                g_tail = grad_chunk[-tail:].float()
                u_tail = upd_chunk[-tail:].float()
                align_tail = float((g_tail * u_tail).mean().item())
                blk_scale_tail = 1.0 + (alpha * (1.0 + math.tanh(10.0 * align_tail)))
                blk_scale_tail = float(min(max(blk_scale_tail, float(s_min)), float(s_max)))
                upd_chunk[-tail:].mul_(blk_scale_tail)
                scale_sum += blk_scale_tail
                scale_sq_sum += blk_scale_tail * blk_scale_tail
                scale_count += 1
                batches += 1

        return scale_sum, scale_sq_sum, scale_count, batches

    @torch.no_grad()
    def set_credit_signal(self, value: float) -> None:
        v = float(value)
        if not math.isfinite(v):
            v = 0.0
        self._credit_signal = float(min(max(v, -4.0), 4.0))

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        out = dict(self._last_metrics)
        out["c3o_credit_signal"] = float(self._credit_signal)
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_elements = 0
        update_norm_acc_t: torch.Tensor | None = None
        scale_sum = 0.0
        scale_sq_sum = 0.0
        scale_count = 0
        scale_batches = 0
        scale_skipped = 0
        trust_exact_recompute = 0
        trust_norm_skipped = 0
        foreach_groups = 0
        grad_clip_foreach_groups = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            damping = float(group["damping"])
            grad_clip = float(group["grad_clip"])
            block_size = int(group["block_size"])
            credit_ema = float(group["credit_ema"])
            credit_gain = float(group["credit_gain"])
            credit_skip_threshold = float(group.get("credit_skip_threshold", 1e-3))
            s_min = float(group["block_scale_min"])
            s_max = float(group["block_scale_max"])
            trust_radius = float(group["trust_radius"])
            trust_norm_refresh_steps = max(1, int(group.get("trust_norm_refresh_steps", 8)))
            trust_norm_refresh_drift = max(0.0, float(group.get("trust_norm_refresh_drift", 0.02)))
            foreach_fused = bool(group.get("foreach_fused", True))
            state_dtype = str(group.get("state_dtype", "auto"))

            entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, int, float]] = []
            param_updates_by_dtype: dict[torch.dtype, list[tuple[torch.Tensor, torch.Tensor]]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("C3O does not support sparse gradients.")

                g = p.grad.detach()
                st = self.state[p]
                if len(st) == 0:
                    st_dtype = self._resolve_state_dtype(p, state_dtype)
                    st["m"] = torch.zeros_like(p, dtype=st_dtype)
                    st["h"] = torch.zeros_like(p, dtype=st_dtype)
                    st["step"] = 0
                    st["credit"] = 0.0

                m = st["m"]
                h = st["h"]
                st["step"] = int(st["step"]) + 1
                step_idx = int(st["step"])
                st_credit = float(st.get("credit", 0.0))
                st_credit = (credit_ema * st_credit) + ((1.0 - credit_ema) * self._credit_signal)
                st["credit"] = st_credit

                entries.append((p, g, m, h, st, step_idx, st_credit))

            if not entries:
                continue

            if grad_clip > 0:
                if foreach_fused and hasattr(torch, "_foreach_clamp_"):
                    grad_dtype_buckets: dict[torch.dtype, list[torch.Tensor]] = {}
                    for _, g, _, _, _, _, _ in entries:
                        grad_dtype_buckets.setdefault(g.dtype, []).append(g)
                    for g_list in grad_dtype_buckets.values():
                        torch._foreach_clamp_(g_list, min=-grad_clip, max=grad_clip)
                    grad_clip_foreach_groups += len(grad_dtype_buckets)
                else:
                    for _, g, _, _, _, _, _ in entries:
                        g.clamp_(min=-grad_clip, max=grad_clip)

            if foreach_fused:
                dtype_buckets: dict[torch.dtype, list[int]] = {}
                for idx, (_, _, m, _, _, _, _) in enumerate(entries):
                    dtype_buckets.setdefault(m.dtype, []).append(idx)
                for idxs in dtype_buckets.values():
                    m_list = [entries[i][2] for i in idxs]
                    h_list = [entries[i][3] for i in idxs]
                    target_dtype = m_list[0].dtype
                    g_list = [
                        entries[i][1] if entries[i][1].dtype == target_dtype else entries[i][1].to(dtype=target_dtype)
                        for i in idxs
                    ]
                    torch._foreach_mul_(m_list, beta1)
                    torch._foreach_add_(m_list, g_list, alpha=1.0 - beta1)
                    torch._foreach_mul_(h_list, beta2)
                    torch._foreach_addcmul_(h_list, g_list, g_list, value=1.0 - beta2)
                foreach_groups += len(dtype_buckets)
            else:
                for _, g, m, h, _, _, _ in entries:
                    g_state = g.to(dtype=m.dtype)
                    m.mul_(beta1).add_(g_state, alpha=1.0 - beta1)
                    h.mul_(beta2).addcmul_(g_state, g_state, value=1.0 - beta2)

            for p, g, m, h, st, step_idx, st_credit in entries:
                bc1 = 1.0 - (beta1**step_idx)
                bc2 = 1.0 - (beta2**step_idx)
                m_hat = m / max(bc1, 1e-8)
                h_hat = h / max(bc2, 1e-8)
                precond = torch.rsqrt(h_hat + damping + eps)
                upd = (m_hat * precond).to(dtype=p.dtype)

                n = int(upd.numel())
                if n <= 0:
                    continue

                # Per-block scaling from counterfactual credit + local alignment.
                use_block_scaling = not ((abs(st_credit) < credit_skip_threshold) or (credit_gain == 0.0))
                if not use_block_scaling:
                    blk_sum, blk_sq_sum, blk_count, blk_batches = 0.0, 0.0, 0, 0
                    scale_skipped += 1
                else:
                    flat_upd = upd.view(-1)
                    flat_grad = g.view(-1)
                    blk_sum, blk_sq_sum, blk_count, blk_batches = self._apply_block_scaling_vectorized_(
                        flat_upd=flat_upd,
                        flat_grad=flat_grad,
                        block_size=block_size,
                        credit_gain=credit_gain,
                        credit_value=st_credit,
                        s_min=s_min,
                        s_max=s_max,
                        chunk_blocks=4096,
                    )
                scale_sum += blk_sum
                scale_sq_sum += blk_sq_sum
                scale_count += blk_count
                scale_batches += blk_batches

                # Trust region clipping on update norm.
                upd_norm_t: torch.Tensor
                if trust_radius > 0.0:
                    cache_step = int(st.get("data_norm_step", -1))
                    data_norm_t = st.get("data_norm_t")
                    need_exact = (not isinstance(data_norm_t, torch.Tensor)) or (
                        (step_idx - cache_step) >= trust_norm_refresh_steps
                    )
                    if need_exact:
                        data_norm_t = torch.linalg.vector_norm(p.data.float()).detach()
                        st["data_norm_t"] = data_norm_t
                        st["data_norm_step"] = step_idx
                        trust_exact_recompute += 1
                    assert isinstance(data_norm_t, torch.Tensor)
                    max_upd_t = trust_radius * torch.clamp(data_norm_t, min=1e-6)
                    prev_upd_norm_t = st.get("prev_upd_norm_t")
                    prev_step = int(st.get("upd_norm_step", -1))
                    age = step_idx - prev_step
                    can_skip_norm = False
                    if (
                        isinstance(prev_upd_norm_t, torch.Tensor)
                        and age > 0
                        and age < trust_norm_refresh_steps
                        and trust_norm_refresh_drift > 0.0
                    ):
                        margin_t = max_upd_t * (1.0 - trust_norm_refresh_drift)
                        if bool((prev_upd_norm_t <= margin_t).item()):
                            can_skip_norm = True

                    if can_skip_norm:
                        upd_norm_t = prev_upd_norm_t.detach()
                        trust_norm_skipped += 1
                    else:
                        upd_norm_t = torch.linalg.vector_norm(upd.float())
                        scale_t = torch.clamp(max_upd_t / torch.clamp(upd_norm_t, min=1e-12), max=1.0)
                        upd.mul_(scale_t.to(dtype=upd.dtype))
                        upd_norm_t = upd_norm_t * scale_t
                        st["prev_upd_norm_t"] = upd_norm_t.detach()
                        st["upd_norm_step"] = step_idx
                else:
                    upd_norm_t = torch.linalg.vector_norm(upd.float())

                total_elements += n
                if update_norm_acc_t is None:
                    update_norm_acc_t = upd_norm_t.detach().float()
                else:
                    update_norm_acc_t = update_norm_acc_t + upd_norm_t.detach().float()
                param_updates_by_dtype.setdefault(p.data.dtype, []).append((p.data, upd))

            if foreach_fused and hasattr(torch, "_foreach_add_") and hasattr(torch, "_foreach_mul_"):
                for pairs in param_updates_by_dtype.values():
                    if not pairs:
                        continue
                    p_list = [pp for pp, _ in pairs]
                    upd_list = [uu for _, uu in pairs]
                    if wd != 0.0:
                        torch._foreach_mul_(p_list, 1.0 - (lr * wd))
                    torch._foreach_add_(p_list, upd_list, alpha=-lr)
            else:
                for pairs in param_updates_by_dtype.values():
                    for p_data, upd in pairs:
                        if wd != 0.0:
                            p_data.mul_(1.0 - (lr * wd))
                        p_data.add_(upd, alpha=-lr)

        if scale_count > 0:
            mean_scale = float(scale_sum / float(scale_count))
            var_scale = float((scale_sq_sum / float(scale_count)) - (mean_scale * mean_scale))
            std_scale = math.sqrt(max(0.0, var_scale))
        else:
            mean_scale = 1.0
            std_scale = 0.0

        update_norm_acc = float(update_norm_acc_t.item()) if isinstance(update_norm_acc_t, torch.Tensor) else 0.0
        self._last_metrics = {
            "c3o_credit_signal": float(self._credit_signal),
            "c3o_block_scale_mean": float(mean_scale),
            "c3o_block_scale_std": float(std_scale),
            "c3o_step_elements": float(total_elements),
            "c3o_update_norm": float(update_norm_acc),
            "c3o_block_scaling_batches": float(scale_batches),
            "c3o_block_scaling_skipped": float(scale_skipped),
            "c3o_trust_exact_recompute": float(trust_exact_recompute),
            "c3o_trust_norm_skipped": float(trust_norm_skipped),
            "c3o_foreach_groups": float(foreach_groups),
            "c3o_grad_clip_foreach_groups": float(grad_clip_foreach_groups),
        }
        return loss
