from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from .optimizer_c3o import C3O, C3OConfig
from .optimizer_gn import GNProx, GNProxConfig


def _matrix_update_scale(shape: tuple[int, ...], mode: str, extra_scale_factor: float) -> float:
    m = int(shape[0])
    n = 1
    for d in shape[1:]:
        n *= int(d)
    if mode == "none":
        scale = 1.0
    elif mode == "spectral":
        scale = (max(m, n) / max(min(m, n), 1)) ** 0.5
    elif mode == "row":
        scale = (m / max(n, 1)) ** 0.5 if m > n else 1.0
    else:
        raise ValueError(f"Unsupported Muon scale mode: {mode}")
    return float(scale * extra_scale_factor)


def _ns_coefficients(coefficient_type: str) -> tuple[float, float, float]:
    # Coefficients aligned with modern Muon implementations.
    if coefficient_type == "simple":
        return 1.5, -0.5, 0.0
    if coefficient_type == "quintic":
        return 3.4445, -4.7750, 2.0315
    if coefficient_type == "polar_express":
        return 1.8750, -1.2500, 0.3750
    raise ValueError(f"Unsupported Muon coefficient_type: {coefficient_type}")


def _orthogonalize_newton_schulz(
    update: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: str = "quintic",
) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization used by Muon/NorMuon-style optimizers.
    Operates on 2D tensors in fp32 for stability.
    """
    if update.ndim != 2:
        raise ValueError(f"Expected rank-2 update tensor, got shape {tuple(update.shape)}")

    a, b, c = _ns_coefficients(coefficient_type)
    original_dtype = update.dtype
    x = update.float()
    transposed = False
    if x.size(0) < x.size(1):
        x = x.t()
        transposed = True

    # Normalize to keep the iteration stable.
    x = x / (x.norm() + eps)

    for _ in range(steps):
        xxt = x @ x.t()
        if c != 0.0:
            xxt2 = xxt @ xxt
            x = a * x + (b * xxt + c * xxt2) @ x
        else:
            x = a * x + b * (xxt @ x)

    if transposed:
        x = x.t()
    return x.to(dtype=original_dtype)


class Muon(Optimizer):
    """
    Momentum Orthogonalized optimizer with modern extensions.

    - Matrix parameters (ndim >= 2) use momentum + orthogonalized updates.
    - Vector/scalar params fallback to AdamW-style moments.
    - Parameter groups may set `use_muon_for_matrix=False` to force Adam updates.
    - `row_adapt_beta2 > 0` enables NorMuon-style row-wise adaptation.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        coefficient_type: str = "quintic",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        row_adapt_beta2: float = 0.0,
        row_adapt_eps: float = 1e-8,
        orthogonalize_every: int = 1,
        orthogonalize_every_end: int = 1,
        orthogonalize_schedule_steps: int = 0,
        matrix_adaptive: bool = False,
        matrix_beta2: float = 0.99,
        matrix_eps: float = 1e-8,
        block_periodic_orth: bool = False,
        bp_full_orth_period: int = 4,
        bp_block_rows: int = 128,
        teon_coupling: float = 0.0,
        teon_eps: float = 1e-8,
        teon_clip_min: float = 0.5,
        teon_clip_max: float = 2.0,
        adaptive_orth: bool = True,
        adaptive_orth_beta: float = 0.95,
        adaptive_orth_tol: float = 0.05,
        adaptive_orth_max_skip: int = 8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            coefficient_type=coefficient_type,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            row_adapt_beta2=row_adapt_beta2,
            row_adapt_eps=row_adapt_eps,
            orthogonalize_every=orthogonalize_every,
            orthogonalize_every_end=orthogonalize_every_end,
            orthogonalize_schedule_steps=orthogonalize_schedule_steps,
            matrix_adaptive=matrix_adaptive,
            matrix_beta2=matrix_beta2,
            matrix_eps=matrix_eps,
            block_periodic_orth=block_periodic_orth,
            bp_full_orth_period=bp_full_orth_period,
            bp_block_rows=bp_block_rows,
            teon_coupling=teon_coupling,
            teon_eps=teon_eps,
            teon_clip_min=teon_clip_min,
            teon_clip_max=teon_clip_max,
            adaptive_orth=adaptive_orth,
            adaptive_orth_beta=adaptive_orth_beta,
            adaptive_orth_tol=adaptive_orth_tol,
            adaptive_orth_max_skip=adaptive_orth_max_skip,
        )
        super().__init__(params, defaults)
        self._last_step_metrics: dict[str, float] = {
            "muon_full_orth_updates": 0.0,
            "muon_block_orth_updates": 0.0,
            "muon_fast_updates": 0.0,
            "muon_adaptive_skip_ratio": 0.0,
        }

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        return dict(self._last_step_metrics)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_full_orth = 0
        total_block_orth = 0
        total_fast = 0
        total_adaptive_skips = 0
        total_scheduled_full = 0

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            beta1 = group["adam_beta1"]
            beta2 = group["adam_beta2"]
            coefficient_type = group["coefficient_type"]
            scale_mode = group["scale_mode"]
            extra_scale_factor = group["extra_scale_factor"]
            row_adapt_beta2 = group["row_adapt_beta2"]
            row_adapt_eps = group["row_adapt_eps"]
            orthogonalize_every_start = max(1, int(group["orthogonalize_every"]))
            orthogonalize_every_end = max(1, int(group.get("orthogonalize_every_end", orthogonalize_every_start)))
            orthogonalize_schedule_steps = max(0, int(group.get("orthogonalize_schedule_steps", 0)))
            matrix_adaptive = bool(group["matrix_adaptive"])
            matrix_beta2 = group["matrix_beta2"]
            matrix_eps = group["matrix_eps"]
            use_muon_for_matrix = bool(group.get("use_muon_for_matrix", True))
            block_periodic_orth = bool(group.get("block_periodic_orth", False))
            bp_full_orth_period = max(1, int(group.get("bp_full_orth_period", 4)))
            bp_block_rows = max(1, int(group.get("bp_block_rows", 128)))
            teon_coupling = float(group.get("teon_coupling", 0.0))
            teon_eps = float(group.get("teon_eps", 1e-8))
            teon_clip_min = float(group.get("teon_clip_min", 0.5))
            teon_clip_max = float(group.get("teon_clip_max", 2.0))
            adaptive_orth = bool(group.get("adaptive_orth", True))
            adaptive_orth_beta = float(group.get("adaptive_orth_beta", 0.95))
            adaptive_orth_tol = float(group.get("adaptive_orth_tol", 0.05))
            adaptive_orth_max_skip = max(0, int(group.get("adaptive_orth_max_skip", 8)))

            matrix_updates: list[tuple[torch.nn.Parameter, torch.Tensor, float]] = []
            global_norm_sq = 0.0
            local_norms: list[float] = []

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    if grad.ndim >= 2 and use_muon_for_matrix:
                        state["momentum_buffer"] = torch.zeros_like(param)
                        if matrix_adaptive:
                            state["matrix_exp_avg_sq"] = torch.zeros(
                                (param.size(0), int(param.numel() / max(param.size(0), 1))),
                                device=param.device,
                                dtype=param.dtype,
                            )
                    else:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1

                if weight_decay != 0:
                    param.mul_(1.0 - lr * weight_decay)

                if grad.ndim >= 2 and use_muon_for_matrix:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf

                    update_2d = update.reshape(update.size(0), -1)
                    if orthogonalize_schedule_steps > 0:
                        progress = min(state["step"] / max(orthogonalize_schedule_steps, 1), 1.0)
                        current_orthogonalize_every = int(
                            round(
                                orthogonalize_every_start
                                + (orthogonalize_every_end - orthogonalize_every_start) * progress
                            )
                        )
                        current_orthogonalize_every = max(1, current_orthogonalize_every)
                    else:
                        current_orthogonalize_every = orthogonalize_every_start

                    do_full_orth = (state["step"] % current_orthogonalize_every) == 0
                    if do_full_orth:
                        total_scheduled_full += 1
                    if adaptive_orth and do_full_orth:
                        update_norm = float(update_2d.norm().item())
                        ema = float(state.get("orth_grad_norm_ema", update_norm))
                        beta = min(max(adaptive_orth_beta, 0.0), 0.9999)
                        ema = (beta * ema) + ((1.0 - beta) * update_norm)
                        state["orth_grad_norm_ema"] = ema
                        rel_dev = abs(update_norm - ema) / max(abs(ema), eps)
                        skip_streak = int(state.get("orth_skip_streak", 0))
                        if rel_dev < adaptive_orth_tol and skip_streak < adaptive_orth_max_skip:
                            do_full_orth = False
                            state["orth_skip_streak"] = skip_streak + 1
                            total_adaptive_skips += 1
                        else:
                            state["orth_skip_streak"] = 0
                    if block_periodic_orth and do_full_orth and (state["step"] % bp_full_orth_period) != 0:
                        # MuonBP-style approximation: orthogonalize row-blocks on most steps,
                        # keep full orthogonalization periodically.
                        out_chunks: list[torch.Tensor] = []
                        for start in range(0, update_2d.size(0), bp_block_rows):
                            stop = min(start + bp_block_rows, update_2d.size(0))
                            chunk = _orthogonalize_newton_schulz(
                                update_2d[start:stop],
                                steps=ns_steps,
                                eps=eps,
                                coefficient_type=coefficient_type,
                            )
                            out_chunks.append(chunk)
                        update_2d = torch.cat(out_chunks, dim=0)
                        total_block_orth += 1
                    elif do_full_orth:
                        update_2d = _orthogonalize_newton_schulz(
                            update_2d,
                            steps=ns_steps,
                            eps=eps,
                            coefficient_type=coefficient_type,
                        )
                        total_full_orth += 1
                    else:
                        # Fast step between orthogonalization updates.
                        update_2d = update_2d / update_2d.norm(dim=1, keepdim=True).clamp_min(eps)
                        total_fast += 1

                    if row_adapt_beta2 > 0:
                        row_sq = update_2d.float().pow(2).mean(dim=1, keepdim=True)
                        row_var = state.get("row_var")
                        if row_var is None:
                            row_var = torch.zeros_like(row_sq)
                        row_var.mul_(row_adapt_beta2).add_(row_sq, alpha=1.0 - row_adapt_beta2)
                        state["row_var"] = row_var
                        update_2d = update_2d / row_var.sqrt().add(row_adapt_eps)

                    if matrix_adaptive:
                        matrix_var = state.get("matrix_exp_avg_sq")
                        if matrix_var is None:
                            matrix_var = torch.zeros_like(update_2d)
                        matrix_var.mul_(matrix_beta2).addcmul_(update_2d, update_2d, value=1.0 - matrix_beta2)
                        state["matrix_exp_avg_sq"] = matrix_var
                        update_2d = update_2d / matrix_var.sqrt().add(matrix_eps)

                    scale = _matrix_update_scale(tuple(param.shape), scale_mode, extra_scale_factor)
                    local = float(update_2d.norm().item())
                    local_norms.append(local)
                    global_norm_sq += (local * local)
                    matrix_updates.append((param, update_2d.view_as(param), scale))
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    step = state["step"]
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)
                    step_size = lr / bias_correction1
                    param.addcdiv_(exp_avg, denom, value=-step_size)

            if matrix_updates:
                global_norm = (global_norm_sq + teon_eps) ** 0.5
                for idx, (param, upd, scale) in enumerate(matrix_updates):
                    if teon_coupling > 0:
                        local = max(local_norms[idx], teon_eps)
                        # TEON-inspired global coupling across matrix updates.
                        ratio = (global_norm / local) ** teon_coupling
                        ratio = float(min(max(ratio, teon_clip_min), teon_clip_max))
                    else:
                        ratio = 1.0
                    param.add_(upd, alpha=-lr * scale * ratio)

        denom = float(max(total_scheduled_full, 1))
        self._last_step_metrics = {
            "muon_full_orth_updates": float(total_full_orth),
            "muon_block_orth_updates": float(total_block_orth),
            "muon_fast_updates": float(total_fast),
            "muon_adaptive_skip_ratio": float(total_adaptive_skips / denom),
        }

        return loss


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    use_muon: bool | None = None
    optimizer_name: str = "normuon"

    # Muon / NorMuon
    muon_momentum: float = 0.95
    muon_ns_steps: int = 3
    muon_coefficient_type: str = "simple"
    muon_scale_mode: str = "spectral"
    muon_extra_scale_factor: float = 1.0
    muon_row_adapt_beta2: float = 0.95
    muon_row_adapt_eps: float = 1e-8
    muon_orthogonalize_every: int = 1
    muon_orthogonalize_every_end: int = 4
    muon_orthogonalize_schedule_steps: int = 2000
    muon_matrix_adaptive_beta2: float = 0.99
    muon_matrix_adaptive_eps: float = 1e-8
    muon_block_periodic_orth: bool = True
    muon_bp_full_orth_period: int = 4
    muon_bp_block_rows: int = 128
    muon_teon_coupling: float = 0.15
    muon_teon_eps: float = 1e-8
    muon_teon_clip_min: float = 0.5
    muon_teon_clip_max: float = 2.0
    muon_adaptive_orth: bool = True
    muon_adaptive_orth_beta: float = 0.95
    muon_adaptive_orth_tol: float = 0.05
    muon_adaptive_orth_max_skip: int = 8
    muon_strict_split: bool = True
    muon_exclude_embeddings: bool = True
    muon_exclude_lm_head: bool = True
    muon_max_orthogonalized_dim: int = 8192

    # AdamW fallback
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # GN-Prox
    gn_beta1: float = 0.9
    gn_beta2: float = 0.98
    gn_eps: float = 1e-8
    gn_damping: float = 0.06
    gn_damping_min: float = 1e-4
    gn_damping_max: float = 1.0
    gn_damping_up: float = 1.08
    gn_damping_down: float = 0.97
    gn_damping_gain: float = 0.20
    gn_ns_steps: int = 5
    gn_block_size: int = 1024
    gn_clip_grad: float = 1.0

    # C3O (Counterfactual Credit-Carrying Optimizer)
    c3o_beta1: float = 0.9
    c3o_beta2: float = 0.98
    c3o_eps: float = 1e-8
    c3o_damping: float = 1e-4
    c3o_grad_clip: float = 1.0
    c3o_block_size: int = 4096
    c3o_credit_ema: float = 0.92
    c3o_credit_gain: float = 0.40
    c3o_credit_skip_threshold: float = 1e-2
    c3o_block_scale_min: float = 0.60
    c3o_block_scale_max: float = 1.80
    c3o_trust_radius: float = 0.15
    c3o_trust_norm_refresh_steps: int = 8
    c3o_trust_norm_refresh_drift: float = 0.02
    c3o_foreach_fused: bool = True
    c3o_state_dtype: str = "auto"

    def normalized_optimizer_name(self) -> str:
        if self.use_muon is not None:
            return "muon" if self.use_muon else "adamw"
        return self.optimizer_name


def _should_exclude_from_muon(name: str, param: torch.nn.Parameter, config: OptimizerConfig) -> bool:
    n = name.lower()
    if config.muon_exclude_embeddings and ("embed" in n or "token_emb" in n):
        return True
    if config.muon_exclude_lm_head and ("lm_head" in n or "output" in n or "mtp_head" in n or "mtp_heads" in n):
        return True
    if param.ndim >= 2:
        m = int(param.shape[0])
        n_dim = 1
        for d in param.shape[1:]:
            n_dim *= int(d)
        if max(m, n_dim) > config.muon_max_orthogonalized_dim:
            return True
    return False


def split_params(
    model: torch.nn.Module,
    *,
    strict_split: bool = True,
    exclude_embeddings: bool = True,
    exclude_lm_head: bool = True,
    max_orthogonalized_dim: int = 8192,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Split model parameters into Muon-compatible and AdamW-compatible groups.

    Default behavior follows orthogonal-split best practice:
    - Muon group: matrix-like tensors (ndim >= 2), excluding embedding/output heads.
    - AdamW group: vectors/scalars (ndim < 2) + excluded matrix tensors.
    """
    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
    strict = bool(strict_split)
    split_cfg = OptimizerConfig(
        muon_exclude_embeddings=exclude_embeddings,
        muon_exclude_lm_head=exclude_lm_head,
        muon_max_orthogonalized_dim=max_orthogonalized_dim,
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if strict:
            if param.ndim >= 2 and not _should_exclude_from_muon(name, param, split_cfg):
                muon_params.append(param)
            else:
                adam_params.append(param)
            continue

        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params


def _build_muon_param_groups(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> tuple[list[dict[str, object]], int, int]:
    muon_params, adam_params = split_params(
        model,
        strict_split=config.muon_strict_split,
        exclude_embeddings=config.muon_exclude_embeddings,
        exclude_lm_head=config.muon_exclude_lm_head,
        max_orthogonalized_dim=config.muon_max_orthogonalized_dim,
    )

    groups: list[dict[str, object]] = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon_for_matrix": True})
    if adam_params:
        groups.append({"params": adam_params, "use_muon_for_matrix": False})
    return groups, len(muon_params), len(adam_params)


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig, device: str | None = None) -> Optimizer:
    opt_name = config.normalized_optimizer_name().lower()
    if opt_name == "c3o":
        print("[optimizer] type=c3o (counterfactual credit-carrying second-order preconditioner)")
        c3o_cfg = C3OConfig(
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.c3o_beta1,
            beta2=config.c3o_beta2,
            eps=config.c3o_eps,
            damping=config.c3o_damping,
            grad_clip=config.c3o_grad_clip,
            block_size=config.c3o_block_size,
            credit_ema=config.c3o_credit_ema,
            credit_gain=config.c3o_credit_gain,
            credit_skip_threshold=config.c3o_credit_skip_threshold,
            block_scale_min=config.c3o_block_scale_min,
            block_scale_max=config.c3o_block_scale_max,
            trust_radius=config.c3o_trust_radius,
            trust_norm_refresh_steps=config.c3o_trust_norm_refresh_steps,
            trust_norm_refresh_drift=config.c3o_trust_norm_refresh_drift,
            foreach_fused=config.c3o_foreach_fused,
            state_dtype=config.c3o_state_dtype,
        )
        return C3O(model.parameters(), c3o_cfg)

    if opt_name == "gnprox":
        if device is not None and not str(device).startswith("cuda"):
            raise RuntimeError("GNProx requires CUDA device.")
        print("[optimizer] type=gnprox (triton fused diagonal gauss-newton)")
        gn_cfg = GNProxConfig(
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.gn_beta1,
            beta2=config.gn_beta2,
            eps=config.gn_eps,
            damping=config.gn_damping,
            damping_min=config.gn_damping_min,
            damping_max=config.gn_damping_max,
            damping_up=config.gn_damping_up,
            damping_down=config.gn_damping_down,
            damping_gain=config.gn_damping_gain,
            ns_steps=config.gn_ns_steps,
            block_size=config.gn_block_size,
            clip_grad=config.gn_clip_grad,
        )
        return GNProx(model.parameters(), gn_cfg)

    if opt_name in {"muon", "normuon", "adamuon"}:
        row_beta2 = config.muon_row_adapt_beta2 if opt_name in {"normuon", "adamuon"} else 0.0
        groups, muon_count, adam_count = _build_muon_param_groups(model, config)
        print(f"[optimizer] type={opt_name} muon_params={muon_count} adam_params={adam_count}")
        return Muon(
            groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.muon_momentum,
            ns_steps=config.muon_ns_steps,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            coefficient_type=config.muon_coefficient_type,
            scale_mode=config.muon_scale_mode,
            extra_scale_factor=config.muon_extra_scale_factor,
            row_adapt_beta2=row_beta2,
            row_adapt_eps=config.muon_row_adapt_eps,
            orthogonalize_every=config.muon_orthogonalize_every,
            orthogonalize_every_end=config.muon_orthogonalize_every_end,
            orthogonalize_schedule_steps=config.muon_orthogonalize_schedule_steps,
            matrix_adaptive=(opt_name == "adamuon"),
            matrix_beta2=config.muon_matrix_adaptive_beta2,
            matrix_eps=config.muon_matrix_adaptive_eps,
            block_periodic_orth=config.muon_block_periodic_orth,
            bp_full_orth_period=config.muon_bp_full_orth_period,
            bp_block_rows=config.muon_bp_block_rows,
            teon_coupling=config.muon_teon_coupling,
            teon_eps=config.muon_teon_eps,
            teon_clip_min=config.muon_teon_clip_min,
            teon_clip_max=config.muon_teon_clip_max,
            adaptive_orth=config.muon_adaptive_orth,
            adaptive_orth_beta=config.muon_adaptive_orth_beta,
            adaptive_orth_tol=config.muon_adaptive_orth_tol,
            adaptive_orth_max_skip=config.muon_adaptive_orth_max_skip,
        )

    adamw_kwargs = dict(
        params=model.parameters(),
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    # Fused AdamW improves throughput on modern CUDA GPUs.
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        adamw_kwargs["fused"] = True

    return torch.optim.AdamW(**adamw_kwargs)
