from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SigmaRLConfig:
    mode: str = "igrpo"
    clip_eps: float = 0.2
    entropy_weight: float = 0.001
    adv_norm: bool = True
    kl_weight: float = 0.02
    dispo_logit_temp: float = 1.0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom
    if x.ndim == 2:
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom
    raise RuntimeError(f"Unsupported masked mean rank: {x.ndim}")


def _normalize_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Group-normalized advantage signal.
    centered = rewards - rewards.mean()
    return centered / centered.std(unbiased=False).clamp_min(eps)


def _compute_ratio(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sequence-level log-prob and PPO-style ratio.
    seq_new = _masked_mean(logp_new, mask)
    seq_old = _masked_mean(logp_old, mask)
    ratio = torch.exp(seq_new - seq_old).clamp(1e-4, 1e4)
    return seq_new, ratio


def _entropy_from_logprobs(logp_new: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Approximate entropy term from selected-action log probs.
    ent = -_masked_mean(logp_new, mask)
    if ent.ndim == 0:
        return ent
    return ent.mean()


def _igrpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    cfg: SigmaRLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    adv = rewards
    if cfg.adv_norm:
        adv = _normalize_advantages(adv)
    seq_new, ratio = _compute_ratio(logp_new, logp_old, mask)
    clip = float(max(1e-4, cfg.clip_eps))
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    adv_scalar = adv.mean()
    surrogate = torch.minimum(ratio, clipped) * adv_scalar
    surrogate = surrogate.mean()
    entropy = _entropy_from_logprobs(logp_new, mask)
    loss = -(surrogate + (cfg.entropy_weight * entropy))
    metrics = {
        "sigma_rl_adv_mean": float(adv.mean().item()),
        "sigma_rl_ratio": float(ratio.mean().item()),
        "sigma_rl_entropy": float(entropy.item()),
        "sigma_rl_seq_logp": float(seq_new.mean().item()),
        "sigma_rl_clip_eps": float(clip),
    }
    return loss, metrics


def _dispo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    cfg: SigmaRLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Preference-style objective where reward defines a soft preference target.
    adv = rewards
    if cfg.adv_norm:
        adv = _normalize_advantages(adv)
    seq_new = _masked_mean(logp_new, mask)
    seq_old = _masked_mean(logp_old, mask)
    temp = float(max(1e-4, cfg.dispo_logit_temp))
    margin = ((seq_new - seq_old) / temp).mean()
    target = torch.sigmoid(adv.mean())
    pred = torch.sigmoid(margin)
    bce = -(target * torch.log(pred.clamp_min(1e-8)) + (1.0 - target) * torch.log((1.0 - pred).clamp_min(1e-8)))
    entropy = _entropy_from_logprobs(logp_new, mask)
    loss = bce - (cfg.entropy_weight * entropy)
    metrics = {
        "sigma_rl_adv_mean": float(adv.mean().item()),
        "sigma_rl_margin": float(margin.item()),
        "sigma_rl_target": float(target.item()),
        "sigma_rl_entropy": float(entropy.item()),
    }
    return loss, metrics


def _gspo_like_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    cfg: SigmaRLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Stable policy optimization with explicit KL control to old policy.
    adv = rewards
    if cfg.adv_norm:
        adv = _normalize_advantages(adv)
    seq_new, ratio = _compute_ratio(logp_new, logp_old, mask)
    seq_old = _masked_mean(logp_old, mask)
    kl = (seq_old - seq_new)
    kl_mean = kl.mean()
    policy_gain = (ratio * adv.mean()).mean()
    entropy = _entropy_from_logprobs(logp_new, mask)
    loss = -(policy_gain + (cfg.entropy_weight * entropy)) + (cfg.kl_weight * kl_mean)
    metrics = {
        "sigma_rl_adv_mean": float(adv.mean().item()),
        "sigma_rl_ratio": float(ratio.mean().item()),
        "sigma_rl_kl": float(kl_mean.item()),
        "sigma_rl_entropy": float(entropy.item()),
    }
    return loss, metrics


def _cispo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    cfg: SigmaRLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    CISPO-inspired objective:
    clip importance weights directly and optimize a weighted policy-gain term.
    """
    adv = rewards
    if cfg.adv_norm:
        adv = _normalize_advantages(adv)
    seq_new, ratio = _compute_ratio(logp_new, logp_old, mask)
    clip = float(max(1e-4, cfg.clip_eps))
    importance = torch.clamp(ratio.detach(), 1.0 - clip, 1.0 + clip)
    adv_scalar = adv.mean()
    policy_gain = (importance * seq_new * adv_scalar).mean()
    entropy = _entropy_from_logprobs(logp_new, mask)
    loss = -(policy_gain + (cfg.entropy_weight * entropy))
    metrics = {
        "sigma_rl_adv_mean": float(adv.mean().item()),
        "sigma_rl_importance_mean": float(importance.mean().item()),
        "sigma_rl_ratio": float(ratio.mean().item()),
        "sigma_rl_entropy": float(entropy.item()),
        "sigma_rl_seq_logp": float(seq_new.mean().item()),
        "sigma_rl_clip_eps": float(clip),
    }
    return loss, metrics


def compute_sigma_rl_loss(
    *,
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    cfg: SigmaRLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if logp_new.shape != logp_old.shape or logp_new.shape != mask.shape:
        raise RuntimeError(
            f"RL loss shape mismatch. new={tuple(logp_new.shape)} old={tuple(logp_old.shape)} mask={tuple(mask.shape)}"
        )
    if rewards.ndim != 1:
        raise RuntimeError(f"rewards must be rank-1, got {tuple(rewards.shape)}")
    mode = cfg.mode.lower().strip()
    if mode == "igrpo":
        return _igrpo_loss(logp_new=logp_new, logp_old=logp_old, rewards=rewards, mask=mask, cfg=cfg)
    if mode == "dispo":
        return _dispo_loss(logp_new=logp_new, logp_old=logp_old, rewards=rewards, mask=mask, cfg=cfg)
    if mode == "gspo":
        return _gspo_like_loss(logp_new=logp_new, logp_old=logp_old, rewards=rewards, mask=mask, cfg=cfg)
    if mode == "cispo":
        return _cispo_loss(logp_new=logp_new, logp_old=logp_old, rewards=rewards, mask=mask, cfg=cfg)
    raise RuntimeError(f"Unsupported sigma RL mode: {cfg.mode}")
