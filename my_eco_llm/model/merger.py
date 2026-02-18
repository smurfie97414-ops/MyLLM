from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch


@dataclass
class MergeConfig:
    method: str = "ties"  # ties|dare_ties
    density: float = 1.0
    seed: int = 20260216

    def normalize(self) -> "MergeConfig":
        method = str(self.method).strip().lower()
        if method not in {"ties", "dare_ties"}:
            raise ValueError(f"Unsupported merge method: {self.method}")
        density = float(min(max(self.density, 1e-3), 1.0))
        return MergeConfig(method=method, density=density, seed=int(self.seed))


def _intersect_keys(states: Iterable[dict[str, torch.Tensor]]) -> list[str]:
    key_sets = [set(s.keys()) for s in states]
    if not key_sets:
        return []
    common = set.intersection(*key_sets)
    return sorted(common)


def _apply_density_mask(x: torch.Tensor, density: float, seed: int, key: str) -> torch.Tensor:
    if density >= 0.9999:
        return x
    g = torch.Generator(device="cpu")
    seed_i = int((hash(key) & 0x7FFFFFFF) ^ int(seed))
    g.manual_seed(seed_i)
    mask = (torch.rand(x.numel(), generator=g) < density).to(x.dtype).view_as(x)
    return x * mask


def merge_adapter_states(
    states: dict[str, dict[str, torch.Tensor]],
    weights: dict[str, float],
    cfg: MergeConfig,
) -> dict[str, torch.Tensor]:
    """
    Merge per-domain LoRA delta tensors using TIES / DARE-TIES style aggregation.

    Inputs:
      - states: {domain -> {param_name -> tensor}}
      - weights: non-negative domain weights
      - cfg: merge method and sparsity density
    """
    cfg = cfg.normalize()
    domains = [d for d in states.keys() if d in weights and float(weights[d]) > 0.0]
    if not domains:
        return {}

    raw_w = torch.tensor([float(weights[d]) for d in domains], dtype=torch.float32)
    raw_w = raw_w / raw_w.sum().clamp_min(1e-9)
    domain_w = {d: float(w.item()) for d, w in zip(domains, raw_w)}

    key_list = _intersect_keys([states[d] for d in domains])
    out: dict[str, torch.Tensor] = {}
    for key in key_list:
        tensors = [states[d][key].detach().float() for d in domains]
        device = tensors[0].device
        weighted = torch.stack([t * domain_w[d] for t, d in zip(tensors, domains)], dim=0)
        mean = weighted.sum(dim=0)

        signs = torch.sign(torch.stack(tensors, dim=0))
        sign_vote = signs.sum(dim=0)
        agree = sign_vote.abs() >= max(1.0, math.ceil(len(domains) / 2.0))
        agree = agree.to(mean.dtype)

        ties = (mean * agree) + (mean * (1.0 - agree))
        if cfg.method == "dare_ties":
            ties = _apply_density_mask(ties.cpu(), cfg.density, cfg.seed, key).to(device=device)

        out[key] = ties.to(dtype=states[domains[0]][key].dtype)
    return out

