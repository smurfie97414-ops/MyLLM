from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class SelfImproverConfig:
    enabled: bool = True
    interval: int = 2
    mutation_sigma: float = 0.08
    frontier_alpha: float = 0.6
    adaptive_sigma_enabled: bool = True
    sigma_ema_beta: float = 0.90
    sigma_gain_up: float = 1.06
    sigma_gain_down: float = 0.94
    sigma_min: float = 0.02
    sigma_max: float = 0.30
    seed: int = 20260216


class SigmaSelfImprover:
    """
    Meta-controller that improves the self-improvement loop itself.

    Parameters are mutated in a small ES-style loop and selected against a
    composite reward frontier (speed, quality, verifier pass, stability).
    """

    def __init__(self, cfg: SelfImproverConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # theta: [lr_mul, ttrl_interval_mul, ttrl_budget_bias, replay_bias]
        self.theta = [0.0, 0.0, 0.0, 0.0]
        self.best_theta = list(self.theta)
        self.best_reward = -1e30
        self.last_mutation: list[float] = [0.0 for _ in self.theta]
        self.phase = 0
        self.plus_reward = 0.0
        self.minus_reward = 0.0
        self.reward_ema = 0.0
        self.reward_gap_ema = 0.0
        self.current_sigma = float(cfg.mutation_sigma)

    def _sample_mutation(self) -> list[float]:
        sigma = max(1e-4, float(self.current_sigma))
        return [self.rng.gauss(0.0, sigma) for _ in self.theta]

    def _reward_center(self, reward: float) -> float:
        beta = 0.92
        self.reward_ema = (beta * self.reward_ema) + ((1.0 - beta) * float(reward))
        return float(reward - self.reward_ema)

    def propose(self, step: int) -> dict[str, float]:
        if not self.cfg.enabled:
            return {}
        if (step % max(1, self.cfg.interval)) != 0:
            return {}
        if self.phase == 0:
            self.last_mutation = self._sample_mutation()
            self.phase = 1
            vec = [a + b for a, b in zip(self.theta, self.last_mutation)]
        elif self.phase == 1:
            self.phase = -1
            vec = [a - b for a, b in zip(self.theta, self.last_mutation)]
        else:
            self.phase = 0
            vec = list(self.theta)
        return self._to_policy(vec)

    def update(self, step: int, reward: float) -> dict[str, float]:
        if not self.cfg.enabled:
            return {}
        if (step % max(1, self.cfg.interval)) != 0:
            return {}
        centered = self._reward_center(reward)
        if self.phase == 1:
            self.plus_reward = centered
        elif self.phase == -1:
            self.minus_reward = centered
            gap = self.plus_reward - self.minus_reward
            if self.cfg.adaptive_sigma_enabled:
                beta = min(max(float(self.cfg.sigma_ema_beta), 0.0), 0.9999)
                abs_gap = abs(gap)
                self.reward_gap_ema = (beta * self.reward_gap_ema) + ((1.0 - beta) * abs_gap)
                if abs_gap < self.reward_gap_ema:
                    self.current_sigma *= float(self.cfg.sigma_gain_up)
                else:
                    self.current_sigma *= float(self.cfg.sigma_gain_down)
                self.current_sigma = min(max(self.current_sigma, float(self.cfg.sigma_min)), float(self.cfg.sigma_max))
            # Simple antithetic ES update.
            lr = 0.12
            denom = max(1e-5, self.current_sigma**2)
            grad = [(gap * m) / denom for m in self.last_mutation]
            self.theta = [max(-4.0, min(4.0, t + (lr * g))) for t, g in zip(self.theta, grad)]
            self.phase = 0
        if centered > self.best_reward:
            self.best_reward = centered
            self.best_theta = list(self.theta)
        out = self._to_policy(self.theta)
        out.update(
            {
                "self_improver_reward_centered": float(centered),
                "self_improver_best_reward": float(self.best_reward),
                "self_improver_phase": float(self.phase),
                "self_improver_sigma": float(self.current_sigma),
                "self_improver_gap_ema": float(self.reward_gap_ema),
            }
        )
        return out

    def _to_policy(self, vec: list[float]) -> dict[str, float]:
        lr_mul = 1.0 + (0.15 * math.tanh(vec[0]))
        ttrl_interval_mul = 1.0 + (0.35 * math.tanh(vec[1]))
        ttrl_budget_bias = 1.0 + (0.50 * math.tanh(vec[2]))
        replay_bias = 1.0 + (0.40 * math.tanh(vec[3]))
        return {
            "policy_lr_mul": float(lr_mul),
            "policy_ttrl_interval_mul": float(ttrl_interval_mul),
            "policy_ttrl_budget_bias": float(ttrl_budget_bias),
            "policy_replay_bias": float(replay_bias),
        }
