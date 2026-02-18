from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class CounterfactualCreditConfig:
    enabled: bool = True
    ema_beta: float = 0.92
    loss_weight: float = 0.45
    verifier_weight: float = 0.35
    speed_weight: float = 0.20
    clip_value: float = 2.0


class CounterfactualCreditEstimator:
    """
    C3O helper: estimate a compact counterfactual credit signal from learning-curve deltas.

    The signal is deliberately lightweight and robust:
    - uses only recent loss/verifier/speed movements
    - centered with EMA baseline
    - clipped to avoid optimizer instability
    """

    def __init__(self, cfg: CounterfactualCreditConfig) -> None:
        self.cfg = cfg
        self._ema_reward = 0.0
        self._last_loss: float | None = None
        self._last_tps: float | None = None
        self._signal = 0.0

    def state_dict(self) -> dict[str, float | None]:
        return {
            "ema_reward": float(self._ema_reward),
            "last_loss": None if self._last_loss is None else float(self._last_loss),
            "last_tps": None if self._last_tps is None else float(self._last_tps),
            "signal": float(self._signal),
        }

    def load_state_dict(self, payload: dict[str, float | None]) -> None:
        self._ema_reward = float(payload.get("ema_reward", 0.0) or 0.0)
        ll = payload.get("last_loss", None)
        lt = payload.get("last_tps", None)
        self._last_loss = None if ll is None else float(ll)
        self._last_tps = None if lt is None else float(lt)
        self._signal = float(payload.get("signal", 0.0) or 0.0)

    def current_signal(self) -> float:
        if not self.cfg.enabled:
            return 0.0
        return float(self._signal)

    def observe(self, *, loss: float, verifier_pass: float, tokens_per_s: float) -> dict[str, float]:
        if not self.cfg.enabled:
            self._signal = 0.0
            return {
                "c3o_credit_signal": 0.0,
                "c3o_credit_centered": 0.0,
                "c3o_loss_gain": 0.0,
                "c3o_speed_gain": 0.0,
            }

        loss_f = float(loss)
        ver_f = float(verifier_pass)
        tps_f = float(tokens_per_s)

        if self._last_loss is None:
            self._last_loss = loss_f
        if self._last_tps is None:
            self._last_tps = tps_f

        loss_gain = float((self._last_loss - loss_f) / max(abs(self._last_loss), 1e-6))
        speed_gain = float((tps_f - self._last_tps) / max(abs(self._last_tps), 1e-6))
        reward = (
            (self.cfg.loss_weight * loss_gain)
            + (self.cfg.verifier_weight * ver_f)
            + (self.cfg.speed_weight * speed_gain)
        )
        beta = float(min(max(self.cfg.ema_beta, 0.0), 0.9999))
        self._ema_reward = (beta * self._ema_reward) + ((1.0 - beta) * reward)
        centered = reward - self._ema_reward

        clip_v = max(0.1, float(self.cfg.clip_value))
        self._signal = float(min(max(centered, -clip_v), clip_v))

        self._last_loss = loss_f
        self._last_tps = tps_f
        return {
            "c3o_credit_signal": float(self._signal),
            "c3o_credit_centered": float(centered),
            "c3o_loss_gain": float(loss_gain),
            "c3o_speed_gain": float(speed_gain),
            "c3o_verifier": float(ver_f),
        }
