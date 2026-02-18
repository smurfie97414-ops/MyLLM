from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class CausalReplayConfig:
    enabled: bool = True
    alpha: float = 0.20
    ema_beta: float = 0.95
    novelty_weight: float = 0.20
    verifier_weight: float = 0.60
    quality_weight: float = 0.20
    horizon_steps: int = 6
    horizon_decay: float = 0.90


class CausalReplayScorer:
    """
    Causal replay scorer.

    Tracks whether replayed samples are associated with measurable short-horizon
    gains and uses that to bias replay priorities toward causally useful samples.
    """

    def __init__(self, cfg: CausalReplayConfig) -> None:
        self.cfg = cfg
        self._causal_gain_ema = 0.0
        self._last_loss: float | None = None
        self._last_verifier: float | None = None
        self._last_replay_used = 0.0
        # Pending delayed-credit windows: [remaining_steps, accum_gain, norm]
        self._pending_windows: list[list[float]] = []
        self._last_horizon_gain = 0.0

    def state_dict(self) -> dict[str, float | None]:
        return {
            "causal_gain_ema": float(self._causal_gain_ema),
            "last_loss": None if self._last_loss is None else float(self._last_loss),
            "last_verifier": None if self._last_verifier is None else float(self._last_verifier),
            "last_replay_used": float(self._last_replay_used),
            "last_horizon_gain": float(self._last_horizon_gain),
            "pending_windows": [
                [float(win[0]), float(win[1]), float(win[2])]
                for win in self._pending_windows
                if len(win) == 3
            ],
        }

    def load_state_dict(self, payload: dict[str, float | None]) -> None:
        self._causal_gain_ema = float(payload.get("causal_gain_ema", 0.0) or 0.0)
        ll = payload.get("last_loss", None)
        lv = payload.get("last_verifier", None)
        self._last_loss = None if ll is None else float(ll)
        self._last_verifier = None if lv is None else float(lv)
        self._last_replay_used = float(payload.get("last_replay_used", 0.0) or 0.0)
        self._last_horizon_gain = float(payload.get("last_horizon_gain", 0.0) or 0.0)
        self._pending_windows = []
        pending = payload.get("pending_windows", [])
        if isinstance(pending, list):
            for item in pending:
                if not isinstance(item, list) or len(item) != 3:
                    continue
                rem = max(0.0, float(item[0]))
                acc = float(item[1])
                norm = max(1e-6, float(item[2]))
                self._pending_windows.append([rem, acc, norm])

    def observe_step(self, *, loss: float, verifier_pass: float, replay_used: bool) -> dict[str, float]:
        if not self.cfg.enabled:
            return {
                "causal_replay_gain_ema": 0.0,
                "causal_replay_step_gain": 0.0,
                "causal_replay_horizon_gain": 0.0,
                "causal_replay_pending_windows": 0.0,
                "causal_replay_used": 0.0,
            }
        lf = float(loss)
        vf = float(verifier_pass)
        if self._last_loss is None:
            self._last_loss = lf
        if self._last_verifier is None:
            self._last_verifier = vf

        loss_gain = float((self._last_loss - lf) / max(abs(self._last_loss), 1e-6))
        verifier_gain = float(vf - float(self._last_verifier))
        step_gain = loss_gain + verifier_gain

        replay_flag = 1.0 if replay_used else 0.0
        horizon_steps = max(1, int(self.cfg.horizon_steps))
        decay = float(min(max(self.cfg.horizon_decay, 0.0), 1.0))
        matured_gains: list[float] = []
        next_pending: list[list[float]] = []
        for win in self._pending_windows:
            rem, acc, norm = float(win[0]), float(win[1]), float(win[2])
            weight = decay ** float(horizon_steps - max(0.0, rem))
            acc += float(weight * step_gain)
            norm += float(weight)
            rem -= 1.0
            if rem <= 0.0:
                matured_gains.append(float(acc / max(norm, 1e-6)))
            else:
                next_pending.append([rem, acc, norm])
        self._pending_windows = next_pending
        if replay_flag > 0.5:
            self._pending_windows.append([float(horizon_steps), 0.0, 0.0])

        if matured_gains:
            horizon_gain = float(sum(matured_gains) / max(len(matured_gains), 1))
        else:
            horizon_gain = float(step_gain if replay_flag > 0.5 else 0.0)
        self._last_horizon_gain = float(horizon_gain)

        if replay_flag > 0.5 or matured_gains:
            b = float(min(max(self.cfg.ema_beta, 0.0), 0.9999))
            self._causal_gain_ema = (b * self._causal_gain_ema) + ((1.0 - b) * horizon_gain)

        self._last_loss = lf
        self._last_verifier = vf
        self._last_replay_used = replay_flag

        return {
            "causal_replay_gain_ema": float(self._causal_gain_ema),
            "causal_replay_step_gain": float(step_gain),
            "causal_replay_horizon_gain": float(self._last_horizon_gain),
            "causal_replay_pending_windows": float(len(self._pending_windows)),
            "causal_replay_used": float(replay_flag),
        }

    def score(self, *, seq: torch.Tensor, verifier_score: float, quality_score: float) -> float:
        if not self.cfg.enabled:
            return float(max(1e-6, quality_score))
        seq_f = seq.float().view(-1)
        novelty = float(torch.unique(seq_f).numel() / max(seq_f.numel(), 1))
        base = (
            (self.cfg.verifier_weight * float(verifier_score))
            + (self.cfg.quality_weight * float(quality_score))
            + (self.cfg.novelty_weight * novelty)
        )
        boosted = base * (1.0 + (self.cfg.alpha * self._causal_gain_ema))
        if not math.isfinite(boosted):
            boosted = base
        return float(max(1e-6, boosted))
