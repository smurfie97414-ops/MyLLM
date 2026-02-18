from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import ast
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any


@dataclass
class RewardWeights:
    w_verifier: float = 0.30
    w_loss_delta: float = 0.30
    w_speed: float = 0.20
    w_stability: float = 0.10
    w_entropy: float = 0.05
    w_replay_quality: float = 0.05

    def clipped(self) -> "RewardWeights":
        vals = {
            "w_verifier": float(min(max(self.w_verifier, 0.0), 1.0)),
            "w_loss_delta": float(min(max(self.w_loss_delta, 0.0), 1.0)),
            "w_speed": float(min(max(self.w_speed, 0.0), 1.0)),
            "w_stability": float(min(max(self.w_stability, 0.0), 1.0)),
            "w_entropy": float(min(max(self.w_entropy, 0.0), 1.0)),
            "w_replay_quality": float(min(max(self.w_replay_quality, 0.0), 1.0)),
        }
        total = sum(vals.values())
        if total <= 1e-8:
            return RewardWeights()
        for k in vals.keys():
            vals[k] = vals[k] / total
        return RewardWeights(**vals)

    def to_metrics(self, prefix: str = "uroboros_reward") -> dict[str, float]:
        return {
            f"{prefix}_w_verifier": float(self.w_verifier),
            f"{prefix}_w_loss_delta": float(self.w_loss_delta),
            f"{prefix}_w_speed": float(self.w_speed),
            f"{prefix}_w_stability": float(self.w_stability),
            f"{prefix}_w_entropy": float(self.w_entropy),
            f"{prefix}_w_replay_quality": float(self.w_replay_quality),
        }

    def to_vector(self) -> list[float]:
        return [
            float(self.w_verifier),
            float(self.w_loss_delta),
            float(self.w_speed),
            float(self.w_stability),
            float(self.w_entropy),
            float(self.w_replay_quality),
        ]

    @staticmethod
    def from_vector(vec: list[float]) -> "RewardWeights":
        v = list(vec) + [0.0] * max(0, 6 - len(vec))
        return RewardWeights(
            w_verifier=float(v[0]),
            w_loss_delta=float(v[1]),
            w_speed=float(v[2]),
            w_stability=float(v[3]),
            w_entropy=float(v[4]),
            w_replay_quality=float(v[5]),
        ).clipped()


@dataclass
class RuntimeSnapshot:
    lr_factor: float
    ttrl_interval: int
    ttrl_budget: int
    replay_bias: float
    reward_weights: RewardWeights


@dataclass
class CurvePoint:
    step: int
    loss: float
    tokens_per_s: float
    verifier_pass: float
    hidden_eval_pass: float
    instability_events: float
    train_loss_real_data: float


@dataclass
class PatchCandidate:
    candidate_id: str
    lr_mul: float = 1.0
    interval_mul: float = 1.0
    budget_mul: float = 1.0
    replay_mul: float = 1.0
    patch_values: dict[str, float] = field(default_factory=dict)
    replaced_feature: str = ""
    replacement_feature: str = ""


@dataclass
class GateResult:
    passed: bool
    p_value: float
    effect_size: float
    relative_gain: float
    verifier_regression: bool
    stability_regression: bool
    reason: str = ""


@dataclass
class UroborosConfig:
    enabled: bool = True
    interval_steps: int = 10
    window_size: int = 64
    bo_trials: int = 8
    patch_trials: int = 3
    significance_alpha: float = 0.05
    min_effect_size: float = 0.15
    min_relative_gain: float = 0.15
    trial_horizon_steps: int = 12
    replacement_policy_strict: bool = True
    patch_commit_enabled: bool = True
    patch_history_file: str = "uroboros_patch_history.jsonl"
    trainer_source_path: str = "training/sigma_trainer.py"
    seed: int = 20260216


@dataclass
class _TrialState:
    candidate: PatchCandidate
    baseline_runtime: RuntimeSnapshot
    baseline_window: list[CurvePoint]
    baseline_score: float
    candidate_weights: RewardWeights
    candidate_vector: list[float]
    start_step: int
    end_step: int


class LearningCurveCritic:
    def __init__(self, window_size: int = 64) -> None:
        self.window_size = max(8, int(window_size))
        self.history: deque[CurvePoint] = deque(maxlen=self.window_size)

    def add(self, point: CurvePoint) -> None:
        self.history.append(point)

    def _slope_ratio(self, values: list[float], invert: bool = False) -> float:
        if len(values) < 2:
            return 0.0
        a = float(values[0])
        b = float(values[-1])
        if invert:
            return float((a - b) / max(abs(a), 1e-6))
        return float((b - a) / max(abs(a), 1e-6))

    def score(self, points: list[CurvePoint] | None = None) -> tuple[float, dict[str, float]]:
        pts = list(self.history) if points is None else list(points)
        if len(pts) < 4:
            return 0.0, {
                "curve_loss_gain": 0.0,
                "curve_speed_gain": 0.0,
                "curve_verifier_gain": 0.0,
                "curve_stability": 0.0,
            }

        loss_vals = [float(p.loss) for p in pts if math.isfinite(float(p.loss))]
        speed_vals = [float(p.tokens_per_s) for p in pts if math.isfinite(float(p.tokens_per_s))]
        verifier_vals = [float(max(p.verifier_pass, p.hidden_eval_pass)) for p in pts]
        instability_vals = [float(p.instability_events) for p in pts]

        loss_gain = self._slope_ratio(loss_vals, invert=True) if len(loss_vals) >= 2 else 0.0
        speed_gain = self._slope_ratio(speed_vals, invert=False) if len(speed_vals) >= 2 else 0.0
        verifier_gain = self._slope_ratio(verifier_vals, invert=False)
        stability = -float(sum(instability_vals) / max(len(instability_vals), 1))

        raw = (0.45 * loss_gain) + (0.20 * speed_gain) + (0.25 * verifier_gain) + (0.10 * stability)
        score = float(max(-1.0, min(1.0, raw)))
        metrics = {
            "curve_loss_gain": float(loss_gain),
            "curve_speed_gain": float(speed_gain),
            "curve_verifier_gain": float(verifier_gain),
            "curve_stability": float(stability),
        }
        return score, metrics


class BayesianRewardController:
    """
    Lightweight Bayesian surrogate controller (no external deps).
    Uses kernel regression mean/variance as acquisition proxy.
    """

    def __init__(self, seed: int = 20260216) -> None:
        self.rng = random.Random(seed)
        self._x: list[list[float]] = []
        self._y: list[float] = []
        self._best_x: list[float] = RewardWeights().to_vector()

    def _kernel(self, a: list[float], b: list[float], length_scale: float = 0.18) -> float:
        s = 0.0
        for i in range(min(len(a), len(b))):
            d = float(a[i] - b[i])
            s += d * d
        return math.exp(-s / max(1e-6, (2.0 * length_scale * length_scale)))

    def _surrogate(self, x: list[float]) -> tuple[float, float]:
        if not self._x:
            return 0.0, 1.0
        weights = [self._kernel(x, xx) for xx in self._x]
        sw = sum(weights)
        if sw <= 1e-8:
            return 0.0, 1.0
        mu = sum(w * y for w, y in zip(weights, self._y)) / sw
        var = sum(w * ((y - mu) ** 2) for w, y in zip(weights, self._y)) / sw
        std = math.sqrt(max(1e-9, var))
        novelty = 1.0 - max(weights)
        return float(mu), float(std + (0.25 * novelty))

    def update(self, x: list[float], reward: float) -> None:
        reward_f = float(reward)
        self._x.append(list(x))
        self._y.append(reward_f)
        if not self._y:
            return
        best_idx = max(range(len(self._y)), key=lambda i: self._y[i])
        self._best_x = list(self._x[best_idx])

    def _sample_candidate(self) -> list[float]:
        sigma = 0.10 if self._x else 0.18
        base = list(self._best_x)
        out = [base[i] + self.rng.gauss(0.0, sigma) for i in range(len(base))]
        out = [max(0.001, min(0.98, v)) for v in out]
        total = sum(out)
        if total <= 1e-8:
            return RewardWeights().to_vector()
        return [v / total for v in out]

    def propose(self, trials: int = 8) -> RewardWeights:
        n_trials = max(1, int(trials))
        best_x = RewardWeights().to_vector()
        best_acq = -1e30
        for _ in range(n_trials):
            x = self._sample_candidate()
            mu, std = self._surrogate(x)
            acq = mu + (1.3 * std)
            if acq > best_acq:
                best_acq = acq
                best_x = x
        return RewardWeights.from_vector(best_x)


class SelfPatchEngine:
    PATCHABLE_DICT_NAME = "UROBOROS_PATCHABLE_DEFAULTS"

    def __init__(self, trainer_source_path: str | Path, history_file: str) -> None:
        self.trainer_source_path = Path(trainer_source_path)
        self.history_file = Path(history_file)
        if not self.trainer_source_path.exists():
            raise RuntimeError(f"UROBOROS trainer source path not found: {self.trainer_source_path}")
        self.rng = random.Random(20260216)

    def propose(self, count: int = 3) -> list[PatchCandidate]:
        candidates: list[PatchCandidate] = []
        for idx in range(max(1, int(count))):
            lr_mul = 1.0 + self.rng.uniform(-0.06, 0.06)
            interval_mul = 1.0 + self.rng.uniform(-0.22, 0.22)
            budget_mul = 1.0 + self.rng.uniform(-0.25, 0.25)
            replay_mul = 1.0 + self.rng.uniform(-0.15, 0.15)
            patch_values = {
                "reward_speed_boost": 1.0 + self.rng.uniform(-0.12, 0.12),
                "reward_quality_boost": 1.0 + self.rng.uniform(-0.12, 0.12),
                "reward_verifier_boost": 1.0 + self.rng.uniform(-0.12, 0.12),
            }
            candidate = PatchCandidate(
                candidate_id=f"patch_{int(time.time())}_{idx}",
                lr_mul=float(lr_mul),
                interval_mul=float(interval_mul),
                budget_mul=float(budget_mul),
                replay_mul=float(replay_mul),
                patch_values=patch_values,
            )
            candidates.append(candidate)
        return candidates

    def commit(self, candidate: PatchCandidate, evidence: dict[str, float]) -> None:
        src = self.trainer_source_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        class _PatchRewriter(ast.NodeTransformer):
            def __init__(self, patch_values: dict[str, float]) -> None:
                super().__init__()
                self.patch_values = patch_values
                self.patched = False

            def visit_Assign(self, node: ast.Assign):  # type: ignore[override]
                if len(node.targets) != 1:
                    return self.generic_visit(node)
                target = node.targets[0]
                if not isinstance(target, ast.Name) or target.id != SelfPatchEngine.PATCHABLE_DICT_NAME:
                    return self.generic_visit(node)
                if not isinstance(node.value, ast.Dict):
                    raise RuntimeError(f"{SelfPatchEngine.PATCHABLE_DICT_NAME} must be a dict literal.")
                key_nodes = node.value.keys
                value_nodes = node.value.values
                new_values: list[ast.expr] = []
                for k_node, v_node in zip(key_nodes, value_nodes):
                    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                        key = str(k_node.value)
                        if key in self.patch_values:
                            new_values.append(ast.Constant(value=float(self.patch_values[key])))
                            continue
                    new_values.append(v_node)
                node.value.values = new_values
                self.patched = True
                return node

        rw = _PatchRewriter(candidate.patch_values)
        new_tree = rw.visit(tree)
        ast.fix_missing_locations(new_tree)
        if not rw.patched:
            raise RuntimeError(
                f"Could not patch {SelfPatchEngine.PATCHABLE_DICT_NAME} in {self.trainer_source_path}."
            )
        new_src = ast.unparse(new_tree)
        self.trainer_source_path.write_text(new_src + "\n", encoding="utf-8")

        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "candidate_id": candidate.candidate_id,
            "patch_values": dict(candidate.patch_values),
            "evidence": {k: float(v) for k, v in evidence.items()},
            "target": str(self.trainer_source_path.as_posix()),
        }
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _welch_significance(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    denom = math.sqrt((va / max(len(a), 1)) + (vb / max(len(b), 1)) + 1e-12)
    if denom <= 1e-12:
        return 0.0, 1.0
    z = (mb - ma) / denom
    p_two_sided = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return float(z), float(max(0.0, min(1.0, p_two_sided)))


def _cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    pooled = math.sqrt(max(1e-12, 0.5 * (va + vb)))
    return float((mb - ma) / pooled)


class ReplacementPolicyEnforcer:
    def __init__(self, alpha: float, min_effect_size: float, min_relative_gain: float, strict: bool = True) -> None:
        self.alpha = float(alpha)
        self.min_effect_size = float(min_effect_size)
        self.min_relative_gain = float(min_relative_gain)
        self.strict = bool(strict)

    def evaluate(
        self,
        *,
        baseline_points: list[CurvePoint],
        trial_points: list[CurvePoint],
        baseline_score: float,
        trial_score: float,
        candidate: PatchCandidate,
    ) -> GateResult:
        if not baseline_points or not trial_points:
            return GateResult(
                passed=False,
                p_value=1.0,
                effect_size=0.0,
                relative_gain=0.0,
                verifier_regression=True,
                stability_regression=True,
                reason="insufficient_points",
            )

        base_curve = [self._point_score(p) for p in baseline_points]
        trial_curve = [self._point_score(p) for p in trial_points]
        _, p_value = _welch_significance(base_curve, trial_curve)
        effect = _cohen_d(base_curve, trial_curve)
        rel_gain = float((trial_score - baseline_score) / max(abs(baseline_score), 1e-6))

        base_verifier = statistics.mean([max(p.verifier_pass, p.hidden_eval_pass) for p in baseline_points])
        trial_verifier = statistics.mean([max(p.verifier_pass, p.hidden_eval_pass) for p in trial_points])
        verifier_reg = trial_verifier + 1e-6 < (base_verifier - 0.01)

        base_instability = statistics.mean([p.instability_events for p in baseline_points])
        trial_instability = statistics.mean([p.instability_events for p in trial_points])
        stability_reg = trial_instability > (base_instability + 1e-6)

        sig_ok = p_value <= self.alpha
        effect_ok = effect >= self.min_effect_size
        gain_ok = rel_gain >= self.min_relative_gain
        replacement_ok = True
        if self.strict and candidate.replaced_feature and not candidate.replacement_feature:
            replacement_ok = False

        passed = bool(sig_ok and effect_ok and gain_ok and (not verifier_reg) and (not stability_reg) and replacement_ok)
        reason = "ok" if passed else "gate_failed"
        return GateResult(
            passed=passed,
            p_value=float(p_value),
            effect_size=float(effect),
            relative_gain=float(rel_gain),
            verifier_regression=bool(verifier_reg),
            stability_regression=bool(stability_reg),
            reason=reason,
        )

    def _point_score(self, p: CurvePoint) -> float:
        return float(
            (0.5 * (-p.loss))
            + (0.2 * math.log(max(p.tokens_per_s, 1e-6)))
            + (0.2 * max(p.verifier_pass, p.hidden_eval_pass))
            - (0.1 * p.instability_events)
        )


class UroborosLoop:
    def __init__(self, cfg: UroborosConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.critic = LearningCurveCritic(window_size=cfg.window_size)
        self.controller = BayesianRewardController(seed=cfg.seed)
        self.patch_engine = SelfPatchEngine(
            trainer_source_path=cfg.trainer_source_path,
            history_file=cfg.patch_history_file,
        )
        self.enforcer = ReplacementPolicyEnforcer(
            alpha=cfg.significance_alpha,
            min_effect_size=cfg.min_effect_size,
            min_relative_gain=cfg.min_relative_gain,
            strict=cfg.replacement_policy_strict,
        )
        self.reward_weights = RewardWeights().clipped()
        self.history: deque[CurvePoint] = deque(maxlen=max(128, cfg.window_size * 4))
        self.active_trial: _TrialState | None = None
        self.last_metrics: dict[str, float] = {}

    def state_dict(self) -> dict[str, Any]:
        payload = {
            "reward_weights": self.reward_weights.to_vector(),
            "controller_x": [list(x) for x in self.controller._x],
            "controller_y": [float(y) for y in self.controller._y],
            "active_trial": None,
            "last_metrics": dict(self.last_metrics),
        }
        if self.active_trial is not None:
            payload["active_trial"] = {
                "candidate_id": self.active_trial.candidate.candidate_id,
                "candidate": {
                    "lr_mul": float(self.active_trial.candidate.lr_mul),
                    "interval_mul": float(self.active_trial.candidate.interval_mul),
                    "budget_mul": float(self.active_trial.candidate.budget_mul),
                    "replay_mul": float(self.active_trial.candidate.replay_mul),
                    "patch_values": dict(self.active_trial.candidate.patch_values),
                    "replaced_feature": str(self.active_trial.candidate.replaced_feature),
                    "replacement_feature": str(self.active_trial.candidate.replacement_feature),
                },
                "baseline_runtime": {
                    "lr_factor": float(self.active_trial.baseline_runtime.lr_factor),
                    "ttrl_interval": int(self.active_trial.baseline_runtime.ttrl_interval),
                    "ttrl_budget": int(self.active_trial.baseline_runtime.ttrl_budget),
                    "replay_bias": float(self.active_trial.baseline_runtime.replay_bias),
                    "reward_weights": self.active_trial.baseline_runtime.reward_weights.to_vector(),
                },
                "baseline_score": float(self.active_trial.baseline_score),
                "baseline_window": [
                    {
                        "step": int(p.step),
                        "loss": float(p.loss),
                        "tokens_per_s": float(p.tokens_per_s),
                        "verifier_pass": float(p.verifier_pass),
                        "hidden_eval_pass": float(p.hidden_eval_pass),
                        "instability_events": float(p.instability_events),
                        "train_loss_real_data": float(p.train_loss_real_data),
                    }
                    for p in self.active_trial.baseline_window
                ],
                "candidate_weights": self.active_trial.candidate_weights.to_vector(),
                "candidate_vector": list(self.active_trial.candidate_vector),
                "start_step": int(self.active_trial.start_step),
                "end_step": int(self.active_trial.end_step),
            }
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        rw = payload.get("reward_weights")
        if isinstance(rw, list):
            self.reward_weights = RewardWeights.from_vector([float(x) for x in rw])
        x = payload.get("controller_x")
        y = payload.get("controller_y")
        if isinstance(x, list) and isinstance(y, list):
            self.controller._x = [list(map(float, row)) for row in x if isinstance(row, list)]
            self.controller._y = [float(v) for v in y]
            if self.controller._y:
                bi = max(range(len(self.controller._y)), key=lambda i: self.controller._y[i])
                self.controller._best_x = list(self.controller._x[bi])
        self.last_metrics = {k: float(v) for k, v in payload.get("last_metrics", {}).items() if isinstance(v, (int, float))}
        active = payload.get("active_trial")
        if isinstance(active, dict):
            cand_raw = active.get("candidate", {})
            base_raw = active.get("baseline_runtime", {})
            candidate = PatchCandidate(
                candidate_id=str(active.get("candidate_id", "restored_candidate")),
                lr_mul=float(cand_raw.get("lr_mul", 1.0)),
                interval_mul=float(cand_raw.get("interval_mul", 1.0)),
                budget_mul=float(cand_raw.get("budget_mul", 1.0)),
                replay_mul=float(cand_raw.get("replay_mul", 1.0)),
                patch_values={k: float(v) for k, v in cand_raw.get("patch_values", {}).items() if isinstance(v, (int, float))},
                replaced_feature=str(cand_raw.get("replaced_feature", "")),
                replacement_feature=str(cand_raw.get("replacement_feature", "")),
            )
            base_rw = RewardWeights.from_vector([float(x) for x in base_raw.get("reward_weights", RewardWeights().to_vector())])
            baseline_runtime = RuntimeSnapshot(
                lr_factor=float(base_raw.get("lr_factor", 1.0)),
                ttrl_interval=int(base_raw.get("ttrl_interval", 1)),
                ttrl_budget=int(base_raw.get("ttrl_budget", 1)),
                replay_bias=float(base_raw.get("replay_bias", 1.0)),
                reward_weights=base_rw,
            )
            candidate_weights = RewardWeights.from_vector([float(x) for x in active.get("candidate_weights", RewardWeights().to_vector())])
            cand_vec = [float(x) for x in active.get("candidate_vector", candidate_weights.to_vector())]
            baseline_window_raw = active.get("baseline_window", [])
            baseline_window: list[CurvePoint] = []
            if isinstance(baseline_window_raw, list):
                for row in baseline_window_raw:
                    if not isinstance(row, dict):
                        continue
                    baseline_window.append(
                        CurvePoint(
                            step=int(row.get("step", 0)),
                            loss=float(row.get("loss", 0.0)),
                            tokens_per_s=float(row.get("tokens_per_s", 0.0)),
                            verifier_pass=float(row.get("verifier_pass", 0.0)),
                            hidden_eval_pass=float(row.get("hidden_eval_pass", 0.0)),
                            instability_events=float(row.get("instability_events", 0.0)),
                            train_loss_real_data=float(row.get("train_loss_real_data", row.get("loss", 0.0))),
                        )
                    )
            self.active_trial = _TrialState(
                candidate=candidate,
                baseline_runtime=baseline_runtime,
                baseline_window=baseline_window,
                baseline_score=float(active.get("baseline_score", 0.0)),
                candidate_weights=candidate_weights,
                candidate_vector=cand_vec,
                start_step=int(active.get("start_step", 0)),
                end_step=int(active.get("end_step", 0)),
            )

    def _window_from_history(self, n: int) -> list[CurvePoint]:
        n_use = min(max(1, int(n)), len(self.history))
        if n_use <= 0:
            return []
        return list(self.history)[-n_use:]

    def _trial_window(self, start_step: int) -> list[CurvePoint]:
        return [p for p in self.history if p.step >= start_step]

    def _capture_point(self, step_idx: int, metrics: dict[str, float]) -> CurvePoint:
        return CurvePoint(
            step=int(step_idx),
            loss=float(metrics.get("loss", 0.0)),
            tokens_per_s=float(metrics.get("raw_tokens_per_s", metrics.get("tokens_per_s", 0.0))),
            verifier_pass=float(metrics.get("public_eval_pass", 0.0)),
            hidden_eval_pass=float(metrics.get("hidden_eval_pass", 0.0)),
            instability_events=float(metrics.get("instability_events", 0.0)),
            train_loss_real_data=float(metrics.get("train_loss_real_data", metrics.get("loss", 0.0))),
        )

    def _build_candidate_runtime(self, baseline: RuntimeSnapshot, candidate: PatchCandidate, rw: RewardWeights) -> RuntimeSnapshot:
        ttrl_interval = max(1, int(round(float(baseline.ttrl_interval) * float(candidate.interval_mul))))
        ttrl_budget = max(1, int(round(float(baseline.ttrl_budget) * float(candidate.budget_mul))))
        return RuntimeSnapshot(
            lr_factor=float(max(0.05, baseline.lr_factor * float(candidate.lr_mul))),
            ttrl_interval=ttrl_interval,
            ttrl_budget=ttrl_budget,
            replay_bias=float(min(max(0.5, baseline.replay_bias * float(candidate.replay_mul)), 2.5)),
            reward_weights=rw,
        )

    def step(
        self,
        *,
        step_idx: int,
        metrics: dict[str, float],
        current_runtime: RuntimeSnapshot,
        apply_runtime: Any,
    ) -> dict[str, float]:
        if not self.cfg.enabled:
            return {}

        point = self._capture_point(step_idx=step_idx, metrics=metrics)
        self.history.append(point)
        self.critic.add(point)
        score, curve_metrics = self.critic.score()
        out = {
            "uroboros_curve_score": float(score),
            **{f"uroboros_{k}": float(v) for k, v in curve_metrics.items()},
            **self.reward_weights.to_metrics(),
            "uroboros_trial_active": 1.0 if self.active_trial is not None else 0.0,
        }

        if self.active_trial is not None and step_idx >= self.active_trial.end_step:
            trial_points = self._trial_window(self.active_trial.start_step)
            gate = self.enforcer.evaluate(
                baseline_points=self.active_trial.baseline_window,
                trial_points=trial_points,
                baseline_score=self.active_trial.baseline_score,
                trial_score=score,
                candidate=self.active_trial.candidate,
            )
            out.update(
                {
                    "uroboros_trial_p_value": float(gate.p_value),
                    "uroboros_trial_effect_size": float(gate.effect_size),
                    "uroboros_trial_relative_gain": float(gate.relative_gain),
                    "uroboros_trial_verifier_regression": float(int(gate.verifier_regression)),
                    "uroboros_trial_stability_regression": float(int(gate.stability_regression)),
                }
            )
            self.controller.update(self.active_trial.candidate_vector, reward=float(gate.relative_gain))
            if gate.passed:
                self.reward_weights = self.active_trial.candidate_weights
                out["uroboros_candidate_promoted"] = 1.0
                out["uroboros_candidate_id"] = float(abs(hash(self.active_trial.candidate.candidate_id)) % 1000000)
                if self.cfg.patch_commit_enabled and self.active_trial.candidate.patch_values:
                    self.patch_engine.commit(
                        self.active_trial.candidate,
                        evidence={
                            "relative_gain": float(gate.relative_gain),
                            "effect_size": float(gate.effect_size),
                            "p_value": float(gate.p_value),
                        },
                    )
                    out["uroboros_patch_committed"] = 1.0
                else:
                    out["uroboros_patch_committed"] = 0.0
            else:
                apply_runtime(self.active_trial.baseline_runtime)
                self.reward_weights = self.active_trial.baseline_runtime.reward_weights
                out["uroboros_candidate_promoted"] = 0.0
                out["uroboros_patch_committed"] = 0.0
            self.active_trial = None

        can_propose = (
            self.active_trial is None
            and (step_idx % max(1, int(self.cfg.interval_steps)) == 0)
            and len(self.history) >= max(8, self.cfg.window_size // 2)
        )
        if can_propose:
            baseline_window = self._window_from_history(self.cfg.window_size)
            baseline_score, _ = self.critic.score(points=baseline_window)
            rw = self.controller.propose(trials=self.cfg.bo_trials)
            candidates = self.patch_engine.propose(count=self.cfg.patch_trials)
            candidate = max(candidates, key=lambda c: abs(c.lr_mul - 1.0) + abs(c.interval_mul - 1.0))
            baseline_runtime = RuntimeSnapshot(
                lr_factor=float(current_runtime.lr_factor),
                ttrl_interval=int(current_runtime.ttrl_interval),
                ttrl_budget=int(current_runtime.ttrl_budget),
                replay_bias=float(current_runtime.replay_bias),
                reward_weights=current_runtime.reward_weights,
            )
            candidate_runtime = self._build_candidate_runtime(baseline_runtime, candidate, rw)
            apply_runtime(candidate_runtime)
            self.active_trial = _TrialState(
                candidate=candidate,
                baseline_runtime=baseline_runtime,
                baseline_window=baseline_window,
                baseline_score=float(baseline_score),
                candidate_weights=rw,
                candidate_vector=rw.to_vector(),
                start_step=int(step_idx + 1),
                end_step=int(step_idx + max(2, int(self.cfg.trial_horizon_steps))),
            )
            out["uroboros_trial_active"] = 1.0
            out["uroboros_trial_started"] = 1.0
            out["uroboros_trial_end_step"] = float(self.active_trial.end_step)

        self.last_metrics = dict(out)
        return out
