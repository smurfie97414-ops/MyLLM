from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any


@dataclass
class FractalNASConfig:
    interval_steps: int = 50
    ucb_c: float = 1.3
    seed: int = 1337
    candidate_levels: tuple[int, ...] = (16, 32, 64)
    memory_penalty: float = 0.15
    instant_error_penalty: float = 25.0
    verifier_bonus_weight: float = 1.5
    instability_penalty: float = 2.0


class FractalKernelNAS:
    """
    Fractal kernel search:
    - explores self-similar tile sizes
    - edits Triton kernel source AST defaults
    - applies chosen tiling to runtime Sigma Mamba blocks
    """

    def __init__(
        self,
        config: FractalNASConfig,
        kernel_source_path: str | Path,
    ) -> None:
        self.config = config
        self.kernel_source_path = Path(kernel_source_path)
        self.rng = random.Random(config.seed)
        self.candidates = self._build_candidates()
        self.counts = [0 for _ in self.candidates]
        self.values = [0.0 for _ in self.candidates]
        self.last_idx: int | None = None
        self.last_metrics: dict[str, float] = {}
        if not self.kernel_source_path.exists():
            raise RuntimeError(f"Kernel source path not found: {self.kernel_source_path}")

    def _build_candidates(self) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        for lvl in self.config.candidate_levels:
            # self-similar tilings
            out.append({"block_t": int(lvl), "block_n": int(lvl * 4)})
            out.append({"block_t": int(lvl), "block_n": int(lvl * 8)})
        dedup: dict[tuple[int, int], dict[str, int]] = {}
        for c in out:
            dedup[(c["block_t"], c["block_n"])] = c
        return list(dedup.values())

    def _ucb_select(self) -> int:
        total = max(1, sum(self.counts))
        best_idx = 0
        best_score = -1e30
        for i, _ in enumerate(self.candidates):
            n = self.counts[i]
            if n == 0:
                return i
            bonus = self.config.ucb_c * math.sqrt(math.log(total + 1.0) / n)
            score = self.values[i] + bonus
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _rewrite_kernel_defaults(self, block_t: int, block_n: int) -> None:
        src = self.kernel_source_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        class _Rewriter(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):  # type: ignore[override]
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name == "DEFAULT_BLOCK_T":
                        node.value = ast.Constant(value=int(block_t))
                    elif name == "DEFAULT_BLOCK_N":
                        node.value = ast.Constant(value=int(block_n))
                return node

        new_tree = _Rewriter().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_src = ast.unparse(new_tree)
        self.kernel_source_path.write_text(new_src + "\n", encoding="utf-8")

    def _apply_runtime_tiling(self, model: Any, block_t: int, block_n: int) -> int:
        changed = 0
        for module in model.modules():
            core = getattr(module, "core", None)
            if core is None:
                continue
            if hasattr(core, "set_kernel_tiling"):
                core.set_kernel_tiling(block_t=block_t, block_n=block_n)
                changed += 1
        return changed

    def step(
        self,
        *,
        step_idx: int,
        model: Any,
        tokens_per_s: float,
        loss_value: float,
        gpu_mem_reserved_gb: float = 0.0,
        instant_reconstruction_error: float = 0.0,
        verifier_pass: float = 0.0,
        instability_events: float = 0.0,
    ) -> dict[str, float]:
        if (step_idx % max(1, self.config.interval_steps)) != 0:
            return self.last_metrics

        if self.last_idx is not None:
            reward = float(tokens_per_s / max(loss_value, 1e-5))
            reward = reward / (1.0 + (self.config.memory_penalty * max(0.0, float(gpu_mem_reserved_gb))))
            reward = reward - (self.config.instant_error_penalty * max(0.0, float(instant_reconstruction_error)))
            reward = reward + (self.config.verifier_bonus_weight * max(0.0, float(verifier_pass)))
            reward = reward - (self.config.instability_penalty * max(0.0, float(instability_events)))
            i = self.last_idx
            self.counts[i] += 1
            ema_beta = 0.85
            self.values[i] = (ema_beta * self.values[i]) + ((1.0 - ema_beta) * reward)

        idx = self._ucb_select()
        candidate = self.candidates[idx]
        self._rewrite_kernel_defaults(candidate["block_t"], candidate["block_n"])
        changed = self._apply_runtime_tiling(model, candidate["block_t"], candidate["block_n"])
        self.last_idx = idx
        self.last_metrics = {
            "fractal_idx": float(idx),
            "fractal_block_t": float(candidate["block_t"]),
            "fractal_block_n": float(candidate["block_n"]),
            "fractal_runtime_blocks_changed": float(changed),
            "fractal_arm_value": float(self.values[idx]),
            "fractal_arm_count": float(self.counts[idx]),
            "fractal_reward_memory_penalty": float(self.config.memory_penalty),
            "fractal_reward_instant_penalty": float(self.config.instant_error_penalty),
            "fractal_reward_verifier_bonus": float(self.config.verifier_bonus_weight),
            "fractal_reward_instability_penalty": float(self.config.instability_penalty),
        }
        return dict(self.last_metrics)
