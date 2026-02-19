
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback
from typing import Any

import psutil
import torch
import torch.nn.functional as F
import torch.profiler as torch_profiler

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

from evolution.fractal_nas import FractalKernelNAS, FractalNASConfig
from evolution.meta_loop import RewardWeights, RuntimeSnapshot, UroborosConfig, UroborosLoop
from model.sigma_llm import SigmaLLM
from .integrity_guards import (
    IntegrityConfig,
    build_manifest,
    check_feature_effective_calls,
    ensure_proof_inputs,
)
from .counterfactual_credit import CounterfactualCreditConfig, CounterfactualCreditEstimator
from .memory_hack import apply_instant_memory_hack, collect_instant_metrics, iter_instant_wrappers, verify_reversible_cycle
from .replay_causal import CausalReplayConfig, CausalReplayScorer
from .sigma_rl_objectives import SigmaRLConfig, compute_sigma_rl_loss
from .sigma_self_improver import SelfImproverConfig, SigmaSelfImprover
from .sigma_verifier import SigmaTask, build_code_task, build_math_task, load_task_manifest, parse_int_answer, verify_answer
from .hydra_v21 import HydraV21Config, HydraV21Engine


SYNTHETIC_SOURCE_ID = 4

UROBOROS_PATCHABLE_DEFAULTS: dict[str, float] = {
    "reward_speed_boost": 1.0,
    "reward_quality_boost": 1.0,
    "reward_verifier_boost": 1.0,
}


@dataclass
class SigmaTrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    steps: int = 2000
    grad_clip: float = 1.0
    log_interval: int = 10
    mixed_precision: bool = True
    precision: str = "auto"  # auto|bf16|fp16
    output_dir: str = "runs/sigma"
    metrics_file: str = "metrics.jsonl"
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100
    max_checkpoints: int = 5
    resume_latest: bool = True
    moe_aux_weight: float = 0.003

    # TTRL / RL
    ttrl_group_size: int = 5
    ttrl_interval: int = 8
    ttrl_max_new_tokens: int = 24
    ttrl_temperature: float = 0.8
    ttrl_top_k: int = 40
    ttrl_refine_iters: int = 3
    ttrl_retry_temperature_decay: float = 0.75
    ttrl_retry_top_k_decay: float = 0.85
    ttrl_adaptive_budget: bool = True
    ttrl_budget_min: int = 1
    ttrl_budget_max: int = 4
    ttrl_asym_verify_enabled: bool = True
    ttrl_candidate_pool_multiplier: float = 1.8
    ttrl_discriminative_topk: int = 8
    ttrl_discriminative_weight: float = 0.35
    ttrl_fast_refine_on_fail: bool = True
    ttrl_confidence_replay_min: float = 0.20
    replay_capacity: int = 512
    replay_inject_every: int = 2
    causal_replay_enabled: bool = True
    causal_replay_alpha: float = 0.20
    causal_replay_ema_beta: float = 0.95
    causal_replay_novelty_weight: float = 0.20
    causal_replay_verifier_weight: float = 0.60
    causal_replay_quality_weight: float = 0.20
    causal_replay_horizon_steps: int = 6
    causal_replay_horizon_decay: float = 0.90

    sigma_rl_mode: str = "igrpo"
    sigma_rl_clip_eps: float = 0.2
    sigma_rl_entropy_weight: float = 0.001
    sigma_rl_adv_norm: bool = True
    sigma_rl_kl_weight: float = 0.02
    sigma_rl_dispo_logit_temp: float = 1.0
    sigma_rl_auto_mix_enabled: bool = True
    sigma_rl_auto_mix_floor: float = 0.10

    # Verifier
    verifier_math_enabled: bool = True
    verifier_code_enabled: bool = True
    verifier_timeout_ms: int = 2500
    verifier_eval_every: int = 1
    verifier_cascade_enabled: bool = True
    verifier_refine_top_fraction: float = 0.5
    verifier_min_refine_candidates: int = 2

    # Self-evolution / self-acceleration
    self_evolve_interval: int = 4
    self_evolve_history: int = 24
    self_evolve_lr_up: float = 1.03
    self_evolve_lr_down: float = 0.94
    self_evolve_lr_min_factor: float = 0.50
    self_evolve_lr_max_factor: float = 1.80
    self_evolve_ttrl_interval_min: int = 2
    self_evolve_ttrl_interval_max: int = 16
    self_evolve_target_reserved_gb: float = 10.5

    self_improver_enabled: bool = True
    self_improver_interval: int = 2
    self_improver_mutation_sigma: float = 0.08
    self_improver_frontier_alpha: float = 0.6
    self_improver_adaptive_sigma_enabled: bool = True
    self_improver_sigma_ema_beta: float = 0.90
    self_improver_sigma_gain_up: float = 1.06
    self_improver_sigma_gain_down: float = 0.94
    self_improver_sigma_min: float = 0.02
    self_improver_sigma_max: float = 0.30

    # Integrity / anti-fake
    integrity_guards_enabled: bool = True
    proof_mode: bool = False
    strict_feature_usage: bool = True
    integrity_hidden_eval_manifest: str = ""
    integrity_hidden_eval_every: int = 2

    # Fractal NAS
    fractal_interval_steps: int = 50
    fractal_ucb_c: float = 1.3

    # UROBOROS recursive self-improvement
    uroboros_enabled: bool = True
    uroboros_interval: int = 10
    uroboros_window_size: int = 64
    uroboros_bo_trials: int = 8
    uroboros_patch_trials: int = 3
    uroboros_trial_horizon_steps: int = 12
    uroboros_significance_alpha: float = 0.05
    uroboros_min_effect_size: float = 0.15
    uroboros_min_relative_gain: float = 0.15
    replacement_policy_strict: bool = True
    uroboros_patch_commit_enabled: bool = True

    # C3O credit signal
    c3o_credit_enabled: bool = True
    c3o_credit_ema_beta: float = 0.92
    c3o_credit_loss_weight: float = 0.45
    c3o_credit_verifier_weight: float = 0.35
    c3o_credit_speed_weight: float = 0.20
    c3o_credit_clip_value: float = 2.0
    massive_improvement_enforced: bool = True
    forward_research_loop_enabled: bool = True
    metrics_ema_beta: float = 0.90
    crash_report_enabled: bool = True
    first_batch_timeout_sec: float = 240.0
    startup_meta_ramp_steps: int = 64
    startup_ttrl_interval_multiplier: int = 4
    startup_meta_real_data_buffer_steps: int = 8
    bootstrap_seed_batches: int = 64
    perf_profile_enable: bool = False
    perf_warmup_steps: int = 20
    perf_measure_steps: int = 200
    perf_profiler_steps: int = 50
    perf_report_dir: str = ""
    perf_device_time_mode: str = "auto"  # auto|cuda|device|cpu
    perf_sync_boundaries: bool = True
    torch_compile: bool = True
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_dynamic: bool = True
    torch_compile_fullgraph: bool = False

    # Hydra-V2.1
    hydra_enable: bool = False
    hydra_domains: tuple[str, ...] = ("math", "code", "text")
    hydra_steps_per_phase: int = 5000
    hydra_n_candidates: int = 4
    hydra_unverified_cap_text: float = 0.20
    hydra_rollback_interval: int = 200
    hydra_rollback_threshold: float = 0.01
    hydra_update_interval: int = 16
    hydra_dpo_beta: float = 0.1
    hydra_sampo_verbosity_weight: float = 0.02
    hydra_lora_rank: int = 8
    hydra_lora_alpha: float = 16.0
    hydra_lora_lr: float = 1e-4
    merge_method: str = "ties"
    merge_density: float = 1.0
    merge_fold_into_backbone: bool = False
    unsloth_bootstrap_imported: bool = False
    unsloth_preloaded_trl: bool = False
    unsloth_preloaded_transformers: bool = False
    unsloth_preloaded_peft: bool = False
    unsloth_trl_patch_active: bool = False
    unsloth_trl_patch_hits: int = 0
    unsloth_trl_patch_targets: int = 0


class SigmaTrainer:
    def __init__(
        self,
        model: SigmaLLM,
        optimizer: torch.optim.Optimizer,
        dataloader: Any,
        tokenizer: Any,
        config: SigmaTrainConfig,
    ) -> None:
        if not config.device.startswith("cuda"):
            raise RuntimeError("SIGMA trainer requires CUDA.")
        if not torch.cuda.is_available():
            raise RuntimeError("SIGMA trainer requires CUDA, but CUDA is unavailable.")
        self._startup_wall_t0 = time.perf_counter()

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.rng = random.Random(20260216)

        self.output_dir = Path(config.output_dir).expanduser().resolve(strict=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = (self.output_dir / config.metrics_file).resolve(strict=False)
        perf_dir_name = config.perf_report_dir.strip() if config.perf_report_dir else ""
        self.perf_dir = Path(perf_dir_name) if perf_dir_name else (self.output_dir / "perf")
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(config.checkpoint_dir).expanduser()
        checkpoint_is_relative = not checkpoint_path.is_absolute()
        if checkpoint_is_relative:
            executable_dir = Path(sys.argv[0]).expanduser().resolve(strict=False).parent
            current_dir = Path.cwd().resolve(strict=False)
            if current_dir != executable_dir:
                print(
                    "[startup-warning] checkpoint_dir is relative and resolves from current working directory "
                    f"({current_dir}) instead of executable directory ({executable_dir})."
                )
            checkpoint_path = current_dir / checkpoint_path
        self.checkpoint_dir = checkpoint_path.resolve(strict=False)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[startup-paths] output_dir={self.output_dir}")
        print(f"[startup-paths] checkpoint_dir={self.checkpoint_dir}")
        print(f"[startup-paths] metrics_file={self.metrics_path}")

        self.model.to(self.device)
        self.model.train()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        self._ema_beta = min(max(float(config.metrics_ema_beta), 0.0), 0.9999)
        self._loss_ema: float | None = None
        self._tps_ema: float | None = None
        self._gpu_handle = None
        self._nvml_ready = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                dev_index = int(str(config.device).split(":")[1]) if ":" in str(config.device) else 0
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(dev_index)
                self._nvml_ready = True
            except Exception:
                self._gpu_handle = None
                self._nvml_ready = False

        self.integrity_cfg = IntegrityConfig(
            enabled=bool(config.integrity_guards_enabled),
            proof_mode=bool(config.proof_mode),
            hidden_eval_manifest=str(config.integrity_hidden_eval_manifest),
        )
        if self.integrity_cfg.enabled:
            ensure_proof_inputs(self.integrity_cfg)

        patch_stats = apply_instant_memory_hack(
            self.model,
            comp_dim=self.model.config.instant_comp_dim,
            error_threshold=self.model.config.instant_error_threshold,
            reversible_iters=self.model.config.instant_reversible_iters,
        )
        wrappers = list(iter_instant_wrappers(self.model))
        if not wrappers:
            raise RuntimeError("INSTANT patch did not install any reversible wrappers.")
        rev_err = verify_reversible_cycle(
            wrappers[0],
            d_model=self.model.config.d_model,
            device=self.device,
        )
        if rev_err > 0.05:
            raise RuntimeError(f"INSTANT reversible verification failed, rel_error={rev_err:.6f}")

        self.rl_cfg = SigmaRLConfig(
            mode=config.sigma_rl_mode,
            clip_eps=config.sigma_rl_clip_eps,
            entropy_weight=config.sigma_rl_entropy_weight,
            adv_norm=config.sigma_rl_adv_norm,
            kl_weight=config.sigma_rl_kl_weight,
            dispo_logit_temp=config.sigma_rl_dispo_logit_temp,
        )
        self._rl_auto_mix_enabled = bool(config.sigma_rl_auto_mix_enabled)
        self._rl_auto_mix_floor = float(min(max(config.sigma_rl_auto_mix_floor, 0.0), 0.45))
        self._rl_cfg_igrpo = replace(self.rl_cfg, mode="igrpo")
        self._rl_cfg_dispo = replace(self.rl_cfg, mode="dispo")
        self._rl_cfg_gspo = replace(self.rl_cfg, mode="gspo")
        self._rl_cfg_cispo = replace(self.rl_cfg, mode="cispo")
        self.self_improver = SigmaSelfImprover(
            SelfImproverConfig(
                enabled=config.self_improver_enabled,
                interval=config.self_improver_interval,
                mutation_sigma=config.self_improver_mutation_sigma,
                frontier_alpha=config.self_improver_frontier_alpha,
                adaptive_sigma_enabled=config.self_improver_adaptive_sigma_enabled,
                sigma_ema_beta=config.self_improver_sigma_ema_beta,
                sigma_gain_up=config.self_improver_sigma_gain_up,
                sigma_gain_down=config.self_improver_sigma_gain_down,
                sigma_min=config.self_improver_sigma_min,
                sigma_max=config.self_improver_sigma_max,
            )
        )
        self.causal_replay = CausalReplayScorer(
            CausalReplayConfig(
                enabled=bool(config.causal_replay_enabled),
                alpha=float(config.causal_replay_alpha),
                ema_beta=float(config.causal_replay_ema_beta),
                novelty_weight=float(config.causal_replay_novelty_weight),
                verifier_weight=float(config.causal_replay_verifier_weight),
                quality_weight=float(config.causal_replay_quality_weight),
                horizon_steps=int(config.causal_replay_horizon_steps),
                horizon_decay=float(config.causal_replay_horizon_decay),
            )
        )
        self.credit_estimator = CounterfactualCreditEstimator(
            CounterfactualCreditConfig(
                enabled=bool(config.c3o_credit_enabled),
                ema_beta=float(config.c3o_credit_ema_beta),
                loss_weight=float(config.c3o_credit_loss_weight),
                verifier_weight=float(config.c3o_credit_verifier_weight),
                speed_weight=float(config.c3o_credit_speed_weight),
                clip_value=float(config.c3o_credit_clip_value),
            )
        )
        self.reward_weights = RewardWeights().clipped()
        self.uroboros = UroborosLoop(
            UroborosConfig(
                enabled=bool(config.uroboros_enabled),
                interval_steps=int(config.uroboros_interval),
                window_size=int(config.uroboros_window_size),
                bo_trials=int(config.uroboros_bo_trials),
                patch_trials=int(config.uroboros_patch_trials),
                significance_alpha=float(config.uroboros_significance_alpha),
                min_effect_size=float(config.uroboros_min_effect_size),
                min_relative_gain=float(config.uroboros_min_relative_gain),
                trial_horizon_steps=int(config.uroboros_trial_horizon_steps),
                replacement_policy_strict=bool(config.replacement_policy_strict),
                patch_commit_enabled=bool(config.uroboros_patch_commit_enabled),
                patch_history_file=str((self.output_dir / "uroboros_patch_history.jsonl").as_posix()),
                trainer_source_path=str(Path(__file__).resolve().as_posix()),
            )
        )

        self.hidden_eval_tasks: list[SigmaTask] = []
        if self.integrity_cfg.hidden_eval_manifest:
            self.hidden_eval_tasks = load_task_manifest(self.integrity_cfg.hidden_eval_manifest)

        self.feature_calls: dict[str, int] = {
            "feature_instant": 0,
            "feature_diff_mla": 0,
            "feature_sigma_rl": 0,
            "feature_verifier": 0,
            "feature_verifier_cascade": 0,
            "feature_fractal_nas": 0,
            "feature_self_improver": 0,
            "feature_uroboros": 0,
            "feature_causal_replay": 0,
            "feature_c3o_credit": 0,
            "feature_asym_verify": 0,
            "feature_rl_auto_mix": 0,
            "feature_hydra_v21": 0,
        }

        self.data_iter = iter(self.dataloader)
        self.global_step = 0
        self.tokens_seen = 0
        self.replay: deque[tuple[float, torch.Tensor]] = deque(maxlen=max(1, int(config.replay_capacity)))
        self.loss_history: deque[float] = deque(maxlen=max(4, int(config.self_evolve_history)))
        self.tps_history: deque[float] = deque(maxlen=max(4, int(config.self_evolve_history)))
        self.mem_history: deque[float] = deque(maxlen=max(4, int(config.self_evolve_history)))
        self._base_lrs: list[float] = []
        self._lr_factor = 1.0
        self._adaptive_ttrl_interval = max(1, int(config.ttrl_interval))
        self._meta_activation_step = max(
            0,
            int(config.bootstrap_seed_batches) + max(0, int(config.startup_meta_real_data_buffer_steps)),
        )
        self._ttrl_budget = min(max(1, int(config.ttrl_budget_min)), max(1, int(config.ttrl_budget_max)))
        self._replay_bias = 1.0
        self._last_ttrl_reward = 0.0
        self._last_ttrl_confidence = 0.0
        self._last_ttrl_hidden_pass = 0.0
        self._last_ttrl_public_pass = 0.0
        self._last_replay_injected = False
        self._last_policy: dict[str, float] = {}
        self._checkpoint_quarantine_count = 0
        self._checkpoint_resume_step = 0
        self._last_checkpoint_path: str | None = None
        self._last_checkpoint_time: str | None = None
        self._last_log_feature_calls: dict[str, int] = {
            "feature_sigma_rl": 0,
            "feature_fractal_nas": 0,
            "feature_uroboros": 0,
            "feature_self_improver": 0,
            "feature_hydra_v21": 0,
        }
        self._status_latest_path = self.output_dir / "status_latest.json"
        self._first_batch_wall: float | None = None
        self._first_step_wall: float | None = None
        self._startup_init_s = 0.0
        self._perf_profiler: torch_profiler.profile | None = None
        self._perf_profile_started = False
        self._perf_profile_finished = False
        self._perf_profile_end_step = 0
        self._perf_measure_count = 0
        self._perf_sum_step_time = 0.0
        self._perf_sum_core_step_time = 0.0
        self._perf_sum_effective_tps = 0.0
        self._perf_sum_core_tps = 0.0
        self._perf_sum_phase_data = 0.0
        self._perf_sum_phase_train = 0.0
        self._perf_sum_phase_forward = 0.0
        self._perf_sum_phase_optim = 0.0
        self._perf_sum_phase_backward = 0.0
        self._perf_sum_phase_optstep = 0.0
        self._perf_sum_phase_zero_grad = 0.0
        self._perf_sum_phase_ttrl = 0.0
        self._perf_sum_phase_meta = 0.0
        self._perf_summary_written = False
        self._last_profile_moe_share = 0.0
        self._last_profile_time_backend = "unknown"
        self._perf_window_offset_step = 0
        self._model_compiled = False
        self.last_metrics: dict[str, float] = {
            "instant_patched_mamba": float(patch_stats["patched_mamba"]),
            "instant_patched_mla": float(patch_stats["patched_mla"]),
            "instant_reversible_rel_error": float(rev_err),
            "meta_activation_step": float(self._meta_activation_step),
        }

        kernel_src = Path(__file__).resolve().parents[1] / "model" / "sigma_kernel_src.py"
        self.fractal = FractalKernelNAS(
            config=FractalNASConfig(
                interval_steps=config.fractal_interval_steps,
                ucb_c=config.fractal_ucb_c,
            ),
            kernel_source_path=kernel_src,
        )

        self.autocast_dtype = self._resolve_autocast_dtype()
        scaler_enabled = self.autocast_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self._base_lrs = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]

        self.hydra_engine: HydraV21Engine | None = None
        if bool(config.hydra_enable):
            self.hydra_engine = HydraV21Engine(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                output_dir=self.output_dir,
                config=HydraV21Config(
                    enabled=bool(config.hydra_enable),
                    domains=tuple(config.hydra_domains),
                    steps_per_phase=int(config.hydra_steps_per_phase),
                    n_candidates=int(config.hydra_n_candidates),
                    unverified_cap_text=float(config.hydra_unverified_cap_text),
                    rollback_interval=int(config.hydra_rollback_interval),
                    rollback_threshold=float(config.hydra_rollback_threshold),
                    update_interval=int(config.hydra_update_interval),
                    dpo_beta=float(config.hydra_dpo_beta),
                    sampo_verbosity_weight=float(config.hydra_sampo_verbosity_weight),
                    lora_rank=int(config.hydra_lora_rank),
                    lora_alpha=float(config.hydra_lora_alpha),
                    lora_lr=float(config.hydra_lora_lr),
                    merge_method=str(config.merge_method),
                    merge_density=float(config.merge_density),
                    merge_fold_into_backbone=bool(config.merge_fold_into_backbone),
                ),
                verifier_math_enabled=bool(config.verifier_math_enabled),
                verifier_code_enabled=bool(config.verifier_code_enabled),
            )

        resumed = self._resume_if_available() if config.resume_latest else 0
        if resumed > 0:
            self.global_step = resumed
        self._perf_window_offset_step = int(self.global_step)

        if self.integrity_cfg.enabled:
            tracked_files = [
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "sigma_rl_objectives.py",
                Path(__file__).resolve().parent / "sigma_verifier.py",
                Path(__file__).resolve().parent / "sigma_self_improver.py",
                Path(__file__).resolve().parent / "integrity_guards.py",
                Path(__file__).resolve().parents[1] / "model" / "sigma_llm.py",
                Path(__file__).resolve().parents[1] / "evolution" / "fractal_nas.py",
                Path(__file__).resolve().parents[1] / "evolution" / "meta_loop.py",
            ]
            build_manifest(
                output_dir=self.output_dir,
                config_payload=self.config.__dict__,
                tracked_files=tracked_files,
                seeds={"python_rng": 20260216, "torch_initial_seed": int(torch.initial_seed())},
            )

        print(
            "[sigma-trainer] init "
            f"features={self.model.algorithmic_features()} "
            f"instant={self.last_metrics} "
            f"proof_mode={int(self.config.proof_mode)}"
        )
        self._startup_init_s = max(time.perf_counter() - self._startup_wall_t0, 0.0)

    def _resolve_autocast_dtype(self):
        if not self.config.mixed_precision:
            return None
        precision = self.config.precision.lower()
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "auto":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        raise ValueError(f"Unsupported precision mode: {self.config.precision}")

    def _autocast_ctx(self):
        if self.autocast_dtype is None:
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=True)

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_step_{step:08d}.pt"

    def _checkpoint_candidates(self) -> list[Path]:
        return sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

    def _quarantine_checkpoint(self, ckpt: Path, reason: Exception) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        bad = ckpt.with_name(f"{ckpt.name}.corrupt_{stamp}_{os.getpid()}")
        ckpt.replace(bad)
        self._checkpoint_quarantine_count += 1
        print(f"[sigma-trainer] quarantined corrupted checkpoint: {ckpt} -> {bad}. reason={reason}")

    def _save_checkpoint(self, step: int) -> None:
        replay_payload = [{"score": float(score), "seq": seq.clone().cpu()} for score, seq in list(self.replay)]
        checkpoint_model = getattr(self.model, "_orig_mod", self.model)
        ckpt_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        payload = {
            "step": int(step),
            "model": checkpoint_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "tokens_seen": int(self.tokens_seen),
            "lr_factor": float(self._lr_factor),
            "adaptive_ttrl_interval": int(self._adaptive_ttrl_interval),
            "ttrl_budget": int(self._ttrl_budget),
            "replay_bias": float(self._replay_bias),
            "last_ttrl_reward": float(self._last_ttrl_reward),
            "last_ttrl_confidence": float(self._last_ttrl_confidence),
            "feature_calls": dict(self.feature_calls),
            "replay": replay_payload,
            "reward_weights": self.reward_weights.to_vector(),
            "uroboros_state": self.uroboros.state_dict(),
            "causal_replay_state": self.causal_replay.state_dict(),
            "credit_state": self.credit_estimator.state_dict(),
            "last_replay_injected": bool(self._last_replay_injected),
            "time": ckpt_time,
        }
        if self.hydra_engine is not None:
            payload["hydra_state"] = self.hydra_engine.state_dict()
        path = self._checkpoint_path(step)
        tmp = path.with_name(path.name + ".tmp")
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                torch.save(payload, tmp)
                os.replace(tmp, path)
                last_exc = None
                break
            except PermissionError as exc:
                last_exc = exc
            except OSError as exc:
                # Windows may surface "file in use" as winerror=32.
                if int(getattr(exc, "winerror", 0) or 0) == 32:
                    last_exc = exc
                else:
                    raise
            wait_s = 0.15 * float(attempt + 1)
            time.sleep(wait_s)
        if last_exc is not None:
            raise RuntimeError(
                f"checkpoint save failed after retries: path={path} tmp={tmp} error={last_exc}"
            ) from last_exc
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        checkpoints = self._checkpoint_candidates()
        while len(checkpoints) > self.config.max_checkpoints:
            old = checkpoints.pop(0)
            old.unlink(missing_ok=True)
        self._last_checkpoint_path = str(path)
        self._last_checkpoint_time = ckpt_time
        print(f"[sigma-trainer] checkpoint saved: {path}")
    def _resume_if_available(self) -> int:
        candidates = self._checkpoint_candidates()
        for ckpt in reversed(candidates):
            try:
                payload = torch.load(ckpt, map_location="cpu")
                if not isinstance(payload, dict):
                    raise RuntimeError("checkpoint payload is not a dict")
                model_state = payload["model"]
                resume_model = getattr(self.model, "_orig_mod", self.model)
                load_res = resume_model.load_state_dict(model_state, strict=False)
                missing = list(getattr(load_res, "missing_keys", []))
                unexpected = list(getattr(load_res, "unexpected_keys", []))
                bad_missing = [k for k in missing if ".adapters." not in str(k)]
                bad_unexpected = [k for k in unexpected if ".adapters." not in str(k)]
                if bad_missing or bad_unexpected:
                    raise RuntimeError(
                        "checkpoint model mismatch: "
                        f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}"
                    )
                self.optimizer.load_state_dict(payload["optimizer"])
                scaler_state = payload.get("scaler")
                if scaler_state is not None and self.scaler.is_enabled():
                    self.scaler.load_state_dict(scaler_state)
                self.tokens_seen = int(payload.get("tokens_seen", 0))
                self._lr_factor = float(payload.get("lr_factor", self._lr_factor))
                self._adaptive_ttrl_interval = int(payload.get("adaptive_ttrl_interval", self._adaptive_ttrl_interval))
                self._ttrl_budget = int(payload.get("ttrl_budget", self._ttrl_budget))
                self._replay_bias = float(payload.get("replay_bias", self._replay_bias))
                rw_vec = payload.get("reward_weights")
                if isinstance(rw_vec, list):
                    self.reward_weights = RewardWeights.from_vector([float(x) for x in rw_vec])
                uroboros_state = payload.get("uroboros_state")
                if isinstance(uroboros_state, dict):
                    self.uroboros.load_state_dict(uroboros_state)
                causal_state = payload.get("causal_replay_state")
                if isinstance(causal_state, dict):
                    self.causal_replay.load_state_dict(causal_state)
                credit_state = payload.get("credit_state")
                if isinstance(credit_state, dict):
                    self.credit_estimator.load_state_dict(credit_state)
                hydra_state = payload.get("hydra_state")
                if self.hydra_engine is not None and isinstance(hydra_state, dict):
                    self.hydra_engine.load_state_dict(hydra_state)
                self._last_replay_injected = bool(payload.get("last_replay_injected", False))
                calls = payload.get("feature_calls")
                if isinstance(calls, dict):
                    for k in self.feature_calls.keys():
                        self.feature_calls[k] = int(calls.get(k, self.feature_calls[k]))
                self._set_optimizer_lr_factor(self._lr_factor)
                replay_payload = payload.get("replay", [])
                if isinstance(replay_payload, list):
                    self.replay.clear()
                    for item in replay_payload:
                        if not isinstance(item, dict):
                            continue
                        score = float(item.get("score", 0.0))
                        seq = item.get("seq")
                        if isinstance(seq, torch.Tensor) and seq.numel() >= 2:
                            self.replay.append((score, seq.cpu()))
                step = int(payload["step"])
                self._checkpoint_resume_step = step
                if isinstance(payload.get("time"), str):
                    self._last_checkpoint_time = str(payload.get("time"))
                self._last_checkpoint_path = str(ckpt)
                print(f"[sigma-trainer] resumed from {ckpt} step={step}")
                return step
            except Exception as exc:
                self._quarantine_checkpoint(ckpt, exc)
                continue
        return 0

    def _next_batch(self) -> dict[str, torch.Tensor]:
        def _fetch_once() -> dict[str, torch.Tensor]:
            try:
                return next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                return next(self.data_iter)

        if self.global_step <= 1 and float(self.config.first_batch_timeout_sec) > 0:
            timeout_s = max(1.0, float(self.config.first_batch_timeout_sec))
            result_box: dict[str, Any] = {}
            exc_box: dict[str, BaseException] = {}
            done = threading.Event()

            def _target() -> None:
                try:
                    result_box["batch"] = _fetch_once()
                except BaseException as exc:
                    exc_box["exc"] = exc
                finally:
                    done.set()

            # Use daemon thread so timeout cannot leave a non-daemon worker alive.
            t = threading.Thread(target=_target, name="first_batch", daemon=True)
            t.start()
            if not done.wait(timeout=timeout_s):
                raise RuntimeError(
                    f"first batch timeout after {timeout_s:.1f}s; data source blocked before step start"
                )
            if "exc" in exc_box:
                raise RuntimeError("first batch fetch failed") from exc_box["exc"]
            batch = result_box.get("batch")
            if not isinstance(batch, dict):
                raise RuntimeError("first batch fetch returned invalid payload.")
        else:
            batch = _fetch_once()
        if not isinstance(batch, dict) or "input_ids" not in batch or "labels" not in batch:
            raise RuntimeError("Dataloader batch must be dict with input_ids and labels.")
        if self._first_batch_wall is None:
            self._first_batch_wall = time.perf_counter()
        return batch

    def _effective_ttrl_interval(self) -> int:
        base = max(1, int(self._adaptive_ttrl_interval))
        ramp_steps = max(0, int(self.config.startup_meta_ramp_steps))
        if ramp_steps <= 0 or self.global_step > ramp_steps:
            return base
        mult = max(1, int(self.config.startup_ttrl_interval_multiplier))
        return max(1, base * mult)

    def _meta_features_ready(self) -> bool:
        return bool(self.global_step > int(self._meta_activation_step))

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        source_ids = batch.get("source_id", None)
        if source_ids is not None:
            source_ids = source_ids.to(self.device, non_blocking=True).view(-1)
        if input_ids.ndim != 2 or labels.ndim != 2:
            raise RuntimeError(f"Expected rank-2 input_ids/labels, got {tuple(input_ids.shape)} and {tuple(labels.shape)}")
        return input_ids, labels, source_ids

    def _forward_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        source_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.model(input_ids)
        token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        token_loss = token_loss.view(input_ids.size(0), input_ids.size(1))
        loss = token_loss.mean()
        aux = self.model.collect_aux_loss()
        loss = loss + (self.config.moe_aux_weight * aux)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss detected: {float(loss.detach().item())}")
        if source_ids is None:
            return loss, loss.detach(), torch.tensor(float("nan"), device=loss.device)

        is_synth = (source_ids == SYNTHETIC_SOURCE_ID).view(-1, 1).expand_as(token_loss)
        is_real = ~is_synth
        real_loss = token_loss[is_real].mean() if is_real.any() else torch.tensor(float("nan"), device=loss.device)
        synth_loss = token_loss[is_synth].mean() if is_synth.any() else torch.tensor(float("nan"), device=loss.device)
        return loss, real_loss.detach(), synth_loss.detach()

    def _optim_step(self, loss: torch.Tensor) -> tuple[float, dict[str, float]]:
        t0 = time.perf_counter()
        if bool(self.config.c3o_credit_enabled) and hasattr(self.optimizer, "set_credit_signal"):
            try:
                self.optimizer.set_credit_signal(float(self.credit_estimator.current_signal()))
                self.feature_calls["feature_c3o_credit"] += 1
            except Exception as exc:
                raise RuntimeError(f"failed to set C3O credit signal: {exc}") from exc
        self.optimizer.zero_grad(set_to_none=True)
        t_after_zero = time.perf_counter()
        use_global_clip = bool(self.config.grad_clip > 0) and not bool(
            getattr(self.optimizer, "supports_internal_grad_clip", False)
        )
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if use_global_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            t_after_backward = time.perf_counter()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if use_global_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            t_after_backward = time.perf_counter()
            self.optimizer.step()
        t_end = time.perf_counter()
        return float(loss.detach().item()), {
            "zero_grad_s": float(max(t_after_zero - t0, 0.0)),
            "backward_s": float(max(t_after_backward - t_after_zero, 0.0)),
            "opt_step_s": float(max(t_end - t_after_backward, 0.0)),
            "optim_total_s": float(max(t_end - t0, 0.0)),
        }

    def _set_optimizer_lr_factor(self, factor: float) -> None:
        lo = max(0.05, float(self.config.self_evolve_lr_min_factor))
        hi = max(lo, float(self.config.self_evolve_lr_max_factor))
        self._lr_factor = min(max(float(factor), lo), hi)
        if not self._base_lrs:
            self._base_lrs = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]
        for idx, group in enumerate(self.optimizer.param_groups):
            base_lr = self._base_lrs[min(idx, len(self._base_lrs) - 1)]
            group["lr"] = base_lr * self._lr_factor

    def _sample_replay(self) -> torch.Tensor | None:
        if not self.replay:
            return None
        scores = [max(1e-6, float(s)) for s, _ in self.replay]
        total = sum(scores)
        draw = self.rng.random() * total
        acc = 0.0
        for score, seq in self.replay:
            acc += max(1e-6, float(score))
            if acc >= draw:
                return seq
        return self.replay[-1][1]

    def _push_replay(self, seq: torch.Tensor, score: float) -> None:
        novelty_quality = float(torch.unique(seq).numel() / max(seq.numel(), 1))
        priority = self.causal_replay.score(
            seq=seq.detach().cpu(),
            verifier_score=float(score),
            quality_score=float(novelty_quality),
        )
        self.replay.append((float(priority), seq.detach().cpu()))
        self.feature_calls["feature_causal_replay"] += 1

    def _inject_replay(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_replay_injected = False
        if not self.replay:
            return input_ids, labels
        if (self.global_step % max(1, int(self.config.replay_inject_every))) != 0:
            return input_ids, labels
        sampled = self._sample_replay()
        if sampled is None:
            return input_ids, labels
        seq = sampled.to(self.device)
        if seq.numel() < 2:
            return input_ids, labels
        seq = seq[-(input_ids.size(1) + 1) :]
        if seq.numel() < (input_ids.size(1) + 1):
            pad = torch.full((input_ids.size(1) + 1 - seq.numel(),), fill_value=int(seq[-1]), device=self.device, dtype=seq.dtype)
            seq = torch.cat((pad, seq), dim=0)
        input_ids = input_ids.clone()
        labels = labels.clone()
        input_ids[0] = seq[:-1]
        labels[0] = seq[1:]
        self._last_replay_injected = True
        return input_ids, labels

    def _select_task(self) -> SigmaTask:
        domains: list[str] = []
        if self.config.verifier_math_enabled:
            domains.append("math")
        if self.config.verifier_code_enabled:
            domains.append("code")
        if not domains:
            raise RuntimeError("Verifier is disabled for both math and code.")
        chosen = self.rng.choice(domains)
        if chosen == "math":
            return build_math_task(self.rng)
        return build_code_task(self.rng)

    def _verify_sequence(self, task: SigmaTask, seq: torch.Tensor) -> tuple[float, bool]:
        text = self._safe_decode(seq.tolist())
        if bool(self.config.verifier_cascade_enabled):
            parsed = parse_int_answer(text)
            if not parsed:
                self.feature_calls["feature_verifier"] += 1
                return 0.0, False
            if parsed == task.expected:
                self.feature_calls["feature_verifier"] += 1
                return 1.0, True
        result = verify_answer(task, text)
        self.feature_calls["feature_verifier"] += 1
        return float(result.score), bool(result.passed)

    def _should_refine_candidate(self, *, rank_idx: int, shortlist_len: int) -> bool:
        if not bool(self.config.ttrl_fast_refine_on_fail):
            return False
        if not bool(self.config.verifier_cascade_enabled):
            return True
        frac = float(min(max(self.config.verifier_refine_top_fraction, 0.0), 1.0))
        min_refine = max(0, int(self.config.verifier_min_refine_candidates))
        refine_limit = max(min_refine, int(math.ceil(float(shortlist_len) * frac)))
        refine_limit = min(max(refine_limit, 0), max(shortlist_len, 0))
        return int(rank_idx) < int(refine_limit)

    def _sequence_confidence_scores(self, seq_batch: torch.Tensor, prompt_len: int) -> torch.Tensor:
        x = seq_batch[:, :-1]
        y = seq_batch[:, 1:]
        with torch.no_grad():
            logits = self.model(x)
            logp = F.log_softmax(logits, dim=-1)
            sel = logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        start_idx = max(0, int(prompt_len - 1))
        mask = torch.zeros_like(sel, dtype=torch.float32)
        if start_idx < mask.size(1):
            mask[:, start_idx:] = 1.0
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (sel * mask).sum(dim=1) / denom

    def _generate_one_with_refinement(self, prompt_tensor: torch.Tensor, task: SigmaTask) -> tuple[torch.Tensor, float]:
        best_seq = prompt_tensor
        best_score = -1.0
        for attempt in range(max(1, int(self.config.ttrl_refine_iters))):
            temp = max(0.15, float(self.config.ttrl_temperature) * (float(self.config.ttrl_retry_temperature_decay) ** attempt))
            top_k = max(4, int(float(self.config.ttrl_top_k) * (float(self.config.ttrl_retry_top_k_decay) ** attempt)))
            out = self.model.generate(
                prompt_tensor,
                max_new_tokens=self.config.ttrl_max_new_tokens,
                temperature=temp,
                top_k=top_k,
            )
            score, passed = self._verify_sequence(task=task, seq=out[0])
            if score > best_score:
                best_score = score
                best_seq = out.detach()
            if passed:
                return best_seq, best_score
        return best_seq, best_score

    def _completion_logprobs(
        self,
        seq_batch: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = seq_batch[:, :-1]
        y = seq_batch[:, 1:]
        with torch.no_grad():
            old_logits = self.model(x)
            old_logp = F.log_softmax(old_logits, dim=-1)
            old_sel = old_logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        new_logits = self.model(x)
        new_logp = F.log_softmax(new_logits, dim=-1)
        new_sel = new_logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        start_idx = max(0, int(prompt_len - 1))
        mask = torch.zeros_like(new_sel)
        if start_idx < mask.size(1):
            mask[:, start_idx:] = 1.0
        return new_sel, old_sel, mask

    def _compute_rl_mix_weights(self, rewards: torch.Tensor, diversity: torch.Tensor) -> dict[str, float]:
        reward_mean = float(rewards.mean().item())
        reward_var = float(rewards.var(unbiased=False).item()) if rewards.numel() > 1 else 0.0
        diversity_mean = float(diversity.mean().item()) if diversity.numel() > 0 else 0.0

        exploit = min(max(reward_mean * (1.0 - min(max(reward_var, 0.0), 1.0)), 0.0), 1.0)
        explore = min(max((1.0 - reward_mean) + (0.75 * reward_var), 0.0), 1.0)
        stability = min(max((1.0 - abs((2.0 * reward_mean) - 1.0)) + (0.5 * reward_var), 0.0), 1.0)

        raw = {
            "cispo": float(0.15 + (0.45 * exploit)),
            "igrpo": float(0.10 + (0.30 * exploit)),
            "dispo": float(0.10 + (0.35 * explore)),
            "gspo": float(0.10 + (0.35 * stability) + (0.10 * (1.0 - diversity_mean))),
        }
        floor = float(min(max(self._rl_auto_mix_floor, 0.0), 0.45))
        for k in list(raw.keys()):
            raw[k] = max(floor, raw[k])
        total = sum(raw.values())
        if total <= 1e-8:
            return {"cispo": 0.25, "igrpo": 0.25, "dispo": 0.25, "gspo": 0.25}
        return {k: float(v / total) for k, v in raw.items()}

    def _compute_rl_loss(
        self,
        *,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        diversity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self._rl_auto_mix_enabled:
            return compute_sigma_rl_loss(
                logp_new=logp_new,
                logp_old=logp_old,
                rewards=rewards,
                mask=mask,
                cfg=self.rl_cfg,
            )

        self.feature_calls["feature_rl_auto_mix"] += 1
        mix = self._compute_rl_mix_weights(rewards=rewards, diversity=diversity)
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, float] = {}
        for mode, cfg in (
            ("cispo", self._rl_cfg_cispo),
            ("igrpo", self._rl_cfg_igrpo),
            ("dispo", self._rl_cfg_dispo),
            ("gspo", self._rl_cfg_gspo),
        ):
            loss_i, met_i = compute_sigma_rl_loss(
                logp_new=logp_new,
                logp_old=logp_old,
                rewards=rewards,
                mask=mask,
                cfg=cfg,
            )
            losses[mode] = loss_i
            metrics[f"sigma_rl_component_{mode}_loss"] = float(loss_i.detach().item())
            for mk, mv in met_i.items():
                metrics[f"{mk}_{mode}"] = float(mv)

        combined = (
            (mix["cispo"] * losses["cispo"])
            + (mix["igrpo"] * losses["igrpo"])
            + (mix["dispo"] * losses["dispo"])
            + (mix["gspo"] * losses["gspo"])
        )
        metrics["sigma_rl_auto_mix_enabled"] = 1.0
        metrics["sigma_rl_mix_w_cispo"] = float(mix["cispo"])
        metrics["sigma_rl_mix_w_igrpo"] = float(mix["igrpo"])
        metrics["sigma_rl_mix_w_dispo"] = float(mix["dispo"])
        metrics["sigma_rl_mix_w_gspo"] = float(mix["gspo"])
        return combined, metrics

    def _eval_task_set(self, tasks: list[SigmaTask], max_tasks: int = 4) -> float:
        if not tasks:
            return 0.0
        sample_n = min(max_tasks, len(tasks))
        picked = self.rng.sample(tasks, sample_n) if len(tasks) > sample_n else list(tasks)
        passed = 0
        for task in picked:
            p_ids = self.tokenizer.encode(task.prompt)
            p = torch.tensor(p_ids, device=self.device, dtype=torch.long).unsqueeze(0)
            out = self.model.generate(
                p,
                max_new_tokens=self.config.ttrl_max_new_tokens,
                temperature=max(0.2, self.config.ttrl_temperature),
                top_k=max(8, self.config.ttrl_top_k),
            )
            text = self._safe_decode(out[0].tolist())
            vr = verify_answer(task, text)
            self.feature_calls["feature_verifier"] += 1
            passed += int(vr.passed)
        return float(passed / max(len(picked), 1))

    def _safe_decode(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode([int(x) for x in token_ids])
        except Exception:
            enc = getattr(self.tokenizer, "_enc", None)
            if enc is not None and hasattr(enc, "decode_single_token_bytes"):
                parts: list[str] = []
                for tid in token_ids:
                    try:
                        b = enc.decode_single_token_bytes(int(tid))
                        parts.append(b.decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
                return "".join(parts)
            filtered = [int(t) for t in token_ids if 0 <= int(t) < int(getattr(self.tokenizer, "vocab_size", 10**9))]
            try:
                return self.tokenizer.decode(filtered)
            except Exception:
                return ""
    def _run_ttrl(self) -> dict[str, float]:
        task = self._select_task()
        prompt_ids = self.tokenizer.encode(task.prompt)
        prompt_tensor = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        group_size = max(2, int(self.config.ttrl_group_size))
        budget = max(1, int(self._ttrl_budget))

        seqs: list[torch.Tensor] = []
        rewards: list[float] = []
        asym_enabled = bool(self.config.ttrl_asym_verify_enabled)
        asym_pool = float(group_size)
        asym_verified = float(group_size)
        asym_discriminative_mean = 0.0
        refined_candidates = 0
        if asym_enabled:
            if bool(self.config.verifier_cascade_enabled):
                self.feature_calls["feature_verifier_cascade"] += 1
            pool_mult = max(1.0, float(self.config.ttrl_candidate_pool_multiplier))
            pool_size = max(group_size, int(round(group_size * pool_mult)))
            draft_candidates: list[torch.Tensor] = []
            for _ in range(pool_size):
                out = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=self.config.ttrl_max_new_tokens,
                    temperature=max(0.2, float(self.config.ttrl_temperature) * 1.10),
                    top_k=max(8, int(float(self.config.ttrl_top_k) * 1.10)),
                )
                draft_candidates.append(out[0].detach())
            seq_batch_draft = torch.nn.utils.rnn.pad_sequence(
                draft_candidates,
                batch_first=True,
                padding_value=int(prompt_ids[-1]) if prompt_ids else 0,
            ).to(self.device)
            conf_scores = self._sequence_confidence_scores(seq_batch_draft, prompt_len=len(prompt_ids))
            conf_scores = torch.sigmoid(conf_scores)
            diversity = torch.tensor(
                [float(torch.unique(seq).numel() / max(seq.numel(), 1)) for seq in draft_candidates],
                device=self.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
            discrim_w = min(max(float(self.config.ttrl_discriminative_weight), 0.0), 1.0)
            discriminative = (discrim_w * conf_scores) + ((1.0 - discrim_w) * diversity)
            asym_discriminative_mean = float(discriminative.mean().item())
            shortlist_k = max(group_size, min(pool_size, int(self.config.ttrl_discriminative_topk)))
            top_idx = torch.topk(discriminative, k=shortlist_k, largest=True).indices.tolist()
            shortlisted = [draft_candidates[i] for i in top_idx]
            verified: list[tuple[torch.Tensor, float]] = []
            for cand_rank, seq in enumerate(shortlisted):
                score, passed = self._verify_sequence(task=task, seq=seq)
                best_seq = seq
                best_score = score
                if (not passed) and self._should_refine_candidate(rank_idx=cand_rank, shortlist_len=len(shortlisted)):
                    refined_seq, refined_score = self._generate_one_with_refinement(prompt_tensor=prompt_tensor, task=task)
                    candidate = refined_seq[0]
                    if refined_score > best_score:
                        best_seq = candidate
                        best_score = float(refined_score)
                    refined_candidates += 1
                verified.append((best_seq, float(best_score)))
            verified.sort(key=lambda x: x[1], reverse=True)
            while len(verified) < group_size:
                seq_refined, score_refined = self._generate_one_with_refinement(prompt_tensor=prompt_tensor, task=task)
                verified.append((seq_refined[0], float(score_refined)))
            chosen = verified[:group_size]
            seqs = [seq for seq, _ in chosen]
            rewards = [float(score) for _, score in chosen]
            asym_pool = float(pool_size)
            asym_verified = float(len(shortlisted))
            self.feature_calls["feature_asym_verify"] += 1
        else:
            for _ in range(group_size):
                seq, score = self._generate_one_with_refinement(prompt_tensor=prompt_tensor, task=task)
                seqs.append(seq[0])
                rewards.append(float(score))

        seq_batch = torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=int(prompt_ids[-1]) if prompt_ids else 0,
        )
        raw_rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        rewards_t = raw_rewards_t
        diversity_scores = []
        for seq in seqs:
            unique_tokens = float(torch.unique(seq).numel())
            diversity_scores.append(unique_tokens / max(float(seq.numel()), 1.0))
        diversity_t = torch.tensor(diversity_scores, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        quality_t = (0.7 * rewards_t) + (0.3 * diversity_t)
        reward_boost = float(UROBOROS_PATCHABLE_DEFAULTS.get("reward_verifier_boost", 1.0))
        entropy_boost = float(UROBOROS_PATCHABLE_DEFAULTS.get("reward_quality_boost", 1.0))
        replay_boost = float(UROBOROS_PATCHABLE_DEFAULTS.get("reward_speed_boost", 1.0))
        blend_sum = (
            (self.reward_weights.w_verifier * reward_boost)
            + (self.reward_weights.w_entropy * entropy_boost)
            + (self.reward_weights.w_replay_quality * replay_boost)
        )
        if blend_sum <= 1e-8:
            blend_sum = 1.0
        rewards_t = (
            ((self.reward_weights.w_verifier * reward_boost) * rewards_t)
            + ((self.reward_weights.w_entropy * entropy_boost) * diversity_t)
            + ((self.reward_weights.w_replay_quality * replay_boost) * quality_t)
        ) / blend_sum
        x = seq_batch[:, :-1]
        y = seq_batch[:, 1:]
        start_idx = max(0, int(len(prompt_ids) - 1))
        mask = torch.zeros_like(x, dtype=torch.float32)
        if start_idx < mask.size(1):
            mask[:, start_idx:] = 1.0
        with torch.no_grad():
            old_logits = self.model(x)
            old_logp = F.log_softmax(old_logits, dim=-1)
            old_sel = old_logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1).detach()

        running_loss = 0.0
        rl_metrics: dict[str, float] = {}
        update_repeats = 1 + int(max(0, budget - 1) * max(0.0, 1.0 - float(rewards_t.mean().item())))
        for _ in range(update_repeats):
            new_logits = self.model(x)
            new_logp = F.log_softmax(new_logits, dim=-1)
            new_sel = new_logp.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
            rl_loss, rl_metrics = self._compute_rl_loss(
                logp_new=new_sel,
                logp_old=old_sel,
                rewards=rewards_t,
                mask=mask,
                diversity=diversity_t,
            )
            rl_loss_val, _ = self._optim_step(rl_loss)
            running_loss += float(rl_loss_val)
        rl_loss_val = running_loss / max(update_repeats, 1)
        self.feature_calls["feature_sigma_rl"] += 1

        reward_mean = float(raw_rewards_t.mean().item())
        reward_conf = float((raw_rewards_t > 0.5).float().mean().item())
        majority_pass = 1.0 if (sum(1 for x in rewards if x > 0.5) >= (len(rewards) // 2 + 1)) else 0.0

        replay_threshold = float(self.config.ttrl_confidence_replay_min) / max(1e-6, self._replay_bias)
        for seq, score in zip(seqs, rewards):
            if score >= replay_threshold:
                self._push_replay(seq, score=score)

        if self.config.ttrl_adaptive_budget:
            budget_min = max(1, int(self.config.ttrl_budget_min))
            budget_max = max(budget_min, int(self.config.ttrl_budget_max))
            if reward_mean < 0.5:
                self._ttrl_budget = min(budget_max, self._ttrl_budget + 1)
            elif reward_mean > 0.9 and reward_conf > 0.7:
                self._ttrl_budget = max(budget_min, self._ttrl_budget - 1)

        hidden_pass = self._last_ttrl_hidden_pass
        if self.hidden_eval_tasks and (self.global_step % max(1, int(self.config.integrity_hidden_eval_every)) == 0):
            hidden_pass = self._eval_task_set(self.hidden_eval_tasks, max_tasks=4)
        self._last_ttrl_hidden_pass = hidden_pass
        self._last_ttrl_public_pass = reward_mean

        self._last_ttrl_reward = reward_mean
        self._last_ttrl_confidence = reward_conf

        out = {
            "ttrl_domain_math": 1.0 if task.domain == "math" else 0.0,
            "ttrl_domain_code": 1.0 if task.domain == "code" else 0.0,
            "ttrl_reward_mean": reward_mean,
            "ttrl_confidence": reward_conf,
            "ttrl_majority_pass": majority_pass,
            "ttrl_online_loss": float(rl_loss_val),
            "ttrl_raw_reward_mean": float(raw_rewards_t.mean().item()),
            "ttrl_budget": float(self._ttrl_budget),
            "ttrl_buffer_size": float(len(self.replay)),
            "public_eval_pass": float(self._last_ttrl_public_pass),
            "hidden_eval_pass": float(self._last_ttrl_hidden_pass),
            "ttrl_diversity_mean": float(diversity_t.mean().item()),
            "ttrl_shaped_reward_mean": float(rewards_t.mean().item()),
            "ttrl_asym_verify_enabled": 1.0 if asym_enabled else 0.0,
            "ttrl_asym_candidate_pool": float(asym_pool),
            "ttrl_asym_verified_candidates": float(asym_verified),
            "ttrl_asym_discriminative_mean": float(asym_discriminative_mean),
            "ttrl_refined_candidates": float(refined_candidates),
            "verifier_cascade_enabled": 1.0 if bool(self.config.verifier_cascade_enabled) else 0.0,
            "sigma_rl_auto_mix_enabled": 1.0 if bool(self._rl_auto_mix_enabled) else 0.0,
            "feature_sigma_rl_enabled": 1.0,
        }
        for k, v in rl_metrics.items():
            out[k] = float(v)
        return out

    def _apply_policy(self, policy: dict[str, float]) -> None:
        if not policy:
            return
        self.feature_calls["feature_self_improver"] += 1
        self._last_policy = dict(policy)
        lr_mul = float(policy.get("policy_lr_mul", 1.0))
        new_factor = self._lr_factor * lr_mul
        self._set_optimizer_lr_factor(new_factor)

        interval_mul = float(policy.get("policy_ttrl_interval_mul", 1.0))
        interval = int(round(self._adaptive_ttrl_interval * interval_mul))
        interval = max(int(self.config.self_evolve_ttrl_interval_min), min(int(self.config.self_evolve_ttrl_interval_max), interval))
        self._adaptive_ttrl_interval = interval

        budget_bias = float(policy.get("policy_ttrl_budget_bias", 1.0))
        budget = int(round(self._ttrl_budget * budget_bias))
        budget = max(int(self.config.ttrl_budget_min), min(int(self.config.ttrl_budget_max), budget))
        self._ttrl_budget = budget

        replay_bias = float(policy.get("policy_replay_bias", 1.0))
        self._replay_bias = min(max(replay_bias, 0.5), 2.0)

    def _current_runtime_snapshot(self) -> RuntimeSnapshot:
        rw = RewardWeights.from_vector(self.reward_weights.to_vector())
        return RuntimeSnapshot(
            lr_factor=float(self._lr_factor),
            ttrl_interval=int(self._adaptive_ttrl_interval),
            ttrl_budget=int(self._ttrl_budget),
            replay_bias=float(self._replay_bias),
            reward_weights=rw,
        )

    def _apply_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._set_optimizer_lr_factor(float(snapshot.lr_factor))
        self._adaptive_ttrl_interval = max(1, int(snapshot.ttrl_interval))
        self._ttrl_budget = max(1, int(snapshot.ttrl_budget))
        self._replay_bias = float(min(max(snapshot.replay_bias, 0.5), 2.5))
        self.reward_weights = snapshot.reward_weights.clipped()

    def _uroboros_step(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.config.uroboros_enabled:
            return {}
        out = self.uroboros.step(
            step_idx=self.global_step,
            metrics=metrics,
            current_runtime=self._current_runtime_snapshot(),
            apply_runtime=self._apply_runtime_snapshot,
        )
        if out:
            self.feature_calls["feature_uroboros"] += 1
        rw_metrics = self.reward_weights.to_metrics()
        for k, v in rw_metrics.items():
            out[k] = float(v)
        return out

    def _self_evolve_step(self, *, loss_val: float, tokens_per_s: float, gpu_mem_reserved_gb: float) -> dict[str, float]:
        self.loss_history.append(float(loss_val))
        self.tps_history.append(float(tokens_per_s))
        self.mem_history.append(float(gpu_mem_reserved_gb))
        metrics: dict[str, float] = {
            "self_evolve_lr_factor": float(self._lr_factor),
            "self_evolve_ttrl_interval": float(self._adaptive_ttrl_interval),
            "self_evolve_ttrl_budget": float(self._ttrl_budget),
            "self_evolve_replay_bias": float(self._replay_bias),
        }

        policy = self.self_improver.propose(step=self.global_step)
        self._apply_policy(policy)
        for k, v in policy.items():
            metrics[k] = float(v)

        interval = max(1, int(self.config.self_evolve_interval))
        if self.global_step < max(4, int(self.config.self_evolve_history)):
            return metrics
        if (self.global_step % interval) != 0:
            return metrics

        half = max(2, len(self.loss_history) // 2)
        loss_old = sum(list(self.loss_history)[:half]) / float(half)
        loss_new = sum(list(self.loss_history)[-half:]) / float(half)
        loss_delta = loss_old - loss_new
        tps_old = sum(list(self.tps_history)[:half]) / float(half)
        tps_new = sum(list(self.tps_history)[-half:]) / float(half)
        tps_gain = (tps_new / max(tps_old, 1e-6)) - 1.0
        mem_avg = sum(self.mem_history) / float(max(len(self.mem_history), 1))

        lr_factor = self._lr_factor
        if loss_delta < -0.01:
            lr_factor *= float(self.config.self_evolve_lr_down)
            self._adaptive_ttrl_interval = max(int(self.config.self_evolve_ttrl_interval_min), self._adaptive_ttrl_interval - 1)
            self._ttrl_budget = min(max(1, int(self.config.ttrl_budget_max)), self._ttrl_budget + 1)
        elif loss_delta > 0.02 and tps_gain >= -0.05:
            lr_factor *= float(self.config.self_evolve_lr_up)
            self._adaptive_ttrl_interval = min(int(self.config.self_evolve_ttrl_interval_max), self._adaptive_ttrl_interval + 1)
            self._ttrl_budget = max(max(1, int(self.config.ttrl_budget_min)), self._ttrl_budget - 1)

        mem_target = float(self.config.self_evolve_target_reserved_gb)
        if mem_avg > mem_target:
            self._adaptive_ttrl_interval = min(int(self.config.self_evolve_ttrl_interval_max), self._adaptive_ttrl_interval + 1)
            lr_factor *= 0.98
        elif mem_avg < (0.80 * mem_target) and tps_gain < 0.0:
            lr_factor *= 1.01

        self._set_optimizer_lr_factor(lr_factor)

        speed_boost = float(UROBOROS_PATCHABLE_DEFAULTS.get("reward_speed_boost", 1.0))
        quality_boost = float(UROBOROS_PATCHABLE_DEFAULTS.get("reward_quality_boost", 1.0))
        verifier_boost = float(UROBOROS_PATCHABLE_DEFAULTS.get("reward_verifier_boost", 1.0))
        composite_reward = (
            (self.reward_weights.w_speed * speed_boost * tps_gain)
            + (self.reward_weights.w_loss_delta * quality_boost * loss_delta)
            + (self.reward_weights.w_verifier * verifier_boost * self._last_ttrl_reward)
            + (self.reward_weights.w_stability * (-max(0.0, mem_avg - float(self.config.self_evolve_target_reserved_gb))))
        )
        policy_update = self.self_improver.update(step=self.global_step, reward=composite_reward)
        self._apply_policy(policy_update)
        for k, v in policy_update.items():
            metrics[k] = float(v)

        metrics.update(
            {
                "self_evolve_loss_delta": float(loss_delta),
                "self_evolve_tps_gain": float(tps_gain),
                "self_evolve_mem_avg_gb": float(mem_avg),
                "self_evolve_composite_reward": float(composite_reward),
                "self_evolve_lr_factor": float(self._lr_factor),
                "self_evolve_ttrl_interval": float(self._adaptive_ttrl_interval),
                "self_evolve_ttrl_budget": float(self._ttrl_budget),
                "self_evolve_replay_bias": float(self._replay_bias),
            }
        )
        return metrics

    def _gpu_mem_metrics(self) -> dict[str, float]:
        return {
            "gpu_mem_alloc_gb": float(torch.cuda.memory_allocated(self.device) / (1024**3)),
            "gpu_mem_reserved_gb": float(torch.cuda.memory_reserved(self.device) / (1024**3)),
            "gpu_mem_peak_gb": float(torch.cuda.max_memory_allocated(self.device) / (1024**3)),
        }

    def _gpu_util_metrics(self) -> dict[str, float]:
        if not self._nvml_ready or self._gpu_handle is None or pynvml is None:
            return {"gpu_util_percent": float("nan"), "gpu_power_w": float("nan")}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            power_w = float(pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0)
            return {"gpu_util_percent": float(util.gpu), "gpu_power_w": power_w}
        except Exception:
            return {"gpu_util_percent": float("nan"), "gpu_power_w": float("nan")}

    def _update_emas(self, *, loss_val: float, tokens_per_s: float) -> dict[str, float]:
        if self._loss_ema is None:
            self._loss_ema = float(loss_val)
        else:
            self._loss_ema = (self._ema_beta * self._loss_ema) + ((1.0 - self._ema_beta) * float(loss_val))
        if self._tps_ema is None:
            self._tps_ema = float(tokens_per_s)
        else:
            self._tps_ema = (self._ema_beta * self._tps_ema) + ((1.0 - self._ema_beta) * float(tokens_per_s))
        return {"loss_ema": float(self._loss_ema), "tokens_per_s_ema": float(self._tps_ema)}

    def _emit_crash_report(self, exc: Exception) -> None:
        if not bool(self.config.crash_report_enabled):
            return
        errors_dir = self.output_dir / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        payload = {
            "time": ts,
            "step": int(self.global_step),
            "tokens_seen": int(self.tokens_seen),
            "device": str(self.device),
            "exception_type": exc.__class__.__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "last_metrics": self.last_metrics,
            "feature_calls": dict(self.feature_calls),
            "algorithmic_features": self.model.algorithmic_features(),
            "optimizer": self.optimizer.__class__.__name__,
        }
        path = errors_dir / f"crash_step_{int(self.global_step):08d}_{ts}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"[sigma-trainer] crash report written: {path}")

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    def _write_status_latest(self, metrics: dict[str, float], bottleneck_stage: str) -> None:
        status = {
            "step": int(self.global_step),
            "loss_ema": float(metrics.get("loss_ema", float("nan"))),
            "effective_tokens_per_s": float(metrics.get("effective_tokens_per_s", float("nan"))),
            "bottleneck_stage": str(bottleneck_stage),
            "bottleneck_share": float(metrics.get("perf_bottleneck_share", float("nan"))),
            "feature_sigma_rl_effective_calls": int(self.feature_calls["feature_sigma_rl"]),
            "feature_fractal_nas_effective_calls": int(self.feature_calls["feature_fractal_nas"]),
            "feature_uroboros_effective_calls": int(self.feature_calls["feature_uroboros"]),
            "feature_self_improver_effective_calls": int(self.feature_calls["feature_self_improver"]),
            "feature_hydra_v21_effective_calls": int(self.feature_calls["feature_hydra_v21"]),
        }
        if self._last_checkpoint_path:
            status["last_checkpoint_path"] = self._last_checkpoint_path
        if self._last_checkpoint_time:
            status["last_checkpoint_time"] = self._last_checkpoint_time
        with self._status_latest_path.open("w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=True, separators=(",", ":"))

    def _integrity_flags(self) -> dict[str, float]:
        c3o_effective = bool(self.config.c3o_credit_enabled and hasattr(self.optimizer, "set_credit_signal"))
        return {
            "feature_instant_enabled": 1.0,
            "feature_diff_mla_enabled": 1.0,
            "feature_sigma_rl_enabled": 1.0,
            "feature_verifier_enabled": 1.0,
            "feature_verifier_cascade_enabled": 1.0 if self.config.verifier_cascade_enabled else 0.0,
            "feature_fractal_nas_enabled": 1.0,
            "feature_self_improver_enabled": 1.0 if self.config.self_improver_enabled else 0.0,
            "feature_uroboros_enabled": 1.0 if self.config.uroboros_enabled else 0.0,
            "feature_causal_replay_enabled": 1.0 if self.config.causal_replay_enabled else 0.0,
            "feature_c3o_credit_enabled": 1.0 if c3o_effective else 0.0,
            "feature_asym_verify_enabled": 1.0 if self.config.ttrl_asym_verify_enabled else 0.0,
            "feature_rl_auto_mix_enabled": 1.0 if self._rl_auto_mix_enabled else 0.0,
            "feature_hydra_v21_enabled": 1.0 if self.config.hydra_enable else 0.0,
            "feature_instant_effective_calls": float(self.feature_calls["feature_instant"]),
            "feature_diff_mla_effective_calls": float(self.feature_calls["feature_diff_mla"]),
            "feature_sigma_rl_effective_calls": float(self.feature_calls["feature_sigma_rl"]),
            "feature_verifier_effective_calls": float(self.feature_calls["feature_verifier"]),
            "feature_verifier_cascade_effective_calls": float(self.feature_calls["feature_verifier_cascade"]),
            "feature_fractal_nas_effective_calls": float(self.feature_calls["feature_fractal_nas"]),
            "feature_self_improver_effective_calls": float(self.feature_calls["feature_self_improver"]),
            "feature_uroboros_effective_calls": float(self.feature_calls["feature_uroboros"]),
            "feature_causal_replay_effective_calls": float(self.feature_calls["feature_causal_replay"]),
            "feature_c3o_credit_effective_calls": float(self.feature_calls["feature_c3o_credit"]),
            "feature_asym_verify_effective_calls": float(self.feature_calls["feature_asym_verify"]),
            "feature_rl_auto_mix_effective_calls": float(self.feature_calls["feature_rl_auto_mix"]),
            "feature_hydra_v21_effective_calls": float(self.feature_calls["feature_hydra_v21"]),
            "meta_activation_step": float(self._meta_activation_step),
            "meta_features_ready": 1.0 if self._meta_features_ready() else 0.0,
            "unsloth_bootstrap_imported": 1.0 if self.config.unsloth_bootstrap_imported else 0.0,
            "unsloth_preloaded_trl": 1.0 if self.config.unsloth_preloaded_trl else 0.0,
            "unsloth_preloaded_transformers": 1.0 if self.config.unsloth_preloaded_transformers else 0.0,
            "unsloth_preloaded_peft": 1.0 if self.config.unsloth_preloaded_peft else 0.0,
            "unsloth_trl_patch_active": 1.0 if self.config.unsloth_trl_patch_active else 0.0,
            "unsloth_trl_patch_hits": float(self.config.unsloth_trl_patch_hits),
            "unsloth_trl_patch_targets": float(self.config.unsloth_trl_patch_targets),
        }

    def _perf_window_bounds(self) -> tuple[int, int]:
        warmup = max(0, int(self.config.perf_warmup_steps))
        measure = max(1, int(self.config.perf_measure_steps))
        base = int(self._perf_window_offset_step)
        return base + warmup + 1, base + warmup + measure

    def _maybe_start_profiler(self) -> None:
        if not bool(self.config.perf_profile_enable):
            return
        if self._perf_profile_started:
            return
        start_step, _ = self._perf_window_bounds()
        if self.global_step < start_step:
            return
        prof_steps = max(1, int(self.config.perf_profiler_steps))
        self._perf_profiler = torch_profiler.profile(
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self._perf_profiler.__enter__()
        self._perf_profile_started = True
        self._perf_profile_end_step = int(start_step + prof_steps - 1)

    def _maybe_step_profiler(self) -> None:
        if self._perf_profiler is None:
            return
        self._perf_profiler.step()
        if self.global_step < int(self._perf_profile_end_step):
            return
        self._finalize_profiler()

    def _finalize_profiler(self) -> None:
        if self._perf_profiler is None or self._perf_profile_finished:
            return
        prof = self._perf_profiler
        try:
            if bool(self.config.perf_sync_boundaries) and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            key_avgs = prof.key_averages()
            sort_key = "self_cuda_time_total"
            if str(self.config.perf_device_time_mode).strip().lower() == "cpu":
                sort_key = "self_cpu_time_total"
            table = key_avgs.table(sort_by=sort_key, row_limit=80)
            ops: list[dict[str, float | str]] = []
            total_time = 0.0
            moe_time = 0.0
            mode = str(self.config.perf_device_time_mode).strip().lower()
            if mode not in {"auto", "cuda", "device", "cpu"}:
                mode = "auto"
            for evt in key_avgs:
                name = str(getattr(evt, "key", ""))
                self_cuda = float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0)
                self_device = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
                cuda_total = float(getattr(evt, "cuda_time_total", 0.0) or 0.0)
                device_total = float(getattr(evt, "device_time_total", 0.0) or 0.0)
                self_cpu = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
                cpu_total = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)
                calls = int(getattr(evt, "count", 0) or 0)
                if mode == "cuda":
                    chosen_self = self_cuda
                    chosen_total = cuda_total
                    chosen_backend = "cuda"
                elif mode == "device":
                    chosen_self = self_device if self_device > 0.0 else self_cuda
                    chosen_total = device_total if device_total > 0.0 else cuda_total
                    chosen_backend = "device"
                elif mode == "cpu":
                    chosen_self = self_cpu
                    chosen_total = cpu_total
                    chosen_backend = "cpu"
                else:
                    if self_cuda > 0.0 or cuda_total > 0.0:
                        chosen_self = self_cuda
                        chosen_total = cuda_total
                        chosen_backend = "cuda"
                    elif self_device > 0.0 or device_total > 0.0:
                        chosen_self = self_device
                        chosen_total = device_total
                        chosen_backend = "device"
                    else:
                        chosen_self = self_cpu
                        chosen_total = cpu_total
                        chosen_backend = "cpu"
                total_time += max(chosen_self, 0.0)
                lname = name.lower()
                if any(tok in lname for tok in ("moe", "dispatch", "topk", "index_add", "routed_experts", "shared_experts")):
                    moe_time += max(chosen_self, 0.0)
                ops.append(
                    {
                        "name": name,
                        "calls": float(calls),
                        "self_time_us": chosen_self,
                        "total_time_us": chosen_total,
                        "time_backend": chosen_backend,
                        "self_cuda_us": self_cuda,
                        "self_device_us": self_device,
                        "cuda_total_us": cuda_total,
                        "device_total_us": device_total,
                        "self_cpu_us": self_cpu,
                        "cpu_total_us": cpu_total,
                    }
                )
            ops = sorted(ops, key=lambda x: float(x["self_time_us"]), reverse=True)[:80]
            if total_time <= 0.0:
                # Kineto can report zero device timings on some Windows builds; fallback to
                # trainer-stage timers so perf artifacts remain actionable.
                phase_train = float(self.last_metrics.get("phase_train_s", 0.0))
                moe_stage = float(self.last_metrics.get("moe_stage_router_s", 0.0)) + float(
                    self.last_metrics.get("moe_stage_topk_s", 0.0)
                )
                moe_stage += float(self.last_metrics.get("moe_stage_dispatch_s", 0.0)) + float(
                    self.last_metrics.get("moe_stage_expert_s", 0.0)
                )
                moe_stage += float(self.last_metrics.get("moe_stage_scatter_s", 0.0))
                total_time = max(phase_train * 1e6, 0.0)
                moe_time = max(moe_stage * 1e6, 0.0)
                ops = [
                    {
                        "name": "phase_train_timer",
                        "calls": 1.0,
                        "self_time_us": float(total_time),
                        "total_time_us": float(total_time),
                        "time_backend": "phase_timer_fallback",
                        "self_cuda_us": 0.0,
                        "self_device_us": 0.0,
                        "cuda_total_us": 0.0,
                        "device_total_us": 0.0,
                        "self_cpu_us": 0.0,
                        "cpu_total_us": 0.0,
                    },
                    {
                        "name": "moe_stage_timer",
                        "calls": 1.0,
                        "self_time_us": float(moe_time),
                        "total_time_us": float(moe_time),
                        "time_backend": "phase_timer_fallback",
                        "self_cuda_us": 0.0,
                        "self_device_us": 0.0,
                        "cuda_total_us": 0.0,
                        "device_total_us": 0.0,
                        "self_cpu_us": 0.0,
                        "cpu_total_us": 0.0,
                    },
                ]
            moe_share = float(moe_time / max(total_time, 1e-9))
            self._last_profile_moe_share = moe_share
            self._last_profile_time_backend = str(ops[0]["time_backend"]) if ops else "unknown"
            profile_json = {
                "total_self_time_us": float(total_time),
                "moe_self_time_us": float(moe_time),
                "moe_time_share": float(moe_share),
                "time_backend": self._last_profile_time_backend,
                "top_ops": ops,
            }
            with (self.perf_dir / "baseline_profiler.txt").open("w", encoding="utf-8") as f:
                f.write(table)
                f.write("\n")
            with (self.perf_dir / "baseline_profiler.json").open("w", encoding="utf-8") as f:
                json.dump(profile_json, f, ensure_ascii=True, indent=2)
            with (self.perf_dir / "window_10m_top_ops.json").open("w", encoding="utf-8") as f:
                json.dump({"time_backend": self._last_profile_time_backend, "top_ops": ops[:40]}, f, ensure_ascii=True, indent=2)
        finally:
            if bool(self.config.perf_sync_boundaries) and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            prof.__exit__(None, None, None)
            self._perf_profiler = None
            self._perf_profile_finished = True

    def _record_perf_window(
        self,
        *,
        core_step_time: float,
        effective_step_time: float,
        core_tokens_per_s: float,
        effective_tokens_per_s: float,
        phase_data_s: float,
        phase_train_s: float,
        phase_forward_s: float,
        phase_optim_s: float,
        phase_backward_s: float,
        phase_opt_step_s: float,
        phase_zero_grad_s: float,
        phase_ttrl_s: float,
        phase_meta_s: float,
    ) -> None:
        if not bool(self.config.perf_profile_enable):
            return
        start_step, end_step = self._perf_window_bounds()
        if self.global_step < start_step or self.global_step > end_step:
            return
        self._perf_measure_count += 1
        self._perf_sum_core_step_time += float(core_step_time)
        self._perf_sum_step_time += float(effective_step_time)
        self._perf_sum_core_tps += float(core_tokens_per_s)
        self._perf_sum_effective_tps += float(effective_tokens_per_s)
        self._perf_sum_phase_data += float(phase_data_s)
        self._perf_sum_phase_train += float(phase_train_s)
        self._perf_sum_phase_forward += float(phase_forward_s)
        self._perf_sum_phase_optim += float(phase_optim_s)
        self._perf_sum_phase_backward += float(phase_backward_s)
        self._perf_sum_phase_optstep += float(phase_opt_step_s)
        self._perf_sum_phase_zero_grad += float(phase_zero_grad_s)
        self._perf_sum_phase_ttrl += float(phase_ttrl_s)
        self._perf_sum_phase_meta += float(phase_meta_s)

    def _write_perf_summary(self, *, force: bool = False) -> None:
        if not bool(self.config.perf_profile_enable):
            return
        if self._perf_summary_written:
            return
        _, end_step = self._perf_window_bounds()
        if (not force) and self.global_step < end_step:
            return
        count = max(1, int(self._perf_measure_count))
        avg_step = float(self._perf_sum_step_time / count)
        avg_data = float(self._perf_sum_phase_data / count)
        avg_train = float(self._perf_sum_phase_train / count)
        avg_forward = float(self._perf_sum_phase_forward / count)
        avg_optim = float(self._perf_sum_phase_optim / count)
        avg_backward = float(self._perf_sum_phase_backward / count)
        avg_opt_step = float(self._perf_sum_phase_optstep / count)
        avg_zero_grad = float(self._perf_sum_phase_zero_grad / count)
        avg_ttrl = float(self._perf_sum_phase_ttrl / count)
        avg_meta = float(self._perf_sum_phase_meta / count)
        avg_other = float(max(avg_train - avg_forward - avg_optim, 0.0))
        shares = {
            "data_share_step": float(avg_data / max(avg_step, 1e-9)),
            "train_share_step": float(avg_train / max(avg_step, 1e-9)),
            "forward_share_train": float(avg_forward / max(avg_train, 1e-9)),
            "optim_share_train": float(avg_optim / max(avg_train, 1e-9)),
            "other_train_share_train": float(avg_other / max(avg_train, 1e-9)),
            "ttrl_share_step": float(avg_ttrl / max(avg_step, 1e-9)),
            "meta_share_step": float(avg_meta / max(avg_step, 1e-9)),
            "backward_share_optim": float(avg_backward / max(avg_optim, 1e-9)),
            "opt_step_share_optim": float(avg_opt_step / max(avg_optim, 1e-9)),
            "zero_grad_share_optim": float(avg_zero_grad / max(avg_optim, 1e-9)),
        }
        summary = {
            "steps_measured": float(self._perf_measure_count),
            "avg_step_time_s": avg_step,
            "avg_core_step_time_s": float(self._perf_sum_core_step_time / count),
            "avg_tokens_per_s": float(self._perf_sum_effective_tps / count),
            "avg_effective_tokens_per_s": float(self._perf_sum_effective_tps / count),
            "avg_core_tokens_per_s": float(self._perf_sum_core_tps / count),
            "avg_phase_data_s": avg_data,
            "avg_phase_train_s": avg_train,
            "avg_phase_forward_s": avg_forward,
            "avg_phase_optim_s": avg_optim,
            "avg_phase_backward_s": avg_backward,
            "avg_phase_opt_step_s": avg_opt_step,
            "avg_phase_zero_grad_s": avg_zero_grad,
            "avg_phase_train_other_s": avg_other,
            "avg_phase_ttrl_s": avg_ttrl,
            "avg_phase_meta_s": avg_meta,
            "moe_time_share_profiled": float(self._last_profile_moe_share),
            "profiler_time_backend": str(self._last_profile_time_backend),
            **shares,
        }
        with (self.perf_dir / "baseline_throughput.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        with (self.perf_dir / "window_10m_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        stage_shares = {
            "data": shares["data_share_step"],
            "train": shares["train_share_step"],
            "ttrl": shares["ttrl_share_step"],
            "meta": shares["meta_share_step"],
        }
        sorted_stage_shares = sorted(stage_shares.items(), key=lambda kv: kv[1], reverse=True)
        optim_subshares = {
            "backward": shares["backward_share_optim"],
            "opt_step": shares["opt_step_share_optim"],
            "zero_grad": shares["zero_grad_share_optim"],
            "optim_other": float(max(1.0 - shares["backward_share_optim"] - shares["opt_step_share_optim"] - shares["zero_grad_share_optim"], 0.0)),
        }
        sorted_optim_subshares = sorted(optim_subshares.items(), key=lambda kv: kv[1], reverse=True)
        with (self.perf_dir / "window_10m_bottlenecks.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage_shares": stage_shares,
                    "sorted_stage_shares": [{"stage": k, "share": float(v)} for k, v in sorted_stage_shares],
                    "dominant_stage": sorted_stage_shares[0][0] if sorted_stage_shares else "unknown",
                    "optim_subshares": optim_subshares,
                    "sorted_optim_subshares": [{"stage": k, "share": float(v)} for k, v in sorted_optim_subshares],
                    "dominant_optim_substage": sorted_optim_subshares[0][0] if sorted_optim_subshares else "unknown",
                },
                f,
                ensure_ascii=True,
                indent=2,
            )
        self._perf_summary_written = True

    def _maybe_compile_model(self) -> None:
        if not bool(self.config.torch_compile):
            return
        if self._model_compiled:
            return
        try:
            import torch._inductor.config as inductor_config  # type: ignore

            if hasattr(inductor_config, "disable_cpp_codegen"):
                setattr(inductor_config, "disable_cpp_codegen", True)
            if hasattr(inductor_config, "cpp_wrapper"):
                setattr(inductor_config, "cpp_wrapper", False)
            if hasattr(inductor_config, "cpp"):
                cpp_obj = getattr(inductor_config, "cpp")
                if hasattr(cpp_obj, "wrapper"):
                    setattr(cpp_obj, "wrapper", False)
        except Exception:
            pass
        backend = str(getattr(self.config, "torch_compile_backend", "inductor")).strip().lower()
        if not backend:
            backend = "inductor"
        mode = str(self.config.torch_compile_mode).strip().lower()
        compile_kwargs: dict[str, Any] = {
            "backend": backend,
            "dynamic": bool(self.config.torch_compile_dynamic),
            "fullgraph": bool(self.config.torch_compile_fullgraph),
        }
        if mode and mode != "none":
            compile_kwargs["mode"] = mode
        self.model = torch.compile(self.model, **compile_kwargs)
        self.model.train()
        self._model_compiled = True
        print(
            "[sigma-trainer] torch_compile enabled "
            f"backend={compile_kwargs.get('backend', 'inductor')} "
            f"mode={compile_kwargs.get('mode', 'default')} "
            f"dynamic={int(bool(compile_kwargs.get('dynamic', False)))} "
            f"fullgraph={int(bool(compile_kwargs.get('fullgraph', False)))}"
        )

    def train(self) -> None:
        self._maybe_compile_model()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._log_metrics(
            {
                "step": 0.0,
                "event_startup": 1.0,
                "startup_init_s": float(self._startup_init_s),
                "checkpoint_resume_step": float(self._checkpoint_resume_step),
                "checkpoint_quarantine_count": float(self._checkpoint_quarantine_count),
                "torch_compile_enabled": float(int(bool(self.config.torch_compile))),
                "torch_compile_active": float(int(bool(self._model_compiled))),
            }
        )
        try:
            while self.global_step < self.config.steps:
                self.global_step += 1
                start = time.perf_counter()
                try:
                    self._maybe_start_profiler()
                    phase_data_s = 0.0
                    phase_train_s = 0.0
                    phase_ttrl_s = 0.0
                    phase_hydra_s = 0.0
                    phase_self_evolve_s = 0.0
                    phase_fractal_s = 0.0
                    phase_uroboros_s = 0.0
                    phase_causal_s = 0.0
                    phase_credit_s = 0.0
                    phase_metrics_s = 0.0

                    t_phase = time.perf_counter()
                    batch = self._next_batch()
                    input_ids, labels, source_ids = self._prepare_batch(batch)
                    input_ids, labels = self._inject_replay(input_ids, labels)
                    self.tokens_seen += int(input_ids.numel())
                    phase_data_s = max(time.perf_counter() - t_phase, 0.0)

                    t_phase = time.perf_counter()
                    t_train_fw = time.perf_counter()
                    with self._autocast_ctx():
                        loss, real_loss, synth_loss = self._forward_loss(input_ids, labels, source_ids)
                    phase_forward_s = max(time.perf_counter() - t_train_fw, 0.0)
                    t_train_opt = time.perf_counter()
                    loss_val, optim_breakdown = self._optim_step(loss)
                    phase_opt_s = max(time.perf_counter() - t_train_opt, 0.0)
                    phase_train_s = max(time.perf_counter() - t_phase, 0.0)

                    self.feature_calls["feature_instant"] += 1
                    self.feature_calls["feature_diff_mla"] += 1

                    core_step_time = max(time.perf_counter() - start, 1e-6)
                    tokens_per_s = float(input_ids.numel() / core_step_time)
                    metrics: dict[str, float] = {
                        "step": float(self.global_step),
                        "loss": float(loss_val),
                        "train_loss_real_data": float(real_loss.item()) if torch.isfinite(real_loss) else float("nan"),
                        "train_loss_synth_data": float(synth_loss.item()) if torch.isfinite(synth_loss) else float("nan"),
                        "tokens_per_s": tokens_per_s,
                        "raw_tokens_per_s": tokens_per_s,
                        "step_time_s": float(core_step_time),
                        "tokens_seen_total": float(self.tokens_seen),
                        "cpu_util_percent": float(psutil.cpu_percent(interval=None)),
                        "ram_used_percent": float(psutil.virtual_memory().percent),
                        **self._gpu_mem_metrics(),
                        **self._gpu_util_metrics(),
                    }

                    if hasattr(self.optimizer, "metrics"):
                        opt_metrics = self.optimizer.metrics()
                        if isinstance(opt_metrics, dict):
                            for k, v in opt_metrics.items():
                                metrics[k] = float(v)

                    moe_metrics = self.model.collect_moe_metrics()
                    for k, v in moe_metrics.items():
                        metrics[k] = float(v)
                    attn_metrics = self.model.collect_attention_metrics()
                    for k, v in attn_metrics.items():
                        metrics[k] = float(v)
                    mamba_metrics = self.model.collect_mamba_metrics()
                    for k, v in mamba_metrics.items():
                        metrics[k] = float(v)
                    instant_metrics = collect_instant_metrics(self.model)
                    for k, v in instant_metrics.items():
                        metrics[k] = float(v)

                    if self._meta_features_ready() and (self.global_step % self._effective_ttrl_interval()) == 0:
                        t_phase = time.perf_counter()
                        ttrl_metrics = self._run_ttrl()
                        phase_ttrl_s = max(time.perf_counter() - t_phase, 0.0)
                        for k, v in ttrl_metrics.items():
                            metrics[k] = float(v)

                    if self.hydra_engine is not None and self._meta_features_ready():
                        t_phase = time.perf_counter()
                        hydra_metrics = self.hydra_engine.step(self.global_step)
                        phase_hydra_s = max(time.perf_counter() - t_phase, 0.0)
                        if hydra_metrics:
                            self.feature_calls["feature_hydra_v21"] += 1
                        for k, v in hydra_metrics.items():
                            metrics[k] = float(v)

                    phase_self_evolve_s = 0.0
                    se_metrics: dict[str, float] = {}
                    if self._meta_features_ready():
                        t_phase = time.perf_counter()
                        se_metrics = self._self_evolve_step(
                            loss_val=loss_val,
                            tokens_per_s=tokens_per_s,
                            gpu_mem_reserved_gb=float(metrics.get("gpu_mem_reserved_gb", 0.0)),
                        )
                        phase_self_evolve_s = max(time.perf_counter() - t_phase, 0.0)
                    for k, v in se_metrics.items():
                        metrics[k] = float(v)

                    phase_fractal_s = 0.0
                    fractal_metrics: dict[str, float] = {}
                    if self._meta_features_ready():
                        t_phase = time.perf_counter()
                        fractal_metrics = self.fractal.step(
                            step_idx=self.global_step,
                            model=self.model,
                            tokens_per_s=tokens_per_s,
                            loss_value=loss_val,
                            gpu_mem_reserved_gb=float(metrics.get("gpu_mem_reserved_gb", 0.0)),
                            instant_reconstruction_error=float(metrics.get("instant_reconstruction_error", 0.0)),
                        )
                        phase_fractal_s = max(time.perf_counter() - t_phase, 0.0)
                    if fractal_metrics:
                        self.feature_calls["feature_fractal_nas"] += 1
                    for k, v in fractal_metrics.items():
                        metrics[k] = float(v)

                    phase_uroboros_s = 0.0
                    uroboros_metrics: dict[str, float] = {}
                    if self._meta_features_ready():
                        t_phase = time.perf_counter()
                        uroboros_metrics = self._uroboros_step(metrics)
                        phase_uroboros_s = max(time.perf_counter() - t_phase, 0.0)
                    for k, v in uroboros_metrics.items():
                        metrics[k] = float(v)

                    t_phase = time.perf_counter()
                    causal_metrics = self.causal_replay.observe_step(
                        loss=float(loss_val),
                        verifier_pass=float(metrics.get("public_eval_pass", self._last_ttrl_public_pass)),
                        replay_used=bool(self._last_replay_injected),
                    )
                    phase_causal_s = max(time.perf_counter() - t_phase, 0.0)
                    if bool(self.config.causal_replay_enabled):
                        self.feature_calls["feature_causal_replay"] += 1
                    for k, v in causal_metrics.items():
                        metrics[k] = float(v)
                    t_phase = time.perf_counter()
                    credit_metrics = self.credit_estimator.observe(
                        loss=float(loss_val),
                        verifier_pass=float(metrics.get("hidden_eval_pass", self._last_ttrl_hidden_pass)),
                        tokens_per_s=float(tokens_per_s),
                    )
                    phase_credit_s = max(time.perf_counter() - t_phase, 0.0)
                    for k, v in credit_metrics.items():
                        metrics[k] = float(v)

                    t_phase = time.perf_counter()
                    for k, v in self._integrity_flags().items():
                        metrics[k] = float(v)
                    metrics["proof_mode"] = float(int(self.config.proof_mode))
                    metrics["integrity_guards_enabled"] = float(int(self.config.integrity_guards_enabled))
                    metrics["massive_improvement_enforced"] = float(int(self.config.massive_improvement_enforced))
                    metrics["forward_research_loop_enabled"] = float(int(self.config.forward_research_loop_enabled))
                    metrics["checkpoint_resume_step"] = float(self._checkpoint_resume_step)
                    metrics["checkpoint_quarantine_count"] = float(self._checkpoint_quarantine_count)
                    metrics["meta_activation_step"] = float(self._meta_activation_step)
                    metrics["meta_features_ready"] = 1.0 if self._meta_features_ready() else 0.0
                    phase_metrics_s = max(time.perf_counter() - t_phase, 0.0)

                    total_step_time = max(time.perf_counter() - start, 1e-6)
                    effective_tokens_per_s = float(input_ids.numel() / total_step_time)
                    phase_meta_s = float(
                        phase_hydra_s
                        + phase_self_evolve_s
                        + phase_fractal_s
                        + phase_uroboros_s
                        + phase_causal_s
                        + phase_credit_s
                        + phase_metrics_s
                    )
                    phase_map = {
                        "data": float(phase_data_s),
                        "train": float(phase_train_s),
                        "ttrl": float(phase_ttrl_s),
                        "meta": float(phase_meta_s),
                    }
                    bottleneck_stage = max(phase_map, key=phase_map.get)
                    stage_id = {"data": 1.0, "train": 2.0, "ttrl": 3.0, "meta": 4.0}.get(bottleneck_stage, 0.0)

                    metrics["core_step_time_s"] = float(core_step_time)
                    metrics["core_tokens_per_s"] = float(tokens_per_s)
                    metrics["effective_step_time_s"] = float(total_step_time)
                    metrics["effective_tokens_per_s"] = float(effective_tokens_per_s)
                    metrics["phase_data_s"] = float(phase_data_s)
                    metrics["phase_train_s"] = float(phase_train_s)
                    metrics["phase_forward_s"] = float(phase_forward_s)
                    metrics["phase_optim_s"] = float(phase_opt_s)
                    metrics["phase_backward_s"] = float(optim_breakdown.get("backward_s", 0.0))
                    metrics["phase_opt_step_s"] = float(optim_breakdown.get("opt_step_s", 0.0))
                    metrics["phase_zero_grad_s"] = float(optim_breakdown.get("zero_grad_s", 0.0))
                    metrics["phase_train_other_s"] = float(max(phase_train_s - phase_forward_s - phase_opt_s, 0.0))
                    metrics["phase_ttrl_s"] = float(phase_ttrl_s)
                    metrics["phase_hydra_s"] = float(phase_hydra_s)
                    metrics["phase_meta_s"] = float(phase_meta_s)
                    metrics["phase_self_evolve_s"] = float(phase_self_evolve_s)
                    metrics["phase_fractal_s"] = float(phase_fractal_s)
                    metrics["phase_uroboros_s"] = float(phase_uroboros_s)
                    metrics["phase_causal_s"] = float(phase_causal_s)
                    metrics["phase_credit_s"] = float(phase_credit_s)
                    metrics["phase_metrics_s"] = float(phase_metrics_s)
                    metrics["perf_bottleneck_stage_id"] = float(stage_id)
                    metrics["perf_bottleneck_share"] = float(phase_map[bottleneck_stage] / total_step_time)

                    ema_vals = self._update_emas(loss_val=loss_val, tokens_per_s=effective_tokens_per_s)
                    metrics["loss_ema"] = float(ema_vals["loss_ema"])
                    metrics["tokens_per_s_ema"] = float(ema_vals["tokens_per_s_ema"])
                    metrics["tokens_per_s"] = float(effective_tokens_per_s)
                    metrics["raw_tokens_per_s"] = float(effective_tokens_per_s)
                    metrics["step_time_s"] = float(total_step_time)
                    metrics["perf/tokens_per_s"] = float(metrics["tokens_per_s"])
                    metrics["perf/tokens_per_s_ema"] = float(metrics["tokens_per_s_ema"])
                    metrics["perf/step_time_s"] = float(metrics["step_time_s"])
                    metrics["perf/core_tokens_per_s"] = float(metrics["core_tokens_per_s"])
                    metrics["perf/effective_tokens_per_s"] = float(metrics["effective_tokens_per_s"])
                    metrics["perf/phase_data_s"] = float(metrics["phase_data_s"])
                    metrics["perf/phase_train_s"] = float(metrics["phase_train_s"])
                    metrics["perf/phase_forward_s"] = float(metrics["phase_forward_s"])
                    metrics["perf/phase_optim_s"] = float(metrics["phase_optim_s"])
                    metrics["perf/phase_backward_s"] = float(metrics["phase_backward_s"])
                    metrics["perf/phase_opt_step_s"] = float(metrics["phase_opt_step_s"])
                    metrics["perf/phase_zero_grad_s"] = float(metrics["phase_zero_grad_s"])
                    metrics["perf/phase_ttrl_s"] = float(metrics["phase_ttrl_s"])
                    metrics["perf/phase_hydra_s"] = float(metrics["phase_hydra_s"])
                    metrics["perf/phase_meta_s"] = float(metrics["phase_meta_s"])
                    metrics["perf/bottleneck_stage_id"] = float(metrics["perf_bottleneck_stage_id"])
                    metrics["perf/bottleneck_share"] = float(metrics["perf_bottleneck_share"])
                    metrics["perf/cpu_util_percent"] = float(metrics["cpu_util_percent"])
                    metrics["perf/ram_used_percent"] = float(metrics["ram_used_percent"])
                    metrics["perf/gpu_util_percent"] = float(metrics["gpu_util_percent"])
                    metrics["perf/gpu_mem_alloc_gb"] = float(metrics["gpu_mem_alloc_gb"])
                    metrics["loss/train"] = float(metrics["loss"])
                    metrics["loss/train_ema"] = float(metrics["loss_ema"])
                    metrics["startup_init_s"] = float(self._startup_init_s)
                    if self._first_batch_wall is not None:
                        metrics["time_to_first_batch_s"] = float(max(self._first_batch_wall - self._startup_wall_t0, 0.0))
                    if self._first_step_wall is None:
                        self._first_step_wall = time.perf_counter()
                    if self._first_step_wall is not None:
                        metrics["time_to_first_step_s"] = float(max(self._first_step_wall - self._startup_wall_t0, 0.0))

                    self._record_perf_window(
                        core_step_time=float(core_step_time),
                        effective_step_time=float(total_step_time),
                        core_tokens_per_s=float(tokens_per_s),
                        effective_tokens_per_s=float(effective_tokens_per_s),
                        phase_data_s=float(phase_data_s),
                        phase_train_s=float(phase_train_s),
                        phase_forward_s=float(phase_forward_s),
                        phase_optim_s=float(phase_opt_s),
                        phase_backward_s=float(metrics["phase_backward_s"]),
                        phase_opt_step_s=float(metrics["phase_opt_step_s"]),
                        phase_zero_grad_s=float(metrics["phase_zero_grad_s"]),
                        phase_ttrl_s=float(phase_ttrl_s),
                        phase_meta_s=float(phase_meta_s),
                    )
                    self._maybe_step_profiler()
                    self._write_perf_summary(force=False)

                    self.last_metrics = metrics
                    self._log_metrics(metrics)

                    if (self.global_step % max(1, self.config.log_interval)) == 0:
                        meta_ready = self.global_step > int(self._meta_activation_step)
                        ttrl_interval = max(1, int(self._effective_ttrl_interval()))
                        meta_gated = not meta_ready
                        key_features = (
                            "feature_sigma_rl",
                            "feature_fractal_nas",
                            "feature_uroboros",
                            "feature_self_improver",
                            "feature_hydra_v21",
                        )
                        delta_calls = {
                            key: int(self.feature_calls[key] - self._last_log_feature_calls.get(key, 0))
                            for key in key_features
                        }
                        self._last_log_feature_calls = {key: int(self.feature_calls[key]) for key in key_features}
                        print(
                            f"step={self.global_step} "
                            f"loss={loss_val:.4f} "
                            f"loss_ema={metrics['loss_ema']:.4f} "
                            f"eff_tps={metrics['effective_tokens_per_s']:.2f} "
                            f"core_tps={metrics['core_tokens_per_s']:.2f} "
                            f"tps_ema={metrics['tokens_per_s_ema']:.2f} "
                            f"bnk_id={metrics['perf_bottleneck_stage_id']:.0f} "
                            f"meta_ready={int(meta_ready)} "
                            f"ttrl_int={ttrl_interval} "
                            f"meta_gated={int(meta_gated)} "
                            f"d_sigma={delta_calls['feature_sigma_rl']} "
                            f"d_frac={delta_calls['feature_fractal_nas']} "
                            f"d_uro={delta_calls['feature_uroboros']} "
                            f"d_self={delta_calls['feature_self_improver']} "
                            f"d_hydra={delta_calls['feature_hydra_v21']} "
                            f"gpu_util={metrics.get('gpu_util_percent', float('nan')):.1f}% "
                            f"gpu_mem={metrics['gpu_mem_alloc_gb']:.2f}GB "
                            f"hidden_eval={metrics.get('hidden_eval_pass', self._last_ttrl_hidden_pass):.3f} "
                            f"diff_lam={metrics.get('diff_attn_lambda_mean', 0.0):.3f}"
                        )
                        self._write_status_latest(metrics=metrics, bottleneck_stage=bottleneck_stage)

                    if (self.global_step % max(1, self.config.save_interval)) == 0:
                        self._save_checkpoint(self.global_step)
                except Exception as step_exc:
                    self._emit_crash_report(step_exc)
                    raise

            if (self.global_step % max(1, self.config.save_interval)) != 0:
                self._save_checkpoint(self.global_step)

            required_calls = dict(self.feature_calls)
            if self.global_step <= int(self._meta_activation_step):
                required_calls.pop("feature_sigma_rl", None)
                required_calls.pop("feature_verifier", None)
                required_calls.pop("feature_verifier_cascade", None)
                required_calls.pop("feature_asym_verify", None)
                required_calls.pop("feature_rl_auto_mix", None)
                required_calls.pop("feature_fractal_nas", None)
                required_calls.pop("feature_self_improver", None)
                required_calls.pop("feature_uroboros", None)
                required_calls.pop("feature_hydra_v21", None)
            if self.global_step < (int(self._meta_activation_step) + max(1, int(self.config.fractal_interval_steps))):
                required_calls.pop("feature_fractal_nas", None)
            if self.global_step < (int(self._meta_activation_step) + max(1, int(self.config.uroboros_interval))):
                required_calls.pop("feature_uroboros", None)
            if not self.config.self_improver_enabled:
                required_calls.pop("feature_self_improver", None)
            elif self.global_step < (int(self._meta_activation_step) + max(1, int(self.config.self_improver_interval))):
                required_calls.pop("feature_self_improver", None)
            if not self.config.uroboros_enabled:
                required_calls.pop("feature_uroboros", None)
            if not self.config.causal_replay_enabled:
                required_calls.pop("feature_causal_replay", None)
            ttrl_interval = max(1, int(self._effective_ttrl_interval()))
            if self.global_step < (int(self._meta_activation_step) + ttrl_interval):
                required_calls.pop("feature_sigma_rl", None)
                required_calls.pop("feature_verifier", None)
                required_calls.pop("feature_verifier_cascade", None)
                required_calls.pop("feature_asym_verify", None)
                required_calls.pop("feature_rl_auto_mix", None)
            if not self.config.ttrl_asym_verify_enabled:
                required_calls.pop("feature_asym_verify", None)
            if not self.config.hydra_enable:
                required_calls.pop("feature_hydra_v21", None)
            elif self.global_step < (int(self._meta_activation_step) + max(1, int(self.config.hydra_update_interval))):
                required_calls.pop("feature_hydra_v21", None)
            if not self.config.verifier_cascade_enabled:
                required_calls.pop("feature_verifier_cascade", None)
            if not self.config.c3o_credit_enabled:
                required_calls.pop("feature_c3o_credit", None)
            elif not hasattr(self.optimizer, "set_credit_signal"):
                required_calls.pop("feature_c3o_credit", None)
            if not self._rl_auto_mix_enabled:
                required_calls.pop("feature_rl_auto_mix", None)
            if not (self.config.verifier_math_enabled or self.config.verifier_code_enabled):
                required_calls.pop("feature_verifier", None)
                required_calls.pop("feature_verifier_cascade", None)
                required_calls.pop("feature_sigma_rl", None)
                required_calls.pop("feature_rl_auto_mix", None)
            check_feature_effective_calls(
                proof_mode=bool(self.config.proof_mode),
                strict_mode=bool(self.config.strict_feature_usage),
                feature_calls=required_calls,
            )
        finally:
            self._write_perf_summary(force=True)
            self._finalize_profiler()
            if self._nvml_ready and pynvml is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
