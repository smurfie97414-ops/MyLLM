from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import heapq
import json
import math
import os
from pathlib import Path
import random
import re
import time
import traceback
from typing import Callable

import psutil
import torch
import torch.nn.functional as F

_CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps: int = 1000
    grad_accum_steps: int = 1
    mixed_precision: bool = True
    precision: str = "auto"  # auto | bf16 | fp16
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    detect_anomaly: bool = False
    skip_non_finite_loss: bool = True
    oom_recovery: bool = True
    max_oom_retries_per_step: int = 3
    dataloader_max_retries: int = 5
    output_dir: str = "runs/default"
    metrics_file: str = "metrics.jsonl"
    save_error_batch: bool = True
    empty_cache_interval: int = 100
    device_prefetch: bool = True
    save_interval: int = 100
    max_checkpoints: int = 5
    resume_from_latest: bool = True
    checkpoint_dir: str = "checkpoints"
    strict_fail_fast: bool = True
    gpu_metrics_sync_interval: int = 20
    seq_len_warmup_steps: int = 0
    seq_len_warmup_start: int = 128
    qk_clip_interval: int = 1
    self_accelerate: bool = True
    self_accel_warmup_steps: int = 10
    self_accel_interval: int = 2
    self_accel_lr_up: float = 0.03
    self_accel_lr_down: float = 0.90
    self_accel_lr_min_factor: float = 0.50
    self_accel_lr_max_factor: float = 1.80
    self_accel_grad_ratio_low: float = 0.005
    self_accel_grad_ratio_high: float = 0.03
    self_accel_opt_bottleneck_threshold: float = 0.55
    self_accel_loss_bad_threshold: float = -0.01
    self_accel_max_orth_end: int = 8
    self_accel_moe_imbalance_high: float = 0.25
    self_accel_moe_imbalance_low: float = 0.12
    self_evolve_bandit: bool = True
    self_evolve_bandit_interval: int = 2
    self_evolve_bandit_ucb_c: float = 1.5
    self_evolve_mutation_std: float = 0.06
    self_evolve_source_reweight: bool = True
    self_evolve_source_decay: float = 0.08
    self_evolve_source_weight_min: float = 0.5
    self_evolve_source_weight_max: float = 2.0
    self_evolve_source_loss_ema_beta: float = 0.97
    self_evolve_mutation_adapt: bool = True
    self_evolve_mutation_min: float = 0.02
    self_evolve_mutation_max: float = 0.25
    self_evolve_reward_ema_beta: float = 0.90
    self_evolve_teon_adapt: bool = True
    self_evolve_teon_min: float = 0.05
    self_evolve_teon_max: float = 0.35
    self_evolve_teon_step: float = 0.015
    self_evolve_hard_replay: bool = True
    self_evolve_hard_replay_capacity: int = 192
    self_evolve_hard_replay_interval: int = 3
    self_evolve_hard_replay_start_step: int = 32
    self_evolve_hard_replay_replace_frac: float = 0.15
    self_evolve_hard_replay_topk_per_batch: int = 2
    self_evolve_meta_es: bool = True
    self_evolve_meta_interval: int = 2
    self_evolve_meta_sigma: float = 0.10
    self_evolve_meta_lr: float = 0.12
    self_evolve_meta_reward_beta: float = 0.92
    self_evolve_meta_source_temp_min: float = 0.65
    self_evolve_meta_source_temp_max: float = 1.45
    self_evolve_meta_underseen_boost: float = 0.30
    self_evolve_meta_gain_penalty: float = 0.20


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader,
        config: TrainConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir = self.output_dir / "errors"
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / config.metrics_file
        self.checkpoints_dir = Path(config.checkpoint_dir)
        if not self.checkpoints_dir.is_absolute():
            self.checkpoints_dir = Path.cwd() / self.checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        self.model.to(self.device)
        self._compiled_active = False
        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=self.config.compile_mode)
                self._compiled_active = True
            except Exception as exc:
                if self.config.strict_fail_fast:
                    raise RuntimeError(f"torch.compile failed in strict mode: {exc}") from exc
                print(f"[{_timestamp()}] [warn] torch.compile failed; continuing uncompiled. error={exc}")

        self.autocast_dtype = self._resolve_autocast_dtype()
        scaler_enabled = self.device.type == "cuda" and self.autocast_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self._prefetch_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._last_gpu_metrics: dict[str, float] = {}
        self._last_aux_metrics: dict[str, float] = {}
        self._tokens_seen_total = 0
        self._sa_history: list[dict[str, float]] = []
        self._sa_base_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        self._sa_lr_factor = 1.0
        self._sa_orth_every_end = int(self.optimizer.param_groups[0].get("orthogonalize_every_end", 1))
        self._sa_orth_floor = max(1, self._sa_orth_every_end)
        self._sa_adaptive_orth_tol = float(self.optimizer.param_groups[0].get("adaptive_orth_tol", 0.05))
        self._sa_teon_coupling = float(self.optimizer.param_groups[0].get("teon_coupling", 0.15))
        self._source_name_to_id = {
            "unknown": 0,
            "fineweb": 1,
            "cosmopedia": 2,
            "commoncrawl": 3,
            "synthetic": 4,
        }
        self._source_loss_weights = torch.ones(5, dtype=torch.float32, device=self.device)
        self._source_loss_ema = torch.zeros(5, dtype=torch.float32, device=self.device)
        self._source_seen_ema = torch.zeros(5, dtype=torch.float32, device=self.device)
        self._source_gain_ema = torch.zeros(5, dtype=torch.float32, device=self.device)
        self._bandit_arms = [
            {
                "name": "stabilize",
                "lr_mul": 0.95,
                "orth_delta": -1,
                "adaptive_tol_delta": -0.003,
                "teon_delta": 0.01,
                "source": "fineweb",
            },
            {
                "name": "balanced",
                "lr_mul": 1.00,
                "orth_delta": 0,
                "adaptive_tol_delta": 0.000,
                "teon_delta": 0.00,
                "source": "unknown",
            },
            {
                "name": "speed",
                "lr_mul": 1.03,
                "orth_delta": 1,
                "adaptive_tol_delta": 0.004,
                "teon_delta": -0.01,
                "source": "commoncrawl",
            },
            {
                "name": "reasoning",
                "lr_mul": 1.01,
                "orth_delta": 0,
                "adaptive_tol_delta": -0.001,
                "teon_delta": 0.005,
                "source": "cosmopedia",
            },
            {
                "name": "moe_balance",
                "lr_mul": 0.98,
                "orth_delta": 0,
                "adaptive_tol_delta": -0.002,
                "teon_delta": 0.008,
                "source": "fineweb",
            },
            {
                "name": "data_explore",
                "lr_mul": 1.00,
                "orth_delta": 1,
                "adaptive_tol_delta": 0.002,
                "teon_delta": -0.004,
                "source": "synthetic",
            },
        ]
        self._bandit_counts = [0 for _ in self._bandit_arms]
        self._bandit_value_ema = [0.0 for _ in self._bandit_arms]
        self._bandit_last_arm = 1
        self._bandit_steps = 0
        self._bandit_last_reward = 0.0
        self._self_evolve_reward_ema = 0.0
        self._self_evolve_mutation_scale = float(self.config.self_evolve_mutation_std)
        self._hard_replay_heap: list[tuple[float, int, dict[str, torch.Tensor]]] = []
        self._hard_replay_counter = 0
        self._last_hard_replay_injected = 0
        self._last_hard_replay_mean_loss = 0.0
        self._meta_theta = torch.zeros(5, dtype=torch.float32)
        self._meta_best_theta = self._meta_theta.clone()
        self._meta_best_reward = -1e30
        self._meta_eps = torch.zeros_like(self._meta_theta)
        self._meta_phase = 0  # 0=idle/base, 1=plus probe, -1=minus probe
        self._meta_plus_reward = 0.0
        self._meta_minus_reward = 0.0
        self._meta_reward_ema = 0.0
        self._meta_lr_multiplier = 1.0
        self._meta_source_temp = 1.0
        self._meta_source_decay_mult = 1.0
        self._meta_replay_mult = 1.0
        self._meta_router_reg_mult = 1.0
        self._apply_meta_theta(self._meta_theta)
        self._warn_if_device_mismatch()

    def _is_compile_backend_error(self, exc: BaseException) -> bool:
        msg = str(exc)
        patterns = [
            "Cannot find a working triton installation",
            "TritonMissing",
            "BackendCompilerFailed",
            "torch._inductor",
        ]
        return any(p in msg for p in patterns)

    def _is_checkpoint_error(self, exc: BaseException) -> bool:
        msg = str(exc)
        return "torch.utils.checkpoint.CheckpointError" in msg or "Recomputed values for the following tensors" in msg

    def _disable_compile(self, reason: str) -> None:
        if not self._compiled_active:
            return
        maybe_orig = getattr(self.model, "_orig_mod", None)
        if maybe_orig is not None:
            self.model = maybe_orig
            self.model.to(self.device)
        self._compiled_active = False
        print(f"[{_timestamp()}] [warn] torch.compile disabled at runtime: {reason}")

    def _disable_activation_checkpointing(self, reason: str) -> None:
        model_ref = self._model_ref()
        config = getattr(model_ref, "config", None)
        if config is not None and getattr(config, "activation_checkpointing", False):
            config.activation_checkpointing = False
            print(f"[{_timestamp()}] [warn] activation checkpointing disabled at runtime: {reason}")

    def _model_ref(self) -> torch.nn.Module:
        return getattr(self.model, "_orig_mod", self.model)

    def _warn_if_device_mismatch(self) -> None:
        if self.device.type == "cuda" and not torch.cuda.is_available():
            if self.config.strict_fail_fast:
                raise RuntimeError("Strict mode requested CUDA device, but CUDA is unavailable.")
            print(f"[{_timestamp()}] [warn] Requested CUDA device but CUDA is unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            return

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            print(f"[{_timestamp()}] device={self.device} gpu='{gpu_name}'")
        else:
            print(f"[{_timestamp()}] device={self.device} (CPU)")

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoints_dir / f"checkpoint_step_{step:08d}.pt"

    def _checkpoint_step(self, path: Path) -> int:
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if not match:
            return -1
        return int(match.group(1))

    def _checkpoint_candidates(self) -> list[Path]:
        return sorted(
            self.checkpoints_dir.glob("checkpoint_step_*.pt"),
            key=self._checkpoint_step,
        )

    def _latest_checkpoint(self) -> Path | None:
        checkpoints = self._checkpoint_candidates()
        if not checkpoints:
            return None
        return checkpoints[-1]

    def _quarantine_checkpoint(self, path: Path, reason: BaseException) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        quarantine = path.with_name(f"{path.name}.corrupt_{stamp}_{os.getpid()}")
        try:
            path.replace(quarantine)
            print(
                f"[{_timestamp()}] [warn] checkpoint marked corrupted and quarantined: "
                f"{path} -> {quarantine}. reason={reason}"
            )
        except Exception as exc:
            print(
                f"[{_timestamp()}] [warn] checkpoint appears corrupted but could not be quarantined: "
                f"{path}. reason={reason} quarantine_error={exc}"
            )

    def _validate_checkpoint_payload(self, payload: object, path: Path) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"checkpoint payload at {path} is not a dict")
        for key in ("model", "optimizer", "step"):
            if key not in payload:
                raise RuntimeError(f"checkpoint payload at {path} missing required key '{key}'")
        if not isinstance(payload["model"], dict):
            raise RuntimeError(f"checkpoint payload at {path} has invalid 'model' state")
        if not isinstance(payload["optimizer"], dict):
            raise RuntimeError(f"checkpoint payload at {path} has invalid 'optimizer' state")
        return payload

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _trainer_meta_state(self) -> dict[str, object]:
        return {
            "sa_lr_factor": float(self._sa_lr_factor),
            "sa_orth_every_end": int(self._sa_orth_every_end),
            "sa_orth_floor": int(self._sa_orth_floor),
            "sa_adaptive_orth_tol": float(self._sa_adaptive_orth_tol),
            "sa_teon_coupling": float(self._sa_teon_coupling),
            "source_loss_weights": self._source_loss_weights.detach().float().cpu(),
            "source_loss_ema": self._source_loss_ema.detach().float().cpu(),
            "source_seen_ema": self._source_seen_ema.detach().float().cpu(),
            "source_gain_ema": self._source_gain_ema.detach().float().cpu(),
            "bandit_counts": [int(x) for x in self._bandit_counts],
            "bandit_value_ema": [float(x) for x in self._bandit_value_ema],
            "bandit_last_arm": int(self._bandit_last_arm),
            "bandit_steps": int(self._bandit_steps),
            "bandit_last_reward": float(self._bandit_last_reward),
            "sa_base_lr": float(self._sa_base_lr),
            "self_evolve_reward_ema": float(self._self_evolve_reward_ema),
            "self_evolve_mutation_scale": float(self._self_evolve_mutation_scale),
            "meta_theta": self._meta_theta.detach().float().cpu(),
            "meta_best_theta": self._meta_best_theta.detach().float().cpu(),
            "meta_best_reward": float(self._meta_best_reward),
            "meta_eps": self._meta_eps.detach().float().cpu(),
            "meta_phase": int(self._meta_phase),
            "meta_plus_reward": float(self._meta_plus_reward),
            "meta_minus_reward": float(self._meta_minus_reward),
            "meta_reward_ema": float(self._meta_reward_ema),
        }

    def _load_trainer_meta_state(self, payload: dict[str, object]) -> None:
        trainer_meta = payload.get("trainer_meta")
        if not isinstance(trainer_meta, dict):
            return
        try:
            self._sa_lr_factor = float(trainer_meta.get("sa_lr_factor", self._sa_lr_factor))
            self._sa_orth_every_end = int(trainer_meta.get("sa_orth_every_end", self._sa_orth_every_end))
            self._sa_orth_floor = int(trainer_meta.get("sa_orth_floor", self._sa_orth_floor))
            self._sa_adaptive_orth_tol = float(
                trainer_meta.get("sa_adaptive_orth_tol", self._sa_adaptive_orth_tol)
            )
            self._sa_teon_coupling = float(trainer_meta.get("sa_teon_coupling", self._sa_teon_coupling))
            self._sa_base_lr = float(trainer_meta.get("sa_base_lr", self._sa_base_lr))
            src_w = trainer_meta.get("source_loss_weights")
            if isinstance(src_w, torch.Tensor) and src_w.numel() == self._source_loss_weights.numel():
                self._source_loss_weights.copy_(src_w.to(self.device, dtype=self._source_loss_weights.dtype))
                self._normalize_source_weights()
            src_ema = trainer_meta.get("source_loss_ema")
            if isinstance(src_ema, torch.Tensor) and src_ema.numel() == self._source_loss_ema.numel():
                self._source_loss_ema.copy_(src_ema.to(self.device, dtype=self._source_loss_ema.dtype))
            src_seen = trainer_meta.get("source_seen_ema")
            if isinstance(src_seen, torch.Tensor) and src_seen.numel() == self._source_seen_ema.numel():
                self._source_seen_ema.copy_(src_seen.to(self.device, dtype=self._source_seen_ema.dtype))
            src_gain = trainer_meta.get("source_gain_ema")
            if isinstance(src_gain, torch.Tensor) and src_gain.numel() == self._source_gain_ema.numel():
                self._source_gain_ema.copy_(src_gain.to(self.device, dtype=self._source_gain_ema.dtype))
            counts = trainer_meta.get("bandit_counts")
            values = trainer_meta.get("bandit_value_ema")
            if isinstance(counts, list) and len(counts) == len(self._bandit_counts):
                self._bandit_counts = [int(x) for x in counts]
            if isinstance(values, list) and len(values) == len(self._bandit_value_ema):
                self._bandit_value_ema = [float(x) for x in values]
            self._bandit_last_arm = int(trainer_meta.get("bandit_last_arm", self._bandit_last_arm))
            self._bandit_steps = int(trainer_meta.get("bandit_steps", self._bandit_steps))
            self._bandit_last_reward = float(trainer_meta.get("bandit_last_reward", self._bandit_last_reward))
            self._self_evolve_reward_ema = float(
                trainer_meta.get("self_evolve_reward_ema", self._self_evolve_reward_ema)
            )
            self._self_evolve_mutation_scale = float(
                trainer_meta.get("self_evolve_mutation_scale", self._self_evolve_mutation_scale)
            )
            meta_theta = trainer_meta.get("meta_theta")
            if isinstance(meta_theta, torch.Tensor) and meta_theta.numel() == self._meta_theta.numel():
                self._meta_theta.copy_(meta_theta.to(dtype=self._meta_theta.dtype, device=self._meta_theta.device))
            meta_best_theta = trainer_meta.get("meta_best_theta")
            if isinstance(meta_best_theta, torch.Tensor) and meta_best_theta.numel() == self._meta_best_theta.numel():
                self._meta_best_theta.copy_(
                    meta_best_theta.to(dtype=self._meta_best_theta.dtype, device=self._meta_best_theta.device)
                )
            meta_eps = trainer_meta.get("meta_eps")
            if isinstance(meta_eps, torch.Tensor) and meta_eps.numel() == self._meta_eps.numel():
                self._meta_eps.copy_(meta_eps.to(dtype=self._meta_eps.dtype, device=self._meta_eps.device))
            self._meta_best_reward = float(trainer_meta.get("meta_best_reward", self._meta_best_reward))
            self._meta_phase = int(trainer_meta.get("meta_phase", self._meta_phase))
            self._meta_plus_reward = float(trainer_meta.get("meta_plus_reward", self._meta_plus_reward))
            self._meta_minus_reward = float(trainer_meta.get("meta_minus_reward", self._meta_minus_reward))
            self._meta_reward_ema = float(trainer_meta.get("meta_reward_ema", self._meta_reward_ema))
            self._apply_meta_theta(self._meta_theta)
            self._set_lr_factor(self._sa_lr_factor)
            self._set_muon_runtime_knobs(
                orth_every_end=self._sa_orth_every_end,
                adaptive_tol=self._sa_adaptive_orth_tol,
            )
            self._set_teon_coupling(self._sa_teon_coupling)
        except Exception as exc:
            print(f"[{_timestamp()}] [warn] failed to restore trainer_meta state: {exc}")

    def _save_checkpoint(self, step: int) -> None:
        if self.config.save_interval <= 0:
            return
        payload = {
            "step": step,
            "model": self._model_ref().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "trainer_meta": self._trainer_meta_state(),
            "time": _timestamp(),
        }
        path = self._checkpoint_path(step)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        # Keep only most recent checkpoints.
        checkpoints = self._checkpoint_candidates()
        while len(checkpoints) > self.config.max_checkpoints:
            old = checkpoints.pop(0)
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"[{_timestamp()}] [ckpt] saved {path}")

    def _resume_if_available(self) -> int:
        if not self.config.resume_from_latest:
            return 0
        checkpoints = self._checkpoint_candidates()
        if not checkpoints:
            return 0

        model_ref = self._model_ref()
        ref_state = model_ref.state_dict()
        for candidate in reversed(checkpoints):
            try:
                raw_payload = torch.load(candidate, map_location="cpu")
                payload = self._validate_checkpoint_payload(raw_payload, candidate)
                model_state = payload["model"]
                assert isinstance(model_state, dict)
                model_keys = set(model_state.keys())
                ref_keys = set(ref_state.keys())
                if model_keys != ref_keys:
                    missing = sorted(ref_keys - model_keys)
                    extra = sorted(model_keys - ref_keys)
                    raise RuntimeError(
                        f"state_dict key mismatch for {candidate}: missing={missing[:5]} extra={extra[:5]}"
                    )
                for key, ref_tensor in ref_state.items():
                    ckpt_tensor = model_state[key]
                    if not isinstance(ckpt_tensor, torch.Tensor):
                        raise RuntimeError(f"state_dict[{key}] is not a tensor")
                    if ckpt_tensor.shape != ref_tensor.shape:
                        raise RuntimeError(
                            f"state_dict[{key}] shape mismatch: ckpt={tuple(ckpt_tensor.shape)} "
                            f"model={tuple(ref_tensor.shape)}"
                        )

                model_ref.load_state_dict(model_state, strict=True)
                optimizer_state = payload["optimizer"]
                assert isinstance(optimizer_state, dict)
                self.optimizer.load_state_dict(optimizer_state)
                self._move_optimizer_state_to_device()
                if self.scaler.is_enabled() and payload.get("scaler") is not None:
                    self.scaler.load_state_dict(payload["scaler"])
                self._load_trainer_meta_state(payload)
                step = int(payload.get("step", 0))
                print(f"[{_timestamp()}] [ckpt] resumed from {candidate} (step={step})")
                return step
            except Exception as exc:
                self._quarantine_checkpoint(candidate, exc)
                continue

        print(f"[{_timestamp()}] [warn] no valid checkpoint found; starting from step 0.")
        return 0

    def _resolve_autocast_dtype(self):
        if not self.config.mixed_precision:
            return None
        if self.device.type != "cuda":
            return None
        precision = self.config.precision.lower()
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision != "auto":
            raise ValueError(f"Unsupported precision mode: {self.config.precision}")
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=True)

    def _next_batch(self, data_iter):
        retries = 0
        while True:
            try:
                return next(data_iter), data_iter
            except StopIteration:
                data_iter = iter(self.dataloader)
            except Exception as exc:
                if self.config.strict_fail_fast:
                    raise RuntimeError(f"Data loader failed in strict mode: {exc}") from exc
                retries += 1
                if retries > self.config.dataloader_max_retries:
                    raise RuntimeError(
                        f"Data loader failed after {self.config.dataloader_max_retries} retries."
                    ) from exc
                wait_s = 0.5 * retries
                print(
                    f"[{_timestamp()}] [warn] data fetch failed "
                    f"(retry {retries}/{self.config.dataloader_max_retries}): {exc}. "
                    f"waiting {wait_s:.1f}s"
                )
                time.sleep(wait_s)

    def _active_seq_len(self, full_seq_len: int, step: int) -> int:
        warmup_steps = int(self.config.seq_len_warmup_steps)
        if warmup_steps <= 0:
            return full_seq_len
        if full_seq_len <= 1:
            return full_seq_len
        start_len = int(self.config.seq_len_warmup_start)
        start_len = max(8, min(start_len, full_seq_len))
        if step >= warmup_steps:
            return full_seq_len
        progress = (step - 1) / max(warmup_steps - 1, 1)
        target = int(round(start_len + (full_seq_len - start_len) * progress))
        return max(8, min(target, full_seq_len))

    def _compute_loss(self, batch: dict[str, torch.Tensor], step: int) -> tuple[torch.Tensor, int, int]:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        source_ids = batch.get("source_id")
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device, non_blocking=True)
        if labels.device != self.device:
            labels = labels.to(self.device, non_blocking=True)
        if isinstance(source_ids, torch.Tensor) and source_ids.device != self.device:
            source_ids = source_ids.to(self.device, non_blocking=True)
        active_seq_len = self._active_seq_len(input_ids.size(1), step)
        if active_seq_len < input_ids.size(1):
            input_ids = input_ids[:, :active_seq_len]
            labels = labels[:, :active_seq_len]
        input_ids, labels, source_ids = self._apply_hard_replay(input_ids, labels, source_ids, step)

        with self._autocast_context():
            model_ref = self._model_ref()
            model_cfg = getattr(model_ref, "config", None)
            mtp_tokens = int(getattr(model_cfg, "mtp_tokens", 0)) if model_cfg is not None else 0
            mtp_weight = float(getattr(model_cfg, "mtp_loss_weight", 0.0)) if model_cfg is not None else 0.0
            moe_z_weight = float(getattr(model_cfg, "moe_z_loss_weight", 0.0)) if model_cfg is not None else 0.0
            aux_logits: list[torch.Tensor] = []
            try:
                out = self.model(input_ids, return_aux=(mtp_tokens > 0 and mtp_weight > 0))
            except Exception as exc:
                if self._compiled_active and self._is_compile_backend_error(exc):
                    if self.config.strict_fail_fast:
                        raise RuntimeError(f"torch.compile backend failed in strict mode: {exc}") from exc
                    self._disable_compile(str(exc))
                    out = self.model(input_ids, return_aux=(mtp_tokens > 0 and mtp_weight > 0))
                else:
                    raise
            logits = out
            if isinstance(out, tuple):
                logits = out[0]
                if len(out) > 1 and isinstance(out[1], list):
                    aux_logits = out[1]
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view(input_ids.size(0), -1)
            sample_loss_raw = token_loss.mean(dim=1)
            self._update_source_loss_ema(sample_loss_raw, source_ids if isinstance(source_ids, torch.Tensor) else None)
            sample_loss = sample_loss_raw
            sample_weights: torch.Tensor | None = None
            if isinstance(source_ids, torch.Tensor) and source_ids.numel() == input_ids.size(0):
                sid = source_ids.view(-1).to(torch.long).clamp_(0, self._source_loss_weights.numel() - 1)
                sample_weights = self._source_loss_weights.index_select(0, sid).to(sample_loss.dtype)
                sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
                sample_loss = sample_loss * sample_weights
            loss = sample_loss.mean()
            self._push_hard_replay(
                input_ids=input_ids,
                labels=labels,
                source_ids=source_ids if isinstance(source_ids, torch.Tensor) else None,
                sample_loss_raw=sample_loss_raw,
            )
            if aux_logits and mtp_weight > 0:
                aux_losses: list[torch.Tensor] = []
                for k, aux in enumerate(aux_logits, start=1):
                    if labels.size(1) <= k:
                        continue
                    target = labels[:, k:]
                    aux_token = F.cross_entropy(
                        aux.reshape(-1, aux.size(-1)),
                        target.reshape(-1),
                        reduction="none",
                    ).view(input_ids.size(0), -1)
                    aux_sample = aux_token.mean(dim=1)
                    if sample_weights is not None:
                        aux_sample = aux_sample * sample_weights
                    aux_losses.append(aux_sample.mean())
                if aux_losses:
                    loss = loss + (mtp_weight * torch.stack(aux_losses).mean())
            if moe_z_weight > 0 and hasattr(model_ref, "collect_aux_loss"):
                # Adaptive MoE balancing: strengthen router regularization when expert
                # usage imbalance rises, relax it when routing is already balanced.
                moe_z_weight_eff = moe_z_weight
                if hasattr(model_ref, "collect_moe_metrics"):
                    moe_metrics = model_ref.collect_moe_metrics()
                    imbalance = float(moe_metrics.get("moe_usage_imbalance", 0.0))
                    moe_z_weight_eff = moe_z_weight * max(0.5, min(4.0, 1.0 + (2.0 * imbalance)))
                    self._last_aux_metrics["moe_usage_imbalance_live"] = imbalance
                moe_z = model_ref.collect_aux_loss()
                loss = loss + (moe_z_weight_eff * moe_z)
                self._last_aux_metrics["moe_aux_z_loss"] = float(moe_z.detach().item())
                self._last_aux_metrics["moe_z_weight_eff"] = float(moe_z_weight_eff)
            else:
                self._last_aux_metrics.pop("moe_aux_z_loss", None)
                self._last_aux_metrics.pop("moe_z_weight_eff", None)
        tokens = int(input_ids.numel())
        return loss, tokens, active_seq_len

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.device.type != "cuda":
            return batch
        moved: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _prefetch_next_batch(self, data_iter):
        fetch_start = time.perf_counter()
        batch, data_iter = self._next_batch(data_iter)
        data_wait_s = time.perf_counter() - fetch_start
        if self.device.type == "cuda" and self.config.device_prefetch and self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                batch = self._batch_to_device(batch)
        return batch, data_iter, data_wait_s

    def _compute_grad_norm(self) -> float:
        total = torch.zeros(1, device=self.device)
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().float().pow(2).sum()
        return float(total.sqrt().item())

    def _compute_param_norm(self) -> float:
        total = torch.zeros(1, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad:
                total += p.detach().float().pow(2).sum()
        return float(total.sqrt().item())

    def _has_non_finite_grad(self) -> bool:
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return True
        return False

    def _collect_gpu_metrics(self, sync: bool = False) -> dict[str, float]:
        if self.device.type != "cuda":
            return {}
        if sync:
            torch.cuda.synchronize(self.device)
        metrics = {
            "gpu_mem_alloc_gb": float(torch.cuda.memory_allocated(self.device) / (1024**3)),
            "gpu_mem_reserved_gb": float(torch.cuda.memory_reserved(self.device) / (1024**3)),
            "gpu_mem_peak_gb": float(torch.cuda.max_memory_allocated(self.device) / (1024**3)),
        }
        if sync:
            try:
                metrics["gpu_util_percent"] = float(torch.cuda.utilization(self.device))
            except Exception:
                pass
        elif "gpu_util_percent" in self._last_gpu_metrics:
            metrics["gpu_util_percent"] = self._last_gpu_metrics["gpu_util_percent"]
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            metrics["gpu_mem_used_percent"] = float((1.0 - (free_b / max(total_b, 1))) * 100.0)
        except Exception:
            pass
        self._last_gpu_metrics.update(metrics)
        return metrics

    def _collect_system_metrics(self) -> dict[str, float]:
        return {
            "cpu_util_percent": float(psutil.cpu_percent(interval=None)),
            "ram_used_percent": float(psutil.virtual_memory().percent),
        }

    def _collect_model_metrics(self) -> dict[str, float]:
        model_ref = self._model_ref()
        metrics: dict[str, float] = {}
        if hasattr(model_ref, "collect_moe_metrics"):
            metrics.update(model_ref.collect_moe_metrics())
        if hasattr(model_ref, "collect_qk_metrics"):
            metrics.update(model_ref.collect_qk_metrics())
        if hasattr(model_ref, "collect_memory_metrics"):
            metrics.update(model_ref.collect_memory_metrics())
        if hasattr(self.optimizer, "metrics"):
            try:
                opt_metrics = self.optimizer.metrics()
                if isinstance(opt_metrics, dict):
                    metrics.update({k: float(v) for k, v in opt_metrics.items()})
            except Exception:
                pass
        metrics.update(self._last_aux_metrics)
        return metrics

    def _source_name_from_id(self, source_id: int) -> str:
        for name, idx in self._source_name_to_id.items():
            if idx == source_id:
                return name
        return "unknown"

    @staticmethod
    def _sigmoid_scalar(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _tanh_scalar(x: float) -> float:
        return math.tanh(x)

    def _apply_meta_theta(self, theta: torch.Tensor) -> None:
        if theta.numel() < 5:
            return
        t0 = float(theta[0].item())
        t1 = float(theta[1].item())
        t2 = float(theta[2].item())
        t3 = float(theta[3].item())
        t4 = float(theta[4].item())

        lr_mul = 1.0 + (0.08 * self._tanh_scalar(t0))
        source_temp = 1.0 + (0.35 * self._tanh_scalar(t1))
        source_decay_mult = 1.0 + (0.80 * self._tanh_scalar(t2))
        replay_mult = 1.0 + (0.90 * self._tanh_scalar(t3))
        router_mult = 1.0 + (0.80 * self._tanh_scalar(t4))

        temp_lo = float(self.config.self_evolve_meta_source_temp_min)
        temp_hi = float(self.config.self_evolve_meta_source_temp_max)
        if temp_hi < temp_lo:
            temp_hi = temp_lo
        self._meta_lr_multiplier = min(max(lr_mul, 0.85), 1.15)
        self._meta_source_temp = min(max(source_temp, temp_lo), temp_hi)
        self._meta_source_decay_mult = min(max(source_decay_mult, 0.15), 2.50)
        self._meta_replay_mult = min(max(replay_mult, 0.10), 3.00)
        self._meta_router_reg_mult = min(max(router_mult, 0.10), 3.00)

    @torch.no_grad()
    def _update_source_loss_ema(self, sample_loss_raw: torch.Tensor, source_ids: torch.Tensor | None) -> None:
        if source_ids is None or source_ids.numel() == 0:
            return
        sid = source_ids.view(-1).to(torch.long).clamp_(0, self._source_loss_ema.numel() - 1)
        losses = sample_loss_raw.detach().view(-1).float()
        beta = min(max(float(self.config.self_evolve_source_loss_ema_beta), 0.0), 0.9999)
        seen_decay = 0.995
        self._source_seen_ema.mul_(seen_decay)
        for idx in range(self._source_loss_ema.numel()):
            mask = sid == idx
            if not bool(mask.any().item()):
                continue
            prev = self._source_loss_ema[idx]
            cur = losses[mask].mean()
            updated = (beta * prev) + ((1.0 - beta) * cur)
            self._source_loss_ema[idx] = updated
            gain = (prev - updated).detach()
            self._source_gain_ema[idx] = (0.90 * self._source_gain_ema[idx]) + (0.10 * gain)
            self._source_seen_ema[idx] += float(mask.sum().item())

    @torch.no_grad()
    def _select_source_from_loss_pressure(self) -> tuple[str, float]:
        seen = self._source_seen_ema.detach().float()
        losses = self._source_loss_ema.detach().float()
        total_seen = float(seen.sum().item())
        if total_seen <= 1e-6:
            return "unknown", 0.0
        valid = seen > 1e-3
        if not bool(valid.any().item()):
            return "unknown", 0.0
        weighted_mean = float((losses * seen).sum().item() / max(total_seen, 1e-6))
        pressure = losses - weighted_mean
        pressure = torch.where(valid, pressure, torch.full_like(pressure, -1e9))
        pressure[0] = -1e9  # do not prioritize unknown source id
        top_id = int(torch.argmax(pressure).item())
        top_pressure = float(pressure[top_id].item())
        if not math.isfinite(top_pressure) or top_pressure <= 0.0:
            return "unknown", 0.0
        return self._source_name_from_id(top_id), top_pressure

    @torch.no_grad()
    def _push_hard_replay(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        source_ids: torch.Tensor | None,
        sample_loss_raw: torch.Tensor,
    ) -> None:
        if not self.config.self_evolve_hard_replay:
            return
        capacity = max(0, int(self.config.self_evolve_hard_replay_capacity))
        if capacity <= 0:
            return
        topk = max(0, int(self.config.self_evolve_hard_replay_topk_per_batch))
        if topk <= 0:
            return
        losses = sample_loss_raw.detach().view(-1).float().cpu()
        if losses.numel() == 0:
            return
        k = min(topk, int(losses.numel()))
        vals, idxs = torch.topk(losses, k=k)
        src_cpu = source_ids.detach().view(-1).to(torch.long).cpu() if isinstance(source_ids, torch.Tensor) else None
        in_cpu = input_ids.detach().cpu()
        lab_cpu = labels.detach().cpu()
        for local_i in range(k):
            row = int(idxs[local_i].item())
            item_loss = float(vals[local_i].item())
            entry: dict[str, torch.Tensor] = {
                "input_ids": in_cpu[row].clone(),
                "labels": lab_cpu[row].clone(),
            }
            if src_cpu is not None and src_cpu.numel() > row:
                entry["source_id"] = src_cpu[row].view(1).clone()
            else:
                entry["source_id"] = torch.zeros(1, dtype=torch.long)

            rec = (item_loss, self._hard_replay_counter, entry)
            self._hard_replay_counter += 1
            if len(self._hard_replay_heap) < capacity:
                heapq.heappush(self._hard_replay_heap, rec)
            elif item_loss > self._hard_replay_heap[0][0]:
                heapq.heapreplace(self._hard_replay_heap, rec)

    @torch.no_grad()
    def _apply_hard_replay(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        source_ids: torch.Tensor | None,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._last_hard_replay_injected = 0
        self._last_hard_replay_mean_loss = 0.0
        if not self.config.self_evolve_hard_replay:
            return input_ids, labels, source_ids
        if step < max(1, int(self.config.self_evolve_hard_replay_start_step)):
            return input_ids, labels, source_ids
        interval = max(1, int(self.config.self_evolve_hard_replay_interval))
        if (step % interval) != 0:
            return input_ids, labels, source_ids
        if not self._hard_replay_heap:
            return input_ids, labels, source_ids

        batch_size = int(input_ids.size(0))
        replace_frac = float(self.config.self_evolve_hard_replay_replace_frac) * float(self._meta_replay_mult)
        replace_frac = min(max(replace_frac, 0.0), 1.0)
        replace_n = int(batch_size * replace_frac)
        if replace_n <= 0:
            return input_ids, labels, source_ids
        replace_n = min(replace_n, batch_size, len(self._hard_replay_heap))
        if replace_n <= 0:
            return input_ids, labels, source_ids

        # Bias replay sampling toward the highest-loss stored examples.
        sorted_heap = sorted(self._hard_replay_heap, key=lambda x: x[0], reverse=True)
        candidate_k = max(replace_n, min(len(sorted_heap), replace_n * 3))
        candidates = sorted_heap[:candidate_k]
        if len(candidates) > replace_n:
            chosen = random.sample(candidates, k=replace_n)
        else:
            chosen = candidates

        row_ids = random.sample(range(batch_size), k=len(chosen))
        out_inputs = input_ids.clone()
        out_labels = labels.clone()
        out_sources = source_ids.clone() if isinstance(source_ids, torch.Tensor) else None

        total_loss = 0.0
        injected = 0
        for dst_row, (loss_v, _, entry) in zip(row_ids, chosen):
            src_input = entry["input_ids"].to(device=input_ids.device, dtype=input_ids.dtype)
            src_label = entry["labels"].to(device=labels.device, dtype=labels.dtype)
            if src_input.numel() != input_ids.size(1) or src_label.numel() != labels.size(1):
                continue
            out_inputs[dst_row].copy_(src_input.view_as(out_inputs[dst_row]))
            out_labels[dst_row].copy_(src_label.view_as(out_labels[dst_row]))
            if out_sources is not None and "source_id" in entry:
                src_id = entry["source_id"].to(device=out_sources.device, dtype=out_sources.dtype)
                out_sources[dst_row].copy_(src_id.view_as(out_sources[dst_row]))
            total_loss += float(loss_v)
            injected += 1

        self._last_hard_replay_injected = injected
        if injected > 0:
            self._last_hard_replay_mean_loss = total_loss / injected
        return out_inputs, out_labels, out_sources

    @torch.no_grad()
    def _normalize_source_weights(self) -> None:
        w = self._source_loss_weights
        mean = w.mean().clamp_min(1e-6)
        w.div_(mean)
        w.clamp_(
            min=float(self.config.self_evolve_source_weight_min),
            max=float(self.config.self_evolve_source_weight_max),
        )
        mean2 = w.mean().clamp_min(1e-6)
        w.div_(mean2)

    @torch.no_grad()
    def _apply_source_weight_update(self, target_source: str) -> None:
        if not self.config.self_evolve_source_reweight:
            return
        decay = float(self.config.self_evolve_source_decay)
        decay = min(max(decay, 0.0), 0.5)
        if decay <= 0:
            return
        adaptive_target, pressure = self._select_source_from_loss_pressure()
        if adaptive_target != "unknown":
            target_source = adaptive_target
            pressure_scale = min(max(pressure, 0.0), 2.0)
            decay = min(0.5, decay * (1.0 + (0.25 * pressure_scale)))
        decay = min(0.5, max(0.0, decay * self._meta_source_decay_mult))
        target_id = int(self._source_name_to_id.get(target_source, self._source_name_to_id["unknown"]))

        losses = self._source_loss_ema.detach().float()
        seen = self._source_seen_ema.detach().float()
        gains = self._source_gain_ema.detach().float()
        valid = seen > 1e-3
        if not bool(valid.any().item()):
            return

        seen_mean = seen[valid].mean().clamp_min(1e-6)
        underseen = (seen_mean - seen) / seen_mean
        underseen = underseen.clamp(-1.0, 1.0)
        loss_center = (losses[valid] * seen[valid]).sum() / seen[valid].sum().clamp_min(1e-6)
        loss_std = losses[valid].std(unbiased=False).clamp_min(1e-4)
        difficulty = (losses - loss_center) / loss_std
        gain_std = gains[valid].std(unbiased=False).clamp_min(1e-4)
        gain_norm = (gains - gains[valid].mean()) / gain_std

        utility = difficulty
        utility = utility + (float(self.config.self_evolve_meta_underseen_boost) * underseen)
        utility = utility - (float(self.config.self_evolve_meta_gain_penalty) * gain_norm)
        utility[target_id] = utility[target_id] + 0.35
        utility[0] = utility[0] - 0.50
        utility = torch.where(valid, utility, torch.full_like(utility, -6.0))

        temp = max(float(self._meta_source_temp), 1e-3)
        soft = torch.softmax(utility / temp, dim=0)
        desired = soft * float(self._source_loss_weights.numel())
        w = self._source_loss_weights
        w.mul_(1.0 - decay).add_(desired, alpha=decay)
        self._normalize_source_weights()

    def _source_weight_metrics(self) -> dict[str, float]:
        w = self._source_loss_weights.detach().float().cpu()
        l = self._source_loss_ema.detach().float().cpu()
        s = self._source_seen_ema.detach().float().cpu()
        g = self._source_gain_ema.detach().float().cpu()
        return {
            "src_weight_unknown": float(w[0].item()),
            "src_weight_fineweb": float(w[1].item()),
            "src_weight_cosmopedia": float(w[2].item()),
            "src_weight_commoncrawl": float(w[3].item()),
            "src_weight_synthetic": float(w[4].item()),
            "src_loss_ema_unknown": float(l[0].item()),
            "src_loss_ema_fineweb": float(l[1].item()),
            "src_loss_ema_cosmopedia": float(l[2].item()),
            "src_loss_ema_commoncrawl": float(l[3].item()),
            "src_loss_ema_synthetic": float(l[4].item()),
            "src_seen_ema_unknown": float(s[0].item()),
            "src_seen_ema_fineweb": float(s[1].item()),
            "src_seen_ema_cosmopedia": float(s[2].item()),
            "src_seen_ema_commoncrawl": float(s[3].item()),
            "src_seen_ema_synthetic": float(s[4].item()),
            "src_gain_ema_unknown": float(g[0].item()),
            "src_gain_ema_fineweb": float(g[1].item()),
            "src_gain_ema_cosmopedia": float(g[2].item()),
            "src_gain_ema_commoncrawl": float(g[3].item()),
            "src_gain_ema_synthetic": float(g[4].item()),
            "self_evolve_mutation_scale": float(self._self_evolve_mutation_scale),
            "self_evolve_hard_replay_buffer": float(len(self._hard_replay_heap)),
            "self_evolve_hard_replay_injected": float(self._last_hard_replay_injected),
            "self_evolve_hard_replay_mean_loss": float(self._last_hard_replay_mean_loss),
            "self_evolve_meta_lr_mul": float(self._meta_lr_multiplier),
            "self_evolve_meta_source_temp": float(self._meta_source_temp),
            "self_evolve_meta_decay_mult": float(self._meta_source_decay_mult),
            "self_evolve_meta_replay_mult": float(self._meta_replay_mult),
            "self_evolve_meta_router_mult": float(self._meta_router_reg_mult),
        }

    def _select_bandit_arm(self, step: int) -> int:
        if not self.config.self_evolve_bandit:
            return self._bandit_last_arm
        # Ensure each arm is tried at least once.
        for idx, count in enumerate(self._bandit_counts):
            if count == 0:
                return idx
        total = max(1, sum(self._bandit_counts))
        c = float(self.config.self_evolve_bandit_ucb_c)
        best_idx = 0
        best_score = -1e30
        for idx, value in enumerate(self._bandit_value_ema):
            count = max(1, self._bandit_counts[idx])
            bonus = c * math.sqrt(math.log(total + 1.0) / count)
            score = float(value + bonus)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _update_bandit(self, arm_idx: int, reward: float) -> None:
        self._bandit_steps += 1
        self._bandit_last_arm = arm_idx
        self._bandit_last_reward = float(reward)
        self._bandit_counts[arm_idx] += 1
        # EMA value estimate keeps controller responsive to non-stationary training.
        beta = 0.85
        prev = float(self._bandit_value_ema[arm_idx])
        self._bandit_value_ema[arm_idx] = (beta * prev) + ((1.0 - beta) * float(reward))

    def _apply_bandit_arm(self, arm_idx: int) -> dict[str, float]:
        arm = self._bandit_arms[arm_idx]
        mut_std = max(0.0, float(self._self_evolve_mutation_scale))
        # Small multiplicative mutation keeps exploration active over long runs.
        noise = (torch.randn((), device=self.device).item() * mut_std) if mut_std > 0 else 0.0
        lr_mul = max(0.85, min(1.15, float(arm["lr_mul"]) * (1.0 + noise)))
        orth_delta = int(arm["orth_delta"])
        adaptive_tol_delta = float(arm["adaptive_tol_delta"]) * (1.0 + noise)
        teon_delta = float(arm.get("teon_delta", 0.0)) * (1.0 + noise)

        self._set_lr_factor(self._sa_lr_factor * lr_mul)
        self._set_muon_runtime_knobs(
            orth_every_end=self._sa_orth_every_end + orth_delta,
            adaptive_tol=self._sa_adaptive_orth_tol + adaptive_tol_delta,
        )
        self._set_teon_coupling(self._sa_teon_coupling + teon_delta)
        target_source = str(arm["source"])
        source_pressure = 0.0
        adaptive_target, source_pressure = self._select_source_from_loss_pressure()
        if adaptive_target != "unknown":
            target_source = adaptive_target
        self._apply_source_weight_update(target_source)
        return {
            "self_evolve_arm_index": float(arm_idx),
            "self_evolve_arm_name_hash": float(abs(hash(arm["name"])) % 100000),
            "self_evolve_lr_mul": float(lr_mul),
            "self_evolve_target_source_id": float(self._source_name_to_id.get(target_source, 0)),
            "self_evolve_source_pressure": float(source_pressure),
            "self_evolve_teon_coupling": float(self._sa_teon_coupling),
        }

    def _set_lr_factor(self, factor: float) -> None:
        min_factor = max(float(self.config.self_accel_lr_min_factor), 0.75)
        factor = max(min_factor, min(self.config.self_accel_lr_max_factor, factor))
        self._sa_lr_factor = float(factor)
        target_lr = self._sa_base_lr * self._sa_lr_factor * float(self._meta_lr_multiplier)
        for group in self.optimizer.param_groups:
            group["lr"] = target_lr

    def _set_muon_runtime_knobs(self, orth_every_end: int, adaptive_tol: float) -> None:
        self._sa_orth_every_end = max(self._sa_orth_floor, min(self.config.self_accel_max_orth_end, int(orth_every_end)))
        self._sa_adaptive_orth_tol = max(0.01, min(0.20, float(adaptive_tol)))
        for group in self.optimizer.param_groups:
            if "orthogonalize_every_end" in group:
                group["orthogonalize_every_end"] = self._sa_orth_every_end
            if "adaptive_orth_tol" in group:
                group["adaptive_orth_tol"] = self._sa_adaptive_orth_tol

    def _set_teon_coupling(self, value: float) -> None:
        lo = float(self.config.self_evolve_teon_min)
        hi = float(self.config.self_evolve_teon_max)
        if hi < lo:
            hi = lo
        self._sa_teon_coupling = min(max(float(value), lo), hi)
        for group in self.optimizer.param_groups:
            if "teon_coupling" in group:
                group["teon_coupling"] = self._sa_teon_coupling

    def _tune_teon(self, avg_grad_ratio: float, loss_slope: float) -> None:
        if not self.config.self_evolve_teon_adapt:
            return
        step = max(0.0, float(self.config.self_evolve_teon_step))
        teon = self._sa_teon_coupling
        if loss_slope < self.config.self_accel_loss_bad_threshold or avg_grad_ratio > self.config.self_accel_grad_ratio_high:
            teon += step
        elif loss_slope > 0 and avg_grad_ratio < (self.config.self_accel_grad_ratio_low * 1.2):
            teon -= (0.5 * step)
        self._set_teon_coupling(teon)

    def _adapt_mutation_scale(self, reward: float) -> None:
        if not self.config.self_evolve_mutation_adapt:
            return
        beta = min(max(float(self.config.self_evolve_reward_ema_beta), 0.0), 0.9999)
        prev_ema = float(self._self_evolve_reward_ema)
        new_ema = (beta * prev_ema) + ((1.0 - beta) * float(reward))
        delta = float(reward - new_ema)
        self._self_evolve_reward_ema = new_ema

        scale = float(self._self_evolve_mutation_scale)
        # Increase exploration when reward stagnates/drops; decrease when regime improves.
        if delta < 0.0:
            scale *= 1.06
        elif delta > max(abs(new_ema) * 0.02, 1e-6):
            scale *= 0.96
        lo = float(self.config.self_evolve_mutation_min)
        hi = float(self.config.self_evolve_mutation_max)
        if hi < lo:
            hi = lo
        self._self_evolve_mutation_scale = min(max(scale, lo), hi)

    def _tune_moe_regularization(self, imbalance: float) -> tuple[float, float]:
        model_ref = self._model_ref()
        model_cfg = getattr(model_ref, "config", None)
        moe_z_weight = 0.0
        router_balance_lr = 0.0
        if model_cfg is not None and hasattr(model_cfg, "moe_z_loss_weight"):
            current = float(getattr(model_cfg, "moe_z_loss_weight"))
            if imbalance > self.config.self_accel_moe_imbalance_high:
                current = min(0.05, max(1e-5, current * (1.0 + (0.05 * self._meta_router_reg_mult))))
            elif imbalance < self.config.self_accel_moe_imbalance_low:
                current = max(1e-5, current * (1.0 - (0.02 * self._meta_router_reg_mult)))
            setattr(model_cfg, "moe_z_loss_weight", current)
            moe_z_weight = current

        if imbalance > self.config.self_accel_moe_imbalance_high:
            rb_scale = 1.0 + (0.05 * self._meta_router_reg_mult)
        elif imbalance < self.config.self_accel_moe_imbalance_low * 0.8:
            rb_scale = 1.0 - (0.02 * self._meta_router_reg_mult)
        else:
            rb_scale = 1.0
        if rb_scale != 1.0:
            for module in model_ref.modules():
                if hasattr(module, "router_balance_lr"):
                    current = float(getattr(module, "router_balance_lr"))
                    current = max(1e-6, min(5e-2, current * rb_scale))
                    setattr(module, "router_balance_lr", current)
                    router_balance_lr = current
        return moe_z_weight, router_balance_lr

    def _meta_evolve(self, step: int, reward: float) -> dict[str, float]:
        if not self.config.self_evolve_meta_es:
            return {}
        if step < self.config.self_accel_warmup_steps:
            return {}
        interval = max(1, int(self.config.self_evolve_meta_interval))
        if (step % interval) != 0:
            return {}

        beta = min(max(float(self.config.self_evolve_meta_reward_beta), 0.0), 0.9999)
        self._meta_reward_ema = (beta * self._meta_reward_ema) + ((1.0 - beta) * float(reward))
        centered_reward = float(reward - self._meta_reward_ema)

        sigma = max(1e-4, float(self.config.self_evolve_meta_sigma))
        learn_rate = max(0.0, float(self.config.self_evolve_meta_lr))
        event = 0.0  # 1=probe+, -1=probe-, 2=update
        grad_norm = 0.0
        reward_gap = 0.0

        if self._meta_phase == 0:
            self._meta_eps = torch.randn_like(self._meta_theta)
            self._meta_phase = 1
            self._apply_meta_theta(self._meta_theta + (sigma * self._meta_eps))
            event = 1.0
        elif self._meta_phase == 1:
            self._meta_plus_reward = centered_reward
            self._meta_phase = -1
            self._apply_meta_theta(self._meta_theta - (sigma * self._meta_eps))
            event = -1.0
        else:
            self._meta_minus_reward = centered_reward
            reward_gap = float(self._meta_plus_reward - self._meta_minus_reward)
            grad = (reward_gap / (2.0 * sigma)) * self._meta_eps
            grad_norm = float(grad.norm().item())
            if grad_norm > 0:
                grad = grad / (grad_norm + 1e-6)
            self._meta_theta = (self._meta_theta + (learn_rate * grad)).clamp_(-4.0, 4.0)
            self._apply_meta_theta(self._meta_theta)
            self._meta_phase = 0
            event = 2.0
            if centered_reward > self._meta_best_reward:
                self._meta_best_reward = centered_reward
                self._meta_best_theta.copy_(self._meta_theta)

        # Keep LR in sync after meta change.
        self._set_lr_factor(self._sa_lr_factor)
        return {
            "self_evolve_meta_event": float(event),
            "self_evolve_meta_phase": float(self._meta_phase),
            "self_evolve_meta_reward_centered": float(centered_reward),
            "self_evolve_meta_reward_gap": float(reward_gap),
            "self_evolve_meta_grad_norm": float(grad_norm),
            "self_evolve_meta_lr_mul": float(self._meta_lr_multiplier),
            "self_evolve_meta_source_temp": float(self._meta_source_temp),
            "self_evolve_meta_decay_mult": float(self._meta_source_decay_mult),
            "self_evolve_meta_replay_mult": float(self._meta_replay_mult),
            "self_evolve_meta_router_mult": float(self._meta_router_reg_mult),
            "self_evolve_meta_best_reward": float(self._meta_best_reward),
        }

    def _self_accelerate(self, step: int, record: dict[str, float | int | str]) -> dict[str, float]:
        if not self.config.self_accelerate:
            return {}

        step_loss = float(record.get("loss", 0.0))
        step_time = float(record.get("step_time_s", 0.0))
        tokens_per_s = float(record.get("tokens_per_s", 0.0))
        opt_frac = float(record.get("optimizer_step_frac", 0.0))
        grad_ratio = float(record.get("grad_to_param_ratio", 0.0))
        moe_imbalance = float(record.get("moe_usage_imbalance", 0.0))
        self._sa_history.append(
            {
                "loss": step_loss,
                "step_time_s": step_time,
                "tokens_per_s": tokens_per_s,
                "optimizer_step_frac": opt_frac,
                "grad_to_param_ratio": grad_ratio,
                "moe_usage_imbalance": moe_imbalance,
            }
        )
        history_keep = max(4, self.config.self_accel_warmup_steps + (self.config.self_accel_interval * 4))
        if len(self._sa_history) > history_keep:
            self._sa_history = self._sa_history[-history_keep:]

        if step < self.config.self_accel_warmup_steps:
            return {}
        interval = max(1, int(self.config.self_accel_interval))
        if len(self._sa_history) < interval or (step % interval) != 0:
            return {}

        window = self._sa_history[-interval:]
        loss_slope = (window[0]["loss"] - window[-1]["loss"]) / max(interval - 1, 1)
        avg_tokens = sum(x["tokens_per_s"] for x in window) / interval
        avg_opt_frac = sum(x["optimizer_step_frac"] for x in window) / interval
        avg_grad_ratio = sum(x["grad_to_param_ratio"] for x in window) / interval
        avg_moe_imbalance = sum(x["moe_usage_imbalance"] for x in window) / interval
        # Combined objective: maximize learning-speed while discouraging unstable/imbalanced regimes.
        reward = (
            (loss_slope * max(avg_tokens, 1e-6))
            - (0.15 * avg_opt_frac * max(avg_tokens, 1e-6))
            - (20.0 * max(avg_grad_ratio - self.config.self_accel_grad_ratio_high, 0.0))
            - (2.0 * max(avg_moe_imbalance - self.config.self_accel_moe_imbalance_high, 0.0))
        )

        # 1) Learning-rate adaptation for learning-speed (loss drop per second).
        lr_factor = self._sa_lr_factor
        if loss_slope > 0 and avg_grad_ratio < self.config.self_accel_grad_ratio_low:
            lr_factor *= 1.0 + self.config.self_accel_lr_up
        elif (
            loss_slope < self.config.self_accel_loss_bad_threshold
            and avg_grad_ratio > (self.config.self_accel_grad_ratio_high * 1.15)
        ):
            lr_factor *= self.config.self_accel_lr_down
        self._set_lr_factor(lr_factor)

        # 2) Muon runtime adaptation: trade orth-cost vs quality online.
        orth_end = self._sa_orth_every_end
        adaptive_tol = self._sa_adaptive_orth_tol
        if avg_opt_frac > self.config.self_accel_opt_bottleneck_threshold and loss_slope >= self.config.self_accel_loss_bad_threshold:
            orth_end = min(self.config.self_accel_max_orth_end, orth_end + 1)
            adaptive_tol = min(0.12, adaptive_tol + 0.005)
        elif loss_slope < self.config.self_accel_loss_bad_threshold:
            orth_end = max(self._sa_orth_floor, orth_end - 1)
            adaptive_tol = max(0.02, adaptive_tol - 0.005)
        self._set_muon_runtime_knobs(orth_every_end=orth_end, adaptive_tol=adaptive_tol)
        self._tune_teon(avg_grad_ratio=avg_grad_ratio, loss_slope=loss_slope)

        # 3) MoE stabilization adaptation (avoid collapse -> faster effective learning).
        moe_z_weight, router_balance_lr = self._tune_moe_regularization(avg_moe_imbalance)
        self._adapt_mutation_scale(reward)
        meta_metrics = self._meta_evolve(step=step, reward=reward)

        bandit_metrics: dict[str, float] = {}
        if self.config.self_evolve_bandit:
            # Credit assignment for arm active during this interval.
            self._update_bandit(self._bandit_last_arm, reward)
            if (step % max(1, int(self.config.self_evolve_bandit_interval))) == 0:
                arm_idx = self._select_bandit_arm(step)
                bandit_metrics = self._apply_bandit_arm(arm_idx)
                bandit_metrics["self_evolve_arm_count"] = float(self._bandit_counts[arm_idx])
                bandit_metrics["self_evolve_arm_value"] = float(self._bandit_value_ema[arm_idx])

        metrics_out = {
            "self_accel_reward": float(reward),
            "self_accel_loss_slope": float(loss_slope),
            "self_accel_lr_factor": float(self._sa_lr_factor),
            "self_accel_muon_orth_end": float(self._sa_orth_every_end),
            "self_accel_muon_adaptive_tol": float(self._sa_adaptive_orth_tol),
            "self_accel_muon_teon_coupling": float(self._sa_teon_coupling),
            "self_accel_moe_z_weight": float(moe_z_weight),
            "self_accel_router_balance_lr": float(router_balance_lr),
            "self_accel_avg_opt_frac": float(avg_opt_frac),
            "self_accel_avg_grad_ratio": float(avg_grad_ratio),
            "self_accel_avg_moe_imbalance": float(avg_moe_imbalance),
            "self_evolve_last_arm": float(self._bandit_last_arm),
            "self_evolve_last_reward": float(self._bandit_last_reward),
            "self_evolve_reward_ema": float(self._self_evolve_reward_ema),
        }
        metrics_out.update(self._source_weight_metrics())
        metrics_out.update(meta_metrics)
        metrics_out.update(bandit_metrics)
        return metrics_out

    def _accumulate_source_counts(self, batch: dict[str, torch.Tensor], counts: torch.Tensor) -> None:
        src = batch.get("source_id")
        if not isinstance(src, torch.Tensor):
            return
        src_ids = src.detach().view(-1).to(torch.long)
        if src_ids.is_cuda:
            src_ids = src_ids.cpu()
        src_ids = src_ids.clamp(min=0, max=max(counts.numel() - 1, 0))
        binc = torch.bincount(src_ids, minlength=counts.numel())
        counts += binc[: counts.numel()]

    def _dump_error_state(
        self,
        step: int,
        exception: BaseException,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self.config.save_error_batch:
            return
        payload: dict[str, object] = {
            "step": step,
            "exception": repr(exception),
            "traceback": traceback.format_exc(),
            "time": _timestamp(),
        }
        if batch is not None:
            payload["batch_shape"] = {k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            payload["batch_preview"] = {
                k: v[0, : min(32, v.size(1))].detach().cpu().tolist()
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and v.ndim >= 2
            }
        path = self.errors_dir / f"step_{step:08d}.pt"
        torch.save(payload, path)
        print(f"[{_timestamp()}] [debug] saved failure snapshot to {path}")

    def _write_metrics(self, record: dict[str, float | int | str]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def train(self, max_steps: int | None = None) -> list[float]:
        total_steps = max_steps if max_steps is not None else self.config.max_steps
        losses: list[float] = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_iter = iter(self.dataloader)
        start_step = self._resume_if_available()

        model_ref = self._model_ref()
        if hasattr(model_ref, "algorithmic_features"):
            print(f"[{_timestamp()}] model_features={model_ref.algorithmic_features()}")

        anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True) if self.config.detect_anomaly else nullcontext()
        with anomaly_ctx:
            skipped_non_finite_grad_steps = 0
            for step in range(start_step + 1, total_steps + 1):
                step_start = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                step_loss = 0.0
                step_tokens = 0
                data_wait_s = 0.0
                oom_retries = 0
                fwd_bwd_s = 0.0
                active_seq_len = 0
                source_counts = torch.zeros(5, dtype=torch.long)

                micro_step = 0
                prefetched_batch = None
                if self.device.type == "cuda" and self.config.device_prefetch and self._prefetch_stream is not None:
                    prefetched_batch, data_iter, wait_s = self._prefetch_next_batch(data_iter)
                    data_wait_s += wait_s
                while micro_step < self.config.grad_accum_steps:
                    if prefetched_batch is not None and self.device.type == "cuda" and self._prefetch_stream is not None:
                        torch.cuda.current_stream(self.device).wait_stream(self._prefetch_stream)
                        batch = prefetched_batch
                        prefetched_batch, data_iter, wait_s = self._prefetch_next_batch(data_iter)
                        data_wait_s += wait_s
                    else:
                        fetch_start = time.perf_counter()
                        batch, data_iter = self._next_batch(data_iter)
                        data_wait_s += time.perf_counter() - fetch_start
                    self._accumulate_source_counts(batch, source_counts)

                    try:
                        compute_start = time.perf_counter()
                        loss, tokens, active_seq_len = self._compute_loss(batch, step=step)
                        if not torch.isfinite(loss):
                            msg = f"non-finite loss={float(loss.detach().item())}"
                            if self.config.skip_non_finite_loss:
                                print(f"[{_timestamp()}] [warn] {msg}; micro-step skipped.")
                                self.optimizer.zero_grad(set_to_none=True)
                                micro_step += 1
                                continue
                            raise FloatingPointError(msg)

                        scaled = loss / self.config.grad_accum_steps
                        if self.scaler.is_enabled():
                            self.scaler.scale(scaled).backward()
                        else:
                            scaled.backward()
                        fwd_bwd_s += time.perf_counter() - compute_start
                    except RuntimeError as exc:
                        if self._is_checkpoint_error(exc):
                            if self.config.strict_fail_fast:
                                self._dump_error_state(step, exc, batch)
                                raise RuntimeError("Activation-checkpointing error in strict mode.") from exc
                            self._disable_activation_checkpointing("checkpoint metadata mismatch")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        if _is_oom_error(exc) and self.config.oom_recovery:
                            oom_retries += 1
                            print(
                                f"[{_timestamp()}] [warn] CUDA OOM at step {step}, micro-step {micro_step + 1}. "
                                f"retry {oom_retries}/{self.config.max_oom_retries_per_step}"
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                            if oom_retries > self.config.max_oom_retries_per_step:
                                self._dump_error_state(step, exc, batch)
                                raise
                            continue
                        self._dump_error_state(step, exc, batch)
                        raise
                    except Exception as exc:
                        if self._is_checkpoint_error(exc):
                            if self.config.strict_fail_fast:
                                self._dump_error_state(step, exc, batch)
                                raise RuntimeError("Activation-checkpointing error in strict mode.") from exc
                            self._disable_activation_checkpointing("checkpoint metadata mismatch")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        self._dump_error_state(step, exc, batch)
                        raise

                    step_loss += float(loss.detach().item())
                    step_tokens += tokens
                    micro_step += 1

                if self._has_non_finite_grad():
                    skipped_non_finite_grad_steps += 1
                    print(f"[{_timestamp()}] [warn] non-finite gradient detected at step {step}; step skipped.")
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                if self.config.grad_clip_norm > 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm).item()
                    )
                else:
                    grad_norm = self._compute_grad_norm()

                opt_start = time.perf_counter()
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                qk_clip_scale = 1.0
                if self.config.qk_clip_interval > 0 and (step % self.config.qk_clip_interval) == 0:
                    model_ref = self._model_ref()
                    if hasattr(model_ref, "apply_qk_clip"):
                        qk_clip_scale = float(model_ref.apply_qk_clip())
                self.optimizer.zero_grad(set_to_none=True)
                opt_step_s = time.perf_counter() - opt_start

                step_time_s = time.perf_counter() - step_start
                tokens_per_s = step_tokens / max(step_time_s, 1e-6)
                model_time_s = max(step_time_s - data_wait_s, 1e-6)
                model_tokens_per_s = step_tokens / model_time_s
                data_wait_frac = data_wait_s / max(step_time_s, 1e-6)
                fwd_bwd_frac = fwd_bwd_s / max(step_time_s, 1e-6)
                opt_step_frac = opt_step_s / max(step_time_s, 1e-6)
                if data_wait_frac > max(fwd_bwd_frac, opt_step_frac):
                    bottleneck = "data"
                elif opt_step_frac > max(data_wait_frac, fwd_bwd_frac):
                    bottleneck = "optimizer"
                else:
                    bottleneck = "forward_backward"

                loss_value = step_loss / max(self.config.grad_accum_steps, 1)
                losses.append(loss_value)
                perplexity = float(math.exp(min(loss_value, 20.0)))
                param_norm = self._compute_param_norm()
                grad_to_param_ratio = grad_norm / max(param_norm, 1e-12)
                self._tokens_seen_total += step_tokens

                record: dict[str, float | int | str] = {
                    "time": _timestamp(),
                    "step": step,
                    "loss": loss_value,
                    "perplexity": perplexity,
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_to_param_ratio": grad_to_param_ratio,
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "step_time_s": step_time_s,
                    "data_wait_s": data_wait_s,
                    "data_wait_frac": data_wait_frac,
                    "forward_backward_s": fwd_bwd_s,
                    "forward_backward_frac": fwd_bwd_frac,
                    "optimizer_step_s": opt_step_s,
                    "optimizer_step_frac": opt_step_frac,
                    "bottleneck": bottleneck,
                    "tokens": step_tokens,
                    "tokens_seen_total": self._tokens_seen_total,
                    "active_seq_len": active_seq_len,
                    "tokens_per_s": tokens_per_s,
                    "model_tokens_per_s": model_tokens_per_s,
                    "precision": str(self.autocast_dtype).replace("torch.", "") if self.autocast_dtype else "fp32",
                    "skipped_non_finite_grad_steps": skipped_non_finite_grad_steps,
                    "qk_clip_scale": qk_clip_scale,
                }
                source_total = int(source_counts.sum().item())
                if source_total > 0:
                    record["src_unknown_frac"] = float(source_counts[0].item() / source_total)
                    record["src_fineweb_frac"] = float(source_counts[1].item() / source_total)
                    record["src_cosmopedia_frac"] = float(source_counts[2].item() / source_total)
                    record["src_commoncrawl_frac"] = float(source_counts[3].item() / source_total)
                    record["src_synthetic_frac"] = float(source_counts[4].item() / source_total)
                sync_gpu_metrics = (
                    self.config.gpu_metrics_sync_interval > 0
                    and (step == 1 or step % self.config.gpu_metrics_sync_interval == 0)
                )
                record.update(self._collect_gpu_metrics(sync=sync_gpu_metrics))
                record.update(self._collect_system_metrics())
                record.update(self._collect_model_metrics())
                record.update(self._self_accelerate(step, record))
                self._write_metrics(record)

                if step % self.config.log_interval == 0:
                    printable = " ".join(
                        f"{k}={record[k]:.4f}" if isinstance(record[k], float) else f"{k}={record[k]}"
                        for k in [
                            "step",
                            "loss",
                            "grad_norm",
                            "lr",
                            "tokens_per_s",
                            "gpu_mem_alloc_gb",
                            "gpu_mem_peak_gb",
                            "gpu_util_percent",
                            "cpu_util_percent",
                            "moe_usage_imbalance",
                            "self_accel_lr_factor",
                            "self_accel_muon_orth_end",
                        ]
                        if k in record
                    )
                    print(printable)

                if (
                    self.device.type == "cuda"
                    and self.config.empty_cache_interval > 0
                    and step % self.config.empty_cache_interval == 0
                ):
                    torch.cuda.empty_cache()

                if self.config.save_interval > 0 and (step % self.config.save_interval == 0 or step == total_steps):
                    self._save_checkpoint(step)

        return losses


RewardFn = Callable[[str, str, str | None], float]


@dataclass
class GRPOConfig:
    group_size: int = 8
    max_new_tokens: int = 64
    temperature: float = 0.7
    backend: str = "trl"  # trl | internal
    num_generations: int = 8
    code_timeout_sec: float = 4.0
    reward_success: float = 2.0
    reward_fail: float = -1.0
    advantage_mode: str = "mean"  # mean | cvar
    cvar_alpha: float = 0.25
    normalize_advantages: bool = True


class GRPOTrainer:
    """
    Post-training GRPO scaffold (R1-Zero style):
    - Sample groups of responses for each prompt.
    - Compute rewards per sample.
    - Use group-relative advantages (reward - group_mean_reward).
    - Apply policy gradient update without a critic model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        reward_fn: RewardFn | None = None,
        config: GRPOConfig | None = None,
    ) -> None:
        from .code_sandbox import CodeSandbox

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config or GRPOConfig()
        self.sandbox = CodeSandbox(timeout_sec=self.config.code_timeout_sec)
        self.reward_fn = reward_fn
        self._trl_bridge = None
        if self.reward_fn is None:
            self.reward_fn = self._code_execution_reward
        if str(self.config.backend).strip().lower() not in {"trl", "internal"}:
            raise ValueError(f"Unsupported GRPO backend: {self.config.backend!r}")

    def close(self) -> None:
        self.sandbox.close()

    def _code_execution_reward(self, prompt: str, completion: str, target: str | None) -> float:
        del prompt, target
        return self.sandbox.reward(
            generated_code=completion,
            success_reward=self.config.reward_success,
            fail_reward=self.config.reward_fail,
        )

    def build_trl_trainer(self, *, train_dataset, output_dir: str, **kwargs):
        from .grpo_trl_bridge import TRLGRPOBridge, TRLGRPOBridgeConfig

        if self._trl_bridge is not None:
            self._trl_bridge.close()
        self._trl_bridge = TRLGRPOBridge(
            model=self.model,
            tokenizer=self.tokenizer,
            config=TRLGRPOBridgeConfig(
                num_generations=int(self.config.num_generations),
                max_new_tokens=int(self.config.max_new_tokens),
                temperature=float(self.config.temperature),
                code_timeout_sec=float(self.config.code_timeout_sec),
                success_reward=float(self.config.reward_success),
                fail_reward=float(self.config.reward_fail),
            ),
        )
        return self._trl_bridge.build_trainer(
            train_dataset=train_dataset,
            output_dir=output_dir,
            **kwargs,
        )

    @torch.no_grad()
    def sample_group(self, prompt_ids: torch.Tensor) -> list[torch.Tensor]:
        samples = []
        for _ in range(self.config.group_size):
            out = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            samples.append(out)
        return samples

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        mode = self.config.advantage_mode.strip().lower()
        if mode == "cvar":
            alpha = min(max(float(self.config.cvar_alpha), 1e-3), 1.0)
            k = max(1, int(math.ceil(rewards.numel() * alpha)))
            baseline = torch.topk(rewards, k=k, largest=True).values.mean()
        else:
            baseline = rewards.mean()
        advantages = rewards - baseline
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        return advantages

    def step(self, prompt_ids: torch.Tensor, target_text: str | None = None) -> dict[str, float]:
        """
        Internal GRPO step using ground-truth code execution rewards.
        """
        samples = self.sample_group(prompt_ids)
        texts: list[str] = []
        for sample in samples:
            token_ids = sample[0].tolist()
            texts.append(self.tokenizer.decode(token_ids))

        prompt_text = self.tokenizer.decode(prompt_ids[0].tolist())
        rewards = torch.tensor(
            [float(self.reward_fn(prompt_text, completion, target_text)) for completion in texts],
            dtype=torch.float32,
            device=prompt_ids.device,
        )
        advantages = self.compute_group_advantages(rewards)
        alpha = min(max(float(self.config.cvar_alpha), 1e-3), 1.0)
        cvar_k = max(1, int(math.ceil(rewards.numel() * alpha)))
        reward_cvar = torch.topk(rewards, k=cvar_k, largest=True).values.mean()
        return {
            "mean_reward": float(rewards.mean().item()),
            "std_reward": float(rewards.std(unbiased=False).item()),
            "cvar_reward": float(reward_cvar.item()),
            "mean_advantage": float(advantages.mean().item()),
        }
