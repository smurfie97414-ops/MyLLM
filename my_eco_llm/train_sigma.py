from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import tempfile

_UNSLOTH_COMPILE_DIR = Path.cwd() / "unsloth_compiled_cache"
_UNSLOTH_COMPILE_DIR.mkdir(parents=True, exist_ok=True)
_UNSLOTH_COMPILE_INIT = _UNSLOTH_COMPILE_DIR / "__init__.py"
if not _UNSLOTH_COMPILE_INIT.exists():
    _UNSLOTH_COMPILE_INIT.write_text("", encoding="utf-8")
_CWD_STR = str(Path.cwd())
if _CWD_STR not in sys.path:
    sys.path.insert(0, _CWD_STR)
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
_UNSLOTH_CRITICAL_MODULES = ("trl", "transformers", "peft")
_UNSLOTH_PRELOADED_CRITICAL = {name: int(name in sys.modules) for name in _UNSLOTH_CRITICAL_MODULES}
_UNSLOTH_IMPORT_ERROR = ""
try:
    import unsloth  # type: ignore  # noqa: F401

    _UNSLOTH_IMPORTED = True
except Exception as exc:
    _UNSLOTH_IMPORTED = False
    _UNSLOTH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

import numpy as np
import torch


class _ArgsFileParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        return shlex.split(line)


def _configure_runtime_threads() -> tuple[int, int]:
    cpu_count = max(1, int(os.cpu_count() or 1))
    intra_threads = max(1, min(24, cpu_count - 1))
    interop_threads = max(1, min(6, cpu_count // 4))
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = str(intra_threads)
    torch.set_num_threads(intra_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # Can already be initialized by runtime; keep current interop setting.
        pass
    return intra_threads, interop_threads


def _bootstrap_vs2026_toolchain_env() -> bool:
    if os.name != "nt":
        return False
    if shutil.which("cl.exe"):
        return True
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat",
        r"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat",
    ]
    tool_script = next((p for p in candidates if os.path.exists(p)), "")
    if not tool_script:
        return False

    bat_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False, encoding="utf-8") as f:
            bat_path = f.name
            f.write("@echo off\n")
            if tool_script.lower().endswith("vsdevcmd.bat"):
                f.write(f'call "{tool_script}" -arch=x64 -host_arch=x64 >nul\n')
            else:
                f.write(f'call "{tool_script}" >nul\n')
            f.write("if errorlevel 1 exit /b 1\n")
            f.write("set\n")
        proc = subprocess.run(
            ["cmd.exe", "/d", "/c", bat_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            return False
        for line in proc.stdout.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if not k:
                continue
            os.environ[k] = v
        return shutil.which("cl.exe") is not None
    finally:
        if bat_path and os.path.exists(bat_path):
            try:
                os.remove(bat_path)
            except OSError:
                pass


def _apply_auto_hardware_profile(args: argparse.Namespace) -> tuple[int, int]:
    cpu_count = max(1, int(os.cpu_count() or 1))
    backend = str(getattr(args, "data_backend", "hybrid")).strip().lower()
    hf_constrained = backend in {"hf", "hybrid"}
    if bool(args.auto_hardware_profile):
        if int(args.num_workers) < 0:
            reserve = 2 if cpu_count >= 8 else 1
            args.num_workers = max(1, min(16, cpu_count - reserve))
        if hf_constrained:
            args.num_workers = min(int(args.num_workers), 2)
        args.num_workers = max(0, int(args.num_workers))
        if int(args.prefetch_factor) <= 0:
            args.prefetch_factor = 4
        if args.num_workers == 0:
            args.persistent_workers = False
        else:
            args.persistent_workers = True
            args.prefetch_factor = max(2, int(args.prefetch_factor))
        if int(args.commoncrawl_parallel_files) <= 0:
            args.commoncrawl_parallel_files = max(4, min(32, (args.num_workers * 2) if args.num_workers > 0 else (cpu_count // 2)))
    else:
        args.num_workers = max(0, int(args.num_workers))
        if args.num_workers == 0:
            args.persistent_workers = False
        if int(args.commoncrawl_parallel_files) <= 0:
            args.commoncrawl_parallel_files = 1
    if bool(args.auto_hardware_profile):
        if str(getattr(args, "c3o_state_dtype", "auto")).strip().lower() == "auto":
            if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                args.c3o_state_dtype = "bf16"
    return int(args.num_workers), int(args.prefetch_factor)


def _resolve_model_param_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if bool(args.no_amp):
        return None
    precision = str(args.precision).strip().lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "auto":
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return None


def _configure_runtime_caches(output_dir: str) -> None:
    cache_root = Path.cwd() / ".runtime_cache"
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    runtime_tmp = cache_root / "tmp"
    unsloth_compile_cache = Path.cwd() / "unsloth_compiled_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    unsloth_compile_cache.mkdir(parents=True, exist_ok=True)
    unsloth_init = unsloth_compile_cache / "__init__.py"
    if not unsloth_init.exists():
        unsloth_init.write_text("", encoding="utf-8")
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))
    os.environ.setdefault("TMP", str(runtime_tmp))
    os.environ.setdefault("TEMP", str(runtime_tmp))
    os.environ.setdefault("TMPDIR", str(runtime_tmp))
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
    # PyInstaller EXE runtime lacks some inductor C++ wrapper headers; force Python wrapper path.
    os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER", "0")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE_CPP_CODEGEN", "1")
    os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
    if output_dir:
        run_cache = Path(output_dir) / "cache"
        run_cache.mkdir(parents=True, exist_ok=True)


def _require_module(name: str, required_symbols: tuple[str, ...] = ()) -> None:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"Missing or broken dependency '{name}': {exc}") from exc
    for symbol in required_symbols:
        if not hasattr(mod, symbol):
            raise RuntimeError(f"Dependency '{name}' is installed but missing required symbol '{symbol}'.")


def _audit_enabled_dependencies(args: argparse.Namespace) -> None:
    if not bool(args.strict_runtime_audit):
        return

    _require_module("torch")
    _require_module("numpy")
    _require_module("psutil")
    _require_module("pynvml")
    _require_module("triton")

    if str(args.tokenizer_backend).strip().lower() == "tiktoken":
        _require_module("tiktoken")
        _require_module("tiktoken_ext.openai_public")
    else:
        _require_module("transformers")

    if str(args.data_backend).strip().lower() in {"hf", "hybrid"}:
        _require_module("datasets")

    if str(args.grpo_backend).strip().lower() == "trl":
        _require_module("transformers")
        _require_module("accelerate")
        _require_module("peft")
        _require_module("trl", required_symbols=("GRPOConfig", "GRPOTrainer"))
        _require_module("wandb")
        _require_module("wandb_workspaces")

    if str(args.optimizer).strip().lower() == "gnprox" and not torch.cuda.is_available():
        raise RuntimeError("GNProx optimizer requires CUDA.")

    if bool(args.c3o_credit_enabled) and str(args.optimizer).strip().lower() != "c3o":
        raise RuntimeError(
            "c3o credit is enabled but optimizer is not 'c3o'. "
            "Use --optimizer c3o or disable --c3o-credit-enabled."
        )

    if str(args.quant_mode).strip().lower() == "bitnet_1_58":
        _require_module("unsloth", required_symbols=("FastLanguageModel",))
        _require_module("unsloth_zoo")
        if not _UNSLOTH_IMPORTED:
            raise RuntimeError(f"unsloth import bootstrap failed before trainer initialization: {_UNSLOTH_IMPORT_ERROR}")
        preloaded = [name for name, flag in _UNSLOTH_PRELOADED_CRITICAL.items() if int(flag) == 1]
        if preloaded:
            raise RuntimeError(
                "unsloth was imported after critical modules already loaded: "
                + ", ".join(preloaded)
                + ". This disables real unsloth patching."
            )


def _resolve_startup_meta_real_data_buffer_steps(args: argparse.Namespace) -> int:
    configured = int(getattr(args, "startup_meta_real_data_buffer_steps", -1))
    if configured >= 0:
        return configured
    # Auto mode: wait at least one baseline TTRL interval after bootstrap.
    return max(1, int(getattr(args, "ttrl_interval", 1)))


def _resolve_meta_activation_step(args: argparse.Namespace) -> int:
    bootstrap_batches = max(0, int(getattr(args, "bootstrap_seed_batches", 0)))
    real_data_buffer = max(0, _resolve_startup_meta_real_data_buffer_steps(args))
    return max(0, bootstrap_batches + real_data_buffer)


def _normalize_startup_schedule(args: argparse.Namespace) -> None:
    args.startup_meta_real_data_buffer_steps = _resolve_startup_meta_real_data_buffer_steps(args)


def _detect_unsloth_trl_patch_state() -> dict[str, int]:
    state = {
        "unsloth_trl_patch_targets": 0,
        "unsloth_trl_patch_hits": 0,
        "unsloth_trl_patch_active": 0,
    }
    if not _UNSLOTH_IMPORTED:
        return state
    try:
        trl = importlib.import_module("trl")
    except Exception:
        return state

    probe_classes = (
        "GRPOTrainer",
        "PPOTrainer",
        "SFTTrainer",
        "DPOTrainer",
        "ORPOTrainer",
        "RewardTrainer",
    )
    for name in probe_classes:
        cls = getattr(trl, name, None)
        if cls is None:
            continue
        state["unsloth_trl_patch_targets"] += 1
        cls_name = str(getattr(cls, "__name__", ""))
        if cls_name.startswith("Unsloth"):
            state["unsloth_trl_patch_hits"] += 1
    state["unsloth_trl_patch_active"] = int(state["unsloth_trl_patch_hits"] > 0)
    return state


def _validate_strict_feature_schedule(args: argparse.Namespace) -> None:
    if not bool(args.strict_feature_usage):
        return

    steps = max(1, int(args.steps))
    required_steps: dict[str, int] = {}
    meta_activation_step = _resolve_meta_activation_step(args)

    ttrl_interval = max(1, int(args.ttrl_interval))
    ramp_steps = max(0, int(args.startup_meta_ramp_steps))
    ramp_mult = max(1, int(args.startup_ttrl_interval_multiplier))
    ttrl_effective = ttrl_interval if (ramp_steps <= 0 or steps > ramp_steps) else (ttrl_interval * ramp_mult)

    if bool(args.verifier_math_enabled or args.verifier_code_enabled):
        required_steps["ttrl/verifier/sigma_rl"] = meta_activation_step + ttrl_effective
    if bool(args.ttrl_asym_verify_enabled):
        required_steps["ttrl_asym_verify"] = meta_activation_step + ttrl_effective
    if bool(args.sigma_rl_auto_mix_enabled):
        required_steps["sigma_rl_auto_mix"] = meta_activation_step + ttrl_effective
    if bool(args.verifier_cascade_enabled):
        required_steps["verifier_cascade"] = meta_activation_step + ttrl_effective
    if bool(args.fractal_interval_steps > 0):
        required_steps["fractal_nas"] = meta_activation_step + max(1, int(args.fractal_interval_steps))
    if bool(args.uroboros_enabled):
        required_steps["uroboros"] = meta_activation_step + max(1, int(args.uroboros_interval))
    if bool(args.hydra_enable):
        required_steps["hydra_v21"] = meta_activation_step + max(1, int(args.hydra_update_interval))
    if bool(args.self_improver_enabled):
        required_steps["self_improver"] = meta_activation_step + max(1, int(args.self_improver_interval))

    missing = {name: need for name, need in required_steps.items() if steps < need}
    if missing:
        details = ", ".join(f"{k}:>={v}" for k, v in sorted(missing.items()))
        raise RuntimeError(
            "Strict feature validation is enabled, but --steps is too small to trigger all enabled features. "
            f"Current steps={steps}. Required: {details}. Increase --steps or disable corresponding features."
        )


def parse_args() -> argparse.Namespace:
    p = _ArgsFileParser(
        description="Train SIGMA stack (Mamba-3 MIMO + INSTANT + GN-Prox + TTRL).",
        fromfile_prefix_chars="@",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="runs/sigma")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--max-checkpoints", type=int, default=5)
    p.add_argument("--no-resume", action="store_true")

    p.add_argument("--tokenizer-backend", choices=["tiktoken", "hf"], default="tiktoken")
    p.add_argument("--tokenizer-name", type=str, default="cl100k_base")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--auto-hardware-profile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--synthetic-data", action="store_true")

    p.add_argument("--data-backend", choices=["hf", "commoncrawl", "hybrid"], default="hf")
    p.add_argument("--fineweb-ratio", type=float, default=0.7)
    p.add_argument("--fineweb-min-score", type=float, default=4.0)
    p.add_argument("--hf-token", type=str, default="")
    p.add_argument("--hf-token-env-var", type=str, default="HF_TOKEN")
    p.add_argument("--stream-sample-prefetch", type=int, default=2048)
    p.add_argument("--source-prefetch", type=int, default=128)
    p.add_argument("--source-read-timeout-sec", type=float, default=2.5)
    p.add_argument("--startup-hf-only-samples", type=int, default=1024)
    p.add_argument("--bootstrap-seed-batches", type=int, default=64)
    p.add_argument("--stream-tokenize-batch-size", type=int, default=64)
    p.add_argument("--stream-shuffle-buffer", type=int, default=20000)
    p.add_argument("--dedup-window-size", type=int, default=65536)
    p.add_argument("--dedup-normalize-chars", type=int, default=4096)
    p.add_argument("--commoncrawl-ratio", type=float, default=0.6)
    p.add_argument("--commoncrawl-latest-crawls", type=int, default=2)
    p.add_argument("--commoncrawl-wet-paths-per-crawl", type=int, default=256)
    p.add_argument("--commoncrawl-records-per-file", type=int, default=256)
    p.add_argument("--commoncrawl-max-files-per-worker-cycle", type=int, default=64)
    p.add_argument("--commoncrawl-parallel-files", type=int, default=1)

    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=18)
    p.add_argument("--kv-latent-dim", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--n-shared-experts", type=int, default=1)
    p.add_argument("--n-routed-experts", type=int, default=8)
    p.add_argument("--moe-top-k", type=int, default=2)
    p.add_argument("--moe-dispatch-mode", choices=["legacy", "packed", "grouped"], default="grouped")
    p.add_argument("--router-balance-lr", type=float, default=1e-3)
    p.add_argument("--router-metrics-interval", type=int, default=10)
    p.add_argument("--router-jitter-noise", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--mamba-ratio", type=int, default=5)
    p.add_argument("--attention-ratio", type=int, default=1)
    p.add_argument("--mimo-rank", type=int, default=4)
    p.add_argument("--mamba-block-t", type=int, default=32)
    p.add_argument("--mamba-block-n", type=int, default=128)
    p.add_argument("--mamba-step-scale", type=float, default=0.05)

    p.add_argument("--instant-comp-dim", type=int, default=64)
    p.add_argument("--instant-error-threshold", type=float, default=0.01)
    p.add_argument("--instant-reversible-iters", type=int, default=8)

    p.add_argument("--optimizer", choices=["muon", "normuon", "adamuon", "c3o", "gnprox"], default="c3o")
    p.add_argument("--muon-strict-split", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--muon-exclude-embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--muon-exclude-lm-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--muon-max-orthogonalized-dim", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--quant-mode", choices=["bitnet_1_58"], default="bitnet_1_58")
    p.add_argument("--moe-gate-precision", choices=["bf16", "int8"], default="bf16")
    p.add_argument("--grpo-backend", choices=["trl", "internal"], default="trl")
    p.add_argument("--grpo-num-generations", type=int, default=8)
    p.add_argument("--grpo-code-timeout-sec", type=float, default=4.0)
    p.add_argument("--grpo-reward-success", type=float, default=2.0)
    p.add_argument("--grpo-reward-fail", type=float, default=-1.0)

    p.add_argument("--gn-beta1", type=float, default=0.9)
    p.add_argument("--gn-beta2", type=float, default=0.90)
    p.add_argument("--gn-eps", type=float, default=1e-8)
    p.add_argument("--gn-damping", type=float, default=1e-4)
    p.add_argument("--gn-damping-min", type=float, default=1e-4)
    p.add_argument("--gn-damping-max", type=float, default=1.0)
    p.add_argument("--gn-damping-up", type=float, default=1.08)
    p.add_argument("--gn-damping-down", type=float, default=0.97)
    p.add_argument("--gn-damping-gain", type=float, default=0.02)
    p.add_argument("--gn-ns-steps", type=int, default=5)
    p.add_argument("--gn-block-size", type=int, default=1024)
    p.add_argument("--gn-clip-grad", type=float, default=0.0)
    p.add_argument("--c3o-beta1", type=float, default=0.9)
    p.add_argument("--c3o-beta2", type=float, default=0.98)
    p.add_argument("--c3o-eps", type=float, default=1e-8)
    p.add_argument("--c3o-damping", type=float, default=1e-4)
    p.add_argument("--c3o-grad-clip", type=float, default=1.0)
    p.add_argument("--c3o-block-size", type=int, default=4096)
    p.add_argument("--c3o-credit-ema", type=float, default=0.92)
    p.add_argument("--c3o-credit-gain", type=float, default=0.40)
    p.add_argument("--c3o-credit-skip-threshold", type=float, default=1e-2)
    p.add_argument("--c3o-block-scale-min", type=float, default=0.60)
    p.add_argument("--c3o-block-scale-max", type=float, default=1.80)
    p.add_argument("--c3o-trust-radius", type=float, default=0.15)
    p.add_argument("--c3o-trust-norm-refresh-steps", type=int, default=16)
    p.add_argument("--c3o-trust-norm-refresh-drift", type=float, default=0.02)
    p.add_argument("--c3o-foreach-fused", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--c3o-state-dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")

    p.add_argument("--ttrl-group-size", type=int, default=5)
    p.add_argument("--ttrl-interval", type=int, default=8)
    p.add_argument("--ttrl-max-new-tokens", type=int, default=24)
    p.add_argument("--ttrl-temperature", type=float, default=0.8)
    p.add_argument("--ttrl-top-k", type=int, default=40)
    p.add_argument("--ttrl-refine-iters", type=int, default=3)
    p.add_argument("--ttrl-retry-temperature-decay", type=float, default=0.75)
    p.add_argument("--ttrl-retry-top-k-decay", type=float, default=0.85)
    p.add_argument("--ttrl-budget-min", type=int, default=1)
    p.add_argument("--ttrl-budget-max", type=int, default=4)
    p.add_argument("--ttrl-asym-verify-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ttrl-candidate-pool-multiplier", type=float, default=1.8)
    p.add_argument("--ttrl-discriminative-topk", type=int, default=8)
    p.add_argument("--ttrl-discriminative-weight", type=float, default=0.35)
    p.add_argument("--ttrl-fast-refine-on-fail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ttrl-confidence-replay-min", type=float, default=0.20)
    p.add_argument("--replay-capacity", type=int, default=512)
    p.add_argument("--replay-inject-every", type=int, default=2)
    p.add_argument("--sigma-rl-mode", choices=["igrpo", "dispo", "gspo", "cispo"], default="cispo")
    p.add_argument("--sigma-rl-clip-eps", type=float, default=0.2)
    p.add_argument("--sigma-rl-entropy-weight", type=float, default=0.001)
    p.add_argument("--sigma-rl-kl-weight", type=float, default=0.02)
    p.add_argument("--sigma-rl-dispo-logit-temp", type=float, default=1.0)
    p.add_argument("--no-sigma-rl-adv-norm", action="store_true")
    p.add_argument("--sigma-rl-auto-mix-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sigma-rl-auto-mix-floor", type=float, default=0.10)
    p.add_argument("--verifier-math-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verifier-code-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verifier-timeout-ms", type=int, default=2500)
    p.add_argument("--verifier-eval-every", type=int, default=1)
    p.add_argument("--verifier-cascade-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verifier-refine-top-fraction", type=float, default=0.5)
    p.add_argument("--verifier-min-refine-candidates", type=int, default=2)
    p.add_argument("--causal-replay-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--causal-replay-alpha", type=float, default=0.20)
    p.add_argument("--causal-replay-ema-beta", type=float, default=0.95)
    p.add_argument("--causal-replay-novelty-weight", type=float, default=0.20)
    p.add_argument("--causal-replay-verifier-weight", type=float, default=0.60)
    p.add_argument("--causal-replay-quality-weight", type=float, default=0.20)
    p.add_argument("--causal-replay-horizon-steps", type=int, default=6)
    p.add_argument("--causal-replay-horizon-decay", type=float, default=0.90)
    p.add_argument("--c3o-credit-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--c3o-credit-ema-beta", type=float, default=0.92)
    p.add_argument("--c3o-credit-loss-weight", type=float, default=0.45)
    p.add_argument("--c3o-credit-verifier-weight", type=float, default=0.35)
    p.add_argument("--c3o-credit-speed-weight", type=float, default=0.20)
    p.add_argument("--c3o-credit-clip-value", type=float, default=2.0)
    p.add_argument("--self-evolve-interval", type=int, default=4)
    p.add_argument("--self-evolve-history", type=int, default=24)
    p.add_argument("--self-evolve-lr-up", type=float, default=1.03)
    p.add_argument("--self-evolve-lr-down", type=float, default=0.94)
    p.add_argument("--self-evolve-lr-min-factor", type=float, default=0.50)
    p.add_argument("--self-evolve-lr-max-factor", type=float, default=1.80)
    p.add_argument("--self-evolve-ttrl-interval-min", type=int, default=2)
    p.add_argument("--self-evolve-ttrl-interval-max", type=int, default=16)
    p.add_argument("--self-evolve-target-reserved-gb", type=float, default=10.5)
    p.add_argument("--self-improver-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-improver-interval", type=int, default=2)
    p.add_argument("--self-improver-mutation-sigma", type=float, default=0.08)
    p.add_argument("--self-improver-frontier-alpha", type=float, default=0.6)
    p.add_argument("--self-improver-adaptive-sigma-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self-improver-sigma-ema-beta", type=float, default=0.90)
    p.add_argument("--self-improver-sigma-gain-up", type=float, default=1.06)
    p.add_argument("--self-improver-sigma-gain-down", type=float, default=0.94)
    p.add_argument("--self-improver-sigma-min", type=float, default=0.02)
    p.add_argument("--self-improver-sigma-max", type=float, default=0.30)
    p.add_argument("--integrity-guards-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--proof-mode", action="store_true")
    p.add_argument("--strict-runtime-audit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-feature-usage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-unsloth-trl-patch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--integrity-hidden-eval-manifest", type=str, default="")
    p.add_argument("--integrity-hidden-eval-every", type=int, default=2)
    p.add_argument("--metrics-ema-beta", type=float, default=0.90)
    p.add_argument("--crash-report-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--first-batch-timeout-sec", type=float, default=240.0)
    p.add_argument("--startup-meta-ramp-steps", type=int, default=64)
    p.add_argument("--startup-ttrl-interval-multiplier", type=int, default=4)
    p.add_argument("--startup-meta-real-data-buffer-steps", type=int, default=-1)
    p.add_argument("--perf-profile-enable", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--perf-warmup-steps", type=int, default=20)
    p.add_argument("--perf-measure-steps", type=int, default=200)
    p.add_argument("--perf-profiler-steps", type=int, default=50)
    p.add_argument("--perf-report-dir", type=str, default="")
    p.add_argument("--perf-device-time-mode", choices=["auto", "cuda", "device", "cpu"], default="auto")
    p.add_argument("--perf-sync-boundaries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-compile-backend", type=str, default="inductor")
    p.add_argument("--torch-compile-mode", type=str, default="default")
    p.add_argument("--torch-compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-compile-fullgraph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--diff-attention-lambda-init", type=float, default=0.35)
    p.add_argument("--diff-attention-lambda-min", type=float, default=0.05)
    p.add_argument("--diff-attention-lambda-max", type=float, default=1.20)
    p.add_argument("--rib-router-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rib-info-weight", type=float, default=0.03)
    p.add_argument("--rib-collapse-penalty", type=float, default=0.08)
    p.add_argument("--rib-confidence-weight", type=float, default=0.03)
    p.add_argument("--rib-temp-gain", type=float, default=0.55)

    p.add_argument("--fractal-interval-steps", type=int, default=50)
    p.add_argument("--fractal-ucb-c", type=float, default=1.3)
    p.add_argument("--uroboros-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uroboros-interval", type=int, default=10)
    p.add_argument("--uroboros-window-size", type=int, default=64)
    p.add_argument("--uroboros-bo-trials", type=int, default=8)
    p.add_argument("--uroboros-patch-trials", type=int, default=3)
    p.add_argument("--uroboros-trial-horizon-steps", type=int, default=12)
    p.add_argument("--uroboros-significance-alpha", type=float, default=0.05)
    p.add_argument("--uroboros-min-effect-size", type=float, default=0.15)
    p.add_argument("--uroboros-min-relative-gain", type=float, default=0.15)
    p.add_argument("--replacement-policy-strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uroboros-patch-commit-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--massive-improvement-enforced", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--forward-research-loop-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hydra-enable", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hydra-domains", type=str, default="math,code,text")
    p.add_argument("--hydra-steps-per-phase", type=int, default=5000)
    p.add_argument("--hydra-n-candidates", type=int, default=4)
    p.add_argument("--hydra-unverified-cap-text", type=float, default=0.20)
    p.add_argument("--hydra-rollback-interval", type=int, default=200)
    p.add_argument("--hydra-rollback-threshold", type=float, default=0.01)
    p.add_argument("--hydra-update-interval", type=int, default=16)
    p.add_argument("--hydra-dpo-beta", type=float, default=0.1)
    p.add_argument("--hydra-sampo-verbosity-weight", type=float, default=0.02)
    p.add_argument("--hydra-lora-rank", type=int, default=8)
    p.add_argument("--hydra-lora-alpha", type=float, default=16.0)
    p.add_argument("--hydra-lora-lr", type=float, default=1e-4)
    p.add_argument("--merge-method", choices=["ties", "dare_ties"], default="ties")
    p.add_argument("--merge-density", type=float, default=1.0)
    p.add_argument("--merge-fold-into-backbone", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    from model.sigma_llm import SigmaConfig, SigmaLLM
    from training.data import StreamingDataConfig, build_streaming_dataloader, build_tokenizer
    from training.optimizer import OptimizerConfig, build_optimizer
    from training.sigma_trainer import SigmaTrainConfig, SigmaTrainer

    _configure_runtime_caches(args.output_dir)
    _normalize_startup_schedule(args)
    os.environ.setdefault("TORCH_NVCC_FLAGS", "--allow-unsupported-compiler")
    os.environ.setdefault("CUDAFLAGS", "--allow-unsupported-compiler")
    os.environ.setdefault("CMAKE_CUDA_FLAGS", "--allow-unsupported-compiler")
    toolchain_ready = _bootstrap_vs2026_toolchain_env()
    print(f"[sigma] vs2026_toolchain_ready={int(bool(toolchain_ready))}")
    meta_activation_step = _resolve_meta_activation_step(args)
    print(
        "[sigma] meta_schedule "
        f"bootstrap_seed_batches={int(args.bootstrap_seed_batches)} "
        f"real_data_buffer_steps={int(args.startup_meta_real_data_buffer_steps)} "
        f"meta_activation_step={int(meta_activation_step)}"
    )
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("SIGMA training requires CUDA but CUDA is unavailable.")
    if not args.device.startswith("cuda"):
        raise RuntimeError("SIGMA training requires --device cuda.")
    _audit_enabled_dependencies(args)
    _validate_strict_feature_schedule(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    intra_threads, interop_threads = _configure_runtime_threads()
    workers, prefetch = _apply_auto_hardware_profile(args)
    print(
        "[sigma] hardware_profile "
        f"auto={int(bool(args.auto_hardware_profile))} "
        f"threads_intra={intra_threads} threads_interop={interop_threads} "
        f"num_workers={workers} prefetch_factor={prefetch} "
        f"persistent_workers={int(bool(args.persistent_workers))} "
        f"commoncrawl_parallel_files={int(args.commoncrawl_parallel_files)}"
    )

    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        os.environ[args.hf_token_env_var] = args.hf_token

    tokenizer = build_tokenizer(args.tokenizer_backend, args.tokenizer_name)
    data_cfg = StreamingDataConfig(
        data_backend=args.data_backend,
        fineweb_ratio=args.fineweb_ratio,
        min_score=args.fineweb_min_score,
        hf_token=args.hf_token if args.hf_token else None,
        hf_token_env_var=args.hf_token_env_var,
        sample_prefetch=args.stream_sample_prefetch,
        source_prefetch=args.source_prefetch,
        source_read_timeout_sec=args.source_read_timeout_sec,
        tokenize_batch_size=args.stream_tokenize_batch_size,
        stream_shuffle_buffer=args.stream_shuffle_buffer,
        bootstrap_seed_batches=args.bootstrap_seed_batches,
        dedup_window_size=args.dedup_window_size,
        dedup_normalize_chars=args.dedup_normalize_chars,
        hybrid_warmstart_hf_samples=args.startup_hf_only_samples,
        commoncrawl_ratio=args.commoncrawl_ratio,
        commoncrawl_latest_crawls=args.commoncrawl_latest_crawls,
        commoncrawl_wet_paths_per_crawl=args.commoncrawl_wet_paths_per_crawl,
        commoncrawl_records_per_file=args.commoncrawl_records_per_file,
        commoncrawl_max_files_per_worker_cycle=args.commoncrawl_max_files_per_worker_cycle,
        commoncrawl_parallel_files=args.commoncrawl_parallel_files,
        allow_synthetic_fallback=False,
        synthetic_backfill_on_error=False,
        synthetic_mix_ratio=0.0,
    )
    dataloader = build_streaming_dataloader(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        data_config=data_cfg,
        force_synthetic=bool(args.synthetic_data),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=prefetch,
    )
    probe_iter = iter(dataloader)
    del probe_iter

    sigma_cfg = SigmaConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        kv_latent_dim=args.kv_latent_dim,
        d_ff=args.d_ff,
        n_shared_experts=args.n_shared_experts,
        n_routed_experts=args.n_routed_experts,
        moe_top_k=args.moe_top_k,
        moe_dispatch_mode=args.moe_dispatch_mode,
        router_balance_lr=args.router_balance_lr,
        router_metrics_interval=args.router_metrics_interval,
        router_jitter_noise=args.router_jitter_noise,
        router_precision=args.moe_gate_precision,
        rib_router_enabled=args.rib_router_enabled,
        rib_info_weight=args.rib_info_weight,
        rib_collapse_penalty=args.rib_collapse_penalty,
        rib_confidence_weight=args.rib_confidence_weight,
        rib_temp_gain=args.rib_temp_gain,
        use_triton_bitlinear=True,
        bitlinear_quant_cache_training=True,
        mamba_ratio=args.mamba_ratio,
        attention_ratio=args.attention_ratio,
        mimo_rank=args.mimo_rank,
        mamba_block_t=args.mamba_block_t,
        mamba_block_n=args.mamba_block_n,
        mamba_step_scale=args.mamba_step_scale,
        instant_enabled=True,
        instant_comp_dim=args.instant_comp_dim,
        instant_error_threshold=args.instant_error_threshold,
        instant_reversible_iters=args.instant_reversible_iters,
        ttrl_group_size=args.ttrl_group_size,
        ttrl_interval=args.ttrl_interval,
        ttrl_refine_iters=args.ttrl_refine_iters,
        diff_attention_lambda_init=args.diff_attention_lambda_init,
        diff_attention_lambda_min=args.diff_attention_lambda_min,
        diff_attention_lambda_max=args.diff_attention_lambda_max,
    )
    model = SigmaLLM(sigma_cfg)
    model_dtype = _resolve_model_param_dtype(args)
    if model_dtype is not None:
        model.to(device=torch.device(args.device), dtype=model_dtype)
    else:
        model.to(torch.device(args.device))
    print(f"[sigma] parameters={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"[sigma] model_param_dtype={next(model.parameters()).dtype}")
    print(f"[sigma] algorithmic_features={model.algorithmic_features()}")
    algo_features = model.algorithmic_features()
    bitnet_active = float(algo_features.get("bitnet_1_58_active", 0.0))
    print(
        "[sigma] quantization "
        f"mode={args.quant_mode} bitnet_1_58_active={bitnet_active:.0f} "
        f"moe_gate_precision={args.moe_gate_precision}"
    )
    print(
        "[sigma] grpo "
        f"backend={args.grpo_backend} num_generations={int(args.grpo_num_generations)} "
        f"reward_success={float(args.grpo_reward_success):.3f} reward_fail={float(args.grpo_reward_fail):.3f}"
    )
    print(
        "[sigma] unsloth_bootstrap "
        f"imported={int(_UNSLOTH_IMPORTED)} "
        f"preloaded_trl={int(_UNSLOTH_PRELOADED_CRITICAL.get('trl', 0))} "
        f"preloaded_transformers={int(_UNSLOTH_PRELOADED_CRITICAL.get('transformers', 0))} "
        f"preloaded_peft={int(_UNSLOTH_PRELOADED_CRITICAL.get('peft', 0))}"
    )
    if _UNSLOTH_IMPORT_ERROR:
        print(f"[sigma] unsloth_bootstrap_error={_UNSLOTH_IMPORT_ERROR}")
    trl_grpo_available = 0
    trl_grpo_error = ""
    try:
        from trl import GRPOConfig as _TRLGRPOConfig, GRPOTrainer as _TRLGRPOTrainer  # type: ignore
        del _TRLGRPOConfig, _TRLGRPOTrainer
        trl_grpo_available = 1
    except Exception as exc:
        trl_grpo_available = 0
        trl_grpo_error = f"{type(exc).__name__}: {exc}"
    print(f"[sigma] trl_grpo_available={trl_grpo_available}")
    if trl_grpo_error:
        print(f"[sigma] trl_grpo_error={trl_grpo_error}")
    if str(args.grpo_backend).strip().lower() == "trl" and trl_grpo_available == 0:
        raise RuntimeError("GRPO backend is set to 'trl' but TRL GRPO classes are unavailable.")
    if str(args.grpo_backend).strip().lower() == "trl":
        from training.grpo_trl_bridge import TRLGRPOBridge, TRLGRPOBridgeConfig

        with TRLGRPOBridge(
            model=model,
            tokenizer=tokenizer,
            config=TRLGRPOBridgeConfig(
                num_generations=int(args.grpo_num_generations),
                code_timeout_sec=float(args.grpo_code_timeout_sec),
                success_reward=float(args.grpo_reward_success),
                fail_reward=float(args.grpo_reward_fail),
            ),
        ) as _grpo_bridge:
            _probe = _grpo_bridge.reward_func(prompts=["probe"], completions=["print(1)"])
            if not _probe or float(_probe[0]) != float(args.grpo_reward_success):
                raise RuntimeError("TRL GRPO bridge self-check failed: sandbox reward mismatch.")
        print("[sigma] trl_grpo_bridge_ready=1")
    unsloth_patch_state = _detect_unsloth_trl_patch_state()
    print(
        "[sigma] unsloth_trl_patch "
        f"active={int(unsloth_patch_state['unsloth_trl_patch_active'])} "
        f"hits={int(unsloth_patch_state['unsloth_trl_patch_hits'])} "
        f"targets={int(unsloth_patch_state['unsloth_trl_patch_targets'])}"
    )
    if (
        str(args.grpo_backend).strip().lower() == "trl"
        and bool(args.require_unsloth_trl_patch)
        and int(unsloth_patch_state["unsloth_trl_patch_active"]) == 0
    ):
        raise RuntimeError(
            "unsloth TRL patch is required but not active. "
            "Fix packaging/import order so TRL trainers are actually patched by unsloth."
        )

    optim_cfg = OptimizerConfig(
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_strict_split=args.muon_strict_split,
        muon_exclude_embeddings=args.muon_exclude_embeddings,
        muon_exclude_lm_head=args.muon_exclude_lm_head,
        muon_max_orthogonalized_dim=args.muon_max_orthogonalized_dim,
        gn_beta1=args.gn_beta1,
        gn_beta2=args.gn_beta2,
        gn_eps=args.gn_eps,
        gn_damping=args.gn_damping,
        gn_damping_min=args.gn_damping_min,
        gn_damping_max=args.gn_damping_max,
        gn_damping_up=args.gn_damping_up,
        gn_damping_down=args.gn_damping_down,
        gn_damping_gain=args.gn_damping_gain,
        gn_ns_steps=args.gn_ns_steps,
        gn_block_size=args.gn_block_size,
        gn_clip_grad=args.gn_clip_grad,
        c3o_beta1=args.c3o_beta1,
        c3o_beta2=args.c3o_beta2,
        c3o_eps=args.c3o_eps,
        c3o_damping=args.c3o_damping,
        c3o_grad_clip=args.c3o_grad_clip,
        c3o_block_size=args.c3o_block_size,
        c3o_credit_ema=args.c3o_credit_ema,
        c3o_credit_gain=args.c3o_credit_gain,
        c3o_credit_skip_threshold=args.c3o_credit_skip_threshold,
        c3o_block_scale_min=args.c3o_block_scale_min,
        c3o_block_scale_max=args.c3o_block_scale_max,
        c3o_trust_radius=args.c3o_trust_radius,
        c3o_trust_norm_refresh_steps=args.c3o_trust_norm_refresh_steps,
        c3o_trust_norm_refresh_drift=args.c3o_trust_norm_refresh_drift,
        c3o_foreach_fused=args.c3o_foreach_fused,
        c3o_state_dtype=args.c3o_state_dtype,
    )
    optimizer = build_optimizer(model, optim_cfg, device=args.device)

    trainer_cfg = SigmaTrainConfig(
        device=args.device,
        steps=args.steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        mixed_precision=not args.no_amp,
        precision=args.precision,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        max_checkpoints=args.max_checkpoints,
        resume_latest=not args.no_resume,
        ttrl_group_size=args.ttrl_group_size,
        ttrl_interval=args.ttrl_interval,
        ttrl_max_new_tokens=args.ttrl_max_new_tokens,
        ttrl_temperature=args.ttrl_temperature,
        ttrl_top_k=args.ttrl_top_k,
        ttrl_refine_iters=args.ttrl_refine_iters,
        ttrl_retry_temperature_decay=args.ttrl_retry_temperature_decay,
        ttrl_retry_top_k_decay=args.ttrl_retry_top_k_decay,
        ttrl_budget_min=args.ttrl_budget_min,
        ttrl_budget_max=args.ttrl_budget_max,
        ttrl_asym_verify_enabled=args.ttrl_asym_verify_enabled,
        ttrl_candidate_pool_multiplier=args.ttrl_candidate_pool_multiplier,
        ttrl_discriminative_topk=args.ttrl_discriminative_topk,
        ttrl_discriminative_weight=args.ttrl_discriminative_weight,
        ttrl_fast_refine_on_fail=args.ttrl_fast_refine_on_fail,
        ttrl_confidence_replay_min=args.ttrl_confidence_replay_min,
        sigma_rl_mode=args.sigma_rl_mode,
        sigma_rl_clip_eps=args.sigma_rl_clip_eps,
        sigma_rl_entropy_weight=args.sigma_rl_entropy_weight,
        sigma_rl_adv_norm=(not args.no_sigma_rl_adv_norm),
        sigma_rl_kl_weight=args.sigma_rl_kl_weight,
        sigma_rl_dispo_logit_temp=args.sigma_rl_dispo_logit_temp,
        sigma_rl_auto_mix_enabled=args.sigma_rl_auto_mix_enabled,
        sigma_rl_auto_mix_floor=args.sigma_rl_auto_mix_floor,
        verifier_math_enabled=args.verifier_math_enabled,
        verifier_code_enabled=args.verifier_code_enabled,
        verifier_timeout_ms=args.verifier_timeout_ms,
        verifier_eval_every=args.verifier_eval_every,
        verifier_cascade_enabled=args.verifier_cascade_enabled,
        verifier_refine_top_fraction=args.verifier_refine_top_fraction,
        verifier_min_refine_candidates=args.verifier_min_refine_candidates,
        causal_replay_enabled=args.causal_replay_enabled,
        causal_replay_alpha=args.causal_replay_alpha,
        causal_replay_ema_beta=args.causal_replay_ema_beta,
        causal_replay_novelty_weight=args.causal_replay_novelty_weight,
        causal_replay_verifier_weight=args.causal_replay_verifier_weight,
        causal_replay_quality_weight=args.causal_replay_quality_weight,
        causal_replay_horizon_steps=args.causal_replay_horizon_steps,
        causal_replay_horizon_decay=args.causal_replay_horizon_decay,
        replay_capacity=args.replay_capacity,
        replay_inject_every=args.replay_inject_every,
        self_evolve_interval=args.self_evolve_interval,
        self_evolve_history=args.self_evolve_history,
        self_evolve_lr_up=args.self_evolve_lr_up,
        self_evolve_lr_down=args.self_evolve_lr_down,
        self_evolve_lr_min_factor=args.self_evolve_lr_min_factor,
        self_evolve_lr_max_factor=args.self_evolve_lr_max_factor,
        self_evolve_ttrl_interval_min=args.self_evolve_ttrl_interval_min,
        self_evolve_ttrl_interval_max=args.self_evolve_ttrl_interval_max,
        self_evolve_target_reserved_gb=args.self_evolve_target_reserved_gb,
        self_improver_enabled=args.self_improver_enabled,
        self_improver_interval=args.self_improver_interval,
        self_improver_mutation_sigma=args.self_improver_mutation_sigma,
        self_improver_frontier_alpha=args.self_improver_frontier_alpha,
        self_improver_adaptive_sigma_enabled=args.self_improver_adaptive_sigma_enabled,
        self_improver_sigma_ema_beta=args.self_improver_sigma_ema_beta,
        self_improver_sigma_gain_up=args.self_improver_sigma_gain_up,
        self_improver_sigma_gain_down=args.self_improver_sigma_gain_down,
        self_improver_sigma_min=args.self_improver_sigma_min,
        self_improver_sigma_max=args.self_improver_sigma_max,
        integrity_guards_enabled=args.integrity_guards_enabled,
        proof_mode=args.proof_mode,
        strict_feature_usage=args.strict_feature_usage,
        integrity_hidden_eval_manifest=args.integrity_hidden_eval_manifest,
        integrity_hidden_eval_every=args.integrity_hidden_eval_every,
        metrics_ema_beta=args.metrics_ema_beta,
        crash_report_enabled=args.crash_report_enabled,
        first_batch_timeout_sec=args.first_batch_timeout_sec,
        startup_meta_ramp_steps=args.startup_meta_ramp_steps,
        startup_ttrl_interval_multiplier=args.startup_ttrl_interval_multiplier,
        startup_meta_real_data_buffer_steps=args.startup_meta_real_data_buffer_steps,
        bootstrap_seed_batches=args.bootstrap_seed_batches,
        perf_profile_enable=args.perf_profile_enable,
        perf_warmup_steps=args.perf_warmup_steps,
        perf_measure_steps=args.perf_measure_steps,
        perf_profiler_steps=args.perf_profiler_steps,
        perf_report_dir=args.perf_report_dir,
        perf_device_time_mode=args.perf_device_time_mode,
        perf_sync_boundaries=args.perf_sync_boundaries,
        torch_compile=args.torch_compile,
        torch_compile_backend=args.torch_compile_backend,
        torch_compile_mode=args.torch_compile_mode,
        torch_compile_dynamic=args.torch_compile_dynamic,
        torch_compile_fullgraph=args.torch_compile_fullgraph,
        fractal_interval_steps=args.fractal_interval_steps,
        fractal_ucb_c=args.fractal_ucb_c,
        uroboros_enabled=args.uroboros_enabled,
        uroboros_interval=args.uroboros_interval,
        uroboros_window_size=args.uroboros_window_size,
        uroboros_bo_trials=args.uroboros_bo_trials,
        uroboros_patch_trials=args.uroboros_patch_trials,
        uroboros_trial_horizon_steps=args.uroboros_trial_horizon_steps,
        uroboros_significance_alpha=args.uroboros_significance_alpha,
        uroboros_min_effect_size=args.uroboros_min_effect_size,
        uroboros_min_relative_gain=args.uroboros_min_relative_gain,
        replacement_policy_strict=args.replacement_policy_strict,
        uroboros_patch_commit_enabled=args.uroboros_patch_commit_enabled,
        c3o_credit_enabled=args.c3o_credit_enabled,
        c3o_credit_ema_beta=args.c3o_credit_ema_beta,
        c3o_credit_loss_weight=args.c3o_credit_loss_weight,
        c3o_credit_verifier_weight=args.c3o_credit_verifier_weight,
        c3o_credit_speed_weight=args.c3o_credit_speed_weight,
        c3o_credit_clip_value=args.c3o_credit_clip_value,
        massive_improvement_enforced=args.massive_improvement_enforced,
        forward_research_loop_enabled=args.forward_research_loop_enabled,
        hydra_enable=args.hydra_enable,
        hydra_domains=tuple([d.strip() for d in args.hydra_domains.split(",") if d.strip()]),
        hydra_steps_per_phase=args.hydra_steps_per_phase,
        hydra_n_candidates=args.hydra_n_candidates,
        hydra_unverified_cap_text=args.hydra_unverified_cap_text,
        hydra_rollback_interval=args.hydra_rollback_interval,
        hydra_rollback_threshold=args.hydra_rollback_threshold,
        hydra_update_interval=args.hydra_update_interval,
        hydra_dpo_beta=args.hydra_dpo_beta,
        hydra_sampo_verbosity_weight=args.hydra_sampo_verbosity_weight,
        hydra_lora_rank=args.hydra_lora_rank,
        hydra_lora_alpha=args.hydra_lora_alpha,
        hydra_lora_lr=args.hydra_lora_lr,
        merge_method=args.merge_method,
        merge_density=args.merge_density,
        merge_fold_into_backbone=args.merge_fold_into_backbone,
        unsloth_bootstrap_imported=bool(_UNSLOTH_IMPORTED),
        unsloth_preloaded_trl=bool(_UNSLOTH_PRELOADED_CRITICAL.get("trl", 0)),
        unsloth_preloaded_transformers=bool(_UNSLOTH_PRELOADED_CRITICAL.get("transformers", 0)),
        unsloth_preloaded_peft=bool(_UNSLOTH_PRELOADED_CRITICAL.get("peft", 0)),
        unsloth_trl_patch_active=bool(unsloth_patch_state["unsloth_trl_patch_active"]),
        unsloth_trl_patch_hits=int(unsloth_patch_state["unsloth_trl_patch_hits"]),
        unsloth_trl_patch_targets=int(unsloth_patch_state["unsloth_trl_patch_targets"]),
    )
    trainer = SigmaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tokenizer=tokenizer,
        config=trainer_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    mp.freeze_support()
    main()
