# SIGMA EXE-First Training Project

This project now runs through one official runtime path only.

## Official Workflow
- Build: `build_exe.bat`
- Run: `eco_train.exe @train_sigma_args.txt`

There are no alternate run launchers in the root workflow.

## Reproducible Dependencies (Strict)
- Dependency lock file: `requirements-py313-cu128.txt`
- Runtime verifier: `scripts/verify_runtime.py`
- `build_exe.bat` now:
  1. upgrades `pip`,
  2. installs pinned dependencies from lock file,
  3. runs runtime import/symbol verification,
  4. fails immediately if any required dependency is missing/broken.
- Verified stack includes TRL/Unsloth and integration deps: `trl`, `unsloth`, `kfp`, `wandb-workspaces`.

## Parameter Update Flow
1. Edit `train_sigma_args.txt`.
2. Rebuild if code changed: `build_exe.bat`.
3. Run: `eco_train.exe @train_sigma_args.txt`.

CLI overrides still work:
```bat
eco_train.exe @train_sigma_args.txt --steps 500 --output-dir runs\sigma_main
```

## Default Performance Profile (Enabled)
- Model parameter dtype auto-selects `bf16` on supported CUDA GPUs.
- C3O state dtype defaults to `bf16` in auto hardware profile (`--c3o-state-dtype bf16` in args file).
- This is enabled by default and used by the EXE runtime.

## Default Training Interfaces
- Optimizer default: `muon` with strict orthogonal split enabled.
- GRPO backend default: `trl` (`GRPOConfig` + `GRPOTrainer` required).
- Reward constants are configured for executable-code GRPO path:
  - success: `+2.0`
  - fail/timeout: `-1.0`

## CUDA / VS2026 / libtorch
- Build script loads VS2026 toolchain from:
  - `C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`
- Runtime also tries automatic toolchain bootstrap and logs:
  - `[sigma] vs2026_toolchain_ready=1` when successful.
- `--allow-unsupported-compiler` is exported via:
  - `TORCH_NVCC_FLAGS`, `CUDAFLAGS`, `CMAKE_CUDA_FLAGS`
- libtorch lookup order:
  1. `C:\test5\my_eco_llm\libtorch`
  2. `C:\test5\libtorch`

## Why GPU-Only Is Default
GPU+CPU mixed execution is typically slower for this training stack because:
- most hot kernels are CUDA-only and optimized for contiguous GPU execution,
- cross-device synchronization adds stalls,
- extra host-device transfers increase latency,
- CPU-side compute does not accelerate Triton/CUDA kernels.

Default profile therefore keeps training compute on CUDA and uses CPU mainly for lightweight orchestration/data handling.

## Default Data Path
- Default backend in `train_sigma_args.txt`: `hybrid` (HF + CommonCrawl stream mix)
- Interleaved source mix:
  - FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, score-filtered)
  - Cosmopedia (`HuggingFaceTB/cosmopedia`)
  - CommonCrawl WET streaming
- Storage policy:
  - Fully streaming (no mandatory full dataset download to disk).
- Startup anti-stall:
  - `bootstrap_seed_batches` emits deterministic seed batches first so first metrics appear quickly.
  - Then runtime continues on remote streaming data.

## Hydra-V2.1 Runtime
- Enabled in default args (`train_sigma_args.txt`):
  - domain adapters: `math`, `code`, `text`
  - DPO + anti-verbosity regularization
  - rollback gate
  - elastic adapter merge (`ties` by default)
- Implementation:
  - `training/hydra_v21.py`
  - `model/merger.py`

## Output Layout
- Checkpoints: `checkpoints/`
- Main run folder: `runs/sigma_main/`

## Checkpoints
- Automatic resume from latest valid checkpoint.
- Corrupted latest checkpoint is quarantined automatically and previous valid checkpoint is resumed.
- Retention limit enforced (`--max-checkpoints`, default 5).

## Error Handling
Crash reports are saved to:
- `<output_dir>/errors/crash_step_*.json`

Reports include:
- exception type/message/trace,
- step/tokens,
- latest metrics snapshot,
- feature activation/effective-call counters.

## Metrics (Clear/Bottleneck-Oriented)
`metrics.jsonl` includes:
- speed/time:
  - `core_step_time_s`, `effective_step_time_s`
  - `core_tokens_per_s`, `effective_tokens_per_s`, EMA fields
- resource:
  - `gpu_util_percent`, `gpu_mem_alloc_gb`, `gpu_mem_reserved_gb`, `gpu_power_w`
  - `cpu_util_percent`, `ram_used_percent`
- bottleneck diagnostics:
  - `perf_bottleneck_stage_id`, `perf_bottleneck_share`
  - `phase_*_s` stage timings
- quality/integrity:
  - `loss`, `loss_ema`, verifier metrics
  - `feature_*_enabled`, `feature_*_effective_calls`
  - quant/rl startup checks:
    - `bitnet_1_58_active` startup log field
    - `trl_grpo_available` startup log field
  - startup visibility:
    - `startup_init_s`, `time_to_first_batch_s`, `time_to_first_step_s`
    - `checkpoint_resume_step`, `checkpoint_quarantine_count`
    - `instant_inversion_iters`, `instant_inversion_rel_residual`

## Runtime Check (EXE Only)
Use only the EXE runtime path for performance checks.
Example 5-minute run:
```bat
eco_train.exe @train_sigma_args.txt --checkpoint-dir checkpoints --steps 100000000
```

## File Map
- Entry: `train_sigma.py`
- Build: `build_exe.bat`
- Runtime: `eco_train.exe`
- Model stack: `model/`
- Training stack: `training/`
- Evolution stack: `evolution/`
- Algorithms reference: `README_ALGORITHMS.md`
- Future improvement backlog: `IMPROVEMENT_IDEAS_AND_ROADMAP.md`
