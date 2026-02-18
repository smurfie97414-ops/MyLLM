# README_ALGORITHMS

This file is the authoritative list of active algorithms and optimizations in the SIGMA stack.

## Default-On Core Stack
- BitLinear b1.58 projections: `model/bitlinear.py`
- FastLanguageModel strict 1.58-bit loader path (`load_in_1_58bit=True`): `training/fastlanguage_loader.py`
- Mamba-3 MIMO complex Triton scan: `model/sigma_llm.py`, `model/sigma_kernels.py`
- Differential MLA attention: `model/sigma_llm.py`, `model/attention.py`
- DeepSeekMoE + RIB routing: `model/moe.py`, `model/sigma_llm.py`
- MoE grouped dispatch optimization + stage timing metrics: `model/moe.py`
- INSTANT memory protocol: `training/memory_hack.py`
- INSTANT adaptive reversible inversion (early-stop fixed-point): `training/memory_hack.py`
- C3O optimizer: `training/optimizer_c3o.py` (selected by default)
- C3O reduced-memory state dtype (`bf16` default in auto profile): `training/optimizer_c3o.py`, `train_sigma.py`
- TTRL + Sigma RL objectives: `training/sigma_trainer.py`, `training/sigma_rl_objectives.py`
- Adaptive RL Objective Mixer (auto blend of CISPO/DISPO/IGRPO/GSPO): `training/sigma_trainer.py`
- Asymmetric verification: `training/sigma_trainer.py`
- Verifier cascade refine gating (top-fraction refinement): `training/sigma_trainer.py`
- Causal replay prioritization: `training/replay_causal.py`
- Horizon-delayed causal replay credit: `training/replay_causal.py`
- Counterfactual credit signal: `training/counterfactual_credit.py`
- Self-improver loop: `training/sigma_self_improver.py`
- Fractal NAS + UROBOROS: `evolution/fractal_nas.py`, `evolution/meta_loop.py`
- Hydra-V2.1 loop (domain LoRA + DPO + rollback + merge): `training/hydra_v21.py`, `model/merger.py`
- Hydra-V2.1 batched candidate generation + cached DPO ref logprobs: `training/hydra_v21.py`
- Triton BitLinear static launch heuristic (OOM-safe, no autotune bench allocation): `model/triton_kernels.py`
- Integrity guards / anti-fake checks: `training/integrity_guards.py`
- Startup seed curriculum bootstrap (deterministic synthetic warmstart): `training/data.py`
- GRPO bridge with executable code reward (+2/-1) for TRL: `training/grpo_trl_bridge.py`, `training/code_sandbox.py`

## Important Runtime Guarantees
- SIGMA requires CUDA runtime.
- SIGMA requires Triton Mamba scan path.
- INSTANT patching is mandatory and verified at trainer startup.
- Architecture validation now enforces at least one Mamba block and one MLA block.

## Verification Signals
Use startup log + `metrics.jsonl`:
- startup:
  - `algorithmic_features={...}`
  - non-zero `mamba3_layers`, `mla_layers`, `moe_layers`, `bitlinear_layers`
- runtime flags:
  - `feature_instant_enabled == 1`
  - `feature_diff_mla_enabled == 1`
  - `feature_sigma_rl_enabled == 1`
  - `feature_verifier_enabled == 1`
  - `feature_fractal_nas_enabled == 1`
  - `feature_uroboros_enabled == 1`
  - `feature_causal_replay_enabled == 1`
  - `feature_c3o_credit_enabled == 1`
  - `feature_asym_verify_enabled == 1`
  - `feature_verifier_cascade_enabled == 1`
  - `feature_rl_auto_mix_enabled == 1`
  - `feature_hydra_v21_enabled == 1` (when enabled in args)
- effectiveness:
  - each corresponding `feature_*_effective_calls > 0` on real training runs

## Optimization Notes
- GPU-only training compute is default because mixed GPU+CPU compute introduces synchronization and transfer overhead.
- CPU remains relevant for data feeding and orchestration.
- Hardware profile auto-tuning exists but default profile is set for stable CUDA-dominant throughput.
- Model parameter dtype is auto-promoted to `bf16` when supported for memory/throughput efficiency.

## Optimizer Options
- Default: `muon` (strict matrix/vector split)
- Alternative available: `gnprox`
- Alternative available: `c3o`

## Data Path (Storage-Constrained)
- Streaming-first pipeline (`datasets.load_dataset(..., streaming=True)` + CommonCrawl stream path)
- No mandatory full-dataset disk download
- HF token can be supplied by CLI/env for higher limits
- Default EXE profile uses HF interleave (FineWeb + Cosmopedia) for stable startup.
- Default EXE profile uses `hybrid` streaming backend (HF + CommonCrawl) to reduce rate-limit stalls.
- Bootstrap seed batches are enabled by default to avoid first-step stalls before remote streams are hot.

## Checkpoint Safety
- Resume latest valid checkpoint by default
- Corrupted latest checkpoint is quarantined and previous valid checkpoint is resumed automatically
- Retention cap defaults to 5

## Known Bottleneck Axes to Monitor
- verifier cost (`phase_ttrl_s`, `phase_meta_s`)
- model compute (`phase_train_s`, `core_tokens_per_s`)
- input pipeline (`phase_data_s`)
- bottleneck indicators (`perf_bottleneck_stage_id`, `perf_bottleneck_share`)
- startup readiness (`time_to_first_batch_s`, `time_to_first_step_s`)
