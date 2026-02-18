# IMPROVEMENT_IDEAS_AND_ROADMAP

This file is a focused list of high-impact ideas to improve learning quality and learning speed without regressing capability.

## 1) What Is Already Implemented (Do Not Duplicate)
- BitLinear 1.58-style projection path (ternary/STE style).
- Mamba-3 + Differential MLA hybrid stack.
- DeepSeek-style MoE routing with RIB stabilization terms.
- INSTANT reversible/compressed activation pathway.
- C3O optimizer default (+ GN-Prox alternative).
- Hydra-V2.1 (domain LoRA + DPO + rollback + merge).
- TTRL online RL loop with asymmetric verification shortlist.
- Causal replay priority + counterfactual credit signal.
- Self-improver + UROBOROS + fractal NAS loop.
- Integrity counters for enabled/effective feature validation.

## 2) Current Gap Seen in Real EXE Run
Observed in `runs/sigma_main/metrics.jsonl`:
- EXE path is stable and resumes from latest checkpoint (`checkpoint_step_00000003.pt -> step 4` validated).
- Throughput uplift already achieved after latest forward-fix set:
  - `effective_tokens_per_s` step-1: `8.742 -> 21.901` (`+150.5%`)
  - `effective_step_time_s` step-1: `234.279s -> 93.511s` (`-60.1%`)
  - `gpu_mem_alloc_gb` step-1: `21.344GB -> 12.523GB` (`-41.3%`)
- Remaining primary bottleneck is still model compute:
  - `phase_train_s` dominates effective step time (`~99%` in latest run windows).
  - Hydra/TTRL cost can spike on their update steps, but train phase is the sustained bottleneck.

Top priority therefore is now "phase_train_s reduction without quality loss".

## 3) Research-Backed High-Impact Directions (2025-2026)

1. TTT-E2E / Test-Time-Training memory
- Source: https://arxiv.org/abs/2512.23675
- Gap: not yet integrated as strict TTT state-as-weights module in SIGMA training path.
- Potential: better long-horizon adaptation without KV growth.

2. Differential Transformer attention denoising
- Source: https://www.microsoft.com/en-us/research/publication/differential-transformer/
- Status: partially implemented via Differential MLA.
- Next: layerwise lambda scheduling by retrieval difficulty.

3. Kimi Linear hybrid long-context throughput
- Source: https://arxiv.org/abs/2506.05433
- Gap: no KDA-style sparse linear attention block in current stack.
- Potential: better long-context speed/quality tradeoff.

4. Asymmetric Verification for RLVR
- Source: https://arxiv.org/abs/2510.26692
- Status: partially implemented (candidate prefilter + shortlist).
- Next: multi-stage verifier cascade with learned gate.

5. Titans / nested persistent memory
- Source: https://arxiv.org/abs/2501.00663
- Gap: no explicit persistent memory bank shared across updates.
- Potential: stronger retention with lower re-reading cost.

6. RLVR stability and verifier-aware reward framing
- Source: https://arxiv.org/abs/2508.05428
- Gap: objective switching is still limited.
- Potential: lower policy oscillation and faster convergence.

7. Qwen3 report (reasoning budget + efficient post-training)
- Source: https://arxiv.org/abs/2505.09388
- Gap: no explicit dynamic thinking-budget controller in runtime loop.

8. AlphaEvolve-style algorithmic search loop
- Source: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Gap: self-edit loop exists, but search-space and acceptance protocol are still shallow.

## 4) Novel Methods (Custom Proposals)

## A) SCL: Startup Curriculum Loader (Custom)
- Problem solved: 5-minute no-step startup stalls.
- Idea:
  - Phase 0 (first N steps): deterministic local compressed shard stream (small, high-quality seed).
  - Background thread warms remote streams in parallel.
  - Automatic handoff to full remote mixture after stable step-rate.
- Why it should work: decouples first-step latency from network variability.
- Acceptance target:
  - reduce time-to-first-step by >=70%
  - no loss regression at 500-step gate.

## B) VCD-2: Verifier Cascade Distillation (Custom)
- Idea:
  - Stage 1: cheap lexical/symbolic filter.
  - Stage 2: deterministic lightweight checker.
  - Stage 3: strict verifier only for uncertain samples.
  - Distill pass/fail boundary into a tiny gating head.
- Why: verifier path is often dominant in TTRL loops.
- Acceptance target:
  - >=20% reduction in `phase_ttrl_s`
  - no hidden/public pass regression.

## C) ARMO: Adaptive RL Mixture Orchestrator (Custom)
- Idea:
  - Blend CISPO/DISPO/IGRPO/GSPO each update using reward variance, confidence, and diversity.
  - Keep per-objective floor to avoid collapse.
- Why: fixed objective can be suboptimal across training phases.
- Acceptance target:
  - >=10% lower RL loss volatility
  - >=5% better composite at same wall time.

## D) HDCR: Horizon-Delayed Causal Replay (Custom)
- Idea:
  - assign replay credit over delayed horizon windows rather than single-step gain.
  - use decay-weighted future gains to rank replay samples.
- Why: replay effects are delayed; immediate credit is noisy.
- Acceptance target:
  - higher replay usefulness signal stability
  - improved hidden_eval at equal tokens.

## E) BAA: Bit Allocation Auction for INSTANT (Custom)
- Idea:
  - each layer bids precision budget using reconstruction sensitivity + gradient salience.
  - allocate bit-width dynamically under fixed memory budget.
- Why: uniform compression is inefficient.
- Acceptance target:
  - memory savings while preserving hidden_eval pass.

## 5) Execution Order (Highest ROI First)
1. SCL hardening (already enabled; keep stable).
2. Train-phase kernel/runtime optimization (new top bottleneck).
3. VCD-2 (reduce verifier wall time).
4. ARMO (stability/quality uplift).
5. HDCR (replay sample efficiency).
6. BAA (memory-quality efficiency).

## 6) Promotion Rule (No Fake Gains)
A candidate becomes default only if all pass:
- EXE-only proof run,
- >=15% composite gain OR >=20% lower loss OR >=2x matched-loss speed,
- no regression in hidden/public verifier pass,
- required feature effective-call counters remain non-zero.

## 7) Notes
- Replace existing algorithms only if measured superiority is proven.
- Parameter-only tweaking is not sufficient for promotion.
- Startup bottleneck must be solved first before claiming global learning-speed gains.
