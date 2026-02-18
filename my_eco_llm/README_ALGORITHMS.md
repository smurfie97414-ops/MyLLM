# README_ALGORITHMS

Ce document décrit l'état algorithmique réel du stack SIGMA, avec séparation claire entre:

- fonctionnalités structurelles du code,
- fonctionnalités activées par défaut via `train_sigma_args.txt`,
- fonctionnalités disponibles mais dépendantes de flags/contexte.

## 1) Cœur modèle

### 1.1 SigmaLLM (architecture principale)

- Fichier: `model/sigma_llm.py`
- Composition:
  - `SigmaMambaBlock` (Mamba-3 complexe + MoE)
  - `SigmaMLABlock` (MLA + MoE + differential attention)
- Ratios de stack pilotés par:
  - `--mamba-ratio`
  - `--attention-ratio`

### 1.2 BitLinear

- Fichier: `model/bitlinear.py`
- Rôle:
  - projections linéaires quantifiées
  - chemin de quantification avec cache optionnel
- Kernels associés:
  - `model/triton_kernels.py`
  - source Triton: `model/triton_kernel_src.py`

### 1.3 MLA et MoE

- MLA: `model/attention.py`
- MoE DeepSeek + routing/RIB: `model/moe.py`

## 2) Kernels et exigences d'exécution

### 2.1 Triton BitLinear

- `model/triton_kernels.py`
- Vérifie `TRITON_AVAILABLE` à l'import
- Peut fallback selon contexte, mais les voies optimales exigent Triton fonctionnel

### 2.2 Triton scan Mamba

- `model/sigma_kernels.py`
- `mamba3_complex_scan_interleaved` impose des contraintes runtime strictes (CUDA + disponibilité Triton)
- Source kernel: `model/sigma_kernel_src.py`

### 2.3 INSTANT obligatoire sur Sigma

- `SigmaLLM` exige patch INSTANT avant `forward`
- En cas d'absence de patch: `RuntimeError`
- Patch appliqué côté training via `training/memory_hack.py`

## 3) Pipeline d'entraînement SIGMA

### 3.1 Orchestrateur

- `training/sigma_trainer.py`
- Responsabilités:
  - loop train
  - instrumentation/perf phase timing
  - activation des sous-systèmes RL/verifier/évolution
  - checkpoints atomiques + reprise
  - crash reports détaillés

### 3.2 Objectifs RL et vérification

- Bridge TRL/GRPO: `training/grpo_trl_bridge.py`
- Sandbox exécution code: `training/code_sandbox.py`
- Objectifs RL Sigma: `training/sigma_rl_objectives.py`
- Verifier: `training/sigma_verifier.py`

### 3.3 Replay et crédit

- Replay causal: `training/replay_causal.py`
- Crédit contre-factuel: `training/counterfactual_credit.py`
- Boucles de stabilisation adaptative dans `sigma_trainer.py`

## 4) Optimisation

### 4.1 Choix d'optimiseur exposés

`train_sigma.py` (`--optimizer`):

- `muon`
- `normuon`
- `adamuon`
- `c3o`
- `gnprox`

Implémentation: `training/optimizer.py`

### 4.2 Muon family

- Split strict matrices/vecteurs supporté
- Exclusions embeddings/lm_head configurables
- Hyperparamètres orthogonalisation/TEON/adaptatifs étendus

### 4.3 C3O

- Implémentation dédiée: `training/optimizer_c3o.py`
- Intégration config/build dans `training/optimizer.py`
- Crédit C3O piloté côté trainer (`feature_c3o_credit_*`)

### 4.4 GNProx

- Implémentation: `training/optimizer_gn.py`
- Source kernel: `training/optimizer_gn_kernel_src.py`

## 5) Data pipeline

- Implémentation: `training/data.py`
- Backend par défaut EXE: `hybrid`
- Sources:
  - FineWeb/Cosmopedia (streaming HF)
  - CommonCrawl
- Mécanismes:
  - prefetch
  - fallback résilient
  - seed/bootstrap batches au démarrage

## 6) Évolution et recherche

### 6.1 Fractal NAS

- `evolution/fractal_nas.py`
- Ajustements/essais orientés perf/structure

### 6.2 Uroboros

- `evolution/meta_loop.py`
- Boucle de recherche/expérimentation avec fenêtres d'évaluation

### 6.3 Research loop

- `research/cycle_research_engine.py`
- Support de traçabilité des cycles/hypothèses

## 7) Intégrité, preuve et auditabilité

- Gardes d'intégrité: `training/integrity_guards.py`
- Manifeste hidden-eval: `training/hidden_eval_manifest.json`
- Dans `sigma_trainer.py`, les compteurs `feature_*_effective_calls` matérialisent l'activation réelle des features.

## 8) Features activées par défaut dans le profil EXE (`train_sigma_args.txt`)

Profil actuel notable:

- `--optimizer c3o`
- `--grpo-backend trl`
- `--data-backend hybrid`
- `--no-torch-compile`
- verifier math/code activé
- `--verifier-cascade-enabled`
- `--causal-replay-enabled`
- `--c3o-credit-enabled`
- `--self-improver-enabled`
- `--uroboros-enabled`
- `--hydra-enable`
- `--integrity-guards-enabled`
- `--crash-report-enabled`

Important: la vérité runtime est le fichier d'arguments, pas uniquement les defaults de `parse_args()`.

Compatibilite stricte:

- si `--c3o-credit-enabled` est actif, l'optimiseur doit etre `c3o` (sinon hard fail startup).

## 9) Signaux observables recommandés

Vérifier dans `metrics.jsonl`:

- `feature_*_enabled`
- `feature_*_effective_calls`
- timings phase (`phase_*`)
- débit (`core_tokens_per_s`, `effective_tokens_per_s`)
- goulot (`perf_bottleneck_*`)

Vérifier aussi:

- checkpoints valides créés/réutilisés
- absence d'erreurs dans `<output_dir>/errors/`

## 10) Limites et points de vigilance

- Certaines voies critiques dépendent strictement de CUDA + Triton
- Le mode EXE peut diverger des defaults Python si `train_sigma_args.txt` force des options
- Toute affirmation algorithmique doit être validée par compteur effectif et non par simple présence de code

## 11) Preuves runtime EXE (mesurees)

### 11.1 Preuve standard (`runs/proof_exe_standard50_c3o`)

- `unsloth_trl_patch_active=1` avec `hits=6/targets=6`
- `feature_instant_effective_calls=50`
- `feature_diff_mla_effective_calls=50`
- `feature_sigma_rl_effective_calls=2`
- `feature_verifier_effective_calls=40`
- `feature_verifier_cascade_effective_calls=2`
- `feature_fractal_nas_effective_calls=1`
- `feature_self_improver_effective_calls=32`
- `feature_uroboros_effective_calls=50`
- `feature_causal_replay_effective_calls=50`
- `feature_c3o_credit_effective_calls=54`
- `feature_asym_verify_effective_calls=2`
- `feature_rl_auto_mix_effective_calls=4`
- `feature_hydra_v21_effective_calls=3`

### 11.2 Preuve d'activation acceleree (`runs/proof_exe_activation_c3o`)

- `feature_sigma_rl_effective_calls=3`
- `feature_verifier_effective_calls=60`
- `feature_verifier_cascade_effective_calls=3`
- `feature_asym_verify_effective_calls=3`
- `feature_rl_auto_mix_effective_calls=4`
- `feature_hydra_v21_effective_calls=1`
- `feature_fractal_nas_effective_calls=2`
- `feature_c3o_credit_effective_calls=11`

### 11.3 Correctifs structurels realises

- `training/code_sandbox.py`: warmup explicite de l'executor pour eviter les faux echec reward au premier appel.
- `training/sigma_rl_objectives.py`: correction GSPO (`sigma_rl_kl`) en scalaire (`mean`) pour eviter crash tensor->scalar.
- `train_sigma.py`: bootstrap Unsloth/compile cache en package importable et activation `TRITON_BACKENDS_IN_TREE=1`.
- `build_exe.bat`: packaging source-first de `triton` + metadonnees/dependances critiques (`wandb`, `tokenizers`, etc.).
