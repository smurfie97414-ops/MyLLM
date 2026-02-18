# SIGMA EXE-First Training Project

Ce dépôt est un runtime d'entraînement LLM orienté EXE (`eco_train.exe`) avec une pile CUDA/Triton stricte, une boucle d'entraînement Sigma (RL/verifier/auto-évolution), et un pipeline data streaming hybride.

## 1) Portée réelle du projet

- Entrée principale: `train_sigma.py`
- Runtime officiel: `eco_train.exe @train_sigma_args.txt`
- Build officiel: `build_exe.bat`
- Architecture modèle principale: `model/sigma_llm.py`
- Orchestrateur d'entraînement: `training/sigma_trainer.py`
- Référence algorithmique détaillée: `README_ALGORITHMS.md`

Le projet est conçu en priorité pour Windows + CUDA, avec compilation/payload packagés via PyInstaller.

## 2) Arborescence technique

- `model/`: blocs réseau (BitLinear, MLA, MoE, Sigma Mamba/MLA, kernels Triton)
- `training/`: trainer, optimizers, mémoire INSTANT, RL/verifier, data streaming, hydra, intégrité
- `evolution/`: Fractal NAS + Uroboros (évolution de stratégie)
- `research/`: journalisation des cycles de recherche
- `scripts/verify_runtime.py`: vérification dure des imports/symboles runtime
- `train_sigma_args.txt`: profil d'exécution par défaut de l'EXE
- `requirements-py313-cu128.txt`: lockfile dépendances

## 3) Workflow officiel (et unique)

1. Build

```bat
build_exe.bat
```

2. Run

```bat
eco_train.exe @train_sigma_args.txt
```

3. Overrides ponctuels (optionnel)

```bat
eco_train.exe @train_sigma_args.txt --steps 500 --output-dir runs\sigma_debug
```

## 4) Contraintes environnement strictes

### OS / Toolchain

- Windows (workflow `.bat`)
- Python 3.13 (`py -3.13`)
- Visual Studio 2026 toolchain recherché dans:
  - `C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat`
  - `C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat`

### CUDA / DL stack (lockfile)

- `torch==2.10.0+cu128`
- `torchvision==0.25.0+cu128`
- `torchaudio==2.10.0+cu128`
- `triton-windows==3.6.0.post25`
- `transformers==4.57.6`
- `datasets==4.3.0`
- `trl==0.24.0`
- `unsloth==2025.3.3`
- `unsloth-zoo==2025.3.1`

Le build échoue volontairement si les imports/symboles requis ne sont pas présents (`scripts/verify_runtime.py`).

### libtorch local (optionnel mais supporté)

`build_exe.bat` cherche un libtorch local dans l'ordre:

1. `my_eco_llm/libtorch`
2. `../libtorch`

Puis injecte `PATH/LIB/INCLUDE` en conséquence.

## 5) Ce que fait réellement `build_exe.bat`

- Stoppe si `eco_train.exe` est en cours d'exécution
- Charge l'environnement VS2026 si disponible
- Exporte `--allow-unsupported-compiler` via:
  - `TORCH_NVCC_FLAGS`
  - `CUDAFLAGS`
  - `CMAKE_CUDA_FLAGS`
- Upgrade `pip`
- Installe strictement `requirements-py313-cu128.txt` (index CUDA PyTorch)
- Exécute `scripts/verify_runtime.py`
- Pack `train_sigma.py` en `eco_train.exe` avec PyInstaller
- Copie payload `_internal` au root du projet

## 6) Profil runtime effectif (EXE)

Le profil effectif est défini par `train_sigma_args.txt`.

### Valeurs clés activées

- Device: `cuda` + `--require-cuda`
- Optimizer: `c3o`
- Data backend: `hybrid`
- Modèle: `d_model=768`, `n_layers=18`, `n_heads=12`, `seq_len=512`, `batch_size=4`
- Steps: `20000`
- Save interval: `20`
- Max checkpoints: `5`
- Checkpoint dir: `checkpoints`
- Output dir: `runs/sigma_main`
- Compile Torch: **désactivé dans ce profil** via `--no-torch-compile`

### Ecart important entre defaults Python et profil EXE

- `train_sigma.py` expose `--torch-compile` par défaut à `True`
- `train_sigma_args.txt` force `--no-torch-compile`

Même logique pour certains chemins (`output-dir` Python vs args EXE): la vérité opérationnelle est le fichier d'arguments.

## 7) Architecture runtime synthétique

### Modèle

- `SigmaLLM` alterne blocs Mamba et MLA selon `mamba_ratio/attention_ratio`
- MoE DeepSeek intégré dans les blocs Sigma
- BitLinear utilisé pour les projections
- Kernels Triton dédiés:
  - BitLinear (`model/triton_kernels.py`)
  - Scan Mamba (`model/sigma_kernels.py`, source `model/sigma_kernel_src.py`)

### Entraînement

`SigmaTrainer` orchestre:

- Boucle train + autocast + métriques phase par phase
- Vérification/inférence de récompense (math/code)
- RL objectives + auto-mix
- Replay causal
- Crédit contre-factuel C3O
- Self-improver
- Hydra v2.1
- Fractal NAS + Uroboros

### Data

`training/data.py` implémente un streaming hybride:

- FineWeb/Cosmopedia (HF)
- CommonCrawl
- Préfetch + fallback + bootstrap seed batches

## 8) Checkpoints, reprise, crash reports

### Checkpoints

- Sauvegarde atomique périodique
- Reprise sur dernier checkpoint valide
- Quarantaine des checkpoints corrompus
- Politique de rétention (`max-checkpoints`)

### Crash reports

Écriture de JSON détaillés dans:

- `<output_dir>/errors/crash_step_*.json`

Contenu: exception, trace, métriques récentes, état des features.

## 9) Métriques et preuves d'activation

Sortie centrale:

- `metrics.jsonl`

Familles de métriques:

- Vitesse et phases (`core_*`, timings par stage)
- Ressources (GPU/CPU/RAM)
- Qualité/perte
- Intégrité et état des features (`feature_*_enabled`, `feature_*_effective_calls`)
- Signaux startup/reprise

## 10) Intégrité et garde-fous

- `training/integrity_guards.py` fournit les checks d'empreinte/manifeste
- `training/hidden_eval_manifest.json` sert de référence d'évaluation cachée
- En mode preuve, les features attendues doivent produire des appels effectifs non nuls

## 11) Risques et modes d'échec à connaître

- CUDA requis: `train_sigma.py` refuse `--device` non-CUDA
- Triton requis pour certaines voies critiques (BitLinear/Mamba scan)
- `SigmaLLM` exige patch INSTANT avant `forward` (sinon `RuntimeError`)
- Mismatch possible entre attentes README et profil runtime si `train_sigma_args.txt` est modifié sans mise à jour documentaire

## 12) Checklist d'exécution rigoureuse

1. Vérifier GPU/CUDA visible
2. Lancer `build_exe.bat`
3. Contrôler que `scripts/verify_runtime.py` passe
4. Lancer `eco_train.exe @train_sigma_args.txt`
5. Vérifier apparition de `metrics.jsonl`
6. Vérifier progression checkpoints dans `checkpoints/`
7. Contrôler les `feature_*_effective_calls` sur un run réel

## 13) Fichiers de référence

- `train_sigma.py`
- `train_sigma_args.txt`
- `build_exe.bat`
- `requirements-py313-cu128.txt`
- `training/sigma_trainer.py`
- `training/data.py`
- `training/optimizer.py`
- `model/sigma_llm.py`
- `model/sigma_kernels.py`
- `model/triton_kernels.py`

## 14) Validation EXE (preuves reelles)

Validation effectuee sur `eco_train.exe` sans fallback de desactivation (`num-workers=0` non utilise).

### Run A: profil EXE standard (complet, sans desactivation)

Commande:

```bat
eco_train.exe @train_sigma_args.txt --steps 50 --output-dir runs/proof_exe_standard50_c3o --checkpoint-dir checkpoints_proof_exe_standard50_c3o --save-interval 10 --log-interval 1
```

Constats observes:

- `runs/proof_exe_standard50_c3o/metrics.jsonl` genere (`rows=51`, dernier `step=50`)
- `checkpoints_proof_exe_standard50_c3o/checkpoint_step_00000050.pt` genere
- `unsloth_trl_patch_active=1`, `unsloth_trl_patch_hits=6`, `unsloth_trl_patch_targets=6`
- Compteurs effectifs non nuls: `instant`, `diff_mla`, `sigma_rl`, `verifier`, `verifier_cascade`, `fractal_nas`, `self_improver`, `uroboros`, `causal_replay`, `asym_verify`, `rl_auto_mix`, `hydra_v21`, `c3o_credit`

### Run B: profil EXE d'activation acceleree (sans desactiver de features)

Commande:

```bat
eco_train.exe @train_sigma_args.txt --steps 12 --output-dir runs/proof_exe_activation_c3o --checkpoint-dir checkpoints_proof_exe_activation_c3o --save-interval 6 --log-interval 1 --startup-meta-ramp-steps 0 --ttrl-interval 2 --fractal-interval-steps 6 --hydra-update-interval 4 --uroboros-interval 4
```

Constats observes:

- `runs/proof_exe_activation_c3o/metrics.jsonl` genere (`rows=8`, dernier `step=7`)
- `checkpoints_proof_exe_activation_c3o/checkpoint_step_00000006.pt` genere
- Compteurs effectifs non nuls supplementaires:
  - `feature_sigma_rl_effective_calls=3`
  - `feature_verifier_effective_calls=60`
  - `feature_verifier_cascade_effective_calls=3`
  - `feature_asym_verify_effective_calls=3`
  - `feature_rl_auto_mix_effective_calls=4`
  - `feature_hydra_v21_effective_calls=1`
  - `feature_fractal_nas_effective_calls=2`
  - `feature_c3o_credit_effective_calls=11`

### Note C3O

- Le profil EXE de base est aligne sur `--optimizer c3o` avec `--c3o-credit-enabled`.
- Garde dur: le runtime refuse toute execution si `--c3o-credit-enabled` est actif avec un optimiseur non `c3o`.
- Preuve garde dur (observee): `eco_train.exe @train_sigma_args.txt --optimizer muon --steps 2` => `RuntimeError: c3o credit is enabled but optimizer is not 'c3o'`.

### Run C: demarrage sans override (parametres basiques)

- Commande: `eco_train.exe @train_sigma_args.txt`
- Observation en 10 min: progression reelle jusqu'au `step=39` dans `runs/sigma_main/metrics.jsonl`, avec `feature_c3o_credit_enabled=1` et `feature_c3o_credit_effective_calls=39`.



