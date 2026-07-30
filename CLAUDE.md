# CLAUDE.md — Fibrin Clot CNN Project Reference

## !! CRITICAL: Train/Val/Test Split Integrity !!

**NEVER add grid images (endpoint_64) to a dataset unconditionally.**
Grid images MUST be filtered by the experiments already present in the target split:

```python
# CORRECT — filter to split's experiments before concat
split_exps = set(df["Experiment"])
grid_df = grid_all[
    (grid_all["img_type"] == "endpoint_64") &
    (grid_all["Experiment"].isin(split_exps))
]
self.df = pd.concat([df, grid_df]).reset_index(drop=True)

# WRONG — adds val/test grid images to every dataset partition
grid_df = grid_all[grid_all["img_type"] == "endpoint_64"]  # ← DATA BLEED
self.df = pd.concat([df, grid_df]).reset_index(drop=True)
```

Every code path that appends grid images MUST include the `Experiment.isin(split_exps)` filter.
This was violated in `masked_patch_dataset.py` and `masked_patch_dataset_full.py`, causing data
bleed in models mpatch_v0_{b,d,e,f,g,h,i,f1a} — all required full retraining.

The experiment-level split is created by `get_or_create_split` with `seed=99` and is
deterministic — deleting a model directory causes the same split to be recreated.

---

## Project Goal
Classify 1000 microscopy images of fibrin clots across 5 phenotypes using a PyTorch CNN.

## Classes
| Label | Phenotype | Count |
|-------|-----------|-------|
| NC1   | Normal control | 200 |
| AC3   | Prolonged clotting time | 200 |
| F08D  | Factor VIII deficient — Hemophilia A | 200 |
| F09D  | Factor IX deficient — Hemophilia B | 200 |
| F11D  | Factor XI deficient — Hemophilia C | 200 |

Class map (integer labels): AC3=0, F08D=1, F09D=2, F11D=3, NC1=4

## Data
- **Images**: `data/photos/0000.JPG` – `0999.JPG` (6000×4000 px JPEG)
- **Labels**: `data/endpoint10.db` → table `Images_Endpoint_10`
  - `rowid - 1` = filename integer (row 1 → `0000.JPG`)
  - Key columns: `Experiment`, `Exp_Type` (class label), `Slide_Type` (A/B)
- **50 distinct experiments** (10 per class); images from the same experiment are correlated

## File Structure
```
# Core pipeline — full-image
preprocessing.py             Grayscale + 10× min-pool → (1, 400, 600)
preprocessing_frangi.py      Multi-channel Frangi pipeline (7-ch, 4× mean-pool)
preprocessing_full.py        No-pool preprocessing → (1, 4000, 6000) for full-res patches
augmentation.py              4-fold flip augmentations (full-image)
data_loader.py               DB access, FibrinDataset, split helpers
model.py                     FibrinCNN architecture (~455K params)

# Core pipeline — patch-based (10× min-pool)
augmentation_patch.py        Kornia GPU augmentation module for patches
patch_dataset.py             FibrinPatchDataset (uniform random sampling);
                             split_by_experiment_3way / get_or_create_split /
                             split_by_experiment_kfold / get_lc_train_df
masked_patch_dataset.py      MaskedPatchDataset (content-aware sampling, 10× pool)
model_patch.py               FibrinPatchCNN architecture (~900K params)

# Resolution variants — patch-based
masked_patch_dataset_125s.py MaskedPatchDataset for 1.25× area-resize (4800×3200) images
masked_patch_dataset_2x.py   MaskedPatchDataset for 2× min-pool (2000×3000) images
masked_patch_dataset_full.py MaskedPatchDataset for full-resolution (6000×4000) images
model_patch_125s.py          FibrinPatchCNN for 1.25× scale
model_patch_2x.py            FibrinPatchCNN for 2× min-pool
model_patch_full.py          FibrinPatchCNNFull for full-resolution patches

# ABMIL pipeline
abmil_dataset.py             Bag dataset for attention-based MIL
abmil_model.py               Attention-Based Multiple Instance Learning model
train_abmil.py               ABMIL training loop (separate entry point from train.py)
evaluate_abmil.py            ABMIL evaluation
compare_voting_methods.py    Compare voting / ensemble aggregation methods
test_abmil_pipeline.py       ABMIL pipeline tests

# Shared utilities
compute_masks.py             Precompute foreground masks → masks/<version>/
compute_patch_centers.py     Precompute valid patch centers → masks/patch_centers/
gpu_utils.py                 GPU detection + AMP/compile config per SM version
lr_schedulers.py             LR scheduler utilities
checkpoint_manager.py        Checkpoint save/load logic
configs/training_configs.py  All model hyperparameter configs

# Training
train.py                     Unified entry point (dispatches by --model-type)
train_5class.py              5-class full-image training loop
train_5class_hpc.py          HPC-optimized 5-class (cosine, W&B, checkpointing)
train_3class.py              3-class hemophilia (scratch or finetune)
train_patch.py               Patch + masked-patch unified training loop

# Evaluation
evaluate.py                  Per-class precision/recall/F1, confusion matrix
evaluate_cosine.py           Evaluation for cosine-head full-image models
evaluate_patch.py            Patch cosine soft-vote grid inference;
                             score_patch_grid() returns raw per-patch scores,
                             predict_image() wraps it with the mean+argmax step

# Holdout evaluation (reproducible testing, separate from training-time val/test)
holdout_eval_utils.py            Shared helpers: 125s model loader, best-epoch/val-acc
                                 extraction, checkpoint copying, shared-split assertion
evaluate_holdout_125s_ensemble.py  8-fold CV ensemble holdout eval (mpatch_v1e_125s +
                                   rep_a..g) — solo + ensembled patch/image accuracy
evaluate_holdout_lite_lc.py        Batched holdout eval of all 28 mpatch_v1e_lite_lc*
                                   models over their one shared holdout set
aggregate_learning_curve.py        Aggregates the 28 lite_lc holdout summaries into a
                                   learning-curve CSV/JSON/plot

# Analysis scripts
run_analysis.py              Orchestrates all 11 analysis steps
hemophilia_analysis.py       ROC curves, annotated images, multi-model compare
gradcam.py                   Grad-CAM class saliency maps + full overlay batch
visualize_weights.py         CNN kernel heatmaps + activation maps
activation_analysis.py       CNN activation map spatial analysis
misclassification_report.py  Per-image misclassification analysis
preprocessing_comparison.py  Preprocessing method robustness comparison
plot_training_curves.py      Train/val loss and accuracy curves
analysis_utils.py            Shared inference DataFrame + model registry
replay_wandb.py              Replay offline W&B runs to cloud

# Diagnostics / utilities
tune_patch_threshold.py      Sweep min_fg thresholds for mask tuning
visualize_patch_centers.py   Visualize patch-center validity maps
inspect_masks.py             Inspect foreground mask quality
patch_validate.py            Validate patch sampling distribution
generate_figs.py / figs_best_class.py  Publication figures

# Tests
test_preprocessing.py        Visual + structural tests (original pipeline)
test_preprocessing_frangi.py Tests + visual outputs for Frangi pipeline
test_augmentation.py         Visual + property tests
test_data_loader.py          DB, split, Dataset unit tests
test_patch_pipeline.py       Patch dataset, model, augmentation, split tests
test_lr_scheduler.py         LR scheduler tests
test_abmil_pipeline.py       ABMIL pipeline tests
test_holdout_eval.py         Grid geometry, preprocessing shapes, untrained-model
                             plumbing, evaluate_patch.py refactor regression,
                             holdout_eval_utils fixture tests

# Slurm
slurm/setup_env.sh                       One-time conda env creation on HPC
slurm/submit_all.sh                      Submit all standard training jobs
slurm/submit_mpatch_v0.sh               Submit all mpatch_v0 jobs
slurm/submit_mpatch_v1e_lite_lc.sh      Submit LC "a" series (7 jobs)
slurm/submit_mpatch_v1e_lite_lc_bcd.sh  Submit LC b/c/d series (21 jobs)
slurm/train_*.sh                         Per-model self-resubmitting job scripts
slurm/sync_wandb.sh                      Sync offline W&B runs
slurm/logs/                              Job stdout/stderr
slurm/eval_mpatch_v1e_125s_ensemble.sh   125s 8-fold ensemble holdout eval (single run)
slurm/eval_mpatch_v1e_lite_lc.sh         All-28-model lite_lc holdout eval (single run)
slurm/eval_mpatch_v1e_lite_lc_aggregate.sh  Learning-curve aggregation (CPU-only, tiny)
slurm/submit_holdout_eval.sh             Fan-out wrapper; chains aggregate job via
                                         --dependency=afterok

# Data and outputs
environment.yml              Conda environment
data/photos/                 Raw images (0000–0999.JPG)
data/endpoint10.db           SQLite labels
data/img-metadata.csv        Per-image metadata CSV
data/exp-metadata.csv        Per-experiment metadata CSV
masks/v_intensity/           Precomputed foreground masks
masks/patch_centers/         Precomputed patch-center validity maps

# Model directories
models/5class/               5-class full-image model
models/5class_hpc_v0/        Cosine 5-class HPC model
models/5class_hpc_v1{a,b,c}/ Cosine 5-class variant models
models/3class_hemo/          3-class from-scratch model
models/3class_hemo_finetune/ 3-class fine-tuned model
models/patch_v0/             Patch cosine baseline
models/patch_5class_v1{a,b,c}/  Patch cosine variants
models/patch_ce_v2{a,b,c}/   Patch CE variants
models/mpatch_v0_{a,b,c,d}/  Masked-patch baseline (legacy val; see Note below)
models/mpatch_v0_{e,f,g,h}/  Masked-patch uniform-fraction ablations
models/mpatch_v0_{e,f}_ensw/ Top-k checkpoint ensemble variants of e/f
models/mpatch_v0_f1a/        Fine-tuned from mpatch_v0_f best weights
models/mpatch_v0_i/          torch.compile test (cudagraphs backend)
models/mpatch_v0{e,f,g,h}_lite/     Lite (plateau LR, GPU preload) variants of e/f/g/h
models/mpatch_v0_f_aug/             mpatch_v0_f + brightness/contrast jitter
models/mpatch_v0e_lite_aug/         mpatch_v0e_lite + brightness/contrast jitter
models/mpatch_v0f_lite_aug/         mpatch_v0f_lite + brightness/contrast jitter
models/mpatch_v0f_lite_aug2/        mpatch_v0f_lite_aug + Gaussian noise
models/mpatch_v1_full_{a,b}/        Full-resolution (6000×4000) patch models
models/mpatch_v1e_2xmp/             2× min-pool (2000×3000) patch model
models/mpatch_v1e_125s/             1.25× resize patch model (cosine, T_mult=1.5)
models/mpatch_v1e_125s_flat/        1.25× resize patch model (cosine, T_mult=1.0)
models/mpatch_v1e_125s_plateau/     1.25× resize patch model (plateau LR)
models/mpatch_v1e_125s_rep_{a..g}/  Leave-one-out CV replicates of mpatch_v1e_125s
models/mpatch_v1e_lite_lc{05..35}{a,b,c,d}/  28 learning-curve models (see LC section)

# Holdout evaluation outputs (see Holdout Evaluation Pipeline section)
models/mpatch_v1e_125s_ensemble/holdout_eval/   8-fold ensemble holdout results + copied weights
models/mpatch_v1e_lite_lc*/holdout_eval/        Per-LC-model holdout results + copied weights
models/mpatch_v1e_lite_lc_learning_curve/       Aggregated learning-curve CSV/JSON/plot

test_output/                 Visual outputs from preprocessing/augmentation tests
agent/                       Historical planning documents
```

Note on `mpatch_v0_{a,b,c,d}`: these were trained before `val_uniform_fraction` was fixed at
1.0; their validation used `uniform_fraction=0.20` for val sampling, so val accuracy is not
directly comparable to e–h. Do not compare without accounting for this.

---

## CNN Architecture

### FibrinCNN (full-image, `model.py`)
```
Input: 1 × 600 × 400 (grayscale, after 10× min-pool)
Block 1: Conv2d(1→32,   5×5, pad=2) → BN → ReLU → MaxPool(2×2)  → 32  × 300 × 200
Block 2: Conv2d(32→64,  5×5, pad=2) → BN → ReLU → MaxPool(2×2)  → 64  × 150 × 100
Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  75 ×  50
Block 4: Conv2d(128→256,3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 256 ×  37 ×  25
AdaptiveAvgPool2d(1,1) → 256
Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→5)
~455K params. Receptive field after Block 4: 520 px in original space (≥300 px required).
```

### FibrinPatchCNN (patch-based, `model_patch.py`)
```
Input: 1 × 200 × 200 (grayscale patch after rotation + crop)
Block 1: Conv2d(1→32,  3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 32  × 100 × 100
Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 64  ×  50 ×  50
Block 3: Conv2d(64→128,3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  25 ×  25
Block 4: Conv2d(128→256,3×3,pad=1) → BN → ReLU → MaxPool(2×2)  → 256 ×  12 ×  12
AdaptiveAvgPool2d(1,1) → 256
Linear(256→128) → ReLU → Dropout(p) → Linear(128→5)
~900K params. Cosine head variant: NormalizedLinear replaces final Linear.
```

Resolution-specific variants (`model_patch_125s.py`, `model_patch_2x.py`, `model_patch_full.py`)
have matching architecture scaled for their patch sizes (250×250, 1000×1000, 2000×2000).

---

## Preprocessing

### Original pipeline (`preprocessing.py`)
1. Grayscale via CIE LAB L-channel (`lab_l`, default; also `luminance`, `hsv_v`, `hsv_s`, `green`)
2. 10× **min-pool**: `img.reshape(H//10, 10, W//10, 10).min(axis=(1,3))` → 600×400
3. Normalize to float32 [0, 1], add channel dim → tensor shape `(1, 400, 600)`

### Frangi multi-channel pipeline (`preprocessing_frangi.py`)
1. 4× **mean-pool** (color BGR): → 1000×1500
2. Grayscale (lab_l) → float32 [0, 1]
3. Frangi filter at σ = 1, 2, 4, 6, 10, 20 (one channel each)
4. Stack → tensor shape `(7, 1000, 1500)` — channel 0 = gray, channels 1–6 = Frangi σs
- Memory: ~42 MB per image (vs ~1 MB for original)
- Factory: `make_frangi_preprocessor()` → callable for FibrinDataset

### Full-resolution pipeline (`preprocessing_full.py`)
- No pooling; grayscale → float32 [0, 1] → `(1, 4000, 6000)` for full-res patch extraction.

---

## Augmentation

### Full-image augmentation (`augmentation.py`)
4 variants applied in Dataset via `idx % 4`:
- 0 = identity, 1 = h-flip, 2 = v-flip, 3 = both flips

### Patch augmentation (`augmentation_patch.py`)
Kornia GPU module applied per-batch during training only. Validation always uses center-crop only.
- **Training**: random rotation → oversized crop (283×283) → 200×200 center crop → random flips
- **Optional ColorJitter** (config keys `aug_brightness`, `aug_contrast`): brightness/contrast
  multipliers sampled uniformly within ±value. Added in `_aug` model variants.
  Example: `aug_brightness=0.15` → multiplier ~ U(0.85, 1.15)
- **Optional Gaussian noise** (config key `aug_noise_std`): additive noise after jitter.
  Added in `mpatch_v0f_lite_aug2`. Example: `aug_noise_std=0.03`
- **Validation**: `K.CenterCrop(200)` only — no jitter, no noise, no random rotation.

---

## Train/Validation/Test Splits

### Full-image models (2-way)
- **Experiment-level**: all images from one experiment stay together (prevents leakage)
- Exactly 8 of 10 experiments per class used for training (80% train, 20% val)
- Split saved to `train_record.json`; downstream scripts always load from this file
- `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)` for class imbalance

### Patch models (3-way, `patch_dataset.py`)
- `split_by_experiment_3way(seed=99)`: 7 train / 1 val / 2 test experiments per class
  - train: 7×5=35 experiments (~700 images); val: 5; test: 10
- Saved to `train_record_patch.json`; loaded by `get_or_create_split()` / `load_split_record()`
- `seed=99` is independent from full-image models (`seed=42`) — different partitions

### Leave-one-out CV (mpatch_v1e_125s replicates)
- `split_by_experiment_kfold(fold_idx, base_model_dir)`: rotates the validation experiment
  within `mpatch_v1e_125s`'s non-test pool (8 experiments/class), keeping the same test set
- `get_or_create_kfold_split()`: creates and caches the fold split
- `mpatch_v1e_125s_rep_{a..g}` use `cv_fold_idx=1..7`

### Learning curve split sourcing
- All 28 `mpatch_v1e_lite_lc*` models copy `train_record_patch.json` from `models/mpatch_v0e_lite`
  via `split_source_dir` config key (set at first epoch; never recreated)
- `get_lc_train_df(train_df, n_per_class, seed, model_dir)` in `patch_dataset.py`:
  draws N experiments/class from train_df; persists to `lc_experiments.json` in `model_dir`
  so Slurm restarts reproduce the identical subset

---

## Hyperparameters

### Full-image (train_5class.py)
| Parameter    | Value  |
|--------------|--------|
| batch_size   | 16     |
| lr (Adam)    | 1e-3   |
| weight_decay | 1e-4   |
| num_epochs   | 30     |
| Scheduler    | ReduceLROnPlateau(patience=5, factor=0.5) |

### Patch cosine models (`_PATCH_V0_BASE`)
| Parameter    | Value  |
|--------------|--------|
| batch_size   | 64     |
| lr (Adam)    | 1e-3   |
| weight_decay | 1e-3   |
| Scheduler    | CosineAnnealingWarmRestarts(T_0=100, T_mult=varies) |

### LC / lite models (`_MPATCH_V1E_LITE_LC_BASE`)
| Parameter            | Value  |
|----------------------|--------|
| batch_size           | 32 (pinned; P100 cap) |
| lr (Adam)            | 2e-3   |
| Scheduler            | ReduceLROnPlateau(mode='min', factor=0.95, patience=5, threshold=1e-4) |
| plateau_metric       | val_loss (nats; class-weighted CE from training subset) |
| early_stop_min_epochs | 1000  |
| early_stop_patience  | 30 epochs without ≥1e-4 improvement in val_loss |
| Early stop exit code | sys.exit(100) — Slurm does not resubmit |

---

## Learning Curve Series (`mpatch_v1e_lite_lc*`)

28 models organized as 7 training-set sizes × 4 independent repetitions:

| Size suffix | N experiments/class | Total train experiments |
|-------------|---------------------|-------------------------|
| lc05        | 1                   | 5                       |
| lc10        | 2                   | 10                      |
| lc15        | 3                   | 15                      |
| lc20        | 4                   | 20                      |
| lc25        | 5                   | 25                      |
| lc30        | 6                   | 30                      |
| lc35        | 7                   | 35 (all training exps)  |

| Repetition | lc_seed formula | Purpose |
|------------|-----------------|---------|
| a          | N               | primary draw |
| b          | 100 + N         | independent subset draw |
| c          | 200 + N         | independent subset draw |
| d          | 300 + N         | independent subset draw |

lc35{a,b,c,d}: N=7 exhausts all training experiments, so all four draw identical subsets;
they differ only in weight initialization and batch ordering.

All 28 models share: same val/test split as `mpatch_v0e_lite`, same architecture (FibrinPatchCNN
with 10× min-pool), same plateau scheduler + early stopping, `preload_device="cuda"`.

---

## Holdout Evaluation Pipeline

Reproducible holdout testing for the `mpatch_v1e_125s` CV ensemble and the `mpatch_v1e_lite_lc*`
learning-curve series, local- and HPC-runnable (same entry points either way; device is
auto-detected). Each run copies the evaluated checkpoint(s) and training/split metadata into a
dedicated `holdout_eval/` directory for reproducibility, and reports two accuracy metrics:
- **patch accuracy** — each grid patch's own argmax vs. the image's true label (no averaging)
- **image accuracy** — soft-vote (mean over all grid patches) argmax vs. true label

Both scripts use `evaluate_patch.score_patch_grid()` to get raw per-patch scores (35-patch grid,
`inference_grid_centers()`), then average/argmax as needed per metric above.

### `evaluate_holdout_125s_ensemble.py`
Evaluates the complete 8-fold leave-one-experiment-out CV ensemble (`mpatch_v1e_125s` + `rep_a..g`
— NOT `_flat`/`_plateau`, which are separate LR-schedule experiments, not CV folds). All 8 fold
models share an identical `test_indices` holdout by construction (`split_by_experiment_kfold`
copies the test set through unchanged); `holdout_eval_utils.assert_identical_test_split()` verifies
this before evaluating and raises if it ever doesn't hold. For each holdout image, all 8 models
score the same 1.25×-resize grid (`H=3200, W=4800, stride=800` — physically the same 35 locations
as the 200×200/600×400 grid, scaled 8×); ensemble scores are the per-patch mean across all 8
models. Reports both the ensemble's accuracy and each of the 8 solo models' own accuracy.
Output: `models/mpatch_v1e_125s_ensemble/holdout_eval/` (`weights/` × 8, `training_metadata.json`,
`split_record.json`, `per_image_predictions.csv`, `per_patch_predictions.csv`,
`summary_table.csv`/`summary.json`, confusion matrices under `image/` and `patch/`).

### `evaluate_holdout_lite_lc.py`
Evaluates all 28 `mpatch_v1e_lite_lc*` models **separately** (no ensembling) against their one
shared holdout set (inherited from `mpatch_v0e_lite` via `split_source_dir`). The ~200 holdout
images are decoded/preprocessed once and reused across all 28 model passes. Loads models via the
existing `analysis_utils.load_model_from_registry` (already supports this family — standard
600×400 geometry). Output per model: `models/<lc_key>/holdout_eval/` (kept separate from the
`models/<lc_key>/analysis/test/` produced by a standalone `evaluate_patch.py` run).

### `aggregate_learning_curve.py`
Reads all 28 `models/<lc_key>/holdout_eval/summary.json` files (skips + warns on any missing, so
it's safely re-runnable mid-progress) and writes
`models/mpatch_v1e_lite_lc_learning_curve/learning_curve_summary.{csv,json}` +
`learning_curve_combined.png` (patch- and image-accuracy vs. `lc_n_per_class`, 4 reps per size).

### Why the 125s family isn't in `analysis_utils.MODEL_REGISTRY`
`FibrinPatchCNN125s` loading lives in `holdout_eval_utils.load_125s_model()` instead — registering
it would silently expose the individual fold models (different geometry, different
`PAD`/`OVERSIZED`/grid-stride) to the ordinary single-model `evaluate_patch.py` CLI, which knows
nothing about the 125s grid or the 8-fold ensembling semantics.

### Slurm
Single, non-resubmitting runs (evaluation doesn't need training's `USR1`-trap/resume machinery).
`slurm/submit_holdout_eval.sh` submits both eval jobs and chains
`slurm/eval_mpatch_v1e_lite_lc_aggregate.sh` after the lite_lc job via `--dependency=afterok`.
**Neither eval job pins a GPU partition** (unlike `mpatch_v1e_125s*` training, which requires
H200 for `preload_device`'s ~80GB training-image cache) — eval never preloads anything, it
processes one holdout image at a time, so P100/V100/A100/H100/H200 are all fine; `gpu_utils.py`'s
runtime detection adapts automatically. `slurm/eval_mpatch_v1e_125s_ensemble.sh` requests `mem=12G`
(vs. training's 40G) accordingly.

---

## Mask Pipeline
- **Foreground masks** (`compute_masks.py`): precomputed per image, stored in `masks/<version>/`
  - `v_intensity`: light-background intensity threshold (primary)
  - `v_frangi`, `v_entropy`: alternative mask methods
- **Patch-center validity** (`compute_patch_centers.py`): inscribed-circle rule — center valid if
  ≥`min_fg` fraction of circle (radius=100 px) is foreground. Stored in `masks/patch_centers/<version>/`.
  - `p200_circle_v_intensity`: 200 px diameter, v_intensity mask, min_fg=0.03 (3%)
- **MaskedPatchDataset** (`masked_patch_dataset.py`): blends content-weighted and uniform patch allocation.
  - `uniform_fraction`: fraction of patches drawn uniformly per image
    (default 0.10 in e-lite; 0.00 in f/f-lite; ablated in e–h)
  - `val_uniform_fraction`: always 1.0 (new default) for fair cross-model comparison;
    mpatch_v0_{a-d} used 0.20 (legacy)
  - `preload_device`: optional `torch.device` — preloads unpadded (400×600) float32 tensors onto GPU;
    requires `num_workers=0` in DataLoader (set automatically by `train_patch.py`)
  - **CRITICAL**: when `include_grid=True`, grid images are filtered to `split_exps` only (see warning above)
  - `topk_ensemble_save`: accumulates top-k validation checkpoints for later ensembling
    (enabled in `mpatch_v0_{e,f}_ensw`, all `mpatch_v1e_125s*` models)

---

## GPU Training (HPC)
- Cluster: BYU HPC — P100 (SM 6.x), V100 (SM 7.x), A100 (SM 8.x), H100/H200 (SM 9.x)
- `gpu_utils.get_gpu_config()`: returns AMP dtype, GradScaler flag, batch size, compile flag per GPU
  - SM 6.x (P100):  fp16, GradScaler, **batch_size=32** (hardcoded cap regardless of config)
  - SM 7.x (V100):  fp16, GradScaler, batch_size=64, no compile
  - SM 8.x (A100):  bf16, no GradScaler, no compile (Triton needs `cuda-cudart-dev` in conda to enable)
  - SM 9.x (H100/H200): bf16, no GradScaler, no compile (same)
- LC models (`mpatch_v1e_lite_lc*`) explicitly pin `batch_size=32` in config to enforce P100-cap
  consistency across all GPU types (no partition constraint in their Slurm scripts)
- `force_compile: True` in a model config overrides `gpu_utils` and calls `torch.compile(model, backend=compile_backend)`
- `mpatch_v0_i` uses `compile_backend: "cudagraphs"` as a Triton-free compile test
- Full-res / 2× / 1.25× models require H200 (eng, m13h, mgh partitions) for memory headroom

---

## Running
```bash
conda activate fibrin
pytest -v -s                                    # run all tests
pytest test_preprocessing_frangi.py -v -s       # Frangi pipeline tests + visuals

# Training (--db overrides default data/endpoint10.db path)
python train.py --model-type 5class             # train 5-class → models/5class/
python train.py --model-type 3class_scratch     # train 3-class from scratch
python train.py --model-type 3class_finetune    # fine-tune 5-class → 3-class
python train.py --model-type 5class --force-resplit  # regenerate train/val split

# Patch / masked-patch (HPC self-resubmitting)
python train.py --model-type mpatch_v0e_lite --resume --max-epochs 10000 --epochs-per-job 600

# Evaluation
python evaluate.py --model-type 5class          # per-class metrics, confusion matrix
python evaluate.py --model-type 3class_finetune

# Full analysis suite (runs all 11 steps, outputs → models/<type>/analysis/)
python run_analysis.py --model-type 5class
python run_analysis.py --model-type 3class_finetune

# Individual analysis scripts
python hemophilia_analysis.py --model-type 5class --roc      # OVR ROC / AUC curves
python hemophilia_analysis.py --compare                      # multi-model ROC overlay
python hemophilia_analysis.py --model-type 5class --annotate # annotated test images
python gradcam.py --model-type 5class                        # saliency maps
python gradcam.py --model-type 5class --all-overlays         # full-res overlay per image
python visualize_weights.py --model-dir models/5class        # kernel + activation maps

# Holdout evaluation (local or via sbatch — see Holdout Evaluation Pipeline section)
python evaluate_holdout_125s_ensemble.py                     # 8-fold CV ensemble → models/mpatch_v1e_125s_ensemble/holdout_eval/
python evaluate_holdout_125s_ensemble.py --limit 5            # fast dry run
python evaluate_holdout_lite_lc.py                            # all 28 lc models → models/<key>/holdout_eval/
python evaluate_holdout_lite_lc.py --models mpatch_v1e_lite_lc05a --limit 5   # fast dry run
python aggregate_learning_curve.py                            # → models/mpatch_v1e_lite_lc_learning_curve/
bash slurm/submit_holdout_eval.sh                              # submit both pipelines on HPC
```

---

## data_loader.py Key Exports
- `load_split_from_record(record_path, db_path)` — reconstruct (train_df, val_df) from saved JSON; never re-randomizes
- `filter_classes(df, classes)` — subset DataFrame to specified class names

## patch_dataset.py Key Exports
- `split_by_experiment_3way(df, seed=99)` — 7/1/2 per class; used by all patch models
- `get_or_create_split(model_dir, db_path)` — load from `train_record_patch.json` or create and save
- `split_by_experiment_kfold(df, fold_idx, base_model_dir, db_path)` — leave-one-out CV
- `get_or_create_kfold_split(model_dir, db_path, fold_idx)` — cached kfold split
- `get_lc_train_df(train_df, n_per_class, seed, model_dir)` — LC experiment subsetting;
  saves/loads `lc_experiments.json` for Slurm restart reproducibility

## analysis_utils.py Key Exports
- `MODEL_REGISTRY` — maps model name → (model_dir, num_classes, class_map); covers all current models
- `load_model_from_registry(model_type, device)` — returns (model, model_dir, num_classes, class_map, class_names)
  - Routes `startswith("mpatch_v1_full")` → FibrinPatchCNNFull; all other mpatch → FibrinPatchCNN
- `get_val_split(model_type, db_path)` — loads val_df from train_record.json; filters to hemo classes for 3-class models
- `run_inference_full(model, val_df, photos_dir, device, preprocessor, class_map, class_names)` → DataFrame
  - Columns: idx, Experiment, Exp_Type, Slide_Type, true_label, pred_label, correct, confidence, prob_<class>...

## holdout_eval_utils.py Key Exports
- `load_125s_model(model_dir, device, head_type, num_classes)` — standalone FibrinPatchCNN125s loader
- `get_best_epoch_metadata(model_dir)` — best epoch/val_acc from `training_log_full.csv`
  (falls back to `checkpoints/topk_manifest.json`)
- `copy_best_checkpoint(model_dir, dest_weights_dir, name)` — copies `best_model.pth` for reproducibility
- `assert_identical_test_split(model_dirs)` — verifies shared `test_indices` across model dirs; raises on mismatch
- `extract_split_record_subset(model_dir)` / `accuracy_breakdown(results_df, class_names)` / `write_json(path, obj)`

---

## Trained Model Results (validation set)
| Model              | Val Set | Overall Acc | F08D Acc | Notes |
|--------------------|---------|-------------|----------|-------|
| 5class             | n=200   | 57.7%       | 15.0%    | F08D hardest; top confusion F08D→F09D |
| 3class_hemo_finetune | n=120 | 70.8%       | —        | Fine-tuned from 5class weights (F08D/F09D/F11D only) |
| 3class_hemo        | n=120   | 62.5%       | —        | From-scratch 3-class |

Preprocessing sensitivity (5-class model, validation set):
green=63.2% > luminance=61.7% > lab_l=57.7% > hsv_v=57.2% > hsv_s=30.9%

---

## Known Issues / Fixes Applied
- `ReduceLROnPlateau(verbose=True)` removed — argument dropped in PyTorch 2.4+
- `np.trapz` → `np.trapezoid` — renamed in NumPy 2.0
- No sklearn dependency; metrics computed from scratch in `evaluate.py`
- `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler('cuda', ...)` — deprecated in PyTorch 2.7+
- BatchNorm eval mode: always call `model.eval()` before inference; `predict_all()` in `evaluate.py` does this defensively
- `torch.compile` disabled by default on all GPU types — Triton requires `cuda.h` from `cuda-cudart-dev` conda package which is not installed; install with `CONDA_NO_PLUGINS=true conda install -c nvidia cuda-cudart-dev` then re-enable in `gpu_utils.py`
- `is_full_res` routing in `train_patch.py` and `analysis_utils.py` uses `startswith("mpatch_v1_full")`, NOT `startswith("mpatch_v1")` — the latter would misroute all `mpatch_v1e_lite_lc*` models to FibrinPatchCNNFull
