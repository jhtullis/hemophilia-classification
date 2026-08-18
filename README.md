# Fibrin Clot CNN Classification

---

Classify brightfield microscopy images of fibrin clots across five blood phenotypes using PyTorch CNNs. The pipeline spans full-image baseline models, content-aware patch-based models, resolution ablations, an attention-based MIL variant, and a 28-model learning-curve scaling study. Full-image models use a two-way train/validation split; patch-based models introduce a proper held-out test set for final evaluation.

## Scientific Background

Fibrin clots form when blood plasma coagulates in response to injury. The resulting fibrin network — its fiber density, thickness, branching, and pore size — varies measurably between individuals depending on their coagulation factor profile. In hemophilia, key coagulation factors are absent or deficient, producing clots with a distinctively altered microstructure visible under brightfield microscopy.

This project tests whether a CNN can learn to discriminate these structural differences from images alone, with particular interest in distinguishing between the three hemophilia subtypes (A, B, C).

Initial full-image CNN models were evaluated using Grad-CAM saliency analysis, which revealed that the network had learned to discriminate Factor IX deficiency (F09D) partly by exploiting empty slide background rather than fibrin microstructure — a spurious spatial shortcut. This finding motivated a shift to content-aware patch-based training: patches are sampled exclusively from foreground (fibrin-rich) regions identified by precomputed masks, forcing the model to attend to the fibrin network itself. The patch-based pipeline also introduces a proper three-way train/val/test split with a genuinely held-out test set, enabling unbiased final evaluation independent of model selection.

## AI Use Note

The author of this repository, Jason Henry Tullis (`jhenrytullis@gmail.com`), developed it with extensive use of [Claude AI](https://claude.ai/code) and [Claude Code](https://claude.ai/code) (Anthropic). Subject to the author's steering, review, and correction, Claude Code wrote the code files and the majority of the documentation contained in this repository. Likewise subject to the author's input and review, Claude AI was used to perform online literature searches and compile plans (located in the `agent/` directory) which helped to direct Claude Code's work. The author manually staged and committed files, wrote all commit messages, suggested best practices, ran the code produced by the agents, and inspected and interpreted numeric and visual outputs from the code including performance metrics and graphs. In this process, the author also consulted with other humans (such as research advisors) on best deep learning and AI use practices, and made pivots and revisions when necessary. This `readme.md` file, and in particular this AI Use Note, was reviewed and edited by the human author prior to online publication. This work is associated with the author's Undergraduate Honors Thesis available [here](https://scholarsarchive.byu.edu/studentpub_uht/542/), the text of which was written without AI tools.

---

## Classes

| Label | Condition | Count |
|-------|-----------|-------|
| NC1   | Normal plasma control | 200 |
| AC3   | Prolonged clotting time (non-hemophilic control) | 200 |
| F08D  | Factor VIII deficient — Hemophilia A | 200 |
| F09D  | Factor IX deficient — Hemophilia B | 200 |
| F11D  | Factor XI deficient — Hemophilia C | 200 |

Integer label map (hardcoded): `AC3=0, F08D=1, F09D=2, F11D=3, NC1=4`

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fibrin
```

Key dependencies: PyTorch 2.7, Kornia, OpenCV, SQLAlchemy, pandas, NumPy, scikit-image, SciPy, wandb, pytest.

---

## Data

- **Images**: `data/photos/0000.JPG` – `0999.JPG` (6000×4000 px JPEG, landscape)
- **Labels**: `data/endpoint10.db` — SQLite, table `Images_Endpoint_10`
  - `rowid - 1` = filename integer (row 1 → `0000.JPG`)
  - Key columns: `Experiment`, `Exp_Type` (class), `Slide_Type` (A/B)
- **Metadata CSVs**: `data/img-metadata.csv` (per-image), `data/exp-metadata.csv` (per-experiment)
- **50 distinct experiments** (10 per class); images from the same experiment are correlated and always kept in the same partition
- Source: Cordner, R., & Tullis, J. H. (2026). Brightfield Images of Factor Deficient Plasma Clots. Zenodo. https://doi.org/10.5281/zenodo.19994554

---

## Preprocessing

### Full-image pipeline (`preprocessing.py`)
1. Grayscale via CIE LAB L-channel (`lab_l`, default)
2. **10× min-pool**: 6000×4000 → 600×400. Min-pool preserves dark fibers on a light background.
3. Normalize to float32 [0, 1]; add channel dim → tensor `(1, 400, 600)`. No mean/std normalization.

### Frangi multi-channel pipeline (`preprocessing_frangi.py`)
4× mean-pool + Frangi filter at σ = 1, 2, 4, 6, 10, 20 → 7-channel tensor `(7, 1000, 1500)`.

### Resolution variants
- **2× min-pool** (`preprocessing.py` `min_pool` with factor=2): 6000×4000 → 3000×2000
- **1.25× area-resize** (`preprocessing.py` `area_resize`): 6000×4000 → 4800×3200
- **Full-resolution** (`preprocessing_full.py`): no pooling → `(1, 4000, 6000)`

### Foreground masks (`compute_masks.py`)
Precompute per-image binary foreground masks using intensity, Frangi, and entropy methods. Stored in `masks/<version>/`. Used by the masked-patch pipeline to restrict patch sampling to fibrin-rich regions.

### Patch-center validity maps (`compute_patch_centers.py`)
For each image, determine which patch centers (200 px radius inscribed-circle rule) have ≥3% foreground coverage. Stored in `masks/patch_centers/<version>/`. Used by `MaskedPatchDataset`.

---

## Architectures

### FibrinCNN (`model.py`)
Full-image model. Input: `(1, 400, 600)` grayscale.
```
Block 1: Conv2d(1→32,  5×5) → BN → ReLU → MaxPool(2×2)  →  32×300×200
Block 2: Conv2d(32→64, 5×5) → BN → ReLU → MaxPool(2×2)  →  64×150×100
Block 3: Conv2d(64→128,3×3) → BN → ReLU → MaxPool(2×2)  → 128×75×50
Block 4: Conv2d(128→256,3×3)→ BN → ReLU → MaxPool(2×2)  → 256×37×25
AdaptiveAvgPool2d(1,1) → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→N)
```
~455K parameters. Receptive field: ~520 px in original image space.

### FibrinPatchCNN (`model_patch.py`)
Patch-based model. Input: `(1, 200, 200)`. BN+ReLU follows every individual conv.
```
Block 1: Conv(1→32,  3×3) → BN/ReLU  Conv(32→32,  3×3) → BN/ReLU  Conv(32→32,  3×3, s=2) → BN/ReLU  →  32×100×100
Block 2: Conv(32→64, 3×3) → BN/ReLU  Conv(64→64,  3×3) → BN/ReLU  Conv(64→64,  3×3, s=2) → BN/ReLU  →  64×50×50
Block 3: Conv(64→128,3×3) → BN/ReLU  Conv(128→128,3×3) → BN/ReLU  Conv(128→128,3×3, s=2) → BN/ReLU  → 128×25×25
Conv4:   Conv(128→256,3×3) → BN/ReLU                                                                    → 256×25×25
AdaptiveAvgPool2d(1,1) → Linear(256→128) → ReLU → Dropout → head
```
~900K parameters. No MaxPool — downsampling via stride-2 in the 3rd conv of each block. Receptive field: 59 px in patch space = 590 px in original image space.

Two head types (selected per config):
- **Cosine head** (`NormalizedLinear`): outputs cosine similarities in [−1, 1]; trained with `CosineLoss`
- **CE head** (`Linear`): standard cross-entropy logits

Resolution-specific variants in `model_patch_125s.py` / `model_patch_2x.py` / `model_patch_full.py` scale the same architecture for their patch sizes (250×250, 1000×1000, 2000×2000).

### ABMIL (`abmil_model.py`)
Attention-Based Multiple Instance Learning. Treats all patches from one image as a bag; learns per-patch attention weights aggregated into an image-level prediction. Built on top of the patch feature extractor.

---

## Training

All training (except ABMIL) is dispatched through `train.py`:

```bash
python train.py --model-type <type> [--resume] [--preload] [--max-epochs N] [--epochs-per-job N]
```

### Full-image models

| Model type | Description |
|---|---|
| `5class` | FibrinCNN, 5 classes, ReduceLROnPlateau, 30 epochs |
| `5class_hpc_baseline` | Same, separate directory for HPC comparison |
| `5class_hpc_v0` | FibrinCNNCosine, cosine head, CosineAnnealingWarmRestarts |
| `5class_hpc_v1a/b/c` | Cosine 5-class variants with tuned hyperparameters |
| `3class_scratch` | FibrinCNN (3 classes) from random init (F08D/F09D/F11D only) |
| `3class_finetune` | Fine-tune 5-class weights → 3-class head |

### Standard patch models

| Model type | Head | Notes |
|---|---|---|
| `patch_v0` | cosine | Baseline uniform-random patch model |
| `patch_v1a/b/c` | cosine | Ablation variants |
| `patch_v2a/b/c` | CE | Cross-entropy head variants |

### Masked-patch models (content-aware sampling)

Grad-CAM analysis of the full-image models showed the network learning empty slide background as a spurious discriminative signal for Factor IX deficiency. `MaskedPatchDataset` addresses this by restricting patch sampling to foreground regions identified by precomputed intensity masks, so the model is never trained on empty slide area. All masked-patch models also use the 3-way split (`train_record_patch.json`) with a held-out test set — unlike the full-image models, which used a 2-way train/val split only.

Patches are sampled preferentially from fibrin-rich regions using precomputed validity maps. `uniform_fraction` controls the blend between content-weighted and uniform-per-image sampling. Validation always uses `val_uniform_fraction=1.0` for fair cross-model comparison.

| Model | Head | `uniform_fraction` | Notes |
|---|---|---|---|
| `mpatch_v0_a` | cosine | 0.20 | Legacy — val used 0.20 |
| `mpatch_v0_b` | cosine | 0.20 | + grid images; legacy val |
| `mpatch_v0_c` | CE | 0.20 | Legacy val |
| `mpatch_v0_d` | CE | 0.20 | + grid images; legacy val |
| `mpatch_v0_e` | CE | 0.10 | val uses 1.0 |
| `mpatch_v0_f` | CE | 0.00 | Fully content-weighted; val uses 1.0 |
| `mpatch_v0_g` | CE | 0.05 | val uses 1.0 |
| `mpatch_v0_h` | CE | 0.01 | val uses 1.0 |
| `mpatch_v0_i` | CE | 0.20 | `torch.compile` cudagraphs test |
| `mpatch_v0e_lite` | CE | 0.10 | Plateau LR, GPU preload |
| `mpatch_v0f_lite` | CE | 0.00 | Plateau LR, GPU preload |
| `mpatch_v0g_lite` | CE | 0.05 | Plateau LR, GPU preload |
| `mpatch_v0h_lite` | CE | 0.01 | Plateau LR, GPU preload |
| `mpatch_v0_f_aug` | CE | 0.00 | + brightness/contrast jitter |
| `mpatch_v0e_lite_aug` | CE | 0.10 | Lite + brightness/contrast jitter |
| `mpatch_v0f_lite_aug` | CE | 0.00 | Lite + brightness/contrast jitter |
| `mpatch_v0f_lite_aug2` | CE | 0.00 | Lite_aug + Gaussian noise |
| `mpatch_v0_f1a` | CE | 0.00 | Fine-tuned from mpatch_v0_f best checkpoint |
| `mpatch_v0_e_ensw` | CE | 0.10 | Top-k checkpoint ensemble accumulation |
| `mpatch_v0_f_ensw` | CE | 0.00 | Top-k checkpoint ensemble accumulation |

### Resolution variants

| Model | Resolution | Notes |
|---|---|---|
| `mpatch_v1e_2xmp` | 2× min-pool (3000×2000) | |
| `mpatch_v1e_125s` | 1.25× area-resize (4800×3200) | Cosine LR, T_mult=1.5 |
| `mpatch_v1e_125s_flat` | 1.25× resize | Cosine LR, T_mult=1.0 |
| `mpatch_v1e_125s_plateau` | 1.25× resize | Plateau LR |
| `mpatch_v1e_125s_rep_a..g` | 1.25× resize | 7 leave-one-out CV replicates; together with `_125s` form an 8-fold ensemble |
| `mpatch_v1_full_a/b` | Full-resolution (6000×4000) | Requires H200 |

### Learning Curve Series (`mpatch_v1e_lite_lc*`)

28 models probing how accuracy scales with training data size. 7 training-set sizes × 4 independent random draws (repetitions a/b/c/d). All share the same val/test split as `mpatch_v0e_lite`.

| Size suffix | Experiments/class | Total train exps |
|------------|------------------|-----------------|
| `lc05` | 1 | 5 |
| `lc10` | 2 | 10 |
| `lc15` | 3 | 15 |
| `lc20` | 4 | 20 |
| `lc25` | 5 | 25 |
| `lc30` | 6 | 30 |
| `lc35` | 7 | 35 (all) |

Repetitions b/c/d draw independent random subsets (seeds 100+N, 200+N, 300+N; lc35 is exhaustive so all four draws are identical). Early stopping: no stop before 1000 epochs, then stop after 30 epochs without ≥1e-4 improvement in val_loss.

### ABMIL (`train_abmil.py`)

Separate entry point — not dispatched through `train.py`:

```bash
python train_abmil.py --model-dir models/abmil_mpatch_v0_f
```

Uses `abmil_dataset.py` to build image-level bags from patch features extracted by a frozen or jointly-trained backbone.

---

## HPC / Slurm

Training on BYU HPC cluster (P100, V100, A100, H100, H200 nodes). GPU type is auto-detected at runtime; `gpu_utils.py` selects AMP dtype, GradScaler, and batch size accordingly (P100 capped at batch_size=32).

```bash
# Submit individual model
sbatch slurm/train_mpatch_v0_e.sh

# Batch submission
bash slurm/submit_mpatch_v0.sh                # all mpatch_v0 variants
bash slurm/submit_mpatch_v1e_125s.sh          # 125s base model
bash slurm/submit_mpatch_v1e_125s_reps.sh     # 7 CV replicates
bash slurm/submit_mpatch_v1e_lite_lc.sh       # 7 LC "a" models
bash slurm/submit_mpatch_v1e_lite_lc_bcd.sh   # 21 LC b/c/d models
bash slurm/submit_holdout_eval.sh             # chains holdout eval + aggregate

# Monitor
squeue -u $USER
scancel <jobid>
```

Each per-model script self-resubmits on wall-time (USR1 signal 5 min before limit) and on clean exit (epoch quota met, more epochs remain). Jobs are independent — each model trains on a single GPU.

---

## Weights & Biases

W&B logging is on by default. Jobs run in offline mode on the cluster:

```bash
export WANDB_MODE=offline
# After job completes, sync from login node:
wandb sync models/<type>/wandb/run-*/
# Or batch-sync:
bash slurm/sync_wandb.sh
```

Run ID is stored in each checkpoint and restored on resume — W&B runs are continuous across Slurm job boundaries.

---

## Evaluation

```bash
# Full-image models
python evaluate.py --model-type 5class
python evaluate_cosine.py --model-type 5class_hpc_v0

# Patch models (val or test split)
python evaluate_patch.py --model-type mpatch_v0_e --split val

# ABMIL
python evaluate_abmil.py --model-dir models/abmil_mpatch_v0_f
```

---

## Holdout Evaluation

Reproducible holdout testing, kept entirely separate from training-time val/test selection. Results are written to `models/<key>/holdout_eval/` and to `analysis/` for local inspection.

```bash
# 8-fold CV ensemble (125s family)
python evaluate_holdout_125s_ensemble.py
python evaluate_holdout_125s_ensemble.py --limit 5   # fast dry run

# All 28 LC models
python evaluate_holdout_lite_lc.py
python evaluate_holdout_lite_lc.py --models mpatch_v1e_lite_lc35a --limit 5

# Aggregate LC results into learning-curve plots
python aggregate_learning_curve.py

# Local analysis: recompute loss from saved logits, write analysis/
python analyze_holdout_results.py --which 125s
python analyze_holdout_results.py --which lc
python analyze_holdout_results.py --which all

# Or run both eval jobs + aggregate on HPC
bash slurm/submit_holdout_eval.sh
```

---

## Results

### Validation set (training-time val split)

| Model | Val set | Accuracy | Notes |
|---|---|---|---|
| 5class | n=200 | 57.7% | F08D hardest; top confusion F08D→F09D |
| 3class_finetune | n=120 | 70.8% | Fine-tuned from 5class; hemophilia classes only |
| 3class_scratch | n=120 | 62.5% | From-scratch 3-class |

Preprocessing sensitivity (5-class model, val): green=63.2% > luminance=61.7% > lab_l=57.7% > hsv_v=57.2% > hsv_s=30.9%

### Holdout test set (200 images, never seen during training or model selection)

| Model | Image accuracy | Per-class notes |
|---|---|---|
| **mpatch_v1e_125s 8-fold ensemble** | **82.5%** | F09D/NC1=100%, F08D=70%, AC3=77.5%, F11D=65% |
| mpatch_v1e_lite_lc35 (mean, 4 reps) | ~73% | Full LC training set (35 experiments) |
| mpatch_v1e_lite_lc05 (mean, 4 reps) | ~42% | Smallest LC training set (5 experiments) |

Power-law scaling fit over the LC series projects ~84% image accuracy (95% CI ~76–86%) at approximately 110 training experiments (3× the current training set size).

Confusion matrices, per-image predictions, and per-class breakdowns are in `analysis/125s_ensemble/` and `analysis/learning_curve/`.

---

## Analysis Suite (full-image models)

```bash
python run_analysis.py --model-type 5class
```

Runs 11 steps: per-class metrics, misclassification report, activation maps, preprocessing comparison, Grad-CAM saliency maps, ROC curves, annotated images, weight visualization, training curves, multi-model comparison. Outputs to `models/<type>/analysis/`.

---

## File Structure

```
# Core pipeline — full-image
preprocessing.py             Grayscale + 10× min-pool (also 2×/1.25× variants)
preprocessing_frangi.py      Multi-channel Frangi pipeline (7-channel)
preprocessing_full.py        Full-resolution pipeline (no pooling, (1,4000,6000))
augmentation.py              4-fold flip augmentations (full-image)
data_loader.py               DB access, FibrinDataset, split helpers
model.py                     FibrinCNN (~455K params) + NormalizedLinear + FibrinCNNCosine

# Core pipeline — patch-based
augmentation_patch.py        Kornia GPU augmentation (rotation, crop, flips, jitter, noise)
patch_dataset.py             FibrinPatchDataset, 3-way split, kfold CV, LC subset helper
masked_patch_dataset.py      MaskedPatchDataset — content-aware sampling (10× min-pool)
masked_patch_dataset_125s.py MaskedPatchDataset for 1.25× area-resize images
masked_patch_dataset_2x.py   MaskedPatchDataset for 2× min-pool images
masked_patch_dataset_full.py MaskedPatchDataset for full-resolution images
model_patch.py               FibrinPatchCNN (~900K params)
model_patch_125s.py          FibrinPatchCNN for 1.25× scale (250×250 patches)
model_patch_2x.py            FibrinPatchCNN for 2× min-pool (1000×1000 patches)
model_patch_full.py          FibrinPatchCNNFull for full-resolution (2000×2000 patches)
compute_masks.py             Precompute foreground masks → masks/<version>/
compute_patch_centers.py     Precompute patch-center validity maps → masks/patch_centers/

# ABMIL pipeline
abmil_dataset.py             Bag dataset for attention-based MIL
abmil_model.py               ABMIL model (attention pooling over patch features)
train_abmil.py               ABMIL training entry point (separate from train.py)
evaluate_abmil.py            ABMIL evaluation
compare_voting_methods.py    Compare bag-aggregation / voting strategies

# Shared utilities
gpu_utils.py                 GPU detection + AMP/compile config per SM version
lr_schedulers.py             LR scheduler utilities
checkpoint_manager.py        Checkpoint save/load with top-k ensemble support
configs/training_configs.py  All model hyperparameter configs (base dicts + 50+ entries)

# Training entry points
train.py                     Unified dispatcher (--model-type selects pipeline)
train_5class.py              5-class full-image training loop
train_5class_hpc.py          HPC-optimized 5-class (cosine head, W&B, checkpointing)
train_3class.py              3-class hemophilia (scratch or finetune)
train_patch.py               Patch and masked-patch unified training loop

# Evaluation
evaluate.py                  Per-class metrics, confusion matrix (full-image)
evaluate_cosine.py           Evaluation for cosine-head full-image models
evaluate_patch.py            Patch soft-vote grid inference (val/test split)

# Holdout evaluation (reproducible testing, separate from training-time val/test)
evaluate_holdout_125s_ensemble.py  8-fold CV ensemble holdout eval (125s family)
evaluate_holdout_lite_lc.py        All 28 LC models on shared holdout set
aggregate_learning_curve.py        Aggregate 28 holdout summaries → learning-curve CSV/plots
analyze_holdout_results.py         Local analysis: recompute loss, write analysis/ at repo root
holdout_eval_utils.py              Shared helpers: 125s loader, best-epoch extraction, split checks

# Analysis scripts
run_analysis.py              Full 11-step analysis suite (full-image models)
hemophilia_analysis.py       ROC curves, annotated images, multi-model compare
gradcam.py                   Grad-CAM saliency maps
visualize_weights.py         CNN kernel heatmaps + activation maps
activation_analysis.py       Spatial activation statistics
misclassification_report.py  Per-image error analysis + confusion matrices
preprocessing_comparison.py  Grayscale method robustness comparison
plot_training_curves.py      Train/val loss and accuracy curves
analysis_utils.py            Model registry + shared inference utilities
replay_wandb.py              Replay offline W&B runs to cloud

# Utilities / diagnostics
tune_patch_threshold.py      Sweep min_fg thresholds for mask tuning
visualize_patch_centers.py   Visualize valid patch center maps
inspect_masks.py             Inspect foreground mask quality
patch_validate.py            Validate patch sampling distribution
generate_figs.py             Publication figure generation
figs_best_class.py           Per-class best-image figures

# Tests
test_preprocessing.py        Grayscale methods, min-pool, visual outputs
test_preprocessing_frangi.py Frangi pipeline tests
test_augmentation.py         Flip augmentation tests
test_data_loader.py          DB, split, Dataset unit tests
test_patch_pipeline.py       Patch dataset, model, augmentation, split tests
test_lr_scheduler.py         LR scheduler tests
test_abmil_pipeline.py       ABMIL pipeline tests
test_holdout_eval.py         Holdout eval grid geometry, preprocessing shapes, regression tests

# Slurm
slurm/setup_env.sh                        One-time conda env creation on HPC
slurm/submit_all.sh                       Submit all standard training jobs
slurm/submit_mpatch_v0.sh                 Submit all mpatch_v0 variants
slurm/submit_mpatch_v1e_125s.sh           Submit 125s base model
slurm/submit_mpatch_v1e_125s_reps.sh      Submit 7 CV replicates
slurm/submit_mpatch_v1e_lite_lc.sh        Submit 7 LC "a" models
slurm/submit_mpatch_v1e_lite_lc_bcd.sh    Submit 21 LC b/c/d models
slurm/submit_holdout_eval.sh              Fan-out wrapper: chains eval jobs + aggregate
slurm/eval_mpatch_v1e_125s_ensemble.sh    125s ensemble holdout eval (single Slurm job)
slurm/eval_mpatch_v1e_lite_lc.sh          All-28-model LC holdout eval (single Slurm job)
slurm/eval_mpatch_v1e_lite_lc_aggregate.sh  Learning-curve aggregation (CPU-only)
slurm/train_*.sh                          Per-model self-resubmitting job scripts (~90 files)
slurm/sync_wandb.sh                       Sync offline W&B runs to cloud
slurm/sync_wandb_cron.sh                  Cron-based W&B sync helper
slurm/logs/                               Slurm stdout/stderr logs

# Data and outputs
data/photos/                 6000×4000 JPEG images (0000–0999.JPG)
data/endpoint10.db           SQLite label database
data/img-metadata.csv        Per-image metadata
data/exp-metadata.csv        Per-experiment metadata
masks/v_intensity/           Precomputed foreground masks
masks/patch_centers/         Precomputed patch-center validity maps

# Analysis outputs (tracked in git; regenerated by analyze_holdout_results.py)
analysis/125s_ensemble/      125s ensemble summary (accuracy+loss, per-class), confusion matrices
analysis/learning_curve/     LC learning-curve plots, power-law fit report, summary CSV/JSON

# Model output directories (weights excluded from git; holdout_eval/ dirs tracked)
models/5class/               Full-image 5-class model
models/5class_hpc_v0/        Cosine full-image model
models/5class_hpc_v1{a,b,c}/ Cosine variant models
models/3class_hemo/          3-class from-scratch model
models/3class_hemo_finetune/ 3-class fine-tuned model
models/patch_v0/             Patch cosine baseline
models/patch_5class_v1{a,b,c}/  Patch cosine variants
models/patch_ce_v2{a,b,c}/  Patch CE variants
models/mpatch_v0_{a..i}/     Masked-patch baseline series
models/mpatch_v0{e,f,g,h}_lite/  Lite variants
models/mpatch_v0_f_aug/ mpatch_v0e_lite_aug/ mpatch_v0f_lite_aug/ mpatch_v0f_lite_aug2/  Aug variants
models/mpatch_v0_f1a/        Fine-tuned from mpatch_v0_f
models/mpatch_v0_{e,f}_ensw/ Top-k checkpoint ensemble variants
models/mpatch_v1_full_{a,b}/ Full-resolution models
models/mpatch_v1e_2xmp/      2× min-pool model
models/mpatch_v1e_125s/      1.25× resize model (8-fold CV base)
models/mpatch_v1e_125s_{flat,plateau}/  LR-schedule variants
models/mpatch_v1e_125s_rep_{a..g}/  7 CV replicates
models/mpatch_v1e_125s_ensemble/holdout_eval/  Ensemble holdout results + copied weights
models/mpatch_v1e_lite_lc{05..35}{a,b,c,d}/   28 LC models
models/mpatch_v1e_lite_lc*/holdout_eval/       Per-LC-model holdout results
models/mpatch_v1e_lite_lc_learning_curve/      Aggregated LC plots + summary
```

---

## Tests

```bash
pytest -v -s                              # all tests
pytest test_patch_pipeline.py -v -s       # patch pipeline only
pytest test_holdout_eval.py -v -s         # holdout eval geometry + regression tests
pytest test_abmil_pipeline.py -v -s       # ABMIL pipeline tests
```
