# Fibrin Clot CNN Classification

Classify brightfield microscopy images of fibrin clots across five blood phenotypes using PyTorch CNNs. This branch (`dev-hpc0`) adds GPU-accelerated patch-based training on the BYU HPC cluster alongside the original full-image pipeline.

---

## Scientific Background

Fibrin clots form when blood plasma coagulates in response to injury. The resulting fibrin network — its fiber density, thickness, branching, and pore size — varies measurably between individuals depending on their coagulation factor profile. In hemophilia, key coagulation factors are absent or deficient, producing clots with a distinctively altered microstructure visible under brightfield microscopy.

This project tests whether a CNN can learn to discriminate these structural differences from images alone, with particular interest in distinguishing between the three hemophilia subtypes (A, B, C).

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

Key dependencies: PyTorch, Kornia, OpenCV, SQLAlchemy, pandas, NumPy, scikit-image, wandb, pytest.

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
3. Normalize to float32 [0, 1]; add channel dim → tensor `(1, 400, 600)`

### Frangi multi-channel pipeline (`preprocessing_frangi.py`)
4× mean-pool + Frangi filter at σ = 1, 2, 4, 6, 10, 20 → 7-channel tensor `(7, 1000, 1500)`.

### Foreground masks (`compute_masks.py`)
Precompute per-image binary foreground masks using intensity, Frangi, and entropy methods. Stored in `masks/<version>/`. Used by the masked-patch pipeline to restrict patch sampling to fibrin-rich regions.

### Patch-center validity maps (`compute_patch_centers.py`)
For each image, determine which patch centers (200 px radius inscribed-circle rule) have ≥3% foreground coverage. Stored in `masks/patch_centers/<version>/`. Used by `MaskedPatchDataset`.

---

## Architectures

### FibrinCNN (`model.py`)
Full-image model. Input: `(1, 400, 600)` grayscale.
```
Block 1: Conv2d(1→32,  5×5) → BN → ReLU → MaxPool  →  32×300×200
Block 2: Conv2d(32→64, 5×5) → BN → ReLU → MaxPool  →  64×150×100
Block 3: Conv2d(64→128,3×3) → BN → ReLU → MaxPool  → 128×75×50
Block 4: Conv2d(128→256,3×3)→ BN → ReLU → MaxPool  → 256×37×25
AdaptiveAvgPool2d(1,1) → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→N)
```
~455K parameters. Receptive field: 520 px in original image space.

### FibrinPatchCNN (`model_patch.py`)
Patch-based model. Input: `(1, 200, 200)`.
```
Block 1: Conv3×3 → Conv3×3 → Conv3×3(stride=2) → BN/ReLU  →  32×100×100
Block 2: Conv3×3 → Conv3×3 → Conv3×3(stride=2) → BN/ReLU  →  64×50×50
Block 3: Conv3×3 → Conv3×3 → Conv3×3(stride=2) → BN/ReLU  → 128×25×25
Conv4:   Conv3×3                                → BN/ReLU  → 256×25×25
AdaptiveAvgPool2d(1,1) → Linear(256→128) → ReLU → Dropout → head
```
~900K parameters. Receptive field: 590 px in original image space.

Two head types (selected per config):
- **Cosine head** (`NormalizedLinear`): outputs cosine similarities in [−1, 1]; trained with `CosineLoss`
- **CE head** (`Linear`): standard cross-entropy logits

---

## Training

All training is dispatched through `train.py`:

```bash
python train.py --model-type <type> [--resume] [--preload] [--max-epochs N] [--epochs-per-job N]
```

### Full-image models

| Model type | Description |
|---|---|
| `5class` | FibrinCNN, 5 classes, ReduceLROnPlateau, 30 epochs |
| `5class_hpc_baseline` | Same, saved to separate dir for HPC comparison |
| `5class_hpc_v0` | FibrinCNNCosine, cosine head, CosineAnnealingWarmRestarts |
| `5class_hpc_v1a/b/c` | Variants of cosine 5-class with tuned hyperparameters |
| `3class_scratch` | FibrinCNN(3 classes) from random init (F08D/F09D/F11D only) |
| `3class_finetune` | Fine-tune 5-class weights → 3-class head |

### Standard patch models

| Model type | Head | Grid | Notes |
|---|---|---|---|
| `patch_v0` | cosine | — | Baseline patch model |
| `patch_v1a/b/c` | cosine | — | Ablation variants |
| `patch_v2a/b/c` | CE | — | Cross-entropy head variants |

### Masked-patch models (content-aware sampling)

Patches are sampled preferentially from fibrin-rich image regions using precomputed validity maps. `uniform_fraction` controls the blend between content-weighted and uniform-per-image sampling. Validation always uses `uniform_fraction=1.0` (equal patches per image) for fair cross-model comparison.

| Model | Head | Grid | `uniform_fraction` | Notes |
|---|---|---|---|---|
| `mpatch_v0_a` | cosine | no  | 0.20 | val uses 0.20 (legacy) |
| `mpatch_v0_b` | cosine | yes | 0.20 | val uses 0.20 (legacy) |
| `mpatch_v0_c` | CE     | no  | 0.20 | val uses 0.20 (legacy) |
| `mpatch_v0_d` | CE     | yes | 0.20 | val uses 0.20 (legacy) |
| `mpatch_v0_e` | CE     | yes | 0.10 | val uses 1.0 |
| `mpatch_v0_f` | CE     | yes | 0.00 | val uses 1.0; fully content-weighted |
| `mpatch_v0_g` | CE     | yes | 0.05 | val uses 1.0 |
| `mpatch_v0_h` | CE     | yes | 0.01 | val uses 1.0 |
| `mpatch_v0_i` | CE     | yes | 0.20 | compile test; `cudagraphs` backend |

Grid models include both exploratory and grid-acquisition images during training.

---

## HPC / Slurm

Training on BYU HPC cluster (V100, A100, H100, H200 nodes).

```bash
# Submit individual jobs
sbatch slurm/train_mpatch_v0_e.sh

# Submit multiple independently
for m in e f g h; do sbatch slurm/train_mpatch_v0_${m}.sh; done

# Monitor
squeue -u $USER
scancel <jobid>
```

Each script self-resubmits on wall-time (USR1 signal 5 min before limit) and on clean exit (epoch quota met, more epochs remain). Jobs are independent — each model trains on a single GPU.

GPU optimization (`gpu_utils.py`): BF16 AMP on Ampere/Hopper, FP16+GradScaler on Volta, `cudnn.benchmark=True` on all CUDA GPUs.

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
python evaluate.py --model-type 3class_finetune

# Cosine full-image
python evaluate_cosine.py --model-type 5class_hpc_v0

# Patch models (val or test split)
python evaluate_patch.py --model-type patch_v1a --split val
python evaluate_patch.py --model-type mpatch_v0_e --split val
```

### Trained model results (validation set)

| Model | Val set | Accuracy | Notes |
|---|---|---|---|
| 5class | n=200 | 57.7% | F08D hardest; top confusion F08D→F09D |
| 3class_finetune | n=120 | 70.8% | Fine-tuned from 5class; hemophilia classes only |
| 3class_scratch | n=120 | 62.5% | From-scratch 3-class |

Preprocessing sensitivity (5-class, val): green=63.2% > luminance=61.7% > lab_l=57.7% > hsv_v=57.2% > hsv_s=30.9%

---

## Analysis Suite (full-image models)

```bash
python run_analysis.py --model-type 5class
```

Runs 11 steps: metrics, misclassification, activation maps, preprocessing comparison, Grad-CAM, ROC curves, annotated images, weight visualization, training curves, multi-model comparison. Outputs to `models/<type>/analysis/`.

---

## File Structure

```
# Core pipeline
preprocessing.py             Grayscale + 10× min-pool
preprocessing_frangi.py      Multi-channel Frangi pipeline
augmentation.py              4-fold flip augmentations (full-image)
augmentation_patch.py        Kornia GPU augmentation for patches
data_loader.py               DB access, FibrinDataset, split helpers
model.py                     FibrinCNN (full-image)
model_patch.py               FibrinPatchCNN (patch-based)
patch_dataset.py             FibrinPatchDataset — uniform random sampling
masked_patch_dataset.py      MaskedPatchDataset — content-aware sampling
compute_masks.py             Precompute foreground masks
compute_patch_centers.py     Precompute valid patch centers
gpu_utils.py                 GPU detection + AMP/compile configuration
lr_schedulers.py             LR scheduler utilities
checkpoint_manager.py        Checkpoint save/load logic
configs/training_configs.py  All model hyperparameter configs

# Training entry points
train.py                     Unified dispatcher (--model-type selects pipeline)
train_5class.py              5-class full-image training loop
train_5class_hpc.py          HPC-optimized 5-class (cosine head, W&B, checkpointing)
train_3class.py              3-class hemophilia (scratch or finetune)
train_patch.py               Patch and masked-patch training loop

# Evaluation
evaluate.py                  Per-class metrics, confusion matrix (full-image)
evaluate_cosine.py           Evaluation for cosine-head full-image models
evaluate_patch.py            Cosine soft-vote inference over patch grid

# Analysis scripts
run_analysis.py              Full 11-step analysis suite
hemophilia_analysis.py       ROC curves, annotated images, multi-model compare
gradcam.py                   Grad-CAM saliency maps
visualize_weights.py         CNN kernel heatmaps + activation maps
activation_analysis.py       Spatial activation statistics
misclassification_report.py  Per-image error analysis
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

# Slurm
slurm/setup_env.sh           One-time conda env creation on HPC
slurm/submit_all.sh          Submit all standard training jobs
slurm/submit_mpatch_v0.sh    Submit all mpatch_v0 jobs
slurm/train_*.sh             Per-model job scripts (self-resubmitting)
slurm/sync_wandb.sh          Sync offline W&B runs to cloud
slurm/logs/                  Slurm stdout/stderr logs

# Data and outputs
data/photos/                 6000×4000 JPEG images (0000–0999.JPG)
data/endpoint10.db           SQLite label database
data/img-metadata.csv        Per-image metadata
data/exp-metadata.csv        Per-experiment metadata
masks/v_intensity/           Precomputed foreground masks
masks/patch_centers/         Precomputed patch-center validity maps
models/5class/               Full-image 5-class model
models/5class_hpc_v0/        Cosine full-image model
models/patch_5class_v1a/     Patch cosine model (v1a)
models/patch_ce_v2a/         Patch CE model (v2a)
models/mpatch_v0_*/          Masked-patch models
test_output/                 Visual outputs from preprocessing/augmentation tests
agent/                       Historical planning documents
old/                         Legacy evaluation outputs
```

---

## Tests

```bash
pytest -v -s                          # all tests
pytest test_patch_pipeline.py -v -s   # patch pipeline only
```

---

## AI Use Note

This repository was developed with extensive use of [Claude Code](https://claude.ai/code) (Anthropic). Claude Code wrote the code files and the majority of the documentation contained in this repository. The author, Jason Henry Tullis, provided the initial specification, along with active review, correction, and modification of AI outputs throughout the development process.
