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
preprocessing.py             Grayscale + 10× min-pool
preprocessing_frangi.py      Multi-channel Frangi pipeline (7-ch, 4× mean-pool)
augmentation.py              4-fold flip augmentations (full-image)
data_loader.py               DB access, FibrinDataset, split helpers
model.py                     FibrinCNN architecture (~455K params)

# Core pipeline — patch-based
augmentation_patch.py        Kornia GPU augmentation module for patches
patch_dataset.py             FibrinPatchDataset (uniform random sampling)
masked_patch_dataset.py      MaskedPatchDataset (content-aware sampling)
model_patch.py               FibrinPatchCNN architecture (~900K params)
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
evaluate_patch.py            Patch cosine soft-vote grid inference

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

# Slurm
slurm/setup_env.sh           One-time conda env creation on HPC
slurm/submit_all.sh          Submit all standard training jobs
slurm/submit_mpatch_v0.sh    Submit all mpatch_v0 jobs
slurm/train_*.sh             Per-model self-resubmitting job scripts
slurm/sync_wandb.sh          Sync offline W&B runs
slurm/logs/                  Job stdout/stderr

# Data and outputs
environment.yml              Conda environment
data/photos/                 Raw images (0000–0999.JPG)
data/endpoint10.db           SQLite labels
data/img-metadata.csv        Per-image metadata CSV
data/exp-metadata.csv        Per-experiment metadata CSV
masks/v_intensity/           Precomputed foreground masks
masks/patch_centers/         Precomputed patch-center validity maps
models/5class/               5-class full-image model artifacts
models/5class_hpc_v0/        Cosine 5-class HPC model
models/5class_hpc_v1a-c/     Cosine 5-class variant models
models/3class_hemo/          3-class from-scratch model artifacts
models/3class_hemo_finetune/ 3-class fine-tuned model artifacts
models/patch_5class_v1a-c/   Patch cosine models
models/patch_ce_v2a-c/       Patch CE models
models/mpatch_v0_a-i/        Masked-patch models
test_output/                 Visual outputs from preprocessing/augmentation tests
agent/                       Historical planning documents (patch_cnn_plan.md, etc.)
```

## CNN Architecture
```
Input: 1 × 600 × 400 (grayscale, after 10× min-pool)
Block 1: Conv2d(1→32,   5×5, pad=2) → BN → ReLU → MaxPool(2×2)  → 32  × 300 × 200
Block 2: Conv2d(32→64,  5×5, pad=2) → BN → ReLU → MaxPool(2×2)  → 64  × 150 × 100
Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  75 ×  50
Block 4: Conv2d(128→256,3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 256 ×  37 ×  25
AdaptiveAvgPool2d(1,1) → 256
Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→5)
```
Receptive field after Block 4: **520 px** in original space (requirement: ≥300 px).

## Preprocessing

### Original pipeline (`preprocessing.py`)
1. Grayscale via CIE LAB L-channel (`lab_l`, default; also `luminance`, `hsv_v`, `hsv_s`, `green`)
2. 10× **min-pool**: `img.reshape(H//10, 10, W//10, 10).min(axis=(1,3))` → 600×400
3. Normalize to float32 [0, 1], add channel dim → tensor shape `(1, 400, 600)`

### Frangi multi-channel pipeline (`preprocessing_frangi.py`)
1. 4× **mean-pool** (color BGR): `img.reshape(H//4,4,W//4,4,3).mean(axis=(1,3))` → 1000×1500
2. Grayscale (lab_l default) → float32 [0, 1]
3. Frangi filter at σ = 1, 2, 4, 6, 10, 20 (one channel each, single-scale per call)
4. Stack → tensor shape `(7, 1000, 1500)` — channel 0 = gray, channels 1–6 = Frangi σs
- `CHANNEL_LABELS = ["gray_lab_l", "frangi_s1", "frangi_s2", "frangi_s4", "frangi_s6", "frangi_s10", "frangi_s20"]`
- Memory: ~42 MB per image (vs ~1 MB for original). Requires reduced batch size for training.
- Factory: `make_frangi_preprocessor()` → callable for FibrinDataset

## Augmentation
4 variants (applied in Dataset via `idx % 4`):
- 0 = identity, 1 = h-flip, 2 = v-flip, 3 = both flips

## Train/Validation Split
- **Experiment-level**: all images from one experiment stay together (prevents leakage)
- Exactly 8 of 10 experiments per class used for training (80% train, 20% val)
- Split saved to `train_record.json`; downstream scripts always load from this file
- `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)` for class imbalance

## Hyperparameters (train_5class.py)
| Parameter    | Value  |
|--------------|--------|
| batch_size   | 16     |
| lr (Adam)    | 1e-3   |
| weight_decay | 1e-4   |
| num_epochs   | 30     |
| Scheduler    | ReduceLROnPlateau(patience=5, factor=0.5) |

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
```

## data_loader.py Key Exports
- `load_split_from_record(record_path, db_path)` — reconstruct (train_df, val_df) from saved JSON; never re-randomizes
- `filter_classes(df, classes)` — subset DataFrame to specified class names

## analysis_utils.py Key Exports
- `MODEL_REGISTRY` — maps `"5class"/"3class_scratch"/"3class_finetune"` → (model_dir, num_classes, class_map)
- `load_model_from_registry(model_type, device)` — returns (model, model_dir, num_classes, class_map, class_names)
- `get_val_split(model_type, db_path)` — loads val_df from train_record.json; filters to hemo classes for 3-class models (`get_test_split` is a backwards-compat alias)
- `run_inference_full(model, val_df, photos_dir, device, preprocessor, class_map, class_names)` → DataFrame
  - Columns: idx, Experiment, Exp_Type, Slide_Type, true_label, pred_label, correct, confidence, prob_<class>...

## Trained Model Results (validation set)
| Model              | Val Set | Overall Acc | F08D Acc | Notes |
|--------------------|---------|-------------|----------|-------|
| 5class             | n=200   | 57.7%       | 15.0%    | F08D hardest; top confusion F08D→F09D |
| 3class_hemo_finetune | n=120 | 70.8%       | —        | Fine-tuned from 5class weights (F08D/F09D/F11D only) |
| 3class_hemo        | n=120   | 62.5%       | —        | From-scratch 3-class |

Preprocessing sensitivity (5-class model, validation set):
green=63.2% > luminance=61.7% > lab_l=57.7% > hsv_v=57.2% > hsv_s=30.9%

## Mask Pipeline
- **Foreground masks** (`compute_masks.py`): precomputed per image, stored in `masks/<version>/`
  - `v_intensity`: light-background intensity threshold (primary)
  - `v_frangi`, `v_entropy`: alternative mask methods
- **Patch-center validity** (`compute_patch_centers.py`): inscribed-circle rule — center valid if ≥`min_fg` fraction of circle (radius=100 px) is foreground. Stored in `masks/patch_centers/<version>/`.
  - `p200_circle_v_intensity`: 200 px diameter, v_intensity mask, min_fg=0.03 (3%)
- **MaskedPatchDataset** (`masked_patch_dataset.py`): uses validity maps to blend content-weighted and uniform-per-image patch allocation.
  - `uniform_fraction`: fraction of patches drawn uniformly per image (default 0.20 in a–d; ablated in e–h)
  - `val_uniform_fraction`: always 1.0 (new default) for fair cross-model comparison; a–d used 0.20 (legacy)
  - `preload_device`: optional `torch.device` — preloads unpadded (400×600) float32 tensors onto GPU;
    requires `num_workers=0` in DataLoader (workers are separate processes, cannot access CUDA tensors)
  - **CRITICAL**: when `include_grid=True`, grid images are filtered to `split_exps` only (see warning above)

## GPU Training (HPC)
- Cluster: BYU HPC — V100 (SM 7.x), A100 (SM 8.x), H100/H200 (SM 9.x)
- `gpu_utils.get_gpu_config()`: returns AMP dtype, GradScaler flag, batch size, compile flag per GPU
  - SM 7.x (V100): fp16, GradScaler, no compile
  - SM 8.x (A100): bf16, no GradScaler, no compile (Triton needs `cuda-cudart-dev` in conda to enable)
  - SM 9.x (H100/H200): bf16, no GradScaler, no compile (same)
- `force_compile: True` in a model config overrides gpu_utils and calls `torch.compile(model, backend=compile_backend)`
- `mpatch_v0_i` uses `compile_backend: "cudagraphs"` as a Triton-free compile test

## Known Issues / Fixes Applied
- `ReduceLROnPlateau(verbose=True)` removed — argument dropped in PyTorch 2.4+
- `np.trapz` → `np.trapezoid` — renamed in NumPy 2.0
- No sklearn dependency; metrics computed from scratch in `evaluate.py`
- `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler('cuda', ...)` — deprecated in PyTorch 2.7+
- BatchNorm eval mode: always call `model.eval()` before inference; `predict_all()` in `evaluate.py` does this defensively
- `torch.compile` disabled by default on all GPU types — Triton requires `cuda.h` from `cuda-cudart-dev` conda package which is not installed; install with `CONDA_NO_PLUGINS=true conda install -c nvidia cuda-cudart-dev` then re-enable in `gpu_utils.py`
