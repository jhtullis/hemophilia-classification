# CLAUDE.md — Fibrin Clot CNN Project Reference

## Project Goal
Classify 859 microscopy images of fibrin clots across 5 phenotypes using a PyTorch CNN.

## Classes
| Label | Count |
|-------|-------|
| AC3   | 219   |
| F08D  | 120   |
| F09D  | 180   |
| F11D  | 180   |
| NC1   | 160   |

Class map (integer labels): AC3=0, F08D=1, F09D=2, F11D=3, NC1=4

## Data
- **Images**: `data/photos/0000.JPG` – `0858.JPG` (6000×4000 px JPEG)
- **Labels**: `data/test_db.db` → table `Images_Endpoint_10`
  - `rowid - 1` = filename integer (row 1 → `0000.JPG`)
  - Key columns: `Experiment`, `Exp_Type` (class label), `Slide_Type` (A/B)
- **41 distinct experiments**; images from the same experiment are correlated

## File Structure
```
preprocessing.py           Grayscale + 10× min-pool (original pipeline)
preprocessing_frangi.py    Multi-channel Frangi pipeline (7-ch, 4× mean-pool)
augmentation.py            4-fold flip augmentations
data_loader.py             DB access, Dataset, DataLoaders, split helpers
model.py                   FibrinCNN architecture (~455K params)
train_5class.py            5-class training loop
train_3class.py            3-class hemophilia CNN (--mode scratch|finetune)
evaluate.py                Per-class precision/recall/F1, confusion matrix
hemophilia_analysis.py     ROC curves + annotated images (multi-model)
visualize_weights.py       CNN kernel heatmaps + activation maps
analysis_utils.py          Shared inference DataFrame + model registry wrapper
misclassification_report.py  Module 2: per-image misclassification analysis
activation_analysis.py     Module 3: CNN activation map spatial analysis
preprocessing_comparison.py  Module 5: preprocessing robustness comparison
gradcam.py                 Module 6: Grad-CAM class saliency maps
test_preprocessing.py      Visual + structural tests (original pipeline)
test_preprocessing_frangi.py  Tests + visual outputs for Frangi pipeline
test_augmentation.py       Visual + property tests
test_data_loader.py        DB, split, Dataset unit tests
PLAN.md                    Architecture decisions and rationale
AGENT_EXPORT.md            Context primer for new Claude Code instances
agent/                     Modular analysis plan documents
environment.yml            Conda environment
data/photos/               Raw images
data/test_db.db            SQLite labels
models/5class/             5-class model artifacts
  best_model.pth             Saved weights
  train_record.json          Canonical train/test split (201 test images)
  analysis/                  Analysis outputs (misclassification/, activation/,
                               preprocessing/, gradcam/)
models/3class_hemo/        3-class from-scratch model artifacts (best acc 62.5%)
models/3class_hemo_finetune/ 3-class fine-tuned model artifacts
test_output/frangi/        Visual channel inspection outputs from Frangi tests
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

## Train/Test Split
- **Experiment-level**: all images from one experiment stay together (prevents leakage)
- Target ~75% train per class; F08D uses 5/6 experiments (~83%)
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
pytest -v -s                                  # run all tests
pytest test_preprocessing_frangi.py -v -s     # Frangi pipeline tests + visuals

# Training
python train_5class.py                        # train 5-class → models/5class/
python train_3class.py --mode scratch         # train 3-class from scratch
python train_3class.py --mode finetune        # fine-tune 5-class → 3-class

# Evaluation
python evaluate.py                            # detailed metrics on 5-class test set
python hemophilia_analysis.py --roc           # ROC for F08D/F09D/F11D
python hemophilia_analysis.py --compare       # compare all models side-by-side
python visualize_weights.py                   # kernel heatmaps + activation maps

# Analysis modules (outputs → models/<type>/analysis/<module>/)
python misclassification_report.py --model-type 5class
python activation_analysis.py      --model-type 5class
python preprocessing_comparison.py --model-type 5class
python gradcam.py                  --model-type 5class
```

## data_loader.py Key Exports
- `load_split_from_record(record_path, db_path)` — reconstruct (train_df, test_df) from saved JSON; never re-randomizes
- `filter_classes(df, classes)` — subset DataFrame to specified class names

## analysis_utils.py Key Exports
- `MODEL_REGISTRY` — maps `"5class"/"3class_scratch"/"3class_finetune"` → (model_dir, num_classes, class_map)
- `load_model_from_registry(model_type, device)` — returns (model, model_dir, num_classes, class_map, class_names)
- `get_test_split(model_type, db_path)` — loads from train_record.json; filters to hemo classes for 3-class models
- `run_inference_full(model, test_df, photos_dir, device, preprocessor, class_map, class_names)` → DataFrame
  - Columns: idx, Experiment, Exp_Type, Slide_Type, true_label, pred_label, correct, confidence, prob_<class>...

## Trained Model Results (5-class, test set n=201)
| Model              | Overall Acc | F08D Acc | Notes |
|--------------------|-------------|----------|-------|
| 5class             | 57.7%       | 15.0%    | F08D hardest; top confusion F08D→F09D |
| 3class_hemo_finetune | —         | —        | Fine-tuned from 5class weights |
| 3class_hemo        | 62.5% (train) / 35.8% (eval) | — | From-scratch 3-class |

Preprocessing sensitivity (5-class model, test set):
green=63.2% > luminance=61.7% > lab_l=57.7% > hsv_v=57.2% > hsv_s=30.9%

## Known Issues / Fixes Applied
- `ReduceLROnPlateau(verbose=True)` removed — argument dropped in PyTorch 2.4+
- `np.trapz` → `np.trapezoid` — renamed in NumPy 2.0
- No sklearn dependency; metrics computed from scratch in `evaluate.py`
- Machine has no GPU; training runs on CPU only
