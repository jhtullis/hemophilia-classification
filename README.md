# Fibrin Clot CNN Classification

Classify microscopy images of fibrin clots across five blood phenotypes using a convolutional neural network in PyTorch.

---

## Classes

| Label | Phenotype | Count |
|-------|-----------|-------|
| NC1   | Normal control | 200 |
| AC3   | Prolonged clotting time | 200 |
| F08D  | Factor VIII deficient | 200 |
| F09D  | Factor IX deficient | 200 |
| F11D  | Factor XI deficient | 200 |

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fibrin
```

---

## Data

- **Images**: `data/photos/0000.JPG` – `0999.JPG` (6000 × 4000 px JPEG, landscape)
- **Labels**: `data/endpoint10.db` — SQLite, table `Images_Endpoint_10`
  - Filename integer = `rowid - 1` (row 1 → `0000.JPG`)
  - Key columns: `Experiment`, `Exp_Type` (class), `Slide_Type` (A/B)
- **50 distinct experiments** (10 per class); all images from one experiment stay in the same partition

---

## Preprocessing Pipeline

Every image is normalized to landscape orientation, then:

1. **Grayscale** — L channel from CIE LAB color space (`lab_l`, default). Alternatives (`luminance`, `hsv_v`, `hsv_s`, `green`) are compared in `test_preprocessing.py`.
2. **10× min-pool** — Reduces 6000×4000 → 600×400. Min-pool preserves thin dark fibers on a light background.
3. **Normalize** — Scale uint8 to float32 in [0, 1]; add channel dim → tensor `(1, 400, 600)`.

A second **Frangi multi-channel pipeline** (`preprocessing_frangi.py`) is also implemented: 4× mean-pool + 6 Frangi filter scales → 7-channel tensor `(7, 1000, 1500)`.

---

## CNN Architecture

```
Input:  1 × 600 × 400  (grayscale, after 10× min-pool)

Block 1: Conv2d(1→32,   5×5, pad=2) → BN → ReLU → MaxPool(2×2)  →  32 × 300 × 200
Block 2: Conv2d(32→64,  5×5, pad=2) → BN → ReLU → MaxPool(2×2)  →  64 × 150 × 100
Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  75 ×  50
Block 4: Conv2d(128→256,3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 256 ×  37 ×  25

AdaptiveAvgPool2d(1,1) → 256
Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→num_classes)
```

~455K parameters (5-class) / ~455K (3-class). Receptive field after Block 4: **520 px** in original image space (requirement: ≥300 px).

---

## Data Augmentation

Training only. Each image generates 4 variants:

| Variant | Transform |
|---------|-----------|
| 0 | Original |
| 1 | Horizontal flip |
| 2 | Vertical flip |
| 3 | Both flips (180°) |

---

## Training

Three model types are available via `train.py`:

```bash
python train.py --model-type 5class             # 5-class model → models/5class/
python train.py --model-type 3class_scratch     # 3-class from random init
python train.py --model-type 3class_finetune    # fine-tune 5-class → 3-class
python train.py --model-type 5class --preload   # cache images in RAM (faster on CPU)
python train.py --model-type 5class --force-resplit  # regenerate train/val split
```

The 3-class models train on F08D, F09D, F11D only, reusing the same experiment-level split as the 5-class model. The fine-tune variant uses a two-phase schedule: head-only warmup (5 epochs), then full unfreeze (25 epochs).

**Hyperparameters (5-class):**

| Parameter | Value |
|-----------|-------|
| `batch_size` | 16 |
| `lr` | 1e-3 (Adam) |
| `weight_decay` | 1e-4 |
| `num_epochs` | 30 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

**Outputs** (per model directory):
- `best_model.pth` — weights at highest validation accuracy
- `train_record.json` — canonical train/validation split (never re-randomized)
- `training_history.json` — per-epoch loss and accuracy
- `training_log.txt` — full stdout log

---

## Evaluation

```bash
python evaluate.py --model-type 5class
python evaluate.py --model-type 3class_finetune
```

Prints per-class precision, recall, F1, support, and the confusion matrix.

---

## Analysis Suite

Run all 11 analysis steps in order with a single command:

```bash
python run_analysis.py --model-type 5class
python run_analysis.py --model-type 3class_finetune
```

Steps (outputs go to `models/<type>/analysis/`):

| Step | Script | Output |
|------|--------|--------|
| 1 | `evaluate.py` | Per-class metrics, confusion matrix |
| 2 | `misclassification_report.py` | Per-image error analysis |
| 3 | `activation_analysis.py` | Spatial activation maps |
| 4 | `preprocessing_comparison.py` | Accuracy by grayscale method |
| 5 | `gradcam.py` | Class representative + misclassified saliency maps |
| 6 | `gradcam.py --all-overlays` | Full-resolution Grad-CAM overlay per test image |
| 7 | `hemophilia_analysis.py --roc` | OVR ROC / AUC curves |
| 8 | `hemophilia_analysis.py --annotate` | Annotated test image copies |
| 9 | `visualize_weights.py` | Kernel heatmaps + activation maps |
| 10 | `plot_training_curves.py` | Train/val loss and accuracy curves |
| 11 | `hemophilia_analysis.py --compare` | Multi-model ROC overlay |

---

## Tests

```bash
pytest -v -s
```

| Test file | What it checks |
|-----------|---------------|
| `test_preprocessing.py` | Grayscale methods, min-pool, visual PNGs → `test_output/` |
| `test_preprocessing_frangi.py` | Frangi pipeline channels, visual PNGs → `test_output/` |
| `test_augmentation.py` | All 4 flip variants, self-inverse property, shape preservation |
| `test_data_loader.py` | 1000 rows, correct class counts (200 per class), no experiment overlap between splits |

---

## File Structure

```
preprocessing.py             Grayscale + 10× min-pool (original pipeline)
preprocessing_frangi.py      Multi-channel Frangi pipeline (7-ch, 4× mean-pool)
augmentation.py              4-fold flip augmentations
data_loader.py               DB access, Dataset, balanced DataLoader, split helpers
model.py                     FibrinCNN architecture
train.py                     Unified training entry point
train_5class.py              5-class training loop
train_3class.py              3-class hemophilia training (scratch or finetune)
evaluate.py                  Per-class metrics and confusion matrix
run_analysis.py              Orchestrates all 11 analysis steps in order
hemophilia_analysis.py       ROC curves, annotated images, multi-model compare
visualize_weights.py         CNN kernel heatmaps + activation maps
plot_training_curves.py      Train/val loss and accuracy curves
analysis_utils.py            Shared inference utilities and model registry
misclassification_report.py  Per-image misclassification analysis
activation_analysis.py       CNN activation map spatial analysis
preprocessing_comparison.py  Preprocessing method robustness comparison
gradcam.py                   Grad-CAM class saliency maps + full overlay batch
test_preprocessing.py        Visual + structural tests (original pipeline)
test_preprocessing_frangi.py Tests + visual outputs for Frangi pipeline
test_augmentation.py         Visual + property tests
test_data_loader.py          DB, split, and Dataset unit tests
AGENT_EXPORT.md              Context primer for new Claude Code sessions
environment.yml              Conda environment specification
data/photos/                 6000×4000 JPEG images
data/endpoint10.db           SQLite label database
models/5class/               5-class model artifacts
models/3class_hemo/          3-class from-scratch model artifacts
models/3class_hemo_finetune/ 3-class fine-tuned model artifacts
test_output/                 Visual outputs from preprocessing and augmentation tests
```

---

## Potential Improvements

- **Transfer learning**: pre-trained ImageNet weights (grayscale → 3-channel duplication) could improve accuracy on this small dataset.
- **Additional augmentation**: random brightness/contrast jitter or small rotations (if rotational symmetry holds beyond 0°/90°/180°/270°).
- **Larger dataset**: more experimental replicates would be the highest-leverage improvement.
- **Slide type as auxiliary signal**: `Slide_Type` (A/B) is a known covariate that could be incorporated as metadata.

---

## Authorship Note

This repository was developed with extensive use of [Claude Code](https://claude.ai/code) (Anthropic), with active review, participation, and scientific guidance by the human author throughout.
