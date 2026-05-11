# AGENT_EXPORT.md — Fibrin Clot CNN Project: Context Primer

This document is a comprehensive briefing for a new Claude Code session picking up
exploration of this project. It covers the biological context, data organization,
CNN pipeline, and current state of the codebase.

---

## Scientific Background

### What are fibrin clots?
Fibrin clots form when blood plasma clots in response to injury. The fibrin network
that makes up a clot is a polymer of fibrinogen, assembled into fibers that create
a mesh. The structure of this mesh (fiber density, thickness, branching pattern,
pore size) varies depending on the plasma phenotype of the donor.

### Why do these clots look different?
Several coagulation factors are required to convert fibrinogen into fibrin. When
key factors are absent or deficient (as in hemophilia), the resulting clot has a
measurably different microstructure observable under microscopy. The hypothesis
underlying this project is that a CNN can learn to distinguish these structural
differences from microscopy images alone.

### Class Labels and Biological Meaning
| Label | Condition | Count | Notes |
|-------|-----------|-------|-------|
| AC3   | Prolonged clotting time | 200 | Non-hemophilic control |
| F08D  | Factor VIII (8) deficient — Hemophilia A | 200 | |
| F09D  | Factor IX (9) deficient — Hemophilia B | 200 | |
| F11D  | Factor XI (11) deficient — Hemophilia C | 200 | |
| NC1   | Normal Control 1 — standard normal plasma | 200 | Non-hemophilic control |

The three hemophilia classes (F08D, F09D, F11D) are the primary scientific interest:
can the model distinguish between different hemophilia types? AC3 and NC1 are
normal/control phenotypes that the 5-class model must also accommodate.

**Integer label assignment** (hardcoded in `data_loader.py`):
`AC3=0, F08D=1, F09D=2, F11D=3, NC1=4`

---

## Data Organization

### Images
- Location: `data/photos/`
- Format: JPEG, named `0000.JPG` through `0999.JPG` (1000 images total)
- Resolution: **6000 × 4000 pixels** (landscape orientation)
- Content: microscopy images of fibrin clots — mostly light background with dark
  fibrous patches stretching across the frame. The fiber pattern, density, and
  thickness encode the phenotype information.
- Filename → row mapping: `int(filename_stem) = rowid - 1` in the database
  (e.g., `0000.JPG` = row 1, `0042.JPG` = row 43)

### Database
- Location: `data/endpoint10.db` (SQLite)
- Table: `Images_Endpoint_10`
- Key columns:
  - `rowid`: 1-based row index
  - `Experiment`: experiment ID string (groups correlated images)
  - `Exp_Type`: class label string (AC3, F08D, F09D, F11D, NC1)
  - `Slide_Type`: slide type (A or B — two slides per experiment)
  - `Img_Name`: original image filename
  - `Img_Fp`: original file path
- Access via SQLAlchemy: `load_metadata(engine)` returns a DataFrame with an
  added `idx` column (`rowid - 1`) used to construct filenames.

### Experiments
There are **50 distinct experiments** — 10 per class. Multiple images come from the
same experiment (different slides or fields of view). Images within the same
experiment are correlated — they represent the same plasma sample. This is critical:

> **The train/validation split is performed at the experiment level**, not the image
> level. All images from one experiment always land in the same partition. This
> prevents data leakage where correlated images from the same plasma sample appear
> in both train and validation sets.

Exactly 8 experiments per class are used for training and 2 for validation (80/20).

---

## Canonical Train/Validation Split

The split is generated once and saved to `models/5class/train_record.json`. All
downstream scripts reconstruct the same split from this file rather than
re-randomizing. This ensures fair, consistent evaluation across all model variants.

### Split record format (`train_record.json`)
```json
{
  "split_date": "2025-...",
  "seed": 42,
  "train_ratio": 0.8,
  "train_experiments": { "AC3": [...], "F08D": [...], ... },
  "val_experiments":   { "AC3": [...], "F08D": [...], ... },
  "train_indices": [0, 1, 3, ...],
  "val_indices":   [2, 8, ...],
  "class_distribution": {
    "train": { "AC3": 160, "F08D": 160, ... },
    "val":   { "AC3": 40,  "F08D": 40, ... }
  }
}
```

The `train_indices` and `val_indices` are the `idx` values (0-based integer image
indices), stored as sorted lists. `load_split_from_record(record_path, db_path)`
reconstructs exact `(train_df, val_df)` DataFrames from these lists.

### Split sizes (80/20, exact)
| Class | Train | Val |
|-------|-------|-----|
| AC3   | 160   | 40  |
| F08D  | 160   | 40  |
| F09D  | 160   | 40  |
| F11D  | 160   | 40  |
| NC1   | 160   | 40  |
| **Total** | **800** | **200** |

---

## Preprocessing Pipeline

1. **Grayscale conversion** — CIE LAB L-channel (`lab_l` default):
   `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]`
   Perceptually uniform, good contrast for fiber vs. background.
   Five methods are implemented (lab_l, luminance, hsv_v, hsv_s, green);
   `lab_l` was chosen as default.

2. **10× min-pool** — downsamples 6000×4000 → **600×400**:
   ```python
   img.reshape(H//10, 10, W//10, 10).min(axis=(1, 3))
   ```
   Min-pool (not mean-pool or max-pool) is critical: it preserves the dark
   fiber structure on a light background. Mean-pool would wash out thin fibers;
   max-pool would preserve only the brightest pixels (background).

3. **Normalize + tensorize** → `float32 [0,1]`, shape `(1, 400, 600)`
   Note: PyTorch convention is `(C, H, W)` so height (400) comes before width (600).

The preprocessor is created via `make_preprocessor(gray_method, pool_factor)` from
`preprocessing.py` and passed as a callable to `FibrinDataset`.

---

## Data Augmentation

Training images are augmented 4× using the 4-fold flip symmetry of rectangular images:
- Variant 0: identity (no flip)
- Variant 1: horizontal flip (`np.fliplr`)
- Variant 2: vertical flip (`np.flipud`)
- Variant 3: both flips (equivalent to 180° rotation)

In `FibrinDataset`, `augment=True` exposes each base image as 4 consecutive dataset
entries. The variant is `idx % 4`. This means the training dataset is logically
4× larger than the number of base images.

---

## CNN Architecture (FibrinCNN)

Defined in `model.py`. Accepts `num_classes` parameter (5 for full model, 3 for
hemophilia-only models).

```
Input: (N, 1, 600, 400)  — grayscale, min-pooled

model.features (Sequential, 16 modules):
  [0]  Conv2d(1→32,  5×5, pad=2)   [1]  BN   [2]  ReLU   [3]  MaxPool(2×2)
  →  (N, 32, 300, 200)

  [4]  Conv2d(32→64, 5×5, pad=2)   [5]  BN   [6]  ReLU   [7]  MaxPool(2×2)
  →  (N, 64, 150, 100)

  [8]  Conv2d(64→128,3×3, pad=1)   [9]  BN   [10] ReLU   [11] MaxPool(2×2)
  →  (N, 128, 75, 50)

  [12] Conv2d(128→256,3×3,pad=1)   [13] BN   [14] ReLU   [15] MaxPool(2×2)
  →  (N, 256, 37, 25)

model.global_pool: AdaptiveAvgPool2d(1,1)  →  (N, 256, 1, 1)  →  flatten  →  (N, 256)

model.classifier (Sequential, 4 modules):
  [0]  Linear(256→128)
  [1]  ReLU
  [2]  Dropout(0.5)
  [3]  Linear(128→num_classes)
```

**Receptive field**: 520 px in original 6000×4000 space (requirement was ≥300 px).
**Parameter count**: ~455K (5-class).

### Key architectural indices (important for visualization and fine-tuning)
- Conv layers in `model.features`: indices **0, 4, 8, 12**
- MaxPool layers (activation map hook points): indices **3, 7, 11, 15**
- Output classifier layer: `model.classifier[-1]` = `model.classifier[3]`

---

## Class Imbalance Handling

Three complementary strategies are used simultaneously:
1. `WeightedRandomSampler` — oversample minority classes during training
2. `CrossEntropyLoss(weight=class_weights)` — inverse-frequency loss weights
3. 4× uniform augmentation across all classes (does not worsen imbalance)

---

## Model Directory Structure

```
models/
  5class/
    best_model.pth          Weights with highest val accuracy over 30 epochs
    train_record.json       Canonical train/val split (shared by all models)
    training_log.txt        Full stdout log from training run
    training_history.json   Per-epoch loss and accuracy
    training_curves.png     Train/val loss and accuracy curves
    conv1_kernels.png       Visualization: 32 Conv1 filters
    conv2_kernels.png       Visualization: 64 Conv2 filters (mean over input ch)
    conv3_conv4_kernels.png Visualization: Conv3+Conv4 filters side by side
    activation_maps.png     Visualization: mean activation per block per class
    analysis/               Analysis outputs (generated by run_analysis.py)
      roc/                  ROC / AUC curves
      gradcam/              Grad-CAM saliency maps + class representatives
      misclassification/    Confusion matrix, per-image error analysis
      activation/           Spatial activation statistics
      preprocessing/        Grayscale method comparison

  3class_hemo/              Trained by: python train.py --model-type 3class_scratch
    best_model.pth
    train_record.json
    training_log.txt

  3class_hemo_finetune/     Trained by: python train.py --model-type 3class_finetune
    best_model.pth          (fine-tuned from 5-class weights)
    train_record.json
    training_log.txt
```

---

## Scripts Overview

### `train.py` — Unified training entry point
Dispatches to `train_5class.py` or `train_3class.py` based on `--model-type`.

### `train_5class.py` — 5-class training
Trains `FibrinCNN(num_classes=5)` from scratch. Saves best checkpoint and
full training log to `models/5class/`. Writes `train_record.json` (canonical split).

### `train_3class.py --mode scratch|finetune` — 3-class hemophilia training
Two modes:
- **scratch**: `FibrinCNN(num_classes=3)` trained from random init, same
  hyperparameters as 5-class, 30 epochs. Saves to `models/3class_hemo/`.
- **finetune**: Loads 5-class weights, replaces `classifier[-1]` with
  `Linear(128,3)`. Phase 1 (5 epochs): features frozen, head trains at LR=1e-3.
  Phase 2 (25 epochs): all layers unfrozen at LR=1e-4. Saves to
  `models/3class_hemo_finetune/`.

Both modes reuse the canonical 5-class split, filtered to the 3 hemophilia classes.

**Critical detail**: `FibrinDataset3` subclass remaps labels using
`CLASS_MAP_3 = {"F08D":0, "F09D":1, "F11D":2}` instead of the 5-class CLASS_MAP
(where F11D=3, which would be out-of-range for a 3-class model).

### `run_analysis.py` — Full analysis suite
Runs all 11 analysis steps in order for a given model type. Outputs go to
`models/<type>/analysis/`. Usage: `python run_analysis.py --model-type 5class`.

### `evaluate.py` — Detailed metrics
Loads a model and val set from saved split. Prints per-class
precision/recall/F1, confusion matrix.

### `hemophilia_analysis.py` — ROC curves + annotated images + model comparison
- `--model-type {5class,3class_scratch,3class_finetune}`: select which model
- `--roc`: OVR ROC curves for F08D/F09D/F11D (and optionally other classes)
- `--annotate`: write annotated val images (true class, pred class, softmax probs)
- `--compare`: overlay ROC curves from all three model types + accuracy table
- `--classes`: expand analysis to other classes (default: F08D F09D F11D)

### `gradcam.py` — Grad-CAM saliency maps
- Default: class representative + misclassified Grad-CAM plots
- `--all-overlays`: full-resolution Grad-CAM overlay on all val images

### `visualize_weights.py` — CNN kernel + activation visualization
- Conv1: 32 filters (1×5×5), shown as 4×8 heatmap grid (RdBu_r colormap)
- Conv2: 64 filters, mean over 32 input channels → 8×8 heatmap grid
- Conv3+Conv4: side-by-side 8×16 and 16×16 grids of 3×3 filters
- Activation maps: forward hooks on MaxPool outputs; one val image per class;
  channel-mean activation shown as viridis heatmap

---

## Key API Patterns

### Loading a split and building a DataLoader
```python
from data_loader import load_split_from_record, filter_classes, FibrinDataset, make_balanced_sampler
from preprocessing import make_preprocessor
from torch.utils.data import DataLoader

train_df, val_df = load_split_from_record("models/5class/train_record.json", "data/endpoint10.db")
# Optionally filter to hemophilia classes:
val_df = filter_classes(val_df, ["F08D", "F09D", "F11D"])

preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)
val_ds = FibrinDataset(val_df, "data/photos", preprocessor, augment=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
```

### Loading a model
```python
from model import FibrinCNN
import torch

model = FibrinCNN(num_classes=5)
model.load_state_dict(torch.load("models/5class/best_model.pth", weights_only=True))
model.eval()
```

### Detecting num_classes from a saved checkpoint
```python
sd = torch.load("models/5class/best_model.pth", weights_only=True)
num_classes = sd["classifier.3.bias"].shape[0]  # 5 or 3
```

---

## Known Pitfalls and Fixes

| Issue | Where | Fix |
|-------|-------|-----|
| 5-class labels (F11D=3) out-of-range in 3-class model | `FibrinDataset.__getitem__` | Use `FibrinDataset3` subclass with `CLASS_MAP_3` |
| BN running stats shift during frozen fine-tune Phase 1 | `train_3class.py` finetune | Call `model.features.eval()` each batch in Phase 1 |
| BatchNorm uses batch stats instead of running stats | Any fresh `FibrinCNN()` instance | Always call `model.eval()` before inference; `evaluate.py:predict_all()` does this defensively |
| `ReduceLROnPlateau(verbose=True)` error | PyTorch 2.4+ dropped this arg | Removed `verbose` argument |
| `np.trapz` AttributeError | NumPy 2.0 renamed it | Use `np.trapezoid` instead |
| Val set consistency across models | All eval scripts | Always use `load_split_from_record`, never `create_dataloaders` for eval |
| ROC column index mismatch for 3-class | `hemophilia_analysis.py` | Parameterize `class_map` in `collect_predictions` and `run_roc_analysis` |

---

## Environment

- Machine: System76 Darter Pro, 96 GB RAM, Intel Core Ultra 7 255H (16 threads)
- **No dedicated GPU** — all training runs on CPU
- Conda environment: `fibrin` (see `environment.yml`)
- Key packages: PyTorch, OpenCV, SQLAlchemy, pandas, numpy, matplotlib, pytest

Activate with: `conda activate fibrin`

---

## Suggested Next Steps for Exploration

The following are directions that have been discussed or are natural extensions,
but have not yet been implemented:

1. **Run and compare all three models** — `train_3class.py` has been written but
   the 3-class models have not yet been trained on the new 1000-image dataset.
   Running `--mode scratch` and `--mode finetune` and then
   `hemophilia_analysis.py --compare` would give the first empirical answer to
   "does specializing the model to 3 hemophilia classes improve discrimination?"

2. **Image-level exploration** — Examining which val images are misclassified,
   whether misclassifications cluster by experiment, slide type, or spatial region.
   The annotated val images in `models/5class/analysis/annotated_test/` are a
   starting point.

3. **Activation map analysis** — The `activation_maps.png` from `visualize_weights.py`
   shows where the network responds. Investigating whether it attends to fiber
   junctions, fiber density, or background regions could inform whether the network
   is learning biologically meaningful features.

4. **Hyperparameter tuning** — The current 30-epoch / LR=1e-3 setup was chosen
   without systematic search. Learning rate, weight decay, and number of epochs
   may benefit from tuning once baseline performance is established.

5. **Alternative preprocessing** — Five grayscale methods are implemented but only
   `lab_l` has been used. Comparing performance with `luminance`, `hsv_s`, or
   `green` channels could be informative, especially for classes that differ in
   fiber color/saturation.

6. **Grad-CAM or similar** — Class activation maps beyond simple channel-mean
   activations would give more interpretable saliency maps indicating which image
   regions drive each class decision.
