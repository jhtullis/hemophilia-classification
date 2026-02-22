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
preprocessing.py      Grayscale + 10× min-pool
augmentation.py       4-fold flip augmentations
data_loader.py        DB access, Dataset, DataLoaders
model.py              FibrinCNN architecture (~455K params)
train.py              Training loop → best_model.pth, train_record.json
evaluate.py           Per-class precision/recall/F1, confusion matrix
test_preprocessing.py Visual + structural tests
test_augmentation.py  Visual + property tests
test_data_loader.py   DB, split, Dataset unit tests
PLAN.md               Architecture decisions and rationale
environment.yml       Conda environment
data/photos/          Raw images
data/test_db.db       SQLite labels
best_model.pth        Saved model weights (after training)
train_record.json     Split record (which images → train/test)
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
1. Grayscale via CIE LAB L-channel (`lab_l`, default; also `luminance`, `hsv_v`, `hsv_s`, `green`)
2. 10× **min-pool**: `img.reshape(H//10, 10, W//10, 10).min(axis=(1,3))` → 600×400
3. Normalize to float32 [0, 1], add channel dim → tensor shape `(1, 400, 600)`

## Augmentation
4 variants (applied in Dataset via `idx % 4`):
- 0 = identity, 1 = h-flip, 2 = v-flip, 3 = both flips

## Train/Test Split
- **Experiment-level**: all images from one experiment stay together (prevents leakage)
- Target ~75% train per class; F08D uses 5/6 experiments (~83%)
- `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)` for class imbalance

## Hyperparameters (train.py)
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
pytest -v -s           # run all tests
python train.py        # train → best_model.pth
python evaluate.py     # detailed metrics on test set
```

## Known Issues / Fixes Applied
- `ReduceLROnPlateau(verbose=True)` removed — argument dropped in PyTorch 2.4+
- No sklearn dependency; metrics computed from scratch in `evaluate.py`
- Machine has no GPU; training runs on CPU only
