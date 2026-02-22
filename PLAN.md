# Plan: Fibrin Clot CNN Classification

## Context
No Python code exists yet. The goal is to build a complete CNN pipeline to classify 859 microscopy images of fibrin clots across 5 phenotype classes (AC3, F08D, F09D, F11D, NC1). Images are 6000×4000 px JPEG files; labels come from an SQLite database. The machine has no dedicated GPU (integrated Intel graphics only), so the architecture must be computationally practical on CPU.

---

## Dataset Summary
- **859 images**, 6000×4000 px, stored as `data/photos/0000.JPG` – `0858.JPG`
- **Labels**: `data/test_db.db` → table `Images_Endpoint_10` (Experiment, Exp_Type, Slide_Type, Img_Name, Img_Fp)
- **Row-to-filename mapping**: `rowid - 1` = filename integer (e.g. row 1 → `0000.JPG`)
- **Class distribution**: AC3=219, F08D=120, F09D=180, F11D=180, NC1=160
- **41 distinct experiments**; images from the same experiment are correlated

---

## File Structure
```
data_loader.py        # DB access, Dataset class, train/test split
preprocessing.py      # Grayscale conversion and min-pooling
augmentation.py       # 4-fold flip augmentations
model.py              # FibrinCNN architecture
train.py              # Training loop and orchestration
evaluate.py           # Detailed evaluation and metrics
test_data_loader.py   # Tests: DB access, splitting, Dataset
test_preprocessing.py # Visual tests: grayscale methods, min-pool
test_augmentation.py  # Visual tests: flip augmentations
README.md             # Human-readable documentation
```

---

## CNN Architecture

### Step 1 — Initial Downsampling (before convolutions)
Apply a **10×10 min-pool** (stride 10) to every grayscale image:
- 6000×4000 → **600×400**
- Min-pool (not max-pool) selects the darkest pixel in each 10×10 block, preserving the dark fiber structure on a light background
- Both dimensions are exactly divisible by 10 — no cropping needed

### Receptive Field Analysis
**Requirement**: detect features at ≥ 1/20 × 6000 = **300 px** in original image space.
After 10× downsampling, this maps to ≥ **30 px** in the 600×400 CNN input.

| Layer | Kernel | Cum. Stride | RF (600×400 space) | RF (original) |
|-------|--------|-------------|---------------------|---------------|
| Conv1 (k=5) | 5 | 1 | 5 | 50 |
| Pool1 (2×2) | 2 | 2 | 6 | 60 |
| Conv2 (k=5) | 5 | 2 | 14 | 140 |
| Pool2 (2×2) | 2 | 4 | 16 | 160 |
| Conv3 (k=3) | 3 | 4 | 24 | 240 |
| Pool3 (2×2) | 2 | 8 | 28 | 280 |
| Conv4 (k=3) | 3 | 8 | **44** | **440** |
| Pool4 (2×2) | 2 | 16 | 52 | **520 ✓** |

RF after Block 4 = **520 px** in original space — exceeds 300 px requirement.

### Layer-by-Layer Dimensions
```
Input:  1 × 600 × 400  (grayscale, after min-pool)

Block 1: Conv2d(1→32,   5×5, pad=2) → BN → ReLU → MaxPool(2×2)  →  32 × 300 × 200
Block 2: Conv2d(32→64,  5×5, pad=2) → BN → ReLU → MaxPool(2×2)  →  64 × 150 × 100
Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  75 ×  50
Block 4: Conv2d(128→256,3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 256 ×  37 ×  25

AdaptiveAvgPool2d(1,1) → 256
Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→5)
```
All kernels are **square**. All convolutions use `padding = (k−1)//2` to preserve spatial dimensions (no aspect ratio change). ~1.5M parameters total.

---

## Key Design Decisions

### Grayscale Conversion
Default: **L channel from CIE LAB** (`cv2.COLOR_BGR2LAB`, extract channel 0). Perceptually uniform, good contrast for fiber structure. Five methods are compared visually in `test_preprocessing.py`:
1. `"lab_l"` — L channel from LAB (default)
2. `"luminance"` — BT.601 weighted: `0.299R + 0.587G + 0.114B`
3. `"hsv_v"` — Value channel from HSV (max of R,G,B)
4. `"hsv_s"` — Saturation channel from HSV
5. `"green"` — Green channel only

### Train/Test Split
**Experiment-level split** (per class) to prevent data leakage — all images from the same experiment stay in the same partition.
- Target ~75% train per class; F08D uses 5/6 experiments (83%) due to its smaller size
- Deterministic: fixed `seed=42`
- Training image record saved to `train_record.json`

### Class Imbalance
Three complementary strategies:
1. `WeightedRandomSampler` in training DataLoader (inverse-frequency per-sample weights)
2. `CrossEntropyLoss(weight=class_weights)` with inverse-frequency class weights
3. Uniform 4× augmentation across all classes

### Data Augmentation (training only)
Each training image produces 4 variants (4-fold symmetry of rectangular images):
- Variant 0: original
- Variant 1: horizontal flip
- Variant 2: vertical flip
- Variant 3: both flips (= 180° rotation)

---

## Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 16 | ~3.8 MB input per batch; memory-safe on CPU |
| `lr` | 1e-3 | Standard Adam starting point |
| `weight_decay` | 1e-4 | Light L2 regularization |
| `num_epochs` | 30 | ~4830 gradient updates; sufficient for convergence |
| `train_ratio` | 0.75 | 75% train, 25% test |
| `seed` | 42 | Reproducibility |
| `gray_method` | "lab_l" | Best perceptual contrast |
| `pool_factor` | 10 | 6000×4000 → 600×400 |
| `num_workers` | 4 | Parallel image loading on 16-thread CPU |

Scheduler: `ReduceLROnPlateau(patience=5)` — auto-reduces LR on validation plateau.
Best model saved to `best_model.pth` (by test accuracy).

---

## Implementation Sequence
1. `preprocessing.py` → `test_preprocessing.py` — validate grayscale + pooling visually
2. `augmentation.py` → `test_augmentation.py` — validate flips visually
3. `data_loader.py` → `test_data_loader.py` — validate DB access and splitting
4. `model.py`
5. `evaluate.py`
6. `train.py`
7. `README.md`

---

## Verification Checklist
- [ ] `pytest -v` passes; inspect PNGs in `test_output/` to confirm preprocessing and augmentation
- [ ] `python train.py` completes; `train_record.json` and `best_model.pth` created
- [ ] `python evaluate.py` prints per-class precision/recall/F1
- [ ] No experiment appears in both train and test (asserted in `test_data_loader.py`)
- [ ] Test accuracy >20% (better than 5-class random chance baseline)
