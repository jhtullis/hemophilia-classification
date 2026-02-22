# Fibrin Clot CNN Classification

Classify microscopy images of fibrin clots across five blood phenotypes using a convolutional neural network in PyTorch.

---

## Classes
| Label | Phenotype | Image count |
|-------|-----------|-------------|
| AC3   | Normal plasma (control) | 219 |
| F08D  | Factor VIII deficient | 120 |
| F09D  | Factor IX deficient | 180 |
| F11D  | Factor XI deficient | 180 |
| NC1   | Normal control | 160 |

---

## Environment Setup
```bash
conda env create -f environment.yml
conda activate fibrin
```

---

## Data
- **Images**: `data/photos/0000.JPG` – `0858.JPG` (6000 × 4000 px JPEG, ~5.9 GB)
- **Labels**: `data/test_db.db` — SQLite, table `Images_Endpoint_10`
  - Filename integer = `rowid - 1` in the database
  - Relevant columns: `Experiment`, `Exp_Type` (class), `Slide_Type` (A/B)

---

## Preprocessing Pipeline
Every image goes through:
1. **Grayscale conversion** — L channel from CIE LAB color space (default). Perceptually uniform; good contrast for dark fiber structure. Alternatives (`luminance`, `hsv_v`, `hsv_s`, `green`) are compared visually in `test_preprocessing.py`.
2. **10× min-pool** — Reduces 6000×4000 → 600×400. Min-pool selects the darkest pixel in each 10×10 block, preserving the thin dark fibers on a light background.
3. **Normalization** — Scale uint8 values to float32 in [0, 1].

---

## CNN Architecture
```
Input:  1 × 600 × 400  (grayscale, after 10× min-pool)

Block 1: Conv2d(1→32,   5×5, pad=2) → BN → ReLU → MaxPool(2×2)  →  32 × 300 × 200
Block 2: Conv2d(32→64,  5×5, pad=2) → BN → ReLU → MaxPool(2×2)  →  64 × 150 × 100
Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  75 ×  50
Block 4: Conv2d(128→256,3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 256 ×  37 ×  25

AdaptiveAvgPool2d(1,1) → 256
Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→5)
```

All kernels are **square**. Padding preserves spatial dimensions through each convolution (no aspect ratio change).

### Receptive Field
The spec requires features at ≥ 1/20 × 6000 = **300 px** in original image space to be detectable.

| Layer | Kernel | Cum. Stride | RF (600×400) | RF (original) |
|-------|--------|-------------|--------------|---------------|
| Conv1 (k=5) | 5 | 1 | 5 | 50 |
| Pool1 | 2 | 2 | 6 | 60 |
| Conv2 (k=5) | 5 | 2 | 14 | 140 |
| Pool2 | 2 | 4 | 16 | 160 |
| Conv3 (k=3) | 3 | 4 | 24 | 240 |
| Pool3 | 2 | 8 | 28 | 280 |
| Conv4 (k=3) | 3 | 8 | **44** | **440** |
| Pool4 | 2 | 16 | **52** | **520 ✓** |

---

## Data Augmentation
Training only. Each image generates 4 variants exploiting 4-fold symmetry:
- Variant 0: original
- Variant 1: horizontal flip
- Variant 2: vertical flip
- Variant 3: both flips (180° rotation)

---

## Train / Test Split
- **Experiment-level**: all images from the same experiment stay in the same partition to prevent data leakage.
- Target ~75% train per class (F08D uses 5/6 experiments at ~83%).
- `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)` correct for class imbalance.
- Split record saved to `train_record.json`.

---

## Training

```bash
python train.py
```

Hyperparameters (top of `train.py`):

| Parameter | Value |
|-----------|-------|
| `batch_size` | 16 |
| `lr` | 1e-3 (Adam) |
| `weight_decay` | 1e-4 |
| `num_epochs` | 30 |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

Outputs: `best_model.pth`, `train_record.json`

---

## Evaluation

```bash
python evaluate.py
```

Prints per-class precision, recall, F1, support, and the confusion matrix.

---

## Tests

```bash
pytest -v -s
```

| Test file | What it checks |
|-----------|---------------|
| `test_preprocessing.py` | Grayscale methods comparison, min-pool vs max-pool (visual PNGs saved to `test_output/`) |
| `test_augmentation.py` | All 4 flip variants (visual PNG), self-inverse property, shape preservation |
| `test_data_loader.py` | 859 rows, correct class counts, no experiment overlap between splits, tensor shapes |

---

## File Structure
```
preprocessing.py      Grayscale conversion and min-pooling
augmentation.py       4-fold flip augmentations
data_loader.py        DB access, Dataset, balanced DataLoader
model.py              FibrinCNN architecture
train.py              Training loop
evaluate.py           Metrics: per-class precision/recall/F1, confusion matrix
test_preprocessing.py Visual tests for preprocessing
test_augmentation.py  Visual tests for augmentation
test_data_loader.py   Unit tests for data pipeline
PLAN.md               Architecture decisions and implementation plan
environment.yml       Conda environment specification
data/photos/          6000×4000 JPEG images
data/test_db.db       SQLite label database
```

---

## Potential Improvements
- **Transfer learning**: pre-trained ImageNet weights (grayscale → 3-channel duplication) could significantly improve accuracy on this small dataset.
- **Additional augmentation**: random brightness/contrast jitter or small rotations (if rotational symmetry holds beyond 0°/90°/180°/270°).
- **Larger dataset**: the main bottleneck; more experimental replicates would be the highest-leverage improvement.
- **Slide type as auxiliary signal**: `Slide_Type` (A/B) is a known covariate that could be provided as metadata or used in a stratified analysis.
