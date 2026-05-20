# Implementation Plan: Patch-Based CNN with Stride-2 All-Convolutional Architecture

**For Claude Code — fibrin clot classification project**

This plan adds a new self-contained patch-based training pipeline alongside the
existing full-image pipeline. No existing files are modified except `environment.yml`.

---

## 0. Context and Design Decisions

### Pipeline summary
```
Raw JPEG (6000×4000)
  → 10× min-pool (existing preprocessing.py, reused unchanged)
  → 600×400 grayscale float32 tensor, cached in RAM if preload=True
  → Random center (cy, cx) sampled uniformly from [0, 400) × [0, 600)
      — no safety zone; all positions valid; black fill handles out-of-bounds
  → Extract oversized patch: ceil(200 × √2) = 283 px square, centered at (cy, cx)
      — pre-pad full image with 142 px black border so all centers are always valid
      — extraction is always in-bounds on the padded image; no conditional logic needed
  → Return 283×283 tensor to DataLoader
GPU (inside training_step, via Kornia):
  → RandomRotation(degrees=(-180, 180), padding_mode='zeros')   [full arbitrary rotation]
  → CenterCrop(200)                                              [discard rotated corners]
  → RandomHorizontalFlip(p=0.5)
  → RandomVerticalFlip(p=0.5)
  → Forward through FibrinPatchCNN
```

### Patch size rationale
- 200 pooled px = 2000 px in original image space
- Oversized extraction (283 px) allows rotation at any angle without showing
  non-image content inside the final 200 px crop
- Black pixels that appear in border patches are physically honest
  (no reflection or toroidal wrapping artefacts)
- 200 px divides both image dimensions (GCD(400, 600) = 200), enabling a
  clean inference grid; see section 8

### Architecture rationale
- Stride-2 3×3 convolutions replace all max-pooling
  (learned downsampling, same spatial reduction, slightly larger RF)
- 3 blocks of (Conv3×3 + Conv3×3 + Conv3×3-stride-2) + trailing Conv3×3
- RF = 59 px in the 200-px patch input space
- In original image space: 59 × 10 = 590 px (comparable to existing 520 px)
- Final classifier uses NormalizedLinear (cosine classifier); see section 4
- ~900K parameters; no skip connections needed at this depth

### RF calculation (stride-2 version)
| Layer           | k | stride | cum. stride | RF |
|-----------------|---|--------|-------------|----|
| Conv1a          | 3 |   1    |      1      |  3 |
| Conv1b          | 3 |   1    |      1      |  5 |
| Conv1c (s=2)    | 3 |   2    |      2      |  7 |
| Conv2a          | 3 |   1    |      2      | 11 |
| Conv2b          | 3 |   1    |      2      | 15 |
| Conv2c (s=2)    | 3 |   2    |      4      | 19 |
| Conv3a          | 3 |   1    |      4      | 27 |
| Conv3b          | 3 |   1    |      4      | 35 |
| Conv3c (s=2)    | 3 |   2    |      8      | 43 |
| Conv4           | 3 |   1    |      8      | 59 |
| GlobalAvgPool   | — |   —    |      —      |  — |

### Loss function rationale
Training uses pure `CosineLoss` (no cross-entropy, no temperature).
The final classifier layer is `NormalizedLinear`, which produces cosine similarities
in [−1, 1] rather than raw logits. This keeps the embedding geometry well-structured
and avoids Softmax Collapse. No temperature parameter lives in the model.
All evaluation metrics (accuracy, F1, AUC-ROC, confusion matrix, silhouette score)
are temperature-invariant; no probability conversion is performed anywhere in this
pipeline.

### Kornia note
Kornia augmentation runs as an `nn.Module` inside `training_step`, after the batch
is already on GPU. No CPU round-trip. Requires adding `kornia` to the conda environment.

---

## 1. Environment Update

**File to modify: `environment.yml`**

Add `kornia` to the dependencies list (conda-forge has it):

```yaml
# Add this line under dependencies:
  - kornia
```

After editing, recreate or update the environment on the HPC login node:
```bash
mamba env update -f environment.yml --prune
```

Also update `Proposed_environment_for_HPC_Deep_learning` (project file) to include `kornia`.

---

## 2. New Files to Create

```
patch_dataset.py          Patch-sampling Dataset (CPU side)
model_patch.py            FibrinPatchCNN architecture with cosine classifier
augmentation_patch.py     Kornia GPU augmentation module
train_patch.py            Training loop with checkpointing + Slurm awareness
evaluate_patch.py         Inference via cosine soft-vote over image grid
slurm_patch.sh            HPC job script with chain-restart logic
test_patch_pipeline.py    Visual + structural tests for new pipeline
```

Output directory: `models/patch_5class/`

---

## 3. `patch_dataset.py`

### Responsibilities
- Reuse `make_preprocessor` from `preprocessing.py` (10× min-pool, unchanged)
- Optionally preload all 600×400 tensors into RAM (recommended on HPC)
- Pre-pad each preloaded tensor with 142 px black border
- Sample `patches_per_image` random patch centers per image per `__getitem__` call
- Return 283×283 oversized patches (rotation happens on GPU, not here)
- Reuse experiment-level split from `data_loader.py`

### Key constants
```python
PATCH_SIZE = 200          # final patch size after rotation + crop
OVERSIZED  = 283          # ceil(200 * sqrt(2)); guarantees clean crop at any angle
PAD        = 142          # ceil(283 / 2); pre-pad so all image centers are valid
PATCHES_PER_IMAGE = 20    # patches drawn per image per epoch (tunable)
```

### Class signature
```python
class FibrinPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        photo_dir: str,
        preprocessor: Callable,
        patches_per_image: int = PATCHES_PER_IMAGE,
        augment: bool = False,    # unused at dataset level; kept for API consistency
        preload: bool = True,
    ):
```

### `__len__`
```python
return len(self.df) * self.patches_per_image
```

### `__getitem__` logic
```python
base_idx = idx // self.patches_per_image
tensor   = self._get_base_tensor(base_idx)   # (1, 400+2*PAD, 600+2*PAD) padded
label    = CLASS_MAP[self.df.iloc[base_idx]["Exp_Type"]]

_, H_pad, W_pad = tensor.shape
H_orig = H_pad - 2 * PAD   # 400
W_orig = W_pad - 2 * PAD   # 600

# Sample center uniformly over original image coordinates
cy = torch.randint(0, H_orig, (1,)).item()
cx = torch.randint(0, W_orig, (1,)).item()

# Map to padded-image coordinates and extract oversized patch
cy_pad = cy + PAD
cx_pad = cx + PAD
half   = OVERSIZED // 2
patch  = tensor[:, cy_pad - half : cy_pad + half + 1,
                   cx_pad - half : cx_pad + half + 1]
# patch shape: (1, 283, 283) — always valid; black fill already in padding

return patch, label
```

### Pre-padding (done once at preload time)
```python
import torch.nn.functional as F

def _pad_tensor(t: torch.Tensor, pad: int) -> torch.Tensor:
    """Add black (zero) border of `pad` pixels on all four sides."""
    return F.pad(t, (pad, pad, pad, pad), mode='constant', value=0.0)
```

### Three-way experiment split

This pipeline uses an independent split from the existing full-image models:
**7 train / 1 val / 2 test** per class (10 experiments per class, 50 total).

```python
def split_by_experiment_3way(
    df: pd.DataFrame,
    n_train: int = 7,
    n_val: int = 1,
    n_test: int = 2,
    seed: int = 99,          # different seed from existing 5class split (seed=42)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-way experiment-level split: train / val / test.
    All images from one experiment stay in the same partition.
    Split is stratified: n_train/n_val/n_test experiments per class.

    With 10 experiments per class (5 classes):
      train: 7 × 5 = 35 experiments  (~700 images)
      val:   1 × 5 =  5 experiments  (~100 images)
      test:  2 × 5 = 10 experiments  (~200 images)

    The test set is held out completely. It must not be used during
    training, model selection, or hyperparameter tuning. It is only
    accessed in evaluate_patch.py for final reported results.
    """
    rng = np.random.default_rng(seed)
    train_rows, val_rows, test_rows = [], [], []

    for cls in CLASS_MAP:
        cls_df = df[df["Exp_Type"] == cls]
        exps   = cls_df["Experiment"].unique().tolist()
        assert len(exps) == n_train + n_val + n_test, \
            f"Expected {n_train + n_val + n_test} experiments for {cls}, got {len(exps)}"
        rng.shuffle(exps)
        train_exps = set(exps[:n_train])
        val_exps   = set(exps[n_train:n_train + n_val])
        test_exps  = set(exps[n_train + n_val:])

        train_rows.append(cls_df[cls_df["Experiment"].isin(train_exps)])
        val_rows.append(  cls_df[cls_df["Experiment"].isin(val_exps)])
        test_rows.append( cls_df[cls_df["Experiment"].isin(test_exps)])

    return (pd.concat(train_rows).reset_index(drop=True),
            pd.concat(val_rows).reset_index(drop=True),
            pd.concat(test_rows).reset_index(drop=True))
```

Save the split to `models/patch_5class/train_record_patch.json` at first run,
then reload from it on subsequent runs (same pattern as `load_split_from_record`
in `data_loader.py`, but implemented locally in `patch_dataset.py`).

The record format should include `train_experiments`, `val_experiments`,
`test_experiments`, `train_indices`, `val_indices`, `test_indices`,
`seed`, and `class_distribution` for all three splits.

### Notes for Claude Code
- `CLASS_MAP` is identical to the existing one in `data_loader.py`; import it rather
  than redefining
- `make_balanced_sampler` from `data_loader.py` is compatible; pass the new
  `FibrinPatchDataset` to it directly since the label-per-index logic is preserved
- Do **not** reuse the existing `train_record.json` from `models/5class/` — this is
  an independent split with a different seed and a held-out test partition
- The test DataFrame is passed only to `evaluate_patch.py`; it must never be passed
  to the training DataLoader or used to influence model selection
- The validation dataset uses the same random patch sampling as training; the fixed
  inference grid is implemented in `evaluate_patch.py`, not in this Dataset class

---

## 4. `model_patch.py`

### Architecture: `FibrinPatchCNN`

```
Input:  (N, 1, 200, 200)

Block 1:
  Conv2d(1→32,   k=3, pad=1)  → BN → ReLU
  Conv2d(32→32,  k=3, pad=1)  → BN → ReLU
  Conv2d(32→32,  k=3, pad=1, stride=2) → BN → ReLU    → (N, 32, 100, 100)

Block 2:
  Conv2d(32→64,  k=3, pad=1)  → BN → ReLU
  Conv2d(64→64,  k=3, pad=1)  → BN → ReLU
  Conv2d(64→64,  k=3, pad=1, stride=2) → BN → ReLU    → (N, 64, 50, 50)

Block 3:
  Conv2d(64→128, k=3, pad=1)  → BN → ReLU
  Conv2d(128→128,k=3, pad=1)  → BN → ReLU
  Conv2d(128→128,k=3, pad=1, stride=2) → BN → ReLU    → (N, 128, 25, 25)

Conv4:
  Conv2d(128→256,k=3, pad=1)  → BN → ReLU              → (N, 256, 25, 25)

AdaptiveAvgPool2d(1, 1)        → (N, 256, 1, 1)
Flatten                        → (N, 256)
Linear(256→128) → ReLU → Dropout(p=0.5)
NormalizedLinear(128→num_classes)   → (N, num_classes)  cosine similarities in [−1, 1]
```

### `NormalizedLinear`

The final layer normalizes both the input feature vector and the weight matrix
(class prototypes) before computing their dot product, yielding cosine similarities
directly. No temperature parameter — raw similarities are returned.

```python
class NormalizedLinear(nn.Module):
    """
    Linear layer with L2-normalized weights and inputs.
    Output is cosine similarities in [−1, 1], one per class.
    Used as the final classifier in FibrinPatchCNN.
    No temperature, no softmax — raw cosine similarities only.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x,            dim=1)   # normalize feature vectors
        w_norm = F.normalize(self.weight,  dim=1)   # normalize class prototypes
        return x_norm @ w_norm.T                    # (N, num_classes), in [−1, 1]
```

### Implementation notes
- All convolutions use `bias=False` when followed by BatchNorm (standard practice)
- Implement as `nn.Sequential` blocks stored in `self.features` and `self.classifier`
  to keep the same attribute naming convention as `FibrinCNN` in `model.py`
- The stride-2 convolution in each block is the *third* conv (the downsampling step),
  not the first — spatial reduction happens after two rounds of feature extraction
  at full resolution within the block
- `num_classes` parameter defaults to 5; accepts 3 for hemophilia-only variant
- `NormalizedLinear` replaces the final `Linear` in the classifier — everything
  else in the head (256→128 linear, ReLU, Dropout) is unchanged

### Helper: `make_patch_model`
```python
def make_patch_model(num_classes: int = 5) -> FibrinPatchCNN:
    model = FibrinPatchCNN(num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FibrinPatchCNN: {n_params:,} parameters, {num_classes} classes")
    return model
```

### Approximate parameter count
~900K (5-class). Parameter count is independent of spatial input size because
`AdaptiveAvgPool2d` collapses spatial dims before the classifier.

---

## 5. `augmentation_patch.py`

### Purpose
GPU-side augmentation module. Runs inside `training_step` after the batch is already
on the GPU. Input is the 283×283 oversized patch batch; output is the 200×200
augmented patch ready for the CNN.

### Class: `PatchAugmentation(nn.Module)`

```python
import kornia.augmentation as K
import torch.nn as nn

class PatchAugmentation(nn.Module):
    """
    GPU-side augmentation for oversized (283×283) patch batches.

    Pipeline:
      1. RandomRotation: full 360° arbitrary rotation with black fill
      2. CenterCrop:     crop to final 200×200
      3. RandomHorizontalFlip
      4. RandomVerticalFlip

    Apply in training_step AFTER batch is on GPU:
        patches = self.augmentation(patches)
    Do NOT apply during validation.
    """

    def __init__(
        self,
        patch_size: int = 200,
        p_flip: float = 0.5,
    ) -> None:
        super().__init__()
        self.augment = K.AugmentationSequential(
            K.RandomRotation(
                degrees=180.0,          # samples uniformly from [-180, 180]
                padding_mode='zeros',   # black fill during rotation
                p=1.0,
            ),
            K.CenterCrop(patch_size),   # discard rotated corners → (N,1,200,200)
            K.RandomHorizontalFlip(p=p_flip),
            K.RandomVerticalFlip(p=p_flip),
            data_keys=['input'],
            same_on_batch=False,        # independent augmentation per sample
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 283, 283) on GPU → returns (N, 1, 200, 200) on GPU."""
        return self.augment(x)
```

### Notes
- `@torch.no_grad()` prevents augmentation ops from accumulating gradients,
  saving memory with no accuracy cost
- `degrees=180.0` in Kornia samples uniformly from [-180°, +180°], equivalent
  to full 360° rotational coverage
- `padding_mode='zeros'` produces the black background agreed upon
- The existing flip augmentations in `augmentation.py` are superseded by this
  module for the patch pipeline; do not import the old `augment()` function here

---

## 6. `train_patch.py`

### Responsibilities
- Training loop for `FibrinPatchCNN` using `FibrinPatchDataset`
- GPU-side augmentation via `PatchAugmentation` in each training step
- `CosineLoss` with inverse-frequency class weights (no cross-entropy)
- Checkpointing every 20 epochs (epochs 1–300), then every 100 epochs (epochs 300+)
- Slurm-aware: reads `--start-epoch` argument to resume from checkpoint
- Saves full training history to CSV (appended every epoch, not overwritten on resume)
- Saves best validation model separately from epoch checkpoints

### `CosineLoss`

Define as a module-level function (not `nn.Module`):

```python
def cosine_loss(
    similarities: torch.Tensor,   # (B, C) raw cosine similarities from forward()
    labels: torch.Tensor,         # (B,) class indices
    class_weights: torch.Tensor,  # (C,) inverse-frequency weights, same device
) -> torch.Tensor:
    """
    Pure cosine loss: L = mean over batch of [ w_y * (1 − s_y) ]
    where s_y is the cosine similarity to the correct class prototype,
    and w_y is the inverse-frequency class weight for that class.

    Loss range: [0, 2].  0 = perfect alignment, 2 = perfectly opposite.
    No softmax. No temperature. Gradient flows only from correct-class
    similarity; weight normalization creates implicit competition between
    class prototypes.
    """
    correct_sims = similarities[torch.arange(len(labels)), labels]   # (B,)
    weights      = class_weights[labels]                              # (B,)
    return (weights * (1.0 - correct_sims)).mean()
```

Class weights are computed from training label frequencies using the same
inverse-frequency logic already in `data_loader.py`.

### Validation accuracy with cosine output

Since the model outputs cosine similarities, accuracy is:
```python
pred = similarities.argmax(dim=1)   # highest similarity = predicted class
acc  = (pred == labels).float().mean().item()
```
No softmax, no temperature needed.

### CLI arguments
```
--model-dir       str    default="models/patch_5class"
--db              str    default="data/endpoint10.db"
--photo-dir       str    default="data/photos"
--num-epochs      int    default=10000
--start-epoch     int    default=0      (set by chain script on resume)
--batch-size      int    default=64
--lr              float  default=1e-3
--weight-decay    float  default=1e-3
--patches-per-img int    default=20
--num-classes     int    default=5
--preload         flag   store_true     (recommended on HPC; loads ~700MB into RAM)
--num-workers     int    default=4
```

### Optimizer and scheduler

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=2, eta_min=1e-6
)
```

**Why AdamW:** Decoupled weight decay gives consistent weight norm control across
the full 10,000-epoch run. Loshchilov & Hutter showed better test error over
long runs compared to Adam with L2 regularization.

**Why weight_decay=1e-3:** The grokking literature (Nanda et al. 2023, Power et al.
2022) shows weight decay is the primary driver of the transition from memorizing to
generalizing solutions — zero or very small weight decay prevents delayed
generalization from occurring at all. `1e-3` is the value established in the
other training plan for this project and is more typical of the grokking literature
than the original `1e-4`.

**Why CosineAnnealingWarmRestarts:** `ReduceLROnPlateau` with any patience setting
will drive the LR to near-zero within the first few hundred epochs given the noisy
validation curves this model produces. By epoch 200–300, the LR would be
`1e-3 × 0.5^n ≈ 1e-6`, ending all meaningful weight updates for the remaining
9,700+ epochs. `CosineAnnealingWarmRestarts` instead cycles back to `base_lr`
at each restart, enabling continued exploration throughout the full run.
Restart epochs with `T_0=100, T_mult=2`: 100, 300, 700, 1500, 3100, 6300, …

**Scheduler step:** `scheduler.step(epoch)` is called once per epoch with the
**global epoch number** — not with val_loss, and not with a within-job counter.
Passing the global epoch ensures the cosine schedule is continuous and correct
across Slurm job boundaries without needing special handling.

### Training loop structure

```python
# ── Optimizer and scheduler construction ──
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=2, eta_min=1e-6
)

# ── Checkpoint loading ──
checkpoint_path = find_latest_checkpoint(args.model_dir, args.start_epoch)
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    best_epoch   = checkpoint['best_epoch']
    print(f"Resumed from {checkpoint_path} (epoch {checkpoint['epoch']})")
else:
    start_epoch  = args.start_epoch
    best_val_acc = 0.0
    best_epoch   = -1

# ── Augmentation module ──
augmentation = PatchAugmentation(patch_size=PATCH_SIZE).to(device)

# ── Per-epoch loop ──
for epoch in range(start_epoch, args.num_epochs):

    # Training pass
    model.train()
    augmentation.train()
    for patches, labels in train_loader:
        patches = patches.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        patches = augmentation(patches)        # 283→200, rotation+flips on GPU

        optimizer.zero_grad()
        sims = model(patches)                  # (B, num_classes), cosine similarities
        loss = cosine_loss(sims, labels, class_weights)
        loss.backward()
        optimizer.step()

    # Scheduler step — global epoch, called after optimizer.step()
    # CosineAnnealingWarmRestarts uses epoch to determine position in cosine cycle;
    # passing the global epoch ensures continuity across Slurm job restarts
    scheduler.step(epoch)

    # Validation pass (no augmentation — center crop only)
    model.eval()
    with torch.no_grad():
        for patches, labels in val_loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            patches = K.CenterCrop(PATCH_SIZE)(patches)   # 283→200, no rotation
            sims    = model(patches)
            # accumulate cosine_loss and argmax accuracy

    # Append one row to CSV every epoch (never rewrite; safe across crashes)
    append_csv_row(args.model_dir, {
        'epoch':      epoch,
        'train_loss': train_loss, 'train_acc': train_acc,
        'val_loss':   val_loss,   'val_acc':   val_acc,
        'lr':         optimizer.param_groups[0]['lr'],
    })

    # Save best model (by validation accuracy)
    if val_acc > best_val_acc:
        torch.save(model.state_dict(),
                   os.path.join(args.model_dir, 'best_model.pth'))
        best_val_acc = val_acc
        best_epoch   = epoch

    # Checkpoint
    if should_checkpoint(epoch):
        save_checkpoint(model, optimizer, scheduler, epoch,
                        best_val_acc, best_epoch, args, args.model_dir)
```

### `should_checkpoint(epoch)` logic
```python
def should_checkpoint(epoch: int) -> bool:
    if epoch == 0:
        return False
    if epoch <= 300:
        return epoch % 20 == 0
    return epoch % 100 == 0
```

### Checkpoint filename convention
```
models/patch_5class/checkpoints/epoch_0020.pth
models/patch_5class/checkpoints/epoch_0040.pth
...
models/patch_5class/checkpoints/epoch_0300.pth
models/patch_5class/checkpoints/epoch_0400.pth
...
```

### Checkpoint dict structure
```python
{
    'epoch':                   int,
    'model_state_dict':        dict,
    'optimizer_state_dict':    dict,
    'scheduler_state_dict':    dict,   # includes cosine cycle position
    'best_val_acc':            float,
    'best_epoch':              int,
    'config': {
        'lr':                  1e-3,
        'weight_decay':        1e-3,
        'batch_size':          int,
        'optimizer':           'AdamW',
        'scheduler':           'CosineAnnealingWarmRestarts',
        'scheduler_T0':        100,
        'scheduler_Tmult':     2,
        'scheduler_eta_min':   1e-6,
        'loss':                'CosineLoss',
        'patch_size':          200,
        'patches_per_img':     int,
        'num_classes':         int,
    },
}
```

### Critical scheduler note for Slurm chain restarts

`CosineAnnealingWarmRestarts` carries its cycle position in `scheduler.state_dict()`.
This **must** be saved and restored at every checkpoint. On resume, restoring
`scheduler.state_dict()` recovers the exact position in the cosine cycle — no
additional logic needed because `scheduler.step(epoch)` uses the global epoch
to recompute position deterministically. Do not use a within-job epoch counter;
always pass the global epoch.

### Per-epoch CSV logging

Use `training_log_full.csv` (append-only, one row per epoch). Columns:

```
epoch, train_loss, train_acc, val_loss, val_acc, lr
```

Append-only CSV (never rewrite the file) means the log survives crashes and
Slurm preemptions without data loss. Use `pandas.DataFrame([row]).to_csv(...,
mode='a', header=not os.path.exists(path))` for safe appending.

### Output files
```
models/patch_5class/
  best_model.pth              Weights at best validation accuracy
  training_log_full.csv       Append-only per-epoch log
  train_record.json           Copy of the experiment-level split used
  checkpoints/
    epoch_NNNN.pth            Periodic checkpoints
```

---

## 7. `slurm_patch.sh`

### Single-job version (for initial testing)
```bash
#!/bin/bash --login
#SBATCH --job-name=fibrin_patch
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/patch_%j.out
#SBATCH --error=logs/patch_%j.err

module load miniforge3
conda activate fibrin

python train_patch.py \
  --model-dir models/patch_5class \
  --num-epochs 10000 \
  --batch-size 64 \
  --preload \
  --num-workers 4
```

### Chain-restart version (for long training past walltime)
The chain script submits itself recursively. Each job reads the latest checkpoint
epoch and re-submits before exiting.

```bash
#!/bin/bash --login
#SBATCH --job-name=fibrin_patch_chain
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/patch_chain_%j.out
#SBATCH --error=logs/patch_chain_%j.err

module load miniforge3
conda activate fibrin

MODEL_DIR="models/patch_5class"
TARGET_EPOCHS=10000

# Find the latest checkpoint epoch
LATEST_EPOCH=$(python - <<'EOF'
import os, re, sys
ckpt_dir = "models/patch_5class/checkpoints"
if not os.path.isdir(ckpt_dir):
    print(0); sys.exit()
epochs = [int(re.search(r'epoch_(\d+)', f).group(1))
          for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
print(max(epochs) if epochs else 0)
EOF
)

echo "Resuming from epoch ${LATEST_EPOCH}"

python train_patch.py \
  --model-dir  "${MODEL_DIR}" \
  --num-epochs "${TARGET_EPOCHS}" \
  --start-epoch "${LATEST_EPOCH}" \
  --batch-size 64 \
  --preload \
  --num-workers 4

# Resubmit if not complete
CURRENT_EPOCH=$(python - <<'EOF'
import pandas as pd, os
log = "models/patch_5class/training_log_full.csv"
if not os.path.exists(log):
    print(0)
else:
    df = pd.read_csv(log)
    print(int(df['epoch'].max()) if len(df) else 0)
EOF
)

if [ "${CURRENT_EPOCH}" -lt "$((TARGET_EPOCHS - 1))" ]; then
    echo "Training not complete (epoch ${CURRENT_EPOCH}/${TARGET_EPOCHS}). Resubmitting."
    sbatch slurm_patch.sh
else
    echo "Training complete at epoch ${CURRENT_EPOCH}."
fi
```

### Notes
- Create `logs/` directory before first submission: `mkdir -p logs`
- Adjust `--time` and `--mem` based on actual GPU node availability
- `--mem=16G` should be sufficient for preloaded dataset (~700 MB) + model + batches
- Use `squeue -u $USER` to monitor job status

---

## 8. `evaluate_patch.py`

### Inference grid design

At inference, patch centers are placed on a regular grid with 100 px stride,
running from 0 to the image boundary inclusive (center-anchored). This means
centers fall exactly on image edges, so edge pixels are always at the center of
the receptive field rather than its periphery. Out-of-bounds content is filled
with black via the pre-padded tensor — consistent with the black fill seen
during training.

```
Horizontal centers: 0, 100, 200, 300, 400, 500, 600  → 7 columns
Vertical centers:   0, 100, 200, 300, 400             → 5 rows
Total patches per image: 7 × 5 = 35
```

Coverage properties:
- Interior pixels: covered by exactly 2 patches (double coverage, variance reduction)
- Edge pixels: covered by exactly 1 patch, at RF center (best possible view of edge content)
- No black content at inference for interior patches; edge patches contain black fill
  in proportion to their distance from the image boundary — identical to training distribution

```python
def inference_grid_centers(H: int = 400, W: int = 600,
                            stride: int = 100) -> list[tuple[int, int]]:
    """
    Center-anchored grid from (0,0) to (H,W) inclusive at given stride.
    Returns list of (cy, cx) in original image coordinates.
    """
    centers = []
    for cy in range(0, H + 1, stride):
        for cx in range(0, W + 1, stride):
            centers.append((cy, cx))
    return centers   # 35 centers for H=400, W=600, stride=100
```

### Cosine soft voting

The model outputs cosine similarities in [−1, 1]. Soft voting averages these
directly across patches — no temperature, no softmax, no probability conversion.
This is mathematically valid because all similarities share the same [−1, 1] scale
by construction, making them directly comparable and averageable.

```python
def predict_image(
    model: nn.Module,
    padded_tensor: torch.Tensor,   # (1, 400+2*PAD, 600+2*PAD) on CPU
    centers: list[tuple[int, int]],
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    """
    Returns (predicted_class, mean_cosine_similarities).
    mean_cosine_similarities: (num_classes,) tensor, values in [−1, 1].
    No temperature conversion. No softmax. Argmax gives the prediction.
    """
    all_sims = []
    half = OVERSIZED // 2

    for cy, cx in centers:
        cy_pad = cy + PAD
        cx_pad = cx + PAD
        patch  = padded_tensor[:, cy_pad - half : cy_pad + half + 1,
                                  cx_pad - half : cx_pad + half + 1]
        patch  = K.CenterCrop(PATCH_SIZE)(patch.unsqueeze(0).to(device))
        with torch.no_grad():
            sims = model(patch)   # (1, num_classes), cosine similarities
        all_sims.append(sims.squeeze(0).cpu())

    mean_sims  = torch.stack(all_sims).mean(dim=0)   # (num_classes,)
    pred_class = mean_sims.argmax().item()
    return pred_class, mean_sims
```

### AUC-ROC note

AUC-ROC sweeps thresholds over the full range of the score. For cosine similarities
the natural range is [−1, 1], not [0, 1]. Ensure the AUC implementation uses the
actual score range rather than assuming [0, 1] — `sklearn.metrics.roc_auc_score`
handles this correctly since it ranks scores rather than thresholding at fixed values.

### Evaluation sets

Run `evaluate_patch.py` in two modes:

**Validation mode** (used during development and model selection):
```bash
python evaluate_patch.py --split val --model-dir models/patch_5class
```

**Test mode** (final reported results only — run once after all model selection is complete):
```bash
python evaluate_patch.py --split test --model-dir models/patch_5class
```

The test split must not be used to make any training or architectural decisions.
It exists solely to report final performance on data the model has never influenced.

### Output files
```
models/patch_5class/analysis/
  val/
    confusion_matrix.png
    roc_curves.png
    per_image_predictions.csv    (image idx, true label, pred label, per-class mean similarity)
    patch_confidence_maps/       (optional: per-patch max similarity heatmap per image)
  test/
    confusion_matrix.png
    roc_curves.png
    per_image_predictions.csv
```

---

## 9. `test_patch_pipeline.py`

### Tests to implement

1. **`test_patch_shape`**
   Assert `FibrinPatchDataset[i]` returns tensor of shape `(1, 283, 283)`.

2. **`test_patch_label_range`**
   Assert all labels in `[0, num_classes-1]`.

3. **`test_patch_no_out_of_range_pixels`**
   Assert all pixel values in `[0.0, 1.0]`.

4. **`test_patch_coverage`**
   Sample 1000 patches from a single image; assert center coordinates are drawn
   from the full `[0, H) × [0, W)` range (not restricted to interior).
   Test that patches with centers at (0,0), (0, W-1), (H-1, 0), (H-1, W-1)
   all have partial black content (pixel min = 0.0) and partial real content
   (pixel max > 0.0).

5. **`test_augmentation_output_shape`**
   Create a batch of random (N, 1, 283, 283) tensors on CPU (or GPU if available);
   assert `PatchAugmentation()(batch)` returns shape `(N, 1, 200, 200)`.

6. **`test_augmentation_rotation_invariance_visual`**
   Apply augmentation 8 times to the same patch; save a 2×4 grid PNG to
   `test_output/patch_augmentation_visual.png` for human inspection.
   Assert all outputs have different mean values (rotations differ from identity).

7. **`test_model_forward`**
   Run a random `(4, 1, 200, 200)` batch through `FibrinPatchCNN(num_classes=5)`;
   assert output shape is `(4, 5)`.

8. **`test_model_output_range`**
   Assert all output values from `FibrinPatchCNN` are in `[−1.0, 1.0]`
   (confirming `NormalizedLinear` is producing cosine similarities, not raw logits).

9. **`test_model_parameter_count`**
   Assert parameter count is between 800K and 1M.

10. **`test_cosine_loss_range`**
    Assert `cosine_loss` output is in `[0.0, 2.0]` for random inputs.
    Assert loss = 0.0 when similarities are 1.0 for the correct class.
    Assert loss = 2.0 when similarities are −1.0 for the correct class.

11. **`test_inference_grid`**
    Assert `inference_grid_centers(400, 600, 100)` returns exactly 35 centers.
    Assert (0, 0) and (400, 600) are both in the list (edge centers included).

12. **`test_no_experiment_leakage`**
    Assert no experiment appears in more than one of train, val, or test.
    Assert each class has exactly 7 train, 1 val, 2 test experiments.
    Assert train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅ (all three pairs).

---

## 10. Summary: Files to Create / Modify

| File | Action | Notes |
|------|--------|-------|
| `environment.yml` | **Modify** | Add `- kornia` to dependencies |
| `patch_dataset.py` | **Create** | New Dataset with oversized patch extraction and `split_by_experiment_3way` |
| `model_patch.py` | **Create** | `FibrinPatchCNN` with `NormalizedLinear` cosine classifier |
| `augmentation_patch.py` | **Create** | Kornia GPU augmentation module |
| `train_patch.py` | **Create** | Training loop with `CosineLoss`, AdamW, CosineAnnealingWarmRestarts, checkpointing, Slurm-aware |
| `evaluate_patch.py` | **Create** | Cosine soft-vote inference over 35-patch grid; val and test modes |
| `slurm_patch.sh` | **Create** | HPC job + chain-restart script |
| `test_patch_pipeline.py` | **Create** | pytest visual + structural tests |
| `models/patch_5class/` | **Create dir** | New model output directory |
| `models/patch_5class/checkpoints/` | **Create dir** | Checkpoint storage |
| `logs/` | **Create dir** | Slurm stdout/stderr |

**Files that must NOT be modified:**
`preprocessing.py`, `data_loader.py`, `model.py`, `augmentation.py`,
`train_5class.py`, `train_3class.py`, all existing analysis scripts.
The existing pipeline continues to run independently.

---

## 11. Imports and Dependencies

All new files should import from existing modules where possible:

```python
# Shared across new files
from preprocessing import make_preprocessor       # unchanged
from data_loader import (
    CLASS_MAP, CLASS_NAMES,
    make_balanced_sampler,
    get_engine, load_metadata,
)
# Note: split_by_experiment and load_split_from_record from data_loader.py
# are NOT used here — the patch pipeline uses split_by_experiment_3way
# defined in patch_dataset.py, with an independent seed and a held-out test set.
import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Kornia version: any recent release on conda-forge (≥0.7) is compatible.
`K.AugmentationSequential`, `K.RandomRotation`, `K.CenterCrop`,
`K.RandomHorizontalFlip`, `K.RandomVerticalFlip` are all stable APIs.

---

## 12. Suggested Implementation Order for Claude Code

1. Update `environment.yml` and verify `mamba env update` succeeds
2. Create `model_patch.py`; verify with `test_model_forward`, `test_model_output_range`,
   `test_model_parameter_count`, `test_cosine_loss_range`
3. Create `augmentation_patch.py`; verify with `test_augmentation_output_shape`
4. Create `patch_dataset.py`; verify with `test_patch_shape`, `test_patch_coverage`
5. Create `test_patch_pipeline.py` (all tests) and run `pytest test_patch_pipeline.py -v -s`
6. Create `train_patch.py`
7. Create `evaluate_patch.py`; verify with `test_inference_grid`
8. Create `slurm_patch.sh`
9. Run a short smoke test locally (5 epochs, batch_size=4, no preload, num_workers=0)
   to confirm the end-to-end pipeline before submitting to HPC:
   ```bash
   python train_patch.py --num-epochs 5 --batch-size 4 --num-workers 0
   ```
