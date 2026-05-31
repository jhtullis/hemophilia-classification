# Implementation Plan: Fibrin Content Masks & Mask-Aware Patch Dataset

**For Claude Code — fibrin clot classification project**

This plan adds a self-contained masking subsystem alongside the existing pipeline.
No existing files are modified except `environment.yml`. The new masked dataset class
coexists with `FibrinDataset` and `FibrinPatchDataset` for full backwards compatibility.

---

## 0. Motivation and Design Principles

The existing model was shown (Figure 4.9 of the thesis) to have learned empty slide
background as a predictive feature for Factor IX deficiency. The masking system
addresses this by:

1. **Precomputing pixel-level content masks** for every image, identifying regions
   where fibrin or informative structure is present vs. absent.
2. **Precomputing patch-center masks** that indicate which patch centers yield patches
   containing enough informative content to be worth training on, taking into account
   rotational invariance.
3. **Providing a new `MaskedPatchDataset`** that samples patch centers exclusively
   from valid (non-empty) regions, with optional hard-gating (skip empty images
   entirely) or soft-weighting.
4. **Supporting both exploratory and grid images** in the new dataset class, pending
   grid image metadata.

Design constraints:
- **Versioned masks**: multiple mask types coexist in parallel directories.
  Updating one version never invalidates others.
- **PNG storage**: all masks are saved as viewable PNG files for manual inspection.
- **Lightweight first**: methods range from simple (intensity threshold) to more
  sophisticated (Frangi), all CPU-based and fast per image.
- **Backwards compatible**: `FibrinDataset` and `FibrinPatchDataset` are untouched.

---

## 1. Directory Layout

```
masks/
  v_intensity/
    0000.png          # pixel-level content mask, 600×400, grayscale uint8 (0=bg, 255=fg)
    0001.png
    ...
    flagged_empty.txt # list of image indices flagged as fully empty
    stats.csv         # per-image richness statistics for this mask version

  v_entropy/
    0000.png
    ...
    flagged_empty.txt
    stats.csv

  v_frangi/
    0000.png
    ...
    flagged_empty.txt
    stats.csv

  v_or/
    0000.png          # pixel-level OR of v_intensity, v_entropy, v_frangi
    ...
    flagged_empty.txt
    stats.csv

  v_vote2/
    0000.png          # pixel is fg if ≥2 of the 3 base masks agree
    ...
    flagged_empty.txt
    stats.csv

  patch_centers/
    p200_circle_v_intensity/
      0000.png        # patch-center mask: white pixel = valid center, black = skip
      ...
      stats.csv       # per-image: n_valid_centers, fraction_valid, flagged_empty

    p200_circle_v_entropy/
      ...

    p200_circle_v_frangi/
      ...

    p200_circle_v_or/
      ...

    p200_circle_v_vote2/
      ...
```

### Naming conventions

**Pixel-level mask directories:** `v_<method>` where method is one of:
`intensity`, `entropy`, `frangi`, `or`, `vote2`.

**Patch-center mask directories:** `p<size>_<geometry>_<pixel_mask_version>` where:
- `p200` = patch size 200 px (at preprocessed resolution)
- `circle` = the inscribed-circle geometry rule (described in Section 3)
- `v_intensity` etc. = which pixel mask was used to derive it

This naming means that if patch size changes (e.g. to `p150`), or if the geometry
rule changes (e.g. to `square` or `circle_strict`), or if the pixel mask is updated,
the directory name changes automatically, preventing stale caches.

---

## 2. New Files to Create

```
compute_masks.py           Precompute all pixel-level masks for the whole dataset
compute_patch_centers.py   Derive patch-center masks from pixel-level masks
inspect_masks.py           CLI tool for per-image and per-dataset mask inspection
masked_patch_dataset.py    MaskedPatchDataset class (new, does not modify patch_dataset.py)
```

---

## 3. Mask Generation Methods

All methods operate on the **preprocessed grayscale float32 tensor** at 600×400
resolution (the output of `make_preprocessor`, i.e. after 10× min-pooling).
This ensures the masks are aligned with the tensors the model actually sees.

### Method A — Intensity threshold (`v_intensity`)

Background in brightfield fibrin images is bright; fibrin is darker.
Otsu's method on the inverted image separates fibrin from background automatically
without a fixed threshold.

```python
import numpy as np
from skimage.filters import threshold_otsu

def make_intensity_mask(tensor: np.ndarray) -> np.ndarray:
    """
    tensor: (H, W) float32 in [0, 1], grayscale preprocessed image.
    Returns: (H, W) bool mask, True = foreground (fibrin present).
    """
    # Invert: fibrin is dark → becomes bright after inversion
    inverted = 1.0 - tensor
    thresh = threshold_otsu(inverted)
    return inverted > thresh
```

Fallback if Otsu fails (uniform image): mask is all-False (image flagged as empty).

### Method B — Local entropy (`v_entropy`)

Compute Shannon entropy on a sliding window. Uniform background has low entropy;
fibrin network structure has high entropy.

```python
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

def make_entropy_mask(tensor: np.ndarray, radius: int = 5,
                      threshold_percentile: float = 40.0) -> np.ndarray:
    """
    tensor: (H, W) float32 in [0, 1].
    Returns: (H, W) bool mask.
    radius: disk radius for local entropy computation.
    threshold_percentile: entropy values above this percentile are considered fg.
    """
    img_uint8 = (tensor * 255).astype(np.uint8)
    ent = rank_entropy(img_uint8, disk(radius))         # (H, W) float, entropy map
    thresh = np.percentile(ent, threshold_percentile)
    return ent > thresh
```

Default `threshold_percentile=40` means the top 60% of entropy values are
considered foreground. This is intentionally permissive to start; tighten as needed
after visual inspection.

### Method C — Frangi vesselness (`v_frangi`)

Reuses the multi-scale Frangi implementation already in the project. Max-fuses
vesselness across sigma range then thresholds.

```python
from skimage.filters import frangi

def make_frangi_mask(tensor: np.ndarray,
                     sigmas: tuple = (2, 4, 6, 8, 10),
                     threshold: float = 0.01) -> np.ndarray:
    """
    tensor: (H, W) float32 in [0, 1].
    Returns: (H, W) bool mask.
    sigmas: vesselness sigma values (in preprocessed-image pixels).
    threshold: vesselness response threshold after max fusion.
    """
    vesselness = np.max(
        np.stack([frangi(tensor, sigmas=(s,), black_ridges=True) for s in sigmas]),
        axis=0
    )
    return vesselness > threshold
```

Note: `black_ridges=True` because fibrin is dark on bright background in brightfield.
The sigma range (2–10 px at preprocessed resolution = 20–100 px at original resolution)
covers the thin fibrils and thicker bundles seen in the dataset.

### Composite masks (derived, not recomputed)

These are computed by loading the already-stored PNG masks for A, B, C and combining
them. They do not re-run any detection logic.

**`v_or`**: foreground if ANY of A, B, C says foreground.
```python
mask_or = mask_a | mask_b | mask_c
```

**`v_vote2`**: foreground if AT LEAST 2 of A, B, C say foreground.
```python
mask_vote2 = (mask_a.astype(int) + mask_b.astype(int) + mask_c.astype(int)) >= 2
```

---

## 4. Patch-Center Mask Generation (Inscribed-Circle Rule)

### Geometric rationale

A patch of size `P×P` (P=200 at preprocessed resolution) is drawn centered at `(cy, cx)`.
Under arbitrary rotation, the largest region of the patch that is guaranteed to be
included regardless of rotation angle is the **inscribed circle**: a circle of
radius `r = P // 2 = 100` centered at `(cy, cx)`.

The patch-center mask for a given center asks: **"Is enough of the inscribed circle
covered by foreground in the pixel mask?"**

This is both rotation-invariant (the circle doesn't change under rotation) and
conservative (if the circle passes the threshold, the patch will contain real content
after rotation).

### Algorithm

```python
def make_patch_center_mask(
    pixel_mask: np.ndarray,     # (H, W) bool, at preprocessed resolution
    patch_size: int = 200,
    min_foreground_fraction: float = 0.10,
) -> np.ndarray:
    """
    Returns a (H, W) bool array where True = this center yields a valid patch.

    For each candidate center (cy, cx), computes the fraction of the inscribed
    circle (radius = patch_size // 2) that falls on foreground pixels.
    A center is valid if that fraction exceeds min_foreground_fraction.

    Centers near the image edge where part of the circle falls outside the image
    are evaluated only over the in-bounds portion of the circle (same as training,
    where out-of-bounds areas become black padding → counted as background).

    min_foreground_fraction: fraction of circle area that must be foreground.
    Default 0.10 (10%) is intentionally permissive to start; increase after
    visual inspection of flagged images.
    """
    H, W = pixel_mask.shape
    r = patch_size // 2

    # Build a circular kernel for convolution-based counting
    # This gives the number of foreground pixels within the circle for each center
    Y, X = np.ogrid[-r:r+1, -r:r+1]
    circle_kernel = (X**2 + Y**2 <= r**2).astype(np.float32)
    circle_area = circle_kernel.sum()

    from scipy.ndimage import convolve
    # Count foreground pixels within circle at each center (full convolution)
    fg_count = convolve(pixel_mask.astype(np.float32), circle_kernel, mode='constant', cval=0.0)

    # For edge centers, the denominator should be the in-bounds circle area
    # Compute how much of the circle is in-bounds for each center
    inbounds_count = convolve(np.ones((H, W), dtype=np.float32), circle_kernel, mode='constant', cval=0.0)

    # Fraction of in-bounds circle that is foreground
    fg_fraction = np.where(inbounds_count > 0, fg_count / inbounds_count, 0.0)

    return fg_fraction >= min_foreground_fraction
```

The resulting patch-center mask is the same size as the pixel mask (600×400).
Each pixel in the patch-center mask represents one possible patch center location
and is queried directly at the sampled `(cy, cx)`.

---

## 5. Empty Image Detection and Flagging

An image is flagged as **fully empty** if the fraction of foreground pixels in its
pixel-level mask falls below a threshold:

```python
EMPTY_IMAGE_THRESHOLD = 0.01   # < 1% of pixels are foreground
```

Flagged images are written to `masks/<version>/flagged_empty.txt`, one image index
per line, along with their foreground fraction. This file is human-readable and is
used at training time to skip those images.

Images that are flagged in ALL five mask versions (`v_intensity`, `v_entropy`,
`v_frangi`, `v_or`, `v_vote2`) are the most confidently empty and should receive
priority in manual review.

---

## 6. Statistics Files (`stats.csv`)

Each mask version directory contains a `stats.csv` with one row per image and these columns:

| Column | Description |
|---|---|
| `idx` | Image index (0-based, matches filename) |
| `exp_type` | Class label (AC3, F08D, etc.) |
| `experiment` | Experiment ID |
| `image_type` | `exploratory` or `grid` |
| `fg_fraction` | Fraction of pixels classified as foreground |
| `n_fg_pixels` | Raw count of foreground pixels |
| `flagged_empty` | Bool: True if image falls below empty threshold |

Each `patch_centers/<version>/stats.csv` has:

| Column | Description |
|---|---|
| `idx` | Image index |
| `exp_type` | Class label |
| `experiment` | Experiment ID |
| `image_type` | `exploratory` or `grid` |
| `n_valid_centers` | Number of valid patch centers |
| `fraction_valid_centers` | Fraction of all (H×W) centers that are valid |
| `flagged_empty` | Bool: True if n_valid_centers == 0 |

---

## 7. `compute_masks.py`

### CLI interface

```
python compute_masks.py \
    --photo-dir   data/photos \
    --db-path     data/endpoint10.db \
    --mask-dir    masks \
    --methods     intensity entropy frangi or vote2 \
    [--indices    0 1 5 42]     # optional: compute only for these indices
    [--force]                   # recompute even if mask already exists
```

If `--methods or` or `--methods vote2` is specified, the three base masks
(`intensity`, `entropy`, `frangi`) must exist first. The script checks for this and
raises a clear error if they are missing.

### Processing flow

```
For each image index i:
  1. Load JPG from photo_dir
  2. Apply make_preprocessor() → tensor (1, 400, 600) float32
  3. Squeeze to (400, 600) numpy array
  4. For each requested method:
     a. Compute bool mask (400, 600)
     b. Compute fg_fraction
     c. If fg_fraction < EMPTY_IMAGE_THRESHOLD: append i to flagged_empty list
     d. Save mask as PNG to masks/<version>/<idx:04d>.png
        (True=255, False=0; saves as uint8 grayscale)
  5. For composite methods (or, vote2):
     a. Load the three base PNGs
     b. Combine
     c. Save
  6. After all images: write stats.csv and flagged_empty.txt for each version
```

### Progress reporting

Use `tqdm` for a per-image progress bar. Print a summary at completion:
```
v_intensity: 847/1000 images with >1% foreground, 153 flagged empty
v_entropy:   901/1000 images with >1% foreground, 99 flagged empty
v_frangi:    823/1000 images with >1% foreground, 177 flagged empty
```

---

## 8. `compute_patch_centers.py`

### CLI interface

```
python compute_patch_centers.py \
    --mask-dir      masks \
    --mask-versions intensity entropy frangi or vote2 \
    --patch-size    200 \
    --min-fg        0.10 \
    [--force]
```

For each requested mask version, reads every PNG from `masks/<version>/`,
computes the patch-center mask using the inscribed-circle rule, and saves to
`masks/patch_centers/p200_circle_<version>/`.

Also writes a `stats.csv` in each patch-center directory.

---

## 9. `inspect_masks.py`

A CLI inspection tool for manual verification. Three modes:

### Mode 1 — Single image overlay
```
python inspect_masks.py image \
    --idx 42 \
    --mask-version v_intensity \
    --photo-dir data/photos \
    --mask-dir masks \
    --output-dir masks/inspection
```
Saves a side-by-side PNG: [original image | pixel mask | patch-center mask overlaid on original].

### Mode 2 — Dataset summary
```
python inspect_masks.py summary \
    --mask-dir masks \
    --output masks/inspection/dataset_summary.csv
```
Prints a table comparing all mask versions:

```
version     | n_flagged | mean_fg_frac | min_fg_frac | max_fg_frac | flagged_by_all
------------|-----------|--------------|-------------|-------------|---------------
v_intensity |       153 |        0.412 |       0.000 |       0.981 |              -
v_entropy   |        99 |        0.538 |       0.000 |       0.995 |              -
v_frangi    |       177 |        0.387 |       0.000 |       0.972 |              -
v_or        |        87 |        0.601 |       0.000 |       0.998 |              -
v_vote2     |       131 |        0.452 |       0.000 |       0.989 |             72
```

Also breaks down flagged images by class and experiment — critical for verifying
that empty image prevalence doesn't confound the class split.

### Mode 3 — Class and experiment richness breakdown
```
python inspect_masks.py richness \
    --mask-version v_entropy \
    --mask-dir masks \
    --db-path data/endpoint10.db
```
Outputs a table of mean richness per experiment and class, formatted for easy copy-paste
into a lab notebook or thesis table.

---

## 10. `masked_patch_dataset.py`

### Overview

`MaskedPatchDataset` extends the patch-based approach with:
- Mask-gated center sampling (hard gate or soft weighted)
- Skipping of fully-empty images
- Optional inclusion of grid images alongside exploratory images
- Modular mask version selection

This class does **not** modify `FibrinPatchDataset` in any way.

### Class signature

```python
class MaskedPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,                # metadata DataFrame (exploratory images)
        photo_dir: str,                  # directory containing <idx:04d>.JPG
        preprocessor: Callable,          # from make_preprocessor()
        mask_dir: str,                   # root masks/ directory
        mask_version: str,               # e.g. "v_entropy"
        patch_center_version: str,       # e.g. "p200_circle_v_entropy"
        patches_per_image: int = 20,
        gate_mode: str = "hard",         # "hard" = only valid centers | "soft" = weighted
        include_grid: bool = False,      # if True, grid_df and grid_photo_dir are required
        grid_df: Optional[pd.DataFrame] = None,
        grid_photo_dir: Optional[str] = None,
        preload: bool = True,
    ):
```

### Grid image support (INCOMPLETE — pending metadata)

When `include_grid=True`, the dataset draws from both `df` (exploratory) and
`grid_df` (grid). The grid images must:
- Be split at the experiment level, using the same experiment assignments as `df`
  (i.e. if experiment `F09D_03` is in the training set based on exploratory images,
  all grid images from that experiment are also in training).
- Have their own masks precomputed using `compute_masks.py` (using their image
  paths, not the exploratory image paths).

**⚠️ INCOMPLETE**: The following sections cannot be implemented until the grid image
metadata CSV is provided:

```python
# TODO: implement grid metadata loading once metadata CSV is available.
# Expected columns: same as exploratory (idx, Experiment, Exp_Type, Slide_Type,
# Img_Name, Img_Fp) plus optionally a grid_position column (e.g. "3_5" for row 3, col 5).
# The idx values for grid images must not overlap with exploratory image indices
# (use a separate offset or a separate namespace).
raise NotImplementedError(
    "Grid image support requires the grid metadata CSV. "
    "Provide this file and update _load_grid_metadata() before enabling include_grid=True."
)
```

All other parts of the class (mask loading, hard/soft gating, patch sampling,
preloading) are fully implemented and work for exploratory-only mode.

### `__len__`

```python
# Count only non-empty images toward total length
return self._n_valid_images * self.patches_per_image
```

### `__getitem__` logic

```python
# Map idx → valid image index → base tensor + center mask
img_idx = idx // self.patches_per_image
base_idx = self._valid_image_indices[img_idx]
tensor = self._get_base_tensor(base_idx)
center_mask = self._get_center_mask(base_idx)   # (H, W) bool, True = valid center

if self.gate_mode == "hard":
    # Rejection sample from valid centers
    valid_ys, valid_xs = np.where(center_mask)
    if len(valid_ys) == 0:
        # Fallback: sample randomly (should not occur after empty-image filtering)
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
    else:
        k = np.random.randint(len(valid_ys))
        cy, cx = int(valid_ys[k]), int(valid_xs[k])

elif self.gate_mode == "soft":
    # Sample center proportional to pixel mask value at that location
    # Use the float fg_fraction map (before thresholding) if available,
    # otherwise fall back to the binary center mask
    weights = self._get_center_weights(base_idx)   # (H, W) float32
    weights_flat = weights.ravel()
    weights_flat = weights_flat / weights_flat.sum()
    flat_idx = np.random.choice(len(weights_flat), p=weights_flat)
    cy, cx = divmod(flat_idx, W)

# Oversized patch extraction (same as FibrinPatchDataset)
# ... (identical oversized extraction + return logic)
```

### Mask loading and caching

Center masks are loaded from PNG once at init and stored in RAM as bool arrays
if `preload=True`. This adds negligible memory (~1000 images × 600×400 × 1 byte ≈ 240 MB
for bool arrays — acceptable alongside the existing ~700 MB tensor cache).

```python
def _load_center_mask(self, idx: int) -> np.ndarray:
    mask_path = os.path.join(
        self.mask_dir, "patch_centers", self.patch_center_version,
        f"{idx:04d}.png"
    )
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Patch-center mask not found: {mask_path}\n"
            f"Run: python compute_patch_centers.py --mask-versions {self.mask_version}"
        )
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask > 127    # bool
```

### Empty image handling

At init, load `masks/<pixel_mask_version>/flagged_empty.txt` and exclude those
indices from the active image list. Log a warning:

```
MaskedPatchDataset: skipping 153 images flagged as empty by v_entropy.
Active images: 847/1000 (train: 612, val: 102, test: 133 after split).
```

---

## 11. Integration with `patch_dataset.py` / `train_patch.py`

`MaskedPatchDataset` is a **drop-in replacement** for `FibrinPatchDataset` at the
DataLoader construction step. It returns the same `(patch, label)` tuple shape
and the same oversized patch size (283×283) ready for Kornia GPU augmentation.

To use it in a training script:

```python
from masked_patch_dataset import MaskedPatchDataset

train_dataset = MaskedPatchDataset(
    df=train_df,
    photo_dir=PHOTO_DIR,
    preprocessor=make_preprocessor(...),
    mask_dir="masks",
    mask_version="v_entropy",
    patch_center_version="p200_circle_v_entropy",
    patches_per_image=20,
    gate_mode="hard",
    preload=True,
)
```

The rest of the training loop (`PatchAugmentation`, `FibrinPatchCNN`, checkpointing,
Slurm job chaining) is unchanged.

---

## 12. New Files Summary

| File | Purpose |
|---|---|
| `compute_masks.py` | Precompute pixel-level masks (A, B, C, OR, Vote2) for all images |
| `compute_patch_centers.py` | Derive patch-center masks from pixel-level masks |
| `inspect_masks.py` | CLI inspection tool; single-image overlay, dataset summary, richness table |
| `masked_patch_dataset.py` | `MaskedPatchDataset` class; backwards-compatible replacement for `FibrinPatchDataset` |

---

## 13. Environment Updates

Add to `environment.yml` if not already present:

```yaml
- scikit-image    # threshold_otsu, entropy, frangi (already likely installed)
- scipy           # convolve for patch-center mask computation
- tqdm            # progress bars in compute_masks.py
```

Verify with:
```bash
python -c "from skimage.filters import frangi, threshold_otsu; from scipy.ndimage import convolve; print('OK')"
```

---

## 14. Suggested Execution Order

```bash
# Step 1: Compute base pixel masks (A, B, C)
python compute_masks.py \
    --photo-dir data/photos \
    --db-path data/endpoint10.db \
    --mask-dir masks \
    --methods intensity entropy frangi

# Step 2: Inspect a few images visually from each method
python inspect_masks.py image --idx 0  --mask-version v_intensity --photo-dir data/photos --mask-dir masks --output-dir masks/inspection
python inspect_masks.py image --idx 42 --mask-version v_entropy   --photo-dir data/photos --mask-dir masks --output-dir masks/inspection
python inspect_masks.py image --idx 17 --mask-version v_frangi    --photo-dir data/photos --mask-dir masks --output-dir masks/inspection
# ... also inspect a known-empty image and a factor-IX image

# Step 3: Review dataset summary across all three base masks
python inspect_masks.py summary --mask-dir masks

# Step 4: If satisfied, compute composite masks
python compute_masks.py \
    --photo-dir data/photos \
    --db-path data/endpoint10.db \
    --mask-dir masks \
    --methods or vote2

# Step 5: Compute patch-center masks for all versions
python compute_patch_centers.py \
    --mask-dir masks \
    --mask-versions intensity entropy frangi or vote2 \
    --patch-size 200 \
    --min-fg 0.10

# Step 6: Review richness breakdown by class and experiment
python inspect_masks.py richness --mask-version v_entropy --mask-dir masks --db-path data/endpoint10.db

# Step 7: Train with the masked dataset (choose mask version based on inspection results)
# Modify train_patch.py or create train_patch_masked.py to use MaskedPatchDataset
```

---

## 15. Key Design Decisions and Rationale

**Why PNG and not NPY for mask storage?**
PNGs are directly viewable in any file browser or image viewer. The cost is that
loading requires `cv2.imread` instead of `np.load`, which is only marginally slower.
For float-valued intermediate maps (e.g. the raw entropy or vesselness response),
these are not stored — only the thresholded binary mask is saved. If float maps
are needed later, NPZ storage can be added as a parallel output without changing
the PNG-based pipeline.

**Why inscribed-circle rule for patch-center masks?**
Under arbitrary rotation (which the Kornia augmentation performs), the final
200×200 crop is guaranteed to contain the inscribed circle of the pre-rotation
patch. Therefore, a center is "informative" if and only if its inscribed circle
contains enough foreground content — rotation cannot make a circle-passing center
uninformative. This is the tightest rotation-invariant guarantee available.

**Why query the exact sampled center rather than a coarser grid?**
Preserves the continuous sampling distribution of `FibrinPatchDataset` and avoids
introducing a discrete grid artifact. The center mask is 600×400 (same resolution
as the image), so each possible center point has a direct lookup — no interpolation
or nearest-neighbor rounding needed.

**Why implement `v_or` and `v_vote2` from stored masks rather than recomputing?**
Modularity and speed. If any of A, B, C is updated (e.g. better threshold), the
composite masks can be regenerated by re-running `compute_masks.py --methods or vote2`
without re-running the expensive base methods. It also makes the composite logic
completely transparent and auditable.

**Why `min_foreground_fraction=0.10` as default?**
10% is permissive enough to pass most patches with any fibrin content, while
excluding patches that are almost entirely empty slide background. This is a
starting point; the `inspect_masks.py richness` output should guide tuning.
If the factor-IX class shows unusually few valid centers, the threshold may need
lowering to avoid class imbalance in training.

---

## 16. `tune_patch_threshold.py` — Visual Threshold Sweep

This standalone script helps choose an appropriate `min_foreground_fraction` before
committing to a value in `compute_patch_centers.py`. It randomly selects a small
sample of images, then produces a contact-sheet PNG showing how the patch-center
mask changes across a sweep of threshold values — all without running the full
dataset pipeline.

### Threshold sweep

The sweep covers the following values:

```
0.001, 0.002, 0.003, 0.004, 0.005,      # 0.1% → 0.5%  (very permissive)
0.006, 0.007, 0.008, 0.009, 0.010,      # 0.6% → 1.0%
0.02, 0.03, 0.04, 0.05,                 # 2% → 5%
0.06, 0.07, 0.08, 0.09, 0.10           # 6% → 10%
```

That is 19 threshold levels per image. These are defined as a constant list at the
top of the script so they are easy to edit.

### CLI interface

```bash
python tune_patch_threshold.py \
    --photo-dir   data/photos \
    --db-path     data/endpoint10.db \
    --mask-dir    masks \
    --mask-version v_entropy \           # which pixel mask to use as input
    --n-images    6 \                    # number of randomly sampled images
    --patch-size  200 \
    --seed        0 \                    # for reproducible random image selection
    --output-dir  masks/threshold_tuning
```

**Optional flags:**
- `--indices 42 87 301` — instead of random selection, use specific image indices
- `--include-empty` — include images flagged as fully empty (for comparison)

### Output

For each sampled image, the script saves one PNG contact sheet:

```
masks/threshold_tuning/
  image_<idx>_<exp_type>.png     # one file per sampled image
  summary_grid.png               # all images × all thresholds in one large grid
```

#### Per-image contact sheet layout

Each per-image PNG has two rows:

**Row 1:** The original preprocessed image (grayscale, 600×400) repeated across
all 19 columns, with valid patch centers overlaid as small green dots.

**Row 2:** The pixel-level mask (from `--mask-version`) shown as a binary image,
with the same green dots overlaid.

Each column corresponds to one threshold value, labeled above with the threshold
(e.g. `0.1%`, `0.5%`, `1%`, `2%`, …, `10%`).

Below each column pair, print two numbers:
- `N=412` — number of valid patch centers at this threshold
- `(68%)` — fraction of all possible centers that are valid

This lets you visually see how the green dot density changes as the threshold
tightens, and spot the threshold where obviously-empty regions start being excluded
while content-rich regions remain fully covered.

#### Summary grid layout

`summary_grid.png` is a large grid where:
- **Rows** = sampled images (one image per row; shows original image in leftmost cell)
- **Columns** = threshold values (19 columns)
- Each cell = the patch-center mask for that image at that threshold, with valid
  centers shown as white pixels on a black background

This gives an at-a-glance view across images: at what threshold do the Factor IX
images (which should have more empty regions) lose most of their valid centers
relative to denser images?

### Implementation sketch

```python
THRESHOLDS = [
    0.001, 0.002, 0.003, 0.004, 0.005,
    0.006, 0.007, 0.008, 0.009, 0.010,
    0.02,  0.03,  0.04,  0.05,
    0.06,  0.07,  0.08,  0.09,  0.10,
]

def run_sweep(pixel_mask: np.ndarray, patch_size: int) -> dict:
    """
    For each threshold in THRESHOLDS, compute the patch-center mask.
    Returns: dict mapping threshold → (H, W) bool array.
    """
    # Precompute the fg_fraction map once (expensive part) then threshold N times
    # This requires exposing the float fg_fraction from make_patch_center_mask()
    # rather than only the bool output — refactor accordingly.
    fg_fraction_map = compute_fg_fraction_map(pixel_mask, patch_size)
    return {t: fg_fraction_map >= t for t in THRESHOLDS}

def make_contact_sheet(
    original: np.ndarray,      # (H, W) float32 preprocessed image
    pixel_mask: np.ndarray,    # (H, W) bool
    center_masks: dict,        # threshold → (H, W) bool
    idx: int,
    exp_type: str,
) -> np.ndarray:               # returns RGB uint8 image
    """
    Build the two-row contact sheet for one image.
    Columns = thresholds, rows = [original+dots, pixel_mask+dots].
    """
    ...
```

Key implementation note: `make_patch_center_mask()` in `compute_patch_centers.py`
should be refactored to expose an intermediate `compute_fg_fraction_map()` function
that returns the raw float `(H, W)` fraction map. The sweep then calls this once
and applies 19 different `>=` thresholds — much faster than running the full
convolution 19 times.

### Suggested workflow

```bash
# 1. Compute pixel masks first if not done yet
python compute_masks.py --methods entropy --photo-dir data/photos --db-path data/endpoint10.db --mask-dir masks

# 2. Run threshold sweep on 6 random images, try all 5 mask versions to compare
for version in v_intensity v_entropy v_frangi v_or v_vote2; do
    python tune_patch_threshold.py \
        --photo-dir data/photos --db-path data/endpoint10.db \
        --mask-dir masks --mask-version $version \
        --n-images 6 --seed 0 \
        --output-dir masks/threshold_tuning/$version
done

# 3. Inspect summary grids in masks/threshold_tuning/*/
# 4. Pick threshold; run compute_patch_centers.py with --min-fg <chosen value>
```
