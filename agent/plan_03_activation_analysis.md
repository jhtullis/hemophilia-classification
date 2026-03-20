# Plan: Module 3 — Activation Map Analysis

## Purpose

The existing `visualize_weights.py::plot_activation_maps` shows one representative
test image per class with a simple channel-mean activation. This module goes further:
it captures activations for *all* test images, computes class-aggregate spatial
attention, compares correct vs incorrect predictions, and overlays heatmaps on
real images to show what the network is actually attending to.

---

## File to Create: `activation_analysis.py`

---

## Prerequisites

- `analysis_utils.py` must exist (see `plan_shared_refactor.md`)
- A trained model must exist at `<model_dir>/best_model.pth`

---

## CLI

```
python activation_analysis.py [--model-type TYPE] [--output-dir DIR] [--max-images N]

Arguments:
  --model-type  5class | 3class_scratch | 3class_finetune  (default: 5class)
  --output-dir  Override output directory (default: <model_dir>/analysis/activation/)
  --max-images  Cap number of images processed (default: all; useful for quick testing)
```

---

## Imports

```python
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis_utils import (
    MODEL_REGISTRY, load_model_from_registry, run_inference_full,
    get_test_split, PHOTOS_DIR, DB_PATH
)
from preprocessing import make_preprocessor
```

---

## Hook-Based Activation Capture

### Layer target indices

```python
POOL_INDICES = [3, 7, 11, 15]   # MaxPool outputs of each block

# Spatial dimensions of each block's MaxPool output
BLOCK_SHAPES = {
    3:  (300, 200),  # Block 1: 32×300×200
    7:  (150, 100),  # Block 2: 64×150×100
    11: (75,  50),   # Block 3: 128×75×50
    15: (37,  25),   # Block 4: 256×37×25
}
BLOCK_LABELS = ["Block 1 (300×200)", "Block 2 (150×100)",
                "Block 3 (75×50)",   "Block 4 (37×25)"]
```

### Capture function

```python
def capture_activations_all(model, test_df, photos_dir, device, preprocessor,
                            class_map, class_names, max_images=None):
    """Run all test images through the model and record per-image activations.

    Returns:
        results_df: DataFrame from run_inference_full
        act_dict:   dict mapping image idx → dict {pool_idx: np.ndarray (H, W)}
                    where the value is the channel-mean of the activation map

    Implementation:
    1. Register forward hooks on model.features[3,7,11,15]
    2. For each row in test_df (up to max_images):
       a. Load + preprocess image
       b. model(tensor) — hooks fire and populate activations
       c. For each pool_idx, compute channel-mean: activations[pool_idx].mean(axis=0)
          → shape (H, W)
       d. Store in act_dict[idx][pool_idx]
    3. Remove hooks
    4. Return act_dict and the results_df (call run_inference_full for labels)
    """
```

**Note**: Because hooks capture activation tensors during forward pass, and
`run_inference_full` also does a forward pass internally via `collect_predictions`,
it is more efficient to do ONE combined pass rather than two. Implement the hook
capture loop manually (not via `run_inference_full`) so each image is processed
only once. The results DataFrame can then be assembled manually using the same
logic as `run_inference_full`.

### Single-pass implementation outline

```python
activations_storage = {}

def make_hook(idx):
    def hook(module, inp, out):
        activations_storage[idx] = out.detach().cpu()
    return hook

hooks = [model.features[i].register_forward_hook(make_hook(i))
         for i in POOL_INDICES]

act_dict = {}
rows = []

model.eval()
with torch.no_grad():
    for _, row in test_df.iterrows():
        img_path = os.path.join(photos_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        tensor = preprocessor(img_bgr).unsqueeze(0).to(device)

        logits = model(tensor)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(prob.argmax())
        true_label = class_map[row["Exp_Type"]]

        # Store channel-mean activation maps
        act_dict[int(row["idx"])] = {
            pidx: activations_storage[pidx][0].mean(dim=0).numpy()
            for pidx in POOL_INDICES
        }

        rows.append({
            "idx":        int(row["idx"]),
            "Experiment": row["Experiment"],
            "Exp_Type":   row["Exp_Type"],
            "Slide_Type": row["Slide_Type"],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct":    true_label == pred_label,
            "confidence": float(prob.max()),
        })

for h in hooks:
    h.remove()

results_df = pd.DataFrame(rows)
```

---

## Output A: `aggregate_activations.png`

```python
def plot_aggregate_activations(results_df, act_dict, class_names, out_dir):
```

**Algorithm**:
1. For each class in class_names, collect all image indices with that true label
2. For each block (POOL_INDICES), average their channel-mean activation maps
   → class_avg[cls][pidx] = np.mean([act_dict[idx][pidx] for idx in cls_indices], axis=0)
3. Layout: rows = classes, cols = 4 blocks
   ```
   fig, axes = plt.subplots(len(class_names), 4,
                            figsize=(4*4, 3*len(class_names)), squeeze=False)
   ```
4. Each cell: `ax.imshow(class_avg[cls][pidx], cmap="viridis", interpolation="bilinear")`
5. Row labels (left side): class name — set with `ax.set_ylabel(cls, fontsize=12)`
   on the first column only
6. Column headers (top): block labels — set with `ax.set_title(label, fontsize=10)`
   on the first row only
7. Turn off all axes ticks
8. Add a suptitle: "Class-Average Channel-Mean Activation Maps"
9. Save as `aggregate_activations.png` at dpi=120

---

## Output B: `correct_vs_incorrect_attention.png`

```python
def plot_correct_vs_incorrect(results_df, act_dict, class_names, out_dir,
                               min_incorrect=3):
```

**Algorithm**:
1. For each class, split indices into `correct_idx` and `incorrect_idx`
2. Skip class if `len(incorrect_idx) < min_incorrect` (default 3)
3. For classes with enough incorrect examples, compute:
   - `avg_correct[cls]`   = mean of act_dict[idx][15] (Block 4, pidx=15) over correct_idx
   - `avg_incorrect[cls]` = mean of act_dict[idx][15] over incorrect_idx
4. Layout: one row per qualifying class, 2 columns (Correct | Incorrect)
   ```
   n_rows = number of qualifying classes
   fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.5 * n_rows), squeeze=False)
   ```
5. Each pair: imshow with same vmin/vmax for the pair (normalize jointly)
6. Left column header: "Correct predictions"
7. Right column header: "Incorrect predictions"
8. Row label: class name + f" (n_correct={X}, n_incorrect={Y})"
9. If no classes qualify, print a warning and skip this figure
10. Save as `correct_vs_incorrect_attention.png`

---

## Output C: `spatial_activation_stats.csv`

```python
def save_spatial_stats(results_df, act_dict, out_dir):
```

**Algorithm**:
1. For each image, extract the Block 4 activation map: `a = act_dict[idx][15]`
   Shape: (37, 25)
2. Divide into 4 quadrants:
   - top-left:     `a[:18, :12]`
   - top-right:    `a[:18, 13:]`
   - bottom-left:  `a[19:, :12]`
   - bottom-right: `a[19:, 13:]`
   (Use floor division: H//2=18 rows top, W//2=12 cols left)
3. Compute mean activation in each quadrant
4. Build a DataFrame:
   ```
   idx, Exp_Type, Slide_Type, correct, confidence,
   act_top_left, act_top_right, act_bottom_left, act_bottom_right, act_total_mean
   ```
5. Join with results_df columns, save to `spatial_activation_stats.csv`

---

## Output D: `activation_overlay_examples.png`

```python
def plot_activation_overlays(results_df, act_dict, class_names, photos_dir,
                              preprocessor, out_dir):
```

**Algorithm**:
1. For each class, pick the image with the highest confidence correct prediction
   (from results_df, filter correct==True and Exp_Type==cls, then sort by confidence desc)
2. Load the original image: `img_bgr = cv2.imread(f"{photos_dir}/{idx:04d}.JPG")`
3. Preprocess to get grayscale tensor: `tensor = preprocessor(img_bgr)` → shape (1, 400, 600)
4. Get Block 4 activation: `a = act_dict[idx][15]` → shape (37, 25)
5. Upsample activation to (400, 600) using `cv2.resize(a, (600, 400), interpolation=cv2.INTER_LINEAR)`
   Note: cv2.resize takes (width, height) = (600, 400) for a (400×600) output.
6. Normalize upsampled map to [0, 1]
7. Convert grayscale tensor to uint8: `gray = (tensor.squeeze().numpy() * 255).astype(np.uint8)`
8. Convert gray to BGR: `gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)`
9. Apply jet colormap to activation: `heatmap = cv2.applyColorMap((a_up*255).astype(np.uint8), cv2.COLORMAP_JET)`
10. Blend: `overlay = cv2.addWeighted(gray_bgr, 0.5, heatmap, 0.5, 0)`
11. Layout: 1 row per class, 2 columns (gray | overlay)
    `fig, axes = plt.subplots(len(class_names), 2, figsize=(12, 4*len(class_names)))`
12. Display gray image in col 0, overlay in col 1
13. Annotate col 0 title "Preprocessed" and col 1 title "Block 4 Activation Overlay"
14. Save as `activation_overlay_examples.png`

---

## Main Flow

```python
def main():
    args = parse_args()
    device = torch.device("cpu")

    model, model_dir, num_classes, class_map, class_names = \
        load_model_from_registry(args.model_type, device)
    test_df = get_test_split(args.model_type)
    if args.max_images:
        test_df = test_df.head(args.max_images)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "activation")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Capturing activations for {len(test_df)} images ...")
    results_df, act_dict = capture_activations_all(
        model, test_df, PHOTOS_DIR, device, preprocessor, class_map, class_names
    )
    acc = results_df["correct"].mean()
    print(f"  Accuracy on this subset: {acc:.4f}")

    print("Generating aggregate activation maps ...")
    plot_aggregate_activations(results_df, act_dict, class_names, out_dir)

    print("Generating correct vs incorrect attention comparison ...")
    plot_correct_vs_incorrect(results_df, act_dict, class_names, out_dir)

    print("Computing spatial activation statistics ...")
    save_spatial_stats(results_df, act_dict, out_dir)

    print("Generating activation overlay examples ...")
    plot_activation_overlays(results_df, act_dict, class_names, PHOTOS_DIR,
                             preprocessor, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
```

---

## Verification

```bash
python activation_analysis.py --model-type 5class
```

Expected outputs in `models/5class/analysis/activation/`:
- `aggregate_activations.png` — 5×4 grid of heatmaps (classes × blocks)
- `correct_vs_incorrect_attention.png` — paired Block-4 maps per class (or warning)
- `spatial_activation_stats.csv` — 201 rows with quadrant activation values
- `activation_overlay_examples.png` — 5×2 grid (gray | overlay) for best image per class

Quick smoke test:
```bash
python activation_analysis.py --model-type 5class --max-images 20
# Should complete in <60s and produce all 4 outputs with truncated data
```
