# Plan: Module 6 — Grad-CAM Saliency Maps

## Purpose

Grad-CAM produces class-specific spatial heatmaps showing *which image regions*
the model uses to make each classification decision. Unlike the channel-mean
activation maps in Module 3 (which are class-agnostic), Grad-CAM attributes
spatial importance to a specific target class. This is especially useful for:

- Understanding what features distinguish each fibrin clot phenotype
- Diagnosing misclassifications: what did the model "see" for the wrong class?

---

## File to Create: `gradcam.py`

---

## Prerequisites

- No dependency on `analysis_utils.py` (standalone)
- A trained model must exist at `<model_dir>/best_model.pth`

---

## CLI

```
python gradcam.py [--model-type TYPE] [--output-dir DIR] [--target-class CLASS]
                  [--misclassified-only]

Arguments:
  --model-type        5class | 3class_scratch | 3class_finetune  (default: 5class)
  --output-dir        Override output directory
                      (default: <model_dir>/analysis/gradcam/)
  --target-class      If set, only generate Grad-CAM for this class as target.
                      Default: generate for each image's predicted class.
  --misclassified-only If set, only process misclassified images (skip class_representatives).
```

---

## Imports

```python
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import (CLASS_MAP, CLASS_NAMES, load_split_from_record, filter_classes)
from hemophilia_analysis import load_model, _MODEL_REGISTRY
from model import FibrinCNN
from preprocessing import make_preprocessor
```

---

## Grad-CAM Target Layer

```python
GRADCAM_LAYER_IDX = 14   # model.features[14] = ReLU after Conv4 (before MaxPool4)
                          # Output shape: (256, 75, 50)  — higher resolution than post-MaxPool
```

**Why features[14]?** This is the ReLU activation immediately after the final
Conv2d layer (features[12]) and its BatchNorm (features[13]), but before the
final MaxPool (features[15]). Spatial size is 75×50 — 4× higher resolution than
the MaxPool output (37×25), giving more spatially precise saliency maps.

---

## Class `GradCAM`

```python
class GradCAM:
    """Compute Grad-CAM heatmaps for FibrinCNN.

    Usage:
        gcam = GradCAM(model, target_layer_idx=14)
        heatmap = gcam.compute(tensor, target_class=2)  # shape (75, 50)
        overlay = gcam.overlay(heatmap, tensor)          # shape (400, 600, 3) BGR
        gcam.remove_hooks()
    """

    def __init__(self, model: FibrinCNN, target_layer_idx: int = GRADCAM_LAYER_IDX):
        self.model = model
        self._activation = None
        self._gradient = None

        def save_activation(module, inp, out):
            self._activation = out

        def save_gradient(module, inp, out):
            # Register as a hook on the OUTPUT (not input) of the layer
            # torch does not support register_full_backward_hook cleanly in all versions;
            # instead, register on the activation tensor directly using retain_grad()
            # See implementation note below.
            pass

        self._fwd_hook = model.features[target_layer_idx].register_forward_hook(
            save_activation
        )
        # Gradient hook is set up via retain_grad() on the activation, see compute().
```

**Gradient capture implementation** (handle both old and new PyTorch):

```python
def compute(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """Compute Grad-CAM heatmap for the given target class.

    Args:
        tensor:       Preprocessed image tensor, shape (1, 1, 400, 600).
                      Must NOT have requires_grad (we set it internally).
        target_class: Integer class index.

    Returns:
        heatmap: np.ndarray of shape (75, 50), values in [0, 1].
    """
    self.model.eval()
    self.model.zero_grad()

    # Forward pass — activation is saved by the hook
    logits = self.model(tensor)

    # Make the saved activation require grad so we can compute its gradient
    self._activation.retain_grad()

    # Backpropagate the target class logit
    target = logits[0, target_class]
    target.backward()

    # Gradient w.r.t. activation: shape (1, 256, 75, 50)
    grad = self._activation.grad   # (1, 256, 75, 50)
    act  = self._activation.detach()  # (1, 256, 75, 50)

    # Global average pool over spatial dims → channel weights α_k, shape (256,)
    alpha = grad[0].mean(dim=(1, 2))  # (256,)

    # Weighted sum of activation maps
    cam = (alpha[:, None, None] * act[0]).sum(dim=0)  # (75, 50)

    # ReLU to keep only positive contributions
    cam = F.relu(cam).cpu().detach().numpy()

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def overlay(self, heatmap: np.ndarray,
            tensor: torch.Tensor,
            alpha: float = 0.5) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the preprocessed grayscale image.

    Args:
        heatmap: (75, 50) array in [0, 1].
        tensor:  (1, 1, 400, 600) preprocessed image tensor.
        alpha:   Blend weight for the colormap overlay (default 0.5).

    Returns:
        BGR image of shape (400, 600, 3), uint8.
    """
    # Upsample heatmap to image size: cv2.resize takes (width, height)
    heatmap_up = cv2.resize(heatmap, (600, 400), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (heatmap_up * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # (400, 600, 3) BGR

    # Convert grayscale tensor to 3-channel BGR
    gray = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)  # (400, 600)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)                # (400, 600, 3)

    # Blend
    blended = cv2.addWeighted(gray_bgr, 1 - alpha, colormap, alpha, 0)
    return blended


def remove_hooks(self):
    self._fwd_hook.remove()
```

---

## Helper: Load Test Images

```python
def _load_preprocessed(idx: int, photos_dir: str, preprocessor) -> torch.Tensor:
    """Load and preprocess one image. Returns (1,1,400,600) tensor."""
    img_bgr = cv2.imread(os.path.join(photos_dir, f"{idx:04d}.JPG"))
    if img_bgr is None:
        raise FileNotFoundError(f"Image {idx:04d}.JPG not found")
    return preprocessor(img_bgr).unsqueeze(0)  # (1,1,400,600)
```

---

## Helper: Run Inference

```python
def _collect_results(model, test_df, photos_dir, device, preprocessor,
                     class_map, class_names):
    """Return a list of dicts with idx, Exp_Type, true_label, pred_label, correct, probs."""
    rows = []
    model.eval()
    with torch.no_grad():
        for _, row in test_df.iterrows():
            tensor = _load_preprocessed(int(row["idx"]), photos_dir, preprocessor).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            true_label = class_map[row["Exp_Type"]]
            pred_label = int(probs.argmax())
            rows.append({
                "idx":        int(row["idx"]),
                "Exp_Type":   row["Exp_Type"],
                "true_label": true_label,
                "pred_label": pred_label,
                "correct":    true_label == pred_label,
                "confidence": float(probs.max()),
                "probs":      probs,
            })
    return rows
```

---

## Output A: `class_representatives/` — `<class>_gradcam_top3.png`

```python
def plot_class_representatives(gcam, results, class_names, inv_class_map,
                                photos_dir, device, preprocessor, out_dir):
```

**Algorithm**:
1. For each class in class_names:
   a. Filter results to correct==True AND Exp_Type==cls
   b. Sort by confidence descending, take top 3
   c. For each of the 3 images:
      - Load tensor: `_load_preprocessed(idx, photos_dir, preprocessor)`
      - Compute Grad-CAM for the TRUE class
      - Create side-by-side plot:
        Col 0: grayscale image (imshow with gray cmap)
        Col 1: Grad-CAM overlay (imshow with no cmap, already BGR converted to RGB)
   d. Add title per row: f"#{rank}  idx={idx:04d}  conf={confidence:.3f}"
   e. Add softmax probabilities as subtitle (bottom of figure):
      one line: f"Probs: " + "  ".join(f"{cn}={p:.3f}" for cn, p in zip(class_names, probs))
   f. Overall figure title: f"Grad-CAM: {cls} (True class, top-3 confident correct)"
   g. Save as `<out_dir>/class_representatives/{cls}_gradcam_top3.png`

**Layout**:
```python
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
# row i: top-(i+1) confidence image
# col 0: gray preprocessed image
# col 1: grad-CAM overlay
```

**Note**: If a class has fewer than 3 correct predictions, use however many exist.

---

## Output B: `misclassified/` — `<idx>_true-<true>_pred-<pred>.png`

```python
def plot_misclassified(gcam, results, inv_class_map,
                       photos_dir, device, preprocessor, out_dir):
```

**Algorithm**:
For each incorrect result:
1. Load tensor
2. Compute Grad-CAM for the **TRUE** class
3. Compute Grad-CAM for the **PREDICTED** class
4. Layout: 1 row × 3 columns
   - Col 0: Grayscale image  title="Input (TRUE: {true_cls})"
   - Col 1: Grad-CAM for TRUE class   title=f"Grad-CAM: {true_cls} (true)"
   - Col 2: Grad-CAM for PRED class   title=f"Grad-CAM: {pred_cls} (predicted)"
5. Figure suptitle: f"MISCLASSIFIED  idx={idx:04d}  TRUE={true_cls} → PRED={pred_cls}"
6. Add probability bar below as text:
   f"Probs: " + "  ".join(f"{cn}={p:.3f}" for ...)
7. Save as `{out_dir}/misclassified/{idx:04d}_true-{true_cls}_pred-{pred_cls}.png`

**Important**: Call `model.zero_grad()` between the two Grad-CAM computations to
prevent gradient accumulation.

---

## Output C: `class_comparison_grid.png`

```python
def plot_class_comparison_grid(gcam, results, class_names, inv_class_map,
                                photos_dir, device, preprocessor, out_dir):
```

**Algorithm**:
1. For each class, pick the single highest-confidence correct prediction
2. Layout: len(class_names) rows × 2 columns (gray | Grad-CAM)
3. Each row labeled with class name on the left (ylabel)
4. Column headers: "Preprocessed Image" | "Grad-CAM (predicted class)"
5. Figure title: "Grad-CAM Overview — One Representative Image Per Class"
6. `figsize=(10, 4 * len(class_names))`
7. Save as `class_comparison_grid.png`

---

## Output D: `gradcam_summary.txt`

```python
def write_summary(gcam, results, class_names, inv_class_map,
                  photos_dir, device, preprocessor, out_dir):
```

For each class, process all correct test images. For each:
- Compute Grad-CAM for the true class → heatmap (75, 50)
- Compute centroid:
  ```python
  # Centroid = weighted average of (row, col) coordinates
  h, w = heatmap.shape
  rows_idx, cols_idx = np.mgrid[0:h, 0:w]
  total_weight = heatmap.sum() + 1e-8
  centroid_row = (heatmap * rows_idx).sum() / total_weight  # in [0, 74]
  centroid_col = (heatmap * cols_idx).sum() / total_weight  # in [0, 49]
  # Normalize to [0,1]: row/74, col/49
  centroid_y = centroid_row / (h - 1)   # 0=top, 1=bottom
  centroid_x = centroid_col / (w - 1)   # 0=left, 1=right
  ```
- Average centroid_y and centroid_x over all correct images of that class

**Output file content**:
```
=== Grad-CAM Spatial Attention Summary ===
Model type: 5class
Layer: features[14] (ReLU after Conv4, spatial size 75×50)

Note: centroid position normalized to [0,1].
x: 0=left, 1=right.  y: 0=top, 1=bottom.
A centered network attends around (0.5, 0.5).

Class   N_correct  Centroid_X  Centroid_Y  Interpretation
AC3     XX         0.xxx       0.xxx       (center/left/right, top/bottom)
F08D    XX         0.xxx       0.xxx
F09D    XX         0.xxx       0.xxx
F11D    XX         0.xxx       0.xxx
NC1     XX         0.xxx       0.xxx

Misclassified images: XX / 201
Grad-CAM images saved to: models/5class/analysis/gradcam/
```

---

## Main Flow

```python
def main():
    args = parse_args()
    device = torch.device("cpu")

    model_dir, num_classes, class_map = _MODEL_REGISTRY[args.model_type]
    model_path = os.path.join(model_dir, "best_model.pth")
    model = load_model(model_path, device=device, num_classes=num_classes)
    class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]
    inv_class_map = {v: k for k, v in class_map.items()}

    record_path = os.path.join("models", "5class", "train_record.json")
    DB_PATH_LOCAL = os.path.join(os.path.dirname(__file__), "data", "test_db.db")
    PHOTOS_DIR_LOCAL = os.path.join(os.path.dirname(__file__), "data", "photos")
    _, test_df = load_split_from_record(record_path, DB_PATH_LOCAL)
    if num_classes == 3:
        test_df = filter_classes(test_df, ["F08D", "F09D", "F11D"])

    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)
    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "gradcam")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_dir, "class_representatives")).mkdir(exist_ok=True)
    Path(os.path.join(out_dir, "misclassified")).mkdir(exist_ok=True)

    print(f"Collecting predictions for {len(test_df)} test images ...")
    results = _collect_results(model, test_df, PHOTOS_DIR_LOCAL, device,
                                preprocessor, class_map, class_names)
    n_correct   = sum(r["correct"] for r in results)
    n_incorrect = len(results) - n_correct
    print(f"  Correct: {n_correct}  Incorrect: {n_incorrect}")

    gcam = GradCAM(model, target_layer_idx=GRADCAM_LAYER_IDX)

    if not args.misclassified_only:
        print("Generating class representative Grad-CAMs ...")
        plot_class_representatives(gcam, results, class_names, inv_class_map,
                                   PHOTOS_DIR_LOCAL, device, preprocessor,
                                   os.path.join(out_dir, "class_representatives"))

        print("Generating class comparison grid ...")
        plot_class_comparison_grid(gcam, results, class_names, inv_class_map,
                                   PHOTOS_DIR_LOCAL, device, preprocessor, out_dir)

    print(f"Generating Grad-CAMs for {n_incorrect} misclassified images ...")
    plot_misclassified(gcam, results, inv_class_map, PHOTOS_DIR_LOCAL,
                       device, preprocessor,
                       os.path.join(out_dir, "misclassified"))

    print("Computing spatial attention summary ...")
    write_summary(gcam, results, class_names, inv_class_map,
                  PHOTOS_DIR_LOCAL, device, preprocessor, out_dir)

    gcam.remove_hooks()
    print(f"\nAll outputs saved to: {out_dir}/")
```

---

## Gradient Capture — Implementation Note

The `retain_grad()` approach works reliably:

```python
# In compute():
logits = self.model(tensor)           # forward pass; hook fires, sets self._activation
self._activation.retain_grad()        # allow non-leaf to accumulate grad
target = logits[0, target_class]
target.backward()                     # backprop
grad = self._activation.grad          # now accessible
```

Between processing different images, call `self.model.zero_grad()` to clear
accumulated gradients. Also call `tensor.grad = None` if tensor has requires_grad.

---

## Verification

```bash
python gradcam.py --model-type 5class
```

Expected outputs in `models/5class/analysis/gradcam/`:
- `class_representatives/<class>_gradcam_top3.png` — one file per class
- `misclassified/<idx>_true-<true>_pred-<pred>.png` — one file per misclassified image
- `class_comparison_grid.png`
- `gradcam_summary.txt`

Quick smoke test (just misclassified images, faster):
```bash
python gradcam.py --model-type 5class --misclassified-only
```
