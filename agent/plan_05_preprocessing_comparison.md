# Plan: Module 5 — Preprocessing Comparison

## Purpose

Five grayscale conversion methods are implemented in `preprocessing.py`, but only
`lab_l` has been used for training. This module evaluates the *robustness* of the
trained 5-class model when fed images processed with the other four methods. This is
a sensitivity/robustness test rather than a proper comparison (which would require
retraining a separate model for each method).

**Caveat (must appear in the report output)**: The model was trained on `lab_l` images.
Evaluating it with other preprocessing methods constitutes out-of-distribution input.
Lower accuracy under alternative methods reflects that the model has adapted to `lab_l`
image statistics, not necessarily that those methods are inherently worse. For a
proper comparison, train one model per method. The visual comparison and relative
ranking remain informative nonetheless.

---

## File to Create: `preprocessing_comparison.py`

---

## Prerequisites

- No dependency on `analysis_utils.py` (this module is standalone)
- A trained model must exist at `<model_dir>/best_model.pth`
- This script is primarily designed for the 5-class model; running on 3-class models
  is supported but the visual comparison will only show hemophilia classes

---

## CLI

```
python preprocessing_comparison.py [--model-type TYPE] [--output-dir DIR]

Arguments:
  --model-type  5class | 3class_scratch | 3class_finetune  (default: 5class)
  --output-dir  Override output directory (default: <model_dir>/analysis/preprocessing/)
```

---

## Imports

```python
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import (CLASS_MAP, CLASS_NAMES, FibrinDataset,
                         load_split_from_record, filter_classes)
from evaluate import evaluate_model
from hemophilia_analysis import load_model, _MODEL_REGISTRY
from preprocessing import make_preprocessor, to_grayscale, min_pool
```

---

## Constants

```python
GRAY_METHODS = ["lab_l", "luminance", "hsv_v", "hsv_s", "green"]
POOL_FACTOR  = 10
DB_PATH      = os.path.join(os.path.dirname(__file__), "data", "test_db.db")
PHOTOS_DIR   = os.path.join(os.path.dirname(__file__), "data", "photos")

HEMOPHILIA_CLASSES = ["F08D", "F09D", "F11D"]
CLASS_MAP_3 = {"F08D": 0, "F09D": 1, "F11D": 2}
CLASS_NAMES_3 = ["F08D", "F09D", "F11D"]
```

---

## Helper: `_load_test_df`

```python
def _load_test_df(model_type: str):
    """Return (test_df, class_names) for the given model type."""
    record_path = os.path.join("models", "5class", "train_record.json")
    _, test_df = load_split_from_record(record_path, DB_PATH)
    _, num_classes, _ = _MODEL_REGISTRY[model_type]
    if num_classes == 3:
        test_df = filter_classes(test_df, HEMOPHILIA_CLASSES)
        return test_df, CLASS_NAMES_3
    return test_df, CLASS_NAMES
```

---

## Output A: `method_visual_comparison.png`

```python
def plot_visual_comparison(test_df, class_names, out_dir):
```

**Algorithm**:
1. For each class in class_names, find the first test image by `idx` (smallest idx):
   ```python
   rep_images = {}
   for cls in class_names:
       rows = test_df[test_df["Exp_Type"] == cls].sort_values("idx")
       if rows.empty: continue
       idx = int(rows.iloc[0]["idx"])
       img_bgr = cv2.imread(os.path.join(PHOTOS_DIR, f"{idx:04d}.JPG"))
       if img_bgr is not None:
           rep_images[cls] = img_bgr
   ```
2. Layout: rows = classes, cols = 5 grayscale methods
   - `figsize=(3.5 * 5, 3 * len(class_names))`
3. For each (class, method):
   - Apply `to_grayscale(img_bgr, method=method)` → H×W uint8
   - Apply `min_pool(gray, factor=POOL_FACTOR)` → 600×400 float [0,1]
   - Scale to [0,255] uint8 for display
   - `ax.imshow(processed, cmap="gray", vmin=0, vmax=255)`
   - `ax.set_title(method if row==0 else "")` for column headers on first row
   - `ax.set_ylabel(cls if col==0 else "")` for row labels on first column
   - `ax.axis("off")`
4. `fig.suptitle("Grayscale Method Visual Comparison (min-pooled 600×400)", fontsize=13)`
5. Save as `method_visual_comparison.png` at dpi=100

---

## Output B: Accuracy table + report

```python
def run_accuracy_comparison(model, model_type, test_df, class_names,
                             num_classes, class_map, out_dir):
```

**Algorithm**:
1. For each method in GRAY_METHODS:
   a. Build preprocessor: `preprocessor = make_preprocessor(gray_method=method, pool_factor=POOL_FACTOR)`
   b. Build test dataset: `test_ds = FibrinDataset(test_df, PHOTOS_DIR, preprocessor, augment=False)`
   c. Build DataLoader: `loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)`
   d. Note: FibrinDataset uses CLASS_MAP for labels; for 3-class use FibrinDataset3
      from train_3class.py logic — BUT to keep this module standalone, implement
      inline label remapping:
      If num_classes == 3, wrap FibrinDataset with a local subclass that overrides
      __getitem__ to use class_map for label lookup.
   e. Run `evaluate_model(model, loader, device, class_names)` → result dict
   f. Extract: overall accuracy, per-class F1 (from classification_report output)
2. Collect all results into a list of dicts

For the per-class F1 extraction, `evaluate_model` returns `result["report_str"]`
(text) and `result["accuracy"]`. To get per-class F1 programmatically, call
`evaluate.classification_report(result["preds"], result["labels"], class_names)`
directly — this returns a dict. Check evaluate.py source for the exact return format;
if it only returns a string, parse per-class F1 from the numeric arrays:

```python
from evaluate import classification_report as _clf_report
report = _clf_report(result["preds"], result["labels"], class_names)
# if report is a dict with per-class stats, extract F1
# if report is a string, compute F1 directly:
from evaluate import confusion_matrix as cm_fn
cm = cm_fn(result["preds"], result["labels"], len(class_names))
per_class_f1 = {}
for i, cls in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    per_class_f1[cls] = f1
```

**Save `preprocessing_comparison_table.csv`**:
```
method,overall_acc,<class1>_f1,<class2>_f1,...
lab_l,0.xxxx,0.xxx,...
luminance,...
```

**Save `preprocessing_accuracy_bars.png`**:
- Grouped bar chart
- x-axis: GRAY_METHODS (5 groups)
- Within each group: one bar per class + one bar for "Overall" (dashed outline)
- y-axis: accuracy/F1 value 0–1
- Legend: class colors + "Overall" in black
- Title: "Per-Method Accuracy (model trained on lab_l)"
- Subtitle annotation: "Note: cross-method evaluation; retraining needed for fair comparison"
- `figsize=(12, 6)`

**Save `preprocessing_comparison_report.txt`**:
```
=== Preprocessing Sensitivity Analysis ===

Model type:   5class (trained on lab_l)
Test images:  201

IMPORTANT: This model was trained exclusively on lab_l preprocessing.
Evaluation with other methods tests robustness to preprocessing variation,
NOT the optimal accuracy achievable by training a dedicated model per method.

Method Ranking by Overall Accuracy:
  1. lab_l       0.xxxx  (training method — expected highest)
  2. luminance   0.xxxx
  ...

Per-Class F1 by Method:
         lab_l  luminance  hsv_v  hsv_s  green
AC3      0.xxx  0.xxx      ...
F08D     ...
...

Most Robust Class:     <class> (smallest F1 drop from lab_l)
Least Robust Class:    <class> (largest F1 drop from lab_l)
Most Robust Method:    <method> (closest accuracy to lab_l)

Extension: To measure optimal per-method accuracy, retrain a separate model
for each method using train_5class.py with the GRAY_METHOD constant changed.
```

---

## Inline FibrinDataset Subclass (for 3-class mode)

Since this module must be standalone, define the 3-class dataset subclass locally
if needed:

```python
class _FibrinDataset3(FibrinDataset):
    """Local FibrinDataset subclass that uses a custom class_map."""
    def __init__(self, df, photo_dir, preprocessor, class_map):
        super().__init__(df, photo_dir, preprocessor, augment=False)
        self._class_map = class_map

    def __getitem__(self, idx):
        tensor, _ = super().__getitem__(idx)
        row = self.df.iloc[idx]
        label = self._class_map[row["Exp_Type"]]
        return tensor, label
```

Use `_FibrinDataset3` when num_classes == 3 instead of `FibrinDataset`.

---

## Main Flow

```python
def main():
    args = parse_args()
    device = torch.device("cpu")

    model_dir, num_classes, class_map = _MODEL_REGISTRY[args.model_type]
    model_path = os.path.join(model_dir, "best_model.pth")
    model = load_model(model_path, device=device, num_classes=num_classes)

    test_df, class_names = _load_test_df(args.model_type)
    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "preprocessing")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Generating visual comparison of grayscale methods ...")
    plot_visual_comparison(test_df, class_names, out_dir)

    print("Running accuracy evaluation for each preprocessing method ...")
    run_accuracy_comparison(model, args.model_type, test_df, class_names,
                            num_classes, class_map, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
```

---

## Verification

```bash
python preprocessing_comparison.py --model-type 5class
```

Expected outputs in `models/5class/analysis/preprocessing/`:
- `method_visual_comparison.png` — 5×5 grid (classes × methods)
- `preprocessing_comparison_table.csv` — 5 rows (one per method)
- `preprocessing_accuracy_bars.png` — grouped bar chart
- `preprocessing_comparison_report.txt` — human-readable summary

The `lab_l` row in the table should match the accuracy from `python evaluate.py`
(same model, same test set, same preprocessing).
