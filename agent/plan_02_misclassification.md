# Plan: Module 2 — Misclassification Analysis

## Purpose

Generate a comprehensive breakdown of *which* images the model gets wrong, *why*
they might be harder, and whether errors cluster by experiment or slide type.

---

## File to Create: `misclassification_report.py`

---

## Prerequisites

- `analysis_utils.py` must exist (see `plan_shared_refactor.md`)
- A trained model must exist at `<model_dir>/best_model.pth`

---

## CLI

```
python misclassification_report.py [--model-type TYPE] [--output-dir DIR]

Arguments:
  --model-type  5class | 3class_scratch | 3class_finetune  (default: 5class)
  --output-dir  Override output directory  (default: <model_dir>/analysis/misclassification/)
```

---

## Imports

```python
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis_utils import (
    MODEL_REGISTRY, load_model_from_registry, run_inference_full,
    get_test_split, PHOTOS_DIR, DB_PATH
)
from evaluate import confusion_matrix as compute_confusion_matrix
from preprocessing import make_preprocessor
```

---

## Main Flow (`main()`)

```python
def main():
    args = parse_args()
    device = torch.device("cpu")

    model, model_dir, num_classes, class_map, class_names = \
        load_model_from_registry(args.model_type, device)
    test_df = get_test_split(args.model_type)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    print(f"Running inference on {len(test_df)} test images ...")
    results = run_inference_full(
        model, test_df, PHOTOS_DIR, device, preprocessor, class_map, class_names
    )

    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "misclassification")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(results, class_names, out_dir)
    plot_experiment_accuracy(results, out_dir)
    plot_slide_type_comparison(results, class_names, out_dir)
    plot_confidence_distribution(results, out_dir)
    save_csv(results, out_dir)
    write_summary(results, class_names, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
```

---

## Output A: `per_image_results.csv`

```python
def save_csv(results: pd.DataFrame, out_dir: str) -> None:
    path = os.path.join(out_dir, "per_image_results.csv")
    results.to_csv(path, index=False)
    print(f"  Saved: {path}")
```

Saves the full results DataFrame as-is. One row per test image.

---

## Output B: `experiment_accuracy.png`

```python
def plot_experiment_accuracy(results: pd.DataFrame, out_dir: str) -> None:
```

**Algorithm**:
1. Group `results` by `Experiment`, compute accuracy = correct.mean() for each group
2. Join with results to get the dominant class per experiment
   (most common `Exp_Type` value in that experiment's rows)
3. Sort experiments by accuracy (ascending — worst first)
4. Draw horizontal bar chart:
   - One bar per experiment, colored by dominant class using a fixed class→color palette
   - x-axis: accuracy (0.0 to 1.0), vertical dashed line at overall mean accuracy
   - y-axis: experiment ID strings
   - Legend: class colors
   - Title: "Per-Experiment Accuracy (sorted worst→best)"
5. For experiments with accuracy < 0.5, annotate bar with class name
6. `figsize=(10, max(6, 0.4 * num_experiments))`

**Color palette** (consistent across all modules):
```python
CLASS_COLORS = {
    "AC3":  "#4878CF",  # blue
    "F08D": "#D65F5F",  # red
    "F09D": "#B47CC7",  # purple
    "F11D": "#77BEDB",  # light blue
    "NC1":  "#6ACC65",  # green
}
```
Define this dict in `analysis_utils.py` and import it.

---

## Output C: `slide_type_comparison.png`

```python
def plot_slide_type_comparison(results: pd.DataFrame, class_names: list,
                               out_dir: str) -> None:
```

**Algorithm**:
1. For each class × slide_type (A, B), compute accuracy
2. Build a grouped bar chart:
   - x-axis groups: class names
   - Within each group: two bars — Slide A (solid) and Slide B (hatched)
   - y-axis: accuracy (0 to 1)
   - Error bars optional (standard deviation over images within group)
   - Title: "Accuracy by Class and Slide Type"
3. If a class has only one slide type in the test set, note this with an asterisk
4. `figsize=(max(8, 1.5 * len(class_names)), 5)`

---

## Output D: `confusion_matrix.png`

```python
def plot_confusion_matrix(results: pd.DataFrame, class_names: list,
                          out_dir: str) -> None:
```

**Algorithm**:
1. Call `compute_confusion_matrix(results["pred_label"].values, results["true_label"].values, len(class_names))`
   from `evaluate.py`. Note: evaluate.py signature is `confusion_matrix(preds, labels, num_classes)`.
2. Normalize by true class (row-normalize): `cm_norm = cm / cm.sum(axis=1, keepdims=True)`
3. Plot with `matplotlib.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)`
4. Annotate each cell with the raw count AND normalized value:
   `f"{cm[i,j]}\n({cm_norm[i,j]:.2f})"` — count on line 1, fraction on line 2
5. x-axis: predicted class labels, y-axis: true class labels
6. Title: "Confusion Matrix (normalized by true class)"
7. Colorbar labeled "Fraction of true class"
8. `figsize=(max(6, 1.2*len(class_names)), max(5, 1.2*len(class_names)))`

---

## Output E: `confidence_distribution.png`

```python
def plot_confidence_distribution(results: pd.DataFrame, out_dir: str) -> None:
```

**Algorithm**:
1. Split results into `correct_df` and `incorrect_df` on the `correct` column
2. Plot two overlapping histograms with 20 bins from 0.2 to 1.0:
   - Correct: green, alpha=0.6, label=f"Correct (n={len(correct_df)})"
   - Incorrect: red, alpha=0.6, label=f"Incorrect (n={len(incorrect_df)})"
3. Add vertical dashed lines for mean confidence of each group
4. x-axis: "Softmax Confidence (max probability)", y-axis: "Count"
5. Title: "Prediction Confidence Distribution"
6. `figsize=(8, 5)`

---

## Output F: `misclassification_summary.txt`

```python
def write_summary(results: pd.DataFrame, class_names: list, out_dir: str) -> None:
```

**Content**:

```
=== Misclassification Summary ===
Model:       <model_type>
Test images: <N>
Overall accuracy: <X.XXXX> (<correct>/<total>)

── Per-Class Accuracy ──
Class   Correct  Total  Accuracy
AC3     XX       41     X.XXX
F08D    XX       40     X.XXX
...

── Top Confused Class Pairs ──
True → Predicted   Count
F08D → F09D        XX
...

── Experiments with Accuracy < 50% ──
Experiment   Class   N    Accuracy
...
(None if all experiments >= 50%)

── Slide Type Comparison ──
Class   Slide A Acc  Slide B Acc  Diff
AC3     X.XXX        X.XXX        ±X.XXX
...

── Confidence Statistics ──
Correct predictions:   mean=X.XXX  std=X.XXX
Incorrect predictions: mean=X.XXX  std=X.XXX
```

**Implementation**:
- Use `results.groupby("Exp_Type")["correct"].agg(["sum", "count"])` for per-class stats
- For confused pairs: get the (true_label, pred_label) pairs for incorrect rows,
  map back to class names, count occurrences, sort descending, take top 5
- For slide type: `results.groupby(["Exp_Type", "Slide_Type"])["correct"].mean()`

---

## Verification

```bash
python misclassification_report.py --model-type 5class
```

Expected:
- `models/5class/analysis/misclassification/` created
- 6 files present: per_image_results.csv, experiment_accuracy.png,
  slide_type_comparison.png, confusion_matrix.png, confidence_distribution.png,
  misclassification_summary.txt
- Overall accuracy in summary should match what `python evaluate.py` reports
