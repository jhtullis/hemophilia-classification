# Plan: Shared Utility — analysis_utils.py

## Purpose

Four analysis modules (Modules 2, 3, 5, 6) all need to:
1. Load a trained model from the registry
2. Run inference on the test set
3. Work with per-image results alongside metadata (Experiment, Slide_Type, Exp_Type)

Rather than duplicating this logic, create `analysis_utils.py` as a single shared
utility. This is a **prerequisite** for Modules 2 and 3.

---

## File to Create: `analysis_utils.py`

Location: project root (same directory as all other `.py` files).

---

## Dependencies (imports needed)

```python
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data_loader import CLASS_MAP, CLASS_NAMES, FibrinDataset, load_split_from_record, filter_classes
from hemophilia_analysis import collect_predictions, load_model as _load_model
from preprocessing import make_preprocessor
```

Note: `hemophilia_analysis._MODEL_REGISTRY` is imported here so downstream scripts
only need to import from `analysis_utils` — they don't need to know about
`hemophilia_analysis` directly.

---

## Constants

Mirror the model registry from `hemophilia_analysis.py`:

```python
DB_PATH    = os.path.join(os.path.dirname(__file__), "data", "test_db.db")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
GRAY_METHOD = "lab_l"
POOL_FACTOR = 10

CLASS_MAP_3 = {"F08D": 0, "F09D": 1, "F11D": 2}
CLASS_NAMES_3 = ["F08D", "F09D", "F11D"]
HEMOPHILIA_CLASSES = ["F08D", "F09D", "F11D"]

MODEL_REGISTRY: Dict[str, Tuple[str, int, dict]] = {
    "5class":          ("models/5class",               5, CLASS_MAP),
    "3class_scratch":  ("models/3class_hemo",           3, CLASS_MAP_3),
    "3class_finetune": ("models/3class_hemo_finetune",  3, CLASS_MAP_3),
}
```

---

## Function 1: `load_model_from_registry`

```python
def load_model_from_registry(
    model_type: str,
    device: torch.device,
) -> Tuple[object, str, int, dict, List[str]]:
    """Load a trained model by registry key.

    Args:
        model_type: One of "5class", "3class_scratch", "3class_finetune".
        device:     torch.device to load model onto.

    Returns:
        (model, model_dir, num_classes, class_map, class_names)
        model       — FibrinCNN instance in eval mode
        model_dir   — path to directory containing best_model.pth
        num_classes — 5 or 3
        class_map   — dict mapping class name → int label
        class_names — list of class name strings ordered by label
    """
```

**Implementation**:
```python
model_dir, num_classes, class_map = MODEL_REGISTRY[model_type]
model_path = os.path.join(model_dir, "best_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No trained model at {model_path}")
model = _load_model(model_path, device=device, num_classes=num_classes)
class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]
return model, model_dir, num_classes, class_map, class_names
```

---

## Function 2: `run_inference_full`

```python
def run_inference_full(
    model,
    test_df: pd.DataFrame,
    photos_dir: str,
    device: torch.device,
    preprocessor,
    class_map: dict,
    class_names: List[str],
) -> pd.DataFrame:
    """Run inference and return a rich per-image results DataFrame.

    Args:
        model:        FibrinCNN in eval mode.
        test_df:      DataFrame from load_split_from_record (or filter_classes subset).
        photos_dir:   Path to directory with 0000.JPG ... 0858.JPG.
        device:       torch.device.
        preprocessor: Callable (from make_preprocessor) img_bgr → tensor.
        class_map:    Dict mapping class name string → int label.
        class_names:  List of class name strings, ordered by int label (index = label).

    Returns:
        DataFrame with columns:
          idx         int   — 0-based image index (filename = f"{idx:04d}.JPG")
          Experiment  str   — experiment ID from DB
          Exp_Type    str   — true class label string
          Slide_Type  str   — A or B
          true_label  int   — integer label from class_map
          pred_label  int   — argmax of softmax probabilities
          correct     bool  — true_label == pred_label
          confidence  float — max softmax probability
          prob_<name> float — one column per class (e.g. prob_AC3, prob_F08D, ...)
    """
```

**Implementation**:
```python
true_labels, probs, img_indices = collect_predictions(
    model, test_df, photos_dir, device, preprocessor, class_map=class_map
)

df = pd.DataFrame({
    "idx":        img_indices,
    "true_label": true_labels,
    "pred_label": probs.argmax(axis=1).astype(np.int32),
    "confidence": probs.max(axis=1),
    **{f"prob_{name}": probs[:, i] for i, name in enumerate(class_names)},
})

# Add metadata columns from test_df (join on idx)
meta = test_df[["idx", "Experiment", "Exp_Type", "Slide_Type"]].copy()
df = df.merge(meta, on="idx", how="left")

# Add correct flag
df["correct"] = df["true_label"] == df["pred_label"]

# Reorder columns for readability
front = ["idx", "Experiment", "Exp_Type", "Slide_Type",
         "true_label", "pred_label", "correct", "confidence"]
prob_cols = [f"prob_{n}" for n in class_names]
df = df[front + prob_cols]

return df.reset_index(drop=True)
```

---

## Function 3: `get_test_split`

A convenience function used by all modules to load the test DataFrame:

```python
def get_test_split(
    model_type: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load the test DataFrame for a given model type.

    For 3-class models, filters to hemophilia classes only.
    Always uses models/5class/train_record.json as the canonical split source.

    Returns:
        test_df — DataFrame with idx, Experiment, Exp_Type, Slide_Type columns.
    """
    record_path = os.path.join("models", "5class", "train_record.json")
    _, test_df = load_split_from_record(record_path, db_path)

    _, num_classes, _ = MODEL_REGISTRY[model_type]
    if num_classes == 3:
        test_df = filter_classes(test_df, HEMOPHILIA_CLASSES)

    return test_df
```

---

## Verification

After implementing, run:
```bash
python -c "
from analysis_utils import load_model_from_registry, run_inference_full, get_test_split
import torch
from preprocessing import make_preprocessor

device = torch.device('cpu')
model, model_dir, num_classes, class_map, class_names = load_model_from_registry('5class', device)
test_df = get_test_split('5class')
preprocessor = make_preprocessor(gray_method='lab_l', pool_factor=10)
results = run_inference_full(model, test_df, 'data/photos', device, preprocessor, class_map, class_names)
print(results.shape)         # should be (201, 12)
print(results.columns.tolist())
print(results['correct'].mean())  # overall accuracy
"
```

Expected output: 201 rows, columns including idx/Experiment/Exp_Type/Slide_Type/
true_label/pred_label/correct/confidence/prob_AC3/prob_F08D/prob_F09D/prob_F11D/prob_NC1.
