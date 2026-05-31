"""
analysis_utils.py — Shared utilities for analysis modules 2, 3, 5, and 6.

Provides:
  - MODEL_REGISTRY       dict mapping model-type string → (model_dir, num_classes, class_map)
  - CLASS_COLORS         consistent color palette for plots
  - load_model_from_registry(model_type, device)  → (model, model_dir, num_classes, class_map, class_names)
  - run_inference_full(model, val_df, ...)         → pd.DataFrame with per-image predictions + metadata
  - get_val_split(model_type)                      → val_df
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from data_loader import (CLASS_MAP, CLASS_NAMES, filter_classes,
                         load_split_from_record)
from hemophilia_analysis import collect_predictions
from hemophilia_analysis import load_model as _load_model
from preprocessing import make_preprocessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH    = os.path.join(os.path.dirname(__file__), "data", "endpoint10.db")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
GRAY_METHOD = "lab_l"
POOL_FACTOR = 10

# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

CLASS_MAP_3:   Dict[str, int] = {"F08D": 0, "F09D": 1, "F11D": 2}
CLASS_NAMES_3: List[str]      = ["F08D", "F09D", "F11D"]
HEMOPHILIA_CLASSES: List[str] = ["F08D", "F09D", "F11D"]

# ---------------------------------------------------------------------------
# Model registry — mirrors hemophilia_analysis._MODEL_REGISTRY
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)
MODEL_REGISTRY: Dict[str, Tuple[str, int, dict]] = {
    "5class":               (os.path.join(_DIR, "models", "5class"),               5, CLASS_MAP),
    "5class_hpc_baseline":  (os.path.join(_DIR, "models", "5class_hpc_baseline"),  5, CLASS_MAP),
    "5class_hpc_v0":        (os.path.join(_DIR, "models", "5class_hpc_v0"),        5, CLASS_MAP),
    "5class_hpc_v1a":       (os.path.join(_DIR, "models", "5class_hpc_v1a"),       5, CLASS_MAP),
    "5class_hpc_v1b":       (os.path.join(_DIR, "models", "5class_hpc_v1b"),       5, CLASS_MAP),
    "5class_hpc_v1c":       (os.path.join(_DIR, "models", "5class_hpc_v1c"),       5, CLASS_MAP),
    "3class_scratch":       (os.path.join(_DIR, "models", "3class_hemo"),           3, CLASS_MAP_3),
    "3class_finetune":      (os.path.join(_DIR, "models", "3class_hemo_finetune"),  3, CLASS_MAP_3),
    "patch_v0":             (os.path.join(_DIR, "models", "patch_v0"),              5, CLASS_MAP),
    "patch_v1a":            (os.path.join(_DIR, "models", "patch_5class_v1a"),      5, CLASS_MAP),
    "patch_v1b":            (os.path.join(_DIR, "models", "patch_5class_v1b"),      5, CLASS_MAP),
    "patch_v1c":            (os.path.join(_DIR, "models", "patch_5class_v1c"),      5, CLASS_MAP),
}

# ---------------------------------------------------------------------------
# Consistent color palette (used across all analysis plots)
# ---------------------------------------------------------------------------

CLASS_COLORS: Dict[str, str] = {
    "AC3":  "#4878CF",
    "F08D": "#D65F5F",
    "F09D": "#B47CC7",
    "F11D": "#77BEDB",
    "NC1":  "#6ACC65",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_registry(
    model_type: str,
    device: torch.device,
) -> Tuple[object, str, int, dict, List[str]]:
    """Load a trained model by registry key.

    Args:
        model_type: One of "5class", "5class_hpc_baseline", "5class_hpc_v0",
                    "5class_hpc_v1a/b/c", "3class_scratch", "3class_finetune",
                    "patch_v0", "patch_v1a/b/c".
        device:     torch.device to load model onto.

    Returns:
        (model, model_dir, num_classes, class_map, class_names)
    """
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_type '{model_type}'. "
                       f"Valid: {list(MODEL_REGISTRY)}")
    model_dir, num_classes, class_map = MODEL_REGISTRY[model_type]
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. "
                                "Train the model first.")
    if "patch" in model_type:
        from model_patch import FibrinPatchCNN
        model = FibrinPatchCNN(num_classes=num_classes)
        sd = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.to(device).eval()
    elif "hpc" in model_type:
        from model import FibrinCNNCosine
        model = FibrinCNNCosine(num_classes=num_classes)
        sd = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.to(device).eval()
    else:
        model = _load_model(model_path, device=device, num_classes=num_classes)
    class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]
    return model, model_dir, num_classes, class_map, class_names


# ---------------------------------------------------------------------------
# Test split loading
# ---------------------------------------------------------------------------

def get_val_split(model_type: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load the validation DataFrame for a given model type.

    Always uses models/5class/train_record.json as the canonical split.
    For 3-class models, filters to the hemophilia classes.
    5-class and hpc models get the full 200-image val set.

    Returns:
        val_df with columns: idx, Experiment, Exp_Type, Slide_Type, ...
    """
    record_path = os.path.join(_DIR, "models", "5class", "train_record.json")
    _, val_df = load_split_from_record(record_path, db_path)

    _, num_classes, _ = MODEL_REGISTRY[model_type]
    if num_classes == 3:
        val_df = filter_classes(val_df, HEMOPHILIA_CLASSES)

    return val_df


# Keep legacy alias for backward compatibility
get_test_split = get_val_split


# ---------------------------------------------------------------------------
# Rich inference DataFrame
# ---------------------------------------------------------------------------

def run_inference_full(
    model,
    val_df: pd.DataFrame,
    photos_dir: str,
    device: torch.device,
    preprocessor,
    class_map: dict,
    class_names: List[str],
) -> pd.DataFrame:
    """Run inference on all validation images and return a rich per-image DataFrame.

    Args:
        model:        FibrinCNN in eval mode.
        val_df:       DataFrame from load_split_from_record (or filter_classes subset).
        photos_dir:   Path to directory containing 0000.JPG ... 0999.JPG.
        device:       torch.device.
        preprocessor: Callable img_bgr → tensor (from make_preprocessor).
        class_map:    Dict mapping class name string → int label.
        class_names:  List of class name strings ordered by int label.

    Returns:
        DataFrame with columns:
          idx, Experiment, Exp_Type, Slide_Type,
          true_label, pred_label, correct, confidence,
          prob_<class0>, prob_<class1>, ...
    """
    true_labels, probs, img_indices = collect_predictions(
        model, val_df, photos_dir, device, preprocessor, class_map=class_map
    )

    # Build base DataFrame
    df = pd.DataFrame({
        "idx":        img_indices.astype(np.int32),
        "true_label": true_labels.astype(np.int32),
        "pred_label": probs.argmax(axis=1).astype(np.int32),
        "confidence": probs.max(axis=1).astype(np.float32),
        **{f"prob_{name}": probs[:, i].astype(np.float32)
           for i, name in enumerate(class_names)},
    })

    # Join metadata from val_df (Experiment, Exp_Type, Slide_Type)
    meta = val_df[["idx", "Experiment", "Exp_Type", "Slide_Type"]].copy()
    meta["idx"] = meta["idx"].astype(np.int32)
    df = df.merge(meta, on="idx", how="left")

    # Correct flag
    df["correct"] = df["true_label"] == df["pred_label"]

    # Reorder columns
    front = ["idx", "Experiment", "Exp_Type", "Slide_Type",
             "true_label", "pred_label", "correct", "confidence"]
    prob_cols = [f"prob_{n}" for n in class_names]
    df = df[front + prob_cols]

    return df.reset_index(drop=True)
