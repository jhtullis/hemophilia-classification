"""
holdout_eval_utils.py — Shared helpers for the holdout-testing pipelines
(evaluate_holdout_125s_ensemble.py, evaluate_holdout_lite_lc.py).

Public API:
    load_125s_model(model_dir, device, head_type, num_classes) -> FibrinPatchCNN125s
    get_best_epoch_metadata(model_dir) -> dict
    copy_best_checkpoint(model_dir, dest_weights_dir, name) -> str
    assert_identical_test_split(model_dirs) -> dict
    extract_split_record_subset(model_dir) -> dict
    accuracy_breakdown(results_df, class_names) -> (float, dict)
    write_json(path, obj) -> None
"""

import csv
import json
import os
import shutil
from typing import Dict, List, Tuple

import pandas as pd
import torch


def load_125s_model(
    model_dir: str,
    device: torch.device,
    head_type: str = "ce",
    num_classes: int = 5,
):
    """Load a FibrinPatchCNN125s from model_dir/best_model.pth.

    Mirrors analysis_utils.load_model_from_registry's state-dict loading
    pattern (bare state dict, strip any torch.compile "_orig_mod." prefix).
    Not registered in analysis_utils.MODEL_REGISTRY -- see plan rationale:
    the 125s geometry/ensembling semantics don't fit that registry's
    one-model-one-dir abstraction and would leak into unrelated eval paths.
    """
    from model_patch_125s import FibrinPatchCNN125s

    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. "
                                "Train the model first.")

    model = FibrinPatchCNN125s(num_classes=num_classes, head_type=head_type)
    sd = torch.load(model_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def get_best_epoch_metadata(model_dir: str) -> dict:
    """Best epoch/val_acc for model_dir's best_model.pth.

    Prefers training_log_full.csv's max-val_acc row (present for every patch
    model, cosine or ce head). Falls back to checkpoints/topk_manifest.json's
    top all-time entry if the CSV is unavailable.
    """
    csv_path = os.path.join(model_dir, "training_log_full.csv")
    if os.path.exists(csv_path):
        best_epoch, best_val_acc = None, float("-inf")
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                val_acc = float(row["val_acc"])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = int(row["epoch"])
        if best_epoch is not None:
            return {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "source": "training_log_full.csv",
            }

    manifest_path = os.path.join(model_dir, "checkpoints", "topk_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        alltime = manifest.get("alltime", [])
        if alltime:
            best = max(alltime, key=lambda e: e["val_acc"])
            return {
                "best_epoch": best["epoch"],
                "best_val_acc": best["val_acc"],
                "source": "topk_manifest.json",
            }

    raise FileNotFoundError(
        f"No training_log_full.csv or checkpoints/topk_manifest.json found "
        f"under {model_dir} — cannot determine best epoch/val_acc."
    )


def copy_best_checkpoint(model_dir: str, dest_weights_dir: str, name: str) -> str:
    """Copy model_dir/best_model.pth to dest_weights_dir/<name>_best_model.pth."""
    src = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(src):
        raise FileNotFoundError(f"No trained model at {src}. Train the model first.")
    os.makedirs(dest_weights_dir, exist_ok=True)
    dst = os.path.join(dest_weights_dir, f"{name}_best_model.pth")
    shutil.copy2(src, dst)
    return dst


def _load_split_record(model_dir: str) -> dict:
    record_path = os.path.join(model_dir, "train_record_patch.json")
    with open(record_path) as f:
        return json.load(f)


def assert_identical_test_split(model_dirs: List[str]) -> dict:
    """Assert all model_dirs share an identical test split (test_indices).

    Reads train_record_patch.json directly (no DB access). Raises
    AssertionError with a clear diff if any dir's test_indices disagree —
    this is the automated check on the shared-holdout reproducibility
    guarantee for a CV ensemble.
    """
    if not model_dirs:
        raise ValueError("model_dirs must be non-empty")

    records = {d: _load_split_record(d) for d in model_dirs}
    ref_dir = model_dirs[0]
    ref_indices = set(records[ref_dir]["test_indices"])

    mismatches = {}
    for d in model_dirs[1:]:
        indices = set(records[d]["test_indices"])
        if indices != ref_indices:
            mismatches[d] = sorted(indices.symmetric_difference(ref_indices))

    if mismatches:
        raise AssertionError(
            f"test_indices mismatch relative to {ref_dir}: {mismatches}"
        )

    return {
        "test_indices": sorted(ref_indices),
        "test_experiments": records[ref_dir]["test_experiments"],
    }


def extract_split_record_subset(model_dir: str) -> dict:
    """Test-split portion of model_dir's train_record_patch.json, for
    reproducing in an evaluation output directory's split_record.json."""
    record = _load_split_record(model_dir)
    return {
        "source": os.path.join(model_dir, "train_record_patch.json"),
        "test_indices": record["test_indices"],
        "test_experiments": record["test_experiments"],
        "class_distribution_test": record.get("class_distribution", {}).get("test"),
    }


def accuracy_breakdown(
    results_df: pd.DataFrame, class_names: List[str]
) -> Tuple[float, Dict[str, float]]:
    """(overall_accuracy, {class_name: accuracy}) from a results_df with
    Exp_Type/correct columns."""
    overall = float(results_df["correct"].mean())
    per_class = {}
    for cls in class_names:
        cls_df = results_df[results_df["Exp_Type"] == cls]
        if len(cls_df):
            per_class[cls] = float(cls_df["correct"].mean())
    return overall, per_class


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
