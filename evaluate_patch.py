"""
evaluate_patch.py — Image-level evaluation via soft-vote over a 35-patch grid.

For each image, patches are extracted on a fixed 5×7 grid (stride=100 px in
the 600×400 min-pooled space), model scores are averaged across all 35 patches,
and argmax gives the image-level prediction.  Works for both cosine-head and
CE-head patch models.

Usage:
    python evaluate_patch.py --model-type mpatch_v0_f  --split val
    python evaluate_patch.py --model-type patch_v1a    --split test
    python evaluate_patch.py --model-type mpatch_v0_e  --split val --db data/endpoint10.db

Outputs to models/<model_type>/analysis/{val,test}/:
    confusion_matrix.png
    roc_curves.png
    per_image_predictions.csv   (idx, Experiment, Exp_Type, true_label, pred_label,
                                  correct, score_<class>...)
"""

import argparse
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import kornia.augmentation as K

from analysis_utils import load_model_from_registry, MODEL_REGISTRY
from patch_dataset import PAD, PATCH_SIZE, OVERSIZED, _pad_tensor, load_split_record
from preprocessing import make_preprocessor

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Inference grid
# ---------------------------------------------------------------------------

def inference_grid_centers(
    H: int = 400, W: int = 600, stride: int = 100
) -> List[Tuple[int, int]]:
    """Center-anchored grid from (0,0) to (H,W) inclusive at given stride.

    For H=400, W=600, stride=100: 5 rows × 7 cols = 35 centers.
    Edge centers are included so that boundary pixels sit at the center of
    the receptive field rather than its periphery.
    """
    return [
        (cy, cx)
        for cy in range(0, H + 1, stride)
        for cx in range(0, W + 1, stride)
    ]


def score_patch_grid(
    model: nn.Module,
    padded_tensor: torch.Tensor,
    centers: List[Tuple[int, int]],
    device: torch.device,
    center_crop: nn.Module,
    pad: int = PAD,
    oversized: int = OVERSIZED,
) -> torch.Tensor:
    """Raw (unaveraged) per-patch model outputs over the inference grid.

    Returns:
        scores: (num_patches, num_classes) — one row of raw model outputs
        (cosine similarities or CE logits) per grid center in `centers`.
    """
    scores: list[torch.Tensor] = []
    half = oversized // 2

    model.eval()
    with torch.no_grad():
        for cy, cx in centers:
            cy_pad = cy + pad
            cx_pad = cx + pad
            patch = padded_tensor[
                :, cy_pad - half : cy_pad + half + 1,
                   cx_pad - half : cx_pad + half + 1,
            ]
            patch = center_crop(patch.unsqueeze(0).to(device))
            scores.append(model(patch).squeeze(0).cpu())

    return torch.stack(scores)   # (num_patches, num_classes)


def predict_image(
    model: nn.Module,
    padded_tensor: torch.Tensor,
    centers: List[Tuple[int, int]],
    device: torch.device,
    center_crop: nn.Module,
    pad: int = PAD,
    oversized: int = OVERSIZED,
) -> Tuple[int, torch.Tensor]:
    """Soft-vote prediction over the inference grid.

    Averages raw model outputs (cosine similarities or CE logits) across all
    grid patches and returns the argmax class.  Averaging is valid for both
    head types: cosine outputs share a common [-1, 1] scale; CE logits share
    the same linear scale within a model.

    Returns:
        (predicted_class_index, mean_scores)   — mean_scores: (num_classes,)
    """
    scores = score_patch_grid(model, padded_tensor, centers, device, center_crop,
                               pad=pad, oversized=oversized)
    mean_scores = scores.mean(dim=0)   # (num_classes,)
    return mean_scores.argmax().item(), mean_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model via registry (resolves head_type and model_dir from config)
    model, model_dir, num_classes, class_map, class_names = load_model_from_registry(
        args.model_type, device
    )

    # Load split from the model's own train_record_patch.json
    db_path = args.db or os.path.join(_DIR, "data", "endpoint10.db")
    train_df, val_df, test_df = load_split_record(model_dir, db_path)
    eval_df = val_df if args.split == "val" else test_df
    print(f"Evaluating {args.model_type} on {args.split} set: {len(eval_df)} images")

    preprocessor = make_preprocessor()
    centers = inference_grid_centers()
    center_crop = K.CenterCrop(PATCH_SIZE)
    photo_dir = args.photo_dir or os.path.join(_DIR, "data", "photos")

    # Per-image predictions
    records = []
    for _, row in eval_df.iterrows():
        img_path = os.path.join(photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        tensor = preprocessor(img_bgr)          # (1, 400, 600)
        padded = _pad_tensor(tensor, PAD)       # (1, H+2*PAD, W+2*PAD)

        true_label = class_map[row["Exp_Type"]]
        pred_label, mean_scores = predict_image(
            model, padded, centers, device, center_crop
        )

        rec = {
            "idx":        int(row["idx"]),
            "Experiment": row["Experiment"],
            "Exp_Type":   row["Exp_Type"],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct":    int(pred_label == true_label),
        }
        for i, cls in enumerate(class_names):
            rec[f"score_{cls}"] = round(mean_scores[i].item(), 6)
        records.append(rec)

    results_df = pd.DataFrame(records)

    overall_acc = results_df["correct"].mean()
    print(f"\nOverall accuracy ({args.split}): {overall_acc:.3f}")
    for cls in class_names:
        cls_df = results_df[results_df["Exp_Type"] == cls]
        if len(cls_df):
            print(f"  {cls}: {cls_df['correct'].mean():.3f}  ({len(cls_df)} images)")

    # Save outputs
    out_dir = os.path.join(model_dir, "analysis", args.split)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "per_image_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

    _plot_confusion_matrix(results_df, class_names, args.model_type, out_dir)
    _plot_roc_curves(results_df, class_names, class_map, args.model_type, out_dir)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    results_df: pd.DataFrame,
    class_names: List[str],
    model_type: str,
    out_dir: str,
) -> None:
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for _, row in results_df.iterrows():
        cm[row["true_label"], row["pred_label"]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_type}")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _plot_roc_curves(
    results_df: pd.DataFrame,
    class_names: List[str],
    class_map: dict,
    model_type: str,
    out_dir: str,
) -> None:
    from itertools import cycle
    colors = cycle(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color in zip(class_names, colors):
        score_col = f"score_{cls}"
        if score_col not in results_df.columns:
            continue
        cls_idx = class_map[cls]
        y_true  = (results_df["true_label"] == cls_idx).astype(int).values
        y_score = results_df[score_col].values

        order        = np.argsort(-y_score)
        y_true_s     = y_true[order]
        tp           = np.cumsum(y_true_s)
        fp           = np.cumsum(1 - y_true_s)
        tpr          = tp / max(y_true.sum(), 1)
        fpr          = fp / max((1 - y_true).sum(), 1)
        auc          = abs(float(np.trapezoid(tpr, fpr)
                                 if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)))
        ax.plot(fpr, tpr, color=color, label=f"{cls} (AUC={auc:.3f})", lw=1.5)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC Curves — {model_type}")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    patch_types = [k for k in MODEL_REGISTRY if "patch" in k]
    p = argparse.ArgumentParser(
        description="Image-level patch model evaluation via 35-patch soft vote."
    )
    p.add_argument(
        "--model-type", required=True, choices=patch_types,
        help="Model type to evaluate (must be a patch or mpatch model).",
    )
    p.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to evaluate (default: val).",
    )
    p.add_argument("--db", default=None, help="Path to SQLite database.")
    p.add_argument("--photo-dir", default=None, help="Directory of JPEG images.")
    evaluate(p.parse_args())
