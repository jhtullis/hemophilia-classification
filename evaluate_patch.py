"""
evaluate_patch.py — Inference via cosine soft-vote over a 35-patch grid.

Usage:
    python evaluate_patch.py --split val  --model-dir models/patch_v0
    python evaluate_patch.py --split test --model-dir models/patch_v0

The test split must only be used for final reported results, never during
training or model selection.

Outputs to models/patch_v0/analysis/{val,test}/:
    confusion_matrix.png
    roc_curves.png
    per_image_predictions.csv   (idx, true_label, pred_label, per-class mean similarity)
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import kornia.augmentation as K

from data_loader import CLASS_MAP, CLASS_NAMES, get_engine, load_metadata
from model_patch import FibrinPatchCNN
from patch_dataset import PAD, PATCH_SIZE, OVERSIZED, _pad_tensor, load_split_record
from preprocessing import make_preprocessor


# ---------------------------------------------------------------------------
# Inference grid
# ---------------------------------------------------------------------------

def inference_grid_centers(
    H: int = 400, W: int = 600, stride: int = 100
) -> List[Tuple[int, int]]:
    """Center-anchored grid from (0,0) to (H,W) inclusive at given stride.

    Returns list of (cy, cx) in original image coordinates.
    For H=400, W=600, stride=100: 5 rows × 7 cols = 35 centers.
    Edge centers are included — edge pixels are at the RF center of those patches,
    not its periphery, providing the best possible view of image boundaries.
    """
    centers = []
    for cy in range(0, H + 1, stride):
        for cx in range(0, W + 1, stride):
            centers.append((cy, cx))
    return centers


def predict_image(
    model: nn.Module,
    padded_tensor: torch.Tensor,
    centers: List[Tuple[int, int]],
    device: torch.device,
    center_crop: nn.Module,
) -> Tuple[int, torch.Tensor]:
    """Soft-vote cosine prediction over the inference grid.

    Returns (predicted_class_index, mean_cosine_similarities).
    mean_cosine_similarities: (num_classes,) tensor in [-1, 1].
    No temperature; no softmax; argmax gives the prediction.
    """
    all_sims: list[torch.Tensor] = []
    half = OVERSIZED // 2

    model.eval()
    with torch.no_grad():
        for cy, cx in centers:
            cy_pad = cy + PAD
            cx_pad = cx + PAD
            patch = padded_tensor[
                :, cy_pad - half : cy_pad + half + 1,
                   cx_pad - half : cx_pad + half + 1,
            ]
            patch = center_crop(patch.unsqueeze(0).to(device))
            sims = model(patch)   # (1, num_classes)
            all_sims.append(sims.squeeze(0).cpu())

    mean_sims = torch.stack(all_sims).mean(dim=0)   # (num_classes,)
    return mean_sims.argmax().item(), mean_sims


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load split
    train_df, val_df, test_df = load_split_record(args.model_dir, args.db)
    eval_df = val_df if args.split == "val" else test_df
    print(f"Evaluating on {args.split} set: {len(eval_df)} images")

    # Load model
    best_model_path = os.path.join(args.model_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best_model.pth in {args.model_dir}")

    model = FibrinPatchCNN(num_classes=len(CLASS_MAP)).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preprocessor = make_preprocessor()
    centers = inference_grid_centers()
    center_crop = K.CenterCrop(PATCH_SIZE)

    # Per-image predictions
    records = []
    for _, row in eval_df.iterrows():
        img_path = os.path.join(args.photo_dir, f"{int(row['idx']):04d}.JPG")
        tensor = preprocessor(img_path)          # (1, 400, 600)
        padded = _pad_tensor(tensor, PAD)        # (1, 684, 884)

        true_label = CLASS_MAP[row["Exp_Type"]]
        pred_label, mean_sims = predict_image(
            model, padded, centers, device, center_crop
        )

        rec = {
            "idx": int(row["idx"]),
            "Experiment": row["Experiment"],
            "Exp_Type": row["Exp_Type"],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(pred_label == true_label),
        }
        for i, cls in enumerate(CLASS_NAMES):
            rec[f"sim_{cls}"] = round(mean_sims[i].item(), 6)
        records.append(rec)

    results_df = pd.DataFrame(records)

    overall_acc = results_df["correct"].mean()
    print(f"\nOverall accuracy ({args.split}): {overall_acc:.3f}")
    for cls in CLASS_NAMES:
        cls_df = results_df[results_df["Exp_Type"] == cls]
        acc = cls_df["correct"].mean()
        print(f"  {cls}: {acc:.3f} ({len(cls_df)} images)")

    # Save outputs
    out_dir = os.path.join(args.model_dir, "analysis", args.split)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "per_image_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

    _plot_confusion_matrix(results_df, out_dir)
    _plot_roc_curves(results_df, out_dir)


def _plot_confusion_matrix(results_df: pd.DataFrame, out_dir: str) -> None:
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for _, row in results_df.iterrows():
        cm[row["true_label"], row["pred_label"]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — patch_v0")
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


def _plot_roc_curves(results_df: pd.DataFrame, out_dir: str) -> None:
    from itertools import cycle

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cycle(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])

    for cls, color in zip(CLASS_NAMES, colors):
        cls_idx = CLASS_MAP[cls]
        score_col = f"sim_{cls}"
        if score_col not in results_df.columns:
            continue

        y_true = (results_df["true_label"] == cls_idx).astype(int).values
        y_score = results_df[score_col].values

        # Sort by descending score
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        tp = np.cumsum(y_true_sorted)
        fp = np.cumsum(1 - y_true_sorted)
        tpr = tp / max(y_true.sum(), 1)
        fpr = fp / max((1 - y_true).sum(), 1)

        # AUC via trapezoidal integration (cosine scores: range [-1,1] is fine)
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        auc = abs(auc)   # sign depends on fpr ordering; take absolute value

        ax.plot(fpr, tpr, color=color, label=f"{cls} (AUC={auc:.3f})", lw=1.5)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves — patch_v0")
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
    p = argparse.ArgumentParser(description="Evaluate patch_v0 model.")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="Which split to evaluate (default: val).")
    p.add_argument("--model-dir", default="models/patch_v0",
                   help="Model directory (default: models/patch_v0).")
    p.add_argument("--db", default="data/endpoint10.db",
                   help="Path to SQLite database.")
    p.add_argument("--photo-dir", default="data/photos",
                   help="Directory containing JPEG images.")
    evaluate(p.parse_args())
