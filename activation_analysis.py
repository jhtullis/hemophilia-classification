"""
activation_analysis.py — Spatial activation map analysis for FibrinCNN (Module 3).

Captures MaxPool activations for all test images and produces:
  1. Class-average channel-mean activation maps across all 4 conv blocks
  2. Comparison of correct vs incorrect spatial attention (Block 4)
  3. Spatial quadrant activation statistics CSV
  4. Activation overlaid on representative real images

Usage:
    python activation_analysis.py [--model-type TYPE] [--output-dir DIR] [--max-images N]

Outputs (saved to <model_dir>/analysis/activation/ by default):
    aggregate_activations.png       — class × block grid of average attention
    correct_vs_incorrect_attention.png — Block-4 attention for correct vs wrong predictions
    spatial_activation_stats.csv    — per-image quadrant activation values
    activation_overlay_examples.png — activation overlaid on example images
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis_utils import (DB_PATH, MODEL_REGISTRY, PHOTOS_DIR, get_val_split,
                             load_model_from_registry)
from preprocessing import make_preprocessor

# MaxPool layer indices in model.features
POOL_INDICES = [3, 7, 11, 15]
BLOCK_LABELS = ["Block 1\n(300×200)", "Block 2\n(150×100)",
                "Block 3\n(75×50)",   "Block 4\n(37×25)"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Activation map analysis for FibrinCNN.")
    p.add_argument("--model-type", default="5class",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Which trained model to analyse (default: 5class).")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Override output directory.")
    p.add_argument("--max-images", type=int, default=None, metavar="N",
                   help="Cap images processed (useful for quick testing).")
    p.add_argument("--db", default=DB_PATH, metavar="PATH",
                   help="Path to SQLite database (default: data/endpoint10.db).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-pass activation capture
# ---------------------------------------------------------------------------

def capture_activations_all(
    model,
    val_df: pd.DataFrame,
    photos_dir: str,
    device: torch.device,
    preprocessor,
    class_map: dict,
    class_names: List[str],
) -> Tuple[pd.DataFrame, Dict[int, Dict[int, np.ndarray]]]:
    """Run all validation images through the model, capturing activations and predictions.

    Returns:
        results_df: per-image DataFrame (idx, Exp_Type, Slide_Type, Experiment,
                    true_label, pred_label, correct, confidence)
        act_dict:   {img_idx: {pool_idx: (H, W) channel-mean activation}}
    """
    act_storage: Dict[int, torch.Tensor] = {}

    def make_hook(pidx: int):
        def hook(module, inp, out):
            act_storage[pidx] = out.detach().cpu()
        return hook

    hooks = [model.features[i].register_forward_hook(make_hook(i))
             for i in POOL_INDICES]

    act_dict: Dict[int, Dict[int, np.ndarray]] = {}
    rows = []

    model.eval()
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img_path = os.path.join(photos_dir, f"{int(row['idx']):04d}.JPG")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            tensor = preprocessor(img_bgr).unsqueeze(0).to(device)

            logits = model(tensor)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = int(prob.argmax())
            true_label = class_map[row["Exp_Type"]]

            idx = int(row["idx"])
            act_dict[idx] = {
                pidx: act_storage[pidx][0].mean(dim=0).numpy()
                for pidx in POOL_INDICES
            }

            rows.append({
                "idx":        idx,
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
    return results_df, act_dict


# ---------------------------------------------------------------------------
# Output A: aggregate_activations.png
# ---------------------------------------------------------------------------

def plot_aggregate_activations(results_df: pd.DataFrame,
                                act_dict: Dict[int, Dict[int, np.ndarray]],
                                class_names: List[str],
                                out_dir: str) -> None:
    n_cls = len(class_names)
    n_blk = len(POOL_INDICES)
    fig, axes = plt.subplots(n_cls, n_blk,
                             figsize=(4 * n_blk, 3 * n_cls),
                             squeeze=False)

    for row_i, cls in enumerate(class_names):
        cls_indices = results_df[results_df["Exp_Type"] == cls]["idx"].tolist()
        for col_i, pidx in enumerate(POOL_INDICES):
            maps = [act_dict[idx][pidx] for idx in cls_indices if idx in act_dict]
            ax = axes[row_i, col_i]
            if maps:
                avg = np.mean(maps, axis=0)
                ax.imshow(avg, cmap="viridis", interpolation="bilinear",
                          aspect="auto")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(BLOCK_LABELS[col_i], fontsize=10)
            if col_i == 0:
                ax.set_ylabel(cls, fontsize=12, rotation=0, labelpad=42,
                              va="center")

    fig.suptitle("Class-Average Channel-Mean Activation Maps\n"
                 "(each cell = mean over all test images of that class)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "aggregate_activations.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output B: correct_vs_incorrect_attention.png
# ---------------------------------------------------------------------------

def plot_correct_vs_incorrect(results_df: pd.DataFrame,
                               act_dict: Dict[int, Dict[int, np.ndarray]],
                               class_names: List[str],
                               out_dir: str,
                               min_incorrect: int = 3) -> None:
    qualifying = []
    for cls in class_names:
        cls_rows = results_df[results_df["Exp_Type"] == cls]
        correct_idx   = cls_rows[cls_rows["correct"]]["idx"].tolist()
        incorrect_idx = cls_rows[~cls_rows["correct"]]["idx"].tolist()
        if len(incorrect_idx) >= min_incorrect:
            qualifying.append((cls, correct_idx, incorrect_idx))

    if not qualifying:
        print(f"  No classes with >= {min_incorrect} incorrect predictions; "
              "skipping correct_vs_incorrect_attention.png")
        return

    n_rows = len(qualifying)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3.5 * n_rows), squeeze=False)

    pidx = POOL_INDICES[-1]   # Block 4

    for row_i, (cls, correct_idx, incorrect_idx) in enumerate(qualifying):
        c_maps = [act_dict[idx][pidx] for idx in correct_idx if idx in act_dict]
        w_maps = [act_dict[idx][pidx] for idx in incorrect_idx if idx in act_dict]

        avg_c = np.mean(c_maps, axis=0) if c_maps else None
        avg_w = np.mean(w_maps, axis=0) if w_maps else None

        vmax = max(
            (avg_c.max() if avg_c is not None else 0),
            (avg_w.max() if avg_w is not None else 0),
        ) or 1.0

        for col_i, (avg, label) in enumerate([(avg_c, "Correct"), (avg_w, "Incorrect")]):
            ax = axes[row_i, col_i]
            if avg is not None:
                ax.imshow(avg, cmap="viridis", vmin=0, vmax=vmax,
                          interpolation="bilinear", aspect="auto")
            ax.axis("off")
            n = len(c_maps) if label == "Correct" else len(w_maps)
            if row_i == 0:
                ax.set_title(f"{label} predictions", fontsize=11)
            ax.set_ylabel(f"{cls}\n(n={n})", fontsize=10, rotation=0,
                          labelpad=50, va="center")

    fig.suptitle("Block-4 Spatial Attention: Correct vs Incorrect Predictions",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "correct_vs_incorrect_attention.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output C: spatial_activation_stats.csv
# ---------------------------------------------------------------------------

def save_spatial_stats(results_df: pd.DataFrame,
                       act_dict: Dict[int, Dict[int, np.ndarray]],
                       out_dir: str) -> None:
    pidx = POOL_INDICES[-1]   # Block 4 output: (37, 25)
    rows = []
    for _, row in results_df.iterrows():
        idx = int(row["idx"])
        if idx not in act_dict:
            continue
        a = act_dict[idx][pidx]  # (37, 25)
        h, w = a.shape
        hh, hw = h // 2, w // 2
        rows.append({
            "idx":             idx,
            "Exp_Type":        row["Exp_Type"],
            "Slide_Type":      row["Slide_Type"],
            "correct":         bool(row["correct"]),
            "confidence":      float(row["confidence"]),
            "act_top_left":    float(a[:hh, :hw].mean()),
            "act_top_right":   float(a[:hh, hw:].mean()),
            "act_bottom_left": float(a[hh:, :hw].mean()),
            "act_bottom_right":float(a[hh:, hw:].mean()),
            "act_total_mean":  float(a.mean()),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "spatial_activation_stats.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output D: activation_overlay_examples.png
# ---------------------------------------------------------------------------

def plot_activation_overlays(results_df: pd.DataFrame,
                              act_dict: Dict[int, Dict[int, np.ndarray]],
                              class_names: List[str],
                              photos_dir: str,
                              preprocessor,
                              out_dir: str) -> None:
    pidx = POOL_INDICES[-1]   # Block 4

    # Pick highest-confidence correct image per class
    rep: Dict[str, int] = {}
    for cls in class_names:
        sub = results_df[(results_df["Exp_Type"] == cls) & results_df["correct"]]
        if sub.empty:
            sub = results_df[results_df["Exp_Type"] == cls]
        if sub.empty:
            continue
        rep[cls] = int(sub.sort_values("confidence", ascending=False).iloc[0]["idx"])

    if not rep:
        print("  No representative images found; skipping overlay plot.")
        return

    n_cls = len(rep)
    fig, axes = plt.subplots(n_cls, 2, figsize=(12, 4 * n_cls), squeeze=False)

    for row_i, cls in enumerate(cn for cn in class_names if cn in rep):
        idx = rep[cls]
        img_bgr = cv2.imread(os.path.join(photos_dir, f"{idx:04d}.JPG"))
        if img_bgr is None:
            continue
        tensor = preprocessor(img_bgr)                          # (1, H, W)
        gray = (tensor.squeeze().numpy() * 255).astype(np.uint8)  # (400, 600)

        a = act_dict.get(idx, {}).get(pidx)
        if a is None:
            continue

        # Upsample activation: cv2.resize takes (width, height)
        a_up = cv2.resize(a, (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min() + 1e-8)
        heatmap_uint8 = (a_up * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        gray_bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay   = cv2.addWeighted(gray_bgr, 0.5, colormap, 0.5, 0)

        # matplotlib expects RGB
        ax_gray = axes[row_i, 0]
        ax_over = axes[row_i, 1]
        ax_gray.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax_over.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax_gray.axis("off")
        ax_over.axis("off")

        if row_i == 0:
            ax_gray.set_title("Preprocessed (grayscale)", fontsize=11)
            ax_over.set_title("Block-4 Activation Overlay", fontsize=11)
        ax_gray.set_ylabel(f"{cls}\n(idx={idx:04d})", fontsize=11,
                           rotation=0, labelpad=60, va="center")

    fig.suptitle("Activation Maps Overlaid on Preprocessed Images\n"
                 "(highest-confidence correct test image per class)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "activation_overlay_examples.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")

    print(f"Loading {args.model_type} model ...")
    model, model_dir, _, class_map, class_names = \
        load_model_from_registry(args.model_type, device)
    val_df = get_val_split(args.model_type, db_path=args.db)
    if args.max_images:
        val_df = val_df.head(args.max_images)
        print(f"  (capped at {args.max_images} images)")

    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)
    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "activation")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Capturing activations for {len(val_df)} images ...")
    results_df, act_dict = capture_activations_all(
        model, val_df, PHOTOS_DIR, device, preprocessor, class_map, class_names
    )
    acc = results_df["correct"].mean()
    print(f"  Accuracy on this subset: {acc:.4f}")
    print(f"\nSaving outputs to: {out_dir}/")

    print("Figure 1: aggregate activation maps ...")
    plot_aggregate_activations(results_df, act_dict, class_names, out_dir)

    print("Figure 2: correct vs incorrect attention ...")
    plot_correct_vs_incorrect(results_df, act_dict, class_names, out_dir)

    print("Figure 3: spatial activation stats CSV ...")
    save_spatial_stats(results_df, act_dict, out_dir)

    print("Figure 4: activation overlay examples ...")
    plot_activation_overlays(results_df, act_dict, class_names,
                             PHOTOS_DIR, preprocessor, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
