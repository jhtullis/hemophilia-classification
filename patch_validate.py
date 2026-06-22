"""
patch_validate.py — Patch-model validation visualizer.

For each patch model and each data split (val / train), evaluates all images with
a deterministic grid of patches, produces annotated images with transparent green/
red overlays showing per-patch correctness, and outputs CSV tables + summary plots
with accuracy broken down by class, experiment, and slide.

Usage:
    python patch_validate.py [--models patch_v0 patch_v1a ...] [--splits val train]
    python patch_validate.py --help

Output: patch_validation/{model_type}/{split}/
"""

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis_utils import CLASS_COLORS, MODEL_REGISTRY
from data_loader import CLASS_MAP, CLASS_NAMES
from patch_dataset import PAD, OVERSIZED, PATCH_SIZE, _pad_tensor, load_split_record
from preprocessing import make_preprocessor

# ── constants ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
PATCH_ALPHA = 0.25                                   # fill alpha per patch rectangle
CROP_START = (OVERSIZED - PATCH_SIZE) // 2          # 41  (283-200)//2
ALL_PATCH_MODELS = [k for k in MODEL_REGISTRY if k.startswith("patch_")]

# Slide A/B get distinct colors (used across all slide-comparison figures)
SLIDE_COLORS: Dict[str, str] = {"A": "#4878CF", "B": "#E89B2D"}

_CORRECT_COLOR = "#2ca02c"
_WRONG_COLOR   = "#d62728"

# ── patch grid ────────────────────────────────────────────────────────────────

def make_grid(n_cols: int = 5, n_rows: int = 4) -> List[Tuple[int, int]]:
    """Return (cy, cx) patch centers covering the 600×400 preprocessed image.

    Default 5×4 grid places centers at:
      x ∈ {100, 200, 300, 400, 500}  (width dimension, step=100)
      y ∈ {100, 167, 233, 300}       (height dimension, step≈67)

    Adjacent patches overlap by ~100 px; all centers are ≥100 px from the
    image edge so the 283×283 oversized crop never reads outside the padded tensor.
    """
    xs = np.linspace(100, 500, n_cols).astype(int)
    ys = np.linspace(100, 300, n_rows).astype(int)
    return [(int(cy), int(cx)) for cy in ys for cx in xs]


# ── model loading ─────────────────────────────────────────────────────────────

def load_patch_model(model_type: str) -> Optional[torch.nn.Module]:
    """Load a patch model from best_model.pth; return None if file is missing."""
    model_dir, num_classes, _ = MODEL_REGISTRY[model_type]
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        return None
    from model_patch import FibrinPatchCNN
    head_type = "ce" if "v2" in model_type else "cosine"
    model = FibrinPatchCNN(num_classes=num_classes, head_type=head_type)
    sd = torch.load(model_path, map_location=DEVICE, weights_only=True)
    # Strip _orig_mod. prefix produced by torch.compile if present
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────

def infer_image(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    preprocessor,
    grid: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Preprocess one image, extract all grid patches, run a single batched forward pass.

    Returns:
        preds  (N,)    int32  predicted label per patch
        probs  (N, C)  float32  softmax probabilities per patch
        tensor (1,400,600)  preprocessed image (for visualization)
    """
    tensor = preprocessor(img_bgr)         # (1, 400, 600) float32
    padded = _pad_tensor(tensor, PAD)      # (1, 684, 884)
    half = OVERSIZED // 2                  # 141

    patches = []
    for cy, cx in grid:
        cy_pad, cx_pad = cy + PAD, cx + PAD
        oversized = padded[
            :,
            cy_pad - half : cy_pad + half + 1,
            cx_pad - half : cx_pad + half + 1,
        ]  # (1, 283, 283)
        patch = oversized[
            :,
            CROP_START : CROP_START + PATCH_SIZE,
            CROP_START : CROP_START + PATCH_SIZE,
        ]  # (1, 200, 200) — same deterministic center-crop used at val time
        patches.append(patch)

    batch = torch.stack(patches)           # (N, 1, 200, 200)
    with torch.no_grad():
        logits = model(batch)              # (N, C)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return probs.argmax(axis=1).astype(np.int32), probs, tensor


# ── annotated image ───────────────────────────────────────────────────────────

def draw_annotated_image(
    tensor: torch.Tensor,
    grid: List[Tuple[int, int]],
    preds: np.ndarray,
    true_label: int,
    row: pd.Series,
    voted_class: str,
    is_train: bool,
) -> plt.Figure:
    """Return a matplotlib figure: preprocessed grayscale image + patch overlays.

    Green rectangles = patch predicted correctly.
    Red rectangles   = patch predicted incorrectly.
    Overlapping patches accumulate alpha, showing spatial confidence density.
    """
    img_np = tensor.squeeze(0).numpy()     # (400, 600) float32 in [0, 1]
    n_correct = int((preds == true_label).sum())
    n_total = len(preds)
    true_class = CLASS_NAMES[true_label]
    voted_ok = (voted_class == true_class)

    fig, ax = plt.subplots(figsize=(12, 8.4))
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    # Data coordinates match pixel indices: x=cx (columns), y=cy (rows)

    for i, (cy, cx) in enumerate(grid):
        color = _CORRECT_COLOR if preds[i] == true_label else _WRONG_COLOR
        rect = mpatches.Rectangle(
            (cx - PATCH_SIZE // 2, cy - PATCH_SIZE // 2),
            PATCH_SIZE, PATCH_SIZE,
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=PATCH_ALPHA,
        )
        ax.add_patch(rect)

    title = (
        f"{int(row['idx']):04d}  |  true: {true_class}  |  "
        f"{row['Experiment']}  Slide {row['Slide_Type']}  |  "
        f"{n_correct}/{n_total} patches correct  |  voted: {voted_class}"
    )
    ax.set_title(title, fontsize=9, color=("black" if voted_ok else "red"), pad=5)
    ax.axis("off")

    if is_train:
        ax.text(
            0.01, 0.99, "TRAIN SET",
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            color="red", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="red", pad=3),
        )

    fig.tight_layout(pad=0.2)
    return fig


# ── confusion matrix ──────────────────────────────────────────────────────────

def _confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels.tolist(), pred_labels.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> plt.Figure:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(n * 1.4 + 1, n * 1.4 + 0.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    threshold = cm.max() * 0.6
    row_totals = cm.sum(axis=1, keepdims=True).clip(1)
    for i in range(n):
        for j in range(n):
            pct = cm[i, j] / row_totals[i, 0] * 100
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=7.5, color=color)
    fig.tight_layout()
    return fig


# ── summary figures + stats text ─────────────────────────────────────────────

def save_summary(
    patch_df: pd.DataFrame,
    image_df: pd.DataFrame,
    summary_dir: str,
    model_type: str,
    split: str,
) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    n_cls = len(CLASS_NAMES)
    cls_colors = [CLASS_COLORS.get(c, "#888888") for c in CLASS_NAMES]

    # ── 1. Patch accuracy by class ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    accs = [patch_df[patch_df["Exp_Type"] == c]["correct"].mean() * 100
            for c in CLASS_NAMES]
    overall_patch = patch_df["correct"].mean() * 100
    bars = ax.bar(CLASS_NAMES, accs, color=cls_colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.5, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=8.5)
    ax.axhline(overall_patch, color="gray", linestyle="--", linewidth=1,
               label=f"Overall {overall_patch:.1f}%")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Patch accuracy (%)", fontsize=10)
    ax.set_title(f"{model_type}  ·  {split}  ·  Patch accuracy by class", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "patch_accuracy_by_class.png"), dpi=100)
    plt.close(fig)

    # ── 2. Voting accuracy by class (hard vs soft) ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_cls)
    w = 0.38
    hard_accs = [image_df[image_df["Exp_Type"] == c]["hard_vote_correct"].mean() * 100
                 for c in CLASS_NAMES]
    soft_accs = [image_df[image_df["Exp_Type"] == c]["soft_vote_correct"].mean() * 100
                 for c in CLASS_NAMES]
    b1 = ax.bar(x - w / 2, hard_accs, w, color=cls_colors, edgecolor="white",
                linewidth=0.8, label="Hard vote (majority)")
    b2 = ax.bar(x + w / 2, soft_accs, w, color=cls_colors, edgecolor="black",
                alpha=0.45, linewidth=1.2, label="Soft vote (mean prob)")
    for bars, acc_list in [(b1, hard_accs), (b2, soft_accs)]:
        for bar, acc in zip(bars, acc_list):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.5, f"{acc:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Image accuracy (%)", fontsize=10)
    ax.set_title(f"{model_type}  ·  {split}  ·  Voting accuracy by class", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "voting_accuracy_by_class.png"), dpi=100)
    plt.close(fig)

    # ── 3. Voting accuracy by experiment ──────────────────────────────────────
    exp_data = (
        image_df.groupby(["Experiment", "Exp_Type"])
        .agg(n_correct=("hard_vote_correct", "sum"),
             n_total=("hard_vote_correct", "count"))
        .reset_index()
    )
    exp_data["acc"] = exp_data["n_correct"] / exp_data["n_total"] * 100
    exp_data = exp_data.sort_values("acc").reset_index(drop=True)
    exp_colors = [CLASS_COLORS.get(c, "#888888") for c in exp_data["Exp_Type"]]
    fig, ax = plt.subplots(figsize=(max(8, len(exp_data) * 0.7 + 1.5), 5))
    bars = ax.bar(range(len(exp_data)), exp_data["acc"],
                  color=exp_colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, exp_data["acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.5, f"{acc:.0f}%",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(exp_data)))
    ax.set_xticklabels(
        [f"{r['Experiment']}\n({r['Exp_Type']})" for _, r in exp_data.iterrows()],
        rotation=45, ha="right", fontsize=7.5,
    )
    ax.set_ylim(0, 120)
    ax.set_ylabel("Hard-vote image accuracy (%)", fontsize=10)
    ax.set_title(f"{model_type}  ·  {split}  ·  Voting accuracy by experiment", fontsize=10)
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_NAMES]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "voting_accuracy_by_experiment.png"), dpi=100)
    plt.close(fig)

    # ── 4. Voting accuracy by class and slide (A vs B) ───────────────────────
    slides = sorted(image_df["Slide_Type"].unique())
    slide_acc = (
        image_df.groupby(["Exp_Type", "Slide_Type"])["hard_vote_correct"]
        .mean().reset_index()
    )
    slide_acc["acc"] = slide_acc["hard_vote_correct"] * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_cls)
    w = 0.75 / max(len(slides), 1)
    for si, slide in enumerate(slides):
        s_accs = []
        for cls in CLASS_NAMES:
            mask = (slide_acc["Exp_Type"] == cls) & (slide_acc["Slide_Type"] == slide)
            vals = slide_acc.loc[mask, "acc"].values
            s_accs.append(float(vals[0]) if len(vals) else 0.0)
        offset = (si - (len(slides) - 1) / 2) * w
        ax.bar(x + offset, s_accs, w, label=f"Slide {slide}",
               color=SLIDE_COLORS.get(slide, f"C{si}"), edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Hard-vote image accuracy (%)", fontsize=10)
    ax.set_title(f"{model_type}  ·  {split}  ·  Voting accuracy by class and slide",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "voting_accuracy_by_slide.png"), dpi=100)
    plt.close(fig)

    # ── 5. Confusion matrix — patch level ─────────────────────────────────────
    cm_patch = _confusion_matrix(
        patch_df["true_label"].values, patch_df["pred_label"].values, n_cls)
    fig = plot_confusion_matrix(
        cm_patch, CLASS_NAMES,
        f"{model_type}  ·  {split}  ·  Patch-level confusion")
    fig.savefig(os.path.join(summary_dir, "confusion_patch.png"), dpi=100)
    plt.close(fig)

    # ── 6. Confusion matrix — image voting level ──────────────────────────────
    cm_vote = _confusion_matrix(
        image_df["true_label"].values, image_df["hard_vote_label"].values, n_cls)
    fig = plot_confusion_matrix(
        cm_vote, CLASS_NAMES,
        f"{model_type}  ·  {split}  ·  Image hard-vote confusion")
    fig.savefig(os.path.join(summary_dir, "confusion_voting.png"), dpi=100)
    plt.close(fig)

    # ── 7. overall_stats.txt ──────────────────────────────────────────────────
    n_p = len(patch_df)
    c_p = int(patch_df["correct"].sum())
    n_i = len(image_df)
    c_h = int(image_df["hard_vote_correct"].sum())
    c_s = int(image_df["soft_vote_correct"].sum())

    lines = [
        f"Split: {split}  |  Model: {model_type}",
        "─" * 55,
        f"Patch accuracy:         {c_p/n_p*100:5.1f}%  ({c_p}/{n_p})",
        f"Image accuracy (hard):  {c_h/n_i*100:5.1f}%  ({c_h}/{n_i})",
        f"Image accuracy (soft):  {c_s/n_i*100:5.1f}%  ({c_s}/{n_i})",
        "",
        "Per-class image accuracy (hard vote):",
    ]
    for cls in CLASS_NAMES:
        sub = image_df[image_df["Exp_Type"] == cls]
        if len(sub):
            nc = int(sub["hard_vote_correct"].sum())
            lines.append(f"  {cls:5s}: {nc/len(sub)*100:5.1f}%  ({nc}/{len(sub)})")

    lines += ["", "Per-experiment accuracy (hard vote):"]
    for _, er in exp_data.sort_values("acc", ascending=False).iterrows():
        lines.append(
            f"  {er['Experiment']} ({er['Exp_Type']}): "
            f"{er['acc']:.1f}%  ({er['n_correct']}/{er['n_total']})"
        )

    lines += ["", "Per-slide accuracy (hard vote):"]
    for slide in slides:
        sub = image_df[image_df["Slide_Type"] == slide]
        nc = int(sub["hard_vote_correct"].sum())
        lines.append(f"  Slide {slide}: {nc/len(sub)*100:.1f}%  ({nc}/{len(sub)})")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(summary_dir, "overall_stats.txt"), "w") as f:
        f.write(text)


# ── process one (model × split) combination ───────────────────────────────────

def process_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    split: str,
    model_type: str,
    preprocessor,
    grid: List[Tuple[int, int]],
    photos_dir: str,
    out_dir: str,
) -> None:
    split_dir = os.path.join(out_dir, model_type, split)
    annotated_base = os.path.join(split_dir, "annotated")
    summary_dir = os.path.join(split_dir, "summary")
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(annotated_base, cls), exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    is_train = (split == "train")
    patch_rows: List[dict] = []
    image_rows: List[dict] = []
    n_cls = len(CLASS_NAMES)
    n_img = len(df)

    for img_i, (_, row) in enumerate(df.iterrows()):
        img_path = os.path.join(photos_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: cannot read {img_path} — skipping")
            continue

        true_label = CLASS_MAP[row["Exp_Type"]]
        preds, probs, tensor = infer_image(model, img_bgr, preprocessor, grid)

        # patch-level records
        for pi, (cy, cx) in enumerate(grid):
            patch_rows.append({
                "img_idx": int(row["idx"]),
                "Experiment": row["Experiment"],
                "Exp_Type": row["Exp_Type"],
                "Slide_Type": row["Slide_Type"],
                "split": split,
                "patch_i": pi,
                "center_y": cy,
                "center_x": cx,
                "true_label": true_label,
                "pred_label": int(preds[pi]),
                "correct": bool(preds[pi] == true_label),
                **{f"prob_{CLASS_NAMES[k]}": float(probs[pi, k]) for k in range(n_cls)},
            })

        # image-level voting
        hard_label = int(Counter(preds.tolist()).most_common(1)[0][0])
        mean_probs = probs.mean(axis=0)
        soft_label = int(mean_probs.argmax())

        image_rows.append({
            "img_idx": int(row["idx"]),
            "Experiment": row["Experiment"],
            "Exp_Type": row["Exp_Type"],
            "Slide_Type": row["Slide_Type"],
            "split": split,
            "true_label": true_label,
            "true_class": CLASS_NAMES[true_label],
            "hard_vote_label": hard_label,
            "hard_vote_class": CLASS_NAMES[hard_label],
            "hard_vote_correct": bool(hard_label == true_label),
            "soft_vote_label": soft_label,
            "soft_vote_class": CLASS_NAMES[soft_label],
            "soft_vote_correct": bool(soft_label == true_label),
            "patch_accuracy": float((preds == true_label).mean()),
            **{f"prob_{CLASS_NAMES[k]}": float(mean_probs[k]) for k in range(n_cls)},
        })

        # annotated image — save to annotated/{true_class}/
        voted_class = CLASS_NAMES[hard_label]
        true_class = CLASS_NAMES[true_label]
        fig = draw_annotated_image(
            tensor, grid, preds, true_label, row, voted_class, is_train)
        fname = (
            f"{int(row['idx']):04d}_{row['Experiment']}_{row['Slide_Type']}_"
            f"{true_class}to{voted_class}.png"
        )
        fig.savefig(
            os.path.join(annotated_base, true_class, fname),
            dpi=80, bbox_inches="tight",
        )
        plt.close(fig)

        if (img_i + 1) % 100 == 0 or (img_i + 1) == n_img:
            patch_acc = float((preds == true_label).mean()) * 100
            print(f"  [{img_i+1}/{n_img}]  last: {int(row['idx']):04d} "
                  f"({row['Exp_Type']}) patch_acc={patch_acc:.0f}% "
                  f"voted={voted_class}")

    patch_df = pd.DataFrame(patch_rows)
    image_df = pd.DataFrame(image_rows)
    patch_df.to_csv(os.path.join(split_dir, "patch_results.csv"), index=False)
    image_df.to_csv(os.path.join(split_dir, "image_voting_results.csv"), index=False)
    print(f"  CSVs → {split_dir}")

    save_summary(patch_df, image_df, summary_dir, model_type, split)


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_canonical_model_dir() -> Optional[str]:
    """Return the first patch model dir that has train_record_patch.json."""
    for mt in ALL_PATCH_MODELS:
        d = MODEL_REGISTRY[mt][0]
        if os.path.exists(os.path.join(d, "train_record_patch.json")):
            return d
    return None


def _ensure_gitignore(repo_root: str, entry: str) -> None:
    path = os.path.join(repo_root, ".gitignore")
    if os.path.exists(path):
        with open(path) as f:
            if entry in f.read():
                return
        with open(path, "a") as f:
            f.write(f"\n{entry}\n")
    else:
        with open(path, "w") as f:
            f.write(f"{entry}\n")
    print(f".gitignore: added '{entry}'")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch-model validation visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Patch model types to evaluate (default: all with best_model.pth)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "train"],
        choices=["val", "train", "test"],
        metavar="SPLIT",
        help="Data splits to evaluate: val, train, test (default: val train)",
    )
    parser.add_argument("--db", default="data/endpoint10.db",
                        help="SQLite label database (default: data/endpoint10.db)")
    parser.add_argument("--photos", default="data/photos",
                        help="Directory containing 0000.JPG…0999.JPG")
    parser.add_argument("--out-dir", default="patch_validation",
                        help="Root output directory (default: patch_validation)")
    parser.add_argument("--n-cols", type=int, default=5,
                        help="Patch grid columns (default 5)")
    parser.add_argument("--n-rows", type=int, default=4,
                        help="Patch grid rows (default 4)")
    args = parser.parse_args()

    grid = make_grid(args.n_cols, args.n_rows)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    # Determine which models to evaluate
    candidates = args.models if args.models else ALL_PATCH_MODELS
    models_to_run = []
    for mt in candidates:
        if mt not in MODEL_REGISTRY:
            print(f"WARNING: '{mt}' not in MODEL_REGISTRY — skipping")
            continue
        model_dir, _, _ = MODEL_REGISTRY[mt]
        if not os.path.exists(os.path.join(model_dir, "best_model.pth")):
            print(f"  Skipping {mt}: no best_model.pth in {model_dir}/")
            continue
        models_to_run.append(mt)

    if not models_to_run:
        print("No trained patch models found. Train at least one model first.")
        sys.exit(1)

    # Load canonical split — all patch variants use seed=99, so the partition
    # is identical regardless of which model dir the JSON lives in.
    canonical_dir = _find_canonical_model_dir()
    if canonical_dir is None:
        print(
            "ERROR: train_record_patch.json not found in any patch model directory.\n"
            "       Run at least one patch training job to create the split record."
        )
        sys.exit(1)
    print(f"Loading patch split from {canonical_dir}/train_record_patch.json ...")
    train_df, val_df, test_df = load_split_record(canonical_dir, args.db)
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}

    # Ensure output dir is gitignored
    repo_root = os.path.dirname(os.path.abspath(__file__))
    _ensure_gitignore(repo_root, "patch_validation/")

    print(f"\nGrid:   {args.n_cols}×{args.n_rows} = {len(grid)} patches/image")
    print(f"Models: {models_to_run}")
    print(f"Splits: {args.splits}")
    print(f"Output: {args.out_dir}/\n")

    for mt in models_to_run:
        print("=" * 60)
        print(f"Model: {mt}")
        model = load_patch_model(mt)
        if model is None:
            print("  best_model.pth disappeared — skipping")
            continue
        print("  Weights loaded (CPU, eval mode).")

        for split in args.splits:
            df = split_dfs.get(split)
            if df is None or len(df) == 0:
                print(f"  No images for split '{split}' — skipping")
                continue
            print(f"\n  Split '{split}': {len(df)} images × {len(grid)} patches "
                  f"= {len(df)*len(grid)} forward passes")
            process_split(
                model=model,
                df=df,
                split=split,
                model_type=mt,
                preprocessor=preprocessor,
                grid=grid,
                photos_dir=args.photos,
                out_dir=args.out_dir,
            )

        del model   # free memory before loading next model

    print("\nAll done.")


if __name__ == "__main__":
    main()
