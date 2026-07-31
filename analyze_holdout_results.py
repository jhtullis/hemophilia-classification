"""
analyze_holdout_results.py — Local analysis of holdout-eval outputs already
pulled down from HPC (models/mpatch_v1e_125s_ensemble/holdout_eval/ and
models/mpatch_v1e_lite_lc*/holdout_eval/). Recomputes cross-entropy loss from
the raw per-class score columns already saved in per_image_predictions.csv /
per_patch_predictions.csv -- all evaluated models use head_type="ce", so
score_<class> columns are raw logits, sufficient for both accuracy (already
saved) and loss (not saved, computed here). No model re-run or HPC job needed.

Writes to analysis/ at the repo root:
    analysis/125s_ensemble/summary.json, summary_table.csv
    analysis/learning_curve/learning_curve_summary.{csv,json}
    analysis/learning_curve/learning_curve_accuracy.png
    analysis/learning_curve/learning_curve_loss.png

Usage:
    python analyze_holdout_results.py
    python analyze_holdout_results.py --which 125s
    python analyze_holdout_results.py --which lc
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import CLASS_NAMES
from holdout_eval_utils import accuracy_breakdown, write_json

_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(_DIR, "models")
ANALYSIS_ROOT = os.path.join(_DIR, "analysis")

LC_SIZES = [5, 10, 15, 20, 25, 30, 35]
LC_REPS = ["a", "b", "c", "d"]
LC_KEYS = [f"mpatch_v1e_lite_lc{n:02d}{r}" for n in LC_SIZES for r in LC_REPS]

FOLD_KEYS = [
    "mpatch_v1e_125s",
    "mpatch_v1e_125s_rep_a",
    "mpatch_v1e_125s_rep_b",
    "mpatch_v1e_125s_rep_c",
    "mpatch_v1e_125s_rep_d",
    "mpatch_v1e_125s_rep_e",
    "mpatch_v1e_125s_rep_f",
    "mpatch_v1e_125s_rep_g",
]

# dataviz reference palette, categorical slots 1 (blue) and 2 (orange)
COLOR_PATCH = "#2a78d6"
COLOR_IMAGE = "#eb6834"


def compute_loss(
    df: pd.DataFrame, class_names: List[str] = CLASS_NAMES
) -> Tuple[float, Dict[str, float]]:
    """Mean cross-entropy loss (and per-class breakdown) from raw score_<class>
    logit columns + true_label. Same formula used at training time."""
    logits = torch.tensor(df[[f"score_{c}" for c in class_names]].values, dtype=torch.float32)
    targets = torch.tensor(df["true_label"].values, dtype=torch.long)
    per_row = F.cross_entropy(logits, targets, reduction="none")

    per_class = {}
    for i, cls in enumerate(class_names):
        mask = targets == i
        if mask.any():
            per_class[cls] = per_row[mask].mean().item()
    return per_row.mean().item(), per_class


# ---------------------------------------------------------------------------
# 125s 8-fold ensemble
# ---------------------------------------------------------------------------

def analyze_125s_ensemble() -> None:
    holdout_dir = os.path.join(MODELS_ROOT, "mpatch_v1e_125s_ensemble", "holdout_eval")
    image_df = pd.read_csv(os.path.join(holdout_dir, "per_image_predictions.csv"))
    patch_df = pd.read_csv(os.path.join(holdout_dir, "per_patch_predictions.csv"))

    with open(os.path.join(holdout_dir, "training_metadata.json")) as f:
        fold_metadata = {m["config_key"]: m for m in json.load(f)["models"]}

    img_acc, img_acc_per_class = accuracy_breakdown(image_df, CLASS_NAMES)
    patch_acc, patch_acc_per_class = accuracy_breakdown(patch_df, CLASS_NAMES)
    img_loss, img_loss_per_class = compute_loss(image_df)
    patch_loss, patch_loss_per_class = compute_loss(patch_df)

    # Solo per-fold accuracy is recoverable (pred_label_<key>/correct_<key> were
    # saved); solo per-fold loss is NOT -- only the ensemble's raw scores were
    # persisted per row, not each of the 8 models' own score vectors.
    solo_rows = []
    for key in FOLD_KEYS:
        correct_col = f"correct_{key}"
        if correct_col not in image_df.columns:
            continue
        solo_img_df = image_df[["Exp_Type"]].assign(correct=image_df[correct_col])
        solo_acc, _ = accuracy_breakdown(solo_img_df, CLASS_NAMES)
        meta = fold_metadata.get(key, {})
        solo_rows.append({
            "model_key": key,
            "image_accuracy": solo_acc,
            "image_loss": None,
            "patch_accuracy": None,
            "patch_loss": None,
            "best_epoch": meta.get("best_epoch"),
            "best_val_acc": meta.get("best_val_acc"),
        })

    out_dir = os.path.join(ANALYSIS_ROOT, "125s_ensemble")
    os.makedirs(out_dir, exist_ok=True)

    write_json(os.path.join(out_dir, "summary.json"), {
        "n_test_images": len(image_df),
        "n_grid_patches_per_image": len(patch_df) // len(image_df) if len(image_df) else None,
        "ensemble": {
            "image_accuracy": img_acc,
            "image_loss": img_loss,
            "image_accuracy_per_class": img_acc_per_class,
            "image_loss_per_class": img_loss_per_class,
            "patch_accuracy": patch_acc,
            "patch_loss": patch_loss,
            "patch_accuracy_per_class": patch_acc_per_class,
            "patch_loss_per_class": patch_loss_per_class,
        },
        "solo_models_note": (
            "Per-fold raw scores weren't saved during the HPC run (only "
            "pred_label/correct per fold), so solo-model loss can't be recomputed "
            "locally -- accuracy only. Re-run evaluate_holdout_125s_ensemble.py with "
            "per-fold score columns added if solo loss is needed."
        ),
        "solo_models": solo_rows,
    })

    rows = [{
        "model_key": "ensemble",
        "image_accuracy": img_acc, "image_loss": img_loss,
        "patch_accuracy": patch_acc, "patch_loss": patch_loss,
    }] + solo_rows
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    print(f"125s ensemble -- image: acc={img_acc:.3f} loss={img_loss:.3f}  "
          f"patch: acc={patch_acc:.3f} loss={patch_loss:.3f}")
    print(f"Written to {out_dir}")


# ---------------------------------------------------------------------------
# lite_lc learning curve (accuracy + loss)
# ---------------------------------------------------------------------------

def analyze_learning_curve() -> None:
    rows = []
    missing = []
    for key in LC_KEYS:
        holdout_dir = os.path.join(MODELS_ROOT, key, "holdout_eval")
        img_path = os.path.join(holdout_dir, "per_image_predictions.csv")
        patch_path = os.path.join(holdout_dir, "per_patch_predictions.csv")
        summary_path = os.path.join(holdout_dir, "summary.json")
        if not (os.path.exists(img_path) and os.path.exists(patch_path)):
            missing.append(key)
            continue

        image_df = pd.read_csv(img_path)
        patch_df = pd.read_csv(patch_path)
        img_acc, _ = accuracy_breakdown(image_df, CLASS_NAMES)
        patch_acc, _ = accuracy_breakdown(patch_df, CLASS_NAMES)
        img_loss, _ = compute_loss(image_df)
        patch_loss, _ = compute_loss(patch_df)

        with open(summary_path) as f:
            meta = json.load(f)

        rows.append({
            "model_key": key,
            "lc_n_per_class": meta.get("lc_n_per_class"),
            "rep": key[-1],
            "image_accuracy": img_acc,
            "image_loss": img_loss,
            "patch_accuracy": patch_acc,
            "patch_loss": patch_loss,
            "best_epoch": meta.get("best_epoch"),
            "best_val_acc": meta.get("best_val_acc"),
        })

    if missing:
        print(f"WARNING: {len(missing)} lc models missing holdout_eval outputs, skipped: {missing}")
    if not rows:
        raise FileNotFoundError("No lite_lc holdout_eval outputs found under models/.")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ANALYSIS_ROOT, "learning_curve")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "learning_curve_summary.csv"), index=False)

    metrics = ["image_accuracy", "image_loss", "patch_accuracy", "patch_loss"]
    agg = df.groupby("lc_n_per_class")[metrics].agg(["mean", "std", "count"])
    per_size = {}
    for n in agg.index:
        entry = {"n_reps": int(agg.loc[n, (metrics[0], "count")])}
        for m in metrics:
            std = agg.loc[n, (m, "std")]
            entry[f"{m}_mean"] = float(agg.loc[n, (m, "mean")])
            entry[f"{m}_std"] = float(std) if std == std else 0.0   # NaN -> 0.0 (single rep)
        per_size[int(n)] = entry

    write_json(os.path.join(out_dir, "learning_curve_summary.json"), {
        "n_models_found": len(df),
        "n_models_missing": len(missing),
        "missing_models": missing,
        "per_size": per_size,
    })

    _plot_curve(df, out_dir, ["patch_accuracy", "image_accuracy"],
                "Accuracy", "learning_curve_accuracy.png")
    _plot_curve(df, out_dir, ["patch_loss", "image_loss"],
                "Cross-entropy loss", "learning_curve_loss.png")

    print(f"Learning curve analysis written to {out_dir} ({len(df)}/{len(LC_KEYS)} models)")


_METRIC_LABELS = {
    "patch_accuracy": "Patch accuracy",
    "image_accuracy": "Image accuracy (soft-vote)",
    "patch_loss": "Patch loss",
    "image_loss": "Image loss (soft-vote)",
}


def _plot_curve(
    df: pd.DataFrame, out_dir: str, metrics: List[str], ylabel: str, filename: str
) -> None:
    sizes = sorted(df["lc_n_per_class"].unique())
    colors = {metrics[0]: COLOR_PATCH, metrics[1]: COLOR_IMAGE}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    for metric in metrics:
        means = [df[df["lc_n_per_class"] == n][metric].mean() for n in sizes]
        stds = [df[df["lc_n_per_class"] == n][metric].std() or 0.0 for n in sizes]
        ax.errorbar(sizes, means, yerr=stds, color=colors[metric], lw=2, marker="o",
                    markersize=7, capsize=4, label=_METRIC_LABELS[metric], zorder=3)
        for n in sizes:
            reps = df[df["lc_n_per_class"] == n][metric]
            ax.scatter([n] * len(reps), reps, color=colors[metric], s=24, alpha=0.35, zorder=2)

    ax.set_xlabel("Training experiments per class (lc_n_per_class)", color="#0b0b0b")
    ax.set_ylabel(ylabel, color="#0b0b0b")
    ax.set_title(f"mpatch_v1e_lite_lc -- Holdout Learning Curve ({ylabel})", color="#0b0b0b")
    ax.set_xticks(sizes)
    ax.grid(True, color="#e1e0d9", lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Analyze holdout-eval outputs already pulled from HPC (accuracy + loss)."
    )
    p.add_argument("--which", choices=["125s", "lc", "all"], default="all")
    args = p.parse_args()

    if args.which in ("125s", "all"):
        analyze_125s_ensemble()
    if args.which in ("lc", "all"):
        analyze_learning_curve()
