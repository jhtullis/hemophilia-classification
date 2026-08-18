"""
misclassification_report.py — Per-image misclassification analysis (Module 2).

Generates a detailed breakdown of which test images the model gets wrong,
whether errors cluster by experiment or slide type, and how model confidence
differs between correct and incorrect predictions.

Usage:
    python misclassification_report.py [--model-type TYPE] [--output-dir DIR]

Outputs (saved to <model_dir>/analysis/misclassification/ by default):
    per_image_results.csv       — one row per test image with full prediction info
    experiment_accuracy.png     — per-experiment accuracy bar chart
    slide_type_comparison.png   — accuracy by slide type (A vs B) per class
    confusion_matrix.png        — normalized confusion matrix heatmap
    confidence_distribution.png — correct vs incorrect softmax confidence histograms
    misclassification_summary.txt — text summary of key findings
"""

import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis_utils import (CLASS_COLORS, DB_PATH, MODEL_REGISTRY, PHOTOS_DIR,
                             get_val_split, load_model_from_registry,
                             run_inference_full)
from evaluate import confusion_matrix as compute_confusion_matrix
from preprocessing import make_preprocessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Misclassification analysis for FibrinCNN.")
    p.add_argument("--model-type", default="5class",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Which trained model to analyse (default: 5class).")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Override output directory.")
    p.add_argument("--db", default=DB_PATH, metavar="PATH",
                   help="Path to SQLite database (default: data/endpoint10.db).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output A: CSV
# ---------------------------------------------------------------------------

def save_csv(results: pd.DataFrame, out_dir: str) -> None:
    path = os.path.join(out_dir, "per_image_results.csv")
    results.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output B: Experiment accuracy bar chart
# ---------------------------------------------------------------------------

def plot_experiment_accuracy(results: pd.DataFrame, out_dir: str) -> None:
    exp_stats = (results.groupby("Experiment")
                 .agg(accuracy=("correct", "mean"),
                      n=("correct", "count"),
                      dominant_class=("Exp_Type", lambda x: x.mode()[0]))
                 .reset_index()
                 .sort_values("accuracy"))

    overall_mean = results["correct"].mean()
    n_exp = len(exp_stats)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * n_exp)))

    colors = [CLASS_COLORS.get(cls, "#888888") for cls in exp_stats["dominant_class"]]
    bars = ax.barh(exp_stats["Experiment"], exp_stats["accuracy"],
                   color=colors, edgecolor="white", linewidth=0.5)

    ax.axvline(overall_mean, color="black", linestyle="--", linewidth=1.2,
               label=f"Mean ({overall_mean:.3f})")

    # Annotate bars below 50% with count
    for bar, (_, row) in zip(bars, exp_stats.iterrows()):
        if row["accuracy"] < 0.5:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{row['dominant_class']} n={row['n']}",
                    va="center", ha="left", fontsize=7, color="red")

    # Legend for class colors
    legend_patches = [mpatches.Patch(color=v, label=k)
                      for k, v in CLASS_COLORS.items()
                      if k in exp_stats["dominant_class"].values]
    legend_patches.append(mpatches.Patch(color="none", label=""))
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"Mean ({overall_mean:.3f})")
    ], loc="lower right", fontsize=8)

    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_ylabel("Experiment", fontsize=11)
    ax.set_title("Per-Experiment Accuracy (sorted worst → best)", fontsize=12)
    ax.set_xlim(0, 1.12)
    fig.tight_layout()
    path = os.path.join(out_dir, "experiment_accuracy.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output C: Slide type comparison
# ---------------------------------------------------------------------------

def plot_slide_type_comparison(results: pd.DataFrame, class_names: list,
                               out_dir: str) -> None:
    slide_acc = (results.groupby(["Exp_Type", "Slide_Type"])["correct"]
                 .mean().unstack(fill_value=np.nan))

    # Ensure A and B columns exist even if missing in data
    for col in ["A", "B"]:
        if col not in slide_acc.columns:
            slide_acc[col] = np.nan

    # Reindex to class_names order
    slide_acc = slide_acc.reindex(class_names)

    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(class_names)), 5))

    colors_list = [CLASS_COLORS.get(c, "#888888") for c in class_names]
    ax.bar(x - width / 2, slide_acc["A"].fillna(0), width,
           color=colors_list, label="Slide A", edgecolor="white")
    ax.bar(x + width / 2, slide_acc["B"].fillna(0), width,
           color=colors_list, label="Slide B", hatch="///",
           edgecolor="white", alpha=0.75)

    # Mark missing data
    for i, cls in enumerate(class_names):
        if np.isnan(slide_acc.loc[cls, "A"]):
            ax.text(x[i] - width / 2, 0.02, "N/A", ha="center",
                    va="bottom", fontsize=7, color="gray")
        if np.isnan(slide_acc.loc[cls, "B"]):
            ax.text(x[i] + width / 2, 0.02, "N/A", ha="center",
                    va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy by Class and Slide Type", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "slide_type_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output D: Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    results: pd.DataFrame, class_names: list, out_dir: str,
    title: str = "Confusion Matrix (count / fraction of true class)",
    class_key: dict = None,
    class_order: list = None,
    cell_fontsize: float = 9,
    tick_fontsize: float = 9,
    label_fontsize: float = 11,
    title_fontsize: float = 11,
    key_fontsize: float = 9.5,
) -> None:
    """class_key: optional {abbreviation: full name} dict rendered as a horizontal
    text key to the right of the colorbar (e.g. for a presentation-facing figure
    where the class abbreviations need spelling out). class_order: optional
    reordering of class_names for display (rows/cols of the matrix are permuted to
    match; the underlying cm is always computed in class_names' original label-index
    order, so this is purely cosmetic). The font-size params default to this
    function's original hardcoded sizes, so existing callers are unaffected; pass
    larger values for a single call site without changing the shared defaults."""
    n = len(class_names)
    cm = compute_confusion_matrix(results["pred_label"].values,
                                  results["true_label"].values, n)
    if class_order is not None:
        perm = [class_names.index(c) for c in class_order]
        cm = cm[np.ix_(perm, perm)]
        class_names = class_order
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    cell = max(1.2, 6.0 / n)
    extra_w = 2.6 if class_key else 0.0
    fig, ax = plt.subplots(figsize=(n * cell + 1.5 + extra_w, n * cell + 0.5))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            text_color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=cell_fontsize, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(class_names, fontsize=tick_fontsize)
    ax.set_xlabel("Predicted Class", fontsize=label_fontsize)
    ax.set_ylabel("True Class", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, label="Fraction of true class")
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.ax.yaxis.label.set_size(label_fontsize)
    fig.tight_layout()

    if class_key:
        # Bold, individually-placed entries spread across the colorbar's vertical
        # extent (rather than one centered multi-line block) use the available
        # height instead of clustering at the middle. The horizontal offset is
        # measured from the colorbar's own rendered label extent (not a fixed
        # guess), so entries sit close to the colorbar without overlapping
        # "Fraction of true class" regardless of font-size changes. bbox_inches=
        # "tight" on savefig expands the crop to include text placed beyond the
        # axes' right edge, so no manual figure-width math is needed here.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_bbox = cbar.ax.yaxis.label.get_window_extent(renderer=renderer)
        label_bbox_fig = label_bbox.transformed(fig.transFigure.inverted())
        key_x = label_bbox_fig.x1 + 0.02

        cbar_pos = cbar.ax.get_position()
        items = list(class_key.items())
        if len(items) > 1:
            y_positions = np.linspace(cbar_pos.y1 - 0.02, cbar_pos.y0 + 0.02, len(items))
        else:
            y_positions = [(cbar_pos.y0 + cbar_pos.y1) / 2]
        for (abbr, full), y in zip(items, y_positions):
            fig.text(key_x, y, f"{abbr}:  {full}", transform=fig.transFigure,
                      fontsize=key_fontsize, fontweight="normal", va="center", ha="left",
                      color="#1a1a17")

    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output E: Confidence distribution
# ---------------------------------------------------------------------------

def plot_confidence_distribution(results: pd.DataFrame, out_dir: str) -> None:
    correct   = results[results["correct"]]["confidence"].values
    incorrect = results[~results["correct"]]["confidence"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)

    ax.hist(correct,   bins=bins, alpha=0.6, color="green",
            label=f"Correct   (n={len(correct)},  mean={correct.mean():.3f})")
    ax.hist(incorrect, bins=bins, alpha=0.6, color="red",
            label=f"Incorrect (n={len(incorrect)}, mean={incorrect.mean():.3f})")

    if len(correct) > 0:
        ax.axvline(correct.mean(), color="darkgreen", linestyle="--", linewidth=1.5)
    if len(incorrect) > 0:
        ax.axvline(incorrect.mean(), color="darkred", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Softmax Confidence (max probability)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Prediction Confidence Distribution", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "confidence_distribution.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output F: Text summary
# ---------------------------------------------------------------------------

def write_summary(results: pd.DataFrame, class_names: list,
                  model_type: str, out_dir: str) -> None:
    lines = []
    lines.append("=== Misclassification Summary ===")
    lines.append(f"Model:       {model_type}")
    lines.append(f"Test images: {len(results)}")
    n_correct = results["correct"].sum()
    lines.append(f"Overall accuracy: {n_correct/len(results):.4f}  ({n_correct}/{len(results)})")

    lines.append("\n── Per-Class Accuracy ──")
    lines.append(f"{'Class':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    per_class = results.groupby("Exp_Type")["correct"].agg(["sum", "count"])
    for cls in class_names:
        if cls in per_class.index:
            row = per_class.loc[cls]
            acc = row["sum"] / row["count"]
            lines.append(f"{cls:<8} {int(row['sum']):>8} {int(row['count']):>8} {acc:>10.3f}")

    # Top confused pairs
    wrong = results[~results["correct"]][["true_label", "pred_label"]].copy()
    inv = {v: k for k, v in {name: i for i, name in enumerate(class_names)}.items()}
    if len(wrong) > 0:
        pair_counts = (wrong.groupby(["true_label", "pred_label"])
                       .size().reset_index(name="count")
                       .sort_values("count", ascending=False))
        lines.append("\n── Top Confused Class Pairs ──")
        lines.append(f"{'True → Predicted':<25} {'Count':>6}")
        for _, r in pair_counts.head(5).iterrows():
            true_name = class_names[int(r["true_label"])]
            pred_name = class_names[int(r["pred_label"])]
            lines.append(f"{true_name + ' → ' + pred_name:<25} {int(r['count']):>6}")
    else:
        lines.append("\nNo misclassifications.")

    # Low-accuracy experiments
    exp_acc = results.groupby("Experiment")["correct"].mean()
    low_exp = exp_acc[exp_acc < 0.5]
    lines.append("\n── Experiments with Accuracy < 50% ──")
    if len(low_exp) > 0:
        lines.append(f"{'Experiment':<20} {'Class':<8} {'N':>4} {'Accuracy':>10}")
        for exp, acc in low_exp.items():
            exp_rows = results[results["Experiment"] == exp]
            cls = exp_rows["Exp_Type"].mode()[0]
            n = len(exp_rows)
            lines.append(f"{exp:<20} {cls:<8} {n:>4} {acc:>10.3f}")
    else:
        lines.append("  (None — all experiments ≥ 50%)")

    # Slide type comparison
    slide_acc = (results.groupby(["Exp_Type", "Slide_Type"])["correct"]
                 .mean().unstack(fill_value=np.nan))
    for col in ["A", "B"]:
        if col not in slide_acc.columns:
            slide_acc[col] = np.nan
    lines.append("\n── Slide Type Comparison ──")
    lines.append(f"{'Class':<8} {'Slide A Acc':>12} {'Slide B Acc':>12} {'Diff':>8}")
    for cls in class_names:
        if cls in slide_acc.index:
            a = slide_acc.loc[cls, "A"]
            b = slide_acc.loc[cls, "B"]
            diff = (a - b) if (not np.isnan(a) and not np.isnan(b)) else np.nan
            a_str = f"{a:.3f}" if not np.isnan(a) else "N/A"
            b_str = f"{b:.3f}" if not np.isnan(b) else "N/A"
            d_str = f"{diff:+.3f}" if not np.isnan(diff) else "N/A"
            lines.append(f"{cls:<8} {a_str:>12} {b_str:>12} {d_str:>8}")

    # Confidence stats
    correct_conf   = results[results["correct"]]["confidence"]
    incorrect_conf = results[~results["correct"]]["confidence"]
    lines.append("\n── Confidence Statistics ──")
    if len(correct_conf) > 0:
        lines.append(f"Correct predictions:   "
                     f"mean={correct_conf.mean():.3f}  std={correct_conf.std():.3f}")
    if len(incorrect_conf) > 0:
        lines.append(f"Incorrect predictions: "
                     f"mean={incorrect_conf.mean():.3f}  std={incorrect_conf.std():.3f}")

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "misclassification_summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")

    print(f"Loading {args.model_type} model ...")
    model, model_dir, num_classes, class_map, class_names = \
        load_model_from_registry(args.model_type, device)
    val_df = get_val_split(args.model_type, db_path=args.db)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    print(f"Running inference on {len(val_df)} validation images ...")
    results = run_inference_full(
        model, val_df, PHOTOS_DIR, device, preprocessor, class_map, class_names
    )
    print(f"  Overall accuracy: {results['correct'].mean():.4f}")

    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "misclassification")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {out_dir}/")

    save_csv(results, out_dir)
    plot_confusion_matrix(results, class_names, out_dir)
    plot_experiment_accuracy(results, out_dir)
    plot_slide_type_comparison(results, class_names, out_dir)
    plot_confidence_distribution(results, out_dir)
    write_summary(results, class_names, args.model_type, out_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
