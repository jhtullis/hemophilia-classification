"""
aggregate_learning_curve.py — Aggregates the per-model holdout results written
by evaluate_holdout_lite_lc.py (models/<lc_key>/holdout_eval/summary.json)
into a learning-curve summary: accuracy vs. lc_n_per_class, both patch- and
image-level, with the 4 seed repetitions (a/b/c/d) shown per size.

Safely re-runnable mid-progress — missing model summaries are skipped with a
warning rather than raising.

Usage:
    python aggregate_learning_curve.py
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))

LC_SIZES = [5, 10, 15, 20, 25, 30, 35]
LC_REPS = ["a", "b", "c", "d"]

# dataviz reference palette, categorical slots 1 (blue) and 2 (orange)
COLOR_PATCH = "#2a78d6"
COLOR_IMAGE = "#eb6834"


def aggregate(args: argparse.Namespace) -> None:
    models_root = args.models_root or os.path.join(_DIR, "models")
    out_dir = args.out_dir or os.path.join(_DIR, "models", "mpatch_v1e_lite_lc_learning_curve")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    missing = []
    for n in LC_SIZES:
        for r in LC_REPS:
            key = f"mpatch_v1e_lite_lc{n:02d}{r}"
            summary_path = os.path.join(models_root, key, "holdout_eval", "summary.json")
            if not os.path.exists(summary_path):
                missing.append(key)
                continue
            with open(summary_path) as f:
                s = json.load(f)
            rows.append({
                "model_key": key,
                "lc_n_per_class": s.get("lc_n_per_class", n),
                "rep": r,
                "patch_accuracy": s["patch_accuracy"],
                "image_accuracy": s["image_accuracy"],
                "best_epoch": s.get("best_epoch"),
                "best_val_acc": s.get("best_val_acc"),
            })

    if missing:
        print(f"WARNING: {len(missing)} model summaries not found (skipped): {missing}")
    if not rows:
        raise FileNotFoundError(
            "No holdout_eval/summary.json files found under any lc model directory. "
            "Run evaluate_holdout_lite_lc.py first."
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "learning_curve_summary.csv"), index=False)

    agg = (
        df.groupby("lc_n_per_class")[["patch_accuracy", "image_accuracy"]]
        .agg(["mean", "std", "count"])
    )
    summary = {
        "n_models_found": len(df),
        "n_models_missing": len(missing),
        "missing_models": missing,
        "per_size": {
            int(n): {
                "patch_accuracy_mean": float(agg.loc[n, ("patch_accuracy", "mean")]),
                "patch_accuracy_std": float(agg.loc[n, ("patch_accuracy", "std")] or 0.0),
                "image_accuracy_mean": float(agg.loc[n, ("image_accuracy", "mean")]),
                "image_accuracy_std": float(agg.loc[n, ("image_accuracy", "std")] or 0.0),
                "n_reps": int(agg.loc[n, ("patch_accuracy", "count")]),
            }
            for n in agg.index
        },
    }
    with open(os.path.join(out_dir, "learning_curve_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _plot_learning_curve(df, out_dir)
    print(f"Learning-curve summary written to {out_dir}")


def _plot_learning_curve(df: pd.DataFrame, out_dir: str) -> None:
    sizes = sorted(df["lc_n_per_class"].unique())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    for metric, color, label in [
        ("patch_accuracy", COLOR_PATCH, "Patch accuracy"),
        ("image_accuracy", COLOR_IMAGE, "Image accuracy (soft-vote)"),
    ]:
        means = [df[df["lc_n_per_class"] == n][metric].mean() for n in sizes]
        stds = [df[df["lc_n_per_class"] == n][metric].std() or 0.0 for n in sizes]
        ax.errorbar(sizes, means, yerr=stds, color=color, lw=2, marker="o",
                    markersize=7, capsize=4, label=label, zorder=3)
        for n in sizes:
            reps = df[df["lc_n_per_class"] == n][metric]
            ax.scatter([n] * len(reps), reps, color=color, s=24, alpha=0.35, zorder=2)

    ax.set_xlabel("Training experiments per class (lc_n_per_class)", color="#0b0b0b")
    ax.set_ylabel("Accuracy", color="#0b0b0b")
    ax.set_title("mpatch_v1e_lite_lc — Holdout Learning Curve", color="#0b0b0b")
    ax.set_xticks(sizes)
    ax.grid(True, color="#e1e0d9", lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curve_combined.png")
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate mpatch_v1e_lite_lc* holdout results into a learning-curve summary."
    )
    p.add_argument("--models-root", default=None)
    p.add_argument("--out-dir", default=None)
    aggregate(p.parse_args())
