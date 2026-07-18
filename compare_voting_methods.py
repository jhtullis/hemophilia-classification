"""
compare_voting_methods.py — Statistical comparison of ABMIL vs. hard-vote vs.
soft-vote image-level aggregation, consuming evaluate_abmil.py's
per_image_predictions.csv.

No paired significance test (McNemar's or otherwise) exists elsewhere in this
repo, so it's implemented here from scratch, hand-rolled (no scipy/sklearn —
matches this repo's existing no-sklearn convention, e.g. evaluate.py's
from-scratch metrics and train_patch.py's silhouette score). An exact
two-sided binomial McNemar's test is used rather than the chi-square
approximation, since at this repo's scale (val n~100, test n~200) the number
of discordant pairs is typically well under 25, where the chi-square
approximation is unreliable.

Usage:
    python compare_voting_methods.py --model-dir models/abmil_mpatch_v0_f --split val

Outputs to <model-dir>/analysis/<split>/comparison/:
    mcnemar_results.txt
    per_class_accuracy_table.csv
    per_class_accuracy.png
    confusion_matrices_comparison.png
    summary.txt
"""

import argparse
import math
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import CLASS_COLORS
from data_loader import CLASS_NAMES

_METHODS = ("hard_vote", "soft_vote", "abmil")
_METHOD_LABELS = {"hard_vote": "Hard-vote", "soft_vote": "Soft-vote", "abmil": "ABMIL"}
_METHOD_COLORS = {"hard_vote": "#9E9E9E", "soft_vote": "#F2A65A", "abmil": "#4878CF"}


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def _binom_two_sided_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar's test via the binomial CDF (no scipy)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_le_k = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_le_k)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Paired McNemar's test on two boolean correctness arrays over the SAME
    images. b = a right & b wrong; c = a wrong & b right."""
    a = correct_a.astype(bool)
    b_arr = correct_b.astype(bool)
    b = int((a & ~b_arr).sum())
    c = int((~a & b_arr).sum())
    return {"b": b, "c": c, "n_discordant": b + c, "p_value": _binom_two_sided_pvalue(b, c)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_per_class_accuracy(table_df: pd.DataFrame, class_names, out_path: str) -> None:
    n = len(class_names)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, method in enumerate(_METHODS):
        vals = [table_df.loc[table_df["Exp_Type"] == c, f"{method}_acc"].iloc[0] * 100
                for c in class_names]
        ax.bar(x + (i - 1) * width, vals, width,
               label=_METHOD_LABELS[method], color=_METHOD_COLORS[method])

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    for tick, cls in zip(ax.get_xticklabels(), class_names):
        tick.set_color(CLASS_COLORS.get(cls, "black"))
        tick.set_fontweight("bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-class accuracy — hard-vote vs. soft-vote vs. ABMIL")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_confusion_grid(df: pd.DataFrame, class_names, out_path: str) -> None:
    n = len(class_names)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, method in zip(axes, _METHODS):
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(df["true_label"].values, df[f"{method}_pred"].values):
            cm[t, p] += 1
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        acc = (df["true_label"].values == df[f"{method}_pred"].values).mean()
        ax.set_title(f"{_METHOD_LABELS[method]}  (acc={acc:.3f})")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(args: argparse.Namespace) -> None:
    csv_path = args.predictions or os.path.join(
        args.model_dir, "analysis", args.split, "per_image_predictions.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run evaluate_abmil.py first."
        )
    df = pd.read_csv(csv_path)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(csv_path), "comparison"
    )
    os.makedirs(out_dir, exist_ok=True)

    class_names = [c for c in CLASS_NAMES if c in set(df["Exp_Type"])]

    correct = {m: df[f"{m}_correct"].values.astype(bool) for m in _METHODS}
    accs = {m: float(correct[m].mean()) for m in _METHODS}

    mcnemar_hard = mcnemar_test(correct["abmil"], correct["hard_vote"])
    mcnemar_soft = mcnemar_test(correct["abmil"], correct["soft_vote"])

    # ── per-class accuracy table ────────────────────────────────────────
    rows = []
    for cls in class_names:
        sub = df[df["Exp_Type"] == cls]
        row = {"Exp_Type": cls, "n": len(sub)}
        for m in _METHODS:
            row[f"{m}_acc"] = float(sub[f"{m}_correct"].mean()) if len(sub) else float("nan")
        rows.append(row)
    table_df = pd.DataFrame(rows)
    table_path = os.path.join(out_dir, "per_class_accuracy_table.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Saved {table_path}")

    _plot_per_class_accuracy(table_df, class_names, os.path.join(out_dir, "per_class_accuracy.png"))
    _plot_confusion_grid(df, class_names, os.path.join(out_dir, "confusion_matrices_comparison.png"))

    # ── McNemar results ─────────────────────────────────────────────────
    mcnemar_path = os.path.join(out_dir, "mcnemar_results.txt")
    with open(mcnemar_path, "w") as f:
        f.write("McNemar's exact test (two-sided, binomial CDF)\n")
        f.write("Paired on identical images and identical patches per image.\n")
        f.write("=" * 70 + "\n\n")
        for name, res in (("ABMIL vs hard-vote", mcnemar_hard), ("ABMIL vs soft-vote", mcnemar_soft)):
            f.write(f"{name}:\n")
            f.write(f"  b (ABMIL right, other wrong): {res['b']}\n")
            f.write(f"  c (ABMIL wrong, other right): {res['c']}\n")
            f.write(f"  discordant pairs (b+c):       {res['n_discordant']}\n")
            f.write(f"  p-value:                       {res['p_value']:.4f}\n\n")
    print(f"Saved {mcnemar_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Voting method comparison — {csv_path}\n")
        f.write(f"n = {len(df)}\n")
        f.write("=" * 70 + "\n\n")
        for m in _METHODS:
            f.write(f"{_METHOD_LABELS[m]:12s} overall accuracy: {accs[m]:.3f}\n")
        f.write("\n")
        f.write(f"ABMIL vs hard-vote: p={mcnemar_hard['p_value']:.4f} "
                f"(b={mcnemar_hard['b']}, c={mcnemar_hard['c']})\n")
        f.write(f"ABMIL vs soft-vote: p={mcnemar_soft['p_value']:.4f} "
                f"(b={mcnemar_soft['b']}, c={mcnemar_soft['c']})\n")
    print(f"Saved {summary_path}")

    print(f"\nOverall accuracy — hard-vote: {accs['hard_vote']:.3f}  "
          f"soft-vote: {accs['soft_vote']:.3f}  ABMIL: {accs['abmil']:.3f}")
    print(f"McNemar ABMIL vs hard-vote: p={mcnemar_hard['p_value']:.4f} "
          f"(b={mcnemar_hard['b']}, c={mcnemar_hard['c']})")
    print(f"McNemar ABMIL vs soft-vote: p={mcnemar_soft['p_value']:.4f} "
          f"(b={mcnemar_soft['b']}, c={mcnemar_soft['c']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare ABMIL vs. hard-vote vs. soft-vote via paired McNemar's test."
    )
    p.add_argument("--model-dir", default=None,
                    help="ABMIL model dir (e.g. models/abmil_mpatch_v0_f). "
                         "Used to locate analysis/<split>/per_image_predictions.csv "
                         "unless --predictions is given directly.")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--predictions", default=None,
                    help="Explicit path to a per_image_predictions.csv (overrides --model-dir/--split).")
    p.add_argument("--out-dir", default=None,
                    help="Output dir (default: <predictions dir>/comparison/).")
    args = p.parse_args()
    if args.predictions is None and args.model_dir is None:
        p.error("either --model-dir or --predictions is required")
    compare(args)
