"""
preprocessing_comparison.py — Robustness analysis across grayscale methods (Module 5).

Evaluates the trained model with each of the 5 implemented grayscale methods.
This is a SENSITIVITY/ROBUSTNESS test: the model was trained on lab_l, so results
show how well it generalizes to alternative preprocessing, not the optimal accuracy
achievable by training a dedicated model per method.

Usage:
    python preprocessing_comparison.py [--model-type TYPE] [--output-dir DIR]

Outputs (saved to <model_dir>/analysis/preprocessing/ by default):
    method_visual_comparison.png        — grayscale method comparison grid
    preprocessing_comparison_table.csv — accuracy + per-class F1 for each method
    preprocessing_accuracy_bars.png    — grouped bar chart
    preprocessing_comparison_report.txt — human-readable summary
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import (CLASS_MAP, CLASS_NAMES, FibrinDataset,
                         filter_classes, load_split_from_record)
from evaluate import confusion_matrix as compute_cm
from hemophilia_analysis import _MODEL_REGISTRY, load_model
from preprocessing import make_preprocessor, min_pool, to_grayscale

GRAY_METHODS   = ["lab_l", "luminance", "hsv_v", "hsv_s", "green"]
POOL_FACTOR    = 10
DB_PATH        = os.path.join(os.path.dirname(__file__), "data", "test_db.db")
PHOTOS_DIR     = os.path.join(os.path.dirname(__file__), "data", "photos")
HEMOPHILIA_CLS = ["F08D", "F09D", "F11D"]
CLASS_MAP_3    = {"F08D": 0, "F09D": 1, "F11D": 2}
CLASS_NAMES_3  = ["F08D", "F09D", "F11D"]

METHOD_LABELS = {
    "lab_l":     "LAB-L\n(training)",
    "luminance":  "Luminance\n(BT.601)",
    "hsv_v":      "HSV-Value\n(max RGB)",
    "hsv_s":      "HSV-Sat",
    "green":      "Green\nchannel",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Preprocessing method robustness analysis for FibrinCNN."
    )
    p.add_argument("--model-type", default="5class",
                   choices=list(_MODEL_REGISTRY.keys()),
                   help="Which trained model to analyse (default: 5class).")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Override output directory.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_test_df(model_type: str):
    """Return (test_df, class_names, num_classes, class_map)."""
    record_path = os.path.join("models", "5class", "train_record.json")
    _, test_df = load_split_from_record(record_path, DB_PATH)
    _, num_classes, class_map = _MODEL_REGISTRY[model_type]
    if num_classes == 3:
        test_df = filter_classes(test_df, HEMOPHILIA_CLS)
        return test_df, CLASS_NAMES_3, 3, CLASS_MAP_3
    return test_df, CLASS_NAMES, 5, CLASS_MAP


class _FibrinDataset3(FibrinDataset):
    """FibrinDataset variant that remaps labels via a custom class_map."""
    def __init__(self, df, photo_dir, preprocessor, class_map):
        super().__init__(df, photo_dir, preprocessor, augment=False)
        self._cmap = class_map

    def __getitem__(self, idx):
        tensor, _ = super().__getitem__(idx)
        row = self.df.iloc[idx]
        return tensor, self._cmap[row["Exp_Type"]]


# ---------------------------------------------------------------------------
# Per-class F1 from confusion matrix
# ---------------------------------------------------------------------------

def _f1_from_cm(cm: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    f1 = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[cls] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


# ---------------------------------------------------------------------------
# Output A: method_visual_comparison.png
# ---------------------------------------------------------------------------

def plot_visual_comparison(test_df: pd.DataFrame, class_names: List[str],
                           out_dir: str) -> None:
    # One representative image per class (smallest idx)
    rep: Dict[str, np.ndarray] = {}
    for cls in class_names:
        rows = test_df[test_df["Exp_Type"] == cls].sort_values("idx")
        if rows.empty:
            continue
        idx = int(rows.iloc[0]["idx"])
        img = cv2.imread(os.path.join(PHOTOS_DIR, f"{idx:04d}.JPG"))
        if img is not None:
            rep[cls] = img

    n_cls = len(rep)
    n_mth = len(GRAY_METHODS)
    fig, axes = plt.subplots(n_cls, n_mth,
                             figsize=(3.5 * n_mth, 3 * n_cls),
                             squeeze=False)

    for row_i, cls in enumerate(cn for cn in class_names if cn in rep):
        img_bgr = rep[cls]
        for col_i, method in enumerate(GRAY_METHODS):
            gray  = to_grayscale(img_bgr, method=method)
            pooled = min_pool(gray, factor=POOL_FACTOR)        # float [0,1]
            display = (pooled * 255).astype(np.uint8)

            ax = axes[row_i, col_i]
            ax.imshow(display, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=9)
            if col_i == 0:
                ax.set_ylabel(cls, fontsize=11, rotation=0, labelpad=35, va="center")

    fig.suptitle("Grayscale Method Visual Comparison\n(min-pooled 600×400)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "method_visual_comparison.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output B+C: accuracy evaluation loop
# ---------------------------------------------------------------------------

def run_accuracy_comparison(
    model,
    test_df,
    class_names: List[str],
    num_classes: int,
    class_map: dict,
    device: torch.device,
    out_dir: str,
) -> List[dict]:
    records = []

    for method in GRAY_METHODS:
        print(f"  Evaluating method: {method} ...")
        preprocessor = make_preprocessor(gray_method=method, pool_factor=POOL_FACTOR)

        if num_classes == 3:
            ds = _FibrinDataset3(test_df, PHOTOS_DIR, preprocessor, class_map)
        else:
            ds = FibrinDataset(test_df, PHOTOS_DIR, preprocessor, augment=False)

        loader = DataLoader(ds, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=False)

        # Collect preds + labels manually (evaluate_model uses CLASS_NAMES-length tensors)
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                preds = model(imgs.to(device)).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)
        preds_np  = torch.cat(all_preds).numpy().astype(np.int32)
        labels_np = torch.cat(all_labels).numpy().astype(np.int32)

        acc = float((preds_np == labels_np).mean())
        cm  = compute_cm(preds_np, labels_np, num_classes)
        f1  = _f1_from_cm(cm, class_names)

        rec = {"method": method, "overall_acc": acc}
        for cls in class_names:
            rec[f"{cls}_f1"] = f1[cls]
        records.append(rec)
        print(f"    accuracy: {acc:.4f}")

    # Save CSV
    csv_path = os.path.join(out_dir, "preprocessing_comparison_table.csv")
    f1_cols = [f"{cls}_f1" for cls in class_names]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "overall_acc"] + f1_cols)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {csv_path}")

    return records


# ---------------------------------------------------------------------------
# Output C: preprocessing_accuracy_bars.png
# ---------------------------------------------------------------------------

def plot_accuracy_bars(records: List[dict], class_names: List[str],
                       out_dir: str) -> None:
    methods   = [r["method"] for r in records]
    x         = np.arange(len(methods))
    n_series  = len(class_names) + 1          # classes + Overall
    width     = 0.7 / n_series

    fig, ax = plt.subplots(figsize=(13, 6))

    cls_colors = ["#4878CF", "#D65F5F", "#B47CC7", "#77BEDB", "#6ACC65",
                  "#F0A500", "#888888"]

    for i, cls in enumerate(class_names):
        vals = [r[f"{cls}_f1"] for r in records]
        offsets = x + (i - n_series / 2 + 0.5) * width
        ax.bar(offsets, vals, width, label=f"{cls} F1",
               color=cls_colors[i % len(cls_colors)], alpha=0.85)

    # Overall accuracy bars
    overall = [r["overall_acc"] for r in records]
    offsets = x + (n_series - 1 - n_series / 2 + 0.5) * width
    ax.bar(offsets, overall, width, label="Overall Acc",
           color="black", alpha=0.55, linestyle="--", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m].replace("\n", " ") for m in methods],
                       fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy / F1", fontsize=11)
    ax.set_title("Per-Method Evaluation (model trained on lab_l)\n"
                 "Note: alternative methods are out-of-distribution inputs",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.axhline(records[0]["overall_acc"], color="gray", linestyle=":",
               linewidth=1, label="_nolegend_")

    fig.tight_layout()
    path = os.path.join(out_dir, "preprocessing_accuracy_bars.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output D: text report
# ---------------------------------------------------------------------------

def write_report(records: List[dict], class_names: List[str],
                 model_type: str, n_test: int, out_dir: str) -> None:
    lines = [
        "=== Preprocessing Sensitivity Analysis ===",
        f"Model type:   {model_type} (trained on lab_l)",
        f"Test images:  {n_test}  (same images evaluated under each method)",
        "",
        "IMPORTANT: This model was trained exclusively on lab_l preprocessing.",
        "Evaluation with other methods tests robustness to preprocessing variation,",
        "NOT the optimal accuracy for each method. For a proper comparison, train",
        "a separate model for each method.",
        "",
        "── Method Ranking by Overall Accuracy ──",
    ]

    ranked = sorted(records, key=lambda r: r["overall_acc"], reverse=True)
    for i, r in enumerate(ranked, 1):
        tag = "  ← training method" if r["method"] == "lab_l" else ""
        lines.append(f"  {i}. {r['method']:<12}  {r['overall_acc']:.4f}{tag}")

    lines.append("")
    lines.append("── Per-Class F1 by Method ──")
    header = f"{'Method':<12}" + "".join(f"{cls:>10}" for cls in class_names)
    lines.append(header)
    for r in records:
        row = f"{r['method']:<12}" + "".join(
            f"{r[f'{cls}_f1']:>10.3f}" for cls in class_names
        )
        lines.append(row)

    # Most/least robust class
    base = {cls: records[0][f"{cls}_f1"] for cls in class_names}
    drops = {
        cls: base[cls] - min(r[f"{cls}_f1"] for r in records[1:])
        for cls in class_names
    }
    most_robust  = min(drops, key=drops.get)
    least_robust = max(drops, key=drops.get)
    base_acc = records[0]["overall_acc"]
    method_drops = {r["method"]: base_acc - r["overall_acc"] for r in records[1:]}
    most_robust_method = min(method_drops, key=method_drops.get)

    lines += [
        "",
        f"Most robust class:   {most_robust}  (max F1 drop = {drops[most_robust]:.3f})",
        f"Least robust class:  {least_robust}  (max F1 drop = {drops[least_robust]:.3f})",
        f"Most robust method:  {most_robust_method}  "
        f"(accuracy drop = {method_drops[most_robust_method]:.3f} vs lab_l)",
        "",
        "Extension: To measure optimal per-method accuracy, train a model for each",
        "method by changing GRAY_METHOD in train_5class.py and rerunning training.",
    ]

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "preprocessing_comparison_report.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pandas as pd  # local for _load_test_df return type annotation
    args  = _parse_args()
    device = torch.device("cpu")

    model_dir, num_classes, class_map = _MODEL_REGISTRY[args.model_type]
    model_path = os.path.join(model_dir, "best_model.pth")
    model = load_model(model_path, device=device, num_classes=num_classes)

    test_df, class_names, num_classes, class_map = _load_test_df(args.model_type)

    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "preprocessing")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Outputs → {out_dir}/\n")

    print("Visual comparison of preprocessing methods ...")
    plot_visual_comparison(test_df, class_names, out_dir)

    print("\nAccuracy evaluation for each method ...")
    records = run_accuracy_comparison(model, test_df, class_names,
                                      num_classes, class_map, device, out_dir)

    print("\nBar chart ...")
    plot_accuracy_bars(records, class_names, out_dir)

    write_report(records, class_names, args.model_type, len(test_df), out_dir)
    print("\nDone.")


if __name__ == "__main__":
    import pandas as pd
    main()
