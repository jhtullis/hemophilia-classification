"""
evaluate_cosine.py — Evaluation script for cosine-head FibrinCNN models.

Replaces evaluate.py for 5class_hpc_v0 and any future cosine-loss model types.

Section A (always):
  - Accuracy, per-class precision / recall / F1, confusion matrix
  - OVR ROC curves using raw cosine similarities as scores (thresholds ∈ [−1, 1])
  - Silhouette score in cosine distance space
  - Intra / inter-class similarity bar chart
  - UMAP projection of 128-dim normalized feature vectors
  - metrics_summary.txt

Section B (--calibrate flag only):
  - Post-hoc Platt scaling: find tau_cal minimising CE on val set
  - Calibrated CE loss, ECE, reliability diagram
  Note: tau_cal is NOT a model parameter — it is a post-hoc evaluation tool only.

Usage:
    python evaluate_cosine.py --model-type 5class_hpc_v0
    python evaluate_cosine.py --model-type 5class_hpc_v0 --calibrate
    python evaluate_cosine.py --model-type 5class_hpc_v0 --db path/to/endpoint10.db

Outputs: models/<type>/analysis/cosine_eval/
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import (auc, roc_curve, silhouette_score,
                              confusion_matrix)
from umap import UMAP

from analysis_utils import PHOTOS_DIR, get_val_split, load_model_from_registry
from data_loader import CLASS_NAMES
from preprocessing import make_preprocessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR    = os.path.dirname(__file__)
DB_PATH = os.path.join(_DIR, "data", "endpoint10.db")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_cosine_outputs(
    model,
    val_df,
    photos_dir: str,
    device: torch.device,
    preprocessor,
    class_map: dict,
    class_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect similarities, normalized features, and labels.

    Returns:
        sims_all:     (N, C) cosine similarities
        feats_norm:   (N, D) L2-normalized feature vectors (before classifier)
        labels_all:   (N,) integer class labels
    """
    import cv2
    from data_loader import FibrinDataset
    from torch.utils.data import DataLoader

    num_workers = min(4, os.cpu_count() or 1)
    val_ds = FibrinDataset(val_df, photos_dir, preprocessor,
                           augment=False, preload=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=(num_workers > 0))

    model.eval()
    sims_list, feats_list, label_list = [], [], []

    # Hook captures the 256-dim global_pool output (before classifier)
    # NOTE: if model.global_pool is renamed this hook must be updated.
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        captured["h"] = output.detach().cpu().flatten(1)   # (B, 256)

    handle = model.global_pool.register_forward_hook(_hook)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            sims = model(images).cpu()
            sims_list.append(sims)
            label_list.append(labels)
            feats_list.append(captured["h"])

    handle.remove()

    sims_all   = torch.cat(sims_list,  dim=0).numpy()
    feats_all  = torch.cat(feats_list, dim=0)
    feats_norm = F.normalize(feats_all, p=2, dim=1).numpy()
    labels_all = torch.cat(label_list, dim=0).numpy()

    return sims_all, feats_norm, labels_all


# ---------------------------------------------------------------------------
# Section A helpers
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    sims_all: np.ndarray, labels_all: np.ndarray, class_names: List[str]
) -> dict:
    """Compute accuracy and per-class precision/recall/F1 from argmax."""
    preds = sims_all.argmax(axis=1)
    correct = (preds == labels_all).sum()
    n = len(labels_all)
    accuracy = correct / n

    metrics = {"accuracy": float(accuracy), "n": n, "per_class": {}}
    for c, name in enumerate(class_names):
        tp = int(((preds == c) & (labels_all == c)).sum())
        fp = int(((preds == c) & (labels_all != c)).sum())
        fn = int(((preds != c) & (labels_all == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics["per_class"][name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   int((labels_all == c).sum()),
        }
    return metrics


def plot_confusion_matrix(sims_all, labels_all, class_names, output_dir):
    preds = sims_all.argmax(axis=1)
    cm = confusion_matrix(labels_all, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — cosine model")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def compute_cosine_roc(sims_all, labels_all, class_names, output_dir) -> dict:
    """OVR ROC curves using raw cosine similarities as scores (thresholds ∈ [−1, 1])."""
    results = {}
    fig, ax = plt.subplots(figsize=(7, 6))

    for c, name in enumerate(class_names):
        y_true  = (labels_all == c).astype(int)
        y_score = sims_all[:, c]

        fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        # Sanity check: thresholds are cosine similarities, always in [-1, 1]
        assert thresholds.min() >= -1.0 - 1e-6, \
            f"Threshold below -1 for class {name}: {thresholds.min()}"
        assert thresholds.max() <=  1.0 + 1e-6, \
            f"Threshold above +1 for class {name}: {thresholds.max()}"

        results[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(), "auc": roc_auc,
        }
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("OVR ROC — cosine similarity scores (thresholds ∈ [−1, 1])")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(output_dir, "roc_data.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_intra_inter(feats_norm, labels_all, class_names, output_dir):
    gram = feats_norm @ feats_norm.T
    intra_means, inter_means = [], []

    for c in range(len(class_names)):
        mask_c = labels_all == c
        nc = mask_c.sum()
        block = gram[np.ix_(mask_c, mask_c)]
        intra = float((block.sum() - nc) / (nc * (nc - 1))) if nc > 1 else 0.0
        inter = float(gram[np.ix_(mask_c, ~mask_c)].mean())
        intra_means.append(intra)
        inter_means.append(inter)

    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, intra_means, width, label="Intra-class")
    ax.bar(x + width / 2, inter_means, width, label="Inter-class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Intra vs. inter-class cosine similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "intra_inter_similarity.png"), dpi=150)
    plt.close(fig)


def plot_umap(feats_norm, labels_all, class_names, output_dir):
    reducer = UMAP(n_components=2, random_state=42, metric="cosine")
    embedding = reducer.fit_transform(feats_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for c, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels_all == c
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   label=name, color=color, alpha=0.7, s=20)
    ax.set_title("UMAP of 128-dim ĥ vectors (cosine distance)")
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "embedding_umap.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section B — Platt scaling / calibration
# ---------------------------------------------------------------------------

def fit_temperature(sims_all: np.ndarray, labels_all: np.ndarray) -> float:
    """Post-hoc Platt scaling: find tau_cal minimising CE on val set.

    tau_cal is a post-hoc scalar for evaluation only — it is NOT a model
    parameter and is never used during training.  Deviation of tau_cal from
    1.0 indicates that raw cosine similarities are miscalibrated as probability
    substitutes.
    """
    sims_t   = torch.tensor(sims_all,   dtype=torch.float32)
    labels_t = torch.tensor(labels_all, dtype=torch.long)

    result = minimize_scalar(
        lambda tau: F.cross_entropy(sims_t / tau, labels_t).item(),
        bounds=(0.01, 10.0), method="bounded",
    )
    return float(result.x)


def compute_ece(probs: np.ndarray, labels_all: np.ndarray, n_bins: int = 10) -> float:
    preds   = probs.argmax(axis=1)
    confs   = probs.max(axis=1)
    correct = (preds == labels_all).astype(float)

    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confs[mask].mean()
        ece += mask.mean() * abs(acc_bin - conf_bin)
    return float(ece)


def plot_reliability(probs, labels_all, output_dir, n_bins=10):
    preds   = probs.argmax(axis=1)
    confs   = probs.max(axis=1)
    correct = (preds == labels_all).astype(float)

    bins   = np.linspace(0, 1, n_bins + 1)
    accs, conf_means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        accs.append(correct[mask].mean())
        conf_means.append(confs[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(conf_means, accs, "o-", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (calibrated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reliability_diagram.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a cosine-head FibrinCNN model."
    )
    p.add_argument("--model-type", default="5class_hpc_v0",
                   help="Model type from MODEL_REGISTRY (default: 5class_hpc_v0).")
    p.add_argument("--calibrate", action="store_true",
                   help="Also run Platt scaling / calibration analysis (Section B).")
    p.add_argument("--db", default=None, metavar="PATH",
                   help="Path to SQLite database.")
    args = p.parse_args()

    db_path = args.db or DB_PATH
    device  = torch.device("cpu")

    model, model_dir, num_classes, class_map, class_names = \
        load_model_from_registry(args.model_type, device)

    val_df       = get_val_split(args.model_type, db_path=db_path)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    output_dir = os.path.join(model_dir, "analysis", "cosine_eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running cosine evaluation for {args.model_type} …")
    print(f"Output → {output_dir}")

    # --- Inference ---
    sims_all, feats_norm, labels_all = collect_cosine_outputs(
        model, val_df, PHOTOS_DIR, device, preprocessor, class_map, class_names
    )
    print(f"Collected {len(labels_all)} validation predictions.")

    # ── Section A ────────────────────────────────────────────────────────────

    # 1. Per-class metrics
    clf_metrics = compute_per_class_metrics(sims_all, labels_all, class_names)
    print(f"\nOverall accuracy: {clf_metrics['accuracy']:.4f} ({clf_metrics['n']} images)")

    # 2. Confusion matrix
    plot_confusion_matrix(sims_all, labels_all, class_names, output_dir)
    print("Confusion matrix saved.")

    # 3. OVR ROC curves
    roc_results = compute_cosine_roc(sims_all, labels_all, class_names, output_dir)
    mean_auc = np.mean([v["auc"] for v in roc_results.values()])
    print(f"ROC curves saved. Mean AUC: {mean_auc:.4f}")

    # 4. Silhouette
    sil_score = float(silhouette_score(feats_norm, labels_all, metric="cosine"))
    print(f"Silhouette score (cosine): {sil_score:.4f}")

    # 5. Intra/inter similarity
    plot_intra_inter(feats_norm, labels_all, class_names, output_dir)
    print("Intra/inter similarity chart saved.")

    # 6. UMAP
    plot_umap(feats_norm, labels_all, class_names, output_dir)
    print("UMAP embedding saved.")

    # 7. metrics_summary.txt
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Validation images: {clf_metrics['n']}\n")
        f.write(f"Overall accuracy:  {clf_metrics['accuracy']:.4f}\n")
        f.write(f"Mean OVR AUC:      {mean_auc:.4f}\n")
        f.write(f"Silhouette score:  {sil_score:.4f}\n\n")
        f.write(f"{'Class':<8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
                f"{'Support':>8}  {'AUC':>8}\n")
        f.write("-" * 58 + "\n")
        for name in class_names:
            pc  = clf_metrics["per_class"][name]
            auc_val = roc_results[name]["auc"]
            f.write(f"{name:<8}  {pc['precision']:>10.4f}  {pc['recall']:>8.4f}  "
                    f"{pc['f1']:>8.4f}  {pc['support']:>8d}  {auc_val:>8.4f}\n")
    print(f"Metrics summary saved → {summary_path}")

    # ── Section B — calibration ───────────────────────────────────────────────

    if args.calibrate:
        cal_dir = os.path.join(output_dir, "calibration")
        os.makedirs(cal_dir, exist_ok=True)

        tau_cal = fit_temperature(sims_all, labels_all)
        print(f"\nPlatt scaling: tau_cal = {tau_cal:.4f}  "
              f"(deviation from 1.0 indicates miscalibration)")

        sims_t  = torch.tensor(sims_all, dtype=torch.float32)
        probs   = torch.softmax(sims_t / tau_cal, dim=1).numpy()
        labels_t = torch.tensor(labels_all, dtype=torch.long)
        cal_ce  = float(F.cross_entropy(
            torch.tensor(sims_all / tau_cal), labels_t
        ).item())
        ece     = compute_ece(probs, labels_all)

        with open(os.path.join(cal_dir, "tau_calibrated.txt"), "w") as f:
            f.write(f"tau_cal = {tau_cal:.6f}\n")
            f.write("Note: post-hoc only — not a model parameter.\n")
        with open(os.path.join(cal_dir, "calibrated_ce.txt"), "w") as f:
            f.write(f"Calibrated CE loss: {cal_ce:.6f}\n")
            f.write(f"ECE:                {ece:.6f}\n")
        plot_reliability(probs, labels_all, cal_dir)
        print(f"Calibration outputs saved → {cal_dir}")
        print(f"  Calibrated CE: {cal_ce:.4f}   ECE: {ece:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
