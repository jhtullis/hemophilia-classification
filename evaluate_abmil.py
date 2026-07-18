"""
evaluate_abmil.py — Image-level evaluation of a trained FibrinABMIL model,
producing hard-vote, soft-vote, AND ABMIL predictions from a single forward
pass over the same fixed 35-patch grid used by evaluate_patch.py.

Deriving all three predictions from identical patches (rather than three
independently-sampled patch sets) is required for a fair, paired comparison
between aggregation methods — see compare_voting_methods.py for the paired
McNemar's significance test built on this script's output.

Soft-vote here averages RAW per-patch scores (cosine similarities or CE
logits) then argmaxes — the same convention evaluate_patch.py uses (NOT
patch_validate.py's mean-softmax-probability convention, which can disagree
on argmax). This makes evaluate_abmil.py's soft-vote column a reproducibility
check against the already-computed baseline in
models/<backbone>/analysis/<split>/per_image_predictions.csv.

Usage:
    python evaluate_abmil.py --model-dir models/abmil_mpatch_v0_f --split val
    python evaluate_abmil.py --model-dir models/abmil_mpatch_v0_f --split test

Outputs to <model-dir>/analysis/<split>/:
    per_image_predictions.csv
    attention_weights.csv
    attention_maps/<Exp_Type>/<idx>_<Experiment>.png
    confusion_matrix_{hard,soft,abmil}.png
    roc_curves_soft_vs_abmil.png
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Tuple

import cv2
import kornia.augmentation as K
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from abmil_model import FibrinABMIL
from data_loader import CLASS_MAP, CLASS_NAMES
from evaluate_patch import inference_grid_centers
from hemophilia_analysis import run_roc_analysis
from model_patch import FibrinPatchCNN
from patch_dataset import PAD, OVERSIZED, PATCH_SIZE, _pad_tensor, load_split_record
from preprocessing import make_preprocessor

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_abmil_model(model_dir: str, device: torch.device):
    """Load a trained FibrinABMIL from <model_dir>/best_model.pth, using the
    run metadata written by train_abmil.py to reconstruct the architecture."""
    config_path = os.path.join(model_dir, "abmil_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"abmil_config.json not found in {model_dir}")
    with open(config_path) as f:
        abmil_cfg = json.load(f)

    from analysis_utils import MODEL_REGISTRY
    backbone_model_type = abmil_cfg["backbone_model_type"]
    backbone_model_dir, num_classes, class_map = MODEL_REGISTRY[backbone_model_type]
    head_type = abmil_cfg["head_type"]

    backbone = FibrinPatchCNN(num_classes=num_classes, head_type=head_type)
    model = FibrinABMIL(
        backbone=backbone, num_classes=num_classes, head_type=head_type,
        freeze_backbone=True, attn_hidden_dim=abmil_cfg["attn_hidden_dim"],
    )

    weights_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No trained ABMIL model at {weights_path}.")
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()

    class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]
    return model, abmil_cfg, backbone_model_dir, num_classes, class_map, class_names


# ---------------------------------------------------------------------------
# Inference — single pass, all three predictions
# ---------------------------------------------------------------------------

def predict_image(
    model: FibrinABMIL,
    padded_tensor: torch.Tensor,
    centers: List[Tuple[int, int]],
    device: torch.device,
    center_crop: torch.nn.Module,
):
    """Run ONE forward pass over the fixed grid; derive hard-vote, soft-vote,
    and ABMIL predictions from the same patches.

    Returns:
        hard_pred, soft_pred, abmil_pred (int),
        soft_scores, abmil_scores (torch.Tensor, (C,), raw — not softmax),
        attn (torch.Tensor, (len(centers),))
    """
    half = OVERSIZED // 2
    patches = []
    for cy, cx in centers:
        cy_pad, cx_pad = cy + PAD, cx + PAD
        oversized = padded_tensor[
            :, cy_pad - half : cy_pad + half + 1, cx_pad - half : cx_pad + half + 1,
        ]
        patches.append(center_crop(oversized.unsqueeze(0).to(device)).squeeze(0))
    batch = torch.stack(patches)   # (35, 1, 200, 200)

    model.eval()
    with torch.no_grad():
        raw_scores = model.backbone(batch)                  # (35, C)
        embeddings = model.backbone.get_embeddings(batch)    # (35, 128)
        image_emb, attn = model.attention(embeddings)
        abmil_scores = model.classifier(image_emb.unsqueeze(0)).squeeze(0)   # (C,)

    hard_pred = int(Counter(raw_scores.argmax(1).tolist()).most_common(1)[0][0])
    soft_scores = raw_scores.mean(dim=0)
    soft_pred = int(soft_scores.argmax().item())
    abmil_pred = int(abmil_scores.argmax().item())

    return (hard_pred, soft_pred, abmil_pred,
            soft_scores.cpu(), abmil_scores.cpu(), attn.cpu())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, abmil_cfg, backbone_model_dir, num_classes, class_map, class_names = \
        load_abmil_model(args.model_dir, device)

    db_path = args.db or os.path.join(_DIR, "data", "endpoint10.db")
    train_df, val_df, test_df = load_split_record(backbone_model_dir, db_path)
    eval_df = val_df if args.split == "val" else test_df
    print(f"Evaluating {args.model_dir} on {args.split} set: {len(eval_df)} images")

    preprocessor = make_preprocessor()
    centers = inference_grid_centers()
    center_crop = K.CenterCrop(PATCH_SIZE)
    photo_dir = args.photo_dir or os.path.join(_DIR, "data", "photos")

    out_dir = os.path.join(args.model_dir, "analysis", args.split)
    attn_maps_dir = os.path.join(out_dir, "attention_maps")
    os.makedirs(out_dir, exist_ok=True)

    image_records = []
    attn_records = []

    for _, row in eval_df.iterrows():
        img_path = os.path.join(photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: cannot read {img_path} — skipping")
            continue
        tensor = preprocessor(img_bgr)          # (1, 400, 600)
        padded = _pad_tensor(tensor, PAD)

        true_label = class_map[row["Exp_Type"]]
        hard_pred, soft_pred, abmil_pred, soft_scores, abmil_scores, attn = \
            predict_image(model, padded, centers, device, center_crop)

        attn_np = attn.numpy()
        ent = float(-(np.clip(attn_np, 1e-12, None) * np.log(np.clip(attn_np, 1e-12, None))).sum())
        ent_norm = ent / np.log(len(centers)) if len(centers) > 1 else 0.0

        rec = {
            "idx": int(row["idx"]),
            "Experiment": row["Experiment"],
            "Exp_Type": row["Exp_Type"],
            "true_label": true_label,
            "hard_vote_pred": hard_pred,
            "hard_vote_correct": int(hard_pred == true_label),
            "soft_vote_pred": soft_pred,
            "soft_vote_correct": int(soft_pred == true_label),
            "abmil_pred": abmil_pred,
            "abmil_correct": int(abmil_pred == true_label),
            "attn_entropy": round(ent, 6),
            "attn_entropy_normalized": round(ent_norm, 6),
            "attn_max_weight": round(float(attn_np.max()), 6),
        }
        for i, cls in enumerate(class_names):
            rec[f"soft_vote_score_{cls}"] = round(soft_scores[i].item(), 6)
            rec[f"abmil_score_{cls}"] = round(abmil_scores[i].item(), 6)
        image_records.append(rec)

        for pi, (cy, cx) in enumerate(centers):
            attn_records.append({
                "idx": int(row["idx"]), "patch_i": pi,
                "center_y": cy, "center_x": cx,
                "attn_weight": round(float(attn_np[pi]), 6),
            })

        if args.save_attention_maps:
            os.makedirs(os.path.join(attn_maps_dir, row["Exp_Type"]), exist_ok=True)
            fig = _draw_attention_overlay(
                tensor, centers, attn_np, true_label, row, class_names[abmil_pred], class_names
            )
            fname = f"{int(row['idx']):04d}_{row['Experiment']}.png"
            fig.savefig(os.path.join(attn_maps_dir, row["Exp_Type"], fname),
                        dpi=80, bbox_inches="tight")
            plt.close(fig)

    results_df = pd.DataFrame(image_records)
    attn_df = pd.DataFrame(attn_records)

    for method in ("hard_vote", "soft_vote", "abmil"):
        acc = results_df[f"{method}_correct"].mean()
        print(f"\n{method} overall accuracy ({args.split}): {acc:.3f}")
        for cls in class_names:
            cls_df = results_df[results_df["Exp_Type"] == cls]
            if len(cls_df):
                print(f"  {cls}: {cls_df[f'{method}_correct'].mean():.3f}  ({len(cls_df)} images)")

    csv_path = os.path.join(out_dir, "per_image_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    attn_csv_path = os.path.join(out_dir, "attention_weights.csv")
    attn_df.to_csv(attn_csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")
    print(f"Saved attention weights to {attn_csv_path}")

    for method, pred_col in (("hard", "hard_vote_pred"), ("soft", "soft_vote_pred"), ("abmil", "abmil_pred")):
        _plot_confusion_matrix(
            results_df["true_label"].values, results_df[pred_col].values,
            class_names, f"{method}-vote — {os.path.basename(args.model_dir)}",
            os.path.join(out_dir, f"confusion_matrix_{method}.png"),
        )

    _plot_roc_soft_vs_abmil(results_df, class_names, class_map, out_dir,
                             os.path.basename(args.model_dir))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    true_labels: np.ndarray, pred_labels: np.ndarray,
    class_names: List[str], title: str, path: str,
) -> None:
    n = len(class_names)
    cm_arr = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm_arr[t, p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _plot_roc_soft_vs_abmil(
    results_df: pd.DataFrame, class_names: List[str], class_map: dict,
    out_dir: str, model_name: str,
) -> None:
    true_labels = results_df["true_label"].values

    soft_raw = np.stack([results_df[f"soft_vote_score_{c}"].values for c in class_names], axis=1)
    abmil_raw = np.stack([results_df[f"abmil_score_{c}"].values for c in class_names], axis=1)
    soft_probs = _softmax(soft_raw)
    abmil_probs = _softmax(abmil_raw)

    fig, ax = plt.subplots(figsize=(8, 6))
    run_roc_analysis(true_labels, soft_probs, analysis_classes=class_names,
                      class_map=class_map, ax=ax, label_prefix="soft-vote ", linestyle="-")
    run_roc_analysis(true_labels, abmil_probs, analysis_classes=class_names,
                      class_map=class_map, ax=ax, label_prefix="ABMIL ", linestyle="--")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"OVR ROC — soft-vote vs. ABMIL — {model_name}")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curves_soft_vs_abmil.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _softmax(scores: np.ndarray) -> np.ndarray:
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    return e / e.sum(axis=1, keepdims=True)


def _draw_attention_overlay(
    tensor: torch.Tensor, centers: List[Tuple[int, int]], attn: np.ndarray,
    true_label: int, row: pd.Series, pred_class: str, class_names: List[str],
) -> plt.Figure:
    """Per-patch rectangle overlay colored continuously by attention weight
    (viridis, normalized per-image by max weight). Adapted from
    patch_validate.draw_annotated_image's rectangle-with-alpha technique,
    which is binary correct/incorrect only — this is a new sibling function."""
    img_np = tensor.squeeze(0).numpy()
    true_class = class_names[true_label]

    fig, ax = plt.subplots(figsize=(12, 8.4))
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)

    max_w = max(float(attn.max()), 1e-12)
    colormap = matplotlib.colormaps["viridis"]
    for i, (cy, cx) in enumerate(centers):
        weight_norm = float(attn[i]) / max_w
        color = colormap(weight_norm)
        rect = mpatches.Rectangle(
            (cx - PATCH_SIZE // 2, cy - PATCH_SIZE // 2),
            PATCH_SIZE, PATCH_SIZE,
            linewidth=0.5, edgecolor=color, facecolor=color,
            alpha=0.15 + 0.5 * weight_norm,
        )
        ax.add_patch(rect)

    ok = (pred_class == true_class)
    title = (
        f"{int(row['idx']):04d}  |  true: {true_class}  |  "
        f"{row['Experiment']}  |  ABMIL pred: {pred_class}"
    )
    ax.set_title(title, fontsize=9, color=("black" if ok else "red"), pad=5)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Image-level ABMIL evaluation with paired hard-vote/soft-vote "
                     "baselines from a single forward pass over a fixed patch grid."
    )
    p.add_argument("--model-dir", required=True,
                    help="Trained ABMIL model dir (e.g. models/abmil_mpatch_v0_f).")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--db", default=None)
    p.add_argument("--photo-dir", default=None)
    p.add_argument("--save-attention-maps", action="store_true", default=True)
    p.add_argument("--no-attention-maps", dest="save_attention_maps", action="store_false")
    evaluate(p.parse_args())
