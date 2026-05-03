"""
gradcam.py — Grad-CAM class saliency maps for FibrinCNN (Module 6).

Produces class-specific spatial heatmaps showing which image regions
drive each classification decision.

Target layer: model.features[14] (ReLU after Conv4, spatial 75×50 — higher
resolution than the post-MaxPool output at 37×25).

Usage:
    python gradcam.py [--model-type TYPE] [--output-dir DIR]
                      [--misclassified-only]

Outputs (saved to <model_dir>/analysis/gradcam/ by default):
    class_representatives/<class>_gradcam_top3.png  — top-3 correct per class
    misclassified/<idx>_true-<true>_pred-<pred>.png — all misclassified images
    class_comparison_grid.png                       — one row per class overview
    gradcam_summary.txt                             — spatial centroid statistics
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import (CLASS_MAP, CLASS_NAMES, filter_classes,
                         load_split_from_record)
from hemophilia_analysis import _MODEL_REGISTRY, _draw_banner, load_model
from model import FibrinCNN
from preprocessing import ensure_landscape, make_preprocessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRADCAM_LAYER_IDX = 14      # model.features[14] = ReLU after Conv4, before MaxPool4
                             # spatial size: (256, 75, 50)
DB_PATH    = os.path.join(os.path.dirname(__file__), "data", "endpoint10.db")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
HEMOPHILIA_CLASSES = ["F08D", "F09D", "F11D"]
CLASS_MAP_3 = {"F08D": 0, "F09D": 1, "F11D": 2}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM saliency maps for FibrinCNN.")
    p.add_argument("--model-type", default="5class",
                   choices=list(_MODEL_REGISTRY.keys()),
                   help="Which trained model to use (default: 5class).")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Override output directory.")
    p.add_argument("--misclassified-only", action="store_true",
                   help="Only generate Grad-CAM for misclassified images.")
    p.add_argument("--all-overlays", action="store_true",
                   help="Generate Grad-CAM overlays on full-resolution images "
                        "for every test image (saved to <out_dir>/all_overlays/).")
    p.add_argument("--db", default=DB_PATH, metavar="PATH",
                   help="Path to SQLite database (default: data/endpoint10.db).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GradCAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """Compute Grad-CAM heatmaps for FibrinCNN.

    Hooks model.features[target_layer_idx] to capture activations during
    forward pass and uses retain_grad() to capture gradients on backward pass.

    Usage:
        gcam = GradCAM(model)
        heatmap = gcam.compute(tensor, target_class=2)  # ndarray (75, 50)
        overlay = gcam.overlay(heatmap, tensor)         # BGR ndarray (400, 600, 3)
        gcam.remove_hooks()
    """

    def __init__(self, model: FibrinCNN,
                 target_layer_idx: int = GRADCAM_LAYER_IDX):
        self.model = model
        self._activation: Optional[torch.Tensor] = None

        def _save_activation(module, inp, out):
            self._activation = out

        self._fwd_hook = model.features[target_layer_idx].register_forward_hook(
            _save_activation
        )

    def compute(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Compute Grad-CAM heatmap for one image and target class.

        Args:
            tensor:       Shape (1, 1, 400, 600). Must be on same device as model.
            target_class: Integer class index.

        Returns:
            heatmap: np.ndarray shape (75, 50), values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass — hook fires, self._activation set
        logits = self.model(tensor)

        # Allow gradient accumulation on the (non-leaf) activation tensor
        self._activation.retain_grad()

        # Backpropagate only the target class logit
        logits[0, target_class].backward()

        grad = self._activation.grad     # (1, 256, 75, 50)
        act  = self._activation.detach() # (1, 256, 75, 50)

        # Channel-importance weights: global-average-pool of gradients
        alpha = grad[0].mean(dim=(1, 2))  # (256,)

        # Weighted combination of activation maps
        cam = (alpha[:, None, None] * act[0]).sum(dim=0)  # (75, 50)

        # ReLU: keep only positive contributions
        cam = F.relu(cam).cpu().detach().numpy()

        # Normalize to [0, 1]
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val

        return cam

    def overlay(self, heatmap: np.ndarray, tensor: torch.Tensor,
                alpha: float = 0.5) -> np.ndarray:
        """Overlay Grad-CAM heatmap on the preprocessed grayscale image.

        Args:
            heatmap: (75, 50) array in [0, 1].
            tensor:  (1, 1, 400, 600) preprocessed image tensor.
            alpha:   Colormap blend weight (default 0.5).

        Returns:
            BGR ndarray of shape (400, 600, 3), uint8.
        """
        # Upsample heatmap to image resolution; cv2.resize takes (width, height)
        h_up = cv2.resize(heatmap, (tensor.shape[-1], tensor.shape[-2]),
                          interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap((h_up * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)  # (400,600,3) BGR

        gray = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(gray_bgr, 1 - alpha, colormap, alpha, 0)

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _load_tensor(idx: int, photos_dir: str, preprocessor,
                 device: torch.device) -> torch.Tensor:
    img_bgr = cv2.imread(os.path.join(photos_dir, f"{idx:04d}.JPG"))
    if img_bgr is None:
        raise FileNotFoundError(f"Image {idx:04d}.JPG not found")
    return preprocessor(img_bgr).unsqueeze(0).to(device)


def _collect_results(model: FibrinCNN, test_df, photos_dir: str,
                     device: torch.device, preprocessor,
                     class_map: dict) -> List[dict]:
    """Inference pass — returns list of per-image result dicts."""
    rows = []
    model.eval()
    with torch.no_grad():
        for _, row in test_df.iterrows():
            tensor = _load_tensor(int(row["idx"]), photos_dir,
                                  preprocessor, device)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            tl = class_map[row["Exp_Type"]]
            pl = int(probs.argmax())
            rows.append({
                "idx":        int(row["idx"]),
                "Exp_Type":   row["Exp_Type"],
                "Experiment": row["Experiment"],
                "true_label": tl,
                "pred_label": pl,
                "correct":    tl == pl,
                "confidence": float(probs.max()),
                "probs":      probs,
            })
    return rows


# ---------------------------------------------------------------------------
# Output A: class_representatives/<class>_gradcam_top3.png
# ---------------------------------------------------------------------------

def plot_class_representatives(gcam: GradCAM, results: List[dict],
                                class_names: List[str],
                                inv_class_map: Dict[int, str],
                                photos_dir: str, device: torch.device,
                                preprocessor, out_dir: str) -> None:
    for cls in class_names:
        correct = [r for r in results
                   if r["Exp_Type"] == cls and r["correct"]]
        correct.sort(key=lambda r: r["confidence"], reverse=True)
        top = correct[:3]
        if not top:
            print(f"  {cls}: no correct predictions — skipping.")
            continue

        n = len(top)
        fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n), squeeze=False)

        for rank_i, r in enumerate(top):
            tensor = _load_tensor(r["idx"], photos_dir, preprocessor, device)
            heatmap = gcam.compute(tensor, r["true_label"])
            overlay_bgr = gcam.overlay(heatmap, tensor)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            gray = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

            ax_g = axes[rank_i, 0]
            ax_o = axes[rank_i, 1]
            ax_g.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax_o.imshow(overlay_rgb)
            ax_g.axis("off")
            ax_o.axis("off")

            row_label = (f"#{rank_i+1}  idx={r['idx']:04d}"
                         f"  conf={r['confidence']:.3f}")
            ax_g.set_title(row_label, fontsize=9, loc="left")

            prob_str = "  ".join(
                f"{inv_class_map[i]}={r['probs'][i]:.3f}"
                for i in range(len(r["probs"]))
            )
            ax_o.set_title(f"Probs: {prob_str}", fontsize=7, loc="left")

        axes[0, 0].set_ylabel("Preprocessed", fontsize=10)
        axes[0, 1].set_ylabel("Grad-CAM", fontsize=10)
        fig.suptitle(f"Grad-CAM: {cls} — top-{n} confident correct predictions",
                     fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{cls}_gradcam_top3.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Output B: misclassified/<idx>_true-<true>_pred-<pred>.png
# ---------------------------------------------------------------------------

def plot_misclassified(gcam: GradCAM, results: List[dict],
                       inv_class_map: Dict[int, str],
                       photos_dir: str, device: torch.device,
                       preprocessor, out_dir: str) -> None:
    wrong = [r for r in results if not r["correct"]]
    if not wrong:
        print("  No misclassified images.")
        return

    for r in wrong:
        tensor = _load_tensor(r["idx"], photos_dir, preprocessor, device)
        gray = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

        gcam.model.zero_grad()
        cam_true = gcam.compute(tensor, r["true_label"])
        overlay_true = cv2.cvtColor(gcam.overlay(cam_true, tensor),
                                    cv2.COLOR_BGR2RGB)

        gcam.model.zero_grad()
        cam_pred = gcam.compute(tensor, r["pred_label"])
        overlay_pred = cv2.cvtColor(gcam.overlay(cam_pred, tensor),
                                    cv2.COLOR_BGR2RGB)

        true_cls = inv_class_map[r["true_label"]]
        pred_cls = inv_class_map[r["pred_label"]]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Input\n(TRUE: {true_cls})", fontsize=10)
        axes[1].imshow(overlay_true)
        axes[1].set_title(f"Grad-CAM: {true_cls}\n(true class)", fontsize=10)
        axes[2].imshow(overlay_pred)
        axes[2].set_title(f"Grad-CAM: {pred_cls}\n(predicted class)", fontsize=10)
        for ax in axes:
            ax.axis("off")

        prob_str = "  ".join(
            f"{inv_class_map[i]}={r['probs'][i]:.3f}"
            for i in range(len(r["probs"]))
        )
        fig.suptitle(
            f"MISCLASSIFIED  idx={r['idx']:04d}  TRUE={true_cls} → PRED={pred_cls}",
            fontsize=10,
        )
        fig.text(0.5, 0.04, f"Probs: {prob_str}",
                 ha="center", fontsize=9, color="#444444")
        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        fname = f"{r['idx']:04d}_true-{true_cls}_pred-{pred_cls}.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(wrong)} misclassified image(s) to {out_dir}/")


# ---------------------------------------------------------------------------
# Output C: class_comparison_grid.png
# ---------------------------------------------------------------------------

def plot_class_comparison_grid(gcam: GradCAM, results: List[dict],
                                class_names: List[str],
                                inv_class_map: Dict[int, str],
                                photos_dir: str, device: torch.device,
                                preprocessor, out_dir: str) -> None:
    n = len(class_names)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n), squeeze=False)

    for row_i, cls in enumerate(class_names):
        correct = [r for r in results
                   if r["Exp_Type"] == cls and r["correct"]]
        if not correct:
            correct = [r for r in results if r["Exp_Type"] == cls]
        if not correct:
            continue
        r = max(correct, key=lambda x: x["confidence"])

        tensor = _load_tensor(r["idx"], photos_dir, preprocessor, device)
        gray = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        heatmap = gcam.compute(tensor, r["pred_label"])
        overlay_rgb = cv2.cvtColor(gcam.overlay(heatmap, tensor),
                                   cv2.COLOR_BGR2RGB)

        axes[row_i, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[row_i, 1].imshow(overlay_rgb)
        for col_i in range(2):
            axes[row_i, col_i].axis("off")
        axes[row_i, 0].set_ylabel(cls, fontsize=13, rotation=0,
                                   labelpad=40, va="center")

    axes[0, 0].set_title("Preprocessed Image", fontsize=11)
    axes[0, 1].set_title("Grad-CAM (predicted class)", fontsize=11)
    fig.suptitle("Grad-CAM Overview — One Representative Image Per Class",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "class_comparison_grid.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Output D: gradcam_summary.txt
# ---------------------------------------------------------------------------

def write_summary(gcam: GradCAM, results: List[dict],
                  class_names: List[str], inv_class_map: Dict[int, str],
                  photos_dir: str, device: torch.device,
                  preprocessor, model_type: str, out_dir: str) -> None:
    lines = [
        "=== Grad-CAM Spatial Attention Summary ===",
        f"Model type:   {model_type}",
        f"Layer:        features[{GRADCAM_LAYER_IDX}] (ReLU after Conv4, spatial 75×50)",
        "",
        "Centroid position normalized to [0,1].",
        "x: 0=left, 1=right.   y: 0=top, 1=bottom.",
        "A network with no spatial bias attends around centroid (0.5, 0.5).",
        "",
        f"{'Class':<8} {'N_correct':>10} {'Centroid_X':>12} {'Centroid_Y':>12} {'Region'}",
    ]

    # heatmap shape is (50, 75): H=50 rows (y), W=75 cols (x)
    h_idx, w_idx = np.mgrid[0:50, 0:75]

    for cls in class_names:
        correct = [r for r in results if r["Exp_Type"] == cls and r["correct"]]
        if not correct:
            lines.append(f"{cls:<8} {'0':>10} {'N/A':>12} {'N/A':>12}")
            continue

        cx_list, cy_list = [], []
        for r in correct:
            tensor = _load_tensor(r["idx"], photos_dir, preprocessor, device)
            heatmap = gcam.compute(tensor, r["true_label"])
            total = heatmap.sum() + 1e-8
            cy_list.append((heatmap * h_idx).sum() / total / 49.0)
            cx_list.append((heatmap * w_idx).sum() / total / 74.0)

        cx = float(np.mean(cx_list))
        cy = float(np.mean(cy_list))

        # Qualitative region description
        h_desc = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "center")
        v_desc = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
        region = f"{h_desc}-{v_desc}"

        lines.append(f"{cls:<8} {len(correct):>10} {cx:>12.3f} {cy:>12.3f}  {region}")

    n_total     = len(results)
    n_incorrect = sum(1 for r in results if not r["correct"])
    lines += [
        "",
        f"Misclassified images: {n_incorrect} / {n_total}",
        f"Grad-CAM outputs saved to: {out_dir}/",
    ]

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "gradcam_summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n  Saved: {path}")


# ---------------------------------------------------------------------------
# Output E: all_overlays/<idx>.JPG  — full-resolution GradCAM + banner
# ---------------------------------------------------------------------------

def plot_all_overlays(gcam: GradCAM, results: List[dict],
                      inv_class_map: Dict[int, str],
                      photos_dir: str, device: torch.device,
                      preprocessor, out_dir: str) -> None:
    """Write GradCAM overlay images for every test image at full resolution.

    Each output JPEG shows the original image with the Grad-CAM heatmap blended
    on top, plus the same annotation banner used by hemophilia_analysis --annotate
    (TRUE / PRED class, per-class probabilities, colour-coded correct/wrong).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"   Output folder: {os.path.abspath(out_dir)}/\n")

    n = len(results)
    for i, r in enumerate(results, 1):
        idx      = r["idx"]
        true_cls = inv_class_map[r["true_label"]]
        pred_cls = inv_class_map[r["pred_label"]]

        img_bgr = cv2.imread(os.path.join(photos_dir, f"{idx:04d}.JPG"))
        if img_bgr is None:
            print(f"   [{i:>4}/{n}] {idx:04d}.JPG  SKIPPED (not found)")
            continue
        img_bgr = ensure_landscape(img_bgr)

        # Grad-CAM heatmap on preprocessed tensor (predicted class)
        tensor = _load_tensor(idx, photos_dir, preprocessor, device)
        gcam.model.zero_grad()
        heatmap = gcam.compute(tensor, r["pred_label"])

        # Upsample heatmap to original image resolution and blend
        H, W = img_bgr.shape[:2]
        h_up     = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap((h_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended  = cv2.addWeighted(img_bgr, 0.5, colormap, 0.5, 0)

        # Add annotation banner (matches annotated_test style)
        annotated = _draw_banner(blended, true_cls, pred_cls,
                                 r["probs"], inv_class_map)

        fname = f"{idx:04d}.JPG"
        cv2.imwrite(os.path.join(out_dir, fname), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        status = "OK" if r["correct"] else "MISMATCH"
        print(f"   [{i:>4}/{n}] {fname}  true={true_cls}  pred={pred_cls}  {status}")

    print(f"\n   Done — {n}/{n} images written.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")

    model_dir, num_classes, class_map = _MODEL_REGISTRY[args.model_type]
    model_path = os.path.join(model_dir, "best_model.pth")
    model = load_model(model_path, device=device, num_classes=num_classes)
    class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]
    inv_class_map = {v: k for k, v in class_map.items()}

    record_path = os.path.join("models", "5class", "train_record.json")
    _, val_df = load_split_from_record(record_path, args.db)
    if num_classes == 3:
        val_df = filter_classes(val_df, HEMOPHILIA_CLASSES)

    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    out_dir = args.output_dir or os.path.join(model_dir, "analysis", "gradcam")
    rep_dir  = os.path.join(out_dir, "class_representatives")
    misc_dir = os.path.join(out_dir, "misclassified")
    for d in [out_dir, rep_dir, misc_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"Collecting predictions for {len(val_df)} validation images ...")
    results = _collect_results(model, val_df, PHOTOS_DIR, device,
                                preprocessor, class_map)
    n_correct   = sum(1 for r in results if r["correct"])
    n_incorrect = len(results) - n_correct
    print(f"  Correct: {n_correct}  Incorrect: {n_incorrect}")

    gcam = GradCAM(model, target_layer_idx=GRADCAM_LAYER_IDX)

    if not args.misclassified_only:
        print("\nClass representative Grad-CAMs ...")
        plot_class_representatives(gcam, results, class_names, inv_class_map,
                                   PHOTOS_DIR, device, preprocessor, rep_dir)

        print("\nClass comparison grid ...")
        plot_class_comparison_grid(gcam, results, class_names, inv_class_map,
                                   PHOTOS_DIR, device, preprocessor, out_dir)

    print(f"\nMisclassified image Grad-CAMs ({n_incorrect} images) ...")
    plot_misclassified(gcam, results, inv_class_map,
                       PHOTOS_DIR, device, preprocessor, misc_dir)

    print("\nSpatial attention summary ...")
    write_summary(gcam, results, class_names, inv_class_map,
                  PHOTOS_DIR, device, preprocessor, args.model_type, out_dir)

    if args.all_overlays:
        overlays_dir = os.path.join(out_dir, "all_overlays")
        print(f"\nFull-resolution Grad-CAM overlays ({len(results)} images) ...")
        plot_all_overlays(gcam, results, inv_class_map,
                          PHOTOS_DIR, device, preprocessor, overlays_dir)

    gcam.remove_hooks()
    print("\nDone.")


if __name__ == "__main__":
    main()
