"""
visualize_weights.py — Visualise FibrinCNN kernel weights and activation maps.

Produces four PNG files saved into the model directory:

  conv1_kernels.png       32 filters (1×5×5) — all input channels shown
  conv2_kernels.png       64 filters (32×5×5) — mean over input channels
  conv3_conv4_kernels.png 128 + 256 filters (3×3) — mean over input channels
  activation_maps.png     Mean activation per block for one test image per class

Usage:
    python visualize_weights.py                    # models/5class/
    python visualize_weights.py --model-dir models/3class_hemo
"""

import argparse
import os
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import CLASS_MAP, CLASS_NAMES, filter_classes, load_split_from_record
from model import FibrinCNN
from preprocessing import make_preprocessor

# ── Defaults ──────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), "data", "endpoint10.db")
PHOTOS_DIR = "data/photos"
GRAY_METHOD = "lab_l"
POOL_FACTOR = 10

# FibrinCNN.features Sequential layout (4 blocks, 4 modules each):
#   block k starts at index 4*(k-1)
#   [0]  Conv2d(1→32,  5×5)   [1]  BN   [2]  ReLU  [3]  MaxPool
#   [4]  Conv2d(32→64, 5×5)   [5]  BN   [6]  ReLU  [7]  MaxPool
#   [8]  Conv2d(64→128,3×3)   [9]  BN   [10] ReLU  [11] MaxPool
#   [12] Conv2d(128→256,3×3)  [13] BN   [14] ReLU  [15] MaxPool
CONV_INDICES = [0, 4, 8, 12]
POOL_INDICES = [3, 7, 11, 15]


def _load_model(model_dir: str, device: torch.device) -> FibrinCNN:
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Train first.")

    # Detect num_classes from the saved state dict
    sd = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = sd["classifier.3.bias"].shape[0]

    model = FibrinCNN(num_classes=num_classes)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def _kernel_grid(weights: np.ndarray, nrows: int, ncols: int,
                 title: str, output_path: str) -> None:
    """Render a grid of small kernel heatmaps and save to output_path.

    Args:
        weights:     Array of shape (N, kH, kW) — one 2-D kernel per filter.
        nrows/ncols: Grid dimensions (nrows × ncols == N).
        title:       Figure suptitle.
        output_path: Where to save the PNG.
    """
    assert len(weights) == nrows * ncols, (
        f"Grid {nrows}×{ncols}={nrows*ncols} ≠ {len(weights)} kernels"
    )

    vmax = float(np.abs(weights).max())
    if vmax == 0:
        vmax = 1.0

    cell = max(0.8, 60 / max(nrows, ncols))   # scale cell size to grid
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell, nrows * cell),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)

    im = None
    for idx, (ax, kernel) in enumerate(zip(axes.flat, weights)):
        im = ax.imshow(kernel, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.axis("off")
        if nrows <= 8:   # omit titles for very dense grids
            ax.set_title(f"K{idx}", fontsize=6, pad=1)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {output_path}")


# ── Figure 1: Conv1 kernels ────────────────────────────────────────────────────

def plot_conv1(model: FibrinCNN, output_dir: str) -> None:
    """32 filters of shape (1, 5, 5) — single input channel, fully interpretable."""
    w = model.features[CONV_INDICES[0]].weight.data.cpu().numpy()  # (32,1,5,5)
    kernels = w[:, 0, :, :]  # (32, 5, 5)
    _kernel_grid(kernels, nrows=4, ncols=8,
                 title="Conv1 kernels  (32 filters, 1×5×5)\n"
                       "Red = positive weight, Blue = negative weight",
                 output_path=os.path.join(output_dir, "conv1_kernels.png"))


# ── Figure 2: Conv2 kernels ────────────────────────────────────────────────────

def plot_conv2(model: FibrinCNN, output_dir: str) -> None:
    """64 filters of shape (32, 5, 5) — aggregate by mean over input channels."""
    w = model.features[CONV_INDICES[1]].weight.data.cpu().numpy()  # (64,32,5,5)
    kernels = w.mean(axis=1)  # (64, 5, 5)
    _kernel_grid(kernels, nrows=8, ncols=8,
                 title="Conv2 kernels  (64 filters, mean over 32 input channels, 5×5)\n"
                       "Red = positive, Blue = negative",
                 output_path=os.path.join(output_dir, "conv2_kernels.png"))


# ── Figure 3: Conv3 + Conv4 kernels ───────────────────────────────────────────

def plot_conv3_conv4(model: FibrinCNN, output_dir: str) -> None:
    """128 (3×3) and 256 (3×3) filters side by side, mean over input channels."""
    w3 = model.features[CONV_INDICES[2]].weight.data.cpu().numpy()  # (128,64,3,3)
    w4 = model.features[CONV_INDICES[3]].weight.data.cpu().numpy()  # (256,128,3,3)
    k3 = w3.mean(axis=1)  # (128, 3, 3)
    k4 = w4.mean(axis=1)  # (256, 3, 3)

    vmax = float(max(np.abs(k3).max(), np.abs(k4).max()))
    if vmax == 0:
        vmax = 1.0

    # Conv3: 8×16  Conv4: 16×16  — side by side in one figure
    fig = plt.figure(figsize=(48, 24))
    gs  = fig.add_gridspec(1, 2, wspace=0.06)

    def _fill_panel(sub_gs, kernels, nrows, ncols, label):
        inner = sub_gs.subgridspec(nrows, ncols, wspace=0.02, hspace=0.02)
        for idx, kernel in enumerate(kernels[:nrows * ncols]):
            ax = fig.add_subplot(inner[idx // ncols, idx % ncols])
            ax.imshow(kernel, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.axis("off")
        # Panel title via annotation on the first subplot
        ax0 = fig.add_subplot(inner[0, 0])
        ax0.set_title(label, fontsize=14, pad=4)

    _fill_panel(gs[0], k3, nrows=8,  ncols=16,
                label="Conv3  (128 filters, 3×3, mean over 64 ch)")
    _fill_panel(gs[1], k4, nrows=16, ncols=16,
                label="Conv4  (256 filters, 3×3, mean over 128 ch)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.005)
    fig.suptitle("Conv3 + Conv4 kernel weights", fontsize=16, y=1.005)

    out = os.path.join(output_dir, "conv3_conv4_kernels.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out}")


# ── Figure 4: Activation maps ─────────────────────────────────────────────────

def plot_activation_maps(model: FibrinCNN, model_dir: str,
                         output_dir: str, device: torch.device,
                         preprocessor) -> None:
    """Show mean activation per block for one representative test image per class.

    Layout: rows = classes, cols = conv blocks.
    The activation shown is the channel-mean of the MaxPool output of each block,
    giving a spatial heatmap of where the network responds most strongly.
    """
    # Determine class set from the model's output size
    num_classes = model.classifier[-1].out_features
    if num_classes == 3:
        class_names = ["F08D", "F09D", "F11D"]
        from data_loader import CLASS_MAP_3  # type: ignore[attr-defined]
        cmap = {"F08D": 0, "F09D": 1, "F11D": 2}
    else:
        class_names = CLASS_NAMES
        cmap = CLASS_MAP

    # Load test split and select the first image (by idx) for each class
    record_path = os.path.join(model_dir, "train_record.json")
    if not os.path.exists(record_path):
        record_path = "models/5class/train_record.json"
    _, test_df = load_split_from_record(record_path, args.db)
    if num_classes == 3:
        test_df = filter_classes(test_df, class_names)

    rep_images: Dict[str, np.ndarray] = {}
    for cls in class_names:
        rows = test_df[test_df["Exp_Type"] == cls].sort_values("idx")
        if rows.empty:
            continue
        idx = int(rows.iloc[0]["idx"])
        img_path = os.path.join(PHOTOS_DIR, f"{idx:04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        rep_images[cls] = img_bgr

    if not rep_images:
        print("   WARNING: no test images found; skipping activation maps.")
        return

    # Register hooks on MaxPool outputs (indices 3, 7, 11, 15)
    activations: Dict[int, torch.Tensor] = {}

    def make_hook(block_idx: int):
        def hook(module, input, output):
            activations[block_idx] = output.detach()
        return hook

    hooks = [model.features[i].register_forward_hook(make_hook(i))
             for i in POOL_INDICES]

    block_labels = [
        "Block 1\n300×200", "Block 2\n150×100",
        "Block 3\n75×50",   "Block 4\n37×25",
    ]
    n_cls = len(rep_images)
    n_blocks = len(POOL_INDICES)

    fig, axes = plt.subplots(n_cls, n_blocks,
                             figsize=(n_blocks * 3.5, n_cls * 3.0),
                             squeeze=False)

    for row, cls in enumerate(class_names):
        if cls not in rep_images:
            continue
        tensor = preprocessor(rep_images[cls]).unsqueeze(0).to(device)
        with torch.no_grad():
            model(tensor)

        for col, pidx in enumerate(POOL_INDICES):
            act = activations[pidx].squeeze(0)   # (C, H, W)
            heatmap = act.mean(dim=0).cpu().numpy()  # (H, W) mean over channels

            ax = axes[row, col]
            ax.imshow(heatmap, cmap="viridis", interpolation="bilinear")
            ax.axis("off")

            if row == 0:
                ax.set_title(block_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(cls, fontsize=11, rotation=0, labelpad=40,
                              va="center")

    for h in hooks:
        h.remove()

    fig.suptitle("Mean activation maps (MaxPool output per block)\n"
                 "One representative test image per class",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "activation_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise FibrinCNN kernel weights and activation maps."
    )
    parser.add_argument("--model-dir", default="models/5class", metavar="PATH",
                        help="Directory containing best_model.pth (default: models/5class).")
    parser.add_argument("--db", default=DB_PATH, metavar="PATH",
                        help="Path to SQLite database (default: data/endpoint10.db).")
    args = parser.parse_args()

    device = torch.device("cpu")

    print(f"Loading model from {args.model_dir} …")
    model = _load_model(args.model_dir, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print("\nConv layer indices in model.features:")
    for i, idx in enumerate(CONV_INDICES, 1):
        layer = model.features[idx]
        print(f"   features[{idx:2d}]  {layer}")

    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)

    os.makedirs(args.model_dir, exist_ok=True)

    print("\nFigure 1: Conv1 kernels …")
    plot_conv1(model, args.model_dir)

    print("Figure 2: Conv2 kernels …")
    plot_conv2(model, args.model_dir)

    print("Figure 3: Conv3 + Conv4 kernels …")
    plot_conv3_conv4(model, args.model_dir)

    print("Figure 4: Activation maps …")
    plot_activation_maps(model, args.model_dir, args.model_dir, device, preprocessor)

    print("\nDone.")


if __name__ == "__main__":
    main()
