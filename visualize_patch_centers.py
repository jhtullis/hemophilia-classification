"""
visualize_patch_centers.py — Show patch-center masks with actual patch footprints.

For a given image and mask, draws:
  Left:   Preprocessed image with ~30 randomly sampled patch rectangles
          (green = valid-center patch, red = invalid-center patch for contrast)
          + inscribed circle on one patch to show the coverage-check region
  Middle: Pixel mask (foreground=white) with patch-center validity as overlay
  Right:  Patch-center validity mask (white=valid center, black=invalid)

This makes it visually clear how the green dots in the threshold sweep contact
sheets translate to where patches are actually sampled from.

Usage:
  python visualize_patch_centers.py --idx 340 --mask-version v_entropy
  python visualize_patch_centers.py --idx 340 --mask-version v_intensity --min-fg 0.05
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from compute_patch_centers import compute_fg_fraction_map
from preprocessing import make_preprocessor

PATCH_SIZE = 200
HALF = PATCH_SIZE // 2

PREPROCESSOR = None


def _get_preprocessor():
    global PREPROCESSOR
    if PREPROCESSOR is None:
        PREPROCESSOR = make_preprocessor(gray_method="lab_l", pool_factor=10)
    return PREPROCESSOR


def _load_arr(photo_dir, idx):
    path = os.path.join(photo_dir, f"{idx:04d}.JPG")
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return _get_preprocessor()(bgr).squeeze(0).numpy()  # (400, 600)


def _load_pixel_mask(mask_dir, version, idx):
    name = version if version.startswith("v_") else f"v_{version}"
    path = os.path.join(mask_dir, name, f"{idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Pixel mask not found: {path}. "
            f"Run compute_masks.py --methods {name[2:]} --indices {idx}"
        )
    return img > 127


def make_figure(arr, pixel_mask, center_mask, idx, mask_version, min_fg,
                n_patches=30, seed=42):
    H, W = arr.shape
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"idx={idx:04d}  |  {mask_version}  |  min_fg={min_fg:.3f}  |  "
        f"valid centers: {center_mask.sum()} / {center_mask.size} "
        f"({center_mask.mean():.1%})",
        fontsize=11,
    )

    # ---- Panel 1: image + sampled patch rectangles ----
    ax = axes[0]
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Patch footprints on image\n(green=valid center, red=invalid)", fontsize=9)
    ax.axis("off")

    # Sample valid and invalid centers
    valid_ys, valid_xs = np.where(center_mask)
    invalid_ys, invalid_xs = np.where(~center_mask)

    n_valid = min(n_patches * 3 // 4, len(valid_ys))
    n_invalid = min(n_patches - n_valid, len(invalid_ys))

    drawn_valid_cy = drawn_valid_cx = None  # for annotating the inscribed circle

    for k in range(n_valid):
        i = rng.integers(0, len(valid_ys))
        cy, cx = int(valid_ys[i]), int(valid_xs[i])
        rect = mpatches.Rectangle(
            (cx - HALF, cy - HALF), PATCH_SIZE, PATCH_SIZE,
            linewidth=1.0, edgecolor="#2ca02c", facecolor="#2ca02c",
            alpha=0.18,
        )
        ax.add_patch(rect)
        if k == 0:
            drawn_valid_cy, drawn_valid_cx = cy, cx

    for k in range(n_invalid):
        i = rng.integers(0, len(invalid_ys))
        cy, cx = int(invalid_ys[i]), int(invalid_xs[i])
        rect = mpatches.Rectangle(
            (cx - HALF, cy - HALF), PATCH_SIZE, PATCH_SIZE,
            linewidth=1.0, edgecolor="#d62728", facecolor="#d62728",
            alpha=0.18,
        )
        ax.add_patch(rect)

    # Annotate one valid center with its inscribed circle
    if drawn_valid_cy is not None:
        circle = mpatches.Circle(
            (drawn_valid_cx, drawn_valid_cy), radius=HALF,
            linewidth=1.5, edgecolor="yellow", facecolor="none",
            linestyle="--", label="inscribed circle (r=100)",
        )
        ax.add_patch(circle)
        ax.plot(drawn_valid_cx, drawn_valid_cy, "y+", markersize=8)

    # Patch legend
    ax.legend(handles=[
        mpatches.Patch(facecolor="#2ca02c", alpha=0.6, label="valid center patch"),
        mpatches.Patch(facecolor="#d62728", alpha=0.6, label="invalid center patch"),
        mpatches.Patch(facecolor="yellow", alpha=0.8, label="inscribed circle"),
    ], loc="upper right", fontsize=7)

    # ---- Panel 2: pixel mask + center validity overlay ----
    ax = axes[1]
    # Pixel mask as grayscale background; green overlay where center is valid
    rgb = np.stack([pixel_mask.astype(np.float32)] * 3, axis=-1)
    rgb[center_mask, 1] = 1.0   # green channel = 1 where valid
    rgb[center_mask, 0] = 0.0
    rgb[center_mask, 2] = 0.0
    ax.imshow(rgb)
    fg_frac = pixel_mask.mean()
    ax.set_title(
        f"Pixel mask + valid centers (green)\n"
        f"fg={fg_frac:.1%} foreground pixels",
        fontsize=9,
    )
    ax.axis("off")

    # ---- Panel 3: patch-center validity mask ----
    ax = axes[2]
    ax.imshow(center_mask, cmap="gray")
    ax.set_title(
        f"Patch-center validity mask\n"
        f"white=can sample here, black=skip",
        fontsize=9,
    )
    ax.axis("off")

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idx",          type=int, required=True)
    p.add_argument("--mask-version", default="v_entropy")
    p.add_argument("--min-fg",       type=float, default=0.10)
    p.add_argument("--patch-size",   type=int, default=200)
    p.add_argument("--photo-dir",    default="data/photos")
    p.add_argument("--mask-dir",     default="masks")
    p.add_argument("--output-dir",   default="masks/inspection")
    p.add_argument("--n-patches",    type=int, default=30)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    version = args.mask_version if args.mask_version.startswith("v_") else f"v_{args.mask_version}"

    print(f"Loading image {args.idx:04d}...")
    arr = _load_arr(args.photo_dir, args.idx)
    pixel_mask = _load_pixel_mask(args.mask_dir, version, args.idx)

    print("Computing fg_fraction_map (FFT)...")
    fg_map = compute_fg_fraction_map(pixel_mask, args.patch_size)
    center_mask = fg_map >= args.min_fg

    print(f"Valid centers: {center_mask.sum()} / {center_mask.size} "
          f"({center_mask.mean():.1%}) at min_fg={args.min_fg}")

    fig = make_figure(arr, pixel_mask, center_mask, args.idx, version,
                      args.min_fg, args.n_patches, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"patch_centers_{args.idx:04d}_{version}_minfg{args.min_fg:.3f}.png"
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
