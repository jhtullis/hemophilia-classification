"""
inspect_masks.py — CLI tool for manual mask inspection.

Three modes:

  image    Single-image side-by-side overlay PNG
  summary  Dataset-wide table comparing all mask versions
  richness Per-experiment/class richness table for a given mask version

Usage:
  python inspect_masks.py image \\
      --idx 42 --mask-version v_intensity \\
      --photo-dir data/photos --mask-dir masks --output-dir masks/inspection

  python inspect_masks.py summary --mask-dir masks

  python inspect_masks.py richness \\
      --mask-version v_entropy --mask-dir masks --csv-path data/img-metadata.csv
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_metadata_csv
from preprocessing import make_preprocessor

PREPROCESSOR = None


def _get_preprocessor():
    global PREPROCESSOR
    if PREPROCESSOR is None:
        PREPROCESSOR = make_preprocessor(gray_method="lab_l", pool_factor=10)
    return PREPROCESSOR


# ---------------------------------------------------------------------------
# Mode 1: single image overlay
# ---------------------------------------------------------------------------

def _load_pixel_mask(mask_dir: str, mask_version: str, idx: int) -> np.ndarray:
    name = mask_version if mask_version.startswith("v_") else f"v_{mask_version}"
    path = os.path.join(mask_dir, name, f"{idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Pixel mask not found: {path}. "
                                f"Run compute_masks.py --methods {name[2:]} first.")
    return img > 127


def _find_center_mask(mask_dir: str, mask_version: str, idx: int):
    """Find the first matching patch-center mask for this pixel mask version."""
    name = mask_version if mask_version.startswith("v_") else f"v_{mask_version}"
    pc_root = os.path.join(mask_dir, "patch_centers")
    if not os.path.isdir(pc_root):
        return None
    for subdir in sorted(os.listdir(pc_root)):
        if subdir.endswith(f"_{name}"):
            path = os.path.join(pc_root, subdir, f"{idx:04d}.png")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img > 127
    return None


def mode_image(args):
    idx = args.idx
    photo_path = os.path.join(args.photo_dir, f"{idx:04d}.JPG")
    img_bgr = cv2.imread(photo_path)
    if img_bgr is None:
        print(f"Error: image not found at {photo_path}")
        sys.exit(1)

    # Preprocessed grayscale (H, W) float32
    tensor = _get_preprocessor()(img_bgr)
    arr = tensor.squeeze(0).numpy()   # (400, 600)

    pixel_mask = _load_pixel_mask(args.mask_dir, args.mask_version, idx)
    center_mask = _find_center_mask(args.mask_dir, args.mask_version, idx)

    n_panels = 3 if center_mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 6, 4))

    # Panel 1: preprocessed image
    axes[0].imshow(arr, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Preprocessed (idx={idx:04d})")
    axes[0].axis("off")

    # Panel 2: pixel mask
    axes[1].imshow(pixel_mask, cmap="gray")
    fg_frac = pixel_mask.mean()
    axes[1].set_title(f"{args.mask_version}  fg={fg_frac:.3f}")
    axes[1].axis("off")

    # Panel 3 (optional): center mask overlaid on original
    if center_mask is not None:
        overlay = np.stack([arr, arr, arr], axis=-1)
        overlay[center_mask, 1] = 1.0   # valid centers → green channel
        axes[2].imshow(overlay)
        n_valid = center_mask.sum()
        frac_valid = n_valid / center_mask.size
        axes[2].set_title(f"Valid centers  n={n_valid}  ({frac_valid:.2%})")
        axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    version_str = args.mask_version.replace("v_", "")
    out_path = os.path.join(args.output_dir, f"img_{idx:04d}_{version_str}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Mode 2: dataset summary
# ---------------------------------------------------------------------------

def mode_summary(args):
    mask_dir = args.mask_dir
    versions = [d for d in sorted(os.listdir(mask_dir))
                if d.startswith("v_") and os.path.isdir(os.path.join(mask_dir, d))]

    if not versions:
        print(f"No mask version directories found in {mask_dir}")
        sys.exit(1)

    print(f"\n{'version':<20} {'n_images':>8} {'n_flagged':>9} {'mean_fg':>8} "
          f"{'min_fg':>7} {'max_fg':>7}")
    print("-" * 68)

    all_stats = {}
    for ver in versions:
        stats_path = os.path.join(mask_dir, ver, "stats.csv")
        if not os.path.exists(stats_path):
            print(f"  {ver}: no stats.csv (run compute_masks.py first)")
            continue
        df = pd.read_csv(stats_path)
        all_stats[ver] = df
        n = len(df)
        n_flagged = int(df["flagged_empty"].sum())
        mean_fg = df["fg_fraction"].mean()
        min_fg = df["fg_fraction"].min()
        max_fg = df["fg_fraction"].max()
        print(f"  {ver:<18} {n:>8} {n_flagged:>9} {mean_fg:>8.3f} "
              f"{min_fg:>7.3f} {max_fg:>7.3f}")

    # Cross-version: images flagged by all
    if len(all_stats) >= 3:
        base_versions = [v for v in ["v_intensity", "v_entropy", "v_frangi"]
                         if v in all_stats]
        if len(base_versions) == 3:
            flagged_sets = [set(all_stats[v][all_stats[v]["flagged_empty"]]["idx"])
                            for v in base_versions]
            flagged_by_all = flagged_sets[0] & flagged_sets[1] & flagged_sets[2]
            print(f"\n  Images flagged empty by all 3 base methods: {len(flagged_by_all)}")

    # Per image-type breakdown
    for ver, df in all_stats.items():
        if "image_type" in df.columns and df["image_type"].nunique() > 1:
            print(f"\n  {ver} breakdown by image_type:")
            grp = df.groupby("image_type").agg(
                n=("idx", "count"),
                n_flagged=("flagged_empty", "sum"),
                mean_fg=("fg_fraction", "mean"),
            )
            print(grp.to_string())

    # Per-class breakdown for first available base version
    for ver in ["v_entropy", "v_intensity", "v_frangi"]:
        if ver in all_stats:
            df = all_stats[ver]
            print(f"\n  {ver} fg_fraction by class (exploratory only):")
            exp_df = df[df["image_type"] == "exploratory"] if "image_type" in df.columns else df
            grp = exp_df.groupby("exp_type")["fg_fraction"].agg(["mean", "min", "max"])
            print(grp.to_string())
            break


# ---------------------------------------------------------------------------
# Mode 3: richness by class/experiment
# ---------------------------------------------------------------------------

def mode_richness(args):
    version = args.mask_version
    if not version.startswith("v_"):
        version = f"v_{version}"

    stats_path = os.path.join(args.mask_dir, version, "stats.csv")
    if not os.path.exists(stats_path):
        print(f"Error: stats.csv not found at {stats_path}. "
              f"Run compute_masks.py --methods {version[2:]} first.")
        sys.exit(1)

    df = pd.read_csv(stats_path)

    # Exploratory only for per-experiment table
    if "image_type" in df.columns:
        expl_df = df[df["image_type"] == "exploratory"].copy()
    else:
        expl_df = df.copy()

    print(f"\nRichness by experiment and class ({version}, exploratory images):\n")
    print(f"{'experiment':<20} {'class':<6} {'n_images':>8} "
          f"{'mean_fg':>8} {'min_fg':>7} {'n_flagged':>9}")
    print("-" * 70)

    for cls in sorted(expl_df["exp_type"].unique()):
        cls_df = expl_df[expl_df["exp_type"] == cls]
        for exp in sorted(cls_df["experiment"].unique()):
            exp_df = cls_df[cls_df["experiment"] == exp]
            n = len(exp_df)
            mean_fg = exp_df["fg_fraction"].mean()
            min_fg = exp_df["fg_fraction"].min()
            n_flagged = int(exp_df["flagged_empty"].sum())
            print(f"  {exp:<18} {cls:<6} {n:>8} {mean_fg:>8.3f} "
                  f"{min_fg:>7.3f} {n_flagged:>9}")
        print()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inspect content masks.")
    sub = p.add_subparsers(dest="mode", required=True)

    im = sub.add_parser("image", help="Single-image overlay")
    im.add_argument("--idx",          type=int, required=True)
    im.add_argument("--mask-version", required=True,
                    help="e.g. v_intensity or intensity")
    im.add_argument("--photo-dir",    default="data/photos")
    im.add_argument("--mask-dir",     default="masks")
    im.add_argument("--output-dir",   default="masks/inspection")

    sm = sub.add_parser("summary", help="Dataset summary table")
    sm.add_argument("--mask-dir",     default="masks")

    rm = sub.add_parser("richness", help="Per-experiment richness table")
    rm.add_argument("--mask-version", required=True)
    rm.add_argument("--mask-dir",     default="masks")
    rm.add_argument("--csv-path",     default="data/img-metadata.csv")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "image":
        mode_image(args)
    elif args.mode == "summary":
        mode_summary(args)
    elif args.mode == "richness":
        mode_richness(args)


if __name__ == "__main__":
    main()
