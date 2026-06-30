"""
compute_patch_centers.py — Derive patch-center validity masks from pixel masks.

For each image, computes which pixel positions are valid patch centers using the
inscribed-circle rule: a center is valid if >=min_fg fraction of the circle of
radius (patch_size//2) centered there falls on foreground pixels.

The key module-level functions compute_fg_fraction_map() and make_patch_center_mask()
are importable by tune_patch_threshold.py for threshold sweep visualizations.

Usage:
  python compute_patch_centers.py --mask-versions intensity entropy frangi or vote2
  python compute_patch_centers.py --mask-versions entropy --min-fg 0.05
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from tqdm import tqdm

from patch_dataset import PATCH_SIZE

# ---------------------------------------------------------------------------
# Core geometry functions (module-level for external import)
# ---------------------------------------------------------------------------

def compute_fg_fraction_map(pixel_mask: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Return (H, W) float32 map of inscribed-circle foreground fraction per center.

    For each candidate patch center (cy, cx), computes the fraction of the inscribed
    circle (radius = patch_size // 2) that falls on foreground pixels. Edge positions
    are evaluated only over the in-bounds portion of the circle, consistent with how
    out-of-bounds areas become zero padding during training.

    This is the expensive operation (one convolution). Call it once then apply
    multiple thresholds with compute_fg_fraction_map(mask) >= threshold.
    """
    H, W = pixel_mask.shape
    r = patch_size // 2

    Y, X = np.ogrid[-r:r + 1, -r:r + 1]
    kernel = (X ** 2 + Y ** 2 <= r ** 2).astype(np.float32)

    fg_count = fftconvolve(pixel_mask.astype(np.float32), kernel, mode="same")
    inbounds_count = fftconvolve(np.ones((H, W), dtype=np.float32), kernel, mode="same")
    # Clip small negative values from FFT numerical noise
    fg_count = np.clip(fg_count, 0.0, None)
    inbounds_count = np.clip(inbounds_count, 0.0, None)
    return np.where(inbounds_count > 0, fg_count / inbounds_count, 0.0)


def make_patch_center_mask(
    pixel_mask: np.ndarray,
    patch_size: int = PATCH_SIZE,
    min_fg: float = 0.10,
) -> np.ndarray:
    """Return (H, W) bool mask: True = this center yields a valid patch.

    A center is valid if >=min_fg fraction of its inscribed circle falls on
    foreground pixels in the given pixel mask.
    """
    return compute_fg_fraction_map(pixel_mask, patch_size) >= min_fg


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_pixel_mask(mask_dir: str, version: str, idx: int) -> np.ndarray:
    path = os.path.join(mask_dir, f"v_{version}", f"{idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Pixel mask not found: {path}")
    return img > 127


def _center_mask_dir(mask_dir: str, version: str, patch_size: int) -> str:
    return os.path.join(mask_dir, "patch_centers", f"p{patch_size}_circle_v_{version}")


def _center_mask_path(mask_dir: str, version: str, patch_size: int, idx: int) -> str:
    return os.path.join(_center_mask_dir(mask_dir, version, patch_size), f"{idx:04d}.png")


def _save_center_mask(mask: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mask.astype(np.uint8) * 255)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_version(
    version: str,
    indices: list,
    mask_dir: str,
    patch_size: int,
    min_fg: float,
    df_idx_map: dict,
    force: bool,
    image_type_filter: str,
) -> None:
    vdir = _version_dir_for_pixel(mask_dir, version)
    if not os.path.isdir(vdir):
        print(f"Error: pixel mask directory '{vdir}' not found. "
              f"Run compute_masks.py --methods {version} first.")
        sys.exit(1)

    out_dir = _center_mask_dir(mask_dir, version, patch_size)
    os.makedirs(out_dir, exist_ok=True)

    stats_rows = []

    for idx in tqdm(indices, desc=f"  p{patch_size}_circle_v_{version}", leave=False):
        out_path = _center_mask_path(mask_dir, version, patch_size, idx)
        if not force and os.path.exists(out_path):
            center_mask = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE) > 127
        else:
            pixel_mask = _load_pixel_mask(mask_dir, version, idx)
            center_mask = make_patch_center_mask(pixel_mask, patch_size, min_fg)
            _save_center_mask(center_mask, out_path)

        n_valid = int(center_mask.sum())
        frac_valid = float(n_valid) / center_mask.size

        row = df_idx_map.get(idx, {})
        img_type_raw = row.get("img_type", "")
        image_type = "exploratory" if img_type_raw == "endpoint_10" else "grid"

        stats_rows.append({
            "idx":                    idx,
            "exp_type":               row.get("Exp_Type", ""),
            "experiment":             row.get("Experiment", ""),
            "image_type":             image_type,
            "n_valid_centers":        n_valid,
            "fraction_valid_centers": round(frac_valid, 6),
            "flagged_empty":          n_valid == 0,
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    n_flagged = int(stats_df["flagged_empty"].sum())
    mean_frac = stats_df["fraction_valid_centers"].mean()
    print(f"  p{patch_size}_circle_v_{version}: {len(indices) - n_flagged}/{len(indices)} images "
          f"with valid centers  |  mean fraction_valid={mean_frac:.3f}  |  "
          f"{n_flagged} fully empty")


def _version_dir_for_pixel(mask_dir: str, version: str) -> str:
    return os.path.join(mask_dir, f"v_{version}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Derive patch-center validity masks from pixel-level masks."
    )
    p.add_argument("--mask-dir",       default="masks")
    p.add_argument("--mask-versions",  nargs="+",
                   choices=["intensity", "entropy", "frangi", "or", "vote2"],
                   default=["intensity", "entropy", "frangi", "or", "vote2"])
    p.add_argument("--patch-size",     type=int, default=PATCH_SIZE)
    p.add_argument("--min-fg",         type=float, default=0.10,
                   help="Minimum foreground fraction in inscribed circle to be valid")
    p.add_argument("--image-type",     choices=["all", "exploratory", "grid"],
                   default="all")
    p.add_argument("--csv-path",       default="data/img-metadata.csv",
                   help="Metadata CSV (needed for stats.csv labels)")
    p.add_argument("--indices",        nargs="+", type=int, default=None)
    p.add_argument("--force",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    from data_loader import load_metadata_csv
    df = load_metadata_csv(args.csv_path)
    df_idx_map = {row["idx"]: row for _, row in df.iterrows()}

    if args.indices is not None:
        indices = sorted(args.indices)
    elif args.image_type == "exploratory":
        indices = sorted(df[df["img_type"] == "endpoint_10"]["idx"].tolist())
    elif args.image_type == "grid":
        indices = sorted(df[df["img_type"] == "endpoint_64"]["idx"].tolist())
    else:
        indices = sorted(df["idx"].tolist())

    print(f"Processing {len(indices)} images, min_fg={args.min_fg}, "
          f"patch_size={args.patch_size}")

    for version in args.mask_versions:
        print(f"Computing patch-center masks from v_{version}...")
        process_version(
            version, indices, args.mask_dir, args.patch_size, args.min_fg,
            df_idx_map, args.force, args.image_type,
        )

    print("Done.")


if __name__ == "__main__":
    main()
