"""
tune_patch_threshold.py — Visual threshold sweep for patch-center min_fg.

Produces contact-sheet PNGs showing how the patch-center mask changes across
19 threshold values (0.001 → 0.10), helping you choose --min-fg before
running compute_patch_centers.py on the full dataset.

For each sampled image, saves:
  masks/threshold_tuning/<version>/image_<idx>_<exp_type>.png
    Two rows per image, 19 columns (one per threshold):
      Row 1: preprocessed image with green area overlay where centers are valid
      Row 2: pixel mask with same green overlay
    Column header shows threshold value, N valid centers, and % valid.

Also saves summary_grid.png with all images × all thresholds stacked vertically.

Usage:
  python tune_patch_threshold.py --mask-version v_entropy --n-images 6 --seed 0
  python tune_patch_threshold.py --mask-version v_entropy --indices 0 42 500 700
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compute_patch_centers import compute_fg_fraction_map
from data_loader import load_metadata_csv
from preprocessing import make_preprocessor

# Default 19 threshold values spanning 0.1% → 10%
DEFAULT_THRESHOLDS = [
    0.001, 0.002, 0.003, 0.004, 0.005,
    0.006, 0.007, 0.008, 0.009, 0.010,
    0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10,
]
THRESHOLDS = DEFAULT_THRESHOLDS  # overridden by --thresholds at runtime

PREPROCESSOR = None


def _get_preprocessor():
    global PREPROCESSOR
    if PREPROCESSOR is None:
        PREPROCESSOR = make_preprocessor(gray_method="lab_l", pool_factor=10)
    return PREPROCESSOR


def _load_pixel_mask(mask_dir: str, version: str, idx: int) -> np.ndarray:
    name = version if version.startswith("v_") else f"v_{version}"
    path = os.path.join(mask_dir, name, f"{idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Pixel mask not found: {path}. "
            f"Run: python compute_masks.py --methods {name[2:]}"
        )
    return img > 127


def _load_image_arr(photo_dir: str, idx: int) -> np.ndarray:
    path = os.path.join(photo_dir, f"{idx:04d}.JPG")
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    tensor = _get_preprocessor()(img_bgr)
    return tensor.squeeze(0).numpy()   # (400, 600) float32


def _format_threshold(t: float) -> str:
    if t < 0.01:
        return f"{t*100:.1f}%"
    return f"{t*100:.0f}%"


def _overlay_centers(arr_gray: np.ndarray, center_mask: np.ndarray) -> np.ndarray:
    """Return RGB uint8 image with a semi-transparent green area overlay where centers are valid.

    Uses an area overlay (not individual dots) to avoid row-major stride artifacts
    that cause vertical banding when subsampling np.where coordinate arrays.
    Green channel is boosted by 0.5 in valid regions, giving a clear tint without
    obscuring the underlying image structure.
    """
    base = (arr_gray * 255).astype(np.uint8)
    r = base.copy()
    g = base.copy()
    b = base.copy()
    # In valid-center regions: tint green (boost green channel, suppress red/blue slightly)
    g[center_mask] = np.clip(g[center_mask].astype(np.int16) + 120, 0, 255).astype(np.uint8)
    r[center_mask] = np.clip(r[center_mask].astype(np.int16) - 40,  0, 255).astype(np.uint8)
    b[center_mask] = np.clip(b[center_mask].astype(np.int16) - 40,  0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)   # BGR for OpenCV


def make_contact_sheet(
    arr: np.ndarray,
    pixel_mask: np.ndarray,
    fg_fraction_map: np.ndarray,
    idx: int,
    exp_type: str,
    patch_size: int,
) -> np.ndarray:
    """Build the two-row contact sheet for one image across all thresholds."""
    H, W = arr.shape
    n_thresh = len(THRESHOLDS)
    cell_h, cell_w = H, W
    header_h = 24  # pixels for threshold label

    sheet_h = header_h + 2 * cell_h
    sheet_w = n_thresh * cell_w
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for col, t in enumerate(THRESHOLDS):
        center_mask = fg_fraction_map >= t
        n_valid = int(center_mask.sum())
        frac_valid = n_valid / center_mask.size

        row1 = _overlay_centers(arr, center_mask)
        row2_gray = (pixel_mask.astype(np.float32))
        row2 = _overlay_centers(row2_gray, center_mask)

        x0 = col * cell_w

        # Header: threshold label + stats
        label = f"{_format_threshold(t)} N={n_valid}({frac_valid:.0%})"
        cv2.putText(
            sheet, label, (x0 + 2, header_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA
        )

        sheet[header_h : header_h + cell_h, x0 : x0 + cell_w] = row1
        sheet[header_h + cell_h : header_h + 2 * cell_h, x0 : x0 + cell_w] = row2

    return sheet


def run(args):
    global THRESHOLDS
    if args.thresholds:
        THRESHOLDS = sorted(args.thresholds)
    else:
        THRESHOLDS = DEFAULT_THRESHOLDS

    version = args.mask_version if args.mask_version.startswith("v_") else f"v_{args.mask_version}"

    # Load metadata
    df = load_metadata_csv(args.csv_path)

    # Choose images to sample
    if args.indices:
        selected_indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        pool = df["idx"].tolist()
        if not args.include_empty:
            flagged_path = os.path.join(args.mask_dir, version, "flagged_empty.txt")
            excluded = set()
            if os.path.exists(flagged_path):
                with open(flagged_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            try:
                                excluded.add(int(line.split()[0]))
                            except ValueError:
                                pass
            pool = [i for i in pool if i not in excluded]
        chosen = rng.choice(pool, size=min(args.n_images, len(pool)), replace=False)
        selected_indices = sorted(chosen.tolist())

    os.makedirs(args.output_dir, exist_ok=True)

    sheets = []
    idx_meta = df.set_index("idx")

    for idx in selected_indices:
        print(f"  Processing idx={idx:04d}...")
        arr = _load_image_arr(args.photo_dir, idx)
        pixel_mask = _load_pixel_mask(args.mask_dir, version, idx)
        fg_map = compute_fg_fraction_map(pixel_mask, args.patch_size)

        exp_type = idx_meta.loc[idx, "Exp_Type"] if idx in idx_meta.index else "unknown"
        sheet = make_contact_sheet(arr, pixel_mask, fg_map, idx, exp_type, args.patch_size)
        sheets.append(sheet)

        out_path = os.path.join(args.output_dir, f"image_{idx:04d}_{exp_type}.png")
        cv2.imwrite(out_path, sheet)
        print(f"    Saved: {out_path}")

    # Summary grid: stack all image sheets vertically
    if len(sheets) > 1:
        grid = np.vstack(sheets)
        grid_path = os.path.join(args.output_dir, "summary_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"Summary grid: {grid_path}")
    elif len(sheets) == 1:
        grid_path = os.path.join(args.output_dir, "summary_grid.png")
        cv2.imwrite(grid_path, sheets[0])
        print(f"Summary grid: {grid_path}")

    print("Done. Inspect images to choose --min-fg for compute_patch_centers.py")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visual threshold sweep for patch-center min_fg selection."
    )
    p.add_argument("--mask-version",  required=True,
                   help="Pixel mask version, e.g. v_entropy or entropy")
    p.add_argument("--mask-dir",      default="masks")
    p.add_argument("--photo-dir",     default="data/photos")
    p.add_argument("--csv-path",      default="data/img-metadata.csv")
    p.add_argument("--patch-size",    type=int, default=200)
    p.add_argument("--n-images",      type=int, default=6,
                   help="Number of images to sample randomly")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--indices",       nargs="+", type=int, default=None,
                   help="Use specific image indices instead of random sampling")
    p.add_argument("--include-empty", action="store_true",
                   help="Include images flagged as fully empty in random sample")
    p.add_argument("--thresholds",    nargs="+", type=float, default=None,
                   help="Override threshold list (e.g. --thresholds 0.02 0.025 0.03 0.04 0.05)")
    p.add_argument("--output-dir",    default="masks/threshold_tuning")
    return p.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
