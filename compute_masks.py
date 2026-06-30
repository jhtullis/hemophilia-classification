"""
compute_masks.py — Precompute pixel-level content masks for all images.

Produces one PNG per image per mask method (0=background, 255=foreground)
plus a stats.csv and flagged_empty.txt per method directory.

Methods:
  intensity  Otsu threshold on inverted grayscale (fibrin is dark)
  entropy    Local Shannon entropy with disk radius=5
  frangi     Multi-scale Frangi vesselness (sigma 2,4,6,8,10)
  or         Union of intensity, entropy, frangi (must exist first)
  vote2      >=2/3 agreement of intensity, entropy, frangi (must exist first)

Usage:
  python compute_masks.py --methods intensity entropy frangi
  python compute_masks.py --methods or vote2
  python compute_masks.py --methods intensity --indices 0 1 2   # smoke test
  python compute_masks.py --methods intensity --image-type exploratory
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import CLASS_MAP, load_metadata_csv
from preprocessing import make_preprocessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY_IMAGE_THRESHOLD = 0.01   # fg_fraction below this → flagged as empty
BASE_METHODS = ("intensity", "entropy", "frangi")
COMPOSITE_METHODS = ("or", "vote2")
ALL_METHODS = BASE_METHODS + COMPOSITE_METHODS

PREPROCESSOR = None   # lazily initialized once per run


def _get_preprocessor():
    global PREPROCESSOR
    if PREPROCESSOR is None:
        PREPROCESSOR = make_preprocessor(gray_method="lab_l", pool_factor=10)
    return PREPROCESSOR


# ---------------------------------------------------------------------------
# Mask computation
# ---------------------------------------------------------------------------

def _load_image_tensor(img_path: str) -> np.ndarray:
    """Load JPG, preprocess to 600x400 float32, squeeze to (400, 600)."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    tensor = _get_preprocessor()(img_bgr)   # (1, 400, 600) float32
    return tensor.squeeze(0).numpy()        # (400, 600)


def make_intensity_mask(arr: np.ndarray) -> np.ndarray:
    """Otsu threshold on inverted image (fibrin=dark → bright after inversion)."""
    from skimage.filters import threshold_otsu
    inverted = 1.0 - arr
    try:
        thresh = threshold_otsu(inverted)
        return inverted > thresh
    except Exception:
        return np.zeros(arr.shape, dtype=bool)


def make_entropy_mask(arr: np.ndarray, radius: int = 5,
                      threshold_percentile: float = 40.0) -> np.ndarray:
    """Local Shannon entropy — high entropy = fibrin network structure."""
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk
    img_uint8 = (arr * 255).astype(np.uint8)
    ent = rank_entropy(img_uint8, disk(radius)).astype(np.float32)
    thresh = np.percentile(ent, threshold_percentile)
    return ent > thresh


def make_frangi_mask(arr: np.ndarray,
                     sigmas: tuple = (2, 4, 6, 8, 10),
                     threshold: float = 0.01) -> np.ndarray:
    """Multi-scale Frangi vesselness (black_ridges=True for dark fibrin)."""
    from skimage.filters import frangi
    vesselness = np.max(
        np.stack([frangi(arr, sigmas=(s,), black_ridges=True) for s in sigmas]),
        axis=0,
    )
    return vesselness > threshold


# ---------------------------------------------------------------------------
# PNG I/O
# ---------------------------------------------------------------------------

def _save_mask_png(mask: np.ndarray, path: str) -> None:
    """Save bool mask as uint8 grayscale PNG (True=255, False=0)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mask.astype(np.uint8) * 255)


def _load_mask_png(path: str) -> np.ndarray:
    """Load a previously saved mask PNG as a bool array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask PNG not found: {path}")
    return img > 127


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _version_dir(mask_dir: str, method: str) -> str:
    return os.path.join(mask_dir, f"v_{method}")


def _mask_png_path(mask_dir: str, method: str, idx: int) -> str:
    return os.path.join(_version_dir(mask_dir, method), f"{idx:04d}.png")


def process_base_method(
    method: str,
    indices: list,
    photo_dir: str,
    mask_dir: str,
    df_idx_map: dict,   # idx -> row dict
    force: bool,
) -> tuple:
    """Process one base mask method. Returns (stats_rows, flagged_empty)."""
    vdir = _version_dir(mask_dir, method)
    os.makedirs(vdir, exist_ok=True)

    stats_rows = []
    flagged_empty = []

    for idx in tqdm(indices, desc=f"  {method}", leave=False):
        out_path = _mask_png_path(mask_dir, method, idx)
        if not force and os.path.exists(out_path):
            # Resume: load from existing PNG
            mask = _load_mask_png(out_path)
        else:
            img_path = os.path.join(photo_dir, f"{idx:04d}.JPG")
            arr = _load_image_tensor(img_path)
            if method == "intensity":
                mask = make_intensity_mask(arr)
            elif method == "entropy":
                mask = make_entropy_mask(arr)
            elif method == "frangi":
                mask = make_frangi_mask(arr)
            else:
                raise ValueError(f"Unknown base method: {method}")
            _save_mask_png(mask, out_path)

        fg_count = int(mask.sum())
        fg_fraction = float(fg_count) / mask.size
        is_empty = fg_fraction < EMPTY_IMAGE_THRESHOLD

        row = df_idx_map.get(idx, {})
        img_type_raw = row.get("img_type", "")
        image_type = "exploratory" if img_type_raw == "endpoint_10" else "grid"

        stats_rows.append({
            "idx":          idx,
            "exp_type":     row.get("Exp_Type", ""),
            "experiment":   row.get("Experiment", ""),
            "image_type":   image_type,
            "fg_fraction":  round(fg_fraction, 6),
            "n_fg_pixels":  fg_count,
            "flagged_empty": is_empty,
        })
        if is_empty:
            flagged_empty.append((idx, fg_fraction))

    return stats_rows, flagged_empty


def process_composite_method(
    method: str,
    indices: list,
    mask_dir: str,
    df_idx_map: dict,
    force: bool,
) -> tuple:
    """Process or/vote2 from existing base PNG masks."""
    for base in BASE_METHODS:
        bdir = _version_dir(mask_dir, base)
        if not os.path.isdir(bdir):
            print(f"Error: base mask directory '{bdir}' not found. "
                  f"Run --methods {' '.join(BASE_METHODS)} first.")
            sys.exit(1)

    vdir = _version_dir(mask_dir, method)
    os.makedirs(vdir, exist_ok=True)

    stats_rows = []
    flagged_empty = []

    for idx in tqdm(indices, desc=f"  {method}", leave=False):
        out_path = _mask_png_path(mask_dir, method, idx)
        if not force and os.path.exists(out_path):
            mask = _load_mask_png(out_path)
        else:
            masks = [_load_mask_png(_mask_png_path(mask_dir, b, idx))
                     for b in BASE_METHODS]
            if method == "or":
                mask = masks[0] | masks[1] | masks[2]
            else:  # vote2
                votes = masks[0].astype(np.int8) + masks[1].astype(np.int8) + masks[2].astype(np.int8)
                mask = votes >= 2
            _save_mask_png(mask, out_path)

        fg_count = int(mask.sum())
        fg_fraction = float(fg_count) / mask.size
        is_empty = fg_fraction < EMPTY_IMAGE_THRESHOLD

        row = df_idx_map.get(idx, {})
        img_type_raw = row.get("img_type", "")
        image_type = "exploratory" if img_type_raw == "endpoint_10" else "grid"

        stats_rows.append({
            "idx":          idx,
            "exp_type":     row.get("Exp_Type", ""),
            "experiment":   row.get("Experiment", ""),
            "image_type":   image_type,
            "fg_fraction":  round(fg_fraction, 6),
            "n_fg_pixels":  fg_count,
            "flagged_empty": is_empty,
        })
        if is_empty:
            flagged_empty.append((idx, fg_fraction))

    return stats_rows, flagged_empty


def save_outputs(mask_dir: str, method: str,
                 stats_rows: list, flagged_empty: list) -> None:
    vdir = _version_dir(mask_dir, method)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(vdir, "stats.csv"), index=False)

    flagged_path = os.path.join(vdir, "flagged_empty.txt")
    with open(flagged_path, "w") as f:
        f.write(f"# Images flagged as empty by v_{method} "
                f"(fg_fraction < {EMPTY_IMAGE_THRESHOLD})\n")
        f.write("# idx\tfg_fraction\n")
        for idx, frac in sorted(flagged_empty):
            f.write(f"{idx}\t{frac:.6f}\n")

    n_fg = len(stats_rows) - len(flagged_empty)
    print(f"  v_{method}: {n_fg}/{len(stats_rows)} images with >{EMPTY_IMAGE_THRESHOLD*100:.0f}% "
          f"foreground, {len(flagged_empty)} flagged empty")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute pixel-level content masks for fibrin images."
    )
    p.add_argument("--csv-path",    default="data/img-metadata.csv")
    p.add_argument("--photo-dir",   default="data/photos")
    p.add_argument("--mask-dir",    default="masks")
    p.add_argument("--methods",     nargs="+", choices=list(ALL_METHODS),
                   default=list(BASE_METHODS))
    p.add_argument("--image-type",  choices=["all", "exploratory", "grid"],
                   default="all",
                   help="Which image subset to process (default: all)")
    p.add_argument("--indices",     nargs="+", type=int, default=None,
                   help="Process only these specific image indices (overrides --image-type)")
    p.add_argument("--force",       action="store_true",
                   help="Recompute masks even if PNG already exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Load metadata
    df = load_metadata_csv(args.csv_path)
    df_idx_map = {row["idx"]: row for _, row in df.iterrows()}

    # Determine which indices to process
    if args.indices is not None:
        indices = sorted(args.indices)
        print(f"Processing {len(indices)} specified indices.")
    elif args.image_type == "exploratory":
        indices = sorted(df[df["img_type"] == "endpoint_10"]["idx"].tolist())
        print(f"Processing {len(indices)} exploratory images (endpoint_10).")
    elif args.image_type == "grid":
        indices = sorted(df[df["img_type"] == "endpoint_64"]["idx"].tolist())
        print(f"Processing {len(indices)} grid images (endpoint_64).")
    else:
        indices = sorted(df["idx"].tolist())
        print(f"Processing all {len(indices)} images.")

    os.makedirs(args.mask_dir, exist_ok=True)

    # Separate base and composite methods, maintain order
    base_requested = [m for m in args.methods if m in BASE_METHODS]
    composite_requested = [m for m in args.methods if m in COMPOSITE_METHODS]

    for method in base_requested:
        print(f"Computing v_{method}...")
        stats_rows, flagged_empty = process_base_method(
            method, indices, args.photo_dir, args.mask_dir, df_idx_map, args.force
        )
        save_outputs(args.mask_dir, method, stats_rows, flagged_empty)

    for method in composite_requested:
        print(f"Computing v_{method} (composite)...")
        stats_rows, flagged_empty = process_composite_method(
            method, indices, args.mask_dir, df_idx_map, args.force
        )
        save_outputs(args.mask_dir, method, stats_rows, flagged_empty)

    print("Done.")


if __name__ == "__main__":
    main()
