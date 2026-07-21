"""
masked_patch_dataset_full.py — Full-resolution mask-aware patch sampling.

MaskedFullPatchDataset is the full-resolution counterpart of MaskedPatchDataset.
Images are loaded at 6000×4000 (no min-pool); patches are 2000×2000 (same physical
area as the 200×200 patches in the 600×400 min-pooled pipeline).

Center sampling reuses the existing 600×400 Otsu masks (patch_center_version
"p200_circle_v_intensity") by upscaling them 10× on the fly:

    mask_400 (400×600 bool)  →  np.repeat ×10 each axis  →  mask_full (4000×6000 bool)

Each valid pixel in the 600×400 mask corresponds to a 10×10 block of valid pixels
in the 6000×4000 space, so the physical content constraint is preserved exactly.

Coverage formula note: n_valid_centers in stats.csv is in 600×400 space. The
full-res upscaling gives n_valid_full = n_valid × 100, and PATCH_SIZE_FULL² =
PATCH_SIZE² × 100, so these cancel — the same coverage_multiplier and stats.csv
yield the same number of patches per image as in the 600×400 pipeline.

Preloading is disabled by default: a padded full-res image is ~241 MB (float32),
making preloading 700 images infeasible (~169 GB). Use num_workers ≥ 4 instead.
"""

import bisect
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from data_loader import CLASS_MAP, CLASS_NAMES, load_metadata_csv
from model_patch_full import OVERSIZED_FULL, PAD_FULL, PATCH_SIZE_FULL
from patch_dataset import _pad_tensor

# n_valid in stats.csv is in 600×400 space (PATCH_SIZE=200).
# Use 200² as the denominator so the coverage formula gives the same allocation
# as in MaskedPatchDataset (the ×100 scale factors in n_valid_full / PATCH_SIZE_FULL²
# cancel exactly with the stats.csv scale).
_COVERAGE_PATCH_SIZE = 200

# Full-image dimensions (expected)
_IMG_H = 4000
_IMG_W = 6000
_MASK_H = 400   # center masks are in 600×400 space
_MASK_W = 600
_SCALE  = 10    # _IMG_H / _MASK_H == _IMG_W / _MASK_W


class MaskedFullPatchDataset(Dataset):
    """Mask-aware full-resolution patch sampling.

    Identical interface to MaskedPatchDataset except:
      - Images are not min-pooled (preprocessor returns (1, 4000, 6000) tensors)
      - Patch size is PATCH_SIZE_FULL = 2000 (same physical area as 200 in 600×400)
      - Center masks are upscaled 10× on the fly from the existing 600×400 PNGs
      - Preloading is not supported (images too large)

    Args:
        df:                    Metadata DataFrame from patch_dataset.load_split_record.
        photo_dir:             Directory containing 0000.JPG … 0999.JPG.
        preprocessor:          From make_preprocessor_full() — img_bgr → (1,4000,6000).
        mask_dir:              Root masks/ directory.
        mask_version:          Pixel mask subdirectory, e.g. "v_intensity".
        patch_center_version:  Patch-center mask subdirectory, e.g.
                               "p200_circle_v_intensity" (600×400 PNGs reused).
        coverage_multiplier:   Target patch coverage per valid-center area (default 2.0).
        uniform_fraction:      Fraction of patches allocated uniformly per image.
                               Set 0.0 for fully content-weighted; 1.0 for equal
                               allocation (recommended for validation).
        validate_classes:      Raise ValueError at init if any class has no valid images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        photo_dir: str,
        preprocessor: Callable,
        mask_dir: str,
        mask_version: str,
        patch_center_version: str,
        coverage_multiplier: float = 2.0,
        uniform_fraction: float = 0.0,
        validate_classes: bool = True,
        include_grid: bool = False,
        csv_path: Optional[str] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)

        if include_grid:
            if csv_path is None:
                raise ValueError("csv_path is required when include_grid=True")
            grid_all = load_metadata_csv(csv_path)
            grid_df = grid_all[grid_all["img_type"] == "endpoint_64"].reset_index(drop=True)
            self.df = pd.concat([self.df, grid_df]).reset_index(drop=True)
        self.photo_dir = photo_dir
        self.preprocessor = preprocessor
        self.mask_dir = mask_dir
        self.mask_version = mask_version if mask_version.startswith("v_") else f"v_{mask_version}"
        self.patch_center_version = patch_center_version
        self.coverage_multiplier = coverage_multiplier
        self.uniform_fraction = uniform_fraction

        # Load patch-center stats (n_valid_centers in 600×400 space)
        stats_path = os.path.join(
            mask_dir, "patch_centers", patch_center_version, "stats.csv"
        )
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Patch-center stats not found: {stats_path}\n"
                f"Run: python compute_patch_centers.py --mask-versions "
                f"{self.mask_version[2:]}"
            )
        center_stats = pd.read_csv(stats_path).set_index("idx")
        self._n_valid: Dict[int, int] = center_stats["n_valid_centers"].to_dict()

        # Load empty-image exclusion list
        flagged_path = os.path.join(mask_dir, self.mask_version, "flagged_empty.txt")
        self._excluded_indices: set = set()
        if os.path.exists(flagged_path):
            with open(flagged_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            self._excluded_indices.add(int(line.split()[0]))
                        except ValueError:
                            pass

        # Build valid row positions
        self._valid_row_positions: List[int] = []
        for pos in range(len(self.df)):
            row = self.df.iloc[pos]
            idx = int(row["idx"])
            n_valid = self._n_valid.get(idx, -1)
            if idx in self._excluded_indices or n_valid == 0:
                continue
            self._valid_row_positions.append(pos)

        if len(self._valid_row_positions) == 0:
            raise ValueError(
                "MaskedFullPatchDataset: no valid images found. "
                "Check mask_version and patch_center_version."
            )

        if validate_classes:
            valid_classes = {
                self.df.iloc[pos]["Exp_Type"]
                for pos in self._valid_row_positions
                if "Exp_Type" in self.df.columns
            }
            for cls in CLASS_MAP:
                if cls in set(self.df["Exp_Type"]) and cls not in valid_classes:
                    raise ValueError(
                        f"Class '{cls}' has no valid full-res images under "
                        f"'{patch_center_version}'."
                    )

        # Coverage formula: same denominator as MaskedPatchDataset (n_valid in 600×400
        # space, _COVERAGE_PATCH_SIZE=200) → identical patch allocation per image.
        valid_n_valid = [
            self._n_valid.get(int(self.df.iloc[pos]["idx"]), 0)
            for pos in self._valid_row_positions
        ]
        mean_valid = float(np.mean(valid_n_valid)) if valid_n_valid else 1.0
        raw_uniform = coverage_multiplier * mean_valid / (_COVERAGE_PATCH_SIZE ** 2)

        self._patches_per_image: List[int] = []
        for pos in self._valid_row_positions:
            idx = int(self.df.iloc[pos]["idx"])
            n_valid = self._n_valid.get(idx, 0)
            raw_content = coverage_multiplier * n_valid / (_COVERAGE_PATCH_SIZE ** 2)
            blended = (1 - uniform_fraction) * raw_content + uniform_fraction * raw_uniform
            self._patches_per_image.append(max(1, round(blended)))

        self._start_indices = np.zeros(len(self._valid_row_positions), dtype=np.int64)
        cumsum = 0
        for i, n in enumerate(self._patches_per_image):
            self._start_indices[i] = cumsum
            cumsum += n
        self._total_patches = int(cumsum)

        n_valid_imgs = len(self._valid_row_positions)
        mean_patches = self._total_patches / max(n_valid_imgs, 1)
        print(
            f"MaskedFullPatchDataset: {n_valid_imgs} valid images, "
            f"{self._total_patches} total patch slots "
            f"(mean {mean_patches:.1f} patches/image), "
            f"patch_size={PATCH_SIZE_FULL}"
        )

    # ---------------------------------------------------------------------------
    # Internal loaders
    # ---------------------------------------------------------------------------

    def _load_base_tensor(self, row_pos: int) -> torch.Tensor:
        row = self.df.iloc[row_pos]
        img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return self.preprocessor(img_bgr)   # (1, 4000, 6000) float32

    def _get_padded_tensor(self, slot_idx: int) -> torch.Tensor:
        row_pos = self._valid_row_positions[slot_idx]
        return _pad_tensor(self._load_base_tensor(row_pos), PAD_FULL)

    def _load_center_mask_arr(self, idx: int) -> np.ndarray:
        """Load 600×400 center mask and upscale 10× → 4000×6000."""
        mask_path = os.path.join(
            self.mask_dir, "patch_centers", self.patch_center_version,
            f"{idx:04d}.png"
        )
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Patch-center mask not found: {mask_path}\n"
                f"Run: python compute_patch_centers.py"
            )
        mask_400 = img > 127   # (400, 600) bool
        # Upscale: each valid 600×400 pixel → 10×10 block in 4000×6000 space
        return np.repeat(np.repeat(mask_400, _SCALE, axis=0), _SCALE, axis=1)  # (4000, 6000)

    # ---------------------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        slot_idx = bisect.bisect_right(self._start_indices, idx) - 1

        row_pos = self._valid_row_positions[slot_idx]
        row = self.df.iloc[row_pos]
        label = CLASS_MAP[row["Exp_Type"]]

        padded      = self._get_padded_tensor(slot_idx)
        center_mask = self._load_center_mask_arr(int(row["idx"]))   # (4000, 6000)

        valid_ys, valid_xs = np.where(center_mask)
        if len(valid_ys) == 0:
            cy = np.random.randint(0, _IMG_H)
            cx = np.random.randint(0, _IMG_W)
        else:
            k = np.random.randint(len(valid_ys))
            cy, cx = int(valid_ys[k]), int(valid_xs[k])

        cy_pad = cy + PAD_FULL
        cx_pad = cx + PAD_FULL
        half   = OVERSIZED_FULL // 2
        # .clone() is critical: without it, the returned patch is a view that keeps the
        # entire 241 MB padded tensor alive. With batch_size N, N views → N × 241 MB of
        # padded tensors held simultaneously in the DataLoader assembly buffer (OOM).
        patch  = padded[
            :,
            cy_pad - half : cy_pad + half + 1,
            cx_pad - half : cx_pad + half + 1,
        ].clone()   # (1, OVERSIZED_FULL, OVERSIZED_FULL) — 32 MB, padded freed

        return patch, label

    # ---------------------------------------------------------------------------
    # Sampler
    # ---------------------------------------------------------------------------

    def make_sampler(self) -> WeightedRandomSampler:
        """Class-balanced WeightedRandomSampler (same logic as MaskedPatchDataset)."""
        n_classes = len(CLASS_NAMES)

        class_total: Dict[str, int] = defaultdict(int)
        for row_pos, n_patches in zip(self._valid_row_positions, self._patches_per_image):
            cls = self.df.iloc[row_pos]["Exp_Type"]
            class_total[cls] += n_patches

        weights: List[float] = []
        for row_pos, n_patches in zip(self._valid_row_positions, self._patches_per_image):
            cls = self.df.iloc[row_pos]["Exp_Type"]
            w = (1.0 / n_classes) / class_total[cls]
            weights.extend([w] * n_patches)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=self._total_patches,
            replacement=True,
        )

    # ---------------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------------

    def class_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for pos in self._valid_row_positions:
            counts[self.df.iloc[pos]["Exp_Type"]] += 1
        return dict(counts)
