"""
masked_patch_dataset.py — Mask-aware patch sampling Dataset.

MaskedPatchDataset is a drop-in replacement for FibrinPatchDataset that only
samples patch centers from content-rich regions identified by a precomputed
mask. Fully empty images (n_valid_centers == 0) are excluded at init.

Key design differences from FibrinPatchDataset:
  - Variable patches_per_image per image (based on content richness)
  - Class-balanced sampler that preserves equal class representation
    regardless of content distribution across classes
  - Exploratory-only by default; grid images opt-in via include_grid=True
  - Pre-training validation: raises if any training class has no valid images

Usage in a training script:
    from masked_patch_dataset import MaskedPatchDataset, extend_split_with_grid

    dataset = MaskedPatchDataset(
        df=train_df,                           # exploratory rows from load_split_record
        photo_dir="data/photos",
        preprocessor=make_preprocessor(),
        mask_dir="masks",
        mask_version="v_entropy",
        patch_center_version="p200_circle_v_entropy",
    )
    loader = DataLoader(dataset, batch_size=64, sampler=dataset.make_sampler())
"""

import bisect
import os
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from data_loader import CLASS_MAP, CLASS_NAMES, load_metadata_csv
from patch_dataset import OVERSIZED, PAD, PATCH_SIZE, _pad_tensor


# ---------------------------------------------------------------------------
# Helper: extend exploratory split with grid images
# ---------------------------------------------------------------------------

def extend_split_with_grid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign grid images to the same split partition as their experiment.

    Grid experiments match the exploratory experiments exactly (same 50 experiments,
    same class assignments). Each grid image inherits the split assignment of its
    exploratory partner based on Experiment name — no new split decisions needed.

    Args:
        train_df, val_df, test_df: From patch_dataset.load_split_record (exploratory).
        grid_df: Grid rows from load_metadata_csv(), filtered to endpoint_64 img_type.

    Returns:
        (train_df, val_df, test_df) each extended with grid rows.
    """
    train_exps = set(train_df["Experiment"])
    val_exps   = set(val_df["Experiment"])
    test_exps  = set(test_df["Experiment"])

    return (
        pd.concat([train_df, grid_df[grid_df["Experiment"].isin(train_exps)]]).reset_index(drop=True),
        pd.concat([val_df,   grid_df[grid_df["Experiment"].isin(val_exps)]]).reset_index(drop=True),
        pd.concat([test_df,  grid_df[grid_df["Experiment"].isin(test_exps)]]).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaskedPatchDataset(Dataset):
    """Mask-aware patch sampling from fibrin images.

    Patches are sampled exclusively from valid patch centers (positions where the
    inscribed circle of radius patch_size//2 has >= min_fg foreground coverage).
    Fully empty images are excluded from training entirely.

    Number of patches per image is variable: images with more valid centers
    contribute proportionally more patches per epoch, blended with a uniform
    baseline to prevent starvation of low-content images. This achieves
    approximately 2x coverage of sampleable areas on average.

    Args:
        df:                    Metadata DataFrame (exploratory rows from load_split_record).
        photo_dir:             Directory containing all images (0000.JPG – 7399.JPG).
        preprocessor:          From make_preprocessor() — img_bgr -> (1,400,600) tensor.
        mask_dir:              Root masks/ directory.
        mask_version:          Pixel mask to use for empty-image detection,
                               e.g. "v_entropy" or "entropy".
        patch_center_version:  Patch-center mask directory name,
                               e.g. "p200_circle_v_entropy".
        gate_mode:             "hard" = sample uniformly from valid centers only.
                               "soft" = reserved for future use.
        preload:               Preload all tensors and center masks into RAM.
                               ~1.2 GB for 1000 exploratory images (default True).
                               With grid (~7400 images), ~8.9 GB — set False.
        include_grid:          If True, append grid images (endpoint_64) from csv_path.
        csv_path:              Required when include_grid=True.
        coverage_multiplier:   Target patch coverage per valid-center area (default 2.0).
        uniform_fraction:      Fraction of patches allocated uniformly across images
                               (0.20 = 20% uniform baseline). Prevents starvation of
                               low-content images.
        validate_classes:      If True (default), raise ValueError at init if any class
                               in the DataFrame has zero valid images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        photo_dir: str,
        preprocessor: Callable,
        mask_dir: str,
        mask_version: str,
        patch_center_version: str,
        gate_mode: str = "hard",
        preload: bool = True,
        include_grid: bool = False,
        csv_path: Optional[str] = None,
        coverage_multiplier: float = 2.0,
        uniform_fraction: float = 0.20,
        validate_classes: bool = True,
    ) -> None:
        self.photo_dir = photo_dir
        self.preprocessor = preprocessor
        self.mask_dir = mask_dir
        self.mask_version = mask_version if mask_version.startswith("v_") else f"v_{mask_version}"
        self.patch_center_version = patch_center_version
        self.gate_mode = gate_mode
        self.coverage_multiplier = coverage_multiplier
        self.uniform_fraction = uniform_fraction

        # Optionally extend with grid images
        if include_grid:
            if csv_path is None:
                raise ValueError("csv_path is required when include_grid=True")
            grid_all = load_metadata_csv(csv_path)
            grid_df = grid_all[grid_all["img_type"] == "endpoint_64"].reset_index(drop=True)
            self.df = pd.concat([df, grid_df]).reset_index(drop=True)
            if preload and len(self.df) > 2000:
                warnings.warn(
                    f"MaskedPatchDataset: preloading {len(self.df)} images with grid data "
                    f"will use ~{len(self.df) * 1.22 / 1000:.1f} GB RAM. "
                    "Set preload=False if memory is limited."
                )
        else:
            self.df = df.reset_index(drop=True)

        # Load patch-center stats to get n_valid_centers per image
        stats_path = os.path.join(
            mask_dir, "patch_centers", patch_center_version, "stats.csv"
        )
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Patch-center stats not found: {stats_path}\n"
                f"Run: python compute_patch_centers.py --mask-versions {self.mask_version[2:]}"
            )
        center_stats = pd.read_csv(stats_path).set_index("idx")
        self._n_valid: Dict[int, int] = center_stats["n_valid_centers"].to_dict()
        self._frac_valid: Dict[int, float] = center_stats["fraction_valid_centers"].to_dict()

        # Load empty-image list from pixel mask
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

        # Build list of valid (non-empty) row positions
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
                "MaskedPatchDataset: no valid images found. "
                "All images are either flagged empty or missing patch-center masks. "
                "Check mask_version and patch_center_version."
            )

        # Optional: validate that each class has at least one valid image
        if validate_classes:
            valid_classes = set(
                self.df.iloc[pos]["Exp_Type"]
                for pos in self._valid_row_positions
                if "Exp_Type" in self.df.columns
            )
            for cls in CLASS_MAP:
                if cls in set(self.df["Exp_Type"]) and cls not in valid_classes:
                    raise ValueError(
                        f"Class '{cls}' has no valid images under mask "
                        f"'{patch_center_version}'. "
                        "Lower --min-fg in compute_patch_centers.py or inspect masks."
                    )

        # Compute variable patches per image (2x coverage + uniform blend)
        valid_n_valid = [
            self._n_valid.get(int(self.df.iloc[pos]["idx"]), 0)
            for pos in self._valid_row_positions
        ]
        mean_valid = float(np.mean(valid_n_valid)) if valid_n_valid else 1.0
        raw_uniform = coverage_multiplier * mean_valid / (PATCH_SIZE ** 2)

        self._patches_per_image: List[int] = []
        for pos in self._valid_row_positions:
            idx = int(self.df.iloc[pos]["idx"])
            n_valid = self._n_valid.get(idx, 0)
            raw_content = coverage_multiplier * n_valid / (PATCH_SIZE ** 2)
            blended = (1 - uniform_fraction) * raw_content + uniform_fraction * raw_uniform
            self._patches_per_image.append(max(1, round(blended)))

        # Cumulative start indices for binary-search lookup in __getitem__
        self._start_indices = np.zeros(len(self._valid_row_positions), dtype=np.int64)
        cumsum = 0
        for i, n in enumerate(self._patches_per_image):
            self._start_indices[i] = cumsum
            cumsum += n
        self._total_patches = int(cumsum)

        n_valid_imgs = len(self._valid_row_positions)
        n_excluded = len(self.df) - n_valid_imgs
        mean_patches = self._total_patches / max(n_valid_imgs, 1)
        print(f"MaskedPatchDataset: {n_valid_imgs} valid images "
              f"({n_excluded} excluded), "
              f"{self._total_patches} total patch slots "
              f"(mean {mean_patches:.1f} patches/image)")

        # Preload tensors and center masks
        self._tensors: Optional[List[torch.Tensor]] = None
        self._center_masks: Optional[List[np.ndarray]] = None

        if preload:
            print(f"  Preloading {n_valid_imgs} tensors...")
            self._tensors = []
            for pos in self._valid_row_positions:
                t = self._load_base_tensor(pos)
                self._tensors.append(_pad_tensor(t, PAD))

            print(f"  Preloading {n_valid_imgs} center masks...")
            self._center_masks = []
            for pos in self._valid_row_positions:
                idx = int(self.df.iloc[pos]["idx"])
                self._center_masks.append(self._load_center_mask_arr(idx))

            print("  Preload complete.")

    # ---------------------------------------------------------------------------
    # Internal loaders
    # ---------------------------------------------------------------------------

    def _load_base_tensor(self, row_pos: int) -> torch.Tensor:
        row = self.df.iloc[row_pos]
        img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return self.preprocessor(img_bgr)   # (1, 400, 600) float32

    def _get_padded_tensor(self, slot_idx: int) -> torch.Tensor:
        if self._tensors is not None:
            return self._tensors[slot_idx]
        return _pad_tensor(self._load_base_tensor(self._valid_row_positions[slot_idx]), PAD)

    def _load_center_mask_arr(self, idx: int) -> np.ndarray:
        mask_path = os.path.join(
            self.mask_dir, "patch_centers", self.patch_center_version,
            f"{idx:04d}.png"
        )
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Patch-center mask not found: {mask_path}\n"
                f"Run: python compute_patch_centers.py --mask-versions {self.mask_version[2:]}"
            )
        return img > 127   # bool (H, W)

    def _get_center_mask(self, slot_idx: int) -> np.ndarray:
        if self._center_masks is not None:
            return self._center_masks[slot_idx]
        idx = int(self.df.iloc[self._valid_row_positions[slot_idx]]["idx"])
        return self._load_center_mask_arr(idx)

    # ---------------------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Binary search: find which image slot this global index belongs to
        slot_idx = bisect.bisect_right(self._start_indices, idx) - 1

        row_pos = self._valid_row_positions[slot_idx]
        row = self.df.iloc[row_pos]
        label = CLASS_MAP[row["Exp_Type"]]

        padded = self._get_padded_tensor(slot_idx)  # (1, 400+2*PAD, 600+2*PAD)
        center_mask = self._get_center_mask(slot_idx)  # (400, 600) bool

        # Sample a valid center
        if self.gate_mode == "hard":
            valid_ys, valid_xs = np.where(center_mask)
            if len(valid_ys) == 0:
                # Fallback (should not occur after empty-image filtering)
                cy = np.random.randint(0, 400)
                cx = np.random.randint(0, 600)
            else:
                k = np.random.randint(len(valid_ys))
                cy, cx = int(valid_ys[k]), int(valid_xs[k])
        else:
            raise NotImplementedError(f"gate_mode={self.gate_mode!r} not implemented")

        # Extract oversized (OVERSIZED x OVERSIZED) patch from padded tensor
        cy_pad = cy + PAD
        cx_pad = cx + PAD
        half = OVERSIZED // 2
        patch = padded[
            :,
            cy_pad - half : cy_pad + half + 1,
            cx_pad - half : cx_pad + half + 1,
        ]   # (1, 283, 283)

        return patch, label

    # ---------------------------------------------------------------------------
    # Sampler
    # ---------------------------------------------------------------------------

    def make_sampler(self) -> WeightedRandomSampler:
        """Class-balanced WeightedRandomSampler for the training DataLoader.

        Each class contributes exactly 1/n_classes fraction of sampled patches
        per epoch. Within a class, images are sampled proportional to their
        patch count (which encodes content richness via the coverage formula).

        Do NOT use this sampler for validation or test DataLoaders — use plain
        sequential DataLoader there so every image is visited exactly once.
        """
        n_classes = len(CLASS_NAMES)

        # Total patch slots per class
        class_total: Dict[str, int] = defaultdict(int)
        for slot_idx, (row_pos, n_patches) in enumerate(
            zip(self._valid_row_positions, self._patches_per_image)
        ):
            cls = self.df.iloc[row_pos]["Exp_Type"]
            class_total[cls] += n_patches

        # Assign weight to every dataset slot
        weights: List[float] = []
        for slot_idx, (row_pos, n_patches) in enumerate(
            zip(self._valid_row_positions, self._patches_per_image)
        ):
            cls = self.df.iloc[row_pos]["Exp_Type"]
            # weight = equal share of 1/n_classes, divided by class's total slots
            w = (1.0 / n_classes) / class_total[cls]
            weights.extend([w] * n_patches)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=self._total_patches,
            replacement=True,
        )

    # ---------------------------------------------------------------------------
    # Introspection helpers
    # ---------------------------------------------------------------------------

    def class_distribution(self) -> Dict[str, int]:
        """Count of valid images per class in this dataset."""
        counts: Dict[str, int] = defaultdict(int)
        for pos in self._valid_row_positions:
            cls = self.df.iloc[pos]["Exp_Type"]
            counts[cls] += 1
        return dict(counts)

    def patch_budget_summary(self) -> pd.DataFrame:
        """Per-class summary of patch budgets for inspection."""
        rows = []
        for pos, n_patches in zip(self._valid_row_positions, self._patches_per_image):
            row = self.df.iloc[pos]
            idx = int(row["idx"])
            rows.append({
                "idx":          idx,
                "Exp_Type":     row["Exp_Type"],
                "Experiment":   row["Experiment"],
                "n_valid_centers": self._n_valid.get(idx, 0),
                "patches_allocated": n_patches,
            })
        return pd.DataFrame(rows)
