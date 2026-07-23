"""
masked_patch_dataset_125s.py — 1.25x-area-resize mask-aware patch sampling.

MaskedPatch125sDataset is the 1.25x-area-resize counterpart of MaskedPatchDataset
(10x-pool), MaskedPatch2xDataset (2x min-pool), and MaskedFullPatchDataset (unpooled).
Images are loaded at 3200x4800 (1.25x area-interpolation resize of the original
6000x4000 JPEGs, via cv2.INTER_AREA -- NOT the block-min-pool used by the other tiers);
patches are 1600x1600 (same physical area as the 200x200 patches in the 600x400
min-pooled pipeline, the 1000x1000 patches in the 2x-pool pipeline, and the 2000x2000
patches in the full-res pipeline).

Center sampling reuses the existing 600x400 Otsu masks (patch_center_version
"p200_circle_v_intensity") without recomputing anything: each valid 600x400 mask pixel
corresponds to an exact 8x8 block of pixels in the 3200x4800 space (10/1.25 = 8), so a
coarse center is picked from the native 600x400 mask and then jittered uniformly within
its 8x8 block to land on the finer 1.25x-resize grid:

    (cy_coarse, cx_coarse) --[x8 + random offset in [0,8)]--> (cy, cx) in 3200x4800 space

This is statistically identical to np.repeat-upscaling the mask 8x and sampling uniformly
over it (masked_patch_dataset_full.py's approach), but avoids ever materializing a dense
mask per image. Only the sparse list of valid coarse (y, x) coordinates is kept in
memory, loaded once per image at construction time.

Coverage formula note: n_valid_centers in stats.csv is in 600x400 space. Coarse-to-fine
jittering gives n_valid_fine = n_valid x 64, and PATCH_SIZE_125S^2 = PATCH_SIZE^2 x 64, so
these cancel -- the same coverage_multiplier and stats.csv yield the same number of
patches per image as in the 600x400, 2x-pool, and unpooled pipelines.

GPU preloading stores raw resized grayscale as UNPADDED UINT8 tensors (not normalized
float32): at 3200x4800, uint8 is ~15.36MB/image (~79.6GB for the ~5180 train-tier images
with grid included), leaving comfortable headroom on a single H200 (float32 would be
~61.4MB/image, ~318GB total -- infeasible). Padding and float normalization happen per
__getitem__, applied only to the small sliced OVERSIZED_125S x OVERSIZED_125S crop --
never to the whole padded canvas.

Requires num_workers=0 in the DataLoader when preload_device is set (CUDA tensors are not
fork-safe across DataLoader worker subprocesses).
"""

import bisect
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from data_loader import CLASS_MAP, CLASS_NAMES, load_metadata_csv
from model_patch_125s import OVERSIZED_125S, PAD_125S, PATCH_SIZE_125S
from patch_dataset import _pad_tensor
from preprocessing import ensure_landscape, resize_pool, to_grayscale

# n_valid in stats.csv is in 600x400 space (PATCH_SIZE=200).
# Use 200^2 as the denominator so the coverage formula gives the same allocation as in
# MaskedPatchDataset / MaskedPatch2xDataset / MaskedFullPatchDataset (the scale factors
# cancel exactly).
_COVERAGE_PATCH_SIZE = 200

# 1.25x-resize image dimensions (expected)
_IMG_H = 3200
_IMG_W = 4800
_MASK_H = 400   # center masks are in 600x400 space
_MASK_W = 600
_SCALE  = 8     # _IMG_H / _MASK_H == _IMG_W / _MASK_W == (10x-pool factor) / (1.25x-resize factor)
_RESIZE_FACTOR = 1.25


class MaskedPatch125sDataset(Dataset):
    """Mask-aware 1.25x-area-resize patch sampling.

    Identical interface to MaskedPatch2xDataset except:
      - Images are 1.25x area-resized (preprocessor returns (1, 3200, 4800) tensors)
      - Patch size is PATCH_SIZE_125S = 1600 (same physical area as 200 in 600x400)
      - Center masks are reused from the native 600x400 PNGs; centers are sampled at
        600x400 coarse granularity then jittered to the finer 1.25x-resize grid (not
        upscaled into a dense mask)
      - Optional GPU preloading: unpadded uint8 resized-grayscale tensors on preload_device

    Args:
        df:                    Metadata DataFrame from patch_dataset.load_split_record.
        photo_dir:             Directory containing 0000.JPG ... 0999.JPG.
        preprocessor:          From make_preprocessor_resized(pool_factor=1.25) --
                               img_bgr -> (1, 3200, 4800) float32 tensor. Only exercised
                               on the disk-read fallback path (preload_device=None);
                               when preload_device is set, this class builds its own
                               uint8 resized-grayscale pipeline internally for memory
                               efficiency.
        mask_dir:              Root masks/ directory.
        mask_version:          Pixel mask subdirectory, e.g. "v_intensity".
        patch_center_version:  Patch-center mask subdirectory, e.g.
                               "p200_circle_v_intensity" (600x400 PNGs reused as-is).
        coverage_multiplier:   Target patch coverage per valid-center area (default 2.0).
        uniform_fraction:      Fraction of patches allocated uniformly per image.
                               Set 0.0 for fully content-weighted; 1.0 for equal
                               allocation (recommended for validation).
        validate_classes:      Raise ValueError at init if any class has no valid images.
        include_grid:          If True, append grid images (endpoint_64) from csv_path,
                               filtered to the experiments already present in df.
        csv_path:              Required when include_grid=True.
        preload_device:        If set, GPU-preload unpadded uint8 tensors onto this
                               device. Requires num_workers=0 in the DataLoader.
        gray_method:           Grayscale conversion method (default "lab_l").
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
        preload_device: Optional[torch.device] = None,
        gray_method: str = "lab_l",
    ) -> None:
        self.df = df.reset_index(drop=True)

        # Optionally extend with grid images, restricted to the same experiments present
        # in the passed split -- prevents val/test grid images leaking into train.
        if include_grid:
            if csv_path is None:
                raise ValueError("csv_path is required when include_grid=True")
            grid_all = load_metadata_csv(csv_path)
            split_exps = set(df["Experiment"])
            grid_df = grid_all[
                (grid_all["img_type"] == "endpoint_64") &
                (grid_all["Experiment"].isin(split_exps))
            ].reset_index(drop=True)
            self.df = pd.concat([self.df, grid_df]).reset_index(drop=True)

        self.photo_dir = photo_dir
        self.preprocessor = preprocessor
        self.mask_dir = mask_dir
        self.mask_version = mask_version if mask_version.startswith("v_") else f"v_{mask_version}"
        self.patch_center_version = patch_center_version
        self.coverage_multiplier = coverage_multiplier
        self.uniform_fraction = uniform_fraction
        self.gray_method = gray_method
        self._preload_device = preload_device

        # Load patch-center stats (n_valid_centers in 600x400 space)
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
                "MaskedPatch125sDataset: no valid images found. "
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
                        f"Class '{cls}' has no valid 1.25x-resize images under "
                        f"'{patch_center_version}'."
                    )

        # Coverage formula: same denominator as MaskedPatchDataset (n_valid in 600x400
        # space, _COVERAGE_PATCH_SIZE=200) -> identical patch allocation per image.
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
            f"MaskedPatch125sDataset: {n_valid_imgs} valid images, "
            f"{self._total_patches} total patch slots "
            f"(mean {mean_patches:.1f} patches/image), "
            f"patch_size={PATCH_SIZE_125S}"
        )

        # Sparse coarse valid-coordinate arrays, loaded once from the native 600x400 PNGs.
        # Kept sparse (not upscaled into a dense 3200x4800 mask) to avoid excess CPU RAM
        # and an expensive per-__getitem__ scan that would serialize under num_workers=0.
        print(f"  Loading {n_valid_imgs} coarse center-coordinate lists...")
        self._coarse_coords: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for pos in self._valid_row_positions:
            idx = int(self.df.iloc[pos]["idx"])
            self._coarse_coords[idx] = self._load_coarse_coords(idx)

        # GPU preload: unpadded (1, 3200, 4800) uint8 tensors on preload_device.
        # Padding + float normalization happen per __getitem__, on the small sliced crop
        # only (never on the whole padded canvas).
        self._tensors: Optional[List[torch.Tensor]] = None
        if preload_device is not None:
            print(f"  GPU-preloading {n_valid_imgs} unpadded uint8 tensors to {preload_device}...")
            self._tensors = []
            for pos in self._valid_row_positions:
                t = self._load_base_tensor_uint8(pos)   # (1, 3200, 4800) uint8, CPU
                self._tensors.append(t.to(preload_device, non_blocking=True))
            total_gb = n_valid_imgs * _IMG_H * _IMG_W / 1e9
            print(f"  GPU preload complete: {total_gb:.2f} GB (uint8)")

    # ---------------------------------------------------------------------------
    # Internal loaders
    # ---------------------------------------------------------------------------

    def _load_coarse_coords(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load the native 600x400 center mask once and return valid (ys, xs) coords."""
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
        return np.where(mask_400)   # (ys, xs) in 600x400 space

    def _load_base_tensor_uint8(self, row_pos: int) -> torch.Tensor:
        """Grayscale + 1.25x area-resize, WITHOUT normalization. Returns (1, 3200, 4800) uint8."""
        row = self.df.iloc[row_pos]
        idx = int(row["idx"])
        img_path = os.path.join(self.photo_dir, f"{idx:04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_bgr = ensure_landscape(img_bgr)
        gray = to_grayscale(img_bgr, method=self.gray_method)         # (H, W) uint8
        pooled = resize_pool(gray, factor=_RESIZE_FACTOR)              # (3200, 4800) uint8
        return torch.from_numpy(pooled).unsqueeze(0)                   # (1, 3200, 4800) uint8

    def _load_base_tensor_disk(self, row_pos: int) -> torch.Tensor:
        row = self.df.iloc[row_pos]
        img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return self.preprocessor(img_bgr)   # (1, 3200, 4800) float32, normalized

    def _get_padded_tensor(self, slot_idx: int) -> torch.Tensor:
        if self._tensors is not None:
            return _pad_tensor(self._tensors[slot_idx], PAD_125S)   # uint8, on GPU
        row_pos = self._valid_row_positions[slot_idx]
        return _pad_tensor(self._load_base_tensor_disk(row_pos), PAD_125S)   # float32, CPU

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
        img_idx = int(row["idx"])

        ys, xs = self._coarse_coords[img_idx]
        if len(ys) == 0:
            cy_coarse = np.random.randint(0, _MASK_H)
            cx_coarse = np.random.randint(0, _MASK_W)
        else:
            k = np.random.randint(len(ys))
            cy_coarse, cx_coarse = int(ys[k]), int(xs[k])

        # Jitter uniformly within the 8x8 block -- statistically identical to sampling
        # over a dense np.repeat-upscaled mask, without ever materializing it.
        cy = _SCALE * cy_coarse + np.random.randint(0, _SCALE)
        cx = _SCALE * cx_coarse + np.random.randint(0, _SCALE)

        padded = self._get_padded_tensor(slot_idx)   # (1, 3200+2*PAD_125S, 4800+2*PAD_125S)

        cy_pad = cy + PAD_125S
        cx_pad = cx + PAD_125S
        half = OVERSIZED_125S // 2
        patch = padded[
            :,
            cy_pad - half : cy_pad + half + 1,
            cx_pad - half : cx_pad + half + 1,
        ]   # (1, OVERSIZED_125S, OVERSIZED_125S)

        if patch.dtype != torch.float32:
            patch = patch.float() / 255.0   # only the small crop is converted, not the canvas

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
