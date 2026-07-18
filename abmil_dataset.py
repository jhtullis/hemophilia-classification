"""
abmil_dataset.py — Image-level "bag" Dataset for ABMIL training/evaluation.

One __getitem__ call returns ALL patches sampled for one image (a bag) plus
its single image-level label — unlike patch_dataset.FibrinPatchDataset and
masked_patch_dataset.MaskedPatchDataset, which both yield flat (patch, label)
pairs with no per-image grouping. No existing dataset in this repo has bag
structure, so this is new code, not a wrapper around either of those classes.

Two sampling modes:
    mask_aware=True  (default; use for mpatch_* backbones) — centers are drawn
        from the same patch-center validity maps (masks/patch_centers/<version>/)
        used to train the backbone. This matters: mpatch_v0_f was trained with
        uniform_fraction=0.00, i.e. it has essentially never seen a patch
        outside the foreground mask. Feeding it uniformly-sampled (possibly
        background) patches at ABMIL-training time would be out-of-distribution
        for the frozen embedding extractor.
    mask_aware=False (use for plain patch_v1*/patch_v2* backbones) — centers
        sampled uniformly over the whole (400, 600) preprocessed image, no
        mask files required, matching FibrinPatchDataset's sampling.

Returns OVERSIZED (283x283) patches, not pre-cropped, so the caller applies
the same GPU-side augmentation used for backbone training (train) or a plain
deterministic center-crop (val/eval).
"""

import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_loader import CLASS_MAP
from patch_dataset import OVERSIZED, PAD, _pad_tensor


def _load_center_mask_arr(mask_dir: str, patch_center_version: str, idx: int) -> np.ndarray:
    """Load a precomputed patch-center validity map (see compute_patch_centers.py).

    Deliberately not imported from masked_patch_dataset.py: that logic is a
    private instance method tied to MaskedPatchDataset's flat patch-slot
    indexing, which doesn't fit this dataset's per-image bag structure. The
    on-disk format (grayscale PNG, threshold >127) is stable and shared.
    """
    path = os.path.join(mask_dir, "patch_centers", patch_center_version, f"{idx:04d}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Patch-center mask not found: {path}\n"
            f"Run: python compute_patch_centers.py --mask-versions <version>"
        )
    return img > 127   # bool (H, W)


def _load_flagged_empty(mask_dir: str, mask_version: str) -> set:
    flagged_path = os.path.join(mask_dir, mask_version, "flagged_empty.txt")
    excluded: set = set()
    if os.path.exists(flagged_path):
        with open(flagged_path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        excluded.add(int(line.split()[0]))
                    except ValueError:
                        pass
    return excluded


class FibrinImageBagDataset(Dataset):
    """Image-level bag dataset for ABMIL.

    Args:
        df:                     Metadata DataFrame (one row per image), e.g.
                                 from patch_dataset.load_split_record.
        photo_dir:               Directory of JPEG images.
        preprocessor:            From make_preprocessor(); img_bgr -> (1,400,600) tensor.
        patches_per_image:        Bag size N (default 50). Independent of whatever
                                  patches_per_image the backbone was trained with.
        mask_aware:               See module docstring.
        mask_dir, mask_version,
        patch_center_version:     Required when mask_aware=True.
        resample_each_epoch:      True (default): centers re-sampled on every
                                  __getitem__ call (bag-composition augmentation
                                  across epochs). False: centers sampled once at
                                  init with `seed` and reused every call — use for
                                  validation so val accuracy isn't noisy from
                                  resampling epoch to epoch.
        seed:                     RNG seed, used when resample_each_epoch=False.
        preload:                  Preload base image tensors (and, if mask_aware,
                                  center-validity maps) into RAM.
        exclude_flagged_empty:    If True and mask_aware, drop images listed in
                                  masks/<mask_version>/flagged_empty.txt or with
                                  zero valid centers (mirrors MaskedPatchDataset).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        photo_dir: str,
        preprocessor: Callable,
        patches_per_image: int = 50,
        mask_aware: bool = True,
        mask_dir: Optional[str] = None,
        mask_version: Optional[str] = None,
        patch_center_version: Optional[str] = None,
        resample_each_epoch: bool = True,
        seed: int = 99,
        preload: bool = True,
        exclude_flagged_empty: bool = True,
    ) -> None:
        self.photo_dir = photo_dir
        self.preprocessor = preprocessor
        self.patches_per_image = patches_per_image
        self.mask_aware = mask_aware
        self.resample_each_epoch = resample_each_epoch
        self.seed = seed

        self.mask_dir = mask_dir
        self.patch_center_version = patch_center_version
        self.mask_version: Optional[str] = None

        if mask_aware:
            if not (mask_dir and mask_version and patch_center_version):
                raise ValueError(
                    "mask_dir, mask_version, and patch_center_version are "
                    "required when mask_aware=True."
                )
            self.mask_version = mask_version if mask_version.startswith("v_") else f"v_{mask_version}"

        df = df.reset_index(drop=True)

        if mask_aware and exclude_flagged_empty:
            excluded = _load_flagged_empty(mask_dir, self.mask_version)
            stats_path = os.path.join(mask_dir, "patch_centers", patch_center_version, "stats.csv")
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"Patch-center stats not found: {stats_path}\n"
                    f"Run: python compute_patch_centers.py --mask-versions {self.mask_version[2:]}"
                )
            n_valid = pd.read_csv(stats_path).set_index("idx")["n_valid_centers"].to_dict()
            keep = df["idx"].apply(lambda i: int(i) not in excluded and n_valid.get(int(i), 0) > 0)
            n_dropped = int((~keep).sum())
            df = df[keep].reset_index(drop=True)
            if n_dropped:
                print(f"FibrinImageBagDataset: excluded {n_dropped} empty/flagged images "
                      f"({len(df)} remain)")

        self.df = df
        if len(self.df) == 0:
            raise ValueError("FibrinImageBagDataset: no valid images after filtering.")

        self._tensors: Optional[List[torch.Tensor]] = None
        self._center_masks: Optional[List[np.ndarray]] = None
        if preload:
            print(f"FibrinImageBagDataset: preloading {len(self.df)} images...")
            self._tensors = [self._load_padded_tensor(i) for i in range(len(self.df))]
            if mask_aware:
                self._center_masks = [
                    _load_center_mask_arr(mask_dir, patch_center_version, int(self.df.iloc[i]["idx"]))
                    for i in range(len(self.df))
                ]
            print("FibrinImageBagDataset: preload complete.")

        self._fixed_centers: Optional[List[List[Tuple[int, int]]]] = None
        if not resample_each_epoch:
            rng = np.random.default_rng(seed)
            self._fixed_centers = [self._sample_centers(i, rng) for i in range(len(self.df))]

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_padded_tensor(self, row_pos: int) -> torch.Tensor:
        row = self.df.iloc[row_pos]
        img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        tensor = self.preprocessor(img_bgr)   # (1, 400, 600)
        return _pad_tensor(tensor, PAD)

    def _get_padded_tensor(self, row_pos: int) -> torch.Tensor:
        if self._tensors is not None:
            return self._tensors[row_pos]
        return self._load_padded_tensor(row_pos)

    def _get_center_mask(self, row_pos: int) -> np.ndarray:
        if self._center_masks is not None:
            return self._center_masks[row_pos]
        idx = int(self.df.iloc[row_pos]["idx"])
        return _load_center_mask_arr(self.mask_dir, self.patch_center_version, idx)

    def _sample_centers(self, row_pos: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
        if self.mask_aware:
            center_mask = self._get_center_mask(row_pos)
            valid_ys, valid_xs = np.where(center_mask)
            if len(valid_ys) == 0:
                # Should not occur post-filtering; fall back to uniform.
                cys = rng.integers(0, 400, size=self.patches_per_image)
                cxs = rng.integers(0, 600, size=self.patches_per_image)
            else:
                k = rng.integers(0, len(valid_ys), size=self.patches_per_image)
                cys, cxs = valid_ys[k], valid_xs[k]
        else:
            cys = rng.integers(0, 400, size=self.patches_per_image)
            cxs = rng.integers(0, 600, size=self.patches_per_image)
        return list(zip(cys.tolist(), cxs.tolist()))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        label = CLASS_MAP[row["Exp_Type"]]
        img_idx = int(row["idx"])

        padded = self._get_padded_tensor(idx)   # (1, 400+2*PAD, 600+2*PAD)

        if self._fixed_centers is not None:
            centers = self._fixed_centers[idx]
        else:
            centers = self._sample_centers(idx, np.random.default_rng())

        half = OVERSIZED // 2
        patches = []
        for cy, cx in centers:
            cy_pad, cx_pad = cy + PAD, cx + PAD
            patch = padded[
                :,
                cy_pad - half : cy_pad + half + 1,
                cx_pad - half : cx_pad + half + 1,
            ]   # (1, 283, 283)
            patches.append(patch)

        bag = torch.stack(patches)   # (N, 1, 283, 283)
        return bag, label, img_idx
