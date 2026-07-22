"""
patch_dataset.py — Patch-sampling Dataset for the patch_v0 pipeline.

Three-way experiment-level split (7 train / 1 val / 2 test per class, seed=99),
independent from all existing models' splits (which use seed=42 and 2-way split).

Key constants:
    PATCH_SIZE = 200   final CNN input after rotation + crop
    OVERSIZED  = 283   ceil(200 * sqrt(2)), ensures clean crop at any rotation angle
    PAD        = 142   ceil(283 / 2), pre-pad border so all image centers are valid
    PATCHES_PER_IMAGE = 20   patches per image per epoch
"""

import json
import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from data_loader import CLASS_MAP, CLASS_NAMES, get_engine, load_metadata
from preprocessing import make_preprocessor

PATCH_SIZE = 200
OVERSIZED = 283
PAD = 142
PATCHES_PER_IMAGE = 20


# ---------------------------------------------------------------------------
# Three-way experiment split
# ---------------------------------------------------------------------------

def split_by_experiment_3way(
    df: pd.DataFrame,
    n_train: int = 7,
    n_val: int = 1,
    n_test: int = 2,
    seed: int = 99,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way experiment-level split: train / val / test.

    All images from one experiment stay in the same partition.
    Stratified: n_train/n_val/n_test experiments per class.
    seed=99 is independent from the existing models' seed=42.

    With 10 experiments per class (5 classes):
        train: 7 × 5 = 35 experiments (~700 images)
        val:   1 × 5 =  5 experiments (~100 images)
        test:  2 × 5 = 10 experiments (~200 images)

    The test split must not influence training or model selection.
    """
    rng = np.random.default_rng(seed)
    train_rows, val_rows, test_rows = [], [], []

    for cls in CLASS_MAP:
        cls_df = df[df["Exp_Type"] == cls]
        exps = cls_df["Experiment"].unique().tolist()
        assert len(exps) == n_train + n_val + n_test, (
            f"Expected {n_train + n_val + n_test} experiments for {cls}, "
            f"got {len(exps)}"
        )
        rng.shuffle(exps)
        train_exps = set(exps[:n_train])
        val_exps = set(exps[n_train : n_train + n_val])
        test_exps = set(exps[n_train + n_val :])

        train_rows.append(cls_df[cls_df["Experiment"].isin(train_exps)])
        val_rows.append(cls_df[cls_df["Experiment"].isin(val_exps)])
        test_rows.append(cls_df[cls_df["Experiment"].isin(test_exps)])

    return (
        pd.concat(train_rows).reset_index(drop=True),
        pd.concat(val_rows).reset_index(drop=True),
        pd.concat(test_rows).reset_index(drop=True),
    )


def save_split_record(
    model_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 99,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    record = {
        "seed": seed,
        "train_indices": sorted(train_df["idx"].tolist()),
        "val_indices": sorted(val_df["idx"].tolist()),
        "test_indices": sorted(test_df["idx"].tolist()),
        "train_experiments": {
            cls: sorted(train_df[train_df["Exp_Type"] == cls]["Experiment"].unique().tolist())
            for cls in CLASS_MAP
        },
        "val_experiments": {
            cls: sorted(val_df[val_df["Exp_Type"] == cls]["Experiment"].unique().tolist())
            for cls in CLASS_MAP
        },
        "test_experiments": {
            cls: sorted(test_df[test_df["Exp_Type"] == cls]["Experiment"].unique().tolist())
            for cls in CLASS_MAP
        },
        "class_distribution": {
            "train": {cls: int((train_df["Exp_Type"] == cls).sum()) for cls in CLASS_MAP},
            "val": {cls: int((val_df["Exp_Type"] == cls).sum()) for cls in CLASS_MAP},
            "test": {cls: int((test_df["Exp_Type"] == cls).sum()) for cls in CLASS_MAP},
        },
    }
    record_path = os.path.join(model_dir, "train_record_patch.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Split record saved to {record_path}")


def load_split_record(
    model_dir: str,
    db_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_path = os.path.join(model_dir, "train_record_patch.json")
    with open(record_path) as f:
        record = json.load(f)

    engine = get_engine(db_path)
    df = load_metadata(engine)

    train_df = df[df["idx"].isin(set(record["train_indices"]))].reset_index(drop=True)
    val_df = df[df["idx"].isin(set(record["val_indices"]))].reset_index(drop=True)
    test_df = df[df["idx"].isin(set(record["test_indices"]))].reset_index(drop=True)
    return train_df, val_df, test_df


def get_or_create_split(
    model_dir: str,
    db_path: str,
    seed: int = 99,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load split from record if it exists; otherwise create and save it."""
    record_path = os.path.join(model_dir, "train_record_patch.json")
    if os.path.exists(record_path):
        print(f"Loading split from {record_path}")
        return load_split_record(model_dir, db_path)

    engine = get_engine(db_path)
    df = load_metadata(engine)
    train_df, val_df, test_df = split_by_experiment_3way(df, seed=seed)
    save_split_record(model_dir, train_df, val_df, test_df, seed=seed)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _pad_tensor(t: torch.Tensor, pad: int) -> torch.Tensor:
    """Add black (zero) border of `pad` pixels on all four sides."""
    return F.pad(t, (pad, pad, pad, pad), mode="constant", value=0.0)


class FibrinPatchDataset(Dataset):
    """Sample random 283×283 oversized patches from 600×400 preprocessed images.

    Each __getitem__ call returns one randomly centered 283×283 patch.
    Rotation + final 200×200 crop happen on the GPU in the training loop.

    Dataset length = len(df) * patches_per_image, so one epoch visits every
    image approximately patches_per_image times.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        photo_dir: str,
        preprocessor: Callable,
        patches_per_image: int = PATCHES_PER_IMAGE,
        preload: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.photo_dir = photo_dir
        self.preprocessor = preprocessor
        self.patches_per_image = patches_per_image
        self.preload = preload

        self._padded_tensors: Optional[List[torch.Tensor]] = None
        if preload:
            print(f"Preloading {len(self.df)} images into RAM...")
            self._padded_tensors = [
                _pad_tensor(self._load_tensor(i), PAD)
                for i in range(len(self.df))
            ]
            print("Preload complete.")

    def _load_tensor(self, base_idx: int) -> torch.Tensor:
        row = self.df.iloc[base_idx]
        img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return self.preprocessor(img_bgr)   # (1, 400, 600) float32

    def _get_padded_tensor(self, base_idx: int) -> torch.Tensor:
        if self._padded_tensors is not None:
            return self._padded_tensors[base_idx]
        return _pad_tensor(self._load_tensor(base_idx), PAD)

    def __len__(self) -> int:
        return len(self.df) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = idx // self.patches_per_image
        tensor = self._get_padded_tensor(base_idx)   # (1, 684, 884)
        label = CLASS_MAP[self.df.iloc[base_idx]["Exp_Type"]]

        # Sample center uniformly over original image dimensions
        cy = torch.randint(0, 400, (1,)).item()
        cx = torch.randint(0, 600, (1,)).item()

        # Map to padded-image coordinates
        cy_pad = cy + PAD
        cx_pad = cx + PAD
        half = OVERSIZED // 2

        patch = tensor[
            :,
            cy_pad - half : cy_pad + half + 1,
            cx_pad - half : cx_pad + half + 1,
        ]   # (1, 283, 283) — always valid due to pre-padding

        return patch, label


# ---------------------------------------------------------------------------
# Balanced sampler (patch-dataset-aware replacement for make_balanced_sampler)
# ---------------------------------------------------------------------------

def get_lc_train_df(
    train_df: pd.DataFrame,
    n_per_class: int,
    seed: int,
    model_dir: str,
) -> pd.DataFrame:
    """Return a subset of train_df limited to n_per_class experiments per class.

    Selection is seeded and saved to lc_experiments.json in model_dir so
    Slurm restarts load the identical subset rather than re-sampling.
    """
    record_path = os.path.join(model_dir, "lc_experiments.json")

    if os.path.exists(record_path):
        with open(record_path) as f:
            record = json.load(f)
        selected = set(record["selected_experiments"])
        print(f"  LC subset loaded ({n_per_class}/class, seed={seed}): {len(selected)} experiments")
        return train_df[train_df["Experiment"].isin(selected)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected, by_class = [], {}
    for cls in sorted(train_df["Exp_Type"].unique()):
        pool = sorted(train_df[train_df["Exp_Type"] == cls]["Experiment"].unique().tolist())
        chosen = sorted(rng.choice(pool, size=n_per_class, replace=False).tolist())
        selected.extend(chosen)
        by_class[cls] = chosen

    os.makedirs(model_dir, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(
            {"n_per_class": n_per_class, "seed": seed,
             "selected_experiments": sorted(selected), "by_class": by_class},
            f, indent=2,
        )
    print(f"  LC subset created ({n_per_class}/class, seed={seed}): {len(selected)} experiments")
    return train_df[train_df["Experiment"].isin(set(selected))].reset_index(drop=True)


def make_patch_sampler(dataset: FibrinPatchDataset) -> WeightedRandomSampler:
    """WeightedRandomSampler that equalises class frequency.

    Weight is inverse-frequency by class, replicated across all patches
    for each base image.
    """
    class_counts = dataset.df["Exp_Type"].value_counts().to_dict()
    total = len(dataset.df)

    weights = []
    for i in range(len(dataset.df)):
        cls = dataset.df.iloc[i]["Exp_Type"]
        w = total / class_counts[cls]
        weights.extend([w] * dataset.patches_per_image)

    return WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True
    )
