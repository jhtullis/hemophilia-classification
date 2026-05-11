"""
data_loader.py — Database access, Dataset, and DataLoader construction.

Responsibilities:
  - Load image metadata from the SQLite database via SQLAlchemy
  - Split data at the experiment level (no leakage between train and validation)
  - Provide a PyTorch Dataset that reads images on demand and applies
    preprocessing + optional augmentation
  - Build class-balanced DataLoaders for training and validation
"""

import json
import os
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from augmentation import augment
from preprocessing import make_preprocessor

# Integer label for each phenotype class
CLASS_MAP: Dict[str, int] = {
    "AC3": 0,
    "F08D": 1,
    "F09D": 2,
    "F11D": 3,
    "NC1": 4,
}
CLASS_NAMES: List[str] = [k for k in CLASS_MAP]  # ordered by int value


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_engine(db_path: str):
    """Create a SQLAlchemy engine for the SQLite database."""
    return create_engine(f"sqlite:///{db_path}")


def load_metadata(engine) -> pd.DataFrame:
    """Load image metadata from Images_Endpoint_10.

    The filename integer equals (rowid - 1), so image 0000.JPG is row 1, etc.

    Returns:
        DataFrame with columns: idx, Experiment, Exp_Type, Slide_Type,
        Img_Name, Img_Fp.  idx is the 0-based row index used to construct
        the photo filename: f"{idx:04d}.JPG".
    """
    query = "SELECT rowid - 1 AS idx, Experiment, Exp_Type, Slide_Type, Img_Name, Img_Fp FROM Images_Endpoint_10"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_by_experiment(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Experiment-level train/validation split, stratified by class.

    All images from the same experiment stay in the same partition
    to prevent data leakage between train and validation sets.

    A minimum of 1 validation experiment per class is guaranteed regardless
    of the train_ratio; this may slightly push the effective train ratio above
    the target for classes with very few experiments.

    Args:
        df:          Metadata DataFrame from load_metadata().
        train_ratio: Approximate fraction of images in the training set.
        seed:        Random seed for reproducible splits.

    Returns:
        (train_df, val_df) — non-overlapping subsets of df.
    """
    rng = np.random.default_rng(seed)
    train_rows, val_rows = [], []

    for cls in CLASS_MAP:
        cls_df = df[df["Exp_Type"] == cls]
        experiments = cls_df["Experiment"].unique().tolist()
        rng.shuffle(experiments)

        n_train = max(1, round(len(experiments) * train_ratio))
        n_train = min(n_train, len(experiments) - 1)  # keep ≥1 for validation

        train_exps = set(experiments[:n_train])
        train_rows.append(cls_df[cls_df["Experiment"].isin(train_exps)])
        val_rows.append(cls_df[~cls_df["Experiment"].isin(train_exps)])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    return train_df, val_df


def save_train_record(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      path: str, **kwargs) -> None:
    """Save a JSON record of which images were used for training.

    Args:
        train_df: Training split DataFrame.
        val_df:   Validation split DataFrame.
        path:     Output JSON file path.
        **kwargs: Additional metadata to include (e.g. seed, gray_method).
    """
    record = {
        "split_date": str(date.today()),
        **kwargs,
        "train_experiments": {
            cls: sorted(train_df[train_df["Exp_Type"] == cls]["Experiment"].unique().tolist())
            for cls in CLASS_MAP
        },
        "val_experiments": {
            cls: sorted(val_df[val_df["Exp_Type"] == cls]["Experiment"].unique().tolist())
            for cls in CLASS_MAP
        },
        "train_indices": sorted(train_df["idx"].tolist()),
        "val_indices": sorted(val_df["idx"].tolist()),
        "class_distribution": {
            "train": {cls: int((train_df["Exp_Type"] == cls).sum()) for cls in CLASS_MAP},
            "val":   {cls: int((val_df["Exp_Type"] == cls).sum())   for cls in CLASS_MAP},
        },
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FibrinDataset(Dataset):
    """PyTorch Dataset for fibrin clot images.

    By default images are loaded from disk on demand. When preload=True all
    images are preprocessed at construction time and cached in RAM — this
    eliminates per-epoch disk I/O at the cost of memory (~700 MB for the
    original 10× min-pool pipeline over ~850 images).

    When augment=True the dataset is logically 4× the size of df: each base
    image appears as 4 consecutive entries (variants 0–3).  The variant index
    is derived from the global index: variant = idx % 4.

    Args:
        df:          Metadata DataFrame (subset of load_metadata() output).
        photo_dir:   Directory containing 0000.JPG … 0858.JPG.
        preprocessor: Callable img_bgr → torch.Tensor (from make_preprocessor).
        augment:     If True, expose 4× the images with flip augmentations.
        preload:     If True, preprocess all images into RAM at init time.
    """

    def __init__(self, df: pd.DataFrame, photo_dir: str,
                 preprocessor: Callable, augment: bool = False,
                 preload: bool = False):
        self.df = df.reset_index(drop=True)
        self.photo_dir = photo_dir
        self.preprocessor = preprocessor
        self.do_augment = augment
        self.multiplier = 4 if augment else 1
        self._cache: Optional[List[torch.Tensor]] = None

        if preload:
            self._cache = []
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    raise FileNotFoundError(f"Image not found: {img_path}")
                self._cache.append(self.preprocessor(img_bgr))

    def __len__(self) -> int:
        return len(self.df) * self.multiplier

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = idx // self.multiplier
        variant = idx % self.multiplier

        row = self.df.iloc[base_idx]

        if self._cache is not None:
            tensor = self._cache[base_idx]
        else:
            img_path = os.path.join(self.photo_dir, f"{int(row['idx']):04d}.JPG")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            tensor = self.preprocessor(img_bgr)

        if self.do_augment:
            tensor = augment(tensor, variant)

        label = CLASS_MAP[row["Exp_Type"]]
        return tensor, label


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

def make_balanced_sampler(dataset: FibrinDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that equalises class frequency.

    Each sample receives weight proportional to the inverse of its class
    count in the dataset (before augmentation expansion, since augmentation
    is applied uniformly across all classes).

    Returns:
        WeightedRandomSampler configured to draw len(dataset) samples.
    """
    # Count base images (no augmentation factor) per class
    class_counts = dataset.df["Exp_Type"].value_counts().to_dict()
    total = len(dataset.df)

    # Assign per-sample weight (augmented index → base row → class weight)
    weights = []
    m = dataset.multiplier
    for i in range(len(dataset.df)):
        cls = dataset.df.iloc[i]["Exp_Type"]
        w = total / class_counts[cls]
        weights.extend([w] * m)   # same weight for all 4 augmentation variants

    return WeightedRandomSampler(weights=weights, num_samples=len(dataset),
                                 replacement=True)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_split_from_record(
    record_path: str,
    db_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the train/validation DataFrames from a saved train_record.json.

    Reads the stored integer image indices (not a random re-split) so the
    held-out validation set is always identical regardless of when this is called.
    Accepts both new-style val_* keys and legacy test_* keys for backward
    compatibility with records written before the rename.

    Args:
        record_path: Path to a train_record.json produced by save_train_record().
        db_path:     Path to the SQLite database (for load_metadata).

    Returns:
        (train_df, val_df) — same structure as split_by_experiment output.
    """
    with open(record_path) as f:
        record = json.load(f)

    # Support both new val_* keys and legacy test_* keys
    train_indices = set(record["train_indices"])
    val_indices = set(record.get("val_indices", record.get("test_indices", [])))

    engine = get_engine(db_path)
    df = load_metadata(engine)

    train_df = df[df["idx"].isin(train_indices)].reset_index(drop=True)
    val_df   = df[df["idx"].isin(val_indices)].reset_index(drop=True)

    assert len(train_df) + len(val_df) == len(train_indices) + len(val_indices), (
        f"Record/DB mismatch: expected {len(train_indices)+len(val_indices)} images, "
        f"got {len(train_df)+len(val_df)}"
    )
    return train_df, val_df


def filter_classes(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    """Return a subset of df containing only the specified classes.

    Args:
        df:      Metadata DataFrame (must have an 'Exp_Type' column).
        classes: Class names to keep.  Each must be a key in CLASS_MAP.

    Returns:
        Filtered DataFrame, index reset.

    Raises:
        ValueError: if any element of classes is not in CLASS_MAP.
    """
    unknown = [c for c in classes if c not in CLASS_MAP]
    if unknown:
        raise ValueError(f"Unknown class(es): {unknown}. Valid: {list(CLASS_MAP)}")
    return df[df["Exp_Type"].isin(classes)].reset_index(drop=True)


def create_dataloaders(
    db_path: str,
    photo_dir: str,
    batch_size: int = 16,
    train_ratio: float = 0.75,
    seed: int = 42,
    gray_method: str = "lab_l",
    pool_factor: int = 10,
    num_workers: int = None,
    train_record_path: str = "train_record.json",
    preload: bool = False,
    force_resplit: bool = False,
) -> Tuple[DataLoader, DataLoader, dict]:
    """End-to-end factory: DB → split → datasets → dataloaders.

    If train_record_path already exists and force_resplit is False, the saved
    split is reloaded rather than regenerated.  Pass force_resplit=True (or
    --force-resplit on the CLI) to discard the existing split and create a new
    one, which will overwrite the record file.

    Args:
        db_path:           Path to endpoint10.db.
        photo_dir:         Directory containing numbered JPEGs.
        batch_size:        Images per mini-batch.
        train_ratio:       Approximate fraction assigned to training.
        seed:              Random seed for the split.
        gray_method:       Grayscale conversion method (see preprocessing.py).
        pool_factor:       Min-pool downsampling factor.
        num_workers:       DataLoader worker processes. Defaults to
                           min(8, max(4, cpu_count - 2)).
        train_record_path: Where to write (or read) the JSON training record.
        preload:           If True, load all images into RAM at startup.
        force_resplit:     If True, always regenerate the split even if a
                           record file already exists.

    Returns:
        (train_loader, val_loader, metadata_dict)
        metadata_dict contains class_names, class_counts, split sizes.
    """
    if num_workers is None:
        num_workers = min(8, max(4, (os.cpu_count() or 4) - 2))

    if not force_resplit and os.path.exists(train_record_path):
        print(f"Reusing existing split from {train_record_path}")
        train_df, val_df = load_split_from_record(train_record_path, db_path)
    else:
        engine = get_engine(db_path)
        df = load_metadata(engine)
        train_df, val_df = split_by_experiment(df, train_ratio=train_ratio, seed=seed)
        save_train_record(train_df, val_df, train_record_path,
                          seed=seed, train_ratio=train_ratio,
                          gray_method=gray_method, pool_factor=pool_factor)

    preprocessor = make_preprocessor(gray_method=gray_method, pool_factor=pool_factor)

    train_ds = FibrinDataset(train_df, photo_dir, preprocessor,
                             augment=True, preload=preload)
    val_ds   = FibrinDataset(val_df,   photo_dir, preprocessor,
                             augment=False, preload=preload)

    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=False, persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False, persistent_workers=(num_workers > 0))

    meta = {
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_base_size": len(train_df),
        "val_base_size": len(val_df),
        "train_class_counts": {
            cls: int((train_df["Exp_Type"] == cls).sum()) for cls in CLASS_MAP
        },
        "val_class_counts": {
            cls: int((val_df["Exp_Type"] == cls).sum()) for cls in CLASS_MAP
        },
    }
    return train_loader, val_loader, meta
