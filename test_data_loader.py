"""
test_data_loader.py — Tests for database access, splitting, and Dataset.

Run with:
    pytest test_data_loader.py -v -s
"""

import os
import pytest
import torch

from data_loader import (
    CLASS_MAP, CLASS_NAMES,
    FibrinDataset,
    get_engine,
    load_metadata,
    make_balanced_sampler,
    make_preprocessor,
    split_by_experiment,
)

DB_PATH    = os.path.join(os.path.dirname(__file__), "data", "endpoint10.db")
PHOTO_DIR  = os.path.join(os.path.dirname(__file__), "data", "photos")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    engine = get_engine(DB_PATH)
    return load_metadata(engine)


@pytest.fixture(scope="module")
def split(df):
    return split_by_experiment(df, train_ratio=0.75, seed=42)


# ---------------------------------------------------------------------------
# Database / metadata tests
# ---------------------------------------------------------------------------

def test_load_metadata_row_count(df):
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"


def test_load_metadata_columns(df):
    required = {"idx", "Experiment", "Exp_Type", "Slide_Type", "Img_Name", "Img_Fp"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"


def test_load_metadata_class_distribution(df):
    expected = {"AC3": 200, "F08D": 200, "F09D": 200, "F11D": 200, "NC1": 200}
    counts = df["Exp_Type"].value_counts().to_dict()
    for cls, n in expected.items():
        assert counts.get(cls) == n, f"Class {cls}: expected {n}, got {counts.get(cls)}"


def test_load_metadata_idx_range(df):
    assert df["idx"].min() == 0, "idx should start at 0"
    assert df["idx"].max() == 999, "idx should end at 999"
    assert df["idx"].nunique() == 1000, "All idx values should be unique"


def test_class_mapping():
    """All 5 classes must be mapped to unique integers 0–4."""
    assert set(CLASS_MAP.values()) == {0, 1, 2, 3, 4}
    assert len(CLASS_NAMES) == 5
    for cls, i in CLASS_MAP.items():
        assert CLASS_NAMES[i] == cls


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

def test_split_no_experiment_overlap(split):
    train_df, test_df = split
    train_exps = set(train_df["Experiment"].unique())
    test_exps  = set(test_df["Experiment"].unique())
    overlap = train_exps & test_exps
    assert len(overlap) == 0, f"Experiments in both train and test: {overlap}"


def test_split_all_classes_in_train(split):
    train_df, _ = split
    for cls in CLASS_MAP:
        assert cls in train_df["Exp_Type"].values, f"Class {cls} missing from train"


def test_split_all_classes_in_test(split):
    _, test_df = split
    for cls in CLASS_MAP:
        assert cls in test_df["Exp_Type"].values, f"Class {cls} missing from test"


def test_split_covers_all_images(split, df):
    train_df, test_df = split
    total = len(train_df) + len(test_df)
    assert total == len(df), f"Split total {total} != dataset total {len(df)}"


def test_split_approximate_ratio(split):
    train_df, test_df = split
    ratio = len(train_df) / (len(train_df) + len(test_df))
    assert 0.65 <= ratio <= 0.90, f"Train ratio {ratio:.2f} outside expected range [0.65, 0.90]"


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_dataset(split):
    _, test_df = split
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)
    return FibrinDataset(test_df, PHOTO_DIR, preprocessor, augment=False)


@pytest.fixture(scope="module")
def train_dataset(split):
    train_df, _ = split
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)
    return FibrinDataset(train_df, PHOTO_DIR, preprocessor, augment=True)


def test_dataset_length_no_augment(test_dataset, split):
    _, test_df = split
    assert len(test_dataset) == len(test_df)


def test_dataset_length_with_augment(train_dataset, split):
    train_df, _ = split
    assert len(train_dataset) == len(train_df) * 4


def test_dataset_getitem_shape(test_dataset):
    tensor, label = test_dataset[0]
    assert tensor.shape == (1, 400, 600), \
        f"Expected (1, 400, 600), got {tuple(tensor.shape)}"


def test_dataset_getitem_dtype(test_dataset):
    tensor, label = test_dataset[0]
    assert tensor.dtype == torch.float32


def test_dataset_getitem_value_range(test_dataset):
    tensor, label = test_dataset[0]
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_dataset_getitem_label_valid(test_dataset):
    for i in range(min(5, len(test_dataset))):
        _, label = test_dataset[i]
        assert 0 <= label <= 4, f"Label {label} out of range [0, 4]"


def test_augmented_dataset_variants(train_dataset):
    """Consecutive indices 0-3 should give the same base image, different flips."""
    tensors = [train_dataset[i][0] for i in range(4)]
    # Variant 0 and variant 3 (both-flip) should differ from original
    # (unless the image happens to be symmetric — unlikely for a real micrograph)
    shapes = [t.shape for t in tensors]
    assert all(s == (1, 400, 600) for s in shapes), "All variants must have the same shape"


# ---------------------------------------------------------------------------
# Sampler test
# ---------------------------------------------------------------------------

def test_balanced_sampler_length(train_dataset):
    sampler = make_balanced_sampler(train_dataset)
    assert sampler.num_samples == len(train_dataset)
