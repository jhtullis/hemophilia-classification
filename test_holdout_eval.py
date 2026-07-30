"""
test_holdout_eval.py — Tests for the holdout-testing pipeline pieces that are
verifiable without any trained checkpoints: grid geometry, preprocessing
shapes, plumbing through untrained models, the evaluate_patch.py refactor's
backward compatibility, and holdout_eval_utils' pure-function/fixture-based
helpers.

Run:
    pytest test_holdout_eval.py -v -s
"""

import csv
import json
import os

import cv2
import kornia.augmentation as K
import pandas as pd
import pytest
import torch

from evaluate_patch import PAD, OVERSIZED, PATCH_SIZE, inference_grid_centers, predict_image, score_patch_grid
from holdout_eval_utils import (accuracy_breakdown, assert_identical_test_split,
                                get_best_epoch_metadata, write_json)
from model_patch import FibrinPatchCNN
from model_patch_125s import FibrinPatchCNN125s, OVERSIZED_125S, PAD_125S, PATCH_SIZE_125S
from patch_dataset import _pad_tensor
from preprocessing import make_preprocessor, make_preprocessor_resized

DB_PATH = "data/endpoint10.db"
PHOTO_DIR = "data/photos"
DB_AVAILABLE = os.path.exists(DB_PATH) and os.path.exists(PHOTO_DIR)
SAMPLE_PHOTO = os.path.join(PHOTO_DIR, "0000.JPG")


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def test_125s_grid_matches_scaled_standard_grid():
    standard_centers = inference_grid_centers(H=400, W=600, stride=100)
    resized_centers = inference_grid_centers(H=3200, W=4800, stride=800)

    assert len(standard_centers) == 35
    assert len(resized_centers) == 35
    scaled = [(cy * 8, cx * 8) for cy, cx in standard_centers]
    assert sorted(scaled) == sorted(resized_centers), (
        "125s grid centers should be exactly the 8x-scaled standard grid centers"
    )


# ---------------------------------------------------------------------------
# Preprocessing / padding shapes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="Photo data not available")
def test_make_preprocessor_resized_shape():
    preprocessor = make_preprocessor_resized(pool_factor=1.25)
    img_bgr = cv2.imread(SAMPLE_PHOTO)
    tensor = preprocessor(img_bgr)
    assert tensor.shape == (1, 3200, 4800), f"Expected (1, 3200, 4800), got {tensor.shape}"


def test_pad_tensor_125s_shape():
    tensor = torch.rand(1, 3200, 4800)
    padded = _pad_tensor(tensor, PAD_125S)
    assert padded.shape == (1, 3200 + 2 * PAD_125S, 4800 + 2 * PAD_125S)


# ---------------------------------------------------------------------------
# Untrained-model plumbing smoke tests (no checkpoints required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="Photo data not available")
def test_score_patch_grid_125s_plumbing():
    device = torch.device("cpu")
    model = FibrinPatchCNN125s(num_classes=5, head_type="ce").to(device).eval()

    preprocessor = make_preprocessor_resized(pool_factor=1.25)
    img_bgr = cv2.imread(SAMPLE_PHOTO)
    tensor = preprocessor(img_bgr)
    padded = _pad_tensor(tensor, PAD_125S)

    centers = inference_grid_centers(H=3200, W=4800, stride=800)
    center_crop = K.CenterCrop(PATCH_SIZE_125S)

    scores = score_patch_grid(model, padded, centers, device, center_crop,
                              pad=PAD_125S, oversized=OVERSIZED_125S)
    assert scores.shape == (35, 5)
    assert torch.isfinite(scores).all()


@pytest.mark.skipif(not DB_AVAILABLE, reason="Photo data not available")
def test_predict_image_matches_score_patch_grid_mean():
    """Regression check for the evaluate_patch.py refactor: predict_image's
    output must be bit-identical to score_patch_grid(...).mean(0)."""
    device = torch.device("cpu")
    model = FibrinPatchCNN(num_classes=5, head_type="ce").to(device).eval()

    preprocessor = make_preprocessor()
    img_bgr = cv2.imread(SAMPLE_PHOTO)
    tensor = preprocessor(img_bgr)
    padded = _pad_tensor(tensor, PAD)

    centers = inference_grid_centers()
    center_crop = K.CenterCrop(PATCH_SIZE)

    pred_class, mean_scores = predict_image(model, padded, centers, device, center_crop)
    scores = score_patch_grid(model, padded, centers, device, center_crop)

    assert torch.equal(mean_scores, scores.mean(dim=0))
    assert pred_class == scores.mean(dim=0).argmax().item()


# ---------------------------------------------------------------------------
# holdout_eval_utils — fixture-based tests
# ---------------------------------------------------------------------------

def _write_split_record(model_dir: str, test_indices: list) -> None:
    os.makedirs(model_dir, exist_ok=True)
    record = {
        "seed": 99,
        "train_indices": [1, 2, 3],
        "val_indices": [4, 5],
        "test_indices": test_indices,
        "test_experiments": {"AC3": ["EXP_A", "EXP_B"]},
    }
    with open(os.path.join(model_dir, "train_record_patch.json"), "w") as f:
        json.dump(record, f)


def test_assert_identical_test_split_pass(tmp_path):
    dir_a = str(tmp_path / "fold_a")
    dir_b = str(tmp_path / "fold_b")
    _write_split_record(dir_a, [10, 20, 30])
    _write_split_record(dir_b, [10, 20, 30])

    result = assert_identical_test_split([dir_a, dir_b])
    assert result["test_indices"] == [10, 20, 30]


def test_assert_identical_test_split_mismatch_raises(tmp_path):
    dir_a = str(tmp_path / "fold_a")
    dir_b = str(tmp_path / "fold_b")
    _write_split_record(dir_a, [10, 20, 30])
    _write_split_record(dir_b, [10, 20, 99])

    with pytest.raises(AssertionError):
        assert_identical_test_split([dir_a, dir_b])


def test_get_best_epoch_metadata_from_csv(tmp_path):
    model_dir = str(tmp_path / "model")
    os.makedirs(model_dir)
    csv_path = os.path.join(model_dir, "training_log_full.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_acc"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "val_acc": 0.5})
        writer.writerow({"epoch": 2, "val_acc": 0.8})
        writer.writerow({"epoch": 3, "val_acc": 0.7})

    meta = get_best_epoch_metadata(model_dir)
    assert meta["best_epoch"] == 2
    assert meta["best_val_acc"] == 0.8
    assert meta["source"] == "training_log_full.csv"


def test_get_best_epoch_metadata_falls_back_to_topk_manifest(tmp_path):
    model_dir = str(tmp_path / "model")
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir)
    manifest = {"alltime": [
        {"epoch": 5, "val_acc": 0.6, "period_idx": 0, "path": "topk/x.pth"},
        {"epoch": 9, "val_acc": 0.9, "period_idx": 0, "path": "topk/y.pth"},
    ], "periods": {}}
    with open(os.path.join(ckpt_dir, "topk_manifest.json"), "w") as f:
        json.dump(manifest, f)

    meta = get_best_epoch_metadata(model_dir)
    assert meta["best_epoch"] == 9
    assert meta["best_val_acc"] == 0.9
    assert meta["source"] == "topk_manifest.json"


def test_accuracy_breakdown():
    df = pd.DataFrame([
        {"Exp_Type": "AC3", "correct": 1},
        {"Exp_Type": "AC3", "correct": 0},
        {"Exp_Type": "NC1", "correct": 1},
    ])
    overall, per_class = accuracy_breakdown(df, ["AC3", "NC1"])
    assert abs(overall - (2 / 3)) < 1e-9
    assert abs(per_class["AC3"] - 0.5) < 1e-9
    assert per_class["NC1"] == 1.0


def test_write_json_roundtrip(tmp_path):
    path = str(tmp_path / "nested" / "out.json")
    write_json(path, {"a": 1, "b": [1, 2, 3]})
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == {"a": 1, "b": [1, 2, 3]}
