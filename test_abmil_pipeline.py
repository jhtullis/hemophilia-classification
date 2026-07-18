"""
test_abmil_pipeline.py — Structural tests for the ABMIL pipeline
(abmil_model.py, abmil_dataset.py, compare_voting_methods.py).

Follows test_patch_pipeline.py conventions: flat functions (no fixtures),
Dataset.__new__() + manual attribute injection for synthetic data (no disk
I/O), a DB_AVAILABLE skip-gate for tests needing real data.

Run:
    pytest test_abmil_pipeline.py -v -s
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from abmil_dataset import FibrinImageBagDataset
from abmil_model import FibrinABMIL, GatedAttentionMIL
from model_patch import FibrinPatchCNN
from patch_dataset import OVERSIZED, PAD, PATCH_SIZE, _pad_tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = "data/endpoint10.db"
PHOTO_DIR = "data/photos"
DB_AVAILABLE = os.path.exists(DB_PATH) and os.path.exists(PHOTO_DIR)
BACKBONE_PATH = "models/mpatch_v0_f/best_model.pth"
BACKBONE_AVAILABLE = os.path.exists(BACKBONE_PATH)


def _make_abmil(head_type: str = "ce", freeze_backbone: bool = True) -> FibrinABMIL:
    backbone = FibrinPatchCNN(num_classes=5, head_type=head_type)
    return FibrinABMIL(
        backbone=backbone, num_classes=5, head_type=head_type,
        freeze_backbone=freeze_backbone, attn_hidden_dim=16,
    )


def _random_bag(n: int) -> torch.Tensor:
    return torch.rand(n, 1, PATCH_SIZE, PATCH_SIZE)


def _make_fake_bag_dataset(n_images: int = 6, patches_per_image: int = 10,
                            mask_aware: bool = False, resample_each_epoch: bool = True):
    """FibrinImageBagDataset backed by synthetic random tensors (no disk access)."""
    from data_loader import CLASS_NAMES

    rows = []
    for i in range(n_images):
        cls = CLASS_NAMES[i % len(CLASS_NAMES)]
        rows.append({"idx": i, "Exp_Type": cls, "Experiment": f"EXP_{cls}_{i}"})
    df = pd.DataFrame(rows)

    ds = FibrinImageBagDataset.__new__(FibrinImageBagDataset)
    ds.df = df
    ds.photo_dir = ""
    ds.preprocessor = None
    ds.patches_per_image = patches_per_image
    ds.mask_aware = mask_aware
    ds.resample_each_epoch = resample_each_epoch
    ds.seed = 99
    ds.mask_dir = None
    ds.patch_center_version = None
    ds.mask_version = None

    ds._tensors = [_pad_tensor(torch.rand(1, 400, 600), PAD) for _ in range(n_images)]
    ds._center_masks = None
    ds._fixed_centers = None

    if not resample_each_epoch:
        rng = np.random.default_rng(ds.seed)
        ds._fixed_centers = [ds._sample_centers(i, rng) for i in range(n_images)]

    return ds


def _make_fake_center_mask(valid_y_range=(150, 250), valid_x_range=(200, 400)) -> np.ndarray:
    mask = np.zeros((400, 600), dtype=bool)
    mask[valid_y_range[0]:valid_y_range[1], valid_x_range[0]:valid_x_range[1]] = True
    return mask


# ---------------------------------------------------------------------------
# FibrinABMIL / GatedAttentionMIL — shape, invariance, and freeze behavior
# ---------------------------------------------------------------------------

def test_variable_n_forward():
    model = _make_abmil(head_type="ce")
    for n in (10, 200):
        out, attn = model(_random_bag(n))
        assert out.shape == (5,), f"Expected (5,), got {out.shape} for N={n}"
        assert attn.shape == (n,), f"Expected ({n},), got {attn.shape} for N={n}"


def test_attention_weights_sum_to_one():
    model = _make_abmil(head_type="ce")
    _, attn = model(_random_bag(37))
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5), (
        f"Attention weights should sum to 1.0, got {attn.sum().item()}"
    )


def test_permutation_invariance():
    model = _make_abmil(head_type="ce")
    model.eval()
    bag = _random_bag(25)
    perm = torch.randperm(25)

    with torch.no_grad():
        out1, attn1 = model(bag)
        out2, attn2 = model(bag[perm])

    assert torch.allclose(out1, out2, atol=1e-5), "Output should be permutation-invariant"
    assert torch.allclose(attn1[perm], attn2, atol=1e-5), (
        "Attention weights should permute consistently with the input"
    )


def test_backbone_frozen_by_default():
    model = _make_abmil(head_type="ce", freeze_backbone=True)
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone params should have requires_grad=False"

    # train()-override fix: outer .train() must not flip the backbone to train mode.
    model.train()
    assert model.backbone.training is False, (
        "backbone should remain in eval() mode even after model.train()"
    )

    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    out, _ = model(_random_bag(20))
    loss = out.sum()
    loss.backward()
    optimizer.step()

    after = model.backbone.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), (
            f"Backbone param/buffer '{k}' changed after an optimizer step despite freezing"
        )


def test_ce_head_runs():
    model = _make_abmil(head_type="ce")
    out, attn = model(_random_bag(15))
    assert out.shape == (5,)
    assert attn.shape == (15,)


def test_cosine_head_output_range():
    model = _make_abmil(head_type="cosine")
    out, _ = model(_random_bag(15))
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# FibrinImageBagDataset
# ---------------------------------------------------------------------------

def test_dataset_bag_shape():
    ds = _make_fake_bag_dataset(n_images=5, patches_per_image=12, mask_aware=False)
    bag, label, img_idx = ds[0]
    assert bag.shape == (12, 1, OVERSIZED, OVERSIZED), (
        f"Expected (12, 1, {OVERSIZED}, {OVERSIZED}), got {bag.shape}"
    )
    assert isinstance(label, (int, np.integer))
    assert 0 <= label < 5
    assert img_idx == 0


def test_dataset_mask_aware_centers_within_valid_mask():
    ds = _make_fake_bag_dataset(n_images=3, patches_per_image=50, mask_aware=True)
    valid_mask = _make_fake_center_mask()
    ds._center_masks = [valid_mask for _ in range(3)]

    rng = np.random.default_rng(0)
    centers = ds._sample_centers(0, rng)
    assert len(centers) == 50
    for cy, cx in centers:
        assert valid_mask[cy, cx], f"Center ({cy},{cx}) sampled outside the valid mask"


def test_dataset_resample_each_epoch_varies_centers():
    ds = _make_fake_bag_dataset(n_images=2, patches_per_image=30, mask_aware=False,
                                 resample_each_epoch=True)
    bag1, _, _ = ds[0]
    bag2, _, _ = ds[0]
    # With independent random sampling each call, two draws of 30 patches
    # from a continuous 400x600 space should essentially never be identical.
    assert not torch.equal(bag1, bag2), "Expected different bags across calls when resample_each_epoch=True"


def test_dataset_fixed_centers_are_reproducible():
    ds = _make_fake_bag_dataset(n_images=2, patches_per_image=30, mask_aware=False,
                                 resample_each_epoch=False)
    bag1, _, _ = ds[0]
    bag2, _, _ = ds[0]
    assert torch.equal(bag1, bag2), "Expected identical bags across calls when resample_each_epoch=False"


# ---------------------------------------------------------------------------
# compare_voting_methods — McNemar's test
# ---------------------------------------------------------------------------

def test_mcnemar_known_values():
    from compare_voting_methods import _binom_two_sided_pvalue, mcnemar_test

    # b == c == 0 -> no discordant pairs -> p == 1.0
    assert _binom_two_sided_pvalue(0, 0) == 1.0

    # Symmetry: mcnemar_test(a, b) and mcnemar_test(b, a) should give the
    # same p-value (b and c swap, but the test is symmetric in b/c).
    correct_a = np.array([True, True, False, False, True, False, True, False])
    correct_b = np.array([True, False, True, False, False, True, True, False])
    res_ab = mcnemar_test(correct_a, correct_b)
    res_ba = mcnemar_test(correct_b, correct_a)
    assert res_ab["b"] == res_ba["c"]
    assert res_ab["c"] == res_ba["b"]
    assert res_ab["p_value"] == pytest.approx(res_ba["p_value"])

    # Hand-checked case: b=1, c=9 (n=10) -> two-sided exact binomial p-value.
    import math
    n, k = 10, 1
    p_le_k = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    expected = min(1.0, 2 * p_le_k)
    assert _binom_two_sided_pvalue(1, 9) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    from checkpoint_manager import load_latest_checkpoint, save_checkpoint

    model = _make_abmil(head_type="ce")
    model_dir = str(tmp_path)

    state = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "best_acc": 0.5,
        "history": [],
        "model_type": "abmil_test",
        "phase": "training",
        "config": {},
        "wandb_run_id": None,
    }
    save_checkpoint(state, model_dir, epoch=3, is_best=True, max_epochs=10)

    ckpt = load_latest_checkpoint(model_dir)
    assert ckpt is not None
    assert ckpt["epoch"] == 3

    loaded_model = _make_abmil(head_type="ce")
    loaded_model.load_state_dict(ckpt["model_state_dict"])
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded_model.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"Mismatch after roundtrip for {k1}"

    # best_model.pth should be the raw model_state_dict (matches train_patch.py's
    # / checkpoint_manager's convention — see train_abmil.py's load_frozen_backbone
    # and evaluate_abmil.py's load_abmil_model, which both handle this case).
    best_path = os.path.join(model_dir, "best_model.pth")
    assert os.path.exists(best_path)
    best_sd = torch.load(best_path, weights_only=True)
    assert "model_state_dict" not in best_sd   # it IS the state dict, not wrapped
    assert set(best_sd.keys()) == set(model.state_dict().keys())


def test_orig_mod_prefix_stripping():
    """Exercises the `_orig_mod.` (torch.compile) prefix-stripping pattern used
    identically in train_abmil.py's resume path and evaluate_abmil.py's
    load_abmil_model — a fake torch.compile-wrapped state_dict should load
    cleanly after stripping."""
    model = _make_abmil(head_type="ce")
    sd = model.state_dict()
    compiled_sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

    # Same stripping logic used in train_abmil.py / evaluate_abmil.py.
    if any(k.startswith("_orig_mod.") for k in compiled_sd):
        stripped = {k[len("_orig_mod."):]: v for k, v in compiled_sd.items()}
    else:
        stripped = compiled_sd

    fresh_model = _make_abmil(head_type="ce")
    fresh_model.load_state_dict(stripped)   # should not raise
    for k in sd:
        assert torch.equal(sd[k], fresh_model.state_dict()[k])


# ---------------------------------------------------------------------------
# End-to-end regression check against the existing mpatch_v0_f soft-vote baseline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DB_AVAILABLE and BACKBONE_AVAILABLE),
    reason="Requires data/endpoint10.db, data/photos/, and models/mpatch_v0_f/best_model.pth",
)
def test_evaluate_reproduces_soft_vote_baseline():
    """evaluate_abmil.py's soft-vote path reuses evaluate_patch.py's exact grid
    and raw-score-averaging convention, so it should reproduce the predictions
    already on record in models/mpatch_v0_f/analysis/val/per_image_predictions.csv
    for the real trained backbone (independent of how well ABMIL's attention
    head is trained, since soft/hard-vote never touch the attention module)."""
    import cv2
    import kornia.augmentation as K

    from evaluate_abmil import predict_image
    from evaluate_patch import inference_grid_centers
    from patch_dataset import PAD as _PAD, load_split_record
    from preprocessing import make_preprocessor
    from train_abmil import load_frozen_backbone
    from abmil_model import FibrinABMIL

    baseline_path = "models/mpatch_v0_f/analysis/val/per_image_predictions.csv"
    if not os.path.exists(baseline_path):
        pytest.skip(f"{baseline_path} not found")
    baseline_df = pd.read_csv(baseline_path).set_index("idx")

    device = torch.device("cpu")
    backbone, cfg, model_dir, num_classes, class_map, head_type = \
        load_frozen_backbone("mpatch_v0_f", device)
    model = FibrinABMIL(backbone=backbone, num_classes=num_classes, head_type=head_type,
                         freeze_backbone=True, attn_hidden_dim=16)
    model.eval()

    _, val_df, _ = load_split_record(model_dir, DB_PATH)
    centers = inference_grid_centers()
    center_crop = K.CenterCrop(PATCH_SIZE)
    preprocessor = make_preprocessor()

    class_names = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]

    n_checked = 0
    for _, row in val_df.head(5).iterrows():
        img_idx = int(row["idx"])
        if img_idx not in baseline_df.index:
            continue
        img_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{img_idx:04d}.JPG"))
        tensor = preprocessor(img_bgr)
        padded = _pad_tensor(tensor, _PAD)

        _, soft_pred, _, soft_scores, _, _ = predict_image(model, padded, centers, device, center_crop)

        baseline_row = baseline_df.loc[img_idx]
        assert soft_pred == int(baseline_row["pred_label"]), (
            f"idx={img_idx}: soft-vote pred {soft_pred} != baseline pred "
            f"{int(baseline_row['pred_label'])}"
        )
        for i, cls in enumerate(class_names):
            expected = float(baseline_row[f"score_{cls}"])
            actual = float(soft_scores[i])
            assert actual == pytest.approx(expected, abs=1e-2), (
                f"idx={img_idx} class={cls}: score {actual} != baseline {expected}"
            )
        n_checked += 1

    assert n_checked > 0, "No baseline rows matched — check val split / baseline CSV alignment"
