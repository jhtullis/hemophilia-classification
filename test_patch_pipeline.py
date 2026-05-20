"""
test_patch_pipeline.py — Structural and visual tests for the patch_v0 pipeline.

Tests cover patch extraction, augmentation, model architecture, loss function,
inference grid, experiment split integrity, and cosine metric smoke test.

Run:
    pytest test_patch_pipeline.py -v -s
"""

import os

import numpy as np
import pytest
import torch

from patch_dataset import (
    OVERSIZED,
    PAD,
    PATCH_SIZE,
    PATCHES_PER_IMAGE,
    FibrinPatchDataset,
    _pad_tensor,
    split_by_experiment_3way,
)
from model_patch import FibrinPatchCNN, make_patch_model
from augmentation_patch import PatchAugmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = "data/endpoint10.db"
PHOTO_DIR = "data/photos"
DB_AVAILABLE = os.path.exists(DB_PATH) and os.path.exists(PHOTO_DIR)

def _make_fake_dataset(n_images: int = 10, patches_per_image: int = 4):
    """FibrinPatchDataset backed by synthetic random tensors (no disk access)."""
    import pandas as pd
    from data_loader import CLASS_NAMES

    rows = []
    for i in range(n_images):
        cls = CLASS_NAMES[i % len(CLASS_NAMES)]
        rows.append({"idx": i, "Exp_Type": cls, "Experiment": f"EXP_{cls}_{i}"})
    df = pd.DataFrame(rows)

    ds = FibrinPatchDataset.__new__(FibrinPatchDataset)
    ds.df = df
    ds.photo_dir = ""
    ds.preprocessor = None
    ds.patches_per_image = patches_per_image

    # Preload with properly padded random tensors (real content + zero border)
    ds._padded_tensors = [
        _pad_tensor(torch.rand(1, 400, 600), PAD)
        for _ in range(n_images)
    ]
    return ds


# ---------------------------------------------------------------------------
# Patch shape and content tests
# ---------------------------------------------------------------------------

def test_patch_shape():
    ds = _make_fake_dataset()
    patch, label = ds[0]
    assert patch.shape == (1, OVERSIZED, OVERSIZED), (
        f"Expected (1, {OVERSIZED}, {OVERSIZED}), got {patch.shape}"
    )


def test_patch_label_range():
    ds = _make_fake_dataset(n_images=20)
    for i in range(len(ds)):
        _, label = ds[i]
        assert 0 <= label < 5, f"Label {label} out of range [0, 4]"


def test_patch_no_out_of_range_pixels():
    ds = _make_fake_dataset()
    for i in range(min(20, len(ds))):
        patch, _ = ds[i]
        assert patch.min() >= 0.0, f"Pixel below 0.0 at index {i}"
        assert patch.max() <= 1.0, f"Pixel above 1.0 at index {i}"


def test_patch_coverage():
    """Centers drawn from full [0, H) × [0, W) range; edge patches contain black."""
    ds = _make_fake_dataset(n_images=1, patches_per_image=1000)
    # Collect 1000 patches; check that spatial range is utilised
    cys, cxs = [], []
    for i in range(1000):
        patch, _ = ds[i]
        # Can't recover cy/cx directly, but we can check edge patches have
        # partial black content from the pre-padding
        _ = patch   # just ensure it doesn't crash

    # Check manually constructed edge patches
    padded = ds._padded_tensors[0]
    half = OVERSIZED // 2

    for cy, cx in [(0, 0), (0, 599), (399, 0), (399, 599)]:
        cy_p, cx_p = cy + PAD, cx + PAD
        edge_patch = padded[
            :, cy_p - half : cy_p + half + 1, cx_p - half : cx_p + half + 1
        ]
        assert edge_patch.shape == (1, OVERSIZED, OVERSIZED)
        assert edge_patch.min().item() == 0.0, "Edge patch should have black border"
        assert edge_patch.max().item() > 0.0, "Edge patch should have real content"


def test_pad_tensor():
    t = torch.ones(1, 400, 600)
    padded = _pad_tensor(t, PAD)
    assert padded.shape == (1, 400 + 2 * PAD, 600 + 2 * PAD)
    # Corners should be zero (black padding)
    assert padded[0, 0, 0].item() == 0.0
    assert padded[0, -1, -1].item() == 0.0
    # Interior should still be ones
    assert padded[0, PAD, PAD].item() == 1.0


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

def test_augmentation_output_shape():
    aug = PatchAugmentation(patch_size=PATCH_SIZE)
    batch = torch.rand(4, 1, OVERSIZED, OVERSIZED)
    out = aug(batch)
    assert out.shape == (4, 1, PATCH_SIZE, PATCH_SIZE), (
        f"Expected (4, 1, {PATCH_SIZE}, {PATCH_SIZE}), got {out.shape}"
    )


def test_augmentation_rotation_invariance_visual():
    """Apply augmentation 8 times to the same patch; save grid to test_output/."""
    os.makedirs("test_output", exist_ok=True)
    aug = PatchAugmentation(patch_size=PATCH_SIZE)
    base = torch.rand(1, 1, OVERSIZED, OVERSIZED)

    outputs = [aug(base).squeeze() for _ in range(8)]

    # Assert outputs differ from each other (rotations produce different pixels)
    means = [o.mean().item() for o in outputs]
    assert len(set(round(m, 4) for m in means)) > 1, "All augmented patches are identical"

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax, img in zip(axes.flat, outputs):
            ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.suptitle("PatchAugmentation: 8 random rotations of the same patch")
        plt.tight_layout()
        plt.savefig("test_output/patch_augmentation_visual.png", dpi=80)
        plt.close()
        print("Saved test_output/patch_augmentation_visual.png")
    except Exception as e:
        print(f"Visual save skipped: {e}")


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_model_forward():
    model = FibrinPatchCNN(num_classes=5)
    model.eval()
    x = torch.rand(4, 1, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"


def test_model_output_range():
    """NormalizedLinear must produce cosine similarities in [-1, 1]."""
    model = FibrinPatchCNN(num_classes=5)
    model.eval()
    x = torch.rand(16, 1, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= -1.0 - 1e-5, f"Output below -1: {out.min().item()}"
    assert out.max().item() <= 1.0 + 1e-5, f"Output above +1: {out.max().item()}"


def test_model_parameter_count():
    model = FibrinPatchCNN(num_classes=5)
    n = sum(p.numel() for p in model.parameters())
    assert 800_000 <= n <= 1_100_000, f"Parameter count {n:,} outside expected [800K, 1.1M]"


def test_get_embeddings_shape():
    model = FibrinPatchCNN(num_classes=5)
    model.eval()
    x = torch.rand(4, 1, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        emb = model.get_embeddings(x)
    assert emb.shape == (4, 128), f"Expected (4, 128), got {emb.shape}"
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), (
        f"Embeddings not unit-norm: {norms}"
    )


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

def test_cosine_loss_range():
    # Import here to avoid circular dependency with train_patch before it exists
    try:
        from train_patch import cosine_loss
    except ImportError:
        pytest.skip("train_patch.py not yet created")

    device = torch.device("cpu")
    C = 5
    class_weights = torch.ones(C)

    # Random inputs: loss should be in [0, 2]
    sims = torch.rand(16, C) * 2 - 1   # in [-1, 1]
    labels = torch.randint(0, C, (16,))
    loss = cosine_loss(sims, labels, class_weights)
    assert 0.0 <= loss.item() <= 2.0 + 1e-5, f"Loss {loss.item()} out of [0, 2]"

    # Perfect alignment: similarity = 1.0 for correct class → loss = 0
    perfect_sims = torch.zeros(4, C)
    labels_p = torch.arange(4) % C
    for i, lbl in enumerate(labels_p):
        perfect_sims[i, lbl] = 1.0
    loss_perfect = cosine_loss(perfect_sims, labels_p, class_weights)
    assert abs(loss_perfect.item()) < 1e-5, f"Perfect loss should be 0, got {loss_perfect.item()}"

    # Worst case: similarity = -1.0 for correct class → loss = 2
    worst_sims = torch.zeros(4, C)
    for i, lbl in enumerate(labels_p):
        worst_sims[i, lbl] = -1.0
    loss_worst = cosine_loss(worst_sims, labels_p, class_weights)
    assert abs(loss_worst.item() - 2.0) < 1e-5, f"Worst loss should be 2, got {loss_worst.item()}"


# ---------------------------------------------------------------------------
# Inference grid test
# ---------------------------------------------------------------------------

def test_inference_grid():
    try:
        from evaluate_patch import inference_grid_centers
    except ImportError:
        pytest.skip("evaluate_patch.py not yet created")

    centers = inference_grid_centers(H=400, W=600, stride=100)
    assert len(centers) == 35, f"Expected 35 centers, got {len(centers)}"
    assert (0, 0) in centers, "Top-left corner (0,0) missing"
    assert (400, 600) in centers, "Bottom-right corner (400,600) missing"


# ---------------------------------------------------------------------------
# Experiment split integrity test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="Database not available")
def test_no_experiment_leakage():
    from data_loader import get_engine, load_metadata

    engine = get_engine(DB_PATH)
    df = load_metadata(engine)
    train_df, val_df, test_df = split_by_experiment_3way(df, seed=99)

    from data_loader import CLASS_MAP

    for cls in CLASS_MAP:
        train_exps = set(train_df[train_df["Exp_Type"] == cls]["Experiment"].unique())
        val_exps = set(val_df[val_df["Exp_Type"] == cls]["Experiment"].unique())
        test_exps = set(test_df[test_df["Exp_Type"] == cls]["Experiment"].unique())

        assert len(train_exps) == 7, f"{cls}: expected 7 train experiments, got {len(train_exps)}"
        assert len(val_exps) == 1, f"{cls}: expected 1 val experiment, got {len(val_exps)}"
        assert len(test_exps) == 2, f"{cls}: expected 2 test experiments, got {len(test_exps)}"

        assert train_exps.isdisjoint(val_exps), f"{cls}: train ∩ val not empty"
        assert train_exps.isdisjoint(test_exps), f"{cls}: train ∩ test not empty"
        assert val_exps.isdisjoint(test_exps), f"{cls}: val ∩ test not empty"


# ---------------------------------------------------------------------------
# Cosine metrics smoke test
# ---------------------------------------------------------------------------

def test_cosine_metrics_smoke():
    try:
        from train_patch import compute_cosine_metrics
    except ImportError:
        pytest.skip("train_patch.py not yet created")

    model = FibrinPatchCNN(num_classes=5)
    model.eval()

    # Build a tiny fake loader: 10 batches of 4 samples each
    ds = _make_fake_dataset(n_images=10, patches_per_image=4)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    import kornia.augmentation as K
    center_crop = K.CenterCrop(PATCH_SIZE)

    # Wrap loader items through center_crop so patches are 200×200
    class CroppedLoader:
        def __init__(self, raw_loader):
            self._loader = raw_loader
        def __iter__(self):
            for patches, labels in self._loader:
                yield center_crop(patches), labels
        def __len__(self):
            return len(self._loader)

    device = torch.device("cpu")
    metrics = compute_cosine_metrics(model, CroppedLoader(loader), device, max_samples=40)

    required = {"mean_max_sim", "intra_class_cos", "inter_class_cos", "silhouette"}
    assert required == set(metrics.keys()), f"Missing keys: {required - set(metrics.keys())}"
    assert -1.0 <= metrics["silhouette"] <= 1.0, f"Silhouette {metrics['silhouette']} out of [-1,1]"
