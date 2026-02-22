"""
test_augmentation.py — Visual and structural tests for augmentation.

Run with:
    pytest test_augmentation.py -v -s

Output PNGs are saved to test_output/ for human inspection.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pytest

from preprocessing import preprocess
from augmentation import augment, get_all_augmentations

PHOTO_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
SAMPLE_IMG = os.path.join(PHOTO_DIR, "0000.JPG")


@pytest.fixture(autouse=True)
def make_output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_tensor() -> torch.Tensor:
    img = cv2.imread(SAMPLE_IMG)
    assert img is not None, f"Could not load: {SAMPLE_IMG}"
    return preprocess(img, gray_method="lab_l", pool_factor=10)


def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) float32 tensor to uint8 2-D array for imshow."""
    return (t.squeeze(0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_four_augmentations():
    """Display all 4 augmentation variants in a 2×2 grid.

    Saves to test_output/augmentation_comparison.png.
    Visual inspection should confirm correct flip directions.
    """
    tensor = load_tensor()
    variants = get_all_augmentations(tensor)
    labels = ["Original", "Horizontal flip", "Vertical flip", "Both flips (180°)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, t, label in zip(axes.flatten(), variants, labels):
        ax.imshow(tensor_to_display(t), cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    fig.suptitle("4-fold augmentation variants", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "augmentation_comparison.png")
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # All variants must have same shape as input
    for v, t in enumerate(variants):
        assert t.shape == tensor.shape, f"Variant {v}: shape mismatch"


def test_augmentation_shape_preserved():
    """All variants must have identical shape to the input tensor."""
    tensor = load_tensor()
    for v in range(4):
        out = augment(tensor, v)
        assert out.shape == tensor.shape, f"variant {v}: shape changed"


def test_augmentation_reversibility():
    """Applying the same non-trivial flip twice should recover the original."""
    tensor = load_tensor()

    # Double horizontal flip → identity
    assert torch.equal(augment(augment(tensor, 1), 1), tensor), \
        "Double h-flip should equal original"

    # Double vertical flip → identity
    assert torch.equal(augment(augment(tensor, 2), 2), tensor), \
        "Double v-flip should equal original"

    # Double both-flip → identity
    assert torch.equal(augment(augment(tensor, 3), 3), tensor), \
        "Double both-flip should equal original"


def test_augmentation_variant0_is_identity():
    """Variant 0 must return a tensor equal to the input."""
    tensor = load_tensor()
    assert torch.equal(augment(tensor, 0), tensor), \
        "Variant 0 (identity) changed the tensor"


def test_hflip_differs_from_original():
    """Horizontal flip must actually change the image (non-symmetric input)."""
    tensor = load_tensor()
    flipped = augment(tensor, 1)
    # A real fibrin image is unlikely to be perfectly symmetric
    assert not torch.equal(flipped, tensor), \
        "H-flip is identical to original — image may be symmetric or test image is wrong"


def test_dataset_augmentation_indexing():
    """Simulate FibrinDataset indexing: indices 0–3 map to the 4 variants
    of the same base image."""
    tensor = load_tensor()
    # In FibrinDataset, variant = global_idx % 4
    for global_idx in range(4):
        variant = global_idx % 4
        out = augment(tensor, variant)
        assert out.shape == tensor.shape
