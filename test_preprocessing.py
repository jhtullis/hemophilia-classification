"""
test_preprocessing.py — Visual tests for preprocessing pipeline.

Run with:
    pytest test_preprocessing.py -v -s

Output PNGs are saved to test_output/ for human inspection.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import torch
import pytest

from preprocessing import to_grayscale, min_pool, preprocess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHOTO_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
SAMPLE_IMG = os.path.join(PHOTO_DIR, "0000.JPG")


@pytest.fixture(autouse=True)
def make_output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_sample() -> np.ndarray:
    img = cv2.imread(SAMPLE_IMG)
    assert img is not None, f"Could not load sample image: {SAMPLE_IMG}"
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_grayscale_methods():
    """Compare all 5 grayscale conversion methods on a single sample image.

    Saves a 2×3 grid to test_output/grayscale_comparison.png.
    Visually inspect to choose the best conversion for fiber contrast.
    """
    img = load_sample()
    methods = ["lab_l", "luminance", "hsv_v", "hsv_s", "green"]
    titles = [
        "LAB — L channel (default)",
        "Luminance (BT.601)",
        "HSV — Value",
        "HSV — Saturation",
        "Green channel",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Show a small crop for legibility (top-left quarter)
    crop = (slice(0, 2000), slice(0, 3000))

    for i, (method, title) in enumerate(zip(methods, titles)):
        gray = to_grayscale(img, method=method)
        axes[i].imshow(gray[crop], cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(title, fontsize=11)
        axes[i].axis("off")

    axes[-1].axis("off")  # hide unused 6th panel
    fig.suptitle("Grayscale conversion method comparison (top-left crop)", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "grayscale_comparison.png")
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Structural checks
    for method in methods:
        gray = to_grayscale(img, method=method)
        assert gray.ndim == 2, f"{method}: expected 2-D output"
        assert gray.dtype == np.uint8, f"{method}: expected uint8"
        assert gray.shape == img.shape[:2], f"{method}: shape mismatch"


def test_min_pool_effect():
    """Show original grayscale vs. 10× min-pooled side by side.

    Saves to test_output/min_pool_effect.png.
    """
    img = load_sample()
    gray = to_grayscale(img, method="lab_l")
    pooled = min_pool(gray, factor=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax1.set_title(f"Original grayscale  {gray.shape[1]}×{gray.shape[0]}", fontsize=11)
    ax1.axis("off")

    ax2.imshow(pooled, cmap="gray", vmin=0, vmax=255)
    ax2.set_title(f"After 10× min-pool  {pooled.shape[1]}×{pooled.shape[0]}", fontsize=11)
    ax2.axis("off")

    fig.suptitle("Min-pool downsampling effect", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "min_pool_effect.png")
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Shape check
    h, w = gray.shape
    assert pooled.shape == (h // 10, w // 10)


def test_min_pool_vs_max_pool():
    """Compare min-pool vs max-pool — demonstrates that min-pool retains dark fibers.

    Saves to test_output/min_vs_max_pool.png.
    Dark fibers should be more visible in the min-pool result.
    """
    img = load_sample()
    gray = to_grayscale(img, method="lab_l")

    min_pooled = min_pool(gray, factor=10)

    # Max-pool for comparison (not used in the model)
    h, w = gray.shape
    max_pooled = gray.reshape(h // 10, 10, w // 10, 10).max(axis=(1, 3))

    # Crop to a region likely to contain fibers for easier visual comparison
    crop = (slice(0, 60), slice(0, 90))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray[: 600, : 900], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (600×900 crop)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(min_pooled[crop], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Min-pool ×10 (fiber-preserving)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(max_pooled[crop], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Max-pool ×10 (fiber-suppressing)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle("Min-pool vs Max-pool: dark fiber preservation", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "min_vs_max_pool.png")
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Min-pool values should be <= max-pool values everywhere
    assert (min_pooled <= max_pooled).all()


def test_full_preprocessing_pipeline():
    """Show the complete pipeline output as a displayable image.

    Saves to test_output/full_pipeline_output.png.
    """
    img = load_sample()
    tensor = preprocess(img, gray_method="lab_l", pool_factor=10)

    assert tensor.shape == (1, 400, 600), (
        f"Expected tensor shape (1, 400, 600), got {tuple(tensor.shape)}"
    )
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0

    # Display as image (squeeze channel dim, scale back to 0–255 for imshow)
    display = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(display, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"Full pipeline output: LAB-L → min-pool ×10 → normalized\n"
        f"Shape: {tuple(tensor.shape)}, min={tensor.min():.3f}, max={tensor.max():.3f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "full_pipeline_output.png")
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


def test_preprocess_tensor_properties():
    """Unit checks on preprocess output: shape, dtype, value range."""
    img = load_sample()
    tensor = preprocess(img, gray_method="lab_l", pool_factor=10)

    assert tensor.ndim == 3, "Expected 3-D tensor (C, H, W)"
    assert tensor.shape[0] == 1, "Expected single channel"
    assert tensor.shape[1] == 400, "Expected height 400 after 10× pool of 4000"
    assert tensor.shape[2] == 600, "Expected width 600 after 10× pool of 6000"
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
