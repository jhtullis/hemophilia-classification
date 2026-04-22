"""
test_preprocessing_frangi.py — Tests for the Frangi multi-channel preprocessing pipeline.

Run with:
    pytest test_preprocessing_frangi.py -v -s

Visual outputs are saved to test_output/frangi/ for human inspection.
Key outputs to review:
    channel_grid.png            — 3 images × 7 channels overview grid
    img<idx>_ch<n>_<label>.png  — individual channel images at full subplot resolution
    pool_comparison.png         — min-pool ×10 vs mean-pool ×4 side-by-side
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
from preprocessing_frangi import (
    CHANNEL_LABELS,
    FRANGI_SIGMAS,
    mean_pool_color,
    preprocess_frangi,
    make_frangi_preprocessor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHOTO_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "test_output", "frangi")

SAMPLE_INDICES = [0, 200, 400]   # three representative images


@pytest.fixture(autouse=True)
def make_output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_image(idx: int) -> np.ndarray:
    path = os.path.join(PHOTO_DIR, f"{idx:04d}.JPG")
    img  = cv2.imread(path)
    assert img is not None, f"Could not load image: {path}"
    return img


# ---------------------------------------------------------------------------
# Structural unit tests
# ---------------------------------------------------------------------------

def test_tensor_properties():
    """Shape, dtype, value range, and non-zero channels for default parameters."""
    img    = load_image(0)
    tensor = preprocess_frangi(img)

    assert tensor.shape == (7, 1000, 1500), \
        f"Expected (7, 1000, 1500), got {tuple(tensor.shape)}"
    assert tensor.dtype == torch.float32, \
        f"Expected float32, got {tensor.dtype}"
    assert float(tensor.min()) >= 0.0, "Values must be >= 0"
    assert float(tensor.max()) <= 1.0, "Values must be <= 1"
    assert not torch.isnan(tensor).any(), "NaN found in tensor"
    assert not torch.isinf(tensor).any(), "Inf found in tensor"

    for ch in range(7):
        assert tensor[ch].sum() > 0, \
            f"Channel {ch} ({CHANNEL_LABELS[ch]}) is all zeros"


def test_make_frangi_preprocessor():
    """make_frangi_preprocessor factory returns a callable that matches preprocess_frangi."""
    img = load_image(0)
    preprocessor = make_frangi_preprocessor()
    t1 = preprocess_frangi(img)
    t2 = preprocessor(img)
    assert torch.allclose(t1, t2), "Factory callable output differs from direct call"


def test_mean_pool_color_shape():
    """mean_pool_color reduces spatial dimensions by the given factor."""
    img    = np.ones((4000, 6000, 3), dtype=np.uint8) * 128
    result = mean_pool_color(img, factor=4)
    assert result.shape == (1000, 1500, 3), \
        f"Expected (1000, 1500, 3), got {result.shape}"
    assert result.dtype == np.uint8


def test_mean_pool_color_values():
    """mean_pool_color correctly averages pixel values within each block."""
    # 8×8×3 image; two 4×4 blocks per axis with distinct values
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[0:4, 0:4] = [100, 150, 200]   # top-left block
    img[0:4, 4:8] = [50,  50,  50]    # top-right block
    img[4:8, 0:4] = [200, 200, 200]   # bottom-left block
    img[4:8, 4:8] = [0,   0,   0]     # bottom-right block

    result = mean_pool_color(img, factor=4)
    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[0, 0], [100, 150, 200])
    np.testing.assert_array_equal(result[0, 1], [50,  50,  50])
    np.testing.assert_array_equal(result[1, 0], [200, 200, 200])
    np.testing.assert_array_equal(result[1, 1], [0,   0,   0])


def test_channels_are_distinct():
    """All 7 channels should be spatially different from one another."""
    img    = load_image(0)
    tensor = preprocess_frangi(img).numpy()   # (7, H, W)

    for i in range(7):
        for j in range(i + 1, 7):
            assert not np.allclose(tensor[i], tensor[j]), \
                f"Channels {i} ({CHANNEL_LABELS[i]}) and {j} ({CHANNEL_LABELS[j]}) are identical"


def test_channel_count_matches_sigmas():
    """Number of channels = 1 (gray) + len(FRANGI_SIGMAS)."""
    img    = load_image(0)
    tensor = preprocess_frangi(img)
    assert tensor.shape[0] == 1 + len(FRANGI_SIGMAS)
    assert len(CHANNEL_LABELS) == 1 + len(FRANGI_SIGMAS)


# ---------------------------------------------------------------------------
# Visual / inspection tests — outputs saved to test_output/frangi/
# ---------------------------------------------------------------------------

def test_visual_channel_grid():
    """Save a 3-image × 7-channel overview grid for visual inspection.

    Rows: sample images (indices 0, 200, 400).
    Cols: channels (gray, frangi_σ1, ..., frangi_σ20).
    Saved to: test_output/frangi/channel_grid.png
    """
    n_images   = len(SAMPLE_INDICES)
    n_channels = len(CHANNEL_LABELS)
    cmaps      = ["gray"] + ["viridis"] * (n_channels - 1)

    fig, axes = plt.subplots(n_images, n_channels,
                             figsize=(3 * n_channels, 3 * n_images),
                             squeeze=False)

    for row_i, idx in enumerate(SAMPLE_INDICES):
        img    = load_image(idx)
        tensor = preprocess_frangi(img).numpy()   # (7, H, W)

        for col_i, (label, cmap) in enumerate(zip(CHANNEL_LABELS, cmaps)):
            ch  = tensor[col_i]
            ax  = axes[row_i, col_i]
            ax.imshow(ch, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(label, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"img {idx:04d}", fontsize=8)

    fig.suptitle("Frangi multi-channel preprocessing — 3 images × 7 channels",
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "channel_grid.png")
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


def test_save_individual_sigma_images():
    """Save each channel as a separate PNG for each sample image.

    Files: test_output/frangi/img<idx>_ch<n>_<label>.png
    Allows full-resolution inspection of each Frangi sigma response.
    """
    for idx in SAMPLE_INDICES:
        img    = load_image(idx)
        tensor = preprocess_frangi(img).numpy()   # (7, H, W)

        for ch_i, label in enumerate(CHANNEL_LABELS):
            ch_arr  = (tensor[ch_i] * 255).astype(np.uint8)
            fname   = f"img{idx:04d}_ch{ch_i}_{label}.png"
            out_path = os.path.join(OUT_DIR, fname)

            if ch_i == 0:
                # Grayscale channel — save as-is
                cv2.imwrite(out_path, ch_arr)
            else:
                # Frangi channels — apply a colormap for better contrast
                colored = cv2.applyColorMap(ch_arr, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(out_path, colored)

            print(f"  Saved: {out_path}")


def test_mean_pool_vs_existing_pipeline_visual():
    """Side-by-side comparison of mean-pool ×4 (new) vs min-pool ×10 (existing).

    Saved to: test_output/frangi/pool_comparison.png
    """
    img = load_image(0)

    # Existing pipeline: lab_l grayscale → min-pool ×10
    existing = preprocess(img, gray_method="lab_l", pool_factor=10)
    existing_arr = (existing.squeeze().numpy() * 255).astype(np.uint8)

    # New pipeline: channel 0 is mean-pool ×4 + lab_l grayscale
    new_ch0 = preprocess_frangi(img)[0].numpy()
    new_arr = (new_ch0 * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(existing_arr, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        f"Existing: lab_l → min-pool ×10\n"
        f"Shape: {existing_arr.shape[1]}×{existing_arr.shape[0]}",
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].imshow(new_arr, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"New: mean-pool ×4 → lab_l (ch 0)\n"
        f"Shape: {new_arr.shape[1]}×{new_arr.shape[0]}",
        fontsize=10,
    )
    axes[1].axis("off")

    fig.suptitle("Downsampling strategy comparison (image 0000.JPG)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "pool_comparison.png")
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
