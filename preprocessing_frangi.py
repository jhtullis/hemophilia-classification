"""
preprocessing_frangi.py — Multi-channel Frangi preprocessing pipeline.

Produces a 7-channel tensor from each raw fibrin microscopy image:
  Channel 0 : grayscale (LAB L-channel by default), mean-pooled 4×
  Channels 1–6 : Frangi filter responses at σ = 1, 2, 4, 6, 10, 20

Pipeline for a 6000×4000 raw image:
  1. Mean-pool the color (BGR) image by factor 4  → (1000, 1500, 3) uint8
  2. Convert to grayscale (reuses to_grayscale)   → (1000, 1500) uint8 → float32 [0,1]
  3. Frangi filter at each σ in FRANGI_SIGMAS
     (sigmas=[s] → single-scale response, NOT the max across scales)
  4. Stack → torch.Tensor (7, 1000, 1500) float32 in [0,1]

Usage:
    from preprocessing_frangi import make_frangi_preprocessor, CHANNEL_LABELS
    preprocessor = make_frangi_preprocessor()
    tensor = preprocessor(img_bgr)   # (7, 1000, 1500) float32

Notes:
  - Memory: one tensor ≈ 42 MB. Reduce batch size compared to the 1-channel pipeline.
  - Frangi is slow (~1–3 s per sigma per image on CPU). Pre-cache if doing many passes.
  - skimage.filters.frangi with a single-element sigmas list evaluates at exactly one
    scale and returns values in [0,1]; no further normalization is needed.
"""

import numpy as np
import torch
from functools import partial

from skimage.filters import frangi

from preprocessing import to_grayscale


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRANGI_SIGMAS = [1, 2, 4, 6, 10, 20]

CHANNEL_LABELS = [
    "gray_lab_l",
    "frangi_s1",
    "frangi_s2",
    "frangi_s4",
    "frangi_s6",
    "frangi_s10",
    "frangi_s20",
]


# ---------------------------------------------------------------------------
# Mean-pool for color images
# ---------------------------------------------------------------------------

def mean_pool_color(img_bgr: np.ndarray, factor: int) -> np.ndarray:
    """Non-overlapping block mean-pool on a uint8 BGR image.

    Args:
        img_bgr: H×W×3 uint8 array (as loaded by cv2.imread).
        factor:  Pooling block size. H and W must be divisible by factor.

    Returns:
        (H // factor, W // factor, 3) uint8 array.
    """
    H, W = img_bgr.shape[:2]
    if H % factor != 0 or W % factor != 0:
        H = (H // factor) * factor
        W = (W // factor) * factor
        img_bgr = img_bgr[:H, :W]
    return (
        img_bgr.reshape(H // factor, factor, W // factor, factor, 3)
               .mean(axis=(1, 3))
               .astype(np.uint8)
    )


# ---------------------------------------------------------------------------
# Full Frangi preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_frangi(img_bgr: np.ndarray,
                      gray_method: str = "lab_l",
                      pool_factor: int = 4,
                      sigmas: list = None) -> torch.Tensor:
    """Multi-channel preprocessing: mean-pool → grayscale → Frangi stack.

    Args:
        img_bgr:     H×W×3 uint8 BGR array (from cv2.imread).
        gray_method: Grayscale method passed to to_grayscale (default 'lab_l').
        pool_factor: Mean-pool downsampling factor (default 4).
        sigmas:      List of Frangi sigma values (default FRANGI_SIGMAS).

    Returns:
        torch.Tensor of shape (7, H // pool_factor, W // pool_factor),
        dtype float32, values in [0, 1].
        Channel order: [grayscale, frangi_σ1, frangi_σ2, ..., frangi_σ20]
    """
    if sigmas is None:
        sigmas = FRANGI_SIGMAS

    # Step 1: mean-pool the color image
    pooled_bgr = mean_pool_color(img_bgr, pool_factor)           # uint8 BGR

    # Step 2: grayscale → float32 [0, 1]
    gray_uint8 = to_grayscale(pooled_bgr, method=gray_method)   # uint8
    gray = gray_uint8.astype(np.float32) / 255.0                 # float32 [0,1]

    # Step 3: Frangi filter at each sigma (single-scale per call)
    channels = [gray]
    for s in sigmas:
        f = frangi(gray, sigmas=[s]).astype(np.float32)          # float32 [0,1]
        channels.append(f)

    # Step 4: stack → (C, H, W) tensor
    return torch.from_numpy(np.stack(channels, axis=0))


def make_frangi_preprocessor(gray_method: str = "lab_l",
                              pool_factor: int = 4,
                              sigmas: list = None):
    """Return a Frangi preprocessing callable with fixed parameters.

    The returned callable accepts a uint8 BGR image and returns a
    (7, H//pool_factor, W//pool_factor) float32 tensor.

    Suitable for passing directly to FibrinDataset as the preprocessor.
    """
    return partial(preprocess_frangi,
                   gray_method=gray_method,
                   pool_factor=pool_factor,
                   sigmas=sigmas if sigmas is not None else FRANGI_SIGMAS)
