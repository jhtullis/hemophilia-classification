"""
preprocessing.py — Image preprocessing pipeline for fibrin clot CNN.

Steps applied to every image before it enters the model:
  1. Grayscale conversion (configurable method; default: L channel from CIE LAB)
  2. Non-overlapping block min-pool to reduce 6000×4000 → 600×400
     (min-pool preserves dark fibers on a light background)
  3. Normalize to [0, 1] float32 and return as a (1, H, W) torch tensor
"""

import cv2
import numpy as np
import torch
from functools import partial


# ---------------------------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------------------------

def to_grayscale(img_bgr: np.ndarray, method: str = "lab_l") -> np.ndarray:
    """Convert a BGR image to a single-channel uint8 grayscale array.

    Args:
        img_bgr: H×W×3 uint8 array in BGR order (as loaded by cv2.imread).
        method:  Conversion method. One of:
                   "lab_l"     — L channel from CIE LAB (default; perceptually uniform)
                   "luminance" — BT.601 weighted RGB: 0.299R + 0.587G + 0.114B
                   "hsv_v"     — Value channel from HSV (max of R,G,B)
                   "hsv_s"     — Saturation channel from HSV
                   "green"     — Green channel only

    Returns:
        2-D uint8 array of shape (H, W).
    """
    if method == "lab_l":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]  # L channel is 0–255 in OpenCV's uint8 LAB

    elif method == "luminance":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # BT.601 by default

    elif method == "hsv_v":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]  # Value

    elif method == "hsv_s":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]  # Saturation

    elif method == "green":
        return img_bgr[:, :, 1]  # BGR index 1 = green

    else:
        raise ValueError(f"Unknown grayscale method: '{method}'. "
                         "Choose from: lab_l, luminance, hsv_v, hsv_s, green")


# ---------------------------------------------------------------------------
# Min-pooling
# ---------------------------------------------------------------------------

def min_pool(img_gray: np.ndarray, factor: int = 10) -> np.ndarray:
    """Non-overlapping block min-pool on a 2-D grayscale image.

    Selects the minimum (darkest) pixel in each (factor × factor) block.
    This preserves dark fiber structure when downsampling images that have
    dark fibers on a light background.

    6000×4000 with factor=10 → 600×400 exactly (no cropping needed).

    Args:
        img_gray: 2-D uint8 array of shape (H, W). H and W must be divisible
                  by factor.
        factor:   Pooling block size (default 10).

    Returns:
        2-D uint8 array of shape (H // factor, W // factor).
    """
    h, w = img_gray.shape
    if h % factor != 0 or w % factor != 0:
        # Crop to the nearest multiple of factor
        h = (h // factor) * factor
        w = (w // factor) * factor
        img_gray = img_gray[:h, :w]

    # Reshape into blocks and take the min over each block
    return img_gray.reshape(h // factor, factor,
                            w // factor, factor).min(axis=(1, 3))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray,
               gray_method: str = "lab_l",
               pool_factor: int = 10) -> torch.Tensor:
    """Full preprocessing pipeline: grayscale → min-pool → normalize → tensor.

    Args:
        img_bgr:     H×W×3 uint8 BGR array (as returned by cv2.imread).
        gray_method: Grayscale conversion method (see to_grayscale).
        pool_factor: Min-pool downsampling factor (default 10).

    Returns:
        torch.Tensor of shape (1, H // pool_factor, W // pool_factor),
        dtype float32, values in [0, 1].
    """
    gray = to_grayscale(img_bgr, method=gray_method)          # (H, W) uint8
    pooled = min_pool(gray, factor=pool_factor)                # (H/f, W/f) uint8
    normalized = pooled.astype(np.float32) / 255.0            # [0, 1] float32
    return torch.from_numpy(normalized).unsqueeze(0)           # (1, H/f, W/f)


def make_preprocessor(gray_method: str = "lab_l",
                      pool_factor: int = 10):
    """Return a preprocessing callable with fixed parameters.

    Suitable for passing directly to FibrinDataset.

    Returns:
        Callable: img_bgr -> torch.Tensor
    """
    return partial(preprocess, gray_method=gray_method, pool_factor=pool_factor)
