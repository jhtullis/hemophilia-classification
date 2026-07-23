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
# Orientation normalisation
# ---------------------------------------------------------------------------

def ensure_landscape(img_bgr: np.ndarray) -> np.ndarray:
    """Rotate portrait-oriented images 90° clockwise to landscape.

    All images are expected to be 6000×4000 (width > height). If a JPEG was
    captured or saved in portrait orientation the raw array will have height >
    width; this rotates it back to the orientation the rest of the pipeline
    assumes.
    """
    if img_bgr.shape[0] > img_bgr.shape[1]:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    return img_bgr


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
# Area-interpolation resize (approximate mean-pool, for non-integer factors)
# ---------------------------------------------------------------------------

def resize_pool(img_gray: np.ndarray, factor: float = 1.25) -> np.ndarray:
    """Downsample a 2-D grayscale image via OpenCV area interpolation.

    Unlike min_pool, this is an approximate area/mean pool (via cv2.INTER_AREA)
    rather than an exact darkest-pixel block-min. Appropriate when factor does not
    evenly divide H/W, or when a smoother, less dark-fiber-biased downsample is
    desired. factor need not be an integer.

    6000×4000 with factor=1.25 → 4800×3200 exactly (no cropping needed).

    Args:
        img_gray: 2-D uint8 array of shape (H, W).
        factor:   Downsampling factor (default 1.25). new_h = round(H / factor),
                  new_w = round(W / factor).

    Returns:
        2-D uint8 array of shape (round(H / factor), round(W / factor)).
    """
    h, w = img_gray.shape
    new_h = round(h / factor)
    new_w = round(w / factor)
    return cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
    img_bgr = ensure_landscape(img_bgr)
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


# ---------------------------------------------------------------------------
# Full pipeline (area-interpolation variant)
# ---------------------------------------------------------------------------

def preprocess_resized(img_bgr: np.ndarray,
                       gray_method: str = "lab_l",
                       pool_factor: float = 1.25) -> torch.Tensor:
    """Full preprocessing pipeline: grayscale → area-resize → normalize → tensor.

    Args:
        img_bgr:     H×W×3 uint8 BGR array (as returned by cv2.imread).
        gray_method: Grayscale conversion method (see to_grayscale).
        pool_factor: Area-resize downsampling factor (default 1.25).

    Returns:
        torch.Tensor of shape (1, round(H/pool_factor), round(W/pool_factor)),
        dtype float32, values in [0, 1].
    """
    img_bgr = ensure_landscape(img_bgr)
    gray = to_grayscale(img_bgr, method=gray_method)          # (H, W) uint8
    pooled = resize_pool(gray, factor=pool_factor)             # (H/f, W/f) uint8
    normalized = pooled.astype(np.float32) / 255.0            # [0, 1] float32
    return torch.from_numpy(normalized).unsqueeze(0)           # (1, H/f, W/f)


def make_preprocessor_resized(gray_method: str = "lab_l",
                              pool_factor: float = 1.25):
    """Return an area-resize preprocessing callable with fixed parameters.

    Suitable for passing directly to FibrinDataset.

    Returns:
        Callable: img_bgr -> torch.Tensor
    """
    return partial(preprocess_resized, gray_method=gray_method, pool_factor=pool_factor)
