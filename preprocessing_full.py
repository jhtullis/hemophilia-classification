"""
preprocessing_full.py — Full-resolution preprocessing (no min-pool).

Returns (1, 4000, 6000) float32 tensors directly from 6000×4000 JPEGs.
Identical pipeline to preprocessing.py except the 10× min-pool step is omitted.

Used by the mpatch_v1_full pipeline to feed full-resolution patches to
FibrinPatchCNNFull.
"""

import torch
import numpy as np
from functools import partial

from preprocessing import ensure_landscape, to_grayscale


def preprocess_full(img_bgr: np.ndarray, gray_method: str = "lab_l") -> torch.Tensor:
    """Grayscale + normalize, no pooling. Returns (1, 4000, 6000) float32 tensor."""
    img_bgr  = ensure_landscape(img_bgr)
    gray     = to_grayscale(img_bgr, method=gray_method)   # (H, W) uint8
    norm     = gray.astype(np.float32) / 255.0             # [0, 1] float32
    return torch.from_numpy(norm).unsqueeze(0)             # (1, H, W)


def make_preprocessor_full(gray_method: str = "lab_l"):
    """Return a full-resolution preprocessing callable.

    Returns:
        Callable: img_bgr (H×W×3 uint8 BGR) → torch.Tensor (1, H, W) float32
    """
    return partial(preprocess_full, gray_method=gray_method)
