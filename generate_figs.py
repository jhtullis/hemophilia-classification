"""
generate_figs.py — Generate pipeline illustration figures for figs/.

Produces four images of data/photos/0000.JPG (AC3, class 0) at each stage
of the preprocessing pipeline, plus a Grad-CAM for its correct class:

  figs/original.JPG          - 0000.JPG copied as-is
  figs/step1_grayscale.png   - Full-resolution LAB L-channel grayscale
  figs/step2_minpooled.png   - After 10× min-pool (400×600)
  figs/step3_gradcam.png     - Grad-CAM for AC3 overlaid on min-pooled image

Usage:
    python generate_figs.py
"""

import os
import shutil

import cv2
import torch

from data_loader import CLASS_MAP
from gradcam import GradCAM
from model import FibrinCNN
from preprocessing import ensure_landscape, min_pool, preprocess, to_grayscale

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_DIR       = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(_DIR, "data", "photos")
MODEL_PATH = os.path.join(_DIR, "models", "5class", "best_model.pth")
FIGS_DIR   = os.path.join(_DIR, "figs")

IMG_IDX    = 0                   # 0000.JPG
TRUE_CLASS = CLASS_MAP["AC3"]    # 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(FIGS_DIR, exist_ok=True)

    src_jpg = os.path.join(PHOTOS_DIR, f"{IMG_IDX:04d}.JPG")
    img_bgr = cv2.imread(src_jpg)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {src_jpg}")
    img_bgr = ensure_landscape(img_bgr)

    # 1. Original — copy as-is
    out = os.path.join(FIGS_DIR, "original.JPG")
    shutil.copy(src_jpg, out)
    print(f"Saved → {out}")

    # 2. Step 1: grayscale (full resolution, 4000×6000)
    gray = to_grayscale(img_bgr, method="lab_l")
    out = os.path.join(FIGS_DIR, "step1_grayscale.png")
    cv2.imwrite(out, gray)
    print(f"Saved → {out}  ({gray.shape[1]}×{gray.shape[0]})")

    # 3. Step 2: min-pool → 400×600
    pooled = min_pool(gray, factor=10)
    out = os.path.join(FIGS_DIR, "step2_minpooled.png")
    cv2.imwrite(out, pooled)
    print(f"Saved → {out}  ({pooled.shape[1]}×{pooled.shape[0]})")

    # 4. Step 3: Grad-CAM for true class (AC3)
    tensor = preprocess(img_bgr).unsqueeze(0)   # (1, 1, 400, 600)

    model = FibrinCNN(num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    gcam    = GradCAM(model)
    heatmap = gcam.compute(tensor, target_class=TRUE_CLASS)   # (75, 50)
    overlay = gcam.overlay(heatmap, tensor)                   # (400, 600, 3) BGR
    gcam.remove_hooks()

    out = os.path.join(FIGS_DIR, "step3_gradcam.png")
    cv2.imwrite(out, overlay)
    print(f"Saved → {out}  ({overlay.shape[1]}×{overlay.shape[0]})")


if __name__ == "__main__":
    main()
