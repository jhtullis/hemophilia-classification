"""
figs_best_class.py — Copy the highest-confidence correctly classified validation
image for each class into figs/.

For each of the 5 phenotype classes the script:
  1. Runs inference on the 5-class validation split.
  2. Finds the correctly classified image with the highest softmax confidence.
  3. Copies that original JPEG to figs/best_<CLASS>.JPG.

Outputs (figs/):
  best_AC3.JPG   best_F08D.JPG   best_F09D.JPG   best_F11D.JPG   best_NC1.JPG

Usage:
    python figs_best_class.py
"""

import os
import shutil

import torch

from analysis_utils import (PHOTOS_DIR, get_val_split,
                             load_model_from_registry, run_inference_full)
from data_loader import CLASS_NAMES
from preprocessing import make_preprocessor

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_DIR       = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR   = os.path.join(_DIR, "figs")
MODEL_TYPE = "5class"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(FIGS_DIR, exist_ok=True)
    device = torch.device("cpu")

    model, _model_dir, _num_classes, class_map, class_names = \
        load_model_from_registry(MODEL_TYPE, device)

    val_df = get_val_split(MODEL_TYPE)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)

    df = run_inference_full(model, val_df, PHOTOS_DIR, device,
                            preprocessor, class_map, class_names)

    for cls in CLASS_NAMES:
        correct = df[(df["Exp_Type"] == cls) & (df["correct"])]
        if correct.empty:
            print(f"WARNING: no correct predictions for {cls}, skipping")
            continue
        best = correct.loc[correct["confidence"].idxmax()]
        src = os.path.join(PHOTOS_DIR, f"{int(best['idx']):04d}.JPG")
        dst = os.path.join(FIGS_DIR, f"best_{cls}.JPG")
        shutil.copy(src, dst)
        print(f"Saved → {dst}  (idx={int(best['idx']):04d}, conf={best['confidence']:.3f})")


if __name__ == "__main__":
    main()
