"""
augmentation_patch.py — GPU-side augmentation for the patch_v0 pipeline.

Runs inside the training loop after the batch is already on GPU.
Input:  oversized patches (N, 1, 283, 283)
Output: augmented patches (N, 1, 200, 200) ready for FibrinPatchCNN

NOT used during validation; validation uses a plain center-crop instead.
"""

import torch
import torch.nn as nn
import kornia.augmentation as K


class PatchAugmentation(nn.Module):
    """
    GPU-side augmentation for oversized (283×283) patch batches.

    Pipeline (applied in order, each sample independently):
      1. RandomRotation: uniform ±180° with black (zero) fill
      2. CenterCrop:     discard rotated corners → 200×200
      3. RandomHorizontalFlip
      4. RandomVerticalFlip

    Usage in training_step (after batch is on GPU):
        patches = self.augmentation(patches)

    Do NOT apply during validation — use kornia.geometry.transform.center_crop
    or K.CenterCrop directly for a deterministic 283→200 crop.
    """

    def __init__(self, patch_size: int = 200, p_flip: float = 0.5) -> None:
        super().__init__()
        self.augment = K.AugmentationSequential(
            # RandomAffine with rotation-only + zero padding (RandomRotation in
            # kornia 0.8 lacks padding_mode; RandomAffine has it)
            K.RandomAffine(
                degrees=180.0,
                padding_mode="ZEROS",
                p=1.0,
            ),
            K.CenterCrop(patch_size),
            K.RandomHorizontalFlip(p=p_flip),
            K.RandomVerticalFlip(p=p_flip),
            data_keys=["input"],
            same_on_batch=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 283, 283) on GPU → returns (N, 1, 200, 200) on GPU."""
        return self.augment(x)
