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
      5. ColorJitter:    brightness + contrast (only when brightness > 0 or contrast > 0)

    Args:
        patch_size:  Output spatial size after CenterCrop (default 200).
        p_flip:      Probability of each flip (default 0.5).
        brightness:  ColorJitter brightness factor — samples multiplier from
                     [max(0, 1-b), 1+b]. 0.0 disables brightness jitter.
        contrast:    ColorJitter contrast factor — samples multiplier from
                     [max(0, 1-c), 1+c]. 0.0 disables contrast jitter.
        noise_std:   Standard deviation of additive Gaussian noise (in [0,1] pixel
                     space). Applied after ColorJitter. 0.0 disables noise.

    Usage in training_step (after batch is on GPU):
        patches = self.augmentation(patches)

    Do NOT apply during validation — use kornia.geometry.transform.center_crop
    or K.CenterCrop directly for a deterministic crop.
    """

    def __init__(
        self,
        patch_size: int = 200,
        p_flip: float = 0.5,
        brightness: float = 0.0,
        contrast: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        transforms = [
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
        ]
        if brightness > 0.0 or contrast > 0.0:
            transforms.append(
                K.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=0.0,
                    hue=0.0,
                    p=1.0,
                )
            )
        if noise_std > 0.0:
            transforms.append(
                K.RandomGaussianNoise(mean=0.0, std=noise_std, p=1.0)
            )
        self.augment = K.AugmentationSequential(
            *transforms,
            data_keys=["input"],
            same_on_batch=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, OVERSIZED, OVERSIZED) on GPU → returns (N, 1, patch_size, patch_size)."""
        return self.augment(x)
