"""
augmentation.py — Data augmentation for fibrin clot images.

Exploits the 4-fold symmetry of rectangular microscopy images:
horizontal flip, vertical flip, and both flips (180° rotation).
Augmentation is applied during training only.
"""

from typing import List
import torch


def augment(tensor: torch.Tensor, variant: int) -> torch.Tensor:
    """Apply one of four augmentation variants to a CHW tensor.

    Args:
        tensor:  Torch tensor of shape (C, H, W).
        variant: Integer in {0, 1, 2, 3}:
                   0 — identity (no change)
                   1 — horizontal flip (left-right mirror)
                   2 — vertical flip (top-bottom mirror)
                   3 — both flips (equivalent to 180° rotation)

    Returns:
        Augmented tensor of the same shape as input.
    """
    if variant == 0:
        return tensor
    elif variant == 1:
        return torch.flip(tensor, dims=[2])   # flip width axis
    elif variant == 2:
        return torch.flip(tensor, dims=[1])   # flip height axis
    elif variant == 3:
        return torch.flip(tensor, dims=[1, 2])  # flip both axes
    else:
        raise ValueError(f"variant must be 0–3, got {variant}")


def get_all_augmentations(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Return all 4 augmentation variants of a tensor.

    Useful for visual inspection in tests.

    Args:
        tensor: Torch tensor of shape (C, H, W).

    Returns:
        List of 4 tensors: [original, h-flip, v-flip, both-flip].
    """
    return [augment(tensor, v) for v in range(4)]
