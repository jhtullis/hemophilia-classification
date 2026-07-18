"""
model_patch_full.py — FibrinPatchCNNFull for the mpatch_v1_full pipeline.

Processes 2000×2000 patches extracted directly from 6000×4000 full-resolution
images (no min-pool). A 3-layer stride-2 stem rapidly brings the spatial
resolution down to 250×250 before handing off to the same 3-block + Conv4
structure as FibrinPatchCNN, keeping the parameter count essentially identical
(~818K vs ~810K).

Architecture:
    Input: (N, 1, 2000, 2000)

    Stem (rapid downsampling, 3 × stride-2 conv-BN-ReLU):
      Conv(1→8,  3×3, s=2, p=1) → BN → ReLU :  (N,  8, 1000, 1000)
      Conv(8→16, 3×3, s=2, p=1) → BN → ReLU :  (N, 16,  500,  500)
      Conv(16→16,3×3, s=2, p=1) → BN → ReLU :  (N, 16,  250,  250)

    Block 1: _make_block(16→32)  → (N, 32, 125, 125)
    Block 2: _make_block(32→64)  → (N, 64,  63,  63)
    Block 3: _make_block(64→128) → (N,128,  32,  32)

    Conv4:   Conv(128→256, 3×3, s=1) + BN + ReLU → (N, 256, 32, 32)

    AdaptiveAvgPool2d(1,1) → Flatten → (N, 256)
    Linear(256→128) → ReLU → Dropout → <head>

Patch constants (exported for use in dataset and training scripts):
    PATCH_SIZE_FULL = 2000   final CNN input after rotation + crop
    OVERSIZED_FULL  = 2829   ceil(2000 * sqrt(2)), ensures clean crop at any angle
    PAD_FULL        = 1415   ceil(2829 / 2), pre-pad border so all image centers valid
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NormalizedLinear
from model_patch import _conv_bn_relu, _make_block

PATCH_SIZE_FULL = 2000
OVERSIZED_FULL  = math.ceil(PATCH_SIZE_FULL * math.sqrt(2))   # 2829
PAD_FULL        = math.ceil(OVERSIZED_FULL / 2)               # 1415


class FibrinPatchCNNFull(nn.Module):
    """Full-resolution patch CNN with stride-2 stem + 3 main blocks.

    Args:
        num_classes: Number of output classes (default 5).
        dropout_p:   Dropout probability in classifier head (default 0.5).
        head_type:   "cosine" — NormalizedLinear classifier (cosine similarities).
                     "ce"     — standard Linear classifier (CE logits).
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout_p: float = 0.5,
        head_type: str = "ce",
    ) -> None:
        super().__init__()
        if head_type not in ("cosine", "ce"):
            raise ValueError(f"head_type must be 'cosine' or 'ce', got {head_type!r}")
        self.head_type = head_type

        self.stem = nn.Sequential(
            _conv_bn_relu(1,  8,  stride=2),    # 2000 → 1000
            _conv_bn_relu(8,  16, stride=2),    # 1000 →  500
            _conv_bn_relu(16, 16, stride=2),    #  500 →  250
        )
        self.features = nn.Sequential(
            _make_block(16, 32),                # 250 → 125
            _make_block(32, 64),                # 125 →  63
            _make_block(64, 128),               #  63 →  32
            _conv_bn_relu(128, 256, stride=1),  #  32 →  32
            nn.AdaptiveAvgPool2d((1, 1)),        #  32 →   1
        )
        final_layer = (
            NormalizedLinear(128, num_classes)
            if head_type == "cosine"
            else nn.Linear(128, num_classes)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            final_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-dim penultimate features (L2-normalized for cosine head)."""
        x = self.stem(x)
        x = self.features(x)
        x = x.flatten(1)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return F.normalize(x, dim=1) if self.head_type == "cosine" else x


def make_full_patch_model(
    num_classes: int = 5,
    dropout_p: float = 0.5,
    head_type: str = "ce",
) -> FibrinPatchCNNFull:
    model = FibrinPatchCNNFull(
        num_classes=num_classes, dropout_p=dropout_p, head_type=head_type
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FibrinPatchCNNFull ({head_type} head): {n_params:,} parameters, "
          f"{num_classes} classes")
    return model
