"""
model_patch_2x.py — FibrinPatchCNN2x for the mpatch_v1e_2xmp pipeline.

Processes 1000×1000 patches extracted from images 2×-min-pooled from the original
6000×4000 JPEGs (2000×3000). Same physical patch footprint (2000px in original-image
space) as the 200×200 patches in the 600×400 min-pooled pipeline and the 2000×2000
patches in the full-resolution pipeline.

A 2-layer stride-2 stem rapidly brings the spatial resolution down to 250×250 (the same
body-input scale as FibrinPatchCNNFull's 3-layer stem — 2×-pool input only needs 4×
further reduction vs full-res's 8×) before handing off to the same 3-block + Conv4
structure as FibrinPatchCNN. The stem is deliberately wider (32 channels) and uses a
larger first kernel (7×7) than the other tiers' stems, giving a modest, deliberate
parameter increase (~825-840K vs ~810K for the 200px model) appropriate to the extra
input resolution.

Architecture:
    Input: (N, 1, 1000, 1000)

    Stem (2 × stride-2 conv-BN-ReLU, wider + larger first kernel):
      Conv(1→32,  7×7, s=2, p=3) → BN → ReLU :  (N, 32, 500, 500)
      Conv(32→32, 3×3, s=2, p=1) → BN → ReLU :  (N, 32, 250, 250)

    Block 1: _make_block(32→32)  → (N, 32, 125, 125)
    Block 2: _make_block(32→64)  → (N, 64,  63,  63)
    Block 3: _make_block(64→128) → (N,128,  32,  32)

    Conv4:   Conv(128→256, 3×3, s=1) + BN + ReLU → (N, 256, 32, 32)

    AdaptiveAvgPool2d(1,1) → Flatten → (N, 256)
    Linear(256→128) → ReLU → Dropout → <head>

Patch constants (exported for use in dataset and training scripts):
    PATCH_SIZE_2X = 1000   final CNN input after rotation + crop (200 * 10/2, same
                            physical footprint as PATCH_SIZE=200 at 10x-pool)
    OVERSIZED_2X  = 1415   ceil(1000 * sqrt(2)), ensures clean crop at any angle
    PAD_2X        =  708   ceil(1415 / 2), pre-pad border so all image centers valid
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NormalizedLinear
from model_patch import _conv_bn_relu, _make_block

PATCH_SIZE_2X = 1000
OVERSIZED_2X  = math.ceil(PATCH_SIZE_2X * math.sqrt(2))   # 1415
PAD_2X        = math.ceil(OVERSIZED_2X / 2)                # 708


class FibrinPatchCNN2x(nn.Module):
    """2x-min-pool patch CNN with a wide stride-2 stem + 3 main blocks.

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
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),                 # 1000 → 500
            _conv_bn_relu(32, 32, stride=2),        #  500 → 250
        )
        self.features = nn.Sequential(
            _make_block(32, 32),                    # 250 → 125
            _make_block(32, 64),                    # 125 →  63
            _make_block(64, 128),                   #  63 →  32
            _conv_bn_relu(128, 256, stride=1),      #  32 →  32
            nn.AdaptiveAvgPool2d((1, 1)),            #  32 →   1
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


def make_2x_patch_model(
    num_classes: int = 5,
    dropout_p: float = 0.5,
    head_type: str = "ce",
) -> FibrinPatchCNN2x:
    model = FibrinPatchCNN2x(
        num_classes=num_classes, dropout_p=dropout_p, head_type=head_type
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FibrinPatchCNN2x ({head_type} head): {n_params:,} parameters, "
          f"{num_classes} classes")
    return model
