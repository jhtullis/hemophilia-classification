"""
model_patch_125s.py — FibrinPatchCNN125s for the mpatch_v1e_125s pipeline.

Processes 1600×1600 patches extracted from images downscaled 1.25× (area-interpolation
resize via cv2.INTER_AREA, NOT block-min-pool) from the original 6000×4000 JPEGs
(4800×3200). Same physical patch footprint (2000px in original-image space) as the
200×200 patches in the 600×400 min-pooled pipeline, the 1000×1000 patches in the
2x-pool pipeline, and the 2000×2000 patches in the full-resolution pipeline.

A wider, deeper stem+body than the other tiers is used deliberately for extra capacity
at this resolution (~3.3M params vs ~810-840K for the other patch tiers), to exploit the
richer detail preserved by area-resize (vs the darkest-pixel-only signal of min_pool).

Architecture:
    Input: (N, 1, 1600, 1600)

    Stem (3 × stride-2 conv-BN-ReLU, wide):
      Conv(1→32,  7×7, s=2, p=3) → BN → ReLU :  (N, 32, 800, 800)
      Conv(32→48, 3×3, s=2, p=1) → BN → ReLU :  (N, 48, 400, 400)
      Conv(48→48, 3×3, s=2, p=1) → BN → ReLU :  (N, 48, 200, 200)

    Block 1: _make_block(48→96)   → (N,  96, 100, 100)
    Block 2: _make_block(96→192)  → (N, 192,  50,  50)
    Block 3: _make_block(192→256) → (N, 256,  25,  25)

    Conv4:   Conv(256→256, 3×3, s=1) + BN + ReLU → (N, 256, 25, 25)

    AdaptiveAvgPool2d(1,1) → Flatten → (N, 256)
    Linear(256→128) → ReLU → Dropout → <head>

Patch constants (exported for use in dataset and training scripts):
    PATCH_SIZE_125S = 1600   final CNN input after rotation + crop (200 * 10/1.25, same
                              physical footprint as PATCH_SIZE=200 at 10x-pool)
    OVERSIZED_125S  = 2263   ceil(1600 * sqrt(2)), ensures clean crop at any angle
    PAD_125S        = 1132   ceil(2263 / 2), pre-pad border so all image centers valid
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NormalizedLinear
from model_patch import _conv_bn_relu, _make_block

PATCH_SIZE_125S = 1600
OVERSIZED_125S  = math.ceil(PATCH_SIZE_125S * math.sqrt(2))   # 2263
PAD_125S        = math.ceil(OVERSIZED_125S / 2)                # 1132


class FibrinPatchCNN125s(nn.Module):
    """1.25x-area-resize patch CNN with a wide 3-layer stride-2 stem + 3 wide main blocks.

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
            nn.ReLU(inplace=True),                 # 1600 → 800
            _conv_bn_relu(32, 48, stride=2),        #  800 → 400
            _conv_bn_relu(48, 48, stride=2),        #  400 → 200
        )
        self.features = nn.Sequential(
            _make_block(48, 96),                    # 200 → 100
            _make_block(96, 192),                   # 100 →  50
            _make_block(192, 256),                  #  50 →  25
            _conv_bn_relu(256, 256, stride=1),      #  25 →  25
            nn.AdaptiveAvgPool2d((1, 1)),            #  25 →   1
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


def make_125s_patch_model(
    num_classes: int = 5,
    dropout_p: float = 0.5,
    head_type: str = "ce",
) -> FibrinPatchCNN125s:
    model = FibrinPatchCNN125s(
        num_classes=num_classes, dropout_p=dropout_p, head_type=head_type
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FibrinPatchCNN125s ({head_type} head): {n_params:,} parameters, "
          f"{num_classes} classes")
    return model
