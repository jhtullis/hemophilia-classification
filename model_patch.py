"""
model_patch.py — FibrinPatchCNN architecture for the patch_v0 pipeline.

Stride-2 all-convolutional CNN (~900K params) with a cosine classifier.
No MaxPool; spatial downsampling is done by stride-2 convolutions.
NormalizedLinear imported from model.py — not redefined here.

Architecture:
    Input: (N, 1, 200, 200)

    Block 1: Conv(1→32, k=3, p=1) → BN → ReLU
             Conv(32→32, k=3, p=1) → BN → ReLU
             Conv(32→32, k=3, p=1, s=2) → BN → ReLU   → (N, 32, 100, 100)

    Block 2: Conv(32→64, k=3, p=1) → BN → ReLU
             Conv(64→64, k=3, p=1) → BN → ReLU
             Conv(64→64, k=3, p=1, s=2) → BN → ReLU   → (N, 64, 50, 50)

    Block 3: Conv(64→128, k=3, p=1) → BN → ReLU
             Conv(128→128, k=3, p=1) → BN → ReLU
             Conv(128→128, k=3, p=1, s=2) → BN → ReLU → (N, 128, 25, 25)

    Conv4:   Conv(128→256, k=3, p=1) → BN → ReLU       → (N, 256, 25, 25)

    AdaptiveAvgPool2d(1, 1) → Flatten → (N, 256)
    Linear(256→128) → ReLU → Dropout(0.5) → NormalizedLinear(128→num_classes)
    Output: (N, num_classes) cosine similarities in [-1, 1]

Receptive field in 200-px patch space: 59 px = 590 px in original image space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NormalizedLinear


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        _conv_bn_relu(in_ch, out_ch, stride=1),
        _conv_bn_relu(out_ch, out_ch, stride=1),
        _conv_bn_relu(out_ch, out_ch, stride=2),
    )


class FibrinPatchCNN(nn.Module):
    """Patch-based CNN with stride-2 downsampling and cosine classifier."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _make_block(1, 32),                              # → (N, 32, 100, 100)
            _make_block(32, 64),                             # → (N, 64, 50, 50)
            _make_block(64, 128),                            # → (N, 128, 25, 25)
            _conv_bn_relu(128, 256, stride=1),               # → (N, 256, 25, 25)
            nn.AdaptiveAvgPool2d((1, 1)),                    # → (N, 256, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            NormalizedLinear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized 128-dim embeddings (penultimate layer)."""
        x = self.features(x)
        x = x.flatten(1)
        for layer in self.classifier[:-1]:   # Linear, ReLU, Dropout
            x = layer(x)
        return F.normalize(x, dim=1)


def make_patch_model(num_classes: int = 5) -> FibrinPatchCNN:
    model = FibrinPatchCNN(num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FibrinPatchCNN: {n_params:,} parameters, {num_classes} classes")
    return model
