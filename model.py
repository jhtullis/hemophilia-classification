"""
model.py — FibrinCNN architecture for fibrin clot phenotype classification.

Architecture overview
---------------------
Input: grayscale image, shape (1, 600, 400), preprocessed with 10× min-pool
       from original 6000×4000 photographs.

The network uses four convolutional blocks (square kernels, BatchNorm, ReLU,
2×2 pooling) followed by global average pooling and a small classifier head.

Receptive field analysis (pixels in the 600×400 input space → original space)
-------------------------------------------------------------------------------
Layer           Kernel  Cum.Stride  RF (600×400)  RF (original 6000×4000)
-----------     ------  ----------  ------------  -----------------------
Conv1 (k=5)       5        1             5              50
Pool1 (2×2)       2        2             6              60
Conv2 (k=5)       5        2            14             140
Pool2 (2×2)       2        4            16             160
Conv3 (k=3)       3        4            24             240
Pool3 (2×2)       2        8            28             280
Conv4 (k=3)       3        8            44             440
Pool4 (2×2)       2       16            52             520  ✓ (req ≥ 300 px)

Feature map dimensions after each block (input 1×600×400):
  Block 1 →  32 × 300 × 200
  Block 2 →  64 × 150 × 100
  Block 3 → 128 ×  75 ×  50
  Block 4 → 256 ×  37 ×  25
  GlobalAvgPool → 256 × 1 × 1
  Classifier  → 5 logits

All kernels are square. All convolutions use padding=(k-1)//2 to preserve
spatial dimensions within each block (no aspect ratio changes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FibrinCNN(nn.Module):
    """CNN for fibrin clot phenotype classification.

    Args:
        num_classes: Number of output classes (default 5).
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()

        # --- Convolutional feature extractor ---
        # Each block: Conv(square kernel) → BN → ReLU → MaxPool(2×2)
        # padding = (k-1)//2 preserves spatial size through convolution

        self.features = nn.Sequential(
            # Block 1: 1 × 600 × 400  →  32 × 300 × 200
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 × 300 × 200  →  64 × 150 × 100
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 × 150 × 100  →  128 × 75 × 50
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 × 75 × 50  →  256 × 37 × 25
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pool collapses spatial dims: 256 × 37 × 25 → 256
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (N, 1, 600, 400).

        Returns:
            Logits of shape (N, num_classes).
        """
        x = self.features(x)         # (N, 256, 37, 25)
        x = self.global_pool(x)      # (N, 256, 1, 1)
        x = x.flatten(1)             # (N, 256)
        return self.classifier(x)    # (N, num_classes)

    def summary(self) -> str:
        """Return a human-readable parameter count summary."""
        lines: List[str] = ["FibrinCNN parameter summary", "-" * 40]
        total = 0
        for name, module in self.named_modules():
            if not list(module.children()):  # leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    lines.append(f"  {name:<40s}  {params:>10,d}")
                    total += params
        lines.append("-" * 40)
        lines.append(f"  {'Total':40s}  {total:>10,d}")
        return "\n".join(lines)


class NormalizedLinear(nn.Module):
    """Linear layer where both input features and weight rows are L2-normalized
    before the dot product. Output is cosine similarities in [−1, 1]."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return x_norm @ w_norm.T   # cosine similarities in [-1, 1]


class FibrinCNNCosine(FibrinCNN):
    """FibrinCNN with a normalized cosine classifier head.

    Forward pass returns raw cosine similarities in [−1, 1]; no softmax.
    Training uses cosine loss (train_5class_hpc.py).
    Evaluation uses evaluate_cosine.py.
    Argmax of similarities gives predicted class — identical to logit argmax.

    Args:
        num_classes: Number of output classes (default 5).
        dropout_p:   Dropout probability in classifier head (default 0.5).
    """

    def __init__(self, num_classes: int = 5, dropout_p: float = 0.5):
        super().__init__(num_classes=num_classes)
        # Rebuild the full classifier head with configurable dropout and
        # NormalizedLinear output. Default dropout_p=0.5 preserves v0 behaviour.
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            NormalizedLinear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
