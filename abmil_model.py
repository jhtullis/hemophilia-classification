"""
abmil_model.py — Gated attention-based multiple instance learning (ABMIL)
aggregator wrapping a frozen (or fine-tunable) FibrinPatchCNN backbone.

Reference: Ilse, Tomczak & Welling (2018), "Attention-based Deep Multiple
Instance Learning" (arXiv:1802.04712). Attention is computed independently
per patch (no inter-patch interaction) — appropriate at this project's bag
count (~800 images); TransMIL/DSMIL-style inter-patch attention is out of
scope.

Does not modify model_patch.py. FibrinPatchCNN.get_embeddings() already
exposes the 128-dim penultimate feature for both head types (L2-normalized
for "cosine", raw for "ce") — used directly, no wrapper module needed.
"""

import torch
import torch.nn as nn

from model_patch import FibrinPatchCNN


class GatedAttentionMIL(nn.Module):
    """Ilse et al. (2018) gated attention aggregator.

    Permutation-invariant weighted average over N instance (patch) embeddings.
    No padding/masking needed — operates on a single bag of shape (N, feature_dim)
    at a time (see train_abmil.py's batch-size-1-bag loop).
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attention_V = nn.Linear(feature_dim, hidden_dim)
        self.attention_U = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patch_features: (N, feature_dim)
        A_V = torch.tanh(self.attention_V(patch_features))
        A_U = torch.sigmoid(self.attention_U(patch_features))
        A = self.attention_w(A_V * A_U)              # (N, 1)
        A = torch.softmax(A, dim=0)                   # (N, 1), sums to 1 over N
        image_embedding = (A * patch_features).sum(dim=0)   # (feature_dim,)
        return image_embedding, A.squeeze(-1)          # (feature_dim,), (N,)


class FibrinABMIL(nn.Module):
    """Image-level classifier built on top of a (by default frozen) patch backbone.

    Does not modify or require modification of the wrapped FibrinPatchCNN.

    Args:
        backbone:        A FibrinPatchCNN instance (already loaded with trained
                          weights by the caller — see train_abmil.load_frozen_backbone).
        num_classes:      Number of output classes.
        head_type:        'cosine' -> NormalizedLinear final layer (cosine similarities
                           in [-1, 1]); must match backbone.head_type since
                           get_embeddings() normalization depends on it.
                           'ce'     -> nn.Linear final layer (raw logits).
        freeze_backbone:  If True (default), backbone parameters are excluded from
                           gradient updates and the backbone is kept in eval() mode
                           at all times regardless of the outer module's train/eval
                           state (see train() override below) — this is what actually
                           freezes BatchNorm running stats and Dropout, not just
                           requires_grad=False.
        attn_hidden_dim:  Hidden dim of the attention head (default 64, smaller than
                           the 128-dim input given the small bag count).
    """

    def __init__(
        self,
        backbone: FibrinPatchCNN,
        num_classes: int = 5,
        head_type: str = "ce",
        freeze_backbone: bool = True,
        attn_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if head_type not in ("cosine", "ce"):
            raise ValueError(f"head_type must be 'cosine' or 'ce', got {head_type!r}")
        self.head_type = head_type
        self.freeze_backbone = freeze_backbone

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.attention = GatedAttentionMIL(feature_dim=128, hidden_dim=attn_hidden_dim)

        if head_type == "cosine":
            from model import NormalizedLinear
            self.classifier = NormalizedLinear(128, num_classes)
        else:
            self.classifier = nn.Linear(128, num_classes)

    def train(self, mode: bool = True) -> "FibrinABMIL":
        """Override so the frozen backbone stays in eval() regardless of the
        outer module's mode.

        nn.Module.train() recurses into every submodule by default. Without
        this override, calling model.train() each epoch would silently
        re-enable BatchNorm running-stat updates and Dropout noise in a
        backbone that requires_grad=False was supposed to keep frozen,
        corrupting its statistics over the course of ABMIL training.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: (N, 1, 200, 200) — all patches sampled from ONE image/bag.

        Returns:
            (output, attn_weights) — output: (num_classes,) cosine similarities
            or CE logits; attn_weights: (N,), sums to 1.
        """
        if self.freeze_backbone:
            with torch.no_grad():
                patch_features = self.backbone.get_embeddings(patches)   # (N, 128)
        else:
            patch_features = self.backbone.get_embeddings(patches)

        image_embedding, attn_weights = self.attention(patch_features)
        output = self.classifier(image_embedding.unsqueeze(0)).squeeze(0)   # (num_classes,)
        return output, attn_weights


def make_abmil_model(
    backbone: FibrinPatchCNN,
    num_classes: int = 5,
    head_type: str = "ce",
    freeze_backbone: bool = True,
    attn_hidden_dim: int = 64,
) -> FibrinABMIL:
    model = FibrinABMIL(
        backbone=backbone,
        num_classes=num_classes,
        head_type=head_type,
        freeze_backbone=freeze_backbone,
        attn_hidden_dim=attn_hidden_dim,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"FibrinABMIL ({head_type} head, freeze_backbone={freeze_backbone}): "
          f"{n_trainable:,} trainable / {n_total:,} total parameters, "
          f"{num_classes} classes")
    return model
