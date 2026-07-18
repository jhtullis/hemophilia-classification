# Claude Code Plan — Attention-Based MIL (ABMIL) Image-Level Aggregator
**Repository:** `hemophilia-classification`
**Date:** 2026-06-30
**Purpose:** Add a new model class that aggregates an *existing, already-trained*
patch-level classifier (`FibrinPatchCNN`) into an image-level classifier using
gated attention-based multiple instance learning (ABMIL), as an alternative to the
current cosine soft-vote aggregation in `evaluate_patch.py`.

**This plan must not modify any existing file.** It only adds new files. The
existing patch classifiers in `models/patch_5class/` (and any future CE-head
sibling, see §1) are loaded read-only and frozen by default.

---

## 0. Background and Literature

- **Ilse, Tomczak & Welling (2018)** — *"Attention-based Deep Multiple Instance
  Learning"* (arXiv:1802.04712). Original ABMIL formulation: a gated attention
  mechanism computes a learned, permutation-invariant weighted average over
  instance (patch) embeddings, trained end-to-end with only bag-level (image-level)
  labels. No patch-level annotations required. This is the architecture used below.
- ABMIL aggregates *before* classification (weighted sum of patch feature vectors,
  then one classifier call), unlike the current `evaluate_patch.py` cosine
  soft-vote, which classifies every patch independently and then averages the
  resulting per-patch probabilities/similarities. This lets the model represent
  diagnostic signal that depends on the *combination* of patches in an image,
  not just any single patch in isolation — relevant given the prior Grad-CAM
  finding that large-scale spatial structure (not just local fiber texture)
  appears diagnostically important.
- Known limitation of vanilla ABMIL (worth noting in code comments, not
  addressing here): attention scores are computed independently per patch, so
  inter-patch relationships are not modeled (this is what TransMIL/DSMIL later
  addressed). Given the project's bag count (~800 images), vanilla gated ABMIL is
  the appropriate complexity level — do not implement TransMIL/DSMIL variants.
- The attention head itself is lightweight regardless of backbone: two small
  linear projections (`attention_V`, `attention_U`) plus a scoring layer
  (`attention_w`), on the order of 10⁴–10⁵ parameters. It is not the bottleneck;
  the frozen backbone's patch features are.

---

## 1. Two Backbone Variants to Support

The patch backbone may have either of two final-layer types, and the ABMIL module
must work with both without modification to the backbone files:

| Variant | Final patch-classifier layer | Penultimate feature | Source file |
|---|---|---|---|
| Cosine | `NormalizedLinear(128 → num_classes)`, outputs cosine similarities in [−1, 1] | `Linear(256→128) → ReLU → Dropout` output, 128-dim | `model_patch.py` (`FibrinPatchCNN`) |
| Cross-entropy | `Linear(128 → num_classes)`, outputs raw logits | same 128-dim penultimate feature | `model_patch_ce.py` (`FibrinPatchCNNCE`), if/when it exists |

Both variants share the same convolutional trunk shape (4 blocks →
`AdaptiveAvgPool2d` → `Linear(256→128)`). The ABMIL module attaches at the
128-dim penultimate feature in both cases and never touches the original
classifier head. If `model_patch_ce.py` does not yet exist in the repo, write
the ABMIL code so it works generically against any backbone exposing a
`forward_features(x) -> (N, 128)` method (see §3.1) and degrade gracefully —
do not hard-fail if only the cosine variant is present.

---

## 2. New Files to Create (nothing else touched)

```
abmil_dataset.py     Image-level bag Dataset — wraps FibrinDataset's split logic,
                      yields all patches for one image per __getitem__
abmil_model.py        FibrinABMIL — attention aggregator wrapping a frozen
                      (or fine-tunable) patch backbone
train_abmil.py         Training loop for FibrinABMIL, two CLI-selectable variants
evaluate_abmil.py      Inference + attention-map visualization, parallel to
                      evaluate_patch.py's report format
test_abmil_pipeline.py Structural tests (shape/variable-N correctness)
```

Output directories (new, do not collide with existing ones):
```
models/patch_5class_abmil_cosine/
models/patch_5class_abmil_ce/        (only populated if CE backbone present)
```

---

## 3. `abmil_model.py`

### 3.1 Backbone feature extraction contract

Add a `forward_features` method to `FibrinPatchCNN` usage **without editing
`model_patch.py`** by wrapping it, not subclassing into it destructively:

```python
import torch
import torch.nn as nn


class PatchFeatureExtractor(nn.Module):
    """
    Read-only wrapper around a frozen, pretrained FibrinPatchCNN (or CE
    sibling) that exposes the 128-dim penultimate embedding instead of the
    final classifier output. Does not modify or retrain the wrapped model
    unless `freeze=False` is passed.
    """

    def __init__(self, backbone: nn.Module, freeze: bool = True):
        super().__init__()
        self.backbone = backbone
        # Reuse every layer up to (not including) the final classifier layer.
        # FibrinPatchCNN.forward is: conv blocks -> pool -> flatten ->
        # fc(256->128) -> relu -> dropout -> final_layer(128->C)
        # Expose everything except final_layer.
        self.trunk = nn.Sequential(*list(backbone.children())[:-1])
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 200, 200) -> (N, 128)
        return self.trunk(x)
```

**Important — verify against the actual `model_patch.py` module ordering**
before relying on `list(backbone.children())[:-1]`: if `FibrinPatchCNN` is
defined with the conv blocks and classifier head as a single flat
`nn.Sequential`, this slicing works directly. If instead it's defined as
separate named attributes (e.g. `self.features`, `self.fc`, `self.final_layer`),
write `forward_features` explicitly as:

```python
def forward_features(self, x):
    x = self.features(x)        # adapt names to actual model_patch.py attrs
    x = self.fc(x)               # Linear(256->128) + ReLU + Dropout
    return x                     # (N, 128), pre-final-layer
```

Check `model_patch.py`'s actual attribute names first and use whichever
approach matches — do not guess blindly; print `model.named_children()` on the
loaded checkpoint to confirm before writing this wrapper.

### 3.2 Gated attention aggregator

```python
class GatedAttentionMIL(nn.Module):
    """
    Ilse et al. (2018) gated attention aggregator.
    Permutation-invariant: handles variable patch counts N per image with
    no padding required when used at batch-size-1 (see train_abmil.py §1).
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.attention_V = nn.Linear(feature_dim, hidden_dim)
        self.attention_U = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patch_features: (N, feature_dim)
        A_V = torch.tanh(self.attention_V(patch_features))
        A_U = torch.sigmoid(self.attention_U(patch_features))
        A = self.attention_w(A_V * A_U)            # (N, 1)
        A = torch.softmax(A, dim=0)                 # (N, 1), sums to 1
        image_embedding = (A * patch_features).sum(dim=0)   # (feature_dim,)
        return image_embedding, A.squeeze(-1)        # also return weights for viz
```

Use `hidden_dim=64` (smaller than the 128-dim input) given the small bag count
(~800 images) — keep the attention head's own parameter count low relative to
the backbone to limit overfitting risk on the aggregator itself.

### 3.3 Full model — `FibrinABMIL`

```python
class FibrinABMIL(nn.Module):
    """
    Image-level classifier built on top of a frozen (by default) patch
    backbone. Does not modify or require modification of the patch backbone.

    head_type: 'cosine' -> NormalizedLinear final layer, cosine_loss training
               'ce'      -> nn.Linear final layer, CrossEntropyLoss training
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 5,
                 head_type: str = "cosine", freeze_backbone: bool = True,
                 attn_hidden_dim: int = 64):
        super().__init__()
        assert head_type in ("cosine", "ce")
        self.head_type = head_type
        self.feature_extractor = PatchFeatureExtractor(backbone, freeze=freeze_backbone)
        self.attention = GatedAttentionMIL(feature_dim=128, hidden_dim=attn_hidden_dim)

        if head_type == "cosine":
            from model_patch import NormalizedLinear   # reuse existing class, do not redefine
            self.classifier = NormalizedLinear(128, num_classes)
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patches: (N, 1, 200, 200) — all patches from ONE image
        patch_features = self.feature_extractor(patches)        # (N, 128)
        image_embedding, attn_weights = self.attention(patch_features)
        output = self.classifier(image_embedding.unsqueeze(0))  # (1, num_classes)
        return output.squeeze(0), attn_weights                   # logits/sims, (N,)
```

Import `NormalizedLinear` from `model_patch.py` rather than redefining it, per
the "do not modify, do not duplicate" constraint.

---

## 4. `abmil_dataset.py`

Image-level bag dataset. This is intentionally a thin wrapper, not a new
sampling scheme — reuse the existing image-level train/val split logic
(`load_split_from_record` / `train_record.json`) from `data_loader.py` so the
ABMIL model is trained and evaluated on the *same* image-level split as the
existing classifiers, exactly as requested.

```python
class FibrinImageBagDataset(Dataset):
    """
    One __getitem__ call returns ALL patches sampled from one image, plus
    its single image-level label. N (patches per image) varies — see
    train_abmil.py for how this is handled at the DataLoader level.
    """

    def __init__(self, image_df, photo_dir, preprocessor,
                 patches_per_image: int = 50, patch_size: int = 200,
                 augment: bool = False, seed: int | None = None):
        ...
        # Sample `patches_per_image` patches per image, RE-SAMPLED each
        # __getitem__ call (i.e. each epoch) when augment=True, so the
        # aggregator sees different patch subsets across training epochs
        # rather than a single fixed grid — this acts as data augmentation
        # for the bag composition itself (see prior discussion: re-randomize
        # which patches enter the bag each epoch when subsampling).

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        # Returns (patches: (N,1,200,200) tensor, label: int)
        ...
```

Use the **same** `FibrinPatchAugmentation` (Kornia) pipeline already defined in
`augmentation_patch.py` for the patch-level training, applied identically here
when `augment=True` — do not write a second augmentation pipeline.

`patches_per_image=50` is a reasonable default starting point (between the 20
used for original patch-CNN training and a denser inference grid); expose it
as a CLI flag so it can be tuned.

---

## 5. `train_abmil.py`

### 5.1 Batching strategy: batch size 1, no padding

Given variable N per image and a bag count of ~800, use one image (one bag)
per forward/backward pass rather than padded multi-image batches — this avoids
implementing attention masking entirely and is sufficient at this data scale.
Gradient accumulation over several images before `optimizer.step()` is
optional but recommended (e.g. accumulate 8 images, i.e. an effective batch
size of 8) to stabilize gradient noise without writing a custom collate/mask
function:

```python
accumulation_steps = 8
optimizer.zero_grad()
for i, (patches, label) in enumerate(train_loader):  # batch_size=1 in DataLoader
    patches, label = patches.squeeze(0).to(device), label.to(device)
    output, attn = model(patches)
    loss = loss_fn(output.unsqueeze(0), label.unsqueeze(0)) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5.2 Loss functions — both variants

```python
# CE variant: standard
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Cosine variant: reuse the existing cosine_loss function from train_patch.py,
# do not redefine it — import it.
from train_patch import cosine_loss
```

For the cosine variant, `model(patches)` returns a single `(num_classes,)`
similarity vector (not a batch), so call:
```python
loss = cosine_loss(output.unsqueeze(0), label.unsqueeze(0), class_weights)
```

### 5.3 Loading the pretrained backbone (read-only)

```python
def load_frozen_backbone(checkpoint_path: str, head_type: str, num_classes: int):
    if head_type == "cosine":
        from model_patch import FibrinPatchCNN
        backbone = FibrinPatchCNN(num_classes=num_classes)
    else:
        from model_patch_ce import FibrinPatchCNNCE   # only if this file exists
        backbone = FibrinPatchCNNCE(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    backbone.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    backbone.eval()
    return backbone
```

Default behavior: `freeze_backbone=True` (attention head + classifier only are
trained). Expose `--finetune-backbone` as an optional flag that sets
`freeze_backbone=False` and uses a lower LR for backbone parameters (e.g.
`lr * 0.1`) via parameter groups, for later experimentation — but the default
run should NOT fine-tune the backbone, since the goal here is augmenting the
already-trained classifiers, not retraining them.

### 5.4 CLI arguments

```
--backbone-checkpoint  str    required (path to existing best_model.pth)
--head-type             str    choices=['cosine','ce'], required
--model-dir             str    default="models/patch_5class_abmil_{head_type}"
--db                    str    default="data/endpoint10.db"
--photo-dir             str    default="data/photos"
--num-epochs            int    default=300
--patches-per-image     int    default=50
--attn-hidden-dim        int    default=64
--lr                    float  default=1e-3
--weight-decay          float  default=1e-3
--accumulation-steps     int    default=8
--finetune-backbone      flag   store_true (default off)
--train-record-path      str    default="models/patch_5class/train_record.json"
                                (REUSE the existing image-level split exactly —
                                 do not generate a new split)
```

### 5.5 Logging / checkpointing

Reuse `checkpoint_manager.py` exactly as the existing training scripts do —
no new checkpointing logic. Track per-epoch image-level accuracy (argmax over
classifier output) on train and val, plus mean attention entropy
(`-sum(A * log(A))` averaged over images) as a diagnostic: very low entropy
early in training suggests the attention head has collapsed onto a single
patch, which is worth flagging in logs but not auto-correcting.

---

## 6. `evaluate_abmil.py`

Parallel structure to `evaluate_patch.py`:
- Run inference per image (full patch grid, not a random subsample, for
  reproducible evaluation — same convention as `evaluate_patch.py`'s fixed
  inference grid).
- Save `per_image_predictions.csv` with columns: image idx, true label, pred
  label, per-class output (similarity or logit depending on head_type).
- Additionally save `attention_maps/` — for each validation image, overlay the
  per-patch attention weight back onto the image at patch locations (this is
  the interpretability payoff of ABMIL over voting; reuse the patch coordinate
  bookkeeping already present in `evaluate_patch.py`'s grid construction).
- Same `--split val` / `--split test` convention; test split untouched until
  final reporting, same rule as the existing pipeline.

---

## 7. `test_abmil_pipeline.py`

1. **`test_variable_n_forward`** — construct two synthetic bags of different
   N (e.g. 10 and 200 patches), confirm `FibrinABMIL.forward` runs without
   shape errors on both and returns `(num_classes,)` and `(N,)` respectively.
2. **`test_attention_weights_sum_to_one`** — assert `attn.sum()` ≈ 1.0 for a
   synthetic bag.
3. **`test_permutation_invariance`** — shuffle the patch order of a synthetic
   bag, confirm the output embedding/logits are numerically identical
   (within float tolerance) — this directly verifies the architecture
   property discussed earlier (sum-pooling is order-independent).
4. **`test_backbone_frozen_by_default`** — confirm `requires_grad=False` on
   all backbone trunk parameters when `freeze_backbone=True`, and that a
   training step does not change backbone weights (compare state_dict before
   /after one optimizer step).
5. **`test_cosine_head_output_range`** — for `head_type='cosine'`, assert all
   outputs lie in [−1, 1].

---

## 8. Notes for Claude Code

- Do not edit `model_patch.py`, `train_patch.py`, `evaluate_patch.py`,
  `data_loader.py`, `augmentation_patch.py`, or any existing model checkpoint.
  Only import from them.
- Before writing `PatchFeatureExtractor`, actually load
  `models/patch_5class/best_model.pth` into a fresh `FibrinPatchCNN` instance
  and run `print(list(model.named_children()))` to confirm the real attribute
  names/ordering — do not assume the `nn.Sequential` slicing in §3.1 matches
  without checking; adapt to whatever the real module structure is.
- If `model_patch_ce.py` / a CE-head patch backbone does not exist yet in the
  repo, implement the `head_type='ce'` path in `abmil_model.py` and
  `train_abmil.py` regardless (so the code is ready when that backbone is
  added), but skip CE-specific tests in `test_abmil_pipeline.py` with a
  `pytest.mark.skipif` guarded on the checkpoint file's existence, rather than
  failing.
- `patches_per_image` for training and the fixed evaluation grid size for
  `evaluate_abmil.py` do not need to match — keep them as independent
  parameters, consistent with how `train_patch.py` / `evaluate_patch.py`
  already separate random training patches from the fixed inference grid.
