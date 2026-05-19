# Claude Code Plan — HPC GPU Training, Checkpointing, and Cosine Loss Model
**Repository:** `hemophilia-classification`
**Date:** 2026-05-18
**Purpose:** Single authoritative instruction document for Claude Code. Covers all
changes needed to run on BYU HPC with GPU, add Slurm job-chaining checkpoints, and
introduce a new cosine-loss model variant (`5class_hpc_v0`) with comprehensive
per-epoch logging of both standard and cosine-native metrics.

---

## 0. Background and Literature

### 0.1 Why long training?

The primary motivation is investigating **delayed generalization ("grokking")** — the
phenomenon where a neural network transitions from overfitting to genuine generalization
long after training loss has plateaued. Key literature:

- **Power et al. (2022)** — *"Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets"* (arXiv:2201.02177). Original observation; generalization can
  lag training convergence by 100× or more in step count.

- **Humayun et al. (2024, ICML)** — *"Deep Networks Always Grok and Here is Why"*
  (PMLR 235:20722–20745). Grokking is widespread in CNNs and ResNets trained on
  CIFAR-10/100 and Imagenette. However, their Figure 10 shows BatchNorm removes the
  abrupt local-complexity phase transition that drives delayed *robustness*. Since
  FibrinCNN has BatchNorm, the abrupt grokking jump is unlikely. Gradual continued
  improvement is still possible and not ruled out. The correct thesis framing is
  **"testing for continued generalization under extended training"**, not "testing for
  grokking." The grokking literature motivates the experiment; the outcome is open.

- **Andriushchenko et al. (2025, ICLR)** — *"Grokking at the Edge of Numerical
  Stability."* Identifies Softmax Collapse (SC) — when one logit grows unbounded, the
  softmax saturates, gradients vanish, and learning stops. SC is the primary obstacle
  to grokking with cross-entropy + softmax, especially for long training. Cosine loss
  sidesteps SC entirely.

- **Nanda et al. (2023, ICLR)** — *"Progress Measures for Grokking via Mechanistic
  Interpretability."* Weight decay is critical: it gradually reduces weight norms from
  a memorizing solution toward a generalizing one.

- **Loshchilov & Hutter (2017)** — *"Decoupled Weight Decay Regularization"*
  (arXiv:1711.05101). AdamW outperforms Adam+L2 over 1,800-epoch training runs. This
  is the directly relevant regime for this project.

- **Loshchilov & Hutter (2016)** — *"SGDR: Stochastic Gradient Descent with Warm
  Restarts"* (arXiv:1608.03983). Basis for `CosineAnnealingWarmRestarts`.

- **Barz & Denzler (2020, WACV)** — *"Deep Learning on Small Datasets without
  Pre-Training using Cosine Loss."* Cosine loss outperforms cross-entropy on datasets
  of 10–250 images per class — exactly the regime here (~160 training images per
  class). The loss `L = 1 − ĥ·Ŵ_y` directly optimizes the angular geometry of
  representations on the unit sphere, with no softmax.

### 0.2 Two model streams

This plan produces two parallel model streams:

| Stream | Directory | Loss | Purpose |
|--------|-----------|------|---------|
| Baseline (existing + GPU/checkpoint) | `models/5class/` | CrossEntropyLoss | Unchanged architecture; gets GPU + checkpointing only |
| New cosine model | `models/5class_hpc_v0/` | Cosine loss | New architecture + all training improvements |

The 3-class models (`3class_hemo/`, `3class_hemo_finetune/`) receive the same GPU +
checkpointing treatment as the `5class` baseline, with no loss function changes.

---

## 1. Environment

### 1.1 `environment.yml` (modify existing)

```yaml
name: fibrin
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - ipykernel
  - sqlalchemy
  - scikit-image
  - matplotlib
  - pytest
  - py-opencv
  - scikit-learn
  - scipy
  - umap-learn
  - pytorch=2.7.1=cuda126_mkl*
  - torchvision
  - torchaudio
```

Add `scikit-learn`, `scipy`, and `umap-learn` — required by `evaluate_cosine.py`
(silhouette score, Platt scaling, UMAP projection). Do **not** add the `defaults`
channel; BYU HPC Miniforge3 is licensed for `conda-forge` only.

### 1.2 BYU HPC context

- GPU driver: CUDA 12.8. PyTorch `cuda126` builds are backward-compatible.
- Scheduler: Slurm. Login nodes have internet; compute nodes may not.
- Conda/mamba via `module load miniforge3`. Use `#!/bin/bash --login` on Slurm
  shebangs so `conda activate` works.
- Install environment once on the login node; do not reinstall inside job scripts.

---

## 2. New Files

### 2.1 `checkpoint_manager.py`

Centralises all checkpoint I/O. Used by all training scripts.

**`save_checkpoint(state, model_dir, epoch, is_best=False)`**

- Always saves to `models/{type}/checkpoints/latest.pth` (overwrites; used for
  resume). Creates `checkpoints/` if absent.
- Saves periodic snapshot per `should_save_periodic(epoch)`:
  - Epochs 1–300: every 20 epochs (`epoch % 20 == 0`)
  - Epochs 301–10000: every 100 epochs (`epoch % 100 == 0`)
  - Always save at the final target epoch
- If `is_best=True`, also saves `models/{type}/best_model.pth`.
- The state dict saved must contain exactly:

```python
{
    "epoch":                int,
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_acc":             float,
    "history":              dict,   # full per-epoch history lists accumulated so far
    "model_type":           str,
    "phase":                int,    # 1 or 2 (finetune only; always 1 otherwise)
    "config":               dict,   # hyperparameter snapshot (see §4.3)
}
```

**`load_latest_checkpoint(model_dir)`**

- Loads `models/{type}/checkpoints/latest.pth`. Returns the dict or `None`.
- Prints "Resuming from epoch N" or "Starting from scratch."

**`log_epoch(model_dir, row_dict)`**

- Appends one row to `models/{type}/training_log_full.csv`.
- Creates the file with a header row on first call.
- Uses `csv.DictWriter` in append mode (`newline=''`).
- The row dict passed in differs by model stream. See §4.4 and §5.4 for the exact
  column sets for each stream.

**`should_save_periodic(epoch)`** — standalone helper, encode checkpoint cadence:
```python
def should_save_periodic(epoch: int, max_epochs: int) -> bool:
    if epoch <= 300:
        return epoch % 20 == 0
    else:
        return epoch % 100 == 0
```

**Storage note (add as comment in file):** Each `.pth` for FibrinCNN (~455K params)
is ~2–5 MB. Total accumulation across 10,000 epochs: ~30 periodic files ≈ 150 MB.

---

### 2.2 `train_5class_hpc.py`

New training script for the `5class_hpc_v0` model. Do not modify `train_5class.py`.

Full diff from `train_5class.py`:

| Component | `train_5class.py` | `train_5class_hpc.py` |
|-----------|-------------------|-----------------------|
| `MODEL_DIR` | `models/5class` | `models/5class_hpc_v0` |
| Model class | `FibrinCNN(5)` | `FibrinCNNCosine(5)` |
| Loss | `CrossEntropyLoss(weight=w)` | `cosine_loss(sims, labels, w)` |
| Optimizer | `Adam(lr=1e-3, wd=1e-4)` | `AdamW(lr=1e-3, wd=1e-3)` |
| Scheduler | `ReduceLROnPlateau(patience=5)` | `CosineAnnealingWarmRestarts(T_0=100, T_mult=2, eta_min=1e-6)` |
| Scheduler step | `scheduler.step(val_acc)` | `scheduler.step(epoch)` |
| Per-epoch log columns | Standard set (§4.4) | Extended set (§5.4) |

The `cosine_loss` function is defined at module level in this file:

```python
def cosine_loss(
    similarities: torch.Tensor,   # (B, C) raw cosine similarities from forward()
    labels:       torch.Tensor,   # (B,) class indices
    class_weights: torch.Tensor,  # (C,) inverse-frequency weights
) -> torch.Tensor:
    """
    Pure cosine loss: L = mean_over_batch[ w_y * (1 − s_y) ]
    where s_y is the cosine similarity to the correct class prototype Ŵ_y.
    Loss range: [0, 2]. No softmax. No Softmax Collapse risk.
    Gradient acts only on the correct-class similarity; implicit competition
    arises from weight normalization (rotating Ŵ_y toward ĥ rotates it away
    from other class prototypes on the unit sphere).
    """
    correct_sims = similarities[torch.arange(len(labels)), labels]  # (B,)
    weights      = class_weights[labels]                             # (B,)
    return (weights * (1.0 - correct_sims)).mean()
```

**Per-epoch validation metrics for `5class_hpc_v0`** — computed during the val loop
each epoch, logged to CSV, and saved into the checkpoint's `history` dict. See §5.4
for the full column specification and implementation details.

---

### 2.3 `evaluate_cosine.py`

New evaluation script for cosine-head models. Replaces `evaluate.py` for the
`5class_hpc_v0` model type. `evaluate.py` is unchanged for all other models.

Full specification in §6.

---

### 2.4 `slurm/train_5class.sh`

```bash
#!/bin/bash --login
#SBATCH --job-name=fibrin_5class
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_5class_%j.out
#SBATCH --error=slurm/logs/fibrin_5class_%j.err
#SBATCH --mail-type=FAIL

module load miniforge3
conda activate fibrin

python train.py --model-type 5class \
    --max-epochs 10000 --epochs-per-job 200 --resume

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch slurm/train_5class.sh
fi
```

Exit code convention (enforced in `train.py`):
- `sys.exit(0)` — job quota exhausted, more epochs remain; resubmit
- `sys.exit(100)` — `--max-epochs` reached; do not resubmit

### 2.5 `slurm/train_5class_hpc_v0.sh`

Same structure as §2.4 with:
- `--job-name=fibrin_5class_hpc_v0`
- `--model-type 5class_hpc_v0`
- Log paths: `slurm/logs/fibrin_5class_hpc_v0_%j.{out,err}`
- Resubmit line: `sbatch slurm/train_5class_hpc_v0.sh`

### 2.6 `slurm/train_3class_scratch.sh`

Same structure as §2.4 with `--model-type 3class_scratch` and matching names.

### 2.7 `slurm/train_3class_finetune.sh`

Same structure with `--model-type 3class_finetune`. Note in comments: Phase 1
(5 epochs, head-only) completes in the first job; Phase 2 continues in subsequent
jobs; the checkpoint stores `phase` so resumption is transparent.

### 2.8 `slurm/submit_all.sh`

```bash
#!/bin/bash
sbatch slurm/train_5class.sh
sbatch slurm/train_5class_hpc_v0.sh
sbatch slurm/train_3class_scratch.sh
sbatch slurm/train_3class_finetune.sh
```

All four jobs are independent and can run concurrently if GPUs are available.

### 2.9 `slurm/setup_env.sh`

```bash
#!/bin/bash
# Run once interactively on the login node — do NOT submit via sbatch.
module load miniforge3
mamba env create -f environment.yml
```

### 2.10 `slurm/logs/.gitkeep`

Empty file so git tracks the logs directory.

---

## 3. Files to Modify

### 3.1 `model.py`

Add two new classes. **Do not modify `FibrinCNN`.**

```python
import torch.nn.functional as F

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
        return x_norm @ w_norm.T   # cosine similarities


class FibrinCNNCosine(FibrinCNN):
    """FibrinCNN with a normalized cosine classifier head.
    Forward pass returns raw cosine similarities in [−1, 1].
    No temperature parameter. No softmax during training or inference.
    Training: CosineLoss (train_5class_hpc.py).
    Evaluation: evaluate_cosine.py."""

    def __init__(self, num_classes: int = 5):
        super().__init__(num_classes=num_classes)
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = NormalizedLinear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs the standard FibrinCNN forward; classifier[-1] is now
        # NormalizedLinear so output is cosine similarities, not logits.
        return super().forward(x)
```

### 3.2 `train.py`

**New CLI arguments:**

```python
parser.add_argument("--resume",          action="store_true")
parser.add_argument("--max-epochs",      type=int, default=10000)
parser.add_argument("--epochs-per-job",  type=int, default=200)
```

**New model-type choice:**

```python
choices=["5class", "5class_hpc_v0", "3class_scratch", "3class_finetune"]
```

Dispatch `5class_hpc_v0` to `train_5class_hpc.main(...)`.

**Device detection helper** (add near top):

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU — using CPU")
    return device
```

Pass the returned device into all training functions.

### 3.3 `train_5class.py` (baseline — minimal changes)

Apply only the changes needed for GPU support, checkpointing, and resume. Do **not**
change the optimizer, scheduler, or loss function — this is the baseline.

- Import and use `checkpoint_manager`.
- Accept `--resume`, `--max-epochs`, `--epochs-per-job` forwarded from `train.py`.
- GPU: `pin_memory=True` in DataLoaders when `device == cuda`.
- Scheduler: add `min_lr=1e-7` to `ReduceLROnPlateau` to prevent total LR collapse.
- Training loop: call `checkpoint_manager.save_checkpoint(...)` and
  `checkpoint_manager.log_epoch(...)` each epoch (standard column set, §4.4).
- Append to `training_history.json` on resume (do not overwrite).
- Exit codes: `sys.exit(0)` when job quota done, `sys.exit(100)` when max-epochs
  reached.
- Update `NUM_EPOCHS` → driven by `--max-epochs` / `--epochs-per-job`.

### 3.4 `train_3class.py` (same minimal changes as §3.3)

Additionally: use a unified epoch loop that transitions between Phase 1 and Phase 2
based on epoch number, rather than two separate `_epoch_loop` calls. The checkpoint
stores `"phase"` (1 or 2) so job-boundary resumes pick up in the correct phase:

```python
for epoch in range(start_epoch, end_epoch + 1):
    if epoch <= EPOCHS_PHASE1:
        model.features.requires_grad_(False)
        use_optimizer, use_scheduler, current_phase = opt1, sched1, 1
    else:
        if current_phase == 1:           # first crossing of phase boundary
            model.requires_grad_(True)
        use_optimizer, use_scheduler, current_phase = opt2, sched2, 2
    # ... run one epoch ...
```

### 3.5 `analysis_utils.py`

Add `5class_hpc_v0` to `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {
    "5class":           ("models/5class",              5, CLASS_MAP_5),
    "5class_hpc_v0":    ("models/5class_hpc_v0",       5, CLASS_MAP_5),  # NEW
    "3class_scratch":   ("models/3class_hemo",          3, CLASS_MAP_3),
    "3class_finetune":  ("models/3class_hemo_finetune", 3, CLASS_MAP_3),
}
```

Update `load_model_from_registry` to instantiate `FibrinCNNCosine` for `hpc` types:

```python
def load_model_from_registry(model_type: str, device):
    model_dir, num_classes, class_map = MODEL_REGISTRY[model_type]
    if "hpc" in model_type:
        from model import FibrinCNNCosine
        model = FibrinCNNCosine(num_classes=num_classes)
    else:
        from model import FibrinCNN
        model = FibrinCNN(num_classes=num_classes)
    sd = torch.load(os.path.join(model_dir, "best_model.pth"),
                    map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    # ... rest of function unchanged ...
```

### 3.6 `run_analysis.py`

Add a branch dispatching `hpc` model types to `evaluate_cosine.py`:

```python
if "hpc" in model_type:
    _run("evaluate_cosine.py", mt, db_path=db_path)
else:
    _run("evaluate.py", mt, db_path=db_path)
    # ... existing analysis steps ...
```

GradCAM, `visualize_weights.py`, and `plot_training_curves.py` do not depend on
loss function and can run for `5class_hpc_v0` without modification.

### 3.7 `AGENT_EXPORT.md`

Add a section titled **"HPC Training and Checkpointing"** with:
- One-time setup: `bash slurm/setup_env.sh`
- Submitting: `sbatch slurm/train_5class_hpc_v0.sh`
- Job chaining: self-resubmit via exit code 0/100 convention
- Checkpoint locations: `models/{type}/checkpoints/`
- Cadence: every 20 epochs to 300, then every 100 to 10,000
- Extended log: `models/{type}/training_log_full.csv`
- Resume: `latest.pth` always overwritten; restore model + optimizer + scheduler state
- Grokking / BatchNorm note (Humayun et al. 2024)
- Cosine model notes: pure cosine loss, no temperature during training, evaluate via
  `evaluate_cosine.py`

---

## 4. Baseline Training Details (`5class`, `3class_scratch`, `3class_finetune`)

### 4.1 Optimizer and scheduler (unchanged from existing codebase)

- Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)` — unchanged
- Scheduler: `ReduceLROnPlateau(mode="max", patience=5, factor=0.5, min_lr=1e-7)`
  (add `min_lr=1e-7` to prevent LR collapse over thousands of epochs)
- Loss: `CrossEntropyLoss(weight=class_weights)` — unchanged

### 4.2 GPU DataLoader settings

When `device == cuda`, set `pin_memory=True` in both train and val DataLoaders.
Keep existing `num_workers` logic (`min(8, max(4, cpu_count() - 2))`).

### 4.3 Checkpoint config dict for baseline models

```python
"config": {
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_size":   BATCH_SIZE,
    "gray_method":  GRAY_METHOD,
    "pool_factor":  POOL_FACTOR,
    "optimizer":    "Adam",
    "scheduler":    "ReduceLROnPlateau",
    "loss":         "CrossEntropyLoss",
}
```

### 4.4 Per-epoch CSV columns for baseline models

`checkpoint_manager.log_epoch` is called once per epoch with this row dict:

```
epoch                  integer epoch number (1-based, global — not reset at job boundaries)
train_loss             mean cross-entropy loss over training set
train_acc              classification accuracy over training set
val_loss               mean cross-entropy loss over validation set
val_acc                classification accuracy over validation set
lr                     optimizer.param_groups[0]["lr"] (recorded after scheduler.step)
elapsed_seconds        wall time for this epoch in seconds
wall_clock_time        datetime.utcnow().isoformat() at epoch end
best_val_acc_so_far    running maximum val_acc across all epochs
model_type             e.g. "5class"
phase                  1 or 2 (finetune only; 1 for all others)
```

### 4.5 Scheduler state across job boundaries

`ReduceLROnPlateau` carries internal state (`num_bad_epochs`, `best`, `_last_lr`).
This **must** be saved and restored via `scheduler.state_dict()` /
`scheduler.load_state_dict()`. Failure to restore it resets the patience counter
at every job boundary, causing premature or incorrect LR decay. This is the most
common silent bug in naive checkpoint implementations.

---

## 5. Cosine Model Training Details (`5class_hpc_v0`)

### 5.1 Architecture

`FibrinCNNCosine` (defined in `model.py`, §3.1). All layers identical to `FibrinCNN`
except `classifier[-1]` is replaced with `NormalizedLinear(128, 5)`. Forward pass
returns raw cosine similarities in [−1, 1]. No temperature parameter anywhere in the
model.

### 5.2 Optimizer and scheduler

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=2, eta_min=1e-6
)
```

Rationale:
- **AdamW over Adam:** Decoupled weight decay gives consistent weight norm control.
  Loshchilov & Hutter showed better test error over 1,800-epoch runs — the relevant
  regime here.
- **weight_decay=1e-3:** Stronger than baseline `1e-4`. The grokking literature shows
  weight decay drives the transition from memorising to generalising solutions (Nanda
  et al. 2023). The small dataset independently motivates stronger regularization.
- **CosineAnnealingWarmRestarts over ReduceLROnPlateau:** `ReduceLROnPlateau` with
  patience=5 will drive LR to ~1e-6 within the first few hundred epochs on noisy val
  curves, ending all meaningful updates. `CosineAnnealingWarmRestarts` cycles back to
  base LR at each restart, enabling continued exploration. Restarts at epochs 100,
  300, 700, 1500, 3100, 6300, 12700… (T_0=100, T_mult=2).
- **Scheduler step:** `scheduler.step(epoch)` — called after `optimizer.step()` each
  epoch. The global epoch number is passed, not a within-job counter, so the schedule
  is continuous across Slurm job boundaries.

### 5.3 Checkpoint config dict for `5class_hpc_v0`

```python
"config": {
    "lr":              1e-3,
    "weight_decay":    1e-3,
    "batch_size":      BATCH_SIZE,
    "gray_method":     GRAY_METHOD,
    "pool_factor":     POOL_FACTOR,
    "optimizer":       "AdamW",
    "scheduler":       "CosineAnnealingWarmRestarts",
    "scheduler_T0":    100,
    "scheduler_Tmult": 2,
    "scheduler_eta_min": 1e-6,
    "loss":            "CosineLoss",
}
```

### 5.4 Per-epoch logging for `5class_hpc_v0` — full column specification

`training_log_full.csv` for the cosine model includes all standard columns plus
cosine-native metrics computed on the validation set every epoch.

**Standard columns** (same as §4.4):
```
epoch, train_loss, train_acc, val_loss, val_acc, lr,
elapsed_seconds, wall_clock_time, best_val_acc_so_far, model_type, phase
```

Here `val_loss` is the mean cosine loss `(1 − s_y)` over the validation set, and
`train_loss` is the weighted cosine loss. `val_acc` and `train_acc` are
`argmax(similarities) == labels` — identical computation to the baseline.

**Cosine-native columns** (added each epoch after the standard val loop):
```
val_ovr_auc_mean       mean OVR AUC across all 5 classes
val_ovr_auc_AC3        OVR AUC for class AC3
val_ovr_auc_F08D       OVR AUC for class F08D
val_ovr_auc_F09D       OVR AUC for class F09D
val_ovr_auc_F11D       OVR AUC for class F11D
val_ovr_auc_NC1        OVR AUC for class NC1
val_silhouette         silhouette score in cosine distance space (on 128-dim ĥ)
val_intra_sim_mean     mean of per-class intra-class cosine similarities
val_inter_sim_mean     mean of all cross-class cosine similarities
```

**Implementation of the cosine-native columns:**

All of these are computed during or immediately after the validation loop each epoch.
The validation set has 200 samples — small enough that all pairwise operations are
fast (200×200 matrix).

```python
import numpy as np
from sklearn.metrics import roc_auc_score, silhouette_score
import torch.nn.functional as F

def compute_val_cosine_metrics(model, val_loader, device, class_names):
    """
    Runs one pass over the validation set.
    Returns:
        sims_all:     (N, C) numpy array of raw cosine similarities
        features_all: (N, 128) numpy array of L2-normalized ĥ vectors
        labels_all:   (N,) numpy array of integer class labels
    Then computes and returns a dict of cosine-native metrics.
    """
    model.eval()
    sims_list, feats_list, label_list = [], [], []

    # Hook to capture the 128-dim feature vector before the classifier
    captured = {}
    def _hook(module, input, output):
        captured["h"] = output.detach().cpu()
    handle = model.global_pool.register_forward_hook(_hook)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            sims = model(images).cpu()           # (B, C) cosine similarities
            sims_list.append(sims)
            label_list.append(labels)
            feats_list.append(captured["h"])

    handle.remove()

    sims_all  = torch.cat(sims_list,  dim=0).numpy()   # (N, C)
    feats_all = torch.cat(feats_list, dim=0)            # (N, 128)
    feats_norm = F.normalize(feats_all, p=2, dim=1).numpy()
    labels_all = torch.cat(label_list, dim=0).numpy()  # (N,)

    metrics = {}

    # --- OVR AUC (thresholds sweep [-1, 1] naturally from similarity values) ---
    for c, name in enumerate(class_names):
        y_true  = (labels_all == c).astype(int)
        y_score = sims_all[:, c]                 # raw cosine sims in [-1, 1]
        metrics[f"val_ovr_auc_{name}"] = roc_auc_score(y_true, y_score)
    metrics["val_ovr_auc_mean"] = np.mean(
        [metrics[f"val_ovr_auc_{n}"] for n in class_names]
    )

    # --- Silhouette score (cosine distance = 1 - cosine_similarity) ---
    # Requires at least 2 classes present in val set (always true here).
    metrics["val_silhouette"] = silhouette_score(
        feats_norm, labels_all, metric="cosine"
    )

    # --- Intra/inter-class cosine similarity ---
    gram = feats_norm @ feats_norm.T        # (N, N) pairwise cosine similarities
    n = len(labels_all)
    intra_sims, inter_sims = [], []
    for c in range(len(class_names)):
        mask_c = (labels_all == c)
        # Intra: same-class pairs excluding diagonal
        block = gram[np.ix_(mask_c, mask_c)]
        nc = mask_c.sum()
        if nc > 1:
            intra = (block.sum() - nc) / (nc * (nc - 1))
            intra_sims.append(intra)
        # Inter: all cross-class pairs from this class
        mask_other = ~mask_c
        inter_sims.append(gram[np.ix_(mask_c, mask_other)].mean())

    metrics["val_intra_sim_mean"] = float(np.mean(intra_sims))
    metrics["val_inter_sim_mean"] = float(np.mean(inter_sims))

    return metrics
```

This function is called once per epoch during training, after the standard val loop.
The returned dict is merged into the row passed to `checkpoint_manager.log_epoch`.
The `global_pool` hook adds negligible overhead since it captures data already being
computed in the forward pass.

**Note on AUC threshold range:** `sklearn.metrics.roc_auc_score` (and `roc_curve`)
derives thresholds from the actual score values present in the data. Since cosine
similarities are bounded in [−1, 1], the threshold sweep will automatically cover
that range. No manual specification of threshold bounds is required. The assert in
`evaluate_cosine.py` (§6.2) confirms this as a sanity check.

---

## 6. `evaluate_cosine.py` — Cosine Model Evaluation Script

This is the sole evaluation entry point for all cosine-head models. It replaces
`evaluate.py` for `5class_hpc_v0`. All outputs go to
`models/5class_hpc_v0/analysis/cosine_eval/`.

### 6.1 Script outline

```
main(model_type, db_path, calibrate=False)
├── load FibrinCNNCosine via analysis_utils.load_model_from_registry
├── build val DataLoader (same split, no augmentation)
├── collect_cosine_outputs()
│       → sims_all (N,C), feats_norm (N,128), labels_all (N,)
├── Section A: metrics requiring no temperature
│   ├── accuracy, per-class precision/recall/F1, confusion matrix
│   ├── OVR ROC curves + AUC (threshold range [-1, 1])
│   ├── silhouette score in cosine distance
│   ├── intra/inter-class similarity summary + bar chart
│   └── UMAP projection of feats_norm (saved as PNG)
└── Section B: probabilistic metrics (only with --calibrate flag)
    ├── fit tau_cal via Platt scaling on val set
    ├── compute p = softmax(sims / tau_cal, dim=1)
    └── calibrated CE loss, ECE, reliability diagram
```

### 6.2 OVR ROC implementation — threshold range [−1, 1]

```python
from sklearn.metrics import roc_curve, auc

def compute_cosine_roc(sims_all, labels_all, class_names, output_dir):
    """
    One-vs-rest ROC curves using raw cosine similarities as scores.
    sklearn.metrics.roc_curve derives thresholds from actual score values,
    so the sweep is automatically over the range of cosine similarities
    present in the data (always a subset of [-1, 1]).
    """
    results = {}
    fig, ax = plt.subplots(figsize=(7, 6))

    for c, name in enumerate(class_names):
        y_true  = (labels_all == c).astype(int)
        y_score = sims_all[:, c]           # cosine similarities in [-1, 1]

        fpr, tpr, thresholds = roc_curve(
            y_true, y_score, drop_intermediate=False
        )
        roc_auc = auc(fpr, tpr)

        # Sanity check: confirm thresholds live in cosine similarity range
        assert thresholds.min() >= -1.0 - 1e-6, \
            f"Threshold below -1 for class {name}: {thresholds.min()}"
        assert thresholds.max() <=  1.0 + 1e-6, \
            f"Threshold above +1 for class {name}: {thresholds.max()}"

        results[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(), "auc": roc_auc,
        }
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("OVR ROC — cosine similarity scores (thresholds ∈ [−1, 1])")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(output_dir, "roc_data.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
```

### 6.3 Temperature calibration (Section B, `--calibrate` flag only)

```python
from scipy.optimize import minimize_scalar

def fit_temperature(sims_all, labels_all):
    """Post-hoc Platt scaling: find tau_cal minimising CE on val set.
    tau_cal is a post-hoc scalar for evaluation only — not a model parameter,
    never used during training."""
    sims_t   = torch.tensor(sims_all,   dtype=torch.float32)
    labels_t = torch.tensor(labels_all, dtype=torch.long)
    result = minimize_scalar(
        lambda tau: F.cross_entropy(sims_t / tau, labels_t).item(),
        bounds=(0.01, 10.0), method="bounded",
    )
    return result.x   # tau_calibrated
```

After fitting: compute `probs = softmax(sims / tau_cal, dim=1)`, report calibrated
CE loss, expected calibration error, and reliability diagram. Note `tau_cal`
prominently — deviation from 1.0 indicates miscalibration of raw similarities as
probability substitutes.

### 6.4 Output files

```
models/5class_hpc_v0/analysis/cosine_eval/
    metrics_summary.txt      accuracy, per-class F1, silhouette, intra/inter, AUC
    confusion_matrix.png
    roc_curves.png           OVR ROC; title notes threshold range [-1, 1]
    roc_data.json            fpr, tpr, thresholds, auc per class
    intra_inter_similarity.png  bar chart: intra vs inter per class
    embedding_umap.png       UMAP of 128-dim ĥ vectors colored by class
    calibration/             (created only with --calibrate flag)
        tau_calibrated.txt
        reliability_diagram.png
        calibrated_ce.txt
```

---

## 7. Evaluation Metric Summary

### 7.1 What requires temperature and what doesn't

| Metric | Temperature needed? | Notes |
|--------|-------------------|-------|
| Training loss (cosine) | No | `L = mean(w_y × (1 − s_y))` |
| Classification accuracy | No | `argmax(s)` invariant to τ |
| Per-class F1, confusion matrix | No | Derived from argmax |
| OVR AUC-ROC | No | Rank-based; thresholds sweep [−1, 1] |
| Silhouette score | No | Uses cosine distance on ĥ vectors |
| Intra/inter-class similarity | No | Computed on raw ĥ vectors |
| UMAP projection | No | Visualises raw ĥ vectors |
| GradCAM saliency maps | No | Gradient w.r.t. input pixels |
| Calibrated probabilities | Yes — `tau_cal` | `--calibrate` flag only |
| Reliability diagram, ECE | Yes — `tau_cal` | `--calibrate` flag only |

### 7.2 Per-epoch vs post-hoc metrics

| Metric | Logged per-epoch? | Where |
|--------|------------------|-------|
| train_loss, train_acc | Yes | `training_log_full.csv` |
| val_loss, val_acc | Yes | `training_log_full.csv` |
| val_ovr_auc_* (per class + mean) | Yes | `training_log_full.csv` |
| val_silhouette | Yes | `training_log_full.csv` |
| val_intra_sim_mean, val_inter_sim_mean | Yes | `training_log_full.csv` |
| lr, elapsed_seconds, wall_clock_time | Yes | `training_log_full.csv` |
| Confusion matrix, full ROC curves | Post-hoc | `evaluate_cosine.py` |
| UMAP projection | Post-hoc | `evaluate_cosine.py` |
| Calibrated probabilities | Post-hoc (`--calibrate`) | `evaluate_cosine.py` |

---

## 8. Directory Structure After All Changes

```
environment.yml              (modified — added scikit-learn, scipy, umap-learn)
checkpoint_manager.py        (NEW)
train_5class_hpc.py          (NEW)
evaluate_cosine.py           (NEW)
model.py                     (modified — added NormalizedLinear, FibrinCNNCosine)
train.py                     (modified — new args, new dispatch)
train_5class.py              (modified — GPU/checkpoint only, no loss change)
train_3class.py              (modified — GPU/checkpoint, unified phase loop)
analysis_utils.py            (modified — added 5class_hpc_v0 to registry)
run_analysis.py              (modified — hpc branch dispatches evaluate_cosine.py)
AGENT_EXPORT.md              (modified — new HPC section)

slurm/
    setup_env.sh
    train_5class.sh
    train_5class_hpc_v0.sh
    train_3class_scratch.sh
    train_3class_finetune.sh
    submit_all.sh
    logs/
        .gitkeep

models/
    5class/                          (UNCHANGED baseline)
        best_model.pth
        train_record.json
        training_history.json        (appended to on resume)
        training_log_full.csv        (NEW — standard columns)
        training_log.txt
        checkpoints/
            latest.pth
            epoch_00020.pth … epoch_00300.pth … epoch_10000.pth

    5class_hpc_v0/                   (NEW — cosine model)
        best_model.pth
        train_record.json            (same split seed as 5class)
        training_history.json
        training_log_full.csv        (NEW — standard + cosine-native columns)
        training_log.txt
        checkpoints/
            latest.pth
            epoch_00020.pth …
        analysis/
            cosine_eval/
                metrics_summary.txt
                confusion_matrix.png
                roc_curves.png
                roc_data.json
                intra_inter_similarity.png
                embedding_umap.png
                calibration/        (--calibrate only)

    3class_hemo/                     (modified — GPU/checkpoint, standard columns)
    3class_hemo_finetune/            (modified — GPU/checkpoint, phase tracking)
```

---

## 9. Files to Leave Completely Unchanged

| File | Reason |
|------|--------|
| `evaluate.py` | Baseline models only; FibrinCNN output is still CE logits |
| `hemophilia_analysis.py` | Baseline models only |
| `gradcam.py` | Works on any FibrinCNN; saliency unaffected by loss function |
| `plot_training_curves.py` | Reads `training_history.json`; format unchanged |
| `visualize_weights.py` | Works on any model with `.features` attribute |
| `misclassification_report.py` | Baseline models only |
| `preprocessing.py`, `augmentation.py` | No changes |
| `data_loader.py` | `pin_memory` handled in train scripts; no changes here |
| All `test_*.py` files | No changes |

---

## 10. Implementation Pitfalls

**Scheduler state across job boundaries.** `ReduceLROnPlateau` and
`CosineAnnealingWarmRestarts` both carry internal state that must be saved via
`scheduler.state_dict()` and restored via `scheduler.load_state_dict()`. Omitting
this resets patience counters or cosine phase at every job boundary.

**`training_history.json` on resume.** Load the existing file, then `extend()` each
list — do not overwrite. This keeps `plot_training_curves.py` working without changes.

**`pin_memory` on GPU.** Set `pin_memory=True` in both DataLoaders when
`device == cuda`. Keep `persistent_workers=True` when `num_workers > 0`.

**Global epoch number in scheduler.** For `CosineAnnealingWarmRestarts`, pass the
global (cumulative) epoch, not the within-job epoch, to `scheduler.step()`. Otherwise
each job restart resets the cosine phase and the schedule is discontinuous.

**`val_silhouette` computation uses `model.global_pool` hook.** If the `global_pool`
layer is renamed or restructured, the hook registration must be updated. Add a comment
in `train_5class_hpc.py` noting this dependency.

**Train record for `5class_hpc_v0`.** Generated fresh in `models/5class_hpc_v0/` on
the first run, using the same seed and split logic as `train_5class.py`. Do not pass
`--force-resplit` unless explicitly intended. The split must be comparable to the
`5class` baseline for fair evaluation.

**Compute nodes and internet.** Install all packages on the login node before
submitting jobs. Training scripts must not attempt network access.

---

## 11. Testing Steps (run on login node before submitting to queue)

```bash
# 1. Install environment
bash slurm/setup_env.sh
conda activate fibrin

# 2. Smoke test: baseline 5class with checkpointing
python train.py --model-type 5class --max-epochs 6 --epochs-per-job 3
python train.py --model-type 5class --max-epochs 6 --epochs-per-job 3 --resume
# Expected: "Resuming from epoch 3", runs epochs 4-6

# 3. Smoke test: cosine model
python train.py --model-type 5class_hpc_v0 --max-epochs 6 --epochs-per-job 3
python train.py --model-type 5class_hpc_v0 --max-epochs 6 --epochs-per-job 3 --resume
# Expected: same resume behaviour; cosine-native columns present in CSV

# 4. Verify CSV columns
python -c "
import pandas as pd
df = pd.read_csv('models/5class_hpc_v0/training_log_full.csv')
print(df.columns.tolist())
print(df)
"
# Expected columns include val_ovr_auc_mean, val_silhouette, val_intra_sim_mean, etc.

# 5. Verify checkpoints exist
ls models/5class/checkpoints/
ls models/5class_hpc_v0/checkpoints/

# 6. Run evaluate_cosine.py
python evaluate_cosine.py --model-type 5class_hpc_v0
ls models/5class_hpc_v0/analysis/cosine_eval/

# 7. Full test suite — no regressions
pytest -v -s
```

---

## 12. Slurm Submission Reference

```bash
# One-time setup
bash slurm/setup_env.sh

# Submit all jobs (run in parallel if GPUs available)
bash slurm/submit_all.sh

# Monitor
squeue -u $USER

# Watch cosine model progress live
tail -f models/5class_hpc_v0/training_log_full.csv

# Check a specific metric over time
python -c "
import pandas as pd
df = pd.read_csv('models/5class_hpc_v0/training_log_full.csv')
print(df[['epoch','val_acc','val_ovr_auc_mean','val_silhouette']].tail(20))
"
```

---

## 13. References

- Power, A. et al. (2022). *Grokking: Generalization beyond overfitting on small
  algorithmic datasets.* arXiv:2201.02177.
- Nanda, N. et al. (2023). *Progress measures for grokking via mechanistic
  interpretability.* ICLR 2023.
- Humayun, A. I., Balestriero, R., Baraniuk, R. (2024). *Deep networks always grok
  and here is why.* ICML 2024. PMLR 235:20722–20745.
- Andriushchenko, M. et al. (2025). *Grokking at the edge of numerical stability.*
  ICLR 2025.
- Liu, Z. et al. (2023). *Omnigrok: Grokking beyond algorithmic data.* ICLR 2023.
- Loshchilov, I., Hutter, F. (2017). *Decoupled weight decay regularization.*
  arXiv:1711.05101.
- Loshchilov, I., Hutter, F. (2016). *SGDR: Stochastic gradient descent with warm
  restarts.* arXiv:1608.03983.
- Barz, B., Denzler, J. (2020). *Deep learning on small datasets without pre-training
  using cosine loss.* WACV 2020.
