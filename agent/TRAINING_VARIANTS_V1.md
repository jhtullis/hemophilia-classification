# Training Variants v1a / v1b / v1c

**For Claude Code — fibrin clot classification project**

Creates six new training configurations as variants of the existing
`5class_hpc_v0` and `patch_v0` procedures. Changes include both hyperparameters
(scheduler, weight decay) and one architectural parameter (dropout), which
requires the model classes to accept dropout as a constructor argument.

---

## 0. Background: What Changes and Why

### v0 baseline hyperparameters (both models)
```python
LR           = 1e-3
WEIGHT_DECAY = 1e-3
T_0          = 100
T_MULT       = 2.0
ETA_MIN      = 1e-6
DROPOUT      = 0.5
```

### Problem diagnosed in v0
- LR is too small relative to weight decay, particularly in the low phase of long
  cosine cycles
- With T_mult=2, cycle lengths grow as 100, 200, 400, 800, 1600 … epochs
- By cycle 4–5, the LR spends 700–1400 epochs near eta_min=1e-6 while weight
  decay=1e-3 operates unchallenged → weight norms compress → 20-point accuracy
  drops during troughs
- Training accuracy plateau at ~0.3 with validation at ~0.7–0.8 suggests dropout
  at p=0.5 is suppressing the training signal too aggressively

### Variant parameter changes

| Variant | T_mult | eta_min | weight_decay | dropout | Rationale |
|---------|--------|---------|--------------|---------|-----------|
| v1a | 1.5 | 1e-4 | 1e-3 (unchanged) | 0.3 | Options 2+3 + reduced dropout |
| v1b | 2.0 (unchanged) | 1e-6 (unchanged) | 3e-4 | 0.3 | Reduce WD 3× + reduced dropout |
| v1c | 1.5 | 1e-4 | 5e-4 | 0.3 | Options 2+3 + 2× WD reduction + reduced dropout |

**Why 3× reduction for v1b (1e-3 → 3e-4):** A 2× cut (5e-4) is likely too
conservative given the observed severity of trough degradation. A 5× cut (2e-4)
risks undermining the grokking-motivated weight decay entirely. 3× is the
principled midpoint — it meaningfully reduces the WD/LR ratio from 1:1 to ~3:1
at base LR, while still providing sufficient weight norm pressure to drive the
memorization-to-generalization transition.

**Why 2× reduction for v1c (1e-3 → 5e-4):** v1c already addresses the floor
problem via raised eta_min and slowed cycle growth. The WD reduction is
complementary rather than primary, so a conservative 2× is appropriate.

**Why dropout 0.5 → 0.3 in all v1 variants:** The training accuracy plateau at
~0.3 with validation at ~0.7–0.8 indicates the classifier head is seeing too much
noise to learn reliably from individual patches. Reducing p from 0.5 to 0.3
reduces the effective masking rate without removing regularization entirely.
Since all three v1 variants include this change, any improvement common to all
three can be attributed to the dropout reduction, while differences between them
reflect the scheduler and WD changes.

---

## 1. Implementation Approach: Config System

Rather than creating six near-identical training scripts, implement a lightweight
config system. Each variant is a small Python dict in a `configs/` file. The
existing training scripts are updated to accept `--config` as an argument.

### New file: `configs/training_configs.py`

```python
"""
Hyperparameter configs for all training variants.
Each config overrides only the fields that differ from v0.
"""

# ── Base configs (v0 values — do not modify) ──────────────────────────────

_5CLASS_HPC_V0_BASE = {
    "model_type":    "5class_hpc_v0",
    "model_dir":     "models/5class_hpc_v0",
    "lr":            1e-3,
    "weight_decay":  1e-3,
    "T_0":           100,
    "T_mult":        2.0,
    "eta_min":       1e-6,
    "dropout":       0.5,     # v0 value; v1 variants use 0.3
    "batch_size":    16,
    "max_epochs":    10000,
    "grad_clip":     None,    # not used in v0
}

_PATCH_V0_BASE = {
    "model_type":    "patch_v0",
    "model_dir":     "models/patch_5class_v0",
    "lr":            1e-3,
    "weight_decay":  1e-3,
    "T_0":           100,
    "T_mult":        2.0,
    "eta_min":       1e-6,
    "dropout":       0.5,     # v0 value; v1 variants use 0.3
    "batch_size":    64,
    "patches_per_img": 20,
    "max_epochs":    10000,
    "grad_clip":     None,
}

# ── 5class_hpc variants ────────────────────────────────────────────────────

CONFIGS = {

    "5class_hpc_v0": _5CLASS_HPC_V0_BASE,

    "5class_hpc_v1a": {
        **_5CLASS_HPC_V0_BASE,
        "model_type":   "5class_hpc_v1a",
        "model_dir":    "models/5class_hpc_v1a",
        # Options 2+3: raise floor, slow cycle growth; weight_decay unchanged
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "dropout":      0.3,
    },

    "5class_hpc_v1b": {
        **_5CLASS_HPC_V0_BASE,
        "model_type":   "5class_hpc_v1b",
        "model_dir":    "models/5class_hpc_v1b",
        # WD reduced 3× only; scheduler unchanged
        "weight_decay": 3e-4,
        "dropout":      0.3,
    },

    "5class_hpc_v1c": {
        **_5CLASS_HPC_V0_BASE,
        "model_type":   "5class_hpc_v1c",
        "model_dir":    "models/5class_hpc_v1c",
        # Options 2+3 + 2× WD reduction
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "weight_decay": 5e-4,
        "dropout":      0.3,
    },

    # ── patch variants ─────────────────────────────────────────────────────

    "patch_v0": _PATCH_V0_BASE,

    "patch_v1a": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v1a",
        "model_dir":    "models/patch_5class_v1a",
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "dropout":      0.3,
    },

    "patch_v1b": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v1b",
        "model_dir":    "models/patch_5class_v1b",
        "weight_decay": 3e-4,
        "dropout":      0.3,
    },

    "patch_v1c": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v1c",
        "model_dir":    "models/patch_5class_v1c",
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "weight_decay": 5e-4,
        "dropout":      0.3,
    },
}
```

---

## 2. Changes to Existing Training Scripts

### `train_5class_hpc.py`

Add `--config` CLI argument and replace hardcoded hyperparameters with
config lookups. The change is localized to the argument parsing and
hyperparameter initialization sections only — the training loop itself
is unchanged.

**Add to argument parser:**
```python
p.add_argument(
    "--config",
    default="5class_hpc_v0",
    choices=list(CONFIGS.keys()),
    help="Hyperparameter config name (default: 5class_hpc_v0).",
)
```

**Replace hardcoded constants with config lookups:**
```python
from configs.training_configs import CONFIGS

cfg          = CONFIGS[args.config]
MODEL_DIR    = cfg["model_dir"]
LR           = cfg["lr"]
WEIGHT_DECAY = cfg["weight_decay"]
T_0          = cfg["T_0"]
T_MULT       = cfg["T_mult"]
ETA_MIN      = cfg["eta_min"]
DROPOUT_P    = cfg["dropout"]
BATCH_SIZE   = cfg["batch_size"]
GRAD_CLIP    = cfg.get("grad_clip", None)
```

**Model instantiation — pass dropout from config:**
```python
# For train_5class_hpc.py:
model = FibrinCNNCosine(num_classes=NUM_CLASSES, dropout_p=DROPOUT_P).to(device)

# For train_patch.py:
model = FibrinPatchCNN(num_classes=NUM_CLASSES, dropout_p=DROPOUT_P).to(device)
```

**Scheduler instantiation (replace existing):**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=T_0,
    T_mult=T_MULT,
    eta_min=ETA_MIN,
)
```

**Gradient clipping (add inside training loop after loss.backward()):**
```python
if GRAD_CLIP is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
optimizer.step()
```

**Checkpoint config dict — add new fields:**
```python
"config": {
    ...existing fields...,
    "T_mult":       T_MULT,
    "eta_min":      ETA_MIN,
    "dropout_p":    DROPOUT_P,
    "grad_clip":    GRAD_CLIP,
    "variant":      args.config,
}
```

**Model directory:** all output paths use `MODEL_DIR` from config — no other
path changes needed since the existing code already parameterises on `MODEL_DIR`.

### `train_patch.py`

Apply identical changes: add `--config` argument, replace hardcoded
hyperparameters with config lookups, pass `dropout_p` to model constructor,
add gradient clipping hook.

The `--config` choices for `train_patch.py` should be limited to patch
variants only:
```python
choices=["patch_v0", "patch_v1a", "patch_v1b", "patch_v1c"]
```

### `model.py` — add `dropout_p` parameter to `FibrinCNNCosine`

`FibrinCNNCosine.__init__` currently hardcodes `Dropout(p=0.5)` in the
classifier head. Add `dropout_p: float = 0.5` as a constructor argument
and pass it through:

```python
class FibrinCNNCosine(nn.Module):
    def __init__(self, num_classes: int = 5, dropout_p: float = 0.5):
        super().__init__()
        # ... existing feature blocks unchanged ...
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),    # was hardcoded Dropout(p=0.5)
            NormalizedLinear(128, num_classes),
        )
```

The existing `FibrinCNN` (cross-entropy baseline) is **not** modified —
it does not use this config system.

### `model_patch.py` — add `dropout_p` parameter to `FibrinPatchCNN`

Same change: replace `Dropout(p=0.5)` in the classifier head with
`Dropout(p=dropout_p)` where `dropout_p: float = 0.5` is a constructor
argument. Default of 0.5 preserves v0 behaviour when called without
the argument.

---

## 3. Model Registry Updates

### `analysis_utils.py` — add new model types to `MODEL_REGISTRY`

```python
MODEL_REGISTRY = {
    # ── existing ──────────────────────────────────────────────────────────
    "5class":           ("models/5class",              5, CLASS_MAP_5),
    "5class_hpc_v0":    ("models/5class_hpc_v0",       5, CLASS_MAP_5),
    "3class_scratch":   ("models/3class_hemo",          3, CLASS_MAP_3),
    "3class_finetune":  ("models/3class_hemo_finetune", 3, CLASS_MAP_3),
    # ── new v1 variants ────────────────────────────────────────────────────
    "5class_hpc_v1a":   ("models/5class_hpc_v1a",      5, CLASS_MAP_5),
    "5class_hpc_v1b":   ("models/5class_hpc_v1b",      5, CLASS_MAP_5),
    "5class_hpc_v1c":   ("models/5class_hpc_v1c",      5, CLASS_MAP_5),
    "patch_v0":         ("models/patch_5class_v0",     5, CLASS_MAP_5),
    "patch_v1a":        ("models/patch_5class_v1a",    5, CLASS_MAP_5),
    "patch_v1b":        ("models/patch_5class_v1b",    5, CLASS_MAP_5),
    "patch_v1c":        ("models/patch_5class_v1c",    5, CLASS_MAP_5),
}
```

Update `load_model_from_registry` to recognise patch model types:
```python
if "hpc" in model_type or "patch" in model_type:
    from model import FibrinCNNCosine
    # patch types use FibrinPatchCNN instead:
    if "patch" in model_type:
        from model_patch import FibrinPatchCNN
        model = FibrinPatchCNN(num_classes=num_classes)
    else:
        model = FibrinCNNCosine(num_classes=num_classes)
```

---

## 4. Slurm Scripts

Create one Slurm script per variant. All scripts follow the existing
self-resubmitting chain pattern from `train_5class_hpc_v0.sh`. The only
differences between scripts are the `--config` argument and the log filenames.

### Naming convention
```
slurm/train_5class_hpc_v1a.sh
slurm/train_5class_hpc_v1b.sh
slurm/train_5class_hpc_v1c.sh
slurm/train_patch_v1a.sh
slurm/train_patch_v1b.sh
slurm/train_patch_v1c.sh
```

### Template (5class_hpc_v1a — adapt config name and log paths for others)

```bash
#!/bin/bash --login
#SBATCH --job-name=fibrin_5class_hpc_v1a
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/5class_hpc_v1a_%j.out
#SBATCH --error=logs/5class_hpc_v1a_%j.err

module load miniforge3
conda activate fibrin

CONFIG="5class_hpc_v1a"

python train.py \
  --model-type 5class_hpc_v1a \
  --config     "${CONFIG}" \
  --max-epochs 10000 \
  --epochs-per-job 200 \
  --resume

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Epoch quota exhausted — resubmitting."
    sbatch slurm/train_5class_hpc_v1a.sh
elif [ ${EXIT_CODE} -eq 100 ]; then
    echo "Training complete."
fi
```

For patch variants, replace `--model-type 5class_hpc_v1a` with
`--model-type patch_v1a` and update `--config`, `--job-name`, `--output`,
and `--error` accordingly.

---

## 5. Directory Structure to Create

Before first submission, create output directories:

```bash
mkdir -p models/5class_hpc_v1a/checkpoints
mkdir -p models/5class_hpc_v1b/checkpoints
mkdir -p models/5class_hpc_v1c/checkpoints
mkdir -p models/patch_5class_v1a/checkpoints
mkdir -p models/patch_5class_v1b/checkpoints
mkdir -p models/patch_5class_v1c/checkpoints
mkdir -p logs
```

---

## 6. Files to Create / Modify

| File | Action | Notes |
|------|--------|-------|
| `configs/__init__.py` | **Create** | Empty init to make `configs` a package |
| `configs/training_configs.py` | **Create** | Config dicts for all variants |
| `model.py` | **Modify** | Add `dropout_p` param to `FibrinCNNCosine`; default 0.5 preserves v0 |
| `model_patch.py` | **Modify** | Add `dropout_p` param to `FibrinPatchCNN`; default 0.5 preserves v0 |
| `train_5class_hpc.py` | **Modify** | Add `--config` arg; pass `dropout_p`; replace hardcoded params; add grad clip hook |
| `train_patch.py` | **Modify** | Same changes as above |
| `analysis_utils.py` | **Modify** | Add new model types to `MODEL_REGISTRY` |
| `slurm/train_5class_hpc_v1a.sh` | **Create** | Chain Slurm script |
| `slurm/train_5class_hpc_v1b.sh` | **Create** | Chain Slurm script |
| `slurm/train_5class_hpc_v1c.sh` | **Create** | Chain Slurm script |
| `slurm/train_patch_v1a.sh` | **Create** | Chain Slurm script |
| `slurm/train_patch_v1b.sh` | **Create** | Chain Slurm script |
| `slurm/train_patch_v1c.sh` | **Create** | Chain Slurm script |
| `models/5class_hpc_v1{a,b,c}/checkpoints/` | **Create dirs** | Output directories |
| `models/patch_5class_v1{a,b,c}/checkpoints/` | **Create dirs** | Output directories |

**Files that must NOT be modified:**
`train_5class.py`, `train_3class.py`, `data_loader.py`, `preprocessing.py`.
`FibrinCNN` (cross-entropy baseline) inside `model.py` must not be modified —
only `FibrinCNNCosine` gains the `dropout_p` parameter.
The v0 training configs in `training_configs.py` must not be altered after v0
has started training — add only, never modify existing entries.

---

## 7. Verification

Before submitting to HPC, run a smoke test for each new config:

```bash
# 5class_hpc variants
python train.py --model-type 5class_hpc_v1a --config 5class_hpc_v1a \
    --max-epochs 3 --epochs-per-job 3
python train.py --model-type 5class_hpc_v1b --config 5class_hpc_v1b \
    --max-epochs 3 --epochs-per-job 3
python train.py --model-type 5class_hpc_v1c --config 5class_hpc_v1c \
    --max-epochs 3 --epochs-per-job 3

# patch variants
python train_patch.py --config patch_v1a --max-epochs 3 \
    --epochs-per-job 3 --batch-size 4 --num-workers 0
python train_patch.py --config patch_v1b --max-epochs 3 \
    --epochs-per-job 3 --batch-size 4 --num-workers 0
python train_patch.py --config patch_v1c --max-epochs 3 \
    --epochs-per-job 3 --batch-size 4 --num-workers 0
```

For each smoke test, assert:
- `models/{variant}/training_log_full.csv` contains 3 rows
- `models/{variant}/checkpoints/latest.pth` exists
- Checkpoint config dict contains the correct `T_mult`, `eta_min`,
  `weight_decay`, and `dropout_p` values for that variant:
  ```python
  ckpt = torch.load("models/5class_hpc_v1a/checkpoints/latest.pth")
  assert ckpt["config"]["dropout_p"]    == 0.3
  assert ckpt["config"]["T_mult"]       == 1.5
  assert ckpt["config"]["eta_min"]      == 1e-4
  assert ckpt["config"]["weight_decay"] == 1e-3
  ```

---

## 8. Additional Mutations to Consider

These are not requested for immediate implementation but are worth queuing
as future variants if the v1a–c results provide useful signal.

### v1d — Direct LR increase (most targeted fix for the diagnosis)
```python
"lr":           3e-3,      # raised from 1e-3
"weight_decay": 1e-3,      # unchanged — isolates LR effect
"T_mult":       2.0,       # unchanged
"eta_min":      1e-6,      # unchanged
"dropout":      0.3,       # consistent with v1 variants
"grad_clip":    10.0,      # required at higher LR to prevent restart instability
```
Rationale: v1a–c all address the LR/WD imbalance indirectly (by shrinking WD
or raising the floor). v1d tests the direct fix. Gradient clipping at
max_norm=10.0 (matching Barz & Denzler) is included as a safeguard at the
higher LR. If v1d outperforms v1a–c, the primary bottleneck is LR magnitude
rather than scheduler shape or WD level.

### v1e — Fixed cycle length (T_mult=1, snapshot ensemble at inference)
```python
"T_mult":  1.0,
"T_0":     300,      # fixed 300-epoch cycles throughout
"eta_min": 1e-5,
"dropout": 0.3,
```
Rationale: Eliminates the growing-gap problem entirely by preventing cycles
from ever lengthening. Inference uses a snapshot ensemble of the best model
from each completed cycle. Appropriate if v1a still shows degradation in
long troughs despite the slowed growth rate. Requires changes to
`evaluate_cosine.py` and `evaluate_patch.py` to load and ensemble multiple
checkpoints.

### General recommendation on ordering
Run v1a, v1b, v1c in parallel first. They jointly test the dropout reduction
(common to all three) and isolate three distinct scheduler/WD hypotheses.
Any improvement common to all three variants can be attributed to dropout;
differences between them indicate whether the floor (v1a), WD magnitude (v1b),
or their combination (v1c) is the dominant remaining cause of trough degradation.

Compare trough depths and peak accuracies at epoch ~1500 (after cycle 4 under
v0/v1b's T_mult=2 schedule; after ~cycle 6 under v1a/v1c's T_mult=1.5 schedule).
If v1a and v1c still show meaningful troughs despite slowed cycle growth,
proceed to v1e (fixed cycles). If all v1 variants improve over v0 but
plateaus remain low, v1d (direct LR increase) is the next step.
