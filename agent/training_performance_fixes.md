# Training performance fixes — instructions for Claude Code

This document describes two bugs and one optimisation to apply to the
`dev-hpc0` branch. The changes affect `train_5class_hpc.py`, `train_5class.py`,
and `train_3class.py`. Rationale is included for each change so you can verify
intent before editing.

Note: the `--preload` fix (adding the flag to all four Slurm scripts) has
already been implemented and does not need to be applied.

---

## Background

Per-epoch wall time on the HPC P100 is ~10 minutes. Profiling shows the GPU is
idle for the overwhelming majority of that time. Two bugs in the current code
are responsible; a third change adds runtime GPU detection so the same scripts
run optimally on any of the available GPUs (P100, V100, H200).

Dataset sizes for reference:
- 800 training images (base), 4× augmentation = 3,200 samples, 200 batches at
  `batch_size=16`
- 200 validation images, 13 batches at `batch_size=16`

---

## Fix 1 — Remove double DataLoader construction on GPU

### What to change

In `train_5class_hpc.py` and `train_5class.py`, the `_main` function currently
calls `create_dataloaders()` unconditionally (which builds a DataLoader pair
with `pin_memory=False`), then immediately re-builds a second pair with
`pin_memory=True` when a GPU is present. The first pair is discarded. If
`--preload` is passed, all images are preloaded twice into separate dataset
objects, consuming 2× the memory.

Restructure the data-loading block so `create_dataloaders` is only called on
the CPU path:

```python
pin_memory = device.type == "cuda"

if pin_memory:
    from data_loader import (FibrinDataset, make_balanced_sampler,
                             load_split_from_record)
    from preprocessing import make_preprocessor
    avail_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", None)
                     or os.cpu_count() or 4)
    num_workers = min(8, max(2, avail_cpus - 1))
    train_df, val_df = load_split_from_record(RECORD_PATH, db_path)
    preprocessor = make_preprocessor(gray_method=GRAY_METHOD,
                                     pool_factor=POOL_FACTOR)
    train_ds = FibrinDataset(train_df, PHOTO_DIR, preprocessor,
                             augment=True, preload=preload)
    val_ds   = FibrinDataset(val_df,   PHOTO_DIR, preprocessor,
                             augment=False, preload=preload)
    sampler      = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    meta = {
        "class_names":        CLASS_NAMES,
        "train_size":         len(train_ds),
        "val_size":           len(val_ds),
        "train_class_counts": {
            cls: int((train_df["Exp_Type"] == cls).sum()) for cls in CLASS_NAMES
        },
    }
else:
    train_loader, val_loader, meta = create_dataloaders(
        db_path=db_path,
        photo_dir=PHOTO_DIR,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        gray_method=GRAY_METHOD,
        pool_factor=POOL_FACTOR,
        train_record_path=RECORD_PATH,
        preload=preload,
        force_resplit=False,
    )
```

Apply the equivalent restructuring to `train_5class.py`. Check whether
`train_3class.py` has the same pattern and apply there too if so.

### Why

The original code was structured to build `meta` from `create_dataloaders` and
then override the loaders. This silently doubled the preload cost and left the
CPU-path DataLoaders (`pin_memory=False`) in a variable that was immediately
overwritten. The fix makes the GPU and CPU paths fully independent.

---

## Fix 2 — Fix `torch.set_num_threads` over-subscription

### What to change

In `train_5class_hpc.py` and `train_5class.py`, find the line:

```python
torch.set_num_threads(os.cpu_count() or 4)
```

Replace with:

```python
avail_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", None)
                 or os.cpu_count() or 4)
torch.set_num_threads(avail_cpus)
```

Check `train_3class.py` for the same pattern and apply there too.

### Why

`os.cpu_count()` returns the node's physical core count (typically 32–64 on
BYU HPC nodes), not the 4 cores allocated by `--cpus-per-task=4`. Telling
PyTorch to spawn 32–64 intra-op threads when only 4 cores are available causes
severe context switching and cache thrashing. The corrected version uses the
same `SLURM_CPUS_PER_TASK` environment variable already used correctly for
`num_workers` in the GPU DataLoader path.

---

## Fix 3 — Runtime GPU detection for AMP and batch size

### What to change

Add the following helper function near the top of `train_5class_hpc.py` (after
imports, before the hyperparameter constants). Also add it to `train_5class.py`
and `train_3class.py`:

```python
def get_gpu_config(device: torch.device) -> dict:
    """Return training hyperparameters tuned for the assigned GPU.

    Detects SM version at runtime so the same script runs correctly
    regardless of which GPU Slurm assigns (P100, V100, or H200).

    SM version mapping:
        6.x  — Pascal (P100): no Tensor Cores; AMP saves memory only
        7.x  — Volta  (V100): Tensor Cores; fp16 + GradScaler
        8.x  — Ampere (A100): Tensor Cores; bf16 preferred
        9.x  — Hopper (H200): Tensor Cores + fp8; bf16, no GradScaler
    """
    if device.type != "cuda":
        return {
            "amp_enabled": False,
            "amp_dtype":   None,
            "use_scaler":  False,
            "batch_size":  BATCH_SIZE,
            "use_compile": False,
        }

    props = torch.cuda.get_device_properties(device)
    sm    = props.major
    print(f"GPU: {props.name}  (SM {props.major}.{props.minor}, "
          f"{props.total_memory / 1e9:.1f} GB)")

    if sm >= 9:        # H200 / H100 (Hopper)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.bfloat16,  # stable range; no GradScaler needed
            "use_scaler":  False,
            "batch_size":  64,
            "use_compile": True,            # torch.compile worthwhile on Hopper
        }
    elif sm >= 7:      # V100 (Volta) or A100 (Ampere)
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.float16,   # Tensor Cores; GradScaler required
            "use_scaler":  True,
            "batch_size":  64,
            "use_compile": False,
        }
    else:              # P100 (Pascal, SM 6.x) — no Tensor Cores
        return {
            "amp_enabled": True,
            "amp_dtype":   torch.float16,   # memory saving only; no compute gain
            "use_scaler":  True,
            "batch_size":  32,
            "use_compile": False,
        }
```

### Wire the config into `_main`

In `_main`, after device detection, call `get_gpu_config` and use its values:

```python
gpu_cfg    = get_gpu_config(device)
batch_size = gpu_cfg["batch_size"]   # replaces the hardcoded BATCH_SIZE constant
```

Optionally compile the model for Hopper before the training loop:

```python
model = FibrinCNNCosine(num_classes=len(CLASS_NAMES)).to(device)
if gpu_cfg["use_compile"]:
    print("Compiling model with torch.compile …")
    model = torch.compile(model)
```

Set up the gradient scaler (no-op when `enabled=False`, so safe for all
paths):

```python
scaler = torch.cuda.amp.GradScaler(enabled=gpu_cfg["use_scaler"])
```

Log the resolved config so it appears in the training log:

```python
print(f"AMP: {gpu_cfg['amp_enabled']}  dtype: {gpu_cfg['amp_dtype']}  "
      f"batch_size: {batch_size}  compile: {gpu_cfg['use_compile']}")
```

### Update `train_one_epoch` to use AMP

Replace the current `train_one_epoch` signature and body with:

```python
def train_one_epoch(model, loader, class_weights, optimizer, device,
                    scaler, amp_enabled, amp_dtype) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda",
                            dtype=amp_dtype,
                            enabled=amp_enabled):
            sims = model(images)
            loss = cosine_loss(sims, labels, class_weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct    += (sims.detach().argmax(dim=1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total
```

Update the call site in the epoch loop to pass the new arguments:

```python
train_loss, train_acc = train_one_epoch(
    model, train_loader, class_weights, optimizer, device,
    scaler=scaler,
    amp_enabled=gpu_cfg["amp_enabled"],
    amp_dtype=gpu_cfg["amp_dtype"],
)
```

Apply equivalent AMP changes to `evaluate_loader` and
`compute_val_cosine_metrics` if desired, though the benefit there is smaller
(no gradient computation).

Also add `"amp_enabled"`, `"amp_dtype"`, `"batch_size"` to `_CONFIG` so they
appear in WandB and checkpoint metadata.

### Why

The P100 (Pascal, SM 6.x) has no Tensor Cores. AMP with fp16 still halves
activation memory, allowing a larger batch size (32 instead of 16) and
consequently fewer GPU kernel launches per epoch — but convolutions themselves
do not run faster.

The V100 (Volta, SM 7.x) introduced Tensor Cores. AMP with fp16 accelerates
matrix multiplications and convolutions by 1.5–5× and halves memory, enabling
batch_size=64. A `GradScaler` is required because fp16 has a narrow dynamic
range and gradients can underflow to zero without loss scaling.

The H200 (Hopper, SM 9.x) supports bfloat16 on Tensor Cores with the same
range as float32, so gradient underflow is not a concern and `GradScaler` can
be omitted. `torch.compile` fuses ops into optimised CUDA kernels and is
worthwhile for a 10,000-epoch run on this architecture.

---

## Summary of expected outcomes

| Fix | Expected impact |
|-----|-----------------|
| Remove double DataLoader construction | Halves startup memory when `--preload` is used; eliminates redundant work |
| Fix `torch.set_num_threads` | Removes CPU over-subscription; reduces context switching |
| Runtime GPU detection + AMP | Additional 1.3–5× compute speedup depending on GPU assigned; config automatically correct for P100, V100, or H200 |

Fix 1 is by far the most important. Fixes 2 and 3 are correctness fixes with
secondary performance benefit. Fix 4 provides additional speedup once I/O is
no longer the bottleneck and future-proofs the scripts for whichever GPU Slurm
assigns.
