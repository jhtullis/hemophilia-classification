"""
train_5class_hpc.py — Training loop for the 5-class cosine-head model (5class_hpc_v0).

Architecture: FibrinCNNCosine — identical to FibrinCNN except the final
Linear(128→5) is replaced with NormalizedLinear(128→5), which outputs cosine
similarities in [−1, 1] rather than unconstrained logits.

Loss: cosine_loss — L = mean(w_y × (1 − s_y)). No softmax. No Softmax Collapse
risk. Loss range [0, 2]. Gradient acts on correct-class similarity only; implicit
competition arises from weight normalization.

Optimizer: AdamW (lr=1e-3, weight_decay=1e-3). Stronger weight decay than the
Adam baseline (1e-4) — motivated by the grokking literature (Nanda et al. 2023)
showing weight decay drives the transition from memorizing to generalizing solutions.

Scheduler: CosineAnnealingWarmRestarts (T_0=100, T_mult=2, eta_min=1e-6). Restarts
at epochs 100, 300, 700, 1500, 3100, 6300… enabling continued exploration over
10,000-epoch runs.  ReduceLROnPlateau would drive LR to ~1e-6 within a few hundred
epochs on noisy val curves, ending all meaningful updates.

Per-epoch logging: training_log_full.csv includes standard columns plus cosine-native
metrics (OVR AUC per class, silhouette in cosine distance, intra/inter-class
similarity means) computed on the validation set each epoch.

Run with:
    python train.py --model-type 5class_hpc_v0 --max-epochs 10000 --epochs-per-job 200

HPC / Slurm (self-resubmitting):
    python train.py --model-type 5class_hpc_v0 --max-epochs 10000 --epochs-per-job 200 --resume

Exit codes:
    0   — epochs-per-job budget exhausted; more epochs remain (Slurm: resubmit)
    100 — max-epochs reached; training complete (Slurm: do not resubmit)

Outputs (models/5class_hpc_v0/):
    best_model.pth          Weights of the epoch with highest validation accuracy
    train_record.json       Canonical train/validation split (same seed as 5class)
    training_history.json   Per-epoch train/val loss and accuracy
    training_log.txt        Full stdout log
    training_log_full.csv   Extended per-epoch CSV (standard + cosine-native columns)
    checkpoints/            latest.pth + periodic snapshots
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, silhouette_score
from torch.utils.data import DataLoader

from checkpoint_manager import (finish_wandb, get_wandb_run_id, init_wandb,
                                 load_latest_checkpoint, log_epoch,
                                 save_checkpoint)
from data_loader import CLASS_NAMES, create_dataloaders
from evaluate import evaluate_model
from model import FibrinCNNCosine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR      = os.path.dirname(__file__)
DB_PATH   = os.path.join(_DIR, "data", "endpoint10.db")
PHOTO_DIR = os.path.join(_DIR, "data", "photos")
MODEL_DIR = os.path.join(_DIR, "models", "5class_hpc_v0")
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pth")
RECORD_PATH = os.path.join(MODEL_DIR, "train_record.json")

# Reuse the canonical 5-class split for fair comparison
RECORD_5CLASS = os.path.join(_DIR, "models", "5class", "train_record.json")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-3      # stronger than baseline (1e-4); motivated by grokking lit.
TRAIN_RATIO  = 0.75
SEED         = 42
GRAY_METHOD  = "lab_l"
POOL_FACTOR  = 10

_CONFIG = {
    "lr":                1e-3,
    "weight_decay":      1e-3,
    "batch_size":        BATCH_SIZE,
    "gray_method":       GRAY_METHOD,
    "pool_factor":       POOL_FACTOR,
    "optimizer":         "AdamW",
    "scheduler":         "CosineAnnealingWarmRestarts",
    "scheduler_T0":      100,
    "scheduler_Tmult":   2,
    "scheduler_eta_min": 1e-6,
    "loss":              "CosineLoss",
}


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate all stdout writes to both the terminal and a log file."""

    def __init__(self, path: str):
        self._file   = open(path, "a", buffering=1)   # append — safe on resume
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._stdout
        self._file.close()


# ---------------------------------------------------------------------------
# Cosine loss
# ---------------------------------------------------------------------------

def cosine_loss(
    similarities: torch.Tensor,   # (B, C) raw cosine similarities from forward()
    labels:       torch.Tensor,   # (B,) class indices
    class_weights: torch.Tensor,  # (C,) inverse-frequency weights
) -> torch.Tensor:
    """Pure cosine loss: L = mean_over_batch[ w_y * (1 − s_y) ]

    s_y is the cosine similarity to the correct class prototype Ŵ_y.
    Loss range: [0, 2]. No softmax. No Softmax Collapse risk.
    Gradient acts only on the correct-class similarity; implicit competition
    arises from weight normalization (rotating Ŵ_y toward ĥ rotates it away
    from other class prototypes on the unit sphere).
    """
    correct_sims = similarities[torch.arange(len(labels)), labels]  # (B,)
    weights      = class_weights[labels]                             # (B,)
    return (weights * (1.0 - correct_sims)).mean()


def compute_class_weights(train_class_counts: dict,
                          device: torch.device) -> torch.Tensor:
    total = sum(train_class_counts.values())
    weights = torch.tensor(
        [total / train_class_counts[cls] for cls in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return weights


# ---------------------------------------------------------------------------
# Per-epoch training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, class_weights, optimizer,
                    device) -> Tuple[float, float]:
    """Returns (avg_cosine_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        sims = model(images)                         # (B, C) cosine similarities
        loss = cosine_loss(sims, labels, class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (sims.detach().argmax(dim=1) == labels).sum().item()
        total   += len(labels)
    return total_loss / total, correct / total


def evaluate_loader(model, loader, class_weights,
                    device) -> Tuple[float, float]:
    """Returns (accuracy, avg_cosine_loss)."""
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            sims = model(images)
            total_loss += cosine_loss(sims, labels, class_weights).item() * len(labels)
            correct += (sims.argmax(dim=1) == labels).sum().item()
            total   += len(labels)
    return correct / total, total_loss / total


def compute_val_cosine_metrics(model, val_loader, device,
                               class_names: List[str]) -> Dict[str, float]:
    """Compute cosine-native validation metrics for one epoch.

    Hooks model.global_pool to capture 128-dim pre-classifier features (ĥ).
    NOTE: if model.global_pool is renamed this hook registration must be updated.

    Returns a dict with keys:
        val_ovr_auc_<class>  (one per class)
        val_ovr_auc_mean
        val_silhouette
        val_intra_sim_mean
        val_inter_sim_mean
    """
    model.eval()
    sims_list, feats_list, label_list = [], [], []

    # Hook captures the 128-dim vector after global_pool + flatten,
    # before the classifier. global_pool output is (B, 256, 1, 1); the
    # classifier's first Linear expects (B, 256) — flatten happens in forward().
    # We hook the output of global_pool and flatten it manually here.
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        # output: (B, 256, 1, 1) — flatten to (B, 256)
        captured["h"] = output.detach().cpu().flatten(1)

    handle = model.global_pool.register_forward_hook(_hook)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            sims = model(images).cpu()          # (B, C) cosine similarities
            sims_list.append(sims)
            label_list.append(labels)
            feats_list.append(captured["h"])

    handle.remove()

    sims_all   = torch.cat(sims_list,  dim=0).numpy()   # (N, C)
    feats_all  = torch.cat(feats_list, dim=0)            # (N, 256)
    feats_norm = F.normalize(feats_all, p=2, dim=1).numpy()   # (N, 256) unit vecs
    labels_all = torch.cat(label_list, dim=0).numpy()   # (N,)

    metrics: Dict[str, float] = {}

    # --- OVR AUC ---
    for c, name in enumerate(class_names):
        y_true  = (labels_all == c).astype(int)
        y_score = sims_all[:, c]
        metrics[f"val_ovr_auc_{name}"] = float(roc_auc_score(y_true, y_score))
    metrics["val_ovr_auc_mean"] = float(
        np.mean([metrics[f"val_ovr_auc_{n}"] for n in class_names])
    )

    # --- Silhouette (cosine distance = 1 - cosine_similarity) ---
    metrics["val_silhouette"] = float(
        silhouette_score(feats_norm, labels_all, metric="cosine")
    )

    # --- Intra / inter-class cosine similarity ---
    gram = feats_norm @ feats_norm.T    # (N, N) pairwise cosine similarities
    intra_sims, inter_sims = [], []
    for c in range(len(class_names)):
        mask_c = labels_all == c
        block  = gram[np.ix_(mask_c, mask_c)]
        nc     = mask_c.sum()
        if nc > 1:
            intra = (block.sum() - nc) / (nc * (nc - 1))
            intra_sims.append(float(intra))
        mask_other = ~mask_c
        inter_sims.append(float(gram[np.ix_(mask_c, mask_other)].mean()))

    metrics["val_intra_sim_mean"] = float(np.mean(intra_sims))
    metrics["val_inter_sim_mean"] = float(np.mean(inter_sims))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    preload: bool = False,
    db_path: str = DB_PATH,
    device: torch.device = None,
    resume: bool = False,
    max_epochs: int = 10000,
    epochs_per_job: int = 200,
    wandb_enabled: bool = True,
    wandb_project: str = "fibrin-cnn",
    wandb_run_name: str = None,
) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_path = os.path.join(MODEL_DIR, "training_log.txt")
    print(f"Training log → {log_path}")
    with _Tee(log_path):
        _exit_code = _main(
            preload=preload, db_path=db_path, device=device,
            resume=resume, max_epochs=max_epochs, epochs_per_job=epochs_per_job,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
    sys.exit(_exit_code)


def _main(
    preload: bool = False,
    db_path: str = DB_PATH,
    device: torch.device = None,
    resume: bool = False,
    max_epochs: int = 10000,
    epochs_per_job: int = 200,
    wandb_enabled: bool = True,
    wandb_project: str = "fibrin-cnn",
    wandb_run_name: str = None,
) -> int:
    if device is None:
        device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEED)

    pin_memory = device.type == "cuda"

    # --- Data (reuse 5-class split for fair comparison) ---
    print("Loading data …")
    # Copy the canonical 5-class split into this model's directory on first run
    # so both models train and validate on identical image sets.
    if not os.path.exists(RECORD_PATH) and os.path.exists(RECORD_5CLASS):
        import shutil
        shutil.copy(RECORD_5CLASS, RECORD_PATH)
        print(f"Copied split from {RECORD_5CLASS} → {RECORD_PATH}")

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

    if pin_memory:
        from data_loader import (FibrinDataset, make_balanced_sampler,
                                 load_split_from_record)
        from preprocessing import make_preprocessor
        num_workers = min(8, max(4, (os.cpu_count() or 4) - 2))
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
        meta["train_class_counts"] = {
            cls: int((train_df["Exp_Type"] == cls).sum()) for cls in CLASS_NAMES
        }

    print(f"\nClass names:  {meta['class_names']}")
    print(f"Train images (with 4× augmentation): {meta['train_size']}")
    print(f"Validation images:  {meta['val_size']}")
    print("Train class counts (base):", meta["train_class_counts"])

    # --- Model ---
    model = FibrinCNNCosine(num_classes=len(CLASS_NAMES)).to(device)
    print(f"\n{model.summary()}")

    # --- Class weights ---
    class_weights = compute_class_weights(meta["train_class_counts"], device)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )

    # --- Resume from checkpoint ---
    best_acc    = 0.0
    start_epoch = 1
    history = {"train_loss": [], "train_accuracy": [],
               "val_loss":   [], "val_accuracy":   []}

    if resume:
        ckpt = load_latest_checkpoint(MODEL_DIR)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_acc    = ckpt["best_acc"]
            start_epoch = ckpt["epoch"] + 1
            history     = ckpt["history"]
    else:
        ckpt = None
        load_latest_checkpoint(MODEL_DIR)  # prints "Starting from scratch"

    wandb_run_id = ckpt.get("wandb_run_id") if ckpt else None
    init_wandb(
        config={**_CONFIG, "model_type": "5class_hpc_v0"},
        model_type="5class_hpc_v0",
        project=wandb_project,
        entity=None,
        run_name=wandb_run_name,
        run_id=wandb_run_id,
        resume_run=(resume and wandb_run_id is not None),
        enabled=wandb_enabled,
    )

    history_path = os.path.join(MODEL_DIR, "training_history.json")
    if resume and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    end_epoch = min(start_epoch + epochs_per_job - 1, max_epochs)

    if start_epoch > max_epochs:
        print(f"Already reached max_epochs={max_epochs}. Nothing to do.")
        return 100

    print(f"\nTraining epochs {start_epoch}–{end_epoch} of {max_epochs} "
          f"on {device} (cosine loss, AdamW, CosineWarmRestarts) …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'AUC Mean':>10}  {'Time (s)':>10}")
    print("-" * 80)

    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                class_weights, optimizer, device)
        val_acc, val_loss = evaluate_loader(model, val_loader,
                                            class_weights, device)

        # Pass global epoch to maintain continuous cosine schedule across job boundaries
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Cosine-native metrics (computed on full val set)
        cosine_metrics = compute_val_cosine_metrics(model, val_loader, device,
                                                    CLASS_NAMES)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            tag = "  ← best"
        else:
            tag = ""

        auc_mean = cosine_metrics.get("val_ovr_auc_mean", float("nan"))
        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>10.4f}  "
              f"{val_loss:>10.4f}  {val_acc:>10.4f}  {auc_mean:>10.4f}  "
              f"{elapsed:>10.1f}{tag}")

        # --- Checkpoint ---
        ckpt_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc":             best_acc,
            "history":              history,
            "model_type":           "5class_hpc_v0",
            "phase":                1,
            "config":               _CONFIG,
            "wandb_run_id":         get_wandb_run_id(),
        }
        save_checkpoint(ckpt_state, MODEL_DIR, epoch, is_best, max_epochs)

        row = {
            "epoch":               epoch,
            "train_loss":          round(train_loss, 6),
            "train_acc":           round(train_acc, 6),
            "val_loss":            round(val_loss, 6),
            "val_acc":             round(val_acc, 6),
            "lr":                  current_lr,
            "elapsed_seconds":     round(elapsed, 2),
            "wall_clock_time":     datetime.utcnow().isoformat(),
            "best_val_acc_so_far": round(best_acc, 6),
            "model_type":          "5class_hpc_v0",
            "phase":               1,
        }
        row.update({k: round(v, 6) for k, v in cosine_metrics.items()})
        log_epoch(MODEL_DIR, row)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy: {best_acc:.4f}")

    if end_epoch >= max_epochs:
        print(f"Reached max_epochs={max_epochs}. Training complete.")
        print("\nDetailed evaluation of best model on validation set:")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                         weights_only=True))
        # Evaluate using argmax of cosine similarities (no softmax needed)
        results = evaluate_model(model, val_loader, device, meta["class_names"])
        print(results["report_str"])
        finish_wandb()
        return 100
    else:
        print(f"Job quota reached (epoch {end_epoch}). Resubmit to continue.")
        finish_wandb()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--db", default=DB_PATH, metavar="PATH")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--epochs-per-job", type=int, default=200)
    args = parser.parse_args()
    from train import get_device
    main(preload=args.preload, db_path=args.db, device=get_device(),
         resume=args.resume, max_epochs=args.max_epochs,
         epochs_per_job=args.epochs_per_job)
