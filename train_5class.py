"""
train_5class.py — Training loop for the 5-class FibrinCNN.

Run with:
    python train.py --model-type 5class [options]
    python train.py --model-type 5class_hpc_baseline [options]

Or directly:
    python train_5class.py [--preload] [--resume] [--max-epochs N] [--epochs-per-job N]

Outputs (models/5class/ or models/5class_hpc_baseline/):
    best_model.pth          Weights of the epoch with the highest validation accuracy
    train_record.json       Record of which images were assigned to training
    training_history.json   Per-epoch train/val loss and accuracy
    training_log.txt        Full stdout log
    training_log_full.csv   Extended per-epoch CSV log
    checkpoints/            latest.pth + periodic snapshots
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoint_manager import (finish_wandb, get_wandb_run_id, init_wandb,
                                 load_latest_checkpoint, log_epoch,
                                 save_checkpoint, should_save_periodic)
from data_loader import CLASS_NAMES, create_dataloaders
from evaluate import evaluate_model
from model import FibrinCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)
DB_PATH    = os.path.join(_DIR, "data", "endpoint10.db")
PHOTO_DIR  = os.path.join(_DIR, "data", "photos")
MODEL_DIR  = os.path.join(_DIR, "models", "5class")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
RECORD_PATH = os.path.join(MODEL_DIR, "train_record.json")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO  = 0.75
SEED         = 42
GRAY_METHOD  = "lab_l"
POOL_FACTOR  = 10

_CONFIG = {
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_size":   BATCH_SIZE,
    "gray_method":  GRAY_METHOD,
    "pool_factor":  POOL_FACTOR,
    "optimizer":    "Adam",
    "scheduler":    "ReduceLROnPlateau",
    "loss":         "CrossEntropyLoss",
}


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate all stdout writes to both the terminal and a log file."""

    def __init__(self, path: str):
        self._file   = open(path, "a", buffering=1)  # append — safe on resume
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
# Training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(train_class_counts: dict,
                          device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss, ordered by CLASS_MAP."""
    total = sum(train_class_counts.values())
    weights = torch.tensor(
        [total / train_class_counts[cls] for cls in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return weights


def train_one_epoch(model: FibrinCNN, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy) over all training batches."""
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def evaluate_loader(model: FibrinCNN, loader: DataLoader,
                    criterion: nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    """Returns (accuracy, avg_loss) on a DataLoader without gradients."""
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total, total_loss / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    preload: bool = False,
    db_path: str = DB_PATH,
    force_resplit: bool = False,
    device: torch.device = None,
    resume: bool = False,
    max_epochs: int = 10000,
    epochs_per_job: int = 200,
    model_dir: str = None,
    wandb_enabled: bool = True,
    wandb_project: str = "fibrin-cnn",
    wandb_run_name: str = None,
) -> None:
    effective_dir = model_dir if model_dir is not None else MODEL_DIR
    os.makedirs(effective_dir, exist_ok=True)
    log_path = os.path.join(effective_dir, "training_log.txt")
    print(f"Training log → {log_path}")
    with _Tee(log_path):
        _exit_code = _main(
            preload=preload, db_path=db_path, force_resplit=force_resplit,
            device=device, resume=resume,
            max_epochs=max_epochs, epochs_per_job=epochs_per_job,
            model_dir=effective_dir,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
    sys.exit(_exit_code)


def _main(
    preload: bool = False,
    db_path: str = DB_PATH,
    force_resplit: bool = False,
    device: torch.device = None,
    resume: bool = False,
    max_epochs: int = 10000,
    epochs_per_job: int = 200,
    model_dir: str = None,
    wandb_enabled: bool = True,
    wandb_project: str = "fibrin-cnn",
    wandb_run_name: str = None,
) -> int:
    effective_dir = model_dir if model_dir is not None else MODEL_DIR
    model_type_str = os.path.basename(effective_dir)

    # Resolve record path; copy canonical 5class split on first run for new dirs
    effective_record = os.path.join(effective_dir, "train_record.json")
    if not os.path.exists(effective_record) and effective_dir != MODEL_DIR:
        canonical = os.path.join(MODEL_DIR, "train_record.json")
        if os.path.exists(canonical):
            import shutil
            shutil.copy(canonical, effective_record)
            print(f"Copied split from {canonical} → {effective_record}")

    effective_model_path = os.path.join(effective_dir, "best_model.pth")
    history_path = os.path.join(effective_dir, "training_history.json")

    if device is None:
        device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEED)

    pin_memory = device.type == "cuda"

    # --- Data ---
    print("Loading data …")
    train_loader, val_loader, meta = create_dataloaders(
        db_path=db_path,
        photo_dir=PHOTO_DIR,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        gray_method=GRAY_METHOD,
        pool_factor=POOL_FACTOR,
        train_record_path=effective_record,
        preload=preload,
        force_resplit=force_resplit,
        # Override pin_memory via DataLoader kwargs isn't directly supported
        # in create_dataloaders — handled below by rebuilding loaders if needed.
    )

    # Rebuild loaders with pin_memory=True when on GPU
    if pin_memory:
        from data_loader import (FibrinDataset, make_balanced_sampler,
                                 load_split_from_record, filter_classes)
        from preprocessing import make_preprocessor
        avail_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", None) or os.cpu_count() or 4)
        num_workers = min(8, max(2, avail_cpus - 1))
        train_df, val_df = load_split_from_record(effective_record, db_path)
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
    print("Validation class counts:  ", meta["val_class_counts"])

    # --- Model ---
    model = FibrinCNN(num_classes=len(CLASS_NAMES)).to(device)

    # --- Loss: inverse-frequency class weights ---
    class_weights = compute_class_weights(meta["train_class_counts"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )

    # --- Resume from checkpoint ---
    best_acc    = 0.0
    start_epoch = 1
    history = {"train_loss": [], "train_accuracy": [],
               "val_loss":   [], "val_accuracy":   []}

    if resume:
        ckpt = load_latest_checkpoint(effective_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_acc    = ckpt["best_acc"]
            start_epoch = ckpt["epoch"] + 1
            history     = ckpt["history"]
    else:
        ckpt = None
        load_latest_checkpoint(effective_dir)  # prints "Starting from scratch"

    wandb_run_id = ckpt.get("wandb_run_id") if ckpt else None
    init_wandb(
        config={**_CONFIG, "model_type": model_type_str},
        model_type=model_type_str,
        project=wandb_project,
        entity=None,
        run_name=wandb_run_name,
        run_id=wandb_run_id,
        resume_run=(resume and wandb_run_id is not None),
        enabled=wandb_enabled,
    )

    # Load and extend training_history.json on resume
    if resume and os.path.exists(history_path):
        with open(history_path) as f:
            saved_hist = json.load(f)
        # history from checkpoint is authoritative; sync to json length
        history = saved_hist

    end_epoch = min(start_epoch + epochs_per_job - 1, max_epochs)

    if start_epoch > max_epochs:
        print(f"Already reached max_epochs={max_epochs}. Nothing to do.")
        return 100

    print(f"\nTraining epochs {start_epoch}–{end_epoch} of {max_epochs} "
          f"on {device} …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time (s)':>10}")
    print("-" * 65)

    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_acc, val_loss = evaluate_loader(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            tag = "  ← best"
        else:
            tag = ""

        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>10.4f}  "
              f"{val_loss:>10.4f}  {val_acc:>10.4f}  {elapsed:>10.1f}{tag}")

        # --- Checkpoint ---
        ckpt_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc":             best_acc,
            "history":              history,
            "model_type":           model_type_str,
            "phase":                1,
            "config":               _CONFIG,
            "wandb_run_id":         get_wandb_run_id(),
        }
        save_checkpoint(ckpt_state, effective_dir, epoch, is_best, max_epochs)

        log_epoch(effective_dir, {
            "epoch":               epoch,
            "train_loss":          round(train_loss, 6),
            "train_acc":           round(train_acc, 6),
            "val_loss":            round(val_loss, 6),
            "val_acc":             round(val_acc, 6),
            "lr":                  current_lr,
            "elapsed_seconds":     round(elapsed, 2),
            "wall_clock_time":     datetime.utcnow().isoformat(),
            "best_val_acc_so_far": round(best_acc, 6),
            "model_type":          model_type_str,
            "phase":               1,
        })

        # --- Save training history (for plot_training_curves.py) ---
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy: {best_acc:.4f}")

    # --- Determine exit code ---
    if end_epoch >= max_epochs:
        print(f"Reached max_epochs={max_epochs}. Training complete.")
        # Final detailed evaluation
        print("\nDetailed evaluation of best model on validation set:")
        model.load_state_dict(torch.load(effective_model_path, map_location=device,
                                         weights_only=True))
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
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--epochs-per-job", type=int, default=200)
    args = parser.parse_args()
    from train import get_device
    main(preload=args.preload, db_path=args.db, force_resplit=args.force_resplit,
         device=get_device(), resume=args.resume,
         max_epochs=args.max_epochs, epochs_per_job=args.epochs_per_job)
