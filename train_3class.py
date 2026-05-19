"""
train_3class.py — Train a 3-class FibrinCNN on F08D, F09D, F11D only.

Two modes selectable via --mode:

  scratch   Train FibrinCNN(num_classes=3) from random initialization.
            Same hyperparameters as the 5-class model.

  finetune  Load the trained 5-class model, replace the output head with a
            new Linear(128→3), and fine-tune in two phases:
              Phase 1 (5 epochs):  only the classifier head trains;
                                   feature extractor is frozen.
              Phase 2 (25 epochs): all layers fine-tune with a smaller LR.

Both modes reuse the SAME train/validation split that was used for the 5-class
model (loaded from models/5class/train_record.json) to ensure fair comparison.

Run with:
    python train.py --model-type 3class_scratch [--resume] [--max-epochs N]
    python train.py --model-type 3class_finetune [--resume] [--max-epochs N]

Or directly:
    python train_3class.py --mode scratch [options]
    python train_3class.py --mode finetune [options]

Outputs (saved to MODEL_DIR):
  best_model.pth          Weights of the best epoch (highest validation accuracy)
  train_record.json       Record of the train/validation split used
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
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from checkpoint_manager import (load_latest_checkpoint, log_epoch,
                                 save_checkpoint)
from data_loader import (CLASS_MAP, CLASS_NAMES, FibrinDataset,
                         filter_classes, load_split_from_record,
                         make_balanced_sampler, save_train_record)
from evaluate import evaluate_model
from model import FibrinCNN
from preprocessing import make_preprocessor

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR          = os.path.dirname(__file__)
DB_PATH       = os.path.join(_DIR, "data", "endpoint10.db")
PHOTO_DIR     = os.path.join(_DIR, "data", "photos")
RECORD_5CLASS = os.path.join(_DIR, "models", "5class", "train_record.json")
MODEL_5CLASS  = os.path.join(_DIR, "models", "5class", "best_model.pth")

# ── 3-class label mapping ─────────────────────────────────────────────────────
HEMO_CLASSES  = ["F08D", "F09D", "F11D"]
CLASS_MAP_3   = {"F08D": 0, "F09D": 1, "F11D": 2}
CLASS_NAMES_3 = ["F08D", "F09D", "F11D"]

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE    = 16
LR_HEAD       = 1e-3   # Phase 1 (head-only) and scratch
LR_FULL       = 1e-4   # Phase 2 (full fine-tune)
WEIGHT_DECAY  = 1e-4
EPOCHS_TOTAL  = 30
EPOCHS_PHASE1 = 5      # finetune: head-only warmup
EPOCHS_PHASE2 = 25     # finetune: full unfreeze (must sum to EPOCHS_TOTAL)
GRAY_METHOD   = "lab_l"
POOL_FACTOR   = 10
SEED          = 42


# ── Logging helper ────────────────────────────────────────────────────────────

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


# ── FibrinDataset3 ────────────────────────────────────────────────────────────

class FibrinDataset3(FibrinDataset):
    """FibrinDataset variant that uses the 3-class label mapping.

    FibrinDataset.__getitem__ hard-codes CLASS_MAP which maps F08D→1, F09D→2,
    F11D→3.  For a 3-class model those labels would be out-of-range; this
    subclass remaps them to 0, 1, 2 via CLASS_MAP_3.
    """

    def __getitem__(self, idx: int):
        tensor, _ = super().__getitem__(idx)          # inherit image loading
        base_idx = idx // self.multiplier
        row = self.df.iloc[base_idx]
        label = CLASS_MAP_3[row["Exp_Type"]]          # remap to 0/1/2
        return tensor, label


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_class_weights(train_df, device: torch.device) -> torch.Tensor:
    counts = {cls: int((train_df["Exp_Type"] == cls).sum()) for cls in CLASS_NAMES_3}
    total  = sum(counts.values())
    return torch.tensor([total / counts[c] for c in CLASS_NAMES_3],
                        dtype=torch.float32, device=device)


def _train_one_epoch(model: FibrinCNN, loader: DataLoader,
                     criterion: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device,
                     freeze_features: bool = False) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy) for one training epoch."""
    model.train()
    if freeze_features:
        model.features.eval()
        model.global_pool.eval()

    total_loss = 0.0
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total   += len(labels)
    return total_loss / total, correct / total


def _evaluate_loader(model: FibrinCNN, loader: DataLoader,
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
            total   += len(labels)
    return correct / total, total_loss / total


def _build_loaders(train_df, val_df, preprocessor, preload: bool = False,
                   pin_memory: bool = False):
    num_workers = min(8, max(4, (os.cpu_count() or 4) - 2))
    train_ds = FibrinDataset3(train_df, PHOTO_DIR, preprocessor,
                              augment=True, preload=preload)
    val_ds   = FibrinDataset3(val_df,   PHOTO_DIR, preprocessor,
                              augment=False, preload=preload)
    sampler  = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    return train_loader, val_loader


# ── Unified training loop ─────────────────────────────────────────────────────

def _run_loop(
    model, train_loader, val_loader, criterion,
    opt1, sched1, opt2, sched2,
    device, model_dir, model_type_str,
    start_epoch, end_epoch, max_epochs,
    initial_phase,  # 1 or 2; from checkpoint on resume
    initial_best_acc, initial_history,
) -> Tuple[Dict[str, List[float]], float, int]:
    """Unified epoch loop supporting both phases across Slurm job boundaries.

    Phase 1 (epochs 1..EPOCHS_PHASE1): head-only.
    Phase 2 (epochs EPOCHS_PHASE1+1..): full fine-tune.
    Scratch mode always uses opt1/sched1 and stays in phase 1.
    """
    best_acc = initial_best_acc
    history  = initial_history
    current_phase = initial_phase
    model_path = os.path.join(model_dir, "best_model.pth")

    _config = {
        "lr_head":      LR_HEAD,
        "lr_full":      LR_FULL,
        "weight_decay": WEIGHT_DECAY,
        "batch_size":   BATCH_SIZE,
        "gray_method":  GRAY_METHOD,
        "pool_factor":  POOL_FACTOR,
        "optimizer":    "Adam",
        "scheduler":    "ReduceLROnPlateau",
        "loss":         "CrossEntropyLoss",
    }

    print(f"\nTraining epochs {start_epoch}–{end_epoch} of {max_epochs} …\n")
    print(f"{'Epoch':>6}  {'Phase':>5}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time (s)':>10}")
    print("-" * 76)

    for epoch in range(start_epoch, end_epoch + 1):
        # Determine phase and optimizer/scheduler for this epoch
        if opt2 is None:
            # scratch mode: always phase 1
            use_opt, use_sched, freeze = opt1, sched1, False
            current_phase = 1
        elif epoch <= EPOCHS_PHASE1:
            model.features.requires_grad_(False)
            model.global_pool.requires_grad_(False)
            use_opt, use_sched, freeze = opt1, sched1, True
            current_phase = 1
        else:
            if current_phase == 1:   # first crossing of phase boundary
                model.requires_grad_(True)
            use_opt, use_sched, freeze = opt2, sched2, False
            current_phase = 2

        t0 = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, use_opt, device,
            freeze_features=freeze)
        val_acc, val_loss = _evaluate_loader(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        use_sched.step(val_acc)
        current_lr = use_opt.param_groups[0]["lr"]

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            tag = "  <- best"
        else:
            tag = ""

        print(f"{epoch:>6}  {current_phase:>5}  {train_loss:>12.4f}  "
              f"{train_acc:>10.4f}  {val_loss:>10.4f}  {val_acc:>10.4f}  "
              f"{elapsed:>10.1f}{tag}")

        # Checkpoint — save both opt/sched states so either phase can resume
        ckpt_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": use_opt.state_dict(),
            "scheduler_state_dict": use_sched.state_dict(),
            "best_acc":             best_acc,
            "history":              history,
            "model_type":           model_type_str,
            "phase":                current_phase,
            "config":               _config,
        }
        save_checkpoint(ckpt_state, model_dir, epoch, is_best, max_epochs)

        log_epoch(model_dir, {
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
            "phase":               current_phase,
        })

    return history, best_acc, current_phase


# ── Mode: scratch ─────────────────────────────────────────────────────────────

def train_scratch(model_dir: str, train_df, val_df,
                  device: torch.device, preload: bool = False,
                  resume: bool = False, max_epochs: int = 10000,
                  epochs_per_job: int = 200) -> Tuple[Dict, int]:
    pin_memory = device.type == "cuda"
    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)
    train_loader, val_loader = _build_loaders(train_df, val_df, preprocessor,
                                              preload=preload, pin_memory=pin_memory)

    model     = FibrinCNN(num_classes=3).to(device)
    weights   = _compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_HEAD,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )

    best_acc = 0.0
    start_epoch = 1
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    if resume:
        ckpt = load_latest_checkpoint(model_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_acc    = ckpt["best_acc"]
            start_epoch = ckpt["epoch"] + 1
            history     = ckpt["history"]
    else:
        load_latest_checkpoint(model_dir)

    history_path = os.path.join(model_dir, "training_history.json")
    if resume and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    end_epoch = min(start_epoch + epochs_per_job - 1, max_epochs)

    if start_epoch > max_epochs:
        print(f"Already reached max_epochs={max_epochs}. Nothing to do.")
        return history, 100

    history, best, _ = _run_loop(
        model, train_loader, val_loader, criterion,
        opt1=optimizer, sched1=scheduler, opt2=None, sched2=None,
        device=device, model_dir=model_dir, model_type_str="3class_scratch",
        start_epoch=start_epoch, end_epoch=end_epoch, max_epochs=max_epochs,
        initial_phase=1, initial_best_acc=best_acc, initial_history=history,
    )
    print(f"\nBest validation accuracy: {best:.4f}")
    return history, 100 if end_epoch >= max_epochs else 0


# ── Mode: finetune ────────────────────────────────────────────────────────────

def train_finetune(model_dir: str, train_df, val_df,
                   device: torch.device, preload: bool = False,
                   resume: bool = False, max_epochs: int = 10000,
                   epochs_per_job: int = 200) -> Tuple[Dict, int]:
    if not resume and not os.path.exists(MODEL_5CLASS):
        raise FileNotFoundError(
            f"5-class model not found at {MODEL_5CLASS}. "
            "Run: python train.py --model-type 5class"
        )

    pin_memory = device.type == "cuda"
    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)
    train_loader, val_loader = _build_loaders(train_df, val_df, preprocessor,
                                              preload=preload, pin_memory=pin_memory)

    model = FibrinCNN(num_classes=5)
    model.classifier[-1] = nn.Linear(128, 3)
    model.to(device)

    weights   = _compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Phase 1 optimizer (head only)
    model.features.requires_grad_(False)
    model.global_pool.requires_grad_(False)
    trainable_head = [p for p in model.parameters() if p.requires_grad]
    opt1   = torch.optim.Adam(trainable_head, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )

    # Phase 2 optimizer (all params; created now so we can restore state on resume)
    model.requires_grad_(True)
    opt2   = torch.optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )
    # Re-freeze for phase 1 start
    model.features.requires_grad_(False)
    model.global_pool.requires_grad_(False)

    best_acc      = 0.0
    start_epoch   = 1
    initial_phase = 1
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    if resume:
        ckpt = load_latest_checkpoint(model_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            initial_phase = ckpt.get("phase", 1)
            # Restore the optimizer that was active at checkpoint time
            if initial_phase == 1:
                opt1.load_state_dict(ckpt["optimizer_state_dict"])
                sched1.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                opt2.load_state_dict(ckpt["optimizer_state_dict"])
                sched2.load_state_dict(ckpt["scheduler_state_dict"])
                model.requires_grad_(True)   # ensure unfrozen for phase 2
            best_acc    = ckpt["best_acc"]
            start_epoch = ckpt["epoch"] + 1
            history     = ckpt["history"]
    else:
        # Load pretrained 5-class weights
        model.load_state_dict(
            torch.load(MODEL_5CLASS, map_location=device, weights_only=True),
            strict=False,   # classifier[-1] shape differs; that's expected
        )
        load_latest_checkpoint(model_dir)  # prints "Starting from scratch"

    history_path = os.path.join(model_dir, "training_history.json")
    if resume and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    end_epoch = min(start_epoch + epochs_per_job - 1, max_epochs)

    if start_epoch > max_epochs:
        print(f"Already reached max_epochs={max_epochs}. Nothing to do.")
        return history, 100

    history, best, _ = _run_loop(
        model, train_loader, val_loader, criterion,
        opt1=opt1, sched1=sched1, opt2=opt2, sched2=sched2,
        device=device, model_dir=model_dir, model_type_str="3class_finetune",
        start_epoch=start_epoch, end_epoch=end_epoch, max_epochs=max_epochs,
        initial_phase=initial_phase, initial_best_acc=best_acc,
        initial_history=history,
    )
    print(f"\nBest validation accuracy: {best:.4f}")
    return history, 100 if end_epoch >= max_epochs else 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main(mode: str = None, preload: bool = False, db_path: str = DB_PATH,
         device: torch.device = None, resume: bool = False,
         max_epochs: int = 10000, epochs_per_job: int = 200) -> None:
    if mode is None:
        parser = argparse.ArgumentParser(
            description="Train a 3-class hemophilia CNN.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument("--mode", choices=["scratch", "finetune"], required=True)
        parser.add_argument("--preload", action="store_true")
        parser.add_argument("--db", default=DB_PATH, metavar="PATH")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--max-epochs", type=int, default=10000)
        parser.add_argument("--epochs-per-job", type=int, default=200)
        args = parser.parse_args()
        mode         = args.mode
        preload      = args.preload
        db_path      = args.db
        resume       = args.resume
        max_epochs   = args.max_epochs
        epochs_per_job = args.epochs_per_job
        from train import get_device
        device = get_device()

    if device is None:
        device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEED)

    model_dir = os.path.join(
        _DIR, "models",
        "3class_hemo" if mode == "scratch" else "3class_hemo_finetune",
    )
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(model_dir, "training_log.txt")
    print(f"Training log → {log_path}")

    with _Tee(log_path):
        _exit_code = _main_inner(mode, device, model_dir, preload=preload,
                                  db_path=db_path, resume=resume,
                                  max_epochs=max_epochs,
                                  epochs_per_job=epochs_per_job)
    sys.exit(_exit_code)


def _main_inner(mode: str, device: torch.device, model_dir: str,
                preload: bool = False, db_path: str = DB_PATH,
                resume: bool = False, max_epochs: int = 10000,
                epochs_per_job: int = 200) -> int:
    print(f"Loading split from {RECORD_5CLASS} …")
    train_df, val_df = load_split_from_record(RECORD_5CLASS, db_path)
    train_df = filter_classes(train_df, HEMO_CLASSES)
    val_df   = filter_classes(val_df,   HEMO_CLASSES)

    print(f"Train: {len(train_df)} images  Validation: {len(val_df)} images")
    print("Train class counts:     ",
          {c: int((train_df["Exp_Type"] == c).sum()) for c in HEMO_CLASSES})
    print("Validation class counts:",
          {c: int((val_df["Exp_Type"]  == c).sum()) for c in HEMO_CLASSES})

    record_path = os.path.join(model_dir, "train_record.json")
    save_train_record(train_df, val_df, record_path,
                      mode=mode, class_names=CLASS_NAMES_3,
                      note="Hemophilia subset of the 5-class split")

    train_kwargs = dict(
        model_dir=model_dir, train_df=train_df, val_df=val_df,
        device=device, preload=preload, resume=resume,
        max_epochs=max_epochs, epochs_per_job=epochs_per_job,
    )
    if mode == "scratch":
        history, exit_code = train_scratch(**train_kwargs)
    else:
        history, exit_code = train_finetune(**train_kwargs)

    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    if exit_code == 100:
        print("\nDetailed evaluation of best model on 3-class hemophilia validation set:")
        model_path = os.path.join(model_dir, "best_model.pth")
        final_model = FibrinCNN(num_classes=3).to(device)
        final_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        final_model.eval()
        preprocessor = make_preprocessor(gray_method=GRAY_METHOD,
                                         pool_factor=POOL_FACTOR)
        _, val_loader = _build_loaders(train_df, val_df, preprocessor)
        results = evaluate_model(final_model, val_loader, device, CLASS_NAMES_3)
        print(results["report_str"])

    return exit_code


if __name__ == "__main__":
    main()
