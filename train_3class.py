"""
train_3class.py — Train a 3-class FibrinCNN on F08D, F09D, F11D only.

Two modes selectable via --mode:

  scratch   Train FibrinCNN(num_classes=3) from random initialization.
            Same hyperparameters as the 5-class model; 30 epochs total.

  finetune  Load the trained 5-class model, replace the output head with a
            new Linear(128→3), and fine-tune in two phases:
              Phase 1 (5 epochs):  only the classifier head trains;
                                   feature extractor is frozen.
              Phase 2 (25 epochs): all layers fine-tune with a smaller LR.

Both modes reuse the SAME train/validation split that was used for the 5-class
model (loaded from models/5class/train_record.json) to ensure fair comparison.

Run with:
    python train.py --model-type 3class_scratch [--preload]
    python train.py --model-type 3class_finetune [--preload]

Or directly:
    python train_3class.py --mode scratch [--preload]
    python train_3class.py --mode finetune [--preload]

Outputs (saved to MODEL_DIR):
  best_model.pth        Weights of the best epoch (highest validation accuracy)
  train_record.json     Record of the train/validation split used
  training_history.json Per-epoch train/val loss and accuracy
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
        self._file   = open(path, "w", buffering=1)  # line-buffered
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


def _build_loaders(train_df, val_df, preprocessor, preload: bool = False):
    num_workers = min(8, max(4, (os.cpu_count() or 4) - 2))
    train_ds = FibrinDataset3(train_df, PHOTO_DIR, preprocessor,
                              augment=True, preload=preload)
    val_ds   = FibrinDataset3(val_df,   PHOTO_DIR, preprocessor,
                              augment=False, preload=preload)
    sampler  = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=(num_workers > 0))
    return train_loader, val_loader


def _epoch_loop(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs, model_path,
                freeze_features=False, start_epoch=1) -> Dict[str, List[float]]:
    """Run training loop; return per-epoch history dict."""
    best_acc = 0.0
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_accuracy": [],
        "val_loss":   [], "val_accuracy":   [],
    }
    for epoch in range(start_epoch, start_epoch + num_epochs):
        t0 = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            freeze_features=freeze_features)
        val_acc, val_loss = _evaluate_loader(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            tag = "  <- best"
        else:
            tag = ""
        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>10.4f}  "
              f"{val_loss:>10.4f}  {val_acc:>10.4f}  {elapsed:>10.1f}{tag}")
    return history


# ── Mode: scratch ─────────────────────────────────────────────────────────────

def train_scratch(model_dir: str, train_df, val_df,
                  device: torch.device, preload: bool = False) -> Dict:
    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)
    train_loader, val_loader = _build_loaders(train_df, val_df, preprocessor,
                                              preload=preload)

    model     = FibrinCNN(num_classes=3).to(device)
    weights   = _compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_HEAD,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    model_path = os.path.join(model_dir, "best_model.pth")

    print(f"\nTraining 3-class model from scratch for {EPOCHS_TOTAL} epochs …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time (s)':>10}")
    print("-" * 65)

    history = _epoch_loop(model, train_loader, val_loader, criterion, optimizer,
                          scheduler, device, EPOCHS_TOTAL, model_path)
    best = max(history["val_accuracy"])
    print(f"\nBest validation accuracy: {best:.4f}")
    print(f"Weights saved to:         {model_path}")
    return history


# ── Mode: finetune ────────────────────────────────────────────────────────────

def train_finetune(model_dir: str, train_df, val_df,
                   device: torch.device, preload: bool = False) -> Dict:
    if not os.path.exists(MODEL_5CLASS):
        raise FileNotFoundError(
            f"5-class model not found at {MODEL_5CLASS}. "
            "Run: python train.py --model-type 5class"
        )

    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)
    train_loader, val_loader = _build_loaders(train_df, val_df, preprocessor,
                                              preload=preload)

    model = FibrinCNN(num_classes=5)
    model.load_state_dict(
        torch.load(MODEL_5CLASS, map_location=device, weights_only=True)
    )
    model.classifier[-1] = nn.Linear(128, 3)
    model.to(device)

    weights   = _compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model_path = os.path.join(model_dir, "best_model.pth")

    # ── Phase 1: freeze feature extractor, train classifier only ──
    model.features.requires_grad_(False)
    model.global_pool.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.Adam(trainable, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode="max", factor=0.5, patience=5
    )

    print(f"\nPhase 1 — head-only training ({EPOCHS_PHASE1} epochs) …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time (s)':>10}")
    print("-" * 65)
    history1 = _epoch_loop(model, train_loader, val_loader, criterion, opt1, sched1,
                            device, EPOCHS_PHASE1, model_path,
                            freeze_features=True, start_epoch=1)

    # ── Phase 2: unfreeze all layers, fine-tune with smaller LR ──
    model.requires_grad_(True)
    opt2 = torch.optim.Adam(model.parameters(), lr=LR_FULL,
                            weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode="max", factor=0.5, patience=5
    )

    print(f"\nPhase 2 — full fine-tune ({EPOCHS_PHASE2} epochs, LR={LR_FULL}) …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time (s)':>10}")
    print("-" * 65)
    history2 = _epoch_loop(model, train_loader, val_loader, criterion, opt2, sched2,
                            device, EPOCHS_PHASE2, model_path,
                            freeze_features=False,
                            start_epoch=EPOCHS_PHASE1 + 1)

    best = max(history2["val_accuracy"])
    print(f"\nBest Phase-2 validation accuracy: {best:.4f}")
    print(f"Weights saved to:                 {model_path}")

    # Concatenate history across both phases
    history = {k: history1[k] + history2[k] for k in history1}
    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main(mode: str = None, preload: bool = False, db_path: str = DB_PATH) -> None:
    if mode is None:
        parser = argparse.ArgumentParser(
            description="Train a 3-class hemophilia CNN.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument("--mode", choices=["scratch", "finetune"], required=True)
        parser.add_argument("--preload", action="store_true",
                            help="Preload all images into RAM before training.")
        parser.add_argument("--db", default=DB_PATH, metavar="PATH",
                            help="Path to SQLite database (default: data/endpoint10.db).")
        args = parser.parse_args()
        mode = args.mode
        preload = args.preload
        db_path = args.db

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
        _main_inner(mode, device, model_dir, preload=preload, db_path=db_path)


def _main_inner(mode: str, device: torch.device, model_dir: str,
                preload: bool = False, db_path: str = DB_PATH) -> None:
    # Load canonical split and filter to hemophilia classes
    print(f"Loading split from {RECORD_5CLASS} …")
    train_df, val_df = load_split_from_record(RECORD_5CLASS, db_path)
    train_df = filter_classes(train_df, HEMO_CLASSES)
    val_df   = filter_classes(val_df,   HEMO_CLASSES)

    print(f"Train: {len(train_df)} images  Validation: {len(val_df)} images")
    print("Train class counts:     ",
          {c: int((train_df["Exp_Type"] == c).sum()) for c in HEMO_CLASSES})
    print("Validation class counts:",
          {c: int((val_df["Exp_Type"]  == c).sum()) for c in HEMO_CLASSES})

    # Save split record for this model directory
    record_path = os.path.join(model_dir, "train_record.json")
    save_train_record(train_df, val_df, record_path,
                      mode=mode, class_names=CLASS_NAMES_3,
                      note="Hemophilia subset of the 5-class split")

    # Train
    if mode == "scratch":
        history = train_scratch(model_dir, train_df, val_df, device, preload=preload)
    else:
        history = train_finetune(model_dir, train_df, val_df, device, preload=preload)

    # Save training history
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Final evaluation with best checkpoint
    print("\nDetailed evaluation of best model on 3-class hemophilia validation set:")
    model_path = os.path.join(model_dir, "best_model.pth")
    final_model = FibrinCNN(num_classes=3).to(device)
    final_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    final_model.eval()
    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)
    _, val_loader = _build_loaders(train_df, val_df, preprocessor)
    results = evaluate_model(final_model, val_loader, device, CLASS_NAMES_3)
    print(results["report_str"])


if __name__ == "__main__":
    main()
