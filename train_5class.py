"""
train_5class.py — Training loop for the 5-class FibrinCNN.

Run with:
    python train.py --model-type 5class [--preload]

Or directly:
    python train_5class.py [--preload]

Outputs (models/5class/):
    best_model.pth        Weights of the epoch with the highest validation accuracy
    train_record.json     Record of which images were assigned to training
    training_history.json Per-epoch train/val loss and accuracy
    training_log.txt      Full stdout log
"""

import argparse
import json
import os
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
NUM_EPOCHS   = 30
TRAIN_RATIO  = 0.75
SEED         = 42
GRAY_METHOD  = "lab_l"
POOL_FACTOR  = 10


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

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
    """Run one training epoch.

    Returns:
        (avg_loss, accuracy) over all batches in the training set.
    """
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
    """Compute accuracy and average loss on a DataLoader without gradients.

    Returns:
        (accuracy, avg_loss)
    """
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

def main(preload: bool = False, db_path: str = DB_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_path = os.path.join(MODEL_DIR, "training_log.txt")
    print(f"Training log → {log_path}")

    with _Tee(log_path):
        _main(preload=preload, db_path=db_path)


def _main(preload: bool = False, db_path: str = DB_PATH):
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    # --- Data ---
    print("Loading data and performing train/validation split …")
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
    )

    print(f"\nClass names:  {meta['class_names']}")
    print(f"Train images (with 4× augmentation): {meta['train_size']}")
    print(f"Validation images:  {meta['val_size']}")
    print("Train class counts (base):", meta["train_class_counts"])
    print("Validation class counts:  ", meta["val_class_counts"])
    print(f"\nTrain record saved to: {RECORD_PATH}")

    # --- Model ---
    model = FibrinCNN(num_classes=len(CLASS_NAMES)).to(device)
    print("\n" + model.summary())

    # --- Loss: inverse-frequency class weights ---
    class_weights = compute_class_weights(meta["train_class_counts"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # --- Training loop ---
    best_acc = 0.0
    history = {"train_loss": [], "train_accuracy": [],
               "val_loss":   [], "val_accuracy":   []}

    print(f"\nTraining for {NUM_EPOCHS} epochs on {device} …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time (s)':>10}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            tag = "  ← best"
        else:
            tag = ""

        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>10.4f}  "
              f"{val_loss:>10.4f}  {val_acc:>10.4f}  {elapsed:>10.1f}{tag}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")

    # --- Save training history ---
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # --- Detailed evaluation of best model ---
    print("\nDetailed evaluation of best model on validation set:")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                     weights_only=True))
    results = evaluate_model(model, val_loader, device, meta["class_names"])
    print(results["report_str"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preload", action="store_true",
                        help="Preload all images into RAM before training.")
    parser.add_argument("--db", default=DB_PATH, metavar="PATH",
                        help="Path to SQLite database (default: data/endpoint10.db).")
    args = parser.parse_args()
    main(preload=args.preload, db_path=args.db)
