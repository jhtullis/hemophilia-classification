"""
train.py — Training loop for FibrinCNN.

Run with:
    python train.py

Outputs:
    best_model.pth     — Weights of the epoch with the highest test accuracy
    train_record.json  — Record of which images were assigned to training
"""

import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import CLASS_NAMES, create_dataloaders
from evaluate import evaluate_model
from model import FibrinCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH   = os.path.join(os.path.dirname(__file__), "data", "test_db.db")
PHOTO_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
RECORD_PATH = os.path.join(os.path.dirname(__file__), "train_record.json")

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
NUM_WORKERS  = 4


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
                    device: torch.device) -> float:
    """Run one training epoch.

    Returns:
        Average cross-entropy loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def quick_accuracy(model: FibrinCNN, loader: DataLoader,
                   device: torch.device) -> float:
    """Compute accuracy without storing all predictions (used during training)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    # --- Data ---
    print("Loading data and performing train/test split …")
    train_loader, test_loader, meta = create_dataloaders(
        db_path=DB_PATH,
        photo_dir=PHOTO_DIR,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        gray_method=GRAY_METHOD,
        pool_factor=POOL_FACTOR,
        num_workers=NUM_WORKERS,
        train_record_path=RECORD_PATH,
    )

    print(f"\nClass names:  {meta['class_names']}")
    print(f"Train images (with 4× augmentation): {meta['train_size']}")
    print(f"Test images:  {meta['test_size']}")
    print("Train class counts (base):", meta["train_class_counts"])
    print("Test class counts:        ", meta["test_class_counts"])
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
    print(f"\nTraining for {NUM_EPOCHS} epochs on {device} …\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Acc':>10}  {'Time (s)':>10}")
    print("-" * 45)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        test_acc = quick_accuracy(model, test_loader, device)
        elapsed = time.time() - t0

        scheduler.step(test_acc)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            tag = "  ← best"
        else:
            tag = ""

        print(f"{epoch:>6}  {train_loss:>12.4f}  {test_acc:>10.4f}  "
              f"{elapsed:>10.1f}{tag}")

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")

    # --- Detailed evaluation of best model ---
    print("\nDetailed evaluation of best model on test set:")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    results = evaluate_model(model, test_loader, device, meta["class_names"])
    print(results["report_str"])


if __name__ == "__main__":
    main()
