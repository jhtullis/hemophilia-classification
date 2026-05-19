"""
checkpoint_manager.py — Centralised checkpoint I/O for all FibrinCNN training scripts.

Storage note: each .pth for FibrinCNN (~455K params) is ~2–5 MB.
Total across 10,000 epochs: ~30 periodic files ≈ 150 MB per model directory.

Public API:
    should_save_periodic(epoch, max_epochs) -> bool
    save_checkpoint(state, model_dir, epoch, is_best, max_epochs)
    load_latest_checkpoint(model_dir) -> dict | None
    log_epoch(model_dir, row_dict)
"""

import csv
import os

import torch


def should_save_periodic(epoch: int, max_epochs: int) -> bool:
    """Return True if a periodic snapshot should be saved this epoch.

    Cadence:
      epochs   1–300  : every 20 epochs
      epochs 301–10000: every 100 epochs
    Always True at the final target epoch.
    """
    if epoch == max_epochs:
        return True
    if epoch <= 300:
        return epoch % 20 == 0
    return epoch % 100 == 0


def save_checkpoint(
    state: dict,
    model_dir: str,
    epoch: int,
    is_best: bool,
    max_epochs: int,
) -> None:
    """Save training state to disk.

    Always overwrites checkpoints/latest.pth (used for resume).
    Saves a periodic snapshot when should_save_periodic() is True.
    Saves model_dir/best_model.pth when is_best is True.

    Expected keys in state:
        epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
        best_acc, history, model_type, phase, config
    """
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_path = os.path.join(ckpt_dir, "latest.pth")
    torch.save(state, latest_path)

    if should_save_periodic(epoch, max_epochs):
        periodic_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
        torch.save(state, periodic_path)
        print(f"  [ckpt] periodic snapshot → {periodic_path}")

    if is_best:
        best_path = os.path.join(model_dir, "best_model.pth")
        torch.save(state["model_state_dict"], best_path)


def load_latest_checkpoint(model_dir: str) -> dict | None:
    """Load checkpoints/latest.pth if it exists.

    Returns the checkpoint dict or None if no checkpoint is found.
    Prints a status line in either case.
    """
    latest_path = os.path.join(model_dir, "checkpoints", "latest.pth")
    if not os.path.exists(latest_path):
        print("Starting from scratch.")
        return None
    ckpt = torch.load(latest_path, weights_only=False)
    print(f"Resuming from epoch {ckpt['epoch']}.")
    return ckpt


def log_epoch(model_dir: str, row_dict: dict) -> None:
    """Append one row to training_log_full.csv.

    Creates the file with a header on first call; appends on subsequent calls.
    Uses csv.DictWriter so column order matches the first row's keys.
    """
    csv_path = os.path.join(model_dir, "training_log_full.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
