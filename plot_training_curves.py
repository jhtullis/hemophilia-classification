"""
plot_training_curves.py — Plot train/validation accuracy and loss curves.

Reads training_history.json written by the training scripts and saves a
two-subplot figure to the same model directory.

Usage:
    python plot_training_curves.py [--model-type TYPE] [--output PATH]

    TYPE: 5class | 3class_scratch | 3class_finetune  (default: 5class)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

_MODEL_DIRS = {
    "5class":          os.path.join("models", "5class"),
    "3class_scratch":  os.path.join("models", "3class_hemo"),
    "3class_finetune": os.path.join("models", "3class_hemo_finetune"),
}


def plot_curves(history_path: str, output_path: str = None) -> str:
    """Read training_history.json and save a training curves PNG.

    Args:
        history_path: Path to training_history.json.
        output_path:  Override output PNG path. Defaults to
                      <history_path parent>/training_curves.png.

    Returns:
        Path to the saved PNG.
    """
    history_path = Path(history_path)
    with open(history_path) as f:
        h = json.load(f)

    n = len(h["train_loss"])
    epochs = range(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ── Accuracy subplot ──────────────────────────────────────────────────────
    ax1.plot(epochs, h["train_accuracy"], label="Train", color="#4878CF", linewidth=1.8)
    ax1.plot(epochs, h["val_accuracy"],   label="Validation", color="#D65F5F",
             linewidth=1.8, linestyle="--")
    best_epoch = int(max(range(n), key=lambda i: h["val_accuracy"][i])) + 1
    ax1.axvline(best_epoch, color="#888", linestyle=":", linewidth=1,
                label=f"Best val epoch {best_epoch}")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right")
    ax1.set_title("Train / Validation Accuracy")
    ax1.grid(True, alpha=0.3)

    # ── Loss subplot ──────────────────────────────────────────────────────────
    ax2.plot(epochs, h["train_loss"], label="Train", color="#4878CF", linewidth=1.8)
    ax2.plot(epochs, h["val_loss"],   label="Validation", color="#D65F5F",
             linewidth=1.8, linestyle="--")
    ax2.axvline(best_epoch, color="#888", linestyle=":", linewidth=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.set_title("Train / Validation Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out = output_path or str(history_path.parent / "training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Plot training curves from history JSON.")
    p.add_argument("--model-type", default="5class",
                   choices=list(_MODEL_DIRS.keys()),
                   help="Model to plot (default: 5class).")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Override output PNG path.")
    args = p.parse_args()

    history_path = os.path.join(_MODEL_DIRS[args.model_type], "training_history.json")
    if not os.path.exists(history_path):
        print(f"No training history found at {history_path}. "
              f"Train the model first.")
        return

    plot_curves(history_path, args.output)


if __name__ == "__main__":
    main()
