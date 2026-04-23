"""
evaluate.py — Detailed model evaluation with per-class metrics.

Can be imported (use evaluate_model()) or run as a script:
    python evaluate.py
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import CLASS_NAMES, FibrinDataset, load_split_from_record
from model import FibrinCNN
from preprocessing import make_preprocessor

DB_PATH     = os.path.join(os.path.dirname(__file__), "data", "endpoint10.db")
PHOTO_DIR   = os.path.join(os.path.dirname(__file__), "data", "photos")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "5class", "best_model.pth")
RECORD_PATH = os.path.join(os.path.dirname(__file__), "models", "5class", "train_record.json")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, num_classes: int = 5,
               device: torch.device = torch.device("cpu")) -> FibrinCNN:
    """Load a saved FibrinCNN checkpoint."""
    model = FibrinCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_all(model: FibrinCNN, loader: DataLoader,
                device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over an entire DataLoader.

    Returns:
        (predictions, true_labels) — integer arrays of length N.
    """
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Compute N×N confusion matrix (rows = true, cols = predicted)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1
    return cm


def classification_report(preds: np.ndarray, labels: np.ndarray,
                           class_names: List[str]) -> str:
    """Per-class precision, recall, F1, and support; overall accuracy.

    Args:
        preds:       Predicted class integers.
        labels:      True class integers.
        class_names: Ordered list of class name strings.

    Returns:
        Formatted string report.
    """
    n = len(class_names)
    cm = confusion_matrix(preds, labels, n)
    overall_acc = np.diag(cm).sum() / cm.sum()

    lines = [
        "",
        f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 55,
    ]
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        lines.append(
            f"{name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10d}"
        )

    lines += [
        "-" * 55,
        f"{'Overall accuracy':>34}  {overall_acc:>10.3f}",
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        f"{'':12}" + "".join(f"{n:>8}" for n in class_names),
    ]
    for i, name in enumerate(class_names):
        lines.append(f"{name:<12}" + "".join(f"{cm[i, j]:>8d}" for j in range(n)))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_model(model: FibrinCNN, loader: DataLoader,
                   device: torch.device,
                   class_names: List[str] = CLASS_NAMES) -> dict:
    """Run full evaluation and return a metrics dictionary.

    Returns:
        dict with keys: accuracy, preds, labels, confusion_matrix, report_str
    """
    preds, labels = predict_all(model, loader, device)
    cm = confusion_matrix(preds, labels, len(class_names))
    acc = np.diag(cm).sum() / cm.sum()
    report = classification_report(preds, labels, class_names)
    return {
        "accuracy": float(acc),
        "preds": preds,
        "labels": labels,
        "confusion_matrix": cm,
        "report_str": report,
    }


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the 5-class FibrinCNN.")
    parser.add_argument("--db", default=DB_PATH, metavar="PATH",
                        help="Path to SQLite database (default: data/endpoint10.db).")
    args = parser.parse_args()

    device = torch.device("cpu")

    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}. Run train_5class.py first.")
        return

    print("Loading model …")
    model = load_model(MODEL_PATH, device=device)
    print(model.summary())

    print("\nLoading validation data from saved split …")
    _, val_df = load_split_from_record(RECORD_PATH, args.db)
    preprocessor = make_preprocessor(gray_method="lab_l", pool_factor=10)
    val_ds = FibrinDataset(val_df, PHOTO_DIR, preprocessor, augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=False)

    print("Running inference on validation set …")
    results = evaluate_model(model, val_loader, device, CLASS_NAMES)

    print(f"\nValidation accuracy: {results['accuracy']:.4f}")
    print(results["report_str"])


if __name__ == "__main__":
    main()
