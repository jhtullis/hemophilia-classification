"""
train.py — Unified training entry point for FibrinCNN models.

Usage:
    python train.py --model-type 5class             # 5-class model
    python train.py --model-type 3class_scratch     # 3-class from random init
    python train.py --model-type 3class_finetune    # 3-class fine-tuned from 5-class
    python train.py --model-type 5class --preload   # preload all images into RAM

The --preload flag caches preprocessed tensors in RAM before training begins.
For the original 10× min-pool pipeline (~700 MB for ~850 images) this
eliminates per-epoch disk I/O and significantly speeds up training on CPU.

Outputs (written to models/<type>/):
    best_model.pth        Weights of the epoch with the highest validation accuracy
    train_record.json     Canonical train/validation experiment-level split
    training_history.json Per-epoch train_loss, train_accuracy, val_loss, val_accuracy
    training_log.txt      Full stdout log
"""

import argparse
import sys


def main():
    p = argparse.ArgumentParser(
        description="Train a FibrinCNN model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-type",
        required=True,
        choices=["5class", "3class_scratch", "3class_finetune"],
        help="Which model to train.",
    )
    p.add_argument(
        "--preload",
        action="store_true",
        help="Preload all preprocessed images into RAM before training "
             "(eliminates disk I/O; ~700 MB for original pipeline).",
    )
    args = p.parse_args()

    if args.model_type == "5class":
        from train_5class import main as _train
        _train(preload=args.preload)
    elif args.model_type == "3class_scratch":
        from train_3class import main as _train
        _train(mode="scratch", preload=args.preload)
    elif args.model_type == "3class_finetune":
        from train_3class import main as _train
        _train(mode="finetune", preload=args.preload)
    else:
        print(f"Unknown model-type: {args.model_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
