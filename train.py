"""
train.py — Unified training entry point for FibrinCNN models.

Usage:
    python train.py --model-type 5class             # 5-class model
    python train.py --model-type 5class_hpc_v0      # 5-class cosine model (HPC)
    python train.py --model-type 3class_scratch     # 3-class from random init
    python train.py --model-type 3class_finetune    # 3-class fine-tuned from 5-class
    python train.py --model-type 5class --preload   # preload all images into RAM

HPC / Slurm usage (self-resubmitting jobs):
    python train.py --model-type 5class --max-epochs 10000 --epochs-per-job 200 --resume

Exit codes:
    0   — epochs-per-job budget exhausted; more epochs remain (Slurm: resubmit)
    100 — max-epochs reached; training complete (Slurm: do not resubmit)

Outputs (written to models/<type>/):
    best_model.pth          Weights of the epoch with the highest validation accuracy
    train_record.json       Canonical train/validation experiment-level split
    training_history.json   Per-epoch train_loss, train_accuracy, val_loss, val_accuracy
    training_log.txt        Full stdout log
    training_log_full.csv   Extended per-epoch CSV (all models; cosine-native cols for hpc)
    checkpoints/            Periodic and latest checkpoint .pth files
"""

import argparse
import os
import sys

import torch


def get_device() -> torch.device:
    """Return cuda if available, else cpu. Prints the chosen device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU — using CPU")
    return device


def main():
    p = argparse.ArgumentParser(
        description="Train a FibrinCNN model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-type",
        required=True,
        choices=[
            "5class", "5class_hpc_baseline", "5class_hpc_v0",
            "5class_hpc_v1a", "5class_hpc_v1b", "5class_hpc_v1c",
            "3class_scratch", "3class_finetune",
            "patch_v0", "patch_v1a", "patch_v1b", "patch_v1c",
            "patch_v2a", "patch_v2b", "patch_v2c",
            "mpatch_v0_a", "mpatch_v0_b", "mpatch_v0_c", "mpatch_v0_d",
            "mpatch_v0_e", "mpatch_v0_f", "mpatch_v0_g", "mpatch_v0_h",
            "mpatch_v0_i",
            "mpatch_v0_f1a",
            "mpatch_v1_full_a",
        ],
        help="Which model to train.",
    )
    p.add_argument(
        "--preload",
        action="store_true",
        help="Preload all preprocessed images into RAM before training.",
    )
    p.add_argument(
        "--db", default=None, metavar="PATH",
        help="Path to SQLite database (default: data/endpoint10.db).",
    )
    p.add_argument(
        "--force-resplit", action="store_true",
        help="Regenerate the train/val split even if train_record.json already exists.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in models/<type>/checkpoints/.",
    )
    p.add_argument(
        "--max-epochs", type=int, default=10000,
        help="Total training epochs across all jobs (default: 10000).",
    )
    p.add_argument(
        "--epochs-per-job", type=int, default=200,
        help="Epochs to run in this Slurm job before exiting with code 0 (default: 200).",
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging.",
    )
    p.add_argument(
        "--wandb-project", default="fibrin-cnn",
        help="W&B project name (default: fibrin-cnn).",
    )
    p.add_argument(
        "--wandb-run-name", default=None,
        help="W&B run display name (default: model-type).",
    )
    args = p.parse_args()

    device = get_device()

    common = {
        "preload":          args.preload,
        "device":           device,
        "resume":           args.resume,
        "max_epochs":       args.max_epochs,
        "epochs_per_job":   args.epochs_per_job,
        "wandb_enabled":    not args.no_wandb,
        "wandb_project":    args.wandb_project,
        "wandb_run_name":   args.wandb_run_name,
    }
    if args.db:
        common["db_path"] = args.db

    if args.model_type == "5class":
        from train_5class import main as _train
        _train(**common, force_resplit=args.force_resplit)
    elif args.model_type == "5class_hpc_baseline":
        from train_5class import main as _train
        _train(**common, force_resplit=args.force_resplit,
               model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "models", "5class_hpc_baseline"))
    elif args.model_type in ("5class_hpc_v0", "5class_hpc_v1a", "5class_hpc_v1b", "5class_hpc_v1c"):
        from train_5class_hpc import main as _train
        _train(**common, config_name=args.model_type)
    elif args.model_type == "3class_scratch":
        from train_3class import main as _train
        _train(mode="scratch", **common)
    elif args.model_type == "3class_finetune":
        from train_3class import main as _train
        _train(mode="finetune", **common)
    elif args.model_type in (
        "patch_v0", "patch_v1a", "patch_v1b", "patch_v1c",
        "patch_v2a", "patch_v2b", "patch_v2c",
        "mpatch_v0_a", "mpatch_v0_b", "mpatch_v0_c", "mpatch_v0_d",
        "mpatch_v0_e", "mpatch_v0_f", "mpatch_v0_g", "mpatch_v0_h",
        "mpatch_v0_i",
        "mpatch_v0_f1a",
        "mpatch_v1_full_a",
    ):
        from train_patch import main as _train
        _train(**common, config_name=args.model_type)
    else:
        print(f"Unknown model-type: {args.model_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
