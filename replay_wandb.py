#!/usr/bin/env python3
"""
replay_wandb.py — Replay training_log_full.csv history into new W&B runs,
then patch checkpoints/latest.pth so Slurm jobs resume into those runs.

Usage:
    python replay_wandb.py                        # all known models
    python replay_wandb.py --model 5class_hpc_v0  # single model
    python replay_wandb.py --project my-project   # override project name
"""

import argparse
import csv
import os
from datetime import datetime, timezone

import torch
import wandb

MODELS = {
    "5class_hpc_v0": "models/5class_hpc_v0",
    "patch_v0":      "models/patch_v0",
}
WANDB_PROJECT = "fibrin-cnn"
SKIP_KEYS = {"epoch", "model_type", "wall_clock_time"}


def replay(model_type: str, model_dir: str, project: str) -> None:
    csv_path = os.path.join(model_dir, "training_log_full.csv")
    if not os.path.exists(csv_path):
        print(f"[{model_type}] No training_log_full.csv — skipping.")
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[{model_type}] training_log_full.csv is empty — skipping.")
        return

    print(f"[{model_type}] Replaying {len(rows)} epochs into wandb ...")

    run = wandb.init(
        project=project,
        name=model_type,
        config={"model_type": model_type},
        reinit=True,
    )

    for row in rows:
        epoch = int(row["epoch"])
        metrics = {}
        for k, v in row.items():
            if k in SKIP_KEYS:
                continue
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                pass
        if "wall_clock_time" in row:
            try:
                dt = datetime.fromisoformat(row["wall_clock_time"])
                metrics["wall_clock_unix"] = dt.replace(
                    tzinfo=timezone.utc).timestamp()
            except (ValueError, TypeError):
                pass
        wandb.log(metrics, step=epoch)

    run_id = run.id
    run_url = run.url
    wandb.finish()
    print(f"  W&B run: {run_url}  (id={run_id})")

    latest_path = os.path.join(model_dir, "checkpoints", "latest.pth")
    if not os.path.exists(latest_path):
        print(f"  [warn] No checkpoint at {latest_path} — run_id not persisted.")
        print(f"         Add wandb_run_id='{run_id}' to the checkpoint manually.")
        return

    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    ckpt["wandb_run_id"] = run_id
    torch.save(ckpt, latest_path)
    print(f"  Patched {latest_path} with wandb_run_id={run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Replay a single model (default: all)")
    parser.add_argument("--project", default=WANDB_PROJECT,
                        help="W&B project name")
    args = parser.parse_args()

    targets = {args.model: MODELS[args.model]} if args.model else MODELS
    for model_type, model_dir in targets.items():
        replay(model_type, model_dir, args.project)


if __name__ == "__main__":
    main()
```