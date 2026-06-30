"""
checkpoint_manager.py — Centralised checkpoint I/O for all FibrinCNN training scripts.

Storage note: each .pth for FibrinCNN (~455K params) is ~2–5 MB.
Total across 10,000 epochs: ~30 periodic files ≈ 150 MB per model directory.

Public API:
    should_save_periodic(epoch, max_epochs) -> bool
    save_checkpoint(state, model_dir, epoch, is_best, max_epochs)
    load_latest_checkpoint(model_dir) -> dict | None
    log_epoch(model_dir, row_dict)
    init_wandb(config, model_type, project, entity, run_name, run_id, resume_run, enabled)
    finish_wandb()
    get_wandb_run_id() -> str | None
"""

import csv
import os

import torch

# ---------------------------------------------------------------------------
# W&B singleton — set by init_wandb(), consumed by log_epoch()
# ---------------------------------------------------------------------------

_wandb_run = None


def init_wandb(
    config: dict,
    model_type: str,
    project: str = "fibrin-cnn",
    entity: str = None,
    run_name: str = None,
    run_id: str = None,
    resume_run: bool = False,
    enabled: bool = True,
):
    """Initialise a Weights & Biases run.

    Stores the run in a module-level singleton consumed by log_epoch().
    Returns the run object or None if wandb is disabled/unavailable.
    Training continues unaffected if this returns None.

    Args:
        config:      Hyperparameter dict logged to wandb config panel.
        model_type:  Used as the default run name (e.g. "5class_hpc_v0").
        project:     wandb project name.
        entity:      wandb entity (user or team). None uses the default.
        run_name:    Display name override. Defaults to model_type.
        run_id:      Resume an existing run by ID (loaded from checkpoint).
        resume_run:  If True and run_id is set, resume="must"; else "allow".
        enabled:     Set False to skip wandb entirely (--no-wandb flag).
    """
    global _wandb_run
    if not enabled:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping experiment tracking")
        return None

    try:
        init_kwargs = dict(
            project=project,
            config=config,
            name=run_name or model_type,
            resume="allow",
        )
        if entity:
            init_kwargs["entity"] = entity
        if run_id:
            init_kwargs["id"] = run_id

        _wandb_run = wandb.init(**init_kwargs)
        print(f"W&B run: {_wandb_run.url}")
        return _wandb_run
    except Exception as exc:
        print(f"wandb init failed ({exc}) — skipping experiment tracking")
        _wandb_run = None
        return None


def finish_wandb(complete: bool = True) -> None:
    """Flush (and optionally mark finished) the active W&B run.

    Args:
        complete: If True (default), mark the run as finished — use this when
                  max_epochs is reached. If False, call wandb.mark_preempting()
                  which flushes buffered metrics and sets status to "preempted"
                  (not "finished") so the next job can resume the same run.
    """
    global _wandb_run
    if _wandb_run is not None:
        import wandb
        if complete:
            wandb.finish()
            _wandb_run = None
        else:
            # Mark preempted: flushes all pending metrics without closing the run.
            # The next Slurm job resumes via init_wandb(resume_run=True, run_id=...).
            try:
                wandb.mark_preempting()
            except Exception:
                pass


def get_wandb_run_id() -> str | None:
    """Return the active W&B run ID, or None if no run is active."""
    return _wandb_run.id if _wandb_run is not None else None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

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
        best_acc, history, model_type, phase, config, wandb_run_id
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


def load_latest_checkpoint(model_dir: str, map_location=None) -> dict | None:
    """Load checkpoints/latest.pth if it exists.

    Returns the checkpoint dict or None if no checkpoint is found.
    Prints a status line in either case.

    Args:
        map_location: Passed to torch.load. Pass the training device so that
                      CUDA checkpoints load correctly even when CUDA fails to
                      initialise on the node (e.g. Error 802 / is_available=False).
                      Defaults to None (preserves torch.load default behaviour).
    """
    latest_path = os.path.join(model_dir, "checkpoints", "latest.pth")
    if not os.path.exists(latest_path):
        print("Starting from scratch.")
        return None
    ckpt = torch.load(latest_path, weights_only=False, map_location=map_location)
    print(f"Resuming from epoch {ckpt['epoch']}.")
    return ckpt


def log_epoch(model_dir: str, row_dict: dict) -> None:
    """Append one row to training_log_full.csv and forward metrics to W&B.

    Creates the file with a header on first call; appends on subsequent calls.
    Uses csv.DictWriter so column order matches the first row's keys.

    W&B: all numeric columns are forwarded with step=epoch. wall_clock_time
    (ISO string) is converted to wall_clock_unix (Unix timestamp) for
    plottability. Non-numeric metadata keys are skipped.
    """
    csv_path = os.path.join(model_dir, "training_log_full.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)

    if _wandb_run is not None:
        from datetime import datetime, timezone
        skip = {"epoch", "wall_clock_time", "model_type"}
        metrics = {k: v for k, v in row_dict.items()
                   if k not in skip and isinstance(v, (int, float))}
        if "wall_clock_time" in row_dict:
            try:
                dt = datetime.fromisoformat(row_dict["wall_clock_time"])
                metrics["wall_clock_unix"] = dt.replace(
                    tzinfo=timezone.utc).timestamp()
            except (ValueError, TypeError):
                pass
        _wandb_run.log(metrics, step=row_dict["epoch"])
