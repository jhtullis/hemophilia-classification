"""
train_abmil.py — Training loop for FibrinABMIL, an attention-based MIL
aggregator built on top of an already-trained, frozen patch backbone.

Standalone script (not wired into train.py / configs/training_configs.py):
ABMIL trains on top of an already-trained backbone with a structurally
different bag-based loop (batch-size-1 bag + gradient accumulation, frozen
pretrained backbone loading, no from-scratch train/val split creation) rather
than being one more from-scratch config variant.

Usage:
    python train_abmil.py --backbone-model-type mpatch_v0_f \
        --max-epochs 500 --epochs-per-job 500 --resume --preload

    # Laptop smoke test (CPU, few epochs, small bags):
    python train_abmil.py --backbone-model-type mpatch_v0_f \
        --max-epochs 5 --epochs-per-job 5 --patches-per-image 20 \
        --accumulation-steps 4 --no-wandb --preload

Exit codes (consumed by slurm/train_abmil_mpatch_v0_f.sh):
    0   — epochs_per_job exhausted; more epochs remain (Slurm: resubmit)
    100 — max_epochs reached; training complete (Slurm: do not resubmit)

Outputs to models/abmil_<backbone-model-type>/:
    best_model.pth           Full FibrinABMIL state_dict (backbone+attention+classifier)
    abmil_config.json        Run metadata (backbone source, sampling params)
    checkpoints/             Periodic + latest checkpoint .pth files
    training_log_full.csv    Per-epoch metrics
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import kornia.augmentation as K
import torch
from torch.utils.data import DataLoader

from abmil_dataset import FibrinImageBagDataset
from abmil_model import FibrinABMIL, make_abmil_model
from augmentation_patch import PatchAugmentation
from checkpoint_manager import (
    finish_wandb,
    get_wandb_run_id,
    init_wandb,
    load_latest_checkpoint,
    log_epoch,
    save_checkpoint,
)
from lr_schedulers import CosineAnnealingWarmRestartsF
from model_patch import FibrinPatchCNN
from patch_dataset import PATCH_SIZE, load_split_record
from preprocessing import make_preprocessor
from train_patch import _compute_class_weights, cosine_loss

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_frozen_backbone(backbone_model_type: str, device: torch.device):
    """Resolve backbone weights/config/split via the existing registry, and
    load best_model.pth defensively (raw state_dict or wrapped; strips a
    possible torch.compile `_orig_mod.` prefix)."""
    from analysis_utils import MODEL_REGISTRY
    from configs.training_configs import CONFIGS

    if backbone_model_type not in MODEL_REGISTRY:
        raise KeyError(f"Unknown backbone_model_type '{backbone_model_type}'. "
                        f"Valid: {list(MODEL_REGISTRY)}")
    if backbone_model_type not in CONFIGS:
        raise KeyError(f"No training config found for '{backbone_model_type}' "
                        f"in configs/training_configs.py.")

    model_dir, num_classes, class_map = MODEL_REGISTRY[backbone_model_type]
    cfg = CONFIGS[backbone_model_type]
    head_type = cfg.get("head_type", "cosine")

    weights_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No trained backbone at {weights_path}. "
                                 f"Train {backbone_model_type} first.")

    backbone = FibrinPatchCNN(num_classes=num_classes, head_type=head_type)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    sd = state["model_state_dict"] if (isinstance(state, dict) and "model_state_dict" in state) else state
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    backbone.load_state_dict(sd)
    backbone.eval()

    return backbone, cfg, model_dir, num_classes, class_map, head_type


# ---------------------------------------------------------------------------
# Collate (module-level, not a lambda, so it pickles cleanly for DataLoader workers)
# ---------------------------------------------------------------------------

def _bag_collate(batch):
    return batch[0]


# ---------------------------------------------------------------------------
# Attention entropy diagnostic
# ---------------------------------------------------------------------------

def _normalized_entropy(attn: torch.Tensor) -> Optional[float]:
    """-(sum(A*logA)) / log(N). None if N<=1 (entropy undefined/degenerate)."""
    n = attn.shape[0]
    if n <= 1:
        return None
    a = attn.clamp_min(1e-12)
    ent = -(a * a.log()).sum().item()
    return ent / math.log(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone, backbone_cfg, backbone_model_dir, num_classes, class_map, backbone_head_type = \
        load_frozen_backbone(args.backbone_model_type, device)

    head_type = backbone_head_type if args.head_type == "auto" else args.head_type
    mask_aware = args.backbone_model_type.startswith("mpatch_") if args.mask_aware == "auto" \
        else (args.mask_aware == "on")

    model_dir = args.model_dir or os.path.join(_DIR, "models", f"abmil_{args.backbone_model_type}")
    os.makedirs(model_dir, exist_ok=True)

    db_path = args.db or os.path.join(_DIR, "data", "endpoint10.db")
    photo_dir = args.photo_dir or os.path.join(_DIR, "data", "photos")

    avail_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", None) or os.cpu_count() or 4)
    torch.set_num_threads(avail_cpus)
    # With preload=True all bag data already lives in RAM, so DataLoader worker
    # processes would only add fork/fd overhead for cheap in-memory indexing —
    # skip multiprocessing entirely in that case. Only spin up workers when
    # __getitem__ still has to hit disk (preload=False).
    num_workers = 0 if args.preload else min(4, max(0, avail_cpus - 1))

    from gpu_utils import get_gpu_config
    gpu_cfg = get_gpu_config(device, default_batch_size=1)
    print(f"AMP: {gpu_cfg['amp_enabled']}  dtype: {gpu_cfg['amp_dtype']}")

    # ── Split (reused verbatim from the backbone's own model dir) ──────────
    train_df, val_df, _test_df = load_split_record(backbone_model_dir, db_path)
    preprocessor = make_preprocessor()

    mask_kwargs = {}
    if mask_aware:
        mask_kwargs = dict(
            mask_dir=os.path.join(_DIR, backbone_cfg["mask_dir"]),
            mask_version=backbone_cfg["mask_version"],
            patch_center_version=backbone_cfg["patch_center_version"],
        )

    train_ds = FibrinImageBagDataset(
        train_df, photo_dir, preprocessor,
        patches_per_image=args.patches_per_image,
        mask_aware=mask_aware,
        resample_each_epoch=True,
        seed=args.seed,
        preload=args.preload,
        **mask_kwargs,
    )
    val_ds = FibrinImageBagDataset(
        val_df, photo_dir, preprocessor,
        patches_per_image=args.val_patches_per_image,
        mask_aware=mask_aware,
        resample_each_epoch=False,
        seed=args.seed,
        preload=args.preload,
        **mask_kwargs,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=_bag_collate,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=_bag_collate,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────
    freeze_backbone = not args.finetune_backbone
    model = make_abmil_model(
        backbone=backbone, num_classes=num_classes, head_type=head_type,
        freeze_backbone=freeze_backbone, attn_hidden_dim=args.attn_hidden_dim,
    ).to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=gpu_cfg["use_scaler"])
    augmentation = PatchAugmentation(patch_size=PATCH_SIZE).to(device)
    center_crop = K.CenterCrop(PATCH_SIZE)

    class_weights = _compute_class_weights(train_df, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights) if head_type == "ce" else None

    if args.finetune_backbone:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
            {"params": list(model.attention.parameters()) + list(model.classifier.parameters()),
             "lr": args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )
    scheduler = CosineAnnealingWarmRestartsF(
        optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min
    )

    # ── abmil_config.json (run metadata; not a split record) ───────────────
    abmil_config = {
        "backbone_model_type": args.backbone_model_type,
        "backbone_checkpoint": os.path.join(backbone_model_dir, "best_model.pth"),
        "head_type": head_type,
        "mask_aware": mask_aware,
        "patches_per_image": args.patches_per_image,
        "val_patches_per_image": args.val_patches_per_image,
        "attn_hidden_dim": args.attn_hidden_dim,
        "freeze_backbone": freeze_backbone,
        "seed": args.seed,
    }
    with open(os.path.join(model_dir, "abmil_config.json"), "w") as f:
        json.dump(abmil_config, f, indent=2)

    # ── Checkpoint resume ────────────────────────────────────────────────
    ckpt = load_latest_checkpoint(model_dir, map_location=device) if args.resume else None
    if ckpt is not None:
        sd = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", [])
        wandb_run_id = ckpt.get("wandb_run_id")
    else:
        start_epoch = 0
        best_val_acc = 0.0
        history = []
        wandb_run_id = None

    # ── Wandb ────────────────────────────────────────────────────────────
    model_type_str = f"abmil_{args.backbone_model_type}"
    wandb_config = {
        "model_type": model_type_str,
        "backbone_model_type": args.backbone_model_type,
        "head_type": head_type,
        "mask_aware": mask_aware,
        "patches_per_image": args.patches_per_image,
        "val_patches_per_image": args.val_patches_per_image,
        "attn_hidden_dim": args.attn_hidden_dim,
        "freeze_backbone": freeze_backbone,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "accumulation_steps": args.accumulation_steps,
        "grad_clip": args.grad_clip,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_T0": args.t0,
        "scheduler_Tmult": args.t_mult,
        "scheduler_eta_min": args.eta_min,
        "loss": "CrossEntropyLoss" if head_type == "ce" else "CosineLoss",
        "num_classes": num_classes,
        "max_epochs": args.max_epochs,
        "epochs_per_job": args.epochs_per_job,
        "amp_enabled": gpu_cfg["amp_enabled"],
        "amp_dtype": str(gpu_cfg["amp_dtype"]),
        "n_train_images": len(train_ds),
        "n_val_images": len(val_ds),
    }
    init_wandb(
        config=wandb_config, model_type=model_type_str, project=args.wandb_project,
        run_name=args.wandb_run_name or model_type_str, run_id=wandb_run_id,
        resume_run=(args.resume and wandb_run_id is not None),
        enabled=not args.no_wandb, entity=None,
    )

    # ── Training loop ───────────────────────────────────────────────────
    if start_epoch >= args.max_epochs:
        print(f"Already reached max_epochs={args.max_epochs}. Training complete.")
        finish_wandb()
        sys.exit(100)

    job_epochs_done = 0

    for epoch in range(start_epoch, args.max_epochs):
        # ── Train pass ──────────────────────────────────────────────────
        model.train()
        augmentation.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        train_entropies = []

        n_batches = len(train_loader)
        optimizer.zero_grad()
        for i, (patches, label, _img_idx) in enumerate(train_loader):
            patches = patches.to(device, non_blocking=True)
            label_t = torch.tensor([label], device=device)

            patches = augmentation(patches)   # (N,1,283,283) -> (N,1,200,200)

            with torch.autocast(device_type="cuda", dtype=gpu_cfg["amp_dtype"],
                                 enabled=gpu_cfg["amp_enabled"]):
                out, attn = model(patches)
                loss_raw = (criterion(out.unsqueeze(0), label_t) if head_type == "ce"
                            else cosine_loss(out.unsqueeze(0), label_t, class_weights))
                loss = loss_raw / args.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == n_batches:
                if args.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_sum += loss_raw.item()
            train_correct += int(out.argmax().item() == label)
            train_total += 1
            ent = _normalized_entropy(attn.detach())
            if ent is not None:
                train_entropies.append(ent)

        scheduler.step(epoch)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        train_entropy = sum(train_entropies) / len(train_entropies) if train_entropies else float("nan")

        # ── Validation pass ─────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        val_entropies = []

        with torch.no_grad():
            for patches, label, _img_idx in val_loader:
                patches = patches.to(device, non_blocking=True)
                label_t = torch.tensor([label], device=device)
                patches = center_crop(patches)   # 283->200, deterministic

                out, attn = model(patches)
                loss_raw = (criterion(out.unsqueeze(0), label_t) if head_type == "ce"
                            else cosine_loss(out.unsqueeze(0), label_t, class_weights))

                val_loss_sum += loss_raw.item()
                val_correct += int(out.argmax().item() == label)
                val_total += 1
                ent = _normalized_entropy(attn)
                if ent is not None:
                    val_entropies.append(ent)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_entropy = sum(val_entropies) / len(val_entropies) if val_entropies else float("nan")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        if val_entropy == val_entropy and val_entropy < 0.15 and epoch < 50:
            print(f"  WARNING: val attention entropy (normalized) is low "
                  f"({val_entropy:.3f}) at epoch {epoch} — attention may be "
                  f"collapsing onto a single patch.")

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "train_attn_entropy_norm": round(train_entropy, 6) if train_entropy == train_entropy else None,
            "val_attn_entropy_norm": round(val_entropy, 6) if val_entropy == val_entropy else None,
            "lr": lr_now,
            "model_type": model_type_str,
            "wall_clock_time": datetime.now(timezone.utc).isoformat(),
        }
        print(
            f"Epoch {epoch:5d} | train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"attn_ent(tr={train_entropy:.3f} va={val_entropy:.3f}) | lr={lr_now:.2e}"
            + (" <- best" if is_best else "")
        )

        log_epoch(model_dir, row)
        history.append({k: v for k, v in row.items() if k != "wall_clock_time"})

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_val_acc,
            "history": history,
            "model_type": model_type_str,
            "phase": "training",
            "config": wandb_config,
            "wandb_run_id": get_wandb_run_id(),
        }
        save_checkpoint(state, model_dir, epoch, is_best, args.max_epochs)

        job_epochs_done += 1
        if job_epochs_done >= args.epochs_per_job:
            print(f"epochs_per_job={args.epochs_per_job} exhausted at epoch {epoch}. Exiting.")
            finish_wandb(complete=False)
            sys.exit(0)

    print(f"Training complete at epoch {args.max_epochs - 1}.")
    finish_wandb()
    sys.exit(100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a FibrinABMIL attention aggregator on top of a "
                     "frozen, pretrained patch backbone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backbone-model-type", required=True,
                    help="Registry key of the already-trained patch backbone "
                         "(e.g. mpatch_v0_f). Must exist in analysis_utils.MODEL_REGISTRY "
                         "and configs.training_configs.CONFIGS.")
    p.add_argument("--model-dir", default=None,
                    help="Output dir (default: models/abmil_<backbone-model-type>).")
    p.add_argument("--db", default=None, help="Path to SQLite database.")
    p.add_argument("--photo-dir", default=None, help="Directory of JPEG images.")
    p.add_argument("--max-epochs", type=int, default=500,
                    help="Total training epochs across all jobs (default: 500).")
    p.add_argument("--epochs-per-job", type=int, default=500,
                    help="Epochs to run before exiting with code 0 (default: 500).")
    p.add_argument("--patches-per-image", type=int, default=50,
                    help="Training bag size (default: 50).")
    p.add_argument("--val-patches-per-image", type=int, default=50,
                    help="Validation bag size, sampled once and fixed (default: 50).")
    p.add_argument("--mask-aware", choices=["auto", "on", "off"], default="auto",
                    help="Sample bag patches from mask-valid centers only. "
                         "'auto' = on for mpatch_* backbones, off otherwise.")
    p.add_argument("--head-type", choices=["auto", "cosine", "ce"], default="auto",
                    help="ABMIL classifier head type. 'auto' = backbone's own head_type.")
    p.add_argument("--attn-hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--t0", type=int, default=50, help="Scheduler T_0 (default: 50).")
    p.add_argument("--t-mult", type=float, default=1.5, help="Scheduler T_mult (default: 1.5).")
    p.add_argument("--eta-min", type=float, default=1e-4, help="Scheduler eta_min (default: 1e-4).")
    p.add_argument("--accumulation-steps", type=int, default=8,
                    help="Bags accumulated before an optimizer step (default: 8).")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--finetune-backbone", action="store_true", default=False,
                    help="Unfreeze the backbone (off by default: goal is augmenting "
                         "the trained backbone, not retraining it).")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="LR multiplier for backbone params when --finetune-backbone is set.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="fibrin-cnn")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--preload", action="store_true")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
