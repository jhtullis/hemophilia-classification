"""
train_patch.py — Training loop for the patch_v0 patch-based cosine CNN.

Called from train.py:
    python train.py --model-type patch_v0 [options]

Key design choices:
    - CosineLoss (no cross-entropy, no softmax, no temperature)
    - AdamW + CosineAnnealingWarmRestarts(T_0=100, T_mult=2)
    - scheduler.step(global_epoch) for continuity across Slurm job boundaries
    - checkpoint_manager.py for checkpointing + wandb integration
    - wandb_run_id stored in checkpoint → same wandb run across all jobs
    - compute_cosine_metrics() for both train and val every epoch

Exit codes (consumed by slurm/train_patch_v0.sh):
    0   — epochs_per_job exhausted; more epochs remain (Slurm: resubmit)
    100 — max_epochs reached; training complete (Slurm: do not resubmit)
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import kornia.augmentation as K

from augmentation_patch import PatchAugmentation
from lr_schedulers import CosineAnnealingWarmRestartsF
from checkpoint_manager import (
    finish_wandb,
    get_wandb_run_id,
    init_wandb,
    load_latest_checkpoint,
    log_epoch,
    save_checkpoint,
    should_save_periodic,
)
from data_loader import CLASS_MAP, CLASS_NAMES
from model_patch import FibrinPatchCNN, make_patch_model
from patch_dataset import (
    PATCH_SIZE,
    OVERSIZED,
    PATCHES_PER_IMAGE,
    FibrinPatchDataset,
    get_or_create_split,
    make_patch_sampler,
)
from preprocessing import make_preprocessor


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def cosine_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Cosine loss: L = mean(w_y * (1 - s_y)), range [0, 2].

    s_y is the cosine similarity to the true class prototype.
    w_y is the inverse-frequency class weight for that class.
    No softmax; no temperature; gradient flows only from correct-class similarity.
    """
    correct_sims = similarities[torch.arange(len(labels)), labels]
    weights = class_weights[labels]
    return (weights * (1.0 - correct_sims)).mean()


# ---------------------------------------------------------------------------
# Cosine-specific metrics
# ---------------------------------------------------------------------------

def compute_cosine_metrics(
    model: FibrinPatchCNN,
    loader,
    device: torch.device,
    max_samples: int = 500,
) -> dict:
    """Collect embeddings and compute cosine-geometry metrics.

    Collects up to max_samples L2-normalized 128-dim embeddings, then computes:
        mean_max_sim   : mean cosine similarity between each sample and its
                         class prototype (mean of same-class embeddings, normalized)
        intra_class_cos: mean pairwise cosine similarity within same class
        inter_class_cos: mean pairwise cosine similarity across classes
        silhouette     : mean silhouette score using cosine distance (1 - cos_sim)
                         a(i) = mean dist to same-class, b(i) = min mean dist to other class

    Loader must yield (patches, labels) where patches are already 200×200.
    Model is temporarily set to eval mode; restored to its original mode after.
    No sklearn — pure PyTorch/numpy.
    """
    was_training = model.training
    model.eval()

    all_emb: list[torch.Tensor] = []
    all_lbl: list[torch.Tensor] = []

    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            emb = model.get_embeddings(patches)   # (B, 128), L2-normalized
            all_emb.append(emb.cpu())
            all_lbl.append(labels.cpu())
            if sum(e.shape[0] for e in all_emb) >= max_samples:
                break

    emb = torch.cat(all_emb)[:max_samples]   # (N, 128)
    lbl = torch.cat(all_lbl)[:max_samples]   # (N,)

    if was_training:
        model.train()

    N = len(emb)
    if N < 2:
        return {"mean_max_sim": 0.0, "intra_class_cos": 0.0,
                "inter_class_cos": 0.0, "silhouette": 0.0}

    # Cosine similarity matrix — embeddings are L2-normalized so sim = dot product
    sim_matrix = emb @ emb.T   # (N, N), values in [-1, 1]
    sim_matrix = sim_matrix.clamp(-1.0, 1.0)   # guard floating-point overshoot

    unique_classes = lbl.unique().tolist()

    # Class prototypes: mean of embeddings per class, then L2-normalize
    class_prototypes: dict[int, torch.Tensor] = {}
    for c in unique_classes:
        mask = (lbl == c)
        proto = emb[mask].mean(0)
        class_prototypes[c] = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)

    # mean_max_sim: mean cosine similarity of each sample to its class prototype
    mean_max_sim_vals = [
        (emb[i] * class_prototypes[lbl[i].item()]).sum().item()
        for i in range(N)
    ]
    mean_max_sim = float(np.mean(mean_max_sim_vals))

    # Pairwise intra/inter-class cosine similarity
    intra_sims: list[float] = []
    inter_sims: list[float] = []
    for i in range(N):
        for j in range(i + 1, N):
            s = sim_matrix[i, j].item()
            if lbl[i] == lbl[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    intra_class_cos = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_class_cos = float(np.mean(inter_sims)) if inter_sims else 0.0

    # Silhouette score using cosine distance = 1 - cosine_similarity
    dist_matrix = (1.0 - sim_matrix).clamp(min=0.0)   # in [0, 2]; clamp guards fp rounding
    silhouette_vals: list[float] = []
    for i in range(N):
        c_i = lbl[i].item()
        same_mask = (lbl == c_i)
        same_mask[i] = False

        if same_mask.sum() == 0:
            continue   # singleton — skip

        a_i = dist_matrix[i, same_mask].mean().item()

        b_i = float("inf")
        for c2 in unique_classes:
            if c2 == c_i:
                continue
            c2_mask = lbl == c2
            if c2_mask.sum() == 0:
                continue
            b_c2 = dist_matrix[i, c2_mask].mean().item()
            if b_c2 < b_i:
                b_i = b_c2

        if b_i == float("inf"):
            continue

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 1e-10 else 0.0
        silhouette_vals.append(s_i)

    silhouette = float(np.mean(silhouette_vals)) if silhouette_vals else 0.0

    return {
        "mean_max_sim": mean_max_sim,
        "intra_class_cos": intra_class_cos,
        "inter_class_cos": inter_class_cos,
        "silhouette": silhouette,
    }


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def _compute_class_weights(train_df, device: torch.device) -> torch.Tensor:
    class_counts = train_df["Exp_Type"].value_counts().to_dict()
    total = len(train_df)
    weights = torch.tensor(
        [total / class_counts[cls] for cls in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.sum() * len(CLASS_NAMES)   # normalise


def _accuracy(sims: torch.Tensor, labels: torch.Tensor) -> float:
    return (sims.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    preload: bool = False,
    device: torch.device = None,
    resume: bool = False,
    max_epochs: int = 10000,
    epochs_per_job: int = 100,
    wandb_enabled: bool = True,
    wandb_project: str = "fibrin-cnn",
    wandb_run_name: Optional[str] = None,
    db_path: Optional[str] = None,
    photo_dir: str = "data/photos",
    num_classes: int = 5,
    config_name: str = "patch_v0",
) -> None:
    from configs.training_configs import CONFIGS
    cfg          = CONFIGS[config_name]
    model_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                cfg["model_dir"])
    lr             = cfg["lr"]
    weight_decay   = cfg["weight_decay"]
    T_MULT         = cfg.get("T_mult", 2.0)
    ETA_MIN        = cfg.get("eta_min", 1e-6)
    DROPOUT_P      = cfg["dropout"]
    GRAD_CLIP      = cfg.get("grad_clip")
    HEAD_TYPE      = cfg.get("head_type", "cosine")
    SCHEDULER_TYPE = cfg.get("scheduler_type", "cosine")
    patches_per_image = cfg["patches_per_img"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "endpoint10.db"
        )

    os.makedirs(model_dir, exist_ok=True)

    # Thread count and worker count respect SLURM allocation
    avail_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", None) or os.cpu_count() or 4)
    torch.set_num_threads(avail_cpus)
    num_workers = min(8, max(2, avail_cpus - 1))

    from gpu_utils import get_gpu_config
    gpu_cfg = get_gpu_config(device, default_batch_size=cfg["batch_size"])
    if cfg.get("force_compile", False):
        gpu_cfg["use_compile"] = True
    batch_size = gpu_cfg["batch_size"]
    print(f"AMP: {gpu_cfg['amp_enabled']}  dtype: {gpu_cfg['amp_dtype']}  "
          f"batch_size: {batch_size}  compile: {gpu_cfg['use_compile']}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, val_df, _ = get_or_create_split(model_dir, db_path)

    is_full_res = config_name.startswith("mpatch_v1")
    is_masked   = config_name.startswith("mpatch_") and not is_full_res

    if is_full_res:
        from masked_patch_dataset_full import MaskedFullPatchDataset
        from model_patch_full import PATCH_SIZE_FULL, OVERSIZED_FULL
        from preprocessing_full import make_preprocessor_full

        preprocessor = make_preprocessor_full()
        _MASK_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   cfg["mask_dir"])
        _MASK_VERSION = cfg["mask_version"]
        _PC_VERSION   = cfg["patch_center_version"]
        _UNIFORM_FRAC = cfg.get("uniform_fraction", 0.00)
        _INCLUDE_GRID = cfg.get("include_grid", True)
        _CSV_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "img-metadata.csv")

        _PRELOAD = cfg.get("preload", False)

        train_ds = MaskedFullPatchDataset(
            train_df, photo_dir, preprocessor,
            mask_dir=_MASK_DIR,
            mask_version=_MASK_VERSION,
            patch_center_version=_PC_VERSION,
            uniform_fraction=_UNIFORM_FRAC,
            include_grid=_INCLUDE_GRID,
            csv_path=_CSV_PATH if _INCLUDE_GRID else None,
            preload=_PRELOAD,
        )
        val_ds = MaskedFullPatchDataset(
            val_df, photo_dir, preprocessor,
            mask_dir=_MASK_DIR,
            mask_version=_MASK_VERSION,
            patch_center_version=_PC_VERSION,
            uniform_fraction=1.0,   # equal patches per image for fair val comparison
            include_grid=False,     # val always uses primary images only
            preload=_PRELOAD,
        )
        train_sampler = train_ds.make_sampler()

    elif is_masked:
        preprocessor = make_preprocessor()
        from masked_patch_dataset import MaskedPatchDataset
        _MASK_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       cfg["mask_dir"])
        _MASK_VERSION   = cfg["mask_version"]
        _PC_VERSION     = cfg["patch_center_version"]
        _INCLUDE_GRID   = cfg.get("include_grid", False)
        _CSV_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "data", "img-metadata.csv")

        _UNIFORM_FRAC     = cfg.get("uniform_fraction",     0.20)
        _VAL_UNIFORM_FRAC = cfg.get("val_uniform_fraction", 1.0)
        # Default val=1.0 gives equal patch weight per image (comparable across ablations).
        # mpatch_v0_a–d set val_uniform_fraction=0.20 explicitly to preserve the behavior
        # they started training with before this default was introduced.

        # GPU preloading: store unpadded (1,400,600) float32 tensors on device.
        # Requires num_workers=0 — DataLoader workers are separate processes and cannot
        # access CUDA tensors created in the main process.
        _PRELOAD_DEVICE_STR = cfg.get("preload_device", None)
        _PRELOAD_DEVICE = torch.device(_PRELOAD_DEVICE_STR) if _PRELOAD_DEVICE_STR else None

        train_ds = MaskedPatchDataset(
            train_df, photo_dir, preprocessor,
            mask_dir=_MASK_DIR,
            mask_version=_MASK_VERSION,
            patch_center_version=_PC_VERSION,
            include_grid=_INCLUDE_GRID,
            csv_path=_CSV_PATH,
            preload=preload if _PRELOAD_DEVICE is None else False,
            uniform_fraction=_UNIFORM_FRAC,
            preload_device=_PRELOAD_DEVICE,
        )
        val_ds = MaskedPatchDataset(
            val_df, photo_dir, preprocessor,
            mask_dir=_MASK_DIR,
            mask_version=_MASK_VERSION,
            patch_center_version=_PC_VERSION,
            include_grid=False,
            preload=preload if _PRELOAD_DEVICE is None else False,
            uniform_fraction=_VAL_UNIFORM_FRAC,
            preload_device=_PRELOAD_DEVICE,
        )
        train_sampler = train_ds.make_sampler()
    else:
        preprocessor = make_preprocessor()
        train_ds = FibrinPatchDataset(
            train_df, photo_dir, preprocessor,
            patches_per_image=patches_per_image, preload=preload,
        )
        val_ds = FibrinPatchDataset(
            val_df, photo_dir, preprocessor,
            patches_per_image=patches_per_image, preload=preload,
        )
        train_sampler = make_patch_sampler(train_ds)
    # GPU preloading requires num_workers=0: DataLoader workers are forked subprocesses
    # and cannot safely access CUDA tensors created in the main process.
    _gpu_preload = is_masked and cfg.get("preload_device") is not None
    _dl_workers  = 0     if _gpu_preload else num_workers
    _dl_pin      = False if _gpu_preload else True
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=_dl_workers, pin_memory=_dl_pin,
        persistent_workers=(_dl_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=_dl_workers, pin_memory=_dl_pin,
        persistent_workers=(_dl_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    if is_full_res:
        from model_patch_full import FibrinPatchCNNFull, PATCH_SIZE_FULL, OVERSIZED_FULL
        _PATCH_SZ  = PATCH_SIZE_FULL
        _OVERSIZED = OVERSIZED_FULL
        model = FibrinPatchCNNFull(num_classes=num_classes, dropout_p=DROPOUT_P,
                                   head_type=HEAD_TYPE).to(device)
    else:
        _PATCH_SZ  = PATCH_SIZE
        _OVERSIZED = OVERSIZED
        model = make_patch_model(num_classes=num_classes, dropout_p=DROPOUT_P,
                                 head_type=HEAD_TYPE).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=gpu_cfg["use_scaler"])
    if gpu_cfg["use_compile"]:
        _backend = cfg.get("compile_backend", "inductor")
        print(f"Compiling model with torch.compile (backend={_backend}) …")
        model = torch.compile(model, backend=_backend)
    augmentation = PatchAugmentation(
        patch_size=_PATCH_SZ,
        brightness=cfg.get("aug_brightness", 0.0),
        contrast=cfg.get("aug_contrast", 0.0),
        noise_std=cfg.get("aug_noise_std", 0.0),
    ).to(device)
    center_crop = K.CenterCrop(_PATCH_SZ)
    class_weights = _compute_class_weights(train_df, device)
    if HEAD_TYPE == "ce":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if SCHEDULER_TYPE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg.get("plateau_factor", 0.95),
            patience=cfg.get("plateau_patience", 5),
        )
    else:
        scheduler = CosineAnnealingWarmRestartsF(
            optimizer, T_0=cfg["T_0"], T_mult=T_MULT, eta_min=ETA_MIN
        )

    # ── Checkpoint resume ────────────────────────────────────────────────────
    ckpt = load_latest_checkpoint(model_dir, map_location=device) if resume else None
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
        pretrain_dir = cfg.get("pretrain_model_dir")
        if pretrain_dir:
            pretrain_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        pretrain_dir)
            weights_path = os.path.join(pretrain_dir, "best_model.pth")
            sd = torch.load(weights_path, map_location=device, weights_only=True)
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            print(f"Loaded pretrain weights from {weights_path}")

    # ── Wandb ────────────────────────────────────────────────────────────────
    _sched_cfg = (
        {
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor":   cfg.get("plateau_factor", 0.95),
            "plateau_patience": cfg.get("plateau_patience", 5),
        }
        if SCHEDULER_TYPE == "plateau"
        else {
            "scheduler": "CosineAnnealingWarmRestarts",
            "scheduler_T0":    cfg["T_0"],
            "scheduler_Tmult": T_MULT,
            "scheduler_eta_min": ETA_MIN,
        }
    )
    config = {
        "model_type": config_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        **_sched_cfg,
        "scheduler_type": SCHEDULER_TYPE,
        "dropout_p": DROPOUT_P,
        "grad_clip": GRAD_CLIP,
        "loss": "CrossEntropyLoss" if HEAD_TYPE == "ce" else "CosineLoss",
        "head_type": HEAD_TYPE,
        "patch_size": _PATCH_SZ,
        "oversized": _OVERSIZED,
        "patches_per_image": train_ds._total_patches if (is_masked or is_full_res) else patches_per_image,
        "num_classes": num_classes,
        "max_epochs": max_epochs,
        "epochs_per_job": epochs_per_job,
        "amp_enabled": gpu_cfg["amp_enabled"],
        "amp_dtype":   str(gpu_cfg["amp_dtype"]),
        "aug_brightness": cfg.get("aug_brightness", 0.0),
        "aug_contrast":   cfg.get("aug_contrast",   0.0),
        "aug_noise_std":  cfg.get("aug_noise_std",  0.0),
        "variant": config_name,
    }
    if is_full_res:
        config.update({
            "dataset":                   "masked_full_patch",
            "mask_version":              _MASK_VERSION,
            "patch_center_version":      _PC_VERSION,
            "mask_min_fg":               cfg.get("mask_min_fg", 0.03),
            "include_grid":              _INCLUDE_GRID,
            "coverage_multiplier":       train_ds.coverage_multiplier,
            "train_uniform_fraction":    train_ds.uniform_fraction,
            "val_uniform_fraction":      val_ds.uniform_fraction,
            "n_train_images":            len(train_ds._valid_row_positions),
            "preload":                   _PRELOAD,
        })
    elif is_masked:
        config.update({
            "dataset":              "masked_patch",
            "mask_version":         _MASK_VERSION,
            "patch_center_version": _PC_VERSION,
            "mask_min_fg":          cfg.get("mask_min_fg", 0.03),
            "include_grid":         _INCLUDE_GRID,
            "coverage_multiplier":       train_ds.coverage_multiplier,
            "train_uniform_fraction":    train_ds.uniform_fraction,
            "val_uniform_fraction":      val_ds.uniform_fraction,   # 0.20 for a-d, 1.0 for e+
            "n_train_images":            len(train_ds._valid_row_positions),
            "preload_device":            str(_PRELOAD_DEVICE) if _PRELOAD_DEVICE else None,
        })
    init_wandb(
        config=config,
        model_type=config_name,
        project=wandb_project,
        run_name=wandb_run_name or config_name,
        run_id=wandb_run_id,
        resume_run=(resume and wandb_run_id is not None),
        enabled=wandb_enabled,
        entity=None,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    if start_epoch >= max_epochs:
        print(f"Already reached max_epochs={max_epochs}. Training complete.")
        finish_wandb()
        sys.exit(100)

    job_epochs_done = 0

    for epoch in range(start_epoch, max_epochs):

        # Training pass
        model.train()
        augmentation.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for patches, labels in train_loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            patches = augmentation(patches)   # oversized → patch_size with rotation + flips

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=gpu_cfg["amp_dtype"],
                                enabled=gpu_cfg["amp_enabled"]):
                sims = model(patches)
                loss = (criterion(sims, labels) if HEAD_TYPE == "ce"
                        else cosine_loss(sims, labels, class_weights))
            scaler.scale(loss).backward()
            if GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(labels)
            train_correct += (sims.argmax(1) == labels).sum().item()
            train_total += len(labels)

        # Cosine scheduler steps on the global epoch (continuity across Slurm jobs).
        # Plateau scheduler steps after validation on val_acc (see below).
        if SCHEDULER_TYPE == "cosine":
            scheduler.step(epoch)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation pass (center-crop only, no rotation)
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                patches = center_crop(patches)   # oversized → patch_size, deterministic
                sims = model(patches)
                loss = (criterion(sims, labels) if HEAD_TYPE == "ce"
                        else cosine_loss(sims, labels, class_weights))
                val_loss_sum += loss.item() * len(labels)
                val_correct += (sims.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if SCHEDULER_TYPE == "plateau":
            scheduler.step(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        lr_now = optimizer.param_groups[0]["lr"]

        if HEAD_TYPE == "cosine":
            # Embedding-geometry metrics — wrap loaders to deliver 200×200 patches
            class _AugLoader:
                def __init__(self, raw): self._raw = raw
                def __iter__(self):
                    for p, l in self._raw:
                        yield augmentation(p.to(device)).cpu(), l
            class _CropLoader:
                def __init__(self, raw): self._raw = raw
                def __iter__(self):
                    for p, l in self._raw:
                        yield center_crop(p), l

            train_cos = compute_cosine_metrics(
                model, _AugLoader(train_loader), device, max_samples=500
            )
            val_cos = compute_cosine_metrics(
                model, _CropLoader(val_loader), device, max_samples=500
            )

            row = {
                "epoch": epoch,
                "train_cosine_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_cosine_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "train_mean_max_sim": round(train_cos["mean_max_sim"], 6),
                "train_intra_cos": round(train_cos["intra_class_cos"], 6),
                "train_inter_cos": round(train_cos["inter_class_cos"], 6),
                "train_silhouette": round(train_cos["silhouette"], 6),
                "val_mean_max_sim": round(val_cos["mean_max_sim"], 6),
                "val_intra_cos": round(val_cos["intra_class_cos"], 6),
                "val_inter_cos": round(val_cos["inter_class_cos"], 6),
                "val_silhouette": round(val_cos["silhouette"], 6),
                "lr": lr_now,
                "model_type": config_name,
                "wall_clock_time": datetime.now(timezone.utc).isoformat(),
            }
            print(
                f"Epoch {epoch:5d} | "
                f"train cosine_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val cosine_loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"sil(tr={train_cos['silhouette']:.3f} va={val_cos['silhouette']:.3f}) | "
                f"lr={lr_now:.2e}"
                + (" ← best" if is_best else "")
            )
        else:  # HEAD_TYPE == "ce"
            row = {
                "epoch": epoch,
                "train_ce_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_ce_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": lr_now,
                "model_type": config_name,
                "wall_clock_time": datetime.now(timezone.utc).isoformat(),
            }
            print(
                f"Epoch {epoch:5d} | "
                f"train ce_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val ce_loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"lr={lr_now:.2e}"
                + (" ← best" if is_best else "")
            )

        log_epoch(model_dir, row)
        history.append({k: v for k, v in row.items() if k != "wall_clock_time"})

        # Checkpointing
        state = {
            "epoch": epoch,
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_val_acc,
            "history": history,
            "model_type": config_name,
            "phase": "training",
            "config": config,
            "wandb_run_id": get_wandb_run_id(),
        }
        save_checkpoint(state, model_dir, epoch, is_best, max_epochs)

        job_epochs_done += 1
        if job_epochs_done >= epochs_per_job:
            print(f"epochs_per_job={epochs_per_job} exhausted at epoch {epoch}. Exiting.")
            finish_wandb(complete=False)
            sys.exit(0)   # Slurm script resubmits

    print(f"Training complete at epoch {max_epochs - 1}.")
    finish_wandb()
    sys.exit(100)   # Slurm script does not resubmit
