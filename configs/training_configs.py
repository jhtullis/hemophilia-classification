"""
Hyperparameter configs for all training variants.
Each config overrides only the fields that differ from its base.
Do not modify existing entries once a model has started training.
"""

# ── Base configs (v0 values) ──────────────────────────────────────────────────

_5CLASS_HPC_V0_BASE = {
    "model_type":    "5class_hpc_v0",
    "model_dir":     "models/5class_hpc_v0",
    "lr":            1e-3,
    "weight_decay":  1e-3,
    "T_0":           100,
    "T_mult":        2,
    "eta_min":       1e-6,
    "dropout":       0.5,
    "batch_size":    16,
    "max_epochs":    10000,
    "grad_clip":     None,
}

_PATCH_V0_BASE = {
    "model_type":      "patch_v0",
    "model_dir":       "models/patch_v0",
    "lr":              1e-3,
    "weight_decay":    1e-3,
    "T_0":             100,
    "T_mult":          2.0,
    "eta_min":         1e-6,
    "dropout":         0.5,
    "batch_size":      64,
    "patches_per_img": 20,
    "max_epochs":      10000,
    "grad_clip":       None,
}

_MPATCH_V1E_LITE_LC_BASE = {
    **_PATCH_V0_BASE,           # batch_size=64, weight_decay=1e-3, etc.
    "lr":                   2e-3,
    "head_type":            "ce",
    "dropout":              0.3,
    "mask_dir":             "masks",
    "mask_version":         "v_intensity",
    "patch_center_version": "p200_circle_v_intensity",
    "mask_min_fg":          0.03,
    "include_grid":         True,
    "uniform_fraction":     0.10,       # matches mpatch_v0e_lite
    "scheduler_type":       "plateau",
    "plateau_factor":       0.95,
    "plateau_patience":     5,
    "plateau_mode":         "min",      # track val_loss, not val_acc
    "plateau_metric":       "loss",
    "plateau_threshold":    1e-4,
    "preload_device":       "cuda",     # unpadded (1,400,600) float32 on GPU; num_workers=0
    "split_source_dir":     "models/mpatch_v0e_lite",   # share exact split record
    "early_stop":           True,
    "early_stop_min_epochs": 1000,
    "early_stop_patience":  30,
    "early_stop_threshold": 1e-4,
}

# ── 5class_hpc variants ───────────────────────────────────────────────────────

CONFIGS = {

    "5class_hpc_v0": _5CLASS_HPC_V0_BASE,

    "5class_hpc_v1a": {
        **_5CLASS_HPC_V0_BASE,
        "model_type": "5class_hpc_v1a",
        "model_dir":  "models/5class_hpc_v1a",
        # Raise eta_min floor + slow cycle growth (1.5×); weight_decay unchanged
        "T_mult":     1.5,
        "eta_min":    1e-4,
        "dropout":    0.3,
    },

    "5class_hpc_v1b": {
        **_5CLASS_HPC_V0_BASE,
        "model_type":   "5class_hpc_v1b",
        "model_dir":    "models/5class_hpc_v1b",
        # WD reduced 3×; scheduler unchanged
        "weight_decay": 3e-4,
        "dropout":      0.3,
    },

    "5class_hpc_v1c": {
        **_5CLASS_HPC_V0_BASE,
        "model_type":   "5class_hpc_v1c",
        "model_dir":    "models/5class_hpc_v1c",
        # Raise eta_min floor + slow cycle growth + 2× WD reduction
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "weight_decay": 5e-4,
        "dropout":      0.3,
    },

    # ── patch variants ────────────────────────────────────────────────────────

    "patch_v0": _PATCH_V0_BASE,

    "patch_v1a": {
        **_PATCH_V0_BASE,
        "model_type": "patch_v1a",
        "model_dir":  "models/patch_5class_v1a",
        "T_mult":     1.5,
        "eta_min":    1e-4,
        "dropout":    0.3,
    },

    "patch_v1b": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v1b",
        "model_dir":    "models/patch_5class_v1b",
        "weight_decay": 3e-4,
        "dropout":      0.3,
    },

    "patch_v1c": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v1c",
        "model_dir":    "models/patch_5class_v1c",
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "weight_decay": 5e-4,
        "dropout":      0.3,
    },

    # ── patch v2: cross-entropy head, mirrors v1a/b/c hyperparameters ─────────
    # head_type="ce" swaps NormalizedLinear for nn.Linear; loss = CrossEntropyLoss.

    "patch_v2a": {
        **_PATCH_V0_BASE,
        "model_type": "patch_v2a",
        "model_dir":  "models/patch_ce_v2a",
        "head_type":  "ce",
        "T_mult":     1.5,
        "eta_min":    1e-4,
        "dropout":    0.3,
    },

    "patch_v2b": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v2b",
        "model_dir":    "models/patch_ce_v2b",
        "head_type":    "ce",
        "weight_decay": 3e-4,
        "dropout":      0.3,
    },

    "patch_v2c": {
        **_PATCH_V0_BASE,
        "model_type":   "patch_v2c",
        "model_dir":    "models/patch_ce_v2c",
        "head_type":    "ce",
        "T_mult":       1.5,
        "eta_min":      1e-4,
        "weight_decay": 5e-4,
        "dropout":      0.3,
    },

    # ── masked-patch v0: mask-guided sampling, replicates v1a/v2a hyperparams ─
    # Shared mask params (intensity mask, min_fg=0.03, patch_size=200).
    # patches_per_img inherited from _PATCH_V0_BASE but unused by MaskedPatchDataset.
    #
    # NOTE (mpatch_v0_a through d): these models were trained before val_uniform_fraction
    # was fixed at 1.0. Their validation curves used uniform_fraction=0.20 for val,
    # so val patch counts were content-weighted rather than equal per image. Do not
    # compare their val accuracy directly against e-h without accounting for this.

    "mpatch_v0_a": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_a",
        "model_dir":            "models/mpatch_v0_a",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         False,
        "val_uniform_fraction": 0.20,   # legacy: training started before val was fixed at 1.0
    },

    "mpatch_v0_b": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_b",
        "model_dir":            "models/mpatch_v0_b",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "val_uniform_fraction": 0.20,   # legacy: training started before val was fixed at 1.0
    },

    "mpatch_v0_c": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_c",
        "model_dir":            "models/mpatch_v0_c",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         False,
        "val_uniform_fraction": 0.20,   # legacy: training started before val was fixed at 1.0
    },

    "mpatch_v0_d": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_d",
        "model_dir":            "models/mpatch_v0_d",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "val_uniform_fraction": 0.20,   # legacy: training started before val was fixed at 1.0
    },

    # ── mpatch_v0_e/f: ablate uniform_fraction (baseline = 0.20 in a–d) ─────
    # Both use CE head + exploratory+grid, identical to mpatch_v0_d otherwise.

    "mpatch_v0_e": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_e",
        "model_dir":            "models/mpatch_v0_e",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
    },

    "mpatch_v0_f": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_f",
        "model_dir":            "models/mpatch_v0_f",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
    },

    # ── mpatch_v0_e_ensw/f_ensw: exact replicas of e/f with a top-k checkpoint ──
    # pool (top 3 per training period + top 9 all-time, by val_acc, never
    # deleted) accumulated for a later ensembling test.

    "mpatch_v0_e_ensw": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_e_ensw",
        "model_dir":            "models/mpatch_v0_e_ensw",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "topk_ensemble_save":   True,
    },

    "mpatch_v0_f_ensw": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_f_ensw",
        "model_dir":            "models/mpatch_v0_f_ensw",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "topk_ensemble_save":   True,
    },

    "mpatch_v0_g": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_g",
        "model_dir":            "models/mpatch_v0_g",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.05,
    },

    "mpatch_v0_h": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_h",
        "model_dir":            "models/mpatch_v0_h",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.01,
    },

    # ── mpatch_v0_f1a: fine-tune from mpatch_v0_f best weights ───────────────
    # Loads best_model.pth from mpatch_v0_f; optimizer and scheduler start fresh.
    # Fixed 100-epoch cosine cycles (T_mult=1.0) for 10k epochs = 100 cycles.
    # LR ceiling (1e-3) and floor (1e-4) match mpatch_v0_f.
    "mpatch_v0_f1a": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_f1a",
        "model_dir":            "models/mpatch_v0_f1a",
        "head_type":            "ce",
        "T_mult":               1.0,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "pretrain_model_dir":   "models/mpatch_v0_f",
    },

    # ── mpatch_v1_full: full-resolution patch pipeline ───────────────────────
    # Extracts 2000×2000 patches from 6000×4000 JPEGs (no min-pool).
    # Center masks reused from p200_circle_v_intensity (upscaled 10× on the fly).
    # Run on H200 partitions (eng, m13h, mgh); batch_size=512 → ~56 GB peak.
    # include_grid=True is the default for all mpatch_v1_full models.

    "mpatch_v1_full_a": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v1_full_a",
        "model_dir":            "models/mpatch_v1_full_a",
        "lr":                   1e-3,
        "weight_decay":         1e-3,
        "T_0":                  100,
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "batch_size":           512,
        "head_type":            "ce",
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         False,  # original run without grid; keep frozen
        "uniform_fraction":     0.00,
    },

    "mpatch_v1_full_b": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v1_full_b",
        "model_dir":            "models/mpatch_v1_full_b",
        "lr":                   3e-3,   # ×3 sqrt scaling vs mpatch_v0 (batch size 64→512 → √8≈3)
        "weight_decay":         2.5e-3, # ×2.5 scaling vs _a (batch size increase)
        "T_0":                  100,
        "T_mult":               1.5,
        "eta_min":              3e-4,   # ×3 sqrt scaling vs _a
        "dropout":              0.3,
        "batch_size":           512,
        "head_type":            "ce",
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "preload":              True,   # grayscale JPEG preload: ~15–30 GB CPU RAM
    },

    # ── mpatch_v0 "lite" variants: ReduceLROnPlateau + GPU preloading ────────
    # Named <parent>_lite. Training regime differs from the parent:
    #   - lr = 2× parent max LR (1e-3 → 2e-3)
    #   - ReduceLROnPlateau(mode='max', factor=0.95, patience=5) instead of cosine
    #   - preload_device="cuda": unpadded (1,400,600) float32 tensors stored on GPU
    #     → requires num_workers=0 in DataLoader (set automatically by train_patch.py)
    #   - No partition constraint in Slurm scripts → runs on any available GPU
    # Uniform fraction, include_grid, and head_type match each parent model.
    # CRITICAL: include_grid=True uses split-filtered grid images only (see CLAUDE.md §CRITICAL).

    "mpatch_v0e_lite": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0e_lite",
        "model_dir":            "models/mpatch_v0e_lite",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
    },

    "mpatch_v0f_lite": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0f_lite",
        "model_dir":            "models/mpatch_v0f_lite",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
    },

    "mpatch_v0g_lite": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0g_lite",
        "model_dir":            "models/mpatch_v0g_lite",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.05,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
    },

    "mpatch_v0h_lite": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0h_lite",
        "model_dir":            "models/mpatch_v0h_lite",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.01,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
    },

    # ── mpatch_v0_f_aug: standard mpatch_v0_f + brightness/contrast jitter ──
    # Identical to mpatch_v0_f (cosine scheduler, CPU preload, full partition list)
    # but with ColorJitter added to the training augmentation pipeline.

    "mpatch_v0_f_aug": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_f_aug",
        "model_dir":            "models/mpatch_v0_f_aug",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "aug_brightness":       0.15,
        "aug_contrast":         0.20,
    },

    # ── mpatch_v0 "_lite_aug" variants: lite + brightness/contrast jitter ────
    # Identical to the corresponding _lite model but add ColorJitter after flips:
    #   aug_brightness=0.15  → brightness multiplier ~ U(0.85, 1.15)
    #   aug_contrast=0.20    → contrast multiplier  ~ U(0.80, 1.20)
    # Applied only during training (validation uses plain center-crop, no jitter).

    "mpatch_v0e_lite_aug": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0e_lite_aug",
        "model_dir":            "models/mpatch_v0e_lite_aug",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
        "aug_brightness":       0.15,
        "aug_contrast":         0.20,
    },

    "mpatch_v0f_lite_aug": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0f_lite_aug",
        "model_dir":            "models/mpatch_v0f_lite_aug",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
        "aug_brightness":       0.15,
        "aug_contrast":         0.20,
    },

    # ── mpatch_v0f_lite_aug2: lite_aug + additive Gaussian noise ─────────────
    # Identical to mpatch_v0f_lite_aug but adds Gaussian noise (std=0.03) after
    # ColorJitter. Pipeline order: rotation → crop → flips → jitter → noise.

    "mpatch_v0f_lite_aug2": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0f_lite_aug2",
        "model_dir":            "models/mpatch_v0f_lite_aug2",
        "lr":                   2e-3,
        "head_type":            "ce",
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.00,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
        "aug_brightness":       0.15,
        "aug_contrast":         0.20,
        "aug_noise_std":        0.03,
    },

    # ── mpatch_v1e_lite learning curve series ────────────────────────────────
    # Mirrors mpatch_v0e_lite (10x min-pool, FibrinPatchCNN, GPU preload, CE head)
    # but trains on progressively smaller subsets of the training experiments
    # (1–7 per class → 5–35 total), for learning curve analysis.
    # All 7 models share the exact same train/val/test split as mpatch_v0e_lite.
    # Scheduler: ReduceLROnPlateau tracking val_loss (min, threshold=1e-4).
    # Early stopping: after 1000 epochs, 30 consecutive epochs without ≥1e-4
    # improvement in val_loss triggers sys.exit(100) (no Slurm resubmit).

    "mpatch_v1e_lite_lc35a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc35a",
        "model_dir":      "models/mpatch_v1e_lite_lc35a",
        "lc_n_per_class": 7, "lc_seed": 35,
    },

    "mpatch_v1e_lite_lc30a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc30a",
        "model_dir":      "models/mpatch_v1e_lite_lc30a",
        "lc_n_per_class": 6, "lc_seed": 30,
    },

    "mpatch_v1e_lite_lc25a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc25a",
        "model_dir":      "models/mpatch_v1e_lite_lc25a",
        "lc_n_per_class": 5, "lc_seed": 25,
    },

    "mpatch_v1e_lite_lc20a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc20a",
        "model_dir":      "models/mpatch_v1e_lite_lc20a",
        "lc_n_per_class": 4, "lc_seed": 20,
    },

    "mpatch_v1e_lite_lc15a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc15a",
        "model_dir":      "models/mpatch_v1e_lite_lc15a",
        "lc_n_per_class": 3, "lc_seed": 15,
    },

    "mpatch_v1e_lite_lc10a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc10a",
        "model_dir":      "models/mpatch_v1e_lite_lc10a",
        "lc_n_per_class": 2, "lc_seed": 10,
    },

    "mpatch_v1e_lite_lc05a": {
        **_MPATCH_V1E_LITE_LC_BASE,
        "model_type":     "mpatch_v1e_lite_lc05a",
        "model_dir":      "models/mpatch_v1e_lite_lc05a",
        "lc_n_per_class": 1, "lc_seed": 5,
    },

    # ── b repetition: same split, different training-subset seeds (100+N) ─────

    "mpatch_v1e_lite_lc35b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc35b", "model_dir": "models/mpatch_v1e_lite_lc35b",
        "lc_n_per_class": 7, "lc_seed": 135},
    "mpatch_v1e_lite_lc30b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc30b", "model_dir": "models/mpatch_v1e_lite_lc30b",
        "lc_n_per_class": 6, "lc_seed": 130},
    "mpatch_v1e_lite_lc25b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc25b", "model_dir": "models/mpatch_v1e_lite_lc25b",
        "lc_n_per_class": 5, "lc_seed": 125},
    "mpatch_v1e_lite_lc20b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc20b", "model_dir": "models/mpatch_v1e_lite_lc20b",
        "lc_n_per_class": 4, "lc_seed": 120},
    "mpatch_v1e_lite_lc15b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc15b", "model_dir": "models/mpatch_v1e_lite_lc15b",
        "lc_n_per_class": 3, "lc_seed": 115},
    "mpatch_v1e_lite_lc10b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc10b", "model_dir": "models/mpatch_v1e_lite_lc10b",
        "lc_n_per_class": 2, "lc_seed": 110},
    "mpatch_v1e_lite_lc05b": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc05b", "model_dir": "models/mpatch_v1e_lite_lc05b",
        "lc_n_per_class": 1, "lc_seed": 105},

    # ── c repetition: seeds 200+N ─────────────────────────────────────────────

    "mpatch_v1e_lite_lc35c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc35c", "model_dir": "models/mpatch_v1e_lite_lc35c",
        "lc_n_per_class": 7, "lc_seed": 235},
    "mpatch_v1e_lite_lc30c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc30c", "model_dir": "models/mpatch_v1e_lite_lc30c",
        "lc_n_per_class": 6, "lc_seed": 230},
    "mpatch_v1e_lite_lc25c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc25c", "model_dir": "models/mpatch_v1e_lite_lc25c",
        "lc_n_per_class": 5, "lc_seed": 225},
    "mpatch_v1e_lite_lc20c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc20c", "model_dir": "models/mpatch_v1e_lite_lc20c",
        "lc_n_per_class": 4, "lc_seed": 220},
    "mpatch_v1e_lite_lc15c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc15c", "model_dir": "models/mpatch_v1e_lite_lc15c",
        "lc_n_per_class": 3, "lc_seed": 215},
    "mpatch_v1e_lite_lc10c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc10c", "model_dir": "models/mpatch_v1e_lite_lc10c",
        "lc_n_per_class": 2, "lc_seed": 210},
    "mpatch_v1e_lite_lc05c": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc05c", "model_dir": "models/mpatch_v1e_lite_lc05c",
        "lc_n_per_class": 1, "lc_seed": 205},

    # ── d repetition: seeds 300+N ─────────────────────────────────────────────

    "mpatch_v1e_lite_lc35d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc35d", "model_dir": "models/mpatch_v1e_lite_lc35d",
        "lc_n_per_class": 7, "lc_seed": 335},
    "mpatch_v1e_lite_lc30d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc30d", "model_dir": "models/mpatch_v1e_lite_lc30d",
        "lc_n_per_class": 6, "lc_seed": 330},
    "mpatch_v1e_lite_lc25d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc25d", "model_dir": "models/mpatch_v1e_lite_lc25d",
        "lc_n_per_class": 5, "lc_seed": 325},
    "mpatch_v1e_lite_lc20d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc20d", "model_dir": "models/mpatch_v1e_lite_lc20d",
        "lc_n_per_class": 4, "lc_seed": 320},
    "mpatch_v1e_lite_lc15d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc15d", "model_dir": "models/mpatch_v1e_lite_lc15d",
        "lc_n_per_class": 3, "lc_seed": 315},
    "mpatch_v1e_lite_lc10d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc10d", "model_dir": "models/mpatch_v1e_lite_lc10d",
        "lc_n_per_class": 2, "lc_seed": 310},
    "mpatch_v1e_lite_lc05d": {**_MPATCH_V1E_LITE_LC_BASE,
        "model_type": "mpatch_v1e_lite_lc05d", "model_dir": "models/mpatch_v1e_lite_lc05d",
        "lc_n_per_class": 1, "lc_seed": 305},

    # ── mpatch_v0_i: compile test ─────────────────────────────────────────────
    # Replicates mpatch_v0_d (CE head, exploratory+grid, uniform_fraction=0.20)
    # with force_compile=True to verify torch.compile works on the cluster.
    # Uses val_uniform_fraction=1.0 (current default). Needs module load cuda/12.8.1
    # and cuda-cudart-dev in the conda env (or CONDA_NO_PLUGINS=true conda install
    # -c nvidia cuda-cudart-dev).
    "mpatch_v0_i": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v0_i",
        "model_dir":            "models/mpatch_v0_i",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "force_compile":        True,
        "compile_backend":      "cudagraphs",
    },

    # ── mpatch_v1e_2xmp: 2x min-pool high-resolution patch pipeline ──────────
    # Images 2x min-pooled (2000x3000, vs 600x400 at 10x) instead of unpooled or
    # 10x-pooled. Patches are 1000x1000 (PATCH_SIZE_2X), same physical footprint as the
    # 200x200 patches at 10x-pool and the 2000x2000 patches in mpatch_v1_full. Center
    # masks reused from p200_circle_v_intensity (600x400), sampled at coarse granularity
    # then jittered to the finer 2x-pool grid -- see masked_patch_dataset_2x.py.
    # Mirrors mpatch_v0_e's training paradigm (CE head, exploratory+grid,
    # uniform_fraction=0.10). GPU-preloads pooled images as uint8 (~32GB) directly onto
    # the device -- requires an H200 (run on H200 partitions: eng, m13h, mgh).
    "mpatch_v1e_2xmp": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v1e_2xmp",
        "model_dir":            "models/mpatch_v1e_2xmp",
        "resolution_variant":   "2x",
        "head_type":            "ce",
        "T_mult":               1.5,
        "eta_min":              1e-4,
        "dropout":              0.3,
        "batch_size":           64,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "preload_device":       "cuda",
    },

    # ── mpatch_v1e_125s family: 1.25x area-resize patch pipeline, 3 LR-schedule variants ──
    # Images downscaled 1.25x via cv2.INTER_AREA resize (4800x3200) -- see
    # masked_patch_dataset_125s.py / model_patch_125s.py for the shared dataset/model code.
    # All three variants require an H200 (run on H200 partitions: eng, m13h, mgh) --
    # the ~79.6GB GPU image preload is a property of the resolution, not the LR schedule,
    # so this applies even to the plateau variant (unlike mpatch_v0e_lite's precedent).
    # All three set topk_ensemble_save=True (mirrors mpatch_v0_f_ensw) for later ensembling.
    # Verify actual peak GPU memory on first run before trusting batch_size=64
    # (gpu_utils.py forces batch_size=64 on any SM>=9 GPU regardless of this config's value).

    "mpatch_v1e_125s": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v1e_125s",
        "model_dir":            "models/mpatch_v1e_125s",
        "resolution_variant":   "1.25x",
        "head_type":            "ce",
        "T_mult":               1.5,          # geometrically-increasing restarts
        "eta_min":              1e-4,
        "dropout":              0.3,
        "batch_size":           64,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "preload_device":       "cuda",
        "topk_ensemble_save":   True,
    },

    "mpatch_v1e_125s_flat": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v1e_125s_flat",
        "model_dir":            "models/mpatch_v1e_125s_flat",
        "resolution_variant":   "1.25x",
        "head_type":            "ce",
        "T_mult":               1.0,          # flat, non-increasing 100-epoch restarts
        "eta_min":              1e-4,
        "dropout":              0.3,
        "batch_size":           64,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "preload_device":       "cuda",
        "topk_ensemble_save":   True,
    },

    "mpatch_v1e_125s_plateau": {
        **_PATCH_V0_BASE,
        "model_type":           "mpatch_v1e_125s_plateau",
        "model_dir":            "models/mpatch_v1e_125s_plateau",
        "resolution_variant":   "1.25x",
        "lr":                   2e-3,         # 2x base LR, matches mpatch_v0e_lite's full paradigm
        "head_type":            "ce",
        "dropout":              0.3,
        "batch_size":           64,
        "mask_dir":             "masks",
        "mask_version":         "v_intensity",
        "patch_center_version": "p200_circle_v_intensity",
        "mask_min_fg":          0.03,
        "include_grid":         True,
        "uniform_fraction":     0.10,
        "scheduler_type":       "plateau",
        "plateau_factor":       0.95,
        "plateau_patience":     5,
        "preload_device":       "cuda",
        "topk_ensemble_save":   True,
    },
}
