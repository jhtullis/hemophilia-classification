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
    },

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
}
