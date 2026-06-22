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
}
