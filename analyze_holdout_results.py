"""
analyze_holdout_results.py — Local analysis of holdout-eval outputs already
pulled down from HPC (models/mpatch_v1e_125s_ensemble/holdout_eval/ and
models/mpatch_v1e_lite_lc*/holdout_eval/). Recomputes cross-entropy loss from
the raw per-class score columns already saved in per_image_predictions.csv /
per_patch_predictions.csv -- all evaluated models use head_type="ce", so
score_<class> columns are raw logits, sufficient for both accuracy (already
saved) and loss (not saved, computed here). No model re-run or HPC job needed.

Writes to analysis/ at the repo root:
    analysis/125s_ensemble/summary.json, summary_table.csv
    analysis/125s_ensemble/confusion_matrices/final_model_image_classification.png,
        final_model_patch_classification.png, selected_model.png (presentation variant
        of the image one -- larger fonts + spelled-out class key), <fold_key>.png (x8,
        image-level only -- solo per-fold patch predictions weren't persisted) -- reuses
        misclassification_report.plot_confusion_matrix (count + row-fraction style: cell
        color = row-normalized fraction, text = "count\n(fraction)")
    analysis/learning_curve/learning_curve_summary.{csv,json}
    analysis/learning_curve/lc_basic_<col>.png, lc_basic_<col>_loglog.png (x4 metrics)
    analysis/learning_curve/lc_powerlaw_<col>.png (x4 metrics, log-log fit + bootstrap CI +
        in-plot fit-quality/asymptote annotation), lc_powerlaw_<col>_linear.png (accuracy
        metrics only, linear-y fit + CI)
    analysis/learning_curve/power_law_fit_report.{txt,json} -- fit quality (R^2, RMSE),
        bootstrap convergence, loss-floor/accuracy-ceiling percentiles, and extrapolated
        predictions at several dd sizes, for all 4 metrics

Usage:
    python analyze_holdout_results.py
    python analyze_holdout_results.py --which 125s
    python analyze_holdout_results.py --which lc
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

from data_loader import CLASS_NAMES
from holdout_eval_utils import accuracy_breakdown, write_json
from misclassification_report import plot_confusion_matrix

_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(_DIR, "models")
ANALYSIS_ROOT = os.path.join(_DIR, "analysis")

LC_SIZES = [5, 10, 15, 20, 25, 30, 35]
LC_REPS = ["a", "b", "c", "d"]
LC_KEYS = [f"mpatch_v1e_lite_lc{n:02d}{r}" for n in LC_SIZES for r in LC_REPS]

CLASS_KEY = {
    "NC1": "Normal Plasma",
    "AC3": "Prolonged Clotting",
    "F08D": "Hemophilia A",
    "F09D": "Hemophilia B",
    "F11D": "Hemophilia C",
}
CLASS_ORDER = list(CLASS_KEY.keys())

FOLD_KEYS = [
    "mpatch_v1e_125s",
    "mpatch_v1e_125s_rep_a",
    "mpatch_v1e_125s_rep_b",
    "mpatch_v1e_125s_rep_c",
    "mpatch_v1e_125s_rep_d",
    "mpatch_v1e_125s_rep_e",
    "mpatch_v1e_125s_rep_f",
    "mpatch_v1e_125s_rep_g",
]

# dataviz reference palette, categorical slots 1 (blue) and 2 (orange)
COLOR_PATCH = "#2a78d6"
COLOR_IMAGE = "#eb6834"

# Learning-curve power-law fit / bootstrap CI (see lc_basic_*/lc_powerlaw_* plots below)
LC_BOOTSTRAP_ITERS = 1000
LC_BOOTSTRAP_MIN_SUCCESS = 10
LC_BOOTSTRAP_SEED = 0   # fixed -> reproducible CI band across reruns

# Forward extrapolation only -- no backward extrapolation grid is generated below the
# smallest observed dd. RIGHT_EXTRAPOLATION_LOG10=0.5 caps the rightmost extrapolated
# point at dd_max * 10**0.5 (half an order of magnitude past the largest observed dd).
RIGHT_EXTRAPOLATION_LOG10 = 0.5
# Extrapolated dd values reported in the fit report / marked in-plot, as multiples of
# dd_max; the last multiplier equals 10**RIGHT_EXTRAPOLATION_LOG10, i.e. the grid's edge.
EXTRAPOLATION_MULTIPLIERS = [1.15, 1.5, 2.0, 10 ** RIGHT_EXTRAPOLATION_LOG10]


def _extrapolation_grid(dd_sorted: List[int], n: int = 300) -> np.ndarray:
    """Dense x-grid spanning [dd_min, dd_max * 10**RIGHT_EXTRAPOLATION_LOG10] -- no
    backward extrapolation below the smallest observed dd, bounded forward extrapolation
    past the largest."""
    dd_min, dd_max = min(dd_sorted), max(dd_sorted)
    return np.linspace(dd_min, dd_max * (10 ** RIGHT_EXTRAPOLATION_LOG10), n)


def compute_loss(
    df: pd.DataFrame, class_names: List[str] = CLASS_NAMES
) -> Tuple[float, Dict[str, float]]:
    """Mean cross-entropy loss (and per-class breakdown) from raw score_<class>
    logit columns + true_label. Same formula used at training time."""
    logits = torch.tensor(df[[f"score_{c}" for c in class_names]].values, dtype=torch.float32)
    targets = torch.tensor(df["true_label"].values, dtype=torch.long)
    per_row = F.cross_entropy(logits, targets, reduction="none")

    per_class = {}
    for i, cls in enumerate(class_names):
        mask = targets == i
        if mask.any():
            per_class[cls] = per_row[mask].mean().item()
    return per_row.mean().item(), per_class


def _save_named_confusion_matrix(
    df: pd.DataFrame, class_names: List[str], title: str, out_dir: str, filename: str,
    **plot_kwargs,
) -> None:
    """Reuses misclassification_report.plot_confusion_matrix (count + row-fraction
    style, chosen over the plain raw-count style). That function always writes a
    fixed 'confusion_matrix.png', so save into out_dir then rename to a unique
    filename; each call target is otherwise self-contained. plot_kwargs passes
    through optional class_key / font-size overrides for one-off call sites."""
    plot_confusion_matrix(df, class_names, out_dir, title=title, **plot_kwargs)
    os.replace(
        os.path.join(out_dir, "confusion_matrix.png"),
        os.path.join(out_dir, f"{filename}.png"),
    )


# ---------------------------------------------------------------------------
# 125s 8-fold ensemble
# ---------------------------------------------------------------------------

def analyze_125s_ensemble() -> None:
    holdout_dir = os.path.join(MODELS_ROOT, "mpatch_v1e_125s_ensemble", "holdout_eval")
    image_df = pd.read_csv(os.path.join(holdout_dir, "per_image_predictions.csv"))
    patch_df = pd.read_csv(os.path.join(holdout_dir, "per_patch_predictions.csv"))

    with open(os.path.join(holdout_dir, "training_metadata.json")) as f:
        fold_metadata = {m["config_key"]: m for m in json.load(f)["models"]}

    img_acc, img_acc_per_class = accuracy_breakdown(image_df, CLASS_NAMES)
    patch_acc, patch_acc_per_class = accuracy_breakdown(patch_df, CLASS_NAMES)
    img_loss, img_loss_per_class = compute_loss(image_df)
    patch_loss, patch_loss_per_class = compute_loss(patch_df)

    out_dir = os.path.join(ANALYSIS_ROOT, "125s_ensemble")
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    _save_named_confusion_matrix(
        image_df, CLASS_NAMES, "Final Model Image Classification", cm_dir,
        "final_model_image_classification",
    )
    _save_named_confusion_matrix(
        patch_df, CLASS_NAMES, "Final Model Patch Classification", cm_dir,
        "final_model_patch_classification",
    )
    # Presentation-facing variant of final_model_image_classification.png: larger fonts
    # (this call site only -- plot_confusion_matrix's shared defaults are untouched) +
    # a spelled-out class-abbreviation key to the right of the colorbar.
    _save_named_confusion_matrix(
        image_df, CLASS_NAMES, "Selected Model: Holdout Evaluation", cm_dir, "selected_model",
        class_key=CLASS_KEY, class_order=CLASS_ORDER, cell_fontsize=13, tick_fontsize=12,
        label_fontsize=14, title_fontsize=16, key_fontsize=16,
    )

    # Solo per-fold accuracy (and an image-level confusion matrix) is recoverable
    # (pred_label_<key>/correct_<key> were saved); solo per-fold loss and patch-level
    # confusion matrices are NOT -- only the ensemble's raw scores and patch-level
    # predictions were persisted per row, not each of the 8 models' own.
    solo_rows = []
    for key in FOLD_KEYS:
        pred_col, correct_col = f"pred_label_{key}", f"correct_{key}"
        if correct_col not in image_df.columns:
            continue
        solo_img_df = image_df[["Exp_Type"]].assign(correct=image_df[correct_col])
        solo_acc, _ = accuracy_breakdown(solo_img_df, CLASS_NAMES)

        fold_cm_df = pd.DataFrame({
            "true_label": image_df["true_label"],
            "pred_label": image_df[pred_col],
        })
        _save_named_confusion_matrix(fold_cm_df, CLASS_NAMES, f"{key} (image)", cm_dir, key)

        meta = fold_metadata.get(key, {})
        solo_rows.append({
            "model_key": key,
            "image_accuracy": solo_acc,
            "image_loss": None,
            "patch_accuracy": None,
            "patch_loss": None,
            "best_epoch": meta.get("best_epoch"),
            "best_val_acc": meta.get("best_val_acc"),
        })

    write_json(os.path.join(out_dir, "summary.json"), {
        "n_test_images": len(image_df),
        "n_grid_patches_per_image": len(patch_df) // len(image_df) if len(image_df) else None,
        "ensemble": {
            "image_accuracy": img_acc,
            "image_loss": img_loss,
            "image_accuracy_per_class": img_acc_per_class,
            "image_loss_per_class": img_loss_per_class,
            "patch_accuracy": patch_acc,
            "patch_loss": patch_loss,
            "patch_accuracy_per_class": patch_acc_per_class,
            "patch_loss_per_class": patch_loss_per_class,
        },
        "solo_models_note": (
            "Per-fold raw scores weren't saved during the HPC run (only "
            "pred_label/correct per fold), so solo-model loss can't be recomputed "
            "locally -- accuracy only. Re-run evaluate_holdout_125s_ensemble.py with "
            "per-fold score columns added if solo loss is needed."
        ),
        "solo_models": solo_rows,
    })

    rows = [{
        "model_key": "ensemble",
        "image_accuracy": img_acc, "image_loss": img_loss,
        "patch_accuracy": patch_acc, "patch_loss": patch_loss,
    }] + solo_rows
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    print(f"125s ensemble -- image: acc={img_acc:.3f} loss={img_loss:.3f}  "
          f"patch: acc={patch_acc:.3f} loss={patch_loss:.3f}")
    print(f"Written to {out_dir} ({len(solo_rows) + 3} confusion matrices under {cm_dir})")


# ---------------------------------------------------------------------------
# lite_lc learning curve (accuracy + loss)
# ---------------------------------------------------------------------------

def analyze_learning_curve() -> None:
    rows = []
    missing = []
    for key in LC_KEYS:
        holdout_dir = os.path.join(MODELS_ROOT, key, "holdout_eval")
        img_path = os.path.join(holdout_dir, "per_image_predictions.csv")
        patch_path = os.path.join(holdout_dir, "per_patch_predictions.csv")
        summary_path = os.path.join(holdout_dir, "summary.json")
        if not (os.path.exists(img_path) and os.path.exists(patch_path)):
            missing.append(key)
            continue

        image_df = pd.read_csv(img_path)
        patch_df = pd.read_csv(patch_path)
        img_acc, _ = accuracy_breakdown(image_df, CLASS_NAMES)
        patch_acc, _ = accuracy_breakdown(patch_df, CLASS_NAMES)
        img_loss, _ = compute_loss(image_df)
        patch_loss, _ = compute_loss(patch_df)

        with open(summary_path) as f:
            meta = json.load(f)

        rows.append({
            "model_key": key,
            "lc_n_per_class": meta.get("lc_n_per_class"),
            "rep": key[-1],
            "image_accuracy": img_acc,
            "image_loss": img_loss,
            "patch_accuracy": patch_acc,
            "patch_loss": patch_loss,
            "best_epoch": meta.get("best_epoch"),
            "best_val_acc": meta.get("best_val_acc"),
        })

    if missing:
        print(f"WARNING: {len(missing)} lc models missing holdout_eval outputs, skipped: {missing}")
    if not rows:
        raise FileNotFoundError("No lite_lc holdout_eval outputs found under models/.")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ANALYSIS_ROOT, "learning_curve")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "learning_curve_summary.csv"), index=False)

    metrics = ["image_accuracy", "image_loss", "patch_accuracy", "patch_loss"]
    agg = df.groupby("lc_n_per_class")[metrics].agg(["mean", "std", "count"])
    per_size = {}
    for n in agg.index:
        entry = {"n_reps": int(agg.loc[n, (metrics[0], "count")])}
        for m in metrics:
            std = agg.loc[n, (m, "std")]
            entry[f"{m}_mean"] = float(agg.loc[n, (m, "mean")])
            entry[f"{m}_std"] = float(std) if std == std else 0.0   # NaN -> 0.0 (single rep)
        per_size[int(n)] = entry

    write_json(os.path.join(out_dir, "learning_curve_summary.json"), {
        "n_models_found": len(df),
        "n_models_missing": len(missing),
        "missing_models": missing,
        "per_size": per_size,
    })

    plot_learning_curves(df, out_dir)

    print(f"Learning curve analysis written to {out_dir} ({len(df)}/{len(LC_KEYS)} models)")


_METRIC_LABELS = {
    "patch_accuracy": "Patch accuracy",
    "image_accuracy": "Image accuracy (soft-vote)",
    "patch_loss": "Patch loss",
    "image_loss": "Image loss (soft-vote)",
}

# Clean title-only labels (no "(soft-vote)" qualifier, no model name) -- used for
# lc_basic_*/lc_powerlaw_* plot titles per user request; _METRIC_LABELS above is
# still used for axis labels/legends/report text.
_METRIC_TITLE_LABELS = {
    "patch_accuracy": "Patch Accuracy",
    "image_accuracy": "Image Accuracy",
    "patch_loss": "Patch Loss",
    "image_loss": "Image Loss",
}


# ---------------------------------------------------------------------------
# Power-law scaling fit + bootstrap CI
#
# y = a * N^beta + c (3-param; a>=0, beta<=0, c>=0), falling back to y = a * N^beta
# (2-param; no floor) if the 3-param fit doesn't converge. For accuracy columns the
# fit is performed on error (1-acc), then converted back, so the beta<=0/c<=0 bounds
# stay meaningful (error shrinking with more data, floored at 0).
# Cf. Hestness et al. 2017, "Deep Learning Scaling is Predictable, Empirically"
# (arXiv:1712.00409).
# ---------------------------------------------------------------------------

def _power_law3(N, a, beta, c):
    return a * np.power(N, beta) + c


def _power_law2(N, a, beta):
    return a * np.power(N, beta)


def _fit_power_law(dd: np.ndarray, y: np.ndarray) -> Optional[dict]:
    """3-param fit with 2-param (no-floor) fallback. Returns None if len(dd) < 2 or
    both fits fail to converge."""
    if len(dd) < 2:
        return None
    c0 = max(float(np.min(y)), 0.0)
    a0 = max(float(y[0] - c0), 1e-6)
    try:
        popt, _ = curve_fit(_power_law3, dd, y, p0=[a0, -0.5, c0],
                             bounds=([0, -np.inf, 0], [np.inf, 0, np.inf]),
                             maxfev=10000)
        return {"a": popt[0], "beta": popt[1], "c": popt[2], "n_params": 3}
    except (RuntimeError, ValueError):
        pass
    try:
        popt, _ = curve_fit(_power_law2, dd, y, p0=[a0, -0.5],
                             bounds=([0, -np.inf], [np.inf, 0]), maxfev=10000)
        return {"a": popt[0], "beta": popt[1], "c": None, "n_params": 2}
    except (RuntimeError, ValueError):
        return None


def _eval_fit(params: dict, x: np.ndarray) -> np.ndarray:
    if params["n_params"] == 3:
        return _power_law3(x, params["a"], params["beta"], params["c"])
    return _power_law2(x, params["a"], params["beta"])


def _fit_power_law_metric(dd: np.ndarray, y: np.ndarray, is_accuracy: bool) -> Optional[dict]:
    """Fits in error-space (1-y) for accuracy columns, tagging the result so
    _eval_fit_metric can convert back. Loss columns fit directly."""
    target = (1.0 - y) if is_accuracy else y
    params = _fit_power_law(dd, target)
    if params is not None:
        params["is_accuracy"] = is_accuracy
    return params


def _eval_fit_metric(params: dict, x: np.ndarray) -> np.ndarray:
    curve = _eval_fit(params, x)
    return (1.0 - curve) if params["is_accuracy"] else curve


def _analyze_metric(df: pd.DataFrame, metric: str, is_accuracy: bool) -> dict:
    """Single source of truth for a metric's power-law analysis: central fit,
    goodness-of-fit (R^2/RMSE against observed per-dd means), one 1000-iteration
    bootstrap (shared by the plot's CI band and the text/JSON report -- run once,
    not once per consumer), the resulting floor/ceiling percentiles, and fitted +
    extrapolated predictions at several dd sizes."""
    dd_sorted = sorted(df["dd"].unique())
    per_dd = {n: df.loc[df["dd"] == n, metric].to_numpy() for n in dd_sorted}
    means = np.array([per_dd[n].mean() for n in dd_sorted])
    dd_arr = np.array(dd_sorted, dtype=float)
    x_grid = _extrapolation_grid(dd_sorted)

    central = _fit_power_law_metric(dd_arr, means, is_accuracy)
    r2 = rmse = fitted_curve = None
    if central is not None:
        fitted_at_dd = _eval_fit_metric(central, dd_arr)
        resid = means - fitted_at_dd
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((means - means.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        fitted_curve = _eval_fit_metric(central, x_grid)

    # Bootstrap: resample each dd's replicate list with replacement, refit, evaluate
    # on x_grid. A 2-param fallback fit is treated as c=0 (no floor) for the purposes
    # of the floor/ceiling distribution -- it's literally the 3-param model with the
    # floor term dropped, i.e. the best-supported floor was indistinguishable from 0.
    rng = np.random.default_rng(LC_BOOTSTRAP_SEED)
    curves, floor_vals, n_3p, n_2p = [], [], 0, 0
    for _ in range(LC_BOOTSTRAP_ITERS):
        boot_means = np.array([
            rng.choice(per_dd[n], size=len(per_dd[n]), replace=True).mean()
            for n in dd_sorted
        ])
        params = _fit_power_law_metric(dd_arr, boot_means, is_accuracy)
        if params is None:
            continue
        if params["n_params"] == 3:
            n_3p += 1
            floor_err = params["c"]
        else:
            n_2p += 1
            floor_err = 0.0
        floor_vals.append((1.0 - floor_err) if is_accuracy else floor_err)
        curves.append(_eval_fit_metric(params, x_grid))

    n_success = len(curves)
    lower = upper = None
    if n_success >= LC_BOOTSTRAP_MIN_SUCCESS:
        stacked = np.vstack(curves)
        lower = np.percentile(stacked, 2.5, axis=0)
        upper = np.percentile(stacked, 97.5, axis=0)

    floor_percentiles = None
    if floor_vals:
        floor_arr = np.array(floor_vals)
        floor_percentiles = {p: float(np.percentile(floor_arr, p))
                              for p in (2.5, 25, 50, 75, 97.5)}

    predictions = []
    for n in dd_sorted:
        predictions.append({
            "dd": int(n), "kind": "observed",
            "fitted": float(np.interp(n, x_grid, fitted_curve)) if fitted_curve is not None else None,
            "observed_mean": float(df.loc[df["dd"] == n, metric].mean()),
        })
    dd_max = dd_sorted[-1]
    for mult in EXTRAPOLATION_MULTIPLIERS:
        n = dd_max * mult
        entry = {
            "dd": round(n, 1), "kind": "extrapolated", "multiplier": mult,
            "fitted": float(np.interp(n, x_grid, fitted_curve)) if fitted_curve is not None else None,
        }
        if lower is not None:
            entry["ci_lower"] = float(np.interp(n, x_grid, lower))
            entry["ci_upper"] = float(np.interp(n, x_grid, upper))
        predictions.append(entry)

    return {
        "metric": metric, "is_accuracy": is_accuracy, "dd_sorted": dd_sorted,
        "means": means, "x_grid": x_grid, "central": central, "fitted_curve": fitted_curve,
        "r2": r2, "rmse": rmse,
        "boot_lower": lower, "boot_upper": upper,
        "n_boot_success": n_success, "n_boot_failed": LC_BOOTSTRAP_ITERS - n_success,
        "n_boot_3param": n_3p, "n_boot_2param": n_2p,
        "floor_percentiles": floor_percentiles,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Learning curve plots (basic scatter+mean, and power-law fit + bootstrap CI)
# ---------------------------------------------------------------------------

def _style_axes(fig, ax) -> None:
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", lw=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#898781")


def _metric_color(metric: str) -> str:
    return COLOR_PATCH if metric.startswith("patch_") else COLOR_IMAGE


def _plot_basic(df: pd.DataFrame, metric: str, out_dir: str, loglog: bool) -> None:
    dd_sorted = sorted(df["dd"].unique())
    color = _metric_color(metric)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    _style_axes(fig, ax)

    for n in dd_sorted:
        vals = df.loc[df["dd"] == n, metric]
        ax.scatter([n] * len(vals), vals, color=color, s=22, alpha=0.3, zorder=2)
    means = [df.loc[df["dd"] == n, metric].mean() for n in dd_sorted]
    ax.plot(dd_sorted, means, color=color, lw=2, zorder=3)
    ax.scatter(dd_sorted, means, color=color, s=70, edgecolor="#0b0b0b",
               linewidth=0.6, zorder=4)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_xticks(dd_sorted)
    ax.set_xlabel("Total training experiments", color="#0b0b0b")
    ax.set_ylabel(_METRIC_LABELS[metric], color="#0b0b0b")
    ax.set_title(f"{_METRIC_TITLE_LABELS[metric]} vs. Number of Training Experiments",
                 color="#0b0b0b")

    suffix = "_loglog" if loglog else ""
    path = os.path.join(out_dir, f"lc_basic_{metric}{suffix}.png")
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def _plot_powerlaw(analysis: dict, out_dir: str, loglog: bool) -> None:
    metric = analysis["metric"]
    is_accuracy = analysis["is_accuracy"]
    dd_sorted = analysis["dd_sorted"]
    x_grid = analysis["x_grid"]
    central = analysis["central"]
    color = _metric_color(metric)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    _style_axes(fig, ax)

    ax.scatter(dd_sorted, analysis["means"], color=color, s=60, edgecolor="#0b0b0b",
               linewidth=0.6, zorder=4, label="Observed mean")

    if central is None:
        print(f"WARNING: power-law fit failed to converge for {metric}; "
              f"plotting observed means only.")
    else:
        ax.plot(x_grid, analysis["fitted_curve"], color=color, lw=2, ls="--", zorder=3,
                label="Power-law fit")

        if analysis["boot_lower"] is not None:
            ax.fill_between(x_grid, analysis["boot_lower"], analysis["boot_upper"],
                             color=color, alpha=0.15, zorder=1, label="95% bootstrap CI")
        else:
            print(f"NOTE: bootstrap CI skipped for {metric} "
                  f"({analysis['n_boot_success']}/{LC_BOOTSTRAP_ITERS} fits converged, "
                  f"need >= {LC_BOOTSTRAP_MIN_SUCCESS})")

        # Asymptote reference line: bootstrap MEDIAN floor/ceiling (not the central
        # fit's own point estimate) -- with only 7 dd points and a>=0/beta<=0/c>=0
        # bounds, the central fit's c often pins near the c>=0 boundary (a degenerate
        # near-unbounded fit) even when the bootstrap distribution is wide; the median
        # is far more robust to that boundary artifact and matches the percentiles
        # already shown in the stats box / report below.
        asym_label = "ceiling" if is_accuracy else "floor"
        fp = analysis["floor_percentiles"]
        if fp is not None:
            asymptote = fp[50]
            ax.axhline(asymptote, color=color, ls=":", lw=1.2, alpha=0.7, zorder=2,
                        label=f"Asymptotic {asym_label} (median) ≈ {asymptote:.3f}")

        stats_lines = [
            f"{'3' if central['n_params'] == 3 else '2'}-param fit"
            + ("" if central["n_params"] == 3 else " (no floor)"),
            f"R² = {analysis['r2']:.3f}   RMSE = {analysis['rmse']:.3f}",
        ]
        if fp is not None:
            stats_lines.append(
                f"{asym_label.capitalize()} (95% CI): "
                f"{fp[2.5]:.3f}–{fp[97.5]:.3f}"
            )
        if abs(central["beta"]) < 0.05:
            stats_lines.append("β ≈ 0: no significant scaling trend")
        box_y, box_va = (0.03, "bottom") if is_accuracy else (0.97, "top")
        ax.text(0.97, box_y, "\n".join(stats_lines), transform=ax.transAxes,
                fontsize=8.5, color="#4a4a45", ha="right", va=box_va,
                bbox=dict(boxstyle="round", facecolor="#fcfcfb", edgecolor="#c3c2b7", alpha=0.9))

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        step = 10
        x_max = x_grid[-1] if x_grid is not None and len(x_grid) else max(dd_sorted)
        ax.set_xticks(np.arange(0, int(x_max) + step + 1, step))
    ax.set_xlabel("Total training experiments", color="#0b0b0b")
    ax.set_ylabel(_METRIC_LABELS[metric], color="#0b0b0b")
    ax.set_title(f"{_METRIC_TITLE_LABELS[metric]} vs. Number of Training Experiments",
                 color="#0b0b0b")
    ax.legend(loc="upper left" if is_accuracy else "lower left", frameon=False, fontsize=8.5)

    suffix = "" if loglog else "_linear"
    path = os.path.join(out_dir, f"lc_powerlaw_{metric}{suffix}.png")
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_learning_curves(df: pd.DataFrame, out_dir: str) -> Dict[str, dict]:
    plot_df = df.copy()
    plot_df["dd"] = plot_df["lc_n_per_class"] * 5   # total training experiments (5..35)

    for stale in ("learning_curve_accuracy.png", "learning_curve_loss.png"):
        stale_path = os.path.join(out_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    analyses = {}
    for metric in ["patch_accuracy", "image_accuracy", "patch_loss", "image_loss"]:
        is_accuracy = metric.endswith("_accuracy")
        _plot_basic(plot_df, metric, out_dir, loglog=False)
        _plot_basic(plot_df, metric, out_dir, loglog=True)

        analysis = _analyze_metric(plot_df, metric, is_accuracy)
        analyses[metric] = analysis
        _plot_powerlaw(analysis, out_dir, loglog=True)
        if is_accuracy:
            _plot_powerlaw(analysis, out_dir, loglog=False)

    _write_power_law_report(analyses, out_dir)
    return analyses


def _write_power_law_report(analyses: Dict[str, dict], out_dir: str) -> None:
    """Human-readable .txt + machine-readable .json report: fit quality (R^2/RMSE),
    bootstrap convergence, loss-floor/accuracy-ceiling percentiles, and fitted +
    extrapolated predictions, for all 4 metrics. Shares the exact same numbers as
    the lc_powerlaw_*.png annotations (computed once in _analyze_metric)."""
    lines = [
        "Power-law scaling fit report -- mpatch_v1e_lite_lc holdout learning curve",
        "Source: analysis/learning_curve/learning_curve_summary.csv "
        "(28 models, one holdout-eval scalar per model per metric)",
        "Fit model: y = a * N^beta + c  (3-param; a>=0, beta<=0, c>=0), N = dd = total "
        "training experiments (lc_n_per_class * 5);",
        "  falls back to y = a * N^beta (2-param, no floor) if the 3-param fit doesn't "
        "converge. Accuracy metrics are fit in error space (1-accuracy) and converted back.",
        f"Bootstrap: {LC_BOOTSTRAP_ITERS} resamples (with replacement) of each dd's "
        "per-replicate values, refit each time; band/percentiles require >= "
        f"{LC_BOOTSTRAP_MIN_SUCCESS} converged fits.",
        f"Extrapolation shown/reported forward only, out to dd_max * 10^{RIGHT_EXTRAPOLATION_LOG10} "
        "(half an order of magnitude past the largest observed dd); no backward extrapolation.",
    ]

    json_out = {}
    for metric, a in analyses.items():
        label = _METRIC_LABELS[metric]
        lines.append(f"\n{'=' * 3} {metric} ({label}) {'=' * 3}")
        c = a["central"]
        if c is None:
            lines.append("Power-law fit FAILED to converge (both 3-param and 2-param).")
            json_out[metric] = {"fit_failed": True}
            continue

        asym_kind = "ceiling" if a["is_accuracy"] else "floor"
        if c["n_params"] == 3:
            asym_val = (1.0 - c["c"]) if a["is_accuracy"] else c["c"]
            lines.append(f"Fit: 3-parameter (a={c['a']:.4f}, beta={c['beta']:.4f}, "
                         f"c={c['c']:.4f})")
        else:
            asym_val = 1.0 if a["is_accuracy"] else 0.0
            lines.append(f"Fit: 2-parameter fallback (a={c['a']:.4f}, beta={c['beta']:.4f}); "
                         "no detectable floor (3-param fit didn't converge)")
        lines.append(f"Goodness of fit (observed per-dd means vs. fitted): "
                     f"R^2={a['r2']:.4f}  RMSE={a['rmse']:.4f}")
        lines.append(f"Bootstrap convergence: {a['n_boot_success']}/{LC_BOOTSTRAP_ITERS} "
                     f"succeeded ({a['n_boot_3param']} three-param, {a['n_boot_2param']} "
                     f"two-param fallback, {a['n_boot_failed']} failed)")

        fp = a["floor_percentiles"]
        if fp is not None:
            lines.append(f"Asymptotic {asym_kind} (N -> infinity), bootstrap percentiles "
                         f"(central fit: {asym_val:.4f}):")
            lines.append("    " + "   ".join(f"{p}%: {v:.4f}" for p, v in fp.items()))
        else:
            lines.append(f"Asymptotic {asym_kind} bootstrap percentiles unavailable "
                         f"(<{LC_BOOTSTRAP_MIN_SUCCESS} converged fits).")

        lines.append("Fitted vs. observed at training sizes:")
        for p in a["predictions"]:
            if p["kind"] != "observed":
                continue
            lines.append(f"    dd={p['dd']:<4d} fitted={p['fitted']:.4f}  "
                         f"observed_mean={p['observed_mean']:.4f}")

        lines.append("Extrapolated predictions:")
        for p in a["predictions"]:
            if p["kind"] != "extrapolated":
                continue
            ci = f"  [95% CI {p['ci_lower']:.4f}-{p['ci_upper']:.4f}]" if "ci_lower" in p else ""
            lines.append(f"    dd={p['dd']:<6g} ({p['multiplier']:.2f}x): "
                         f"{p['fitted']:.4f}{ci}")

        json_out[metric] = {
            "fit_failed": False,
            "n_params": c["n_params"],
            "a": c["a"], "beta": c["beta"], "c": c["c"],
            "asymptote_kind": asym_kind, "asymptote_central": asym_val,
            "asymptote_bootstrap_percentiles": fp,
            "r2": a["r2"], "rmse": a["rmse"],
            "bootstrap": {
                "n_iters": LC_BOOTSTRAP_ITERS, "n_success": a["n_boot_success"],
                "n_3param": a["n_boot_3param"], "n_2param": a["n_boot_2param"],
                "n_failed": a["n_boot_failed"],
            },
            "predictions": a["predictions"],
        }

    text = "\n".join(lines)
    print("\n" + text)
    txt_path = os.path.join(out_dir, "power_law_fit_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    write_json(os.path.join(out_dir, "power_law_fit_report.json"), json_out)
    print(f"\nSaved {txt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Analyze holdout-eval outputs already pulled from HPC (accuracy + loss)."
    )
    p.add_argument("--which", choices=["125s", "lc", "all"], default="all")
    args = p.parse_args()

    if args.which in ("125s", "all"):
        analyze_125s_ensemble()
    if args.which in ("lc", "all"):
        analyze_learning_curve()
