"""
evaluate_holdout_125s_ensemble.py — Holdout evaluation of the mpatch_v1e_125s
8-fold leave-one-experiment-out CV ensemble (fold 0 = mpatch_v1e_125s, folds
1-7 = mpatch_v1e_125s_rep_a..g).

For each holdout image, all 8 fold models score the same 15-patch (3 rows x
5 cols) border-free interior 1.25x-resize grid -- physically identical grid
locations to the 200x200/600x400 pipeline's interior grid, scaled 8x, each
center kept >= PATCH_SIZE_125S//2 from every edge so no patch reads into
the zero-padded border. Two accuracy metrics are reported, both
solo-per-model and ensembled (scores averaged across all 8 models):
    - patch accuracy:  each grid patch's own argmax vs. the image's true label
    - image accuracy:  soft-vote (mean over all patches) argmax vs. true label

Copies each fold's best_model.pth and training/split metadata into
models/mpatch_v1e_125s_ensemble/holdout_eval/ for reproducibility.

Usage:
    python evaluate_holdout_125s_ensemble.py
    python evaluate_holdout_125s_ensemble.py --limit 5       # fast dry run
    python evaluate_holdout_125s_ensemble.py --split val     # debug only, NOT the reported holdout result
"""

import argparse
import os
from datetime import datetime, timezone

import cv2
import kornia.augmentation as K
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

from data_loader import CLASS_MAP, CLASS_NAMES
from evaluate_patch import _plot_confusion_matrix, _plot_roc_curves, inference_grid_centers, score_patch_grid
from holdout_eval_utils import (accuracy_breakdown, assert_identical_test_split,
                                copy_best_checkpoint, extract_split_record_subset,
                                get_best_epoch_metadata, load_125s_model, write_json)
from model_patch_125s import OVERSIZED_125S, PAD_125S, PATCH_SIZE_125S
from patch_dataset import _pad_tensor, load_split_record
from preprocessing import make_preprocessor_resized

_DIR = os.path.dirname(os.path.abspath(__file__))

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


def evaluate(args: argparse.Namespace) -> None:
    from configs.training_configs import CONFIGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_dirs = {k: os.path.join(_DIR, "models", k) for k in FOLD_KEYS}
    present_keys = [k for k in FOLD_KEYS if os.path.exists(
        os.path.join(model_dirs[k], "best_model.pth"))]
    missing_keys = [k for k in FOLD_KEYS if k not in present_keys]
    if missing_keys and not args.allow_partial:
        raise FileNotFoundError(
            f"Missing best_model.pth for {missing_keys}. "
            f"Pass --allow-partial to proceed with the {len(present_keys)} present models."
        )
    if missing_keys:
        print(f"WARNING: proceeding without {missing_keys} (--allow-partial)")

    out_dir = args.out_dir or os.path.join(_DIR, "models", "mpatch_v1e_125s_ensemble", "holdout_eval")
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Verifying shared test split across {len(present_keys)} fold models...")
    split_info = assert_identical_test_split([model_dirs[k] for k in present_keys])
    print(f"  {len(split_info['test_indices'])} shared holdout images confirmed.")

    db_path = args.db or os.path.join(_DIR, "data", "endpoint10.db")
    train_df, val_df, test_df = load_split_record(model_dirs[present_keys[0]], db_path)
    eval_df = val_df if args.split == "val" else test_df
    if args.limit:
        eval_df = eval_df.iloc[: args.limit]
    print(f"Evaluating ensemble on {args.split} set: {len(eval_df)} images, "
          f"{len(present_keys)} fold models")

    models = {}
    fold_metadata = []
    for fold_idx, key in enumerate(FOLD_KEYS):
        if key not in present_keys:
            continue
        cfg = CONFIGS.get(key, {})
        head_type = cfg.get("head_type", "ce")
        models[key] = load_125s_model(model_dirs[key], device, head_type=head_type)

        weights_path = copy_best_checkpoint(model_dirs[key], weights_dir, key)
        best_meta = get_best_epoch_metadata(model_dirs[key])
        fold_metadata.append({
            "fold_idx": fold_idx,
            "config_key": key,
            "model_dir": os.path.relpath(model_dirs[key], _DIR),
            "weights_copied_to": os.path.relpath(weights_path, out_dir),
            **best_meta,
        })

    write_json(os.path.join(out_dir, "training_metadata.json"), {
        "family": "mpatch_v1e_125s_ensemble",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_test_split_source": os.path.relpath(
            os.path.join(model_dirs[present_keys[0]], "train_record_patch.json"), _DIR),
        "models": fold_metadata,
    })
    write_json(os.path.join(out_dir, "split_record.json"),
               extract_split_record_subset(model_dirs[present_keys[0]]))

    preprocessor = make_preprocessor_resized(pool_factor=1.25)
    centers = inference_grid_centers(H=3200, W=4800, stride=args.stride, patch_size=PATCH_SIZE_125S)
    center_crop = K.CenterCrop(PATCH_SIZE_125S)
    photo_dir = args.photo_dir or os.path.join(_DIR, "data", "photos")

    image_records = []
    patch_records = []
    solo_patch_rows = {key: [] for key in present_keys}
    for _, row in eval_df.iterrows():
        img_path = os.path.join(photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        tensor = preprocessor(img_bgr)
        padded = _pad_tensor(tensor, PAD_125S)
        true_label = CLASS_MAP[row["Exp_Type"]]

        per_model_scores = {}
        for key in present_keys:
            per_model_scores[key] = score_patch_grid(
                models[key], padded, centers, device, center_crop,
                pad=PAD_125S, oversized=OVERSIZED_125S,
            )   # (num_patches, num_classes)

        stacked = torch.stack([per_model_scores[k] for k in present_keys])   # (n_models, n_patches, n_classes)
        ensemble_patch_scores = stacked.mean(dim=0)                          # (n_patches, n_classes)
        ensemble_image_scores = ensemble_patch_scores.mean(dim=0)            # (n_classes,)
        ensemble_pred = ensemble_image_scores.argmax().item()

        img_rec = {
            "idx": int(row["idx"]),
            "Experiment": row["Experiment"],
            "Exp_Type": row["Exp_Type"],
            "true_label": true_label,
            "pred_label": ensemble_pred,
            "correct": int(ensemble_pred == true_label),
        }
        for i, cls in enumerate(CLASS_NAMES):
            img_rec[f"score_{cls}"] = round(ensemble_image_scores[i].item(), 6)
        for key in present_keys:
            solo_pred = per_model_scores[key].mean(dim=0).argmax().item()
            img_rec[f"pred_label_{key}"] = solo_pred
            img_rec[f"correct_{key}"] = int(solo_pred == true_label)
        image_records.append(img_rec)

        for p_idx, (cy, cx) in enumerate(centers):
            patch_pred = ensemble_patch_scores[p_idx].argmax().item()
            patch_rec = {
                "idx": int(row["idx"]),
                "Experiment": row["Experiment"],
                "Exp_Type": row["Exp_Type"],
                "true_label": true_label,
                "patch_idx": p_idx,
                "cy": cy,
                "cx": cx,
                "pred_label": patch_pred,
                "correct": int(patch_pred == true_label),
            }
            for i, cls in enumerate(CLASS_NAMES):
                patch_rec[f"score_{cls}"] = round(ensemble_patch_scores[p_idx, i].item(), 6)
            patch_records.append(patch_rec)

        for key in present_keys:
            solo_patch_preds = per_model_scores[key].argmax(dim=1)
            for p_idx in range(len(centers)):
                solo_patch_rows[key].append({
                    "Exp_Type": row["Exp_Type"],
                    "correct": int(solo_patch_preds[p_idx].item() == true_label),
                })

    image_df = pd.DataFrame(image_records)
    patch_df = pd.DataFrame(patch_records)

    image_df.to_csv(os.path.join(out_dir, "per_image_predictions.csv"), index=False)
    patch_df.to_csv(os.path.join(out_dir, "per_patch_predictions.csv"), index=False)

    summary_rows = []
    ens_img_acc, ens_img_per_class = accuracy_breakdown(image_df, CLASS_NAMES)
    ens_patch_acc, ens_patch_per_class = accuracy_breakdown(patch_df, CLASS_NAMES)
    summary_rows.append({
        "model_key": "ensemble",
        "patch_accuracy": ens_patch_acc,
        "image_accuracy": ens_img_acc,
        **{f"patch_acc_{c}": ens_patch_per_class.get(c) for c in CLASS_NAMES},
        **{f"image_acc_{c}": ens_img_per_class.get(c) for c in CLASS_NAMES},
        "best_epoch": None,
        "best_val_acc": None,
    })

    for key in present_keys:
        solo_img_df = image_df.rename(columns={
            f"pred_label_{key}": "pred_label", f"correct_{key}": "correct"})
        solo_img_acc, solo_img_per_class = accuracy_breakdown(solo_img_df, CLASS_NAMES)

        solo_patch_df = pd.DataFrame(solo_patch_rows[key])
        solo_patch_acc, solo_patch_per_class = accuracy_breakdown(solo_patch_df, CLASS_NAMES)

        meta = next(m for m in fold_metadata if m["config_key"] == key)
        summary_rows.append({
            "model_key": key,
            "patch_accuracy": solo_patch_acc,
            "image_accuracy": solo_img_acc,
            **{f"patch_acc_{c}": solo_patch_per_class.get(c) for c in CLASS_NAMES},
            **{f"image_acc_{c}": solo_img_per_class.get(c) for c in CLASS_NAMES},
            "best_epoch": meta["best_epoch"],
            "best_val_acc": meta["best_val_acc"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)
    write_json(os.path.join(out_dir, "summary.json"), {
        "n_test_images": len(eval_df),
        "n_grid_patches_per_image": len(centers),
        "n_models_used": len(present_keys),
        "models_used": present_keys,
        "rows": summary_rows,
    })

    print(f"\nEnsemble image accuracy ({args.split}): {ens_img_acc:.3f}")
    print(f"Ensemble patch accuracy ({args.split}): {ens_patch_acc:.3f}")
    for cls in CLASS_NAMES:
        print(f"  {cls}: image={ens_img_per_class.get(cls, float('nan')):.3f}  "
              f"patch={ens_patch_per_class.get(cls, float('nan')):.3f}")

    image_out = os.path.join(out_dir, "image")
    patch_out = os.path.join(out_dir, "patch")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(patch_out, exist_ok=True)
    _plot_confusion_matrix(image_df, CLASS_NAMES, "mpatch_v1e_125s_ensemble (image)", image_out)
    _plot_roc_curves(image_df, CLASS_NAMES, CLASS_MAP,
                     "mpatch_v1e_125s_ensemble (image)", image_out)
    _plot_confusion_matrix(patch_df, CLASS_NAMES, "mpatch_v1e_125s_ensemble (patch)", patch_out)

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Holdout evaluation of the mpatch_v1e_125s 8-fold CV ensemble."
    )
    p.add_argument("--db", default=None)
    p.add_argument("--photo-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--stride", type=int, default=800)
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N holdout images (fast dry run).")
    p.add_argument("--allow-partial", action="store_true",
                   help="Proceed even if some of the 8 checkpoints are missing.")
    evaluate(p.parse_args())
