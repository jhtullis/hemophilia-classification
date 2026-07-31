"""
evaluate_holdout_lite_lc.py — Holdout evaluation of the mpatch_v1e_lite_lc*
learning-curve series (7 training-set sizes x 4 seed repetitions = 28
independent models, no ensembling — evaluated separately for
aggregate_learning_curve.py to summarize).

All 28 models share an identical holdout test split (copied verbatim from
mpatch_v0e_lite via split_source_dir at training time), so the ~200 test
images are decoded/preprocessed once and reused across all 28 model passes.

Patches are scored on a border-free interior grid (3 rows x 5 cols = 15
centers, stride=100, each kept >= PATCH_SIZE//2 from every edge) rather
than the 35-center edge-inclusive grid evaluate_patch.py uses elsewhere --
no patch here reads into the zero-padded border.

For each model, two accuracy metrics are reported:
    - patch accuracy:  each grid patch's own argmax vs. the image's true label
    - image accuracy:  soft-vote (mean over all patches) argmax vs. true label

Copies each model's best_model.pth and training/split metadata into
models/<model_key>/holdout_eval/ for reproducibility.

Usage:
    python evaluate_holdout_lite_lc.py
    python evaluate_holdout_lite_lc.py --models mpatch_v1e_lite_lc05a --limit 5   # fast dry run
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

from analysis_utils import load_model_from_registry
from evaluate_patch import (_plot_confusion_matrix, _plot_roc_curves,
                            inference_grid_centers, score_patch_grid)
from holdout_eval_utils import (accuracy_breakdown, assert_identical_test_split,
                                copy_best_checkpoint, extract_split_record_subset,
                                get_best_epoch_metadata, write_json)
from patch_dataset import PAD, PATCH_SIZE, load_split_record, _pad_tensor
from preprocessing import make_preprocessor

_DIR = os.path.dirname(os.path.abspath(__file__))

LC_SIZES = [5, 10, 15, 20, 25, 30, 35]
LC_REPS = ["a", "b", "c", "d"]
DEFAULT_MODELS = [f"mpatch_v1e_lite_lc{n:02d}{r}" for n in LC_SIZES for r in LC_REPS]


def evaluate(args: argparse.Namespace) -> None:
    from configs.training_configs import CONFIGS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_keys = args.models.split(",") if args.models else DEFAULT_MODELS
    model_dirs = {k: os.path.join(_DIR, "models", k) for k in model_keys}
    present_keys = [k for k in model_keys if os.path.exists(
        os.path.join(model_dirs[k], "best_model.pth"))]
    missing_keys = [k for k in model_keys if k not in present_keys]
    if missing_keys:
        print(f"WARNING: skipping {len(missing_keys)} models with no best_model.pth: {missing_keys}")
    if not present_keys:
        raise FileNotFoundError(f"No trained models found among {model_keys}.")

    print(f"Verifying shared test split across {len(present_keys)} lite_lc models...")
    assert_identical_test_split([model_dirs[k] for k in present_keys])

    db_path = args.db or os.path.join(_DIR, "data", "endpoint10.db")
    train_df, val_df, test_df = load_split_record(model_dirs[present_keys[0]], db_path)
    eval_df = val_df if args.split == "val" else test_df
    if args.limit:
        eval_df = eval_df.iloc[: args.limit]
    print(f"Evaluating {len(present_keys)} models on {args.split} set: {len(eval_df)} images")

    preprocessor = make_preprocessor()
    centers = inference_grid_centers(patch_size=PATCH_SIZE)
    center_crop = K.CenterCrop(PATCH_SIZE)
    photo_dir = args.photo_dir or os.path.join(_DIR, "data", "photos")

    print("Preloading and padding shared test images...")
    cached = []
    for _, row in eval_df.iterrows():
        img_path = os.path.join(photo_dir, f"{int(row['idx']):04d}.JPG")
        img_bgr = cv2.imread(img_path)
        tensor = preprocessor(img_bgr)
        padded = _pad_tensor(tensor, PAD)
        cached.append((row, padded))

    for key in present_keys:
        model, model_dir, num_classes, class_map, class_names = load_model_from_registry(key, device)
        cfg = CONFIGS.get(key, {})

        out_dir = os.path.join(model_dir, "holdout_eval")
        weights_dir = os.path.join(out_dir, "weights")
        os.makedirs(out_dir, exist_ok=True)

        weights_path = copy_best_checkpoint(model_dir, weights_dir, key)
        best_meta = get_best_epoch_metadata(model_dir)
        write_json(os.path.join(out_dir, "training_metadata.json"), {
            "config_key": key,
            "model_dir": os.path.relpath(model_dir, _DIR),
            "lc_n_per_class": cfg.get("lc_n_per_class"),
            "lc_seed": cfg.get("lc_seed"),
            "split_source_dir": cfg.get("split_source_dir"),
            "weights_copied_to": os.path.relpath(weights_path, out_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **best_meta,
        })
        write_json(os.path.join(out_dir, "split_record.json"),
                   extract_split_record_subset(model_dir))

        image_records, patch_records = [], []
        for row, padded in cached:
            true_label = class_map[row["Exp_Type"]]
            scores = score_patch_grid(model, padded, centers, device, center_crop)   # (n_patches, n_classes)
            mean_scores = scores.mean(dim=0)
            image_pred = mean_scores.argmax().item()

            img_rec = {
                "idx": int(row["idx"]),
                "Experiment": row["Experiment"],
                "Exp_Type": row["Exp_Type"],
                "true_label": true_label,
                "pred_label": image_pred,
                "correct": int(image_pred == true_label),
            }
            for i, cls in enumerate(class_names):
                img_rec[f"score_{cls}"] = round(mean_scores[i].item(), 6)
            image_records.append(img_rec)

            for p_idx, (cy, cx) in enumerate(centers):
                patch_pred = scores[p_idx].argmax().item()
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
                for i, cls in enumerate(class_names):
                    patch_rec[f"score_{cls}"] = round(scores[p_idx, i].item(), 6)
                patch_records.append(patch_rec)

        image_df = pd.DataFrame(image_records)
        patch_df = pd.DataFrame(patch_records)
        image_df.to_csv(os.path.join(out_dir, "per_image_predictions.csv"), index=False)
        patch_df.to_csv(os.path.join(out_dir, "per_patch_predictions.csv"), index=False)

        img_acc, img_per_class = accuracy_breakdown(image_df, class_names)
        patch_acc, patch_per_class = accuracy_breakdown(patch_df, class_names)

        write_json(os.path.join(out_dir, "summary.json"), {
            "model_key": key,
            "lc_n_per_class": cfg.get("lc_n_per_class"),
            "lc_seed": cfg.get("lc_seed"),
            "n_test_images": len(eval_df),
            "n_grid_patches_per_image": len(centers),
            "patch_accuracy": patch_acc,
            "image_accuracy": img_acc,
            "patch_accuracy_per_class": patch_per_class,
            "image_accuracy_per_class": img_per_class,
            **best_meta,
        })

        _plot_confusion_matrix(image_df, class_names, key, out_dir)
        _plot_roc_curves(image_df, class_names, class_map, key, out_dir)
        _plot_confusion_matrix(patch_df, class_names, f"{key} (patch)", out_dir)

        print(f"{key}: image_acc={img_acc:.3f}  patch_acc={patch_acc:.3f}  "
              f"(lc_n_per_class={cfg.get('lc_n_per_class')}, lc_seed={cfg.get('lc_seed')})")

    print(f"\nDone. Per-model results under models/<key>/holdout_eval/.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Batched holdout evaluation of the mpatch_v1e_lite_lc* learning-curve series."
    )
    p.add_argument("--models", default=None,
                   help="Comma-separated model keys (default: all 28 lc models).")
    p.add_argument("--db", default=None)
    p.add_argument("--photo-dir", default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N holdout images (fast dry run).")
    evaluate(p.parse_args())
