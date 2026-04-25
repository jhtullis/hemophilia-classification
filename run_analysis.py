"""
run_analysis.py — Run all downstream analyses for a trained FibrinCNN model.

Executes the following steps in order:
  1.  evaluate.py                   — per-class precision/recall/F1, confusion matrix
  2.  misclassification_report.py   — per-image errors, confusion matrix, confidence
  3.  activation_analysis.py        — spatial activation maps across conv blocks
  4.  preprocessing_comparison.py   — accuracy sensitivity to grayscale method
  5.  gradcam.py                    — Grad-CAM class saliency maps
  6.  hemophilia_analysis.py --roc  — OVR ROC / AUC curves
  7.  hemophilia_analysis.py --annotate — annotated copies of test images
  8.  visualize_weights.py          — CNN kernel heatmaps + activation maps
  9.  plot_training_curves.py       — train/validation accuracy and loss curves
  10. hemophilia_analysis.py --compare — multi-model ROC overlay

Usage:
    python run_analysis.py [--model-type TYPE] [--db PATH]

    TYPE: 5class | 3class_scratch | 3class_finetune  (default: 5class)

All outputs are saved under models/<type>/analysis/<module>/.
"""

import argparse
import os
import subprocess
import sys


_MODEL_DIRS = {
    "5class":          os.path.join("models", "5class"),
    "3class_scratch":  os.path.join("models", "3class_hemo"),
    "3class_finetune": os.path.join("models", "3class_hemo_finetune"),
}


def _run(script: str, extra_args: list, db_path: str = None) -> bool:
    """Run a script as a subprocess. Returns True on success."""
    cmd = [sys.executable, script] + extra_args
    if db_path:
        cmd += ["--db", db_path]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(
        description="Run all downstream analyses for a trained FibrinCNN model."
    )
    p.add_argument("--model-type", default="5class",
                   choices=list(_MODEL_DIRS.keys()),
                   help="Which trained model to analyse (default: 5class).")
    p.add_argument("--db", default=None, metavar="PATH",
                   help="Path to SQLite database (default: data/endpoint10.db).")
    args = p.parse_args()

    model_type = args.model_type
    model_dir  = _MODEL_DIRS[model_type]
    db_path    = args.db

    if not os.path.exists(os.path.join(model_dir, "best_model.pth")):
        print(f"No trained model found in {model_dir}/. "
              f"Run training first with:\n"
              f"  python train.py --model-type {model_type}")
        sys.exit(1)

    mt = ["--model-type", model_type]

    # Build ordered step list: (label, script, extra_args, pass_db)
    roc_dir = os.path.join(model_dir, "analysis", "roc")
    scripts = [
        ("Evaluation metrics",       "evaluate.py",                mt,                                                                         True),
        ("Misclassification report", "misclassification_report.py", mt,                                                                         True),
        ("Activation analysis",      "activation_analysis.py",      mt,                                                                         True),
        ("Preprocessing comparison", "preprocessing_comparison.py", mt,                                                                         True),
        ("Grad-CAM",                 "gradcam.py",                  mt,                                                                         True),
        ("ROC / AUC curves",         "hemophilia_analysis.py",      mt + ["--roc", "--roc-output",
                                                                          os.path.join(roc_dir, "roc_curves.png")],                             True),
        ("Annotated test images",    "hemophilia_analysis.py",      mt + ["--annotate", "--annotate-dir",
                                                                          os.path.join(model_dir, "analysis", "annotated_test")],               True),
        ("Kernel / activation maps", "visualize_weights.py",        ["--model-dir", model_dir],                                                 False),
    ]

    n = len(scripts) + 2  # +1 training curves, +1 compare
    failed = []

    for i, (label, script, extra_args, pass_db) in enumerate(scripts, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{n}] {label}")
        print("=" * 60)
        ok = _run(script, extra_args, db_path=db_path if pass_db else None)
        if not ok:
            failed.append(label)
            print(f"  WARNING: {label} exited with an error.")

    # Step 9 — Training curves (skip if history absent)
    step9_label = "Training curves"
    print(f"\n{'='*60}")
    print(f"[{n - 1}/{n}] {step9_label}")
    print("=" * 60)
    history_path = os.path.join(model_dir, "training_history.json")
    if os.path.exists(history_path):
        ok = _run("plot_training_curves.py", mt)
        if not ok:
            failed.append(step9_label)
            print(f"  WARNING: {step9_label} exited with an error.")
    else:
        print(f"  Skipping: no training_history.json found at {history_path}.")

    # Step 10 — Multi-model ROC comparison (spans all models; no --model-type)
    step10_label = "Multi-model ROC comparison"
    print(f"\n{'='*60}")
    print(f"[{n}/{n}] {step10_label}")
    print("=" * 60)
    compare_out = os.path.join(roc_dir, "roc_compare.png")
    ok = _run("hemophilia_analysis.py",
              ["--compare", "--roc-output", compare_out],
              db_path=db_path)
    if not ok:
        failed.append(step10_label)
        print(f"  WARNING: {step10_label} exited with an error.")

    print(f"\n{'='*60}")
    if failed:
        print(f"Analysis complete with {len(failed)} failure(s): {failed}")
        sys.exit(1)
    else:
        print(f"All analyses complete. Outputs in {model_dir}/analysis/")


if __name__ == "__main__":
    main()
