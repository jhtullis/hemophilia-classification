"""
run_analysis.py — Run all downstream analyses for a trained FibrinCNN model.

Executes the following steps in order:
  1. misclassification_report.py  — per-image errors, confusion matrix, confidence
  2. activation_analysis.py       — spatial activation maps across conv blocks
  3. preprocessing_comparison.py  — accuracy sensitivity to grayscale method
  4. gradcam.py                   — Grad-CAM class saliency maps
  5. plot_training_curves.py      — train/validation accuracy and loss curves

Usage:
    python run_analysis.py [--model-type TYPE]

    TYPE: 5class | 3class_scratch | 3class_finetune  (default: 5class)

All outputs are saved under models/<type>/analysis/<module>/ (unchanged from
running each script individually).
"""

import argparse
import subprocess
import sys
import os


_SCRIPTS = [
    ("Misclassification report", "misclassification_report.py"),
    ("Activation analysis",      "activation_analysis.py"),
    ("Preprocessing comparison", "preprocessing_comparison.py"),
    ("Grad-CAM",                 "gradcam.py"),
]

_MODEL_DIRS = {
    "5class":          os.path.join("models", "5class"),
    "3class_scratch":  os.path.join("models", "3class_hemo"),
    "3class_finetune": os.path.join("models", "3class_hemo_finetune"),
}


def _run(script: str, model_type: str) -> bool:
    """Run a script as a subprocess. Returns True on success."""
    cmd = [sys.executable, script, "--model-type", model_type]
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(
        description="Run all downstream analyses for a trained FibrinCNN model."
    )
    p.add_argument("--model-type", default="5class",
                   choices=list(_MODEL_DIRS.keys()),
                   help="Which trained model to analyse (default: 5class).")
    args = p.parse_args()

    model_dir = _MODEL_DIRS[args.model_type]
    if not os.path.exists(os.path.join(model_dir, "best_model.pth")):
        print(f"No trained model found in {model_dir}/. "
              f"Run training first with:\n"
              f"  python train.py --model-type {args.model_type}")
        sys.exit(1)

    n = len(_SCRIPTS) + 1  # +1 for curves plot
    failed = []

    for i, (label, script) in enumerate(_SCRIPTS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{n}] {label}")
        print("=" * 60)
        ok = _run(script, args.model_type)
        if not ok:
            failed.append(label)
            print(f"  WARNING: {label} exited with an error.")

    # Training curves (reads training_history.json, no --model-type arg needed)
    print(f"\n{'='*60}")
    print(f"[{n}/{n}] Training curves")
    print("=" * 60)
    history_path = os.path.join(model_dir, "training_history.json")
    if os.path.exists(history_path):
        cmd = [sys.executable, "plot_training_curves.py",
               "--model-type", args.model_type]
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or ".")
        if result.returncode != 0:
            failed.append("Training curves")
            print("  WARNING: plot_training_curves.py exited with an error.")
    else:
        print(f"  Skipping: no training_history.json found at {history_path}.")
        print("  (Re-train the model to generate history.)")

    print(f"\n{'='*60}")
    if failed:
        print(f"Analysis complete with {len(failed)} failure(s): {failed}")
        sys.exit(1)
    else:
        print(f"All analyses complete. Outputs in {model_dir}/analysis/")


if __name__ == "__main__":
    main()
