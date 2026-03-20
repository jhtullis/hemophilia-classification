# Analysis Pipeline Overview

## Purpose

This folder contains implementation plans for four analysis modules that extend the
FibrinCNN project beyond basic training and evaluation. All three models are already
trained; these modules study *how* the models behave and *why* they succeed or fail.

| Plan file | Script to create | Analysis type |
|-----------|-----------------|---------------|
| `plan_shared_refactor.md` | `analysis_utils.py` | Prerequisite shared utilities |
| `plan_02_misclassification.md` | `misclassification_report.py` | Per-image error breakdown |
| `plan_03_activation_analysis.md` | `activation_analysis.py` | Spatial attention maps |
| `plan_05_preprocessing_comparison.md` | `preprocessing_comparison.py` | Grayscale method robustness |
| `plan_06_gradcam.md` | `gradcam.py` | Grad-CAM class saliency |

---

## Codebase Context

Working directory: project root containing all `.py` files.

### Models already trained
```
models/5class/               best_model.pth + train_record.json + visualizations
models/3class_hemo/          best_model.pth + train_record.json
models/3class_hemo_finetune/ best_model.pth + train_record.json + some analysis
```

### Test set (201 images total, from models/5class/train_record.json)
```
AC3=41, F08D=40, F09D=40, F11D=40, NC1=40
```

### Model registry (defined in hemophilia_analysis.py, mirrored in analysis_utils.py)
```python
_MODEL_REGISTRY = {
    "5class":          ("models/5class",               5, CLASS_MAP),
    "3class_scratch":  ("models/3class_hemo",           3, CLASS_MAP_3),
    "3class_finetune": ("models/3class_hemo_finetune",  3, CLASS_MAP_3),
}
```

### FibrinCNN feature layer indices (model.features Sequential)
```
[0]  Conv2d(1→32,   5×5, pad=2)    Block 1
[1]  BatchNorm2d(32)
[2]  ReLU
[3]  MaxPool2d(2×2)                → 32×300×200
[4]  Conv2d(32→64,  5×5, pad=2)    Block 2
[5]  BatchNorm2d(64)
[6]  ReLU
[7]  MaxPool2d(2×2)                → 64×150×100
[8]  Conv2d(64→128, 3×3, pad=1)    Block 3
[9]  BatchNorm2d(128)
[10] ReLU
[11] MaxPool2d(2×2)                → 128×75×50
[12] Conv2d(128→256,3×3, pad=1)    Block 4
[13] BatchNorm2d(256)
[14] ReLU                          → 256×75×50  ← Grad-CAM hook target
[15] MaxPool2d(2×2)                → 256×37×25  ← activation analysis hook target
```

---

## Implementation Order

There is one hard dependency: `analysis_utils.py` must exist before Modules 2 and 3
can be implemented. Modules 5 and 6 have no inter-module dependencies.

```
Step 1  [required first]  analysis_utils.py        (plan_shared_refactor.md)

Step 2  [independent, any order or parallel]
        gradcam.py                                  (plan_06_gradcam.md)
        preprocessing_comparison.py                 (plan_05_preprocessing_comparison.md)

Step 3  [after analysis_utils.py]
        activation_analysis.py                      (plan_03_activation_analysis.md)
        misclassification_report.py                 (plan_02_misclassification.md)
```

---

## Output Directory Convention

Every script saves outputs under `<model_dir>/analysis/<module>/` by default.
For the 5-class model this produces:

```
models/5class/analysis/
  misclassification/
    per_image_results.csv
    experiment_accuracy.png
    slide_type_comparison.png
    confusion_matrix.png
    confidence_distribution.png
    misclassification_summary.txt
  activation/
    aggregate_activations.png
    correct_vs_incorrect_attention.png
    spatial_activation_stats.csv
    activation_overlay_examples.png
  preprocessing/
    method_visual_comparison.png
    preprocessing_comparison_table.csv
    preprocessing_comparison_report.txt
    preprocessing_accuracy_bars.png
  gradcam/
    class_representatives/
      <class>_gradcam_top3.png  (one per class)
    misclassified/
      <idx>_true-<true>_pred-<pred>.png
    class_comparison_grid.png
    gradcam_summary.txt
```

---

## Non-Interference Guarantees

- No existing files are modified. All outputs are new files in analysis/ subdirectories.
- `analysis_utils.py` is a new file; it imports from existing modules but does not
  modify them.
- Each script is standalone (runnable with `python <script>.py`) and importable as a
  module. There are no circular imports.
- All scripts work with any of the three trained models via `--model-type` argument.

---

## Verification Commands

```bash
conda activate fibrin

# After implementing analysis_utils.py:
python -c "from analysis_utils import run_inference_full, load_model_from_registry; print('OK')"

# Run each module on the 5-class model:
python misclassification_report.py --model-type 5class
python activation_analysis.py --model-type 5class
python preprocessing_comparison.py --model-type 5class
python gradcam.py --model-type 5class

# Cross-model comparison with Module 2:
for m in 5class 3class_scratch 3class_finetune; do
    python misclassification_report.py --model-type $m
done
```
