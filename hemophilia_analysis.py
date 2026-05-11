"""
hemophilia_analysis.py — Discrimination analysis for hemophilia classes.

Feature 1 – ROC curves (OVR within the selected subset):
    For each class C in the analysis set, treat C as positive and the other
    selected classes as negative.  Plots ROC curves and reports AUC.

Feature 2 – Annotated test images:
    Copies each test image for the selected classes into an output folder
    with the true label, predicted label, and per-class softmax probabilities
    drawn directly on the image.  Original filenames are preserved.

Feature 3 – Model comparison (--compare):
    Loads all three model types for which a best_model.pth exists, overlays
    their ROC curves on one figure, and prints an accuracy comparison table.

Usage:
    python hemophilia_analysis.py                              # 5-class model, both features
    python hemophilia_analysis.py --model-type 3class_scratch  # 3-class scratch model
    python hemophilia_analysis.py --compare                    # compare all available models
    python hemophilia_analysis.py --roc                        # ROC only
    python hemophilia_analysis.py --annotate                   # annotated images only
    python hemophilia_analysis.py --classes F08D F09D F11D NC1 # expand analysis set
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader import CLASS_MAP, CLASS_NAMES, filter_classes, load_split_from_record
from model import FibrinCNN
from preprocessing import ensure_landscape, make_preprocessor

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), "data", "endpoint10.db")
PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "data", "photos")

HEMOPHILIA_CLASSES: List[str] = ["F08D", "F09D", "F11D"]

CLASS_MAP_3: Dict[str, int] = {"F08D": 0, "F09D": 1, "F11D": 2}
CLASS_NAMES_3: List[str]    = ["F08D", "F09D", "F11D"]

# Maps model-type name → (model directory, num_classes, class_map to use)
_MODEL_REGISTRY: Dict[str, Tuple[str, int, dict]] = {
    "5class":          ("models/5class",               5, CLASS_MAP),
    "3class_scratch":  ("models/3class_hemo",           3, CLASS_MAP_3),
    "3class_finetune": ("models/3class_hemo_finetune",  3, CLASS_MAP_3),
}

GRAY_METHOD = "lab_l"
POOL_FACTOR = 10

# OpenCV BGR colours
_GREEN = (60, 180, 60)
_RED   = (50, 50, 200)
_WHITE = (255, 255, 255)
_BLACK = (0,   0,   0)
_FONT  = cv2.FONT_HERSHEY_SIMPLEX


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device,
               num_classes: int = 5) -> FibrinCNN:
    model = FibrinCNN(num_classes=num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def collect_predictions(
    model: FibrinCNN,
    test_df,
    photos_dir: str,
    device: torch.device,
    preprocessor,
    class_map: dict = CLASS_MAP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on every test image and collect results.

    Args:
        class_map: Maps Exp_Type string → integer label (use CLASS_MAP for
                   5-class models, CLASS_MAP_3 for 3-class models).

    Returns:
        true_labels: int32 array, shape (N,)   — integer class index
        probs:       float32 array, shape (N,K) — softmax probabilities
        img_indices: int32 array, shape (N,)   — 4-digit image index
    """
    true_labels, probs_list, img_indices = [], [], []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = os.path.join(photos_dir, f"{int(row['idx']):04d}.JPG")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            tensor = preprocessor(img_bgr).unsqueeze(0).to(device)
            logits = model(tensor)
            prob   = torch.softmax(logits, dim=1).cpu().numpy()[0]

            true_labels.append(class_map[row["Exp_Type"]])
            probs_list.append(prob)
            img_indices.append(int(row["idx"]))

    return (
        np.array(true_labels,  dtype=np.int32),
        np.array(probs_list,   dtype=np.float32),
        np.array(img_indices,  dtype=np.int32),
    )


# ── ROC utilities (no sklearn) ────────────────────────────────────────────────

def _roc_curve(y_true: np.ndarray,
               y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ROC curve anchored at (0,0) and (1,1), sorted by FPR."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC requires at least one positive and one negative example.")

    thresholds = np.sort(np.unique(y_score))[::-1]
    fprs, tprs = [0.0], [0.0]
    for t in thresholds:
        pred = y_score >= t
        tprs.append(int((pred & (y_true == 1)).sum()) / n_pos)
        fprs.append(int((pred & (y_true == 0)).sum()) / n_neg)
    fprs.append(1.0)
    tprs.append(1.0)

    fpr = np.array(fprs)
    tpr = np.array(tprs)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr))


# ── Feature 1: ROC curves ─────────────────────────────────────────────────────

def run_roc_analysis(
    true_labels: np.ndarray,
    probs: np.ndarray,
    analysis_classes: List[str] = HEMOPHILIA_CLASSES,
    output_path: str = "hemophilia_roc.png",
    class_map: dict = CLASS_MAP,
    ax: Optional[plt.Axes] = None,
    label_prefix: str = "",
    linestyle: str = "-",
) -> Dict[str, float]:
    """Plot OVR ROC curves for each class in analysis_classes.

    Positive  = images whose true class is C.
    Negative  = images whose true class is one of the other analysis_classes.
    Score     = model softmax probability for class C.

    Args:
        ax:           If provided, plot onto this existing Axes (used by --compare).
        label_prefix: String prepended to legend labels (used by --compare).
        linestyle:    Matplotlib line style (used by --compare).

    Returns:
        Dict mapping class name → AUC value.
    """
    class_indices = [class_map[c] for c in analysis_classes]
    mask = np.isin(true_labels, class_indices)
    sub_labels = true_labels[mask]
    sub_probs  = probs[mask]

    colours = plt.cm.tab10(np.linspace(0, 0.5, len(analysis_classes)))
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    aucs: Dict[str, float] = {}

    if not label_prefix:
        print(f"\n── ROC AUC  (OVR, subset: {', '.join(analysis_classes)}) ──")
        print(f"   Subset size: {mask.sum()} test images\n")

    for cls_name, colour in zip(analysis_classes, colours):
        cls_idx    = class_map[cls_name]
        y_true_bin = (sub_labels == cls_idx).astype(np.int32)
        y_score    = sub_probs[:, cls_idx]

        n_pos = int(y_true_bin.sum())
        n_neg = int(len(y_true_bin) - n_pos)
        fpr, tpr = _roc_curve(y_true_bin, y_score)
        auc = _auc(fpr, tpr)
        aucs[cls_name] = auc

        if not label_prefix:
            print(f"   {cls_name:5s}  n_pos={n_pos:3d}  n_neg={n_neg:3d}  AUC = {auc:.4f}")

        legend_label = f"{label_prefix}{cls_name}  (AUC={auc:.3f})"
        ax.plot(fpr, tpr, color=colour, lw=2.5, linestyle=linestyle,
                label=legend_label)

    if own_fig:
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"OVR ROC — Hemophilia Discrimination\n({', '.join(analysis_classes)})",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        fig.tight_layout()
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"\n   ROC plot saved → {output_path}")

    return aucs


# ── Feature 2: Annotated images ───────────────────────────────────────────────

def _draw_banner(
    img: np.ndarray,
    true_cls: str,
    pred_cls: str,
    probs: np.ndarray,
    inv_class_map: Dict[int, str],
) -> np.ndarray:
    """Draw a semi-transparent annotation banner at the top of img.

    Line 1 (large): TRUE: <class>   PRED: <class>   [CORRECT / WRONG]
    Line 2 (small): per-class softmax probabilities

    Text is green when the prediction is correct, red otherwise.
    """
    img = img.copy()
    h, w = img.shape[:2]

    banner_h = max(260, h // 13)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    correct = true_cls == pred_cls
    accent  = _GREEN if correct else _RED
    verdict = "CORRECT" if correct else "WRONG"

    scale1 = w / 1800.0
    scale2 = w / 2500.0
    thick1 = max(3, round(scale1 * 2.4))
    thick2 = max(2, round(scale2 * 2.4))

    line1 = f"TRUE: {true_cls}   PRED: {pred_cls}   [{verdict}]"
    y1 = int(banner_h * 0.47)
    cv2.putText(img, line1, (30, y1), _FONT, scale1, accent, thick1, cv2.LINE_AA)

    prob_parts = [f"{inv_class_map[i]}: {probs[i]:.3f}" for i in range(len(probs))]
    line2 = "    ".join(prob_parts)
    y2 = int(banner_h * 0.87)
    cv2.putText(img, line2, (30, y2), _FONT, scale2, _WHITE, thick2, cv2.LINE_AA)

    return img


def run_annotate_images(
    true_labels: np.ndarray,
    probs: np.ndarray,
    img_indices: np.ndarray,
    photos_dir: str,
    output_dir: str = "annotated_test",
    filter_cls: Optional[List[str]] = None,
    inv_class_map: Optional[Dict[int, str]] = None,
) -> None:
    """Write annotated copies of test images to output_dir.

    Args:
        filter_cls:    If given, only annotate images from these classes.
        inv_class_map: Maps integer label → class name string (for banner text).
                       Defaults to the global 5-class inverse map.
    """
    if inv_class_map is None:
        inv_class_map = {v: k for k, v in CLASS_MAP.items()}

    if filter_cls is not None:
        keep = np.isin(true_labels, [inv_class_map_inv(inv_class_map, c)
                                     for c in filter_cls])
    else:
        keep = np.ones(len(true_labels), dtype=bool)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    indices = np.where(keep)[0]
    n = len(indices)

    cls_label = ", ".join(filter_cls) if filter_cls else "all classes"
    print(f"\n── Annotated images  ({n} images, {cls_label}) ──")
    print(f"   Output folder: {os.path.abspath(output_dir)}/\n")

    written = 0
    for rank, i in enumerate(indices, 1):
        img_idx  = img_indices[i]
        fname    = f"{img_idx:04d}.JPG"
        src_path = os.path.join(photos_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            print(f"   WARNING: could not read {src_path} — skipping.")
            continue
        img_bgr = ensure_landscape(img_bgr)

        true_cls = inv_class_map[int(true_labels[i])]
        pred_cls = inv_class_map[int(probs[i].argmax())]
        annotated = _draw_banner(img_bgr, true_cls, pred_cls, probs[i], inv_class_map)

        cv2.imwrite(dst_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
        print(f"   [{rank:3d}/{n}] {fname}  true={true_cls}  pred={pred_cls}"
              f"  {'OK' if true_cls == pred_cls else 'MISMATCH'}")

    print(f"\n   Done — {written}/{n} images written.")


def inv_class_map_inv(inv_map: Dict[int, str], cls_name: str) -> int:
    """Look up integer label for cls_name in an inverted map."""
    for k, v in inv_map.items():
        if v == cls_name:
            return k
    raise KeyError(cls_name)


# ── Feature 3: Model comparison ───────────────────────────────────────────────

def run_compare(
    analysis_classes: List[str],
    output_path: str,
    device: torch.device,
    preprocessor,
    db_path: str = DB_PATH,
) -> None:
    """Overlay ROC curves from all available models on one figure and print
    an accuracy comparison table."""
    linestyles = ["-", "--", ":"]
    model_labels = {"5class": "5-class",
                    "3class_scratch": "3-class (scratch)",
                    "3class_finetune": "3-class (finetune)"}

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    rows = []
    for (mtype, ls) in zip(_MODEL_REGISTRY, linestyles):
        model_dir, num_classes, cmap = _MODEL_REGISTRY[mtype]
        model_path  = os.path.join(model_dir, "best_model.pth")
        record_path = os.path.join(model_dir, "train_record.json")

        if not os.path.exists(model_path):
            print(f"   Skipping {mtype}: {model_path} not found.")
            continue

        print(f"\n   Loading {mtype} …")
        model = load_model(model_path, device, num_classes=num_classes)

        # Reconstruct test set — for 3-class models use the 5-class record
        # filtered to hemophilia classes; for 5-class use the 5-class record directly
        rec = record_path if os.path.exists(record_path) else "models/5class/train_record.json"
        _, test_df = load_split_from_record(rec, db_path)
        if num_classes == 3:
            test_df = filter_classes(test_df, analysis_classes)

        inv_cmap = {v: k for k, v in cmap.items()}
        true_labels, probs, _ = collect_predictions(
            model, test_df, PHOTOS_DIR, device, preprocessor, class_map=cmap
        )

        aucs = run_roc_analysis(
            true_labels, probs,
            analysis_classes=analysis_classes,
            class_map=cmap,
            ax=ax,
            label_prefix=f"{model_labels[mtype]}: ",
            linestyle=ls,
        )

        # Overall accuracy on hemophilia subset
        class_indices = [cmap[c] for c in analysis_classes]
        mask = np.isin(true_labels, class_indices)
        acc = (true_labels[mask] == probs[mask].argmax(axis=1)).mean()
        rows.append((model_labels[mtype], aucs, float(acc)))

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"OVR ROC Comparison — {', '.join(analysis_classes)}", fontsize=13
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    fig.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n   Comparison ROC plot saved → {output_path}")

    # Print accuracy table
    col_w = max(len(c) for c in analysis_classes) + 6
    header = f"{'Model':<24}" + "".join(f"{c+'-AUC':>{col_w}}" for c in analysis_classes) + f"{'Acc':>10}"
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")
    for label, aucs, acc in rows:
        row = f"{label:<24}" + "".join(f"{aucs.get(c, float('nan')):>{col_w}.3f}" for c in analysis_classes) + f"{acc:>10.3f}"
        print(row)
    print(f"{'─'*len(header)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hemophilia class discrimination analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type",
                        choices=list(_MODEL_REGISTRY.keys()),
                        default="5class",
                        help="Which trained model to use (default: 5class).")
    parser.add_argument("--model-dir", default=None, metavar="PATH",
                        help="Override the model directory (takes precedence over --model-type).")
    parser.add_argument("--roc",      action="store_true",
                        help="Run ROC/AUC analysis only.")
    parser.add_argument("--annotate", action="store_true",
                        help="Write annotated images only.")
    parser.add_argument("--compare",  action="store_true",
                        help="Compare all available models (overrides --roc/--annotate).")
    parser.add_argument("--roc-output", default=None, metavar="PATH",
                        help="Path for the ROC plot (default: <model-dir>/hemophilia_roc.png).")
    parser.add_argument("--annotate-dir", default=None, metavar="DIR",
                        help="Output folder for annotated images (default: <model-dir>/annotated_test).")
    parser.add_argument("--classes", nargs="+", default=HEMOPHILIA_CLASSES,
                        choices=list(CLASS_MAP.keys()), metavar="CLASS",
                        help=f"Classes to include (default: {' '.join(HEMOPHILIA_CLASSES)}).")
    parser.add_argument("--db", default=DB_PATH, metavar="PATH",
                        help="Path to SQLite database (default: data/endpoint10.db).")
    args = parser.parse_args()

    # Resolve model configuration
    if args.model_dir:
        # Manual override — infer num_classes from model file if needed
        model_dir = args.model_dir
        # Try to match against registry for num_classes/class_map
        matched = next(
            ((nd, nc, cm) for _, (nd, nc, cm) in _MODEL_REGISTRY.items()
             if nd == model_dir), None
        )
        if matched:
            _, num_classes, cmap = matched
        else:
            # Fallback: assume 5-class
            num_classes, cmap = 5, CLASS_MAP
    else:
        model_dir, num_classes, cmap = _MODEL_REGISTRY[args.model_type]

    model_path  = os.path.join(model_dir, "best_model.pth")
    record_path = os.path.join(model_dir, "train_record.json")

    roc_output   = args.roc_output   or os.path.join(model_dir, "hemophilia_roc.png")
    annotate_dir = args.annotate_dir or os.path.join(model_dir, "annotated_test")

    device = torch.device("cpu")
    preprocessor = make_preprocessor(gray_method=GRAY_METHOD, pool_factor=POOL_FACTOR)

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        compare_out = args.roc_output or "hemophilia_roc_compare.png"
        run_compare(args.classes, compare_out, device, preprocessor, db_path=args.db)
        return

    # ── Single-model mode ─────────────────────────────────────────────────────
    run_roc = not args.annotate
    run_ann = not args.roc
    if args.roc and args.annotate:
        run_roc = run_ann = True

    print(f"Loading model ({args.model_type}) …")
    model = load_model(model_path, device, num_classes=num_classes)

    # Reconstruct test split — 3-class models reuse the 5-class record, filtered
    rec = record_path if os.path.exists(record_path) else "models/5class/train_record.json"
    print(f"Loading test split from {rec} …")
    _, test_df = load_split_from_record(rec, args.db)
    if num_classes == 3:
        test_df = filter_classes(test_df, args.classes)

    print(f"Running inference ({len(test_df)} test images) …")
    true_labels, probs, img_indices = collect_predictions(
        model, test_df, PHOTOS_DIR, device, preprocessor, class_map=cmap
    )
    print(f"   Done.")

    inv_cmap = {v: k for k, v in cmap.items()}

    if run_roc:
        run_roc_analysis(true_labels, probs,
                         analysis_classes=args.classes,
                         output_path=roc_output,
                         class_map=cmap)

    if run_ann:
        run_annotate_images(true_labels, probs, img_indices, PHOTOS_DIR,
                            output_dir=annotate_dir,
                            filter_cls=args.classes,
                            inv_class_map=inv_cmap)


if __name__ == "__main__":
    main()
