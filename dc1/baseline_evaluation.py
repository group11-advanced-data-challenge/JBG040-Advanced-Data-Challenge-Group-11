"""
baseline_evaluation.py

Evaluate a trained model on the test set using the project's BatchSampler.

UPDATED: --model_path is OPTIONAL.
- If --model_path is not provided, the script automatically loads the latest model file
  from --weights_dir (default: model_weights).

Recommended run (from inside dc1/):
  cd dc1
  python baseline_evaluation.py

Or specify a model explicitly:
  python baseline_evaluation.py --model_path model_weights/model_03_02_10_15.txt

Outputs:
- Accuracy
- Macro-F1
- Per-class precision/recall/F1 (classification report)
- Confusion matrix (printed + saved as PNG)
- Threshold sweep: coverage vs macro-F1 / accuracy (printed + saved as PNG)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Imports that work when running "python baseline_evaluation.py" inside dc1/
from net import Net
from image_dataset import ImageDataset
from batch_sampler import BatchSampler


@dataclass
class EvalResults:
    accuracy: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    class_report: str


def pick_device(force_cpu: bool = False) -> str:
    """Pick device similar to main.py logic."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_latest_model(weights_dir: Path) -> Path:
    """
    Find the most recently modified model file in weights_dir.
    Supports .txt (your current saving) and .pt.
    """
    candidates = list(weights_dir.glob("model_*.txt")) + list(weights_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in '{weights_dir}'.\n"
            "Run main.py first so it creates weights in model_weights/."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


@torch.no_grad()
def run_inference_with_batch_sampler(
    model: torch.nn.Module,
    sampler: BatchSampler,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: [N]
      y_pred: [N]
      y_prob: [N, C] softmax probabilities
    """
    model.eval()

    true_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []

    for x, y in sampler:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)                      # [B, C]
        probs = torch.softmax(logits, dim=1)   # [B, C]
        preds = torch.argmax(probs, dim=1)     # [B]

        true_list.append(y.detach().cpu().numpy())
        pred_list.append(preds.detach().cpu().numpy())
        prob_list.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(true_list, axis=0)
    y_pred = np.concatenate(pred_list, axis=0)
    y_prob = np.concatenate(prob_list, axis=0)

    return y_true, y_pred, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: Optional[List[str]] = None,
) -> EvalResults:
    """Compute accuracy, macro-F1, classification report, confusion matrix."""
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required. Install with:\n"
            "pip install scikit-learn"
        ) from e

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        target_names=class_names if class_names else None,
        digits=4,
        zero_division=0,
    )

    return EvalResults(
        accuracy=acc,
        macro_f1=macro_f1,
        confusion_matrix=cm.tolist(),
        class_report=report,
    )


def save_confusion_matrix_png(
    cm: np.ndarray,
    out_path: Path,
    class_names: Optional[List[str]] = None,
) -> None:
    """Save confusion matrix as a matplotlib figure (no seaborn)."""
    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax = fig.add_subplot(111)

    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    n = cm.shape[0]
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if class_names and len(class_names) == n:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selective prediction:
      keep a sample if max_prob >= threshold.

    Returns:
      coverage: [T]
      acc:      [T]  (on kept samples)
      macro_f1: [T]  (on kept samples)
    """
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required. Install with:\n"
            "pip install scikit-learn"
        ) from e

    max_prob = y_prob.max(axis=1)      # [N]
    y_pred = y_prob.argmax(axis=1)     # [N]

    coverages = []
    accs = []
    macro_f1s = []

    for t in thresholds:
        keep = max_prob >= t
        coverage = float(keep.mean()) if keep.size > 0 else 0.0
        coverages.append(coverage)

        if keep.sum() == 0:
            accs.append(float("nan"))
            macro_f1s.append(float("nan"))
            continue

        accs.append(float(accuracy_score(y_true[keep], y_pred[keep])))
        macro_f1s.append(float(f1_score(y_true[keep], y_pred[keep], average="macro")))

    return np.array(coverages), np.array(accs), np.array(macro_f1s)


def save_coverage_curve_png(
    thresholds: np.ndarray,
    coverage: np.ndarray,
    macro_f1: np.ndarray,
    acc: np.ndarray,
    out_path: Path,
) -> None:
    """Plot coverage vs threshold and metrics."""
    fig = plt.figure(figsize=(7, 5), dpi=150)
    ax = fig.add_subplot(111)

    ax.plot(thresholds, coverage, label="Coverage")
    ax.plot(thresholds, macro_f1, label="Macro-F1 (kept samples)")
    ax.plot(thresholds, acc, label="Accuracy (kept samples)")

    ax.set_title("Threshold sweep (Selective prediction)")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_class_names(s: Optional[str], n_classes: int) -> Optional[List[str]]:
    """Parse comma-separated class names."""
    if not s:
        return None
    names = [x.strip() for x in s.split(",")]
    if len(names) != n_classes:
        raise ValueError(f"--class_names must have exactly {n_classes} comma-separated names.")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation (BatchSampler version).")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to saved model state_dict. If not provided, the latest model in --weights_dir is used.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="model_weights",
        help="Directory containing saved models (default: model_weights).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing X_test.npy and Y_test.npy (default: data).",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Test batch size")
    parser.add_argument("--n_classes", type=int, default=6, help="Number of classes")
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help='Optional comma-separated class names, e.g. "A,B,C,D,E,F"',
    )
    parser.add_argument(
        "--balanced_test",
        action="store_true",
        help="Use balanced sampling on test set (normally keep this OFF).",
    )
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA/MPS is available")

    args = parser.parse_args()

    device = pick_device(force_cpu=args.force_cpu)
    print(f"@@@ Using device: {device}")

    data_dir = Path(args.data_dir)
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "Y_test.npy"

    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Could not find test data at:\n  {x_test_path}\n  {y_test_path}\n\n"
            "Tip: run from inside dc1/ so data_dir='data' works:\n"
            "  cd dc1\n  python baseline_evaluation.py"
        )

    test_dataset = ImageDataset(x_test_path, y_test_path)
    test_sampler = BatchSampler(batch_size=args.batch_size, dataset=test_dataset, balanced=args.balanced_test)

    # Decide which model file to load
    weights_dir = Path(args.weights_dir)
    if args.model_path is None:
        model_path = find_latest_model(weights_dir)
        print(f"@@@ --model_path not provided. Using latest model: {model_path}")
    else:
        model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Tip: if you run inside dc1/, you can omit --model_path and it will auto-pick the latest model."
        )

    model = Net(n_classes=args.n_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"@@@ Loaded model weights from: {model_path}")

    class_names = parse_class_names(args.class_names, args.n_classes)

    # Inference
    y_true, y_pred, y_prob = run_inference_with_batch_sampler(model, test_sampler, device=device)

    # Metrics
    results = compute_metrics(y_true, y_pred, n_classes=args.n_classes, class_names=class_names)

    print("\n=== Baseline evaluation results ===")
    print(f"Test Accuracy:  {results.accuracy:.4f}")
    print(f"Test Macro-F1:  {results.macro_f1:.4f}\n")
    print(results.class_report)

    cm = np.array(results.confusion_matrix, dtype=int)
    print("Confusion Matrix:\n", cm)

    # Output folder (inside dc1/artifacts if you run from dc1/)
    now = datetime.now()
    out_dir = Path("artifacts") / f"baseline_eval_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}"
    os.makedirs(out_dir, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(results.class_report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

    save_confusion_matrix_png(cm=cm, out_path=out_dir / "confusion_matrix.png", class_names=class_names)

    thresholds = np.linspace(0.0, 0.99, 50)
    coverage, acc_t, mf1_t = threshold_sweep(y_true=y_true, y_prob=y_prob, thresholds=thresholds)

    print("\n=== Threshold sweep (coverage / acc / macro-F1 on kept samples) ===")
    for t, cov, a, f in zip(thresholds[::10], coverage[::10], acc_t[::10], mf1_t[::10]):
        print(f"t={t:.2f} | coverage={cov:.3f} | acc={a:.3f} | macroF1={f:.3f}")

    save_coverage_curve_png(
        thresholds=thresholds,
        coverage=coverage,
        macro_f1=mf1_t,
        acc=acc_t,
        out_path=out_dir / "threshold_coverage_curve.png",
    )

    print(f"\n@@@ Saved evaluation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()