from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dc1.baseline_evaluation import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    find_latest_model,
    parse_class_names,
    pick_device,
    run_inference_with_batch_sampler,
    save_confusion_matrix_png,
    save_coverage_curve_png,
    threshold_sweep,
)
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net


def _resolve_path(p: Optional[str], default: Path, project_root: Path) -> Path:
    if p is None:
        return default
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _make_artifact_dir(base_dir: Path) -> Path:
    now = datetime.now()
    out_dir = base_dir / "artifacts" / (
        f"experiment_c_triage_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}"
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_threshold_table_csv(
    out_path: Path,
    thresholds: np.ndarray,
    coverage: np.ndarray,
    acc: np.ndarray,
    macro_f1: np.ndarray,
    total_samples: int,
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "coverage",
                "kept_count",
                "abstained_count",
                "accuracy_kept",
                "macro_f1_kept",
            ]
        )
        for t, cov, a, f1 in zip(thresholds, coverage, acc, macro_f1):
            kept = int(round(float(cov) * total_samples))
            abstained = total_samples - kept
            writer.writerow(
                [
                    f"{float(t):.6f}",
                    f"{float(cov):.6f}",
                    kept,
                    abstained,
                    f"{float(a):.6f}",
                    f"{float(f1):.6f}",
                ]
            )


def _build_threshold_rows(
    thresholds: np.ndarray,
    coverage: np.ndarray,
    acc: np.ndarray,
    macro_f1: np.ndarray,
    total_samples: int,
) -> List[Dict[str, float | int]]:
    rows: List[Dict[str, float | int]] = []
    for t, cov, a, f1 in zip(thresholds, coverage, acc, macro_f1):
        kept = int(round(float(cov) * total_samples))
        abstained = total_samples - kept
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(cov),
                "kept_count": kept,
                "abstained_count": abstained,
                "accuracy_kept": float(a),
                "macro_f1_kept": float(f1),
            }
        )
    return rows


def _select_best_threshold(
    thresholds: np.ndarray,
    coverage: np.ndarray,
    acc: np.ndarray,
    macro_f1: np.ndarray,
) -> int:
    """
    Select the operating point by highest macro-F1.
    Tie-breakers:
    1) higher coverage
    2) lower threshold
    """
    best_idx = 0
    for i in range(1, len(thresholds)):
        better_f1 = macro_f1[i] > macro_f1[best_idx]
        same_f1 = np.isclose(macro_f1[i], macro_f1[best_idx])
        better_cov = coverage[i] > coverage[best_idx]
        same_cov = np.isclose(coverage[i], coverage[best_idx])
        lower_t = thresholds[i] < thresholds[best_idx]

        if better_f1 or (same_f1 and better_cov) or (same_f1 and same_cov and lower_t):
            best_idx = i
    return best_idx


def _save_selected_threshold_outputs(
    out_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    class_names: Optional[List[str]],
) -> Dict[str, object]:
    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    keep = max_prob >= threshold

    kept_count = int(np.sum(keep))
    total_count = int(len(y_true))
    abstained_count = total_count - kept_count
    n_classes = y_prob.shape[1]

    summary: Dict[str, object] = {
        "selected_threshold": float(threshold),
        "total_count": total_count,
        "kept_count": kept_count,
        "abstained_count": abstained_count,
        "coverage": float(kept_count / total_count) if total_count else 0.0,
    }

    if kept_count == 0:
        summary.update(
            {
                "accuracy_kept": 0.0,
                "macro_f1_kept": 0.0,
                "confusion_matrix_kept": [[0 for _ in range(n_classes)] for _ in range(n_classes)],
            }
        )
        return summary

    yt = y_true[keep]
    yp = y_pred[keep]
    cm = compute_confusion_matrix(yt, yp, n_classes=n_classes)
    acc = compute_accuracy(yt, yp)
    macro_f1 = compute_macro_f1(cm)

    summary.update(
        {
            "accuracy_kept": float(acc),
            "macro_f1_kept": float(macro_f1),
            "confusion_matrix_kept": cm.tolist(),
        }
    )

    save_confusion_matrix_png(
        cm=cm,
        out_path=out_dir / "confusion_matrix_selected_threshold.png",
        class_names=class_names,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment C: AI triage via selective prediction / abstention."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to saved model state_dict. If omitted, latest model in --weights_dir is used.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Directory containing saved model weights. If omitted, uses <project_root>/dc1/model_weights.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing X_test.npy and Y_test.npy. If omitted, uses <project_root>/dc1/data.",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Evaluation batch size.")
    parser.add_argument("--n_classes", type=int, default=6, help="Number of classes.")
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help='Optional comma-separated class names, e.g. "NoFinding,Pneumothorax,..."',
    )
    parser.add_argument(
        "--balanced_test",
        action="store_true",
        help="Use balanced sampling on test set. Normally keep this OFF.",
    )
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA/MPS is available.")
    parser.add_argument("--threshold_min", type=float, default=0.0, help="Minimum threshold for sweep.")
    parser.add_argument("--threshold_max", type=float, default=0.99, help="Maximum threshold for sweep.")
    parser.add_argument("--threshold_steps", type=int, default=50, help="Number of threshold values in sweep.")
    parser.add_argument(
        "--selected_threshold",
        type=float,
        default=None,
        help="Optional explicit threshold for selected-threshold outputs. If omitted, best macro-F1 threshold is used.",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]  # .../dc1
    project_root = base_dir.parent

    device = pick_device(force_cpu=args.force_cpu)
    print(f"@@@ Using device: {device}")

    data_dir = _resolve_path(args.data_dir, base_dir / "data", project_root)
    x_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "Y_test.npy"
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Could not find test data at:\n  {x_test_path}\n  {y_test_path}\n\n"
            "Run from the project root with:\n"
            "  python -m dc1.experiments.experiment_c_triage"
        )

    test_dataset = ImageDataset(x_test_path, y_test_path)
    test_sampler = BatchSampler(
        batch_size=args.batch_size,
        dataset=test_dataset,
        balanced=args.balanced_test,
    )

    weights_dir = _resolve_path(args.weights_dir, base_dir / "model_weights", project_root)
    if args.model_path is None:
        model_path = find_latest_model(weights_dir)
        print(f"@@@ --model_path not provided. Using latest model: {model_path}")
    else:
        model_path = _resolve_path(args.model_path, base_dir / "model_weights", project_root)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = Net(n_classes=args.n_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"@@@ Loaded model weights from: {model_path}")

    class_names = parse_class_names(args.class_names, args.n_classes)

    # Run inference once; all threshold sweeps operate on stored probabilities.
    y_true, y_pred, y_prob = run_inference_with_batch_sampler(model, test_sampler, device=device)

    if y_prob.ndim != 2 or y_prob.shape[0] != len(y_true):
        raise RuntimeError(
            "Inference output has inconsistent shape. Expected y_prob to have shape (N, C)."
        )

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    coverage, acc_kept, macro_f1_kept = threshold_sweep(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
    )

    out_dir = _make_artifact_dir(base_dir)

    save_coverage_curve_png(
        thresholds=thresholds,
        coverage=coverage,
        macro_f1=macro_f1_kept,
        acc=acc_kept,
        out_path=out_dir / "coverage_vs_score.png",
    )

    total_samples = len(y_true)
    rows = _build_threshold_rows(
        thresholds=thresholds,
        coverage=coverage,
        acc=acc_kept,
        macro_f1=macro_f1_kept,
        total_samples=total_samples,
    )
    _save_threshold_table_csv(
        out_path=out_dir / "threshold_metrics.csv",
        thresholds=thresholds,
        coverage=coverage,
        acc=acc_kept,
        macro_f1=macro_f1_kept,
        total_samples=total_samples,
    )
    (out_dir / "threshold_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    baseline_cm = compute_confusion_matrix(y_true, y_pred, n_classes=args.n_classes)
    baseline_summary = {
        "accuracy": float(compute_accuracy(y_true, y_pred)),
        "macro_f1": float(compute_macro_f1(baseline_cm)),
        "coverage": 1.0,
        "kept_count": int(total_samples),
        "abstained_count": 0,
        "confusion_matrix": baseline_cm.tolist(),
    }

    if args.selected_threshold is not None:
        selected_threshold = float(args.selected_threshold)
    else:
        best_idx = _select_best_threshold(
            thresholds=thresholds,
            coverage=coverage,
            acc=acc_kept,
            macro_f1=macro_f1_kept,
        )
        selected_threshold = float(thresholds[best_idx])

    selected_summary = _save_selected_threshold_outputs(
        out_dir=out_dir,
        y_true=y_true,
        y_prob=y_prob,
        threshold=selected_threshold,
        class_names=class_names,
    )
    (out_dir / "selected_threshold.json").write_text(
        json.dumps(selected_summary, indent=2),
        encoding="utf-8",
    )

    metrics_summary = {
        "experiment": "Experiment C: AI Triage System",
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "device": device,
        "n_classes": int(args.n_classes),
        "batch_size": int(args.batch_size),
        "balanced_test": bool(args.balanced_test),
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "threshold_steps": int(args.threshold_steps),
        "baseline": baseline_summary,
        "selected_threshold": selected_summary,
    }
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Experiment C: AI Triage System ===")
    print(f"Baseline coverage:   {baseline_summary['coverage']:.3f}")
    print(f"Baseline accuracy:   {baseline_summary['accuracy']:.4f}")
    print(f"Baseline macro-F1:   {baseline_summary['macro_f1']:.4f}")
    print("\nSelected threshold summary")
    print(f"Threshold:           {selected_summary['selected_threshold']:.4f}")
    print(f"Coverage:            {selected_summary['coverage']:.3f}")
    print(f"Kept count:          {selected_summary['kept_count']}")
    print(f"Abstained count:     {selected_summary['abstained_count']}")
    print(f"Accuracy (kept):     {selected_summary['accuracy_kept']:.4f}")
    print(f"Macro-F1 (kept):     {selected_summary['macro_f1_kept']:.4f}")
    print(f"\n@@@ Saved experiment artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
