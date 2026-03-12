import os
import glob
import csv
import shutil
from datetime import datetime
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from net import Net
from image_dataset import ImageDataset
from gradcam import GradCAM


# ======================================================
# Class mapping
# ======================================================

CLASS_NAME_MAP = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax",
}


# ======================================================
# Utility functions
# ======================================================

def get_checkpoint_path(manual_path=None, model_dir=None):
    """
    Priority:
    1. Use manual_path if provided.
    2. Otherwise, automatically select the newest checkpoint from model_dir.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_dir is None:
        model_dir = os.path.join(script_dir, "model_weights")
    model_dir = os.path.abspath(model_dir)

    if manual_path is not None and str(manual_path).strip() != "":
        checkpoint_path = os.path.abspath(manual_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Manual checkpoint path not found: {checkpoint_path}")
        mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        print(f"[Checkpoint] Manual checkpoint selected: {checkpoint_path}")
        print(f"[Checkpoint] Last modified: {mtime}")
        return checkpoint_path

    candidates = glob.glob(os.path.join(model_dir, "*.txt"))
    candidates += glob.glob(os.path.join(model_dir, "*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in: {model_dir}")

    checkpoint_path = max(candidates, key=os.path.getmtime)
    mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    print(f"[Checkpoint] Latest checkpoint selected automatically: {checkpoint_path}")
    print(f"[Checkpoint] Last modified: {mtime}")
    return checkpoint_path


def load_test_dataset():
    """
    Load the test dataset.

    Important:
    ImageDataset expects file paths to .npy files, not loaded arrays.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    x_test_path = os.path.join(script_dir, "data", "X_test.npy")
    y_test_path = os.path.join(script_dir, "data", "Y_test.npy")

    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"X_test.npy not found at: {x_test_path}")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Y_test.npy not found at: {y_test_path}")

    return ImageDataset(x_test_path, y_test_path)


def tensor_to_display_image(image_tensor):
    """
    Convert a tensor image to numpy for display.

    Expected input shape:
        [C, H, W] or [1, H, W]

    Returned shape:
        [H, W]
    """
    return image_tensor.squeeze().detach().cpu().numpy()


def get_case_type(true_class, pred_class):
    return "correct_cases" if true_class == pred_class else "wrong_cases"


def get_pair_folder(save_root, case_type, true_class, pred_class):
    """
    Example:
        gradcam_results/wrong_cases/true_3_pred_2/
        gradcam_results/correct_cases/true_4_pred_4/
    """
    folder = os.path.join(
        save_root,
        case_type,
        f"true_{true_class}_pred_{pred_class}",
    )
    os.makedirs(folder, exist_ok=True)
    return folder


def get_class_label(class_idx):
    """
    Return label string such as:
        class 3 (No Finding)
    """
    return f"class {class_idx} ({CLASS_NAME_MAP.get(class_idx, 'Unknown')})"


# ======================================================
# Grad-CAM figure saving
# ======================================================

def save_gradcam_figure(
    image_np,
    pred_cam,
    true_cam,
    true_class,
    pred_class,
    pred_prob,
    true_prob,
    sample_index,
    save_root,
):
    """
    Save one Grad-CAM figure into the appropriate case/pair folder.
    """
    case_type = get_case_type(true_class, pred_class)
    pair_folder = get_pair_folder(save_root, case_type, true_class, pred_class)
    correctness = "correct" if true_class == pred_class else "wrong"

    file_name = (
        f"sample_{sample_index:04d}"
        f"_true_{true_class}"
        f"_pred_{pred_class}"
        f"_{correctness}.png"
    )
    output_path = os.path.join(pair_folder, file_name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title(f"Original Image\nTrue: {get_class_label(true_class)}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(image_np, cmap="gray")
    axes[1].imshow(pred_cam, cmap="jet", alpha=0.4)
    axes[1].set_title(
        f"Predicted-class CAM\nPred: {get_class_label(pred_class)}\nProb: {pred_prob:.3f}",
        fontsize=11,
    )
    axes[1].axis("off")

    axes[2].imshow(image_np, cmap="gray")
    axes[2].imshow(true_cam, cmap="jet", alpha=0.4)
    axes[2].set_title(
        f"True-class CAM\nTrue: {get_class_label(true_class)}\nProb: {true_prob:.3f}",
        fontsize=11,
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ======================================================
# CSV and report saving
# ======================================================

def save_prediction_summary_csv(records, report_dir):
    csv_path = os.path.join(report_dir, "prediction_summary.csv")
    fieldnames = [
        "sample_index",
        "true_class",
        "true_class_name",
        "pred_class",
        "pred_class_name",
        "correctness",
        "pred_prob",
        "true_prob",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow(
                {
                    "sample_index": r["sample_index"],
                    "true_class": r["true_class"],
                    "true_class_name": CLASS_NAME_MAP.get(r["true_class"], "Unknown"),
                    "pred_class": r["pred_class"],
                    "pred_class_name": CLASS_NAME_MAP.get(r["pred_class"], "Unknown"),
                    "correctness": "correct"
                    if r["true_class"] == r["pred_class"]
                    else "wrong",
                    "pred_prob": f"{r['pred_prob']:.6f}",
                    "true_prob": f"{r['true_prob']:.6f}",
                }
            )

    return csv_path


def save_selection_summary_csv(selected_entries, report_dir):
    csv_path = os.path.join(report_dir, "selection_summary.csv")
    fieldnames = [
        "filename",
        "case_type",
        "sample_index",
        "true_class",
        "true_class_name",
        "pred_class",
        "pred_class_name",
        "pair",
        "pair_label",
        "pred_prob",
        "true_prob",
        "output_path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for e in selected_entries:
            true_name = CLASS_NAME_MAP.get(e["true_class"], "Unknown")
            pred_name = CLASS_NAME_MAP.get(e["pred_class"], "Unknown")

            writer.writerow(
                {
                    "filename": os.path.basename(e["output_path"]),
                    "case_type": e["case_type"],
                    "sample_index": e["sample_index"],
                    "true_class": e["true_class"],
                    "true_class_name": true_name,
                    "pred_class": e["pred_class"],
                    "pred_class_name": pred_name,
                    "pair": f"true_{e['true_class']}_pred_{e['pred_class']}",
                    "pair_label": f"{true_name} -> {pred_name}",
                    "pred_prob": f"{e['pred_prob']:.6f}",
                    "true_prob": f"{e['true_prob']:.6f}",
                    "output_path": e["output_path"],
                }
            )

    return csv_path


def save_annotation_table(selected_entries, report_dir):
    """
    Save an annotation table template for manual review.

    This table is intentionally non-medical and focuses on attention behavior.
    """
    csv_path = os.path.join(report_dir, "annotation_table.csv")
    fieldnames = [
        "filename",
        "case_type",
        "sample_index",
        "true_class",
        "true_class_name",
        "pred_class",
        "pred_class_name",
        "pair",
        "pair_label",
        "pred_prob",
        "true_prob",
        "main_attention_region",
        "attention_outside_lung",
        "marker_or_text_overlap",
        "device_or_wires_overlap",
        "border_or_corner_attention",
        "shortcut_suspected",
        "notes",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for e in selected_entries:
            true_name = CLASS_NAME_MAP.get(e["true_class"], "Unknown")
            pred_name = CLASS_NAME_MAP.get(e["pred_class"], "Unknown")

            writer.writerow(
                {
                    "filename": os.path.basename(e["output_path"]),
                    "case_type": e["case_type"],
                    "sample_index": e["sample_index"],
                    "true_class": e["true_class"],
                    "true_class_name": true_name,
                    "pred_class": e["pred_class"],
                    "pred_class_name": pred_name,
                    "pair": f"true_{e['true_class']}_pred_{e['pred_class']}",
                    "pair_label": f"{true_name} -> {pred_name}",
                    "pred_prob": f"{e['pred_prob']:.6f}",
                    "true_prob": f"{e['true_prob']:.6f}",
                    "main_attention_region": "",
                    "attention_outside_lung": "",
                    "marker_or_text_overlap": "",
                    "device_or_wires_overlap": "",
                    "border_or_corner_attention": "",
                    "shortcut_suspected": "",
                    "notes": "",
                }
            )

    return csv_path


def save_text_report(
    checkpoint_path,
    n_classes,
    all_records,
    selected_entries,
    correct_per_pair,
    wrong_per_pair,
    report_dir,
):
    total = len(all_records)
    correct = sum(1 for r in all_records if r["true_class"] == r["pred_class"])
    wrong = total - correct
    accuracy = correct / total if total > 0 else 0.0

    wrong_pairs = Counter()
    for r in all_records:
        if r["true_class"] != r["pred_class"]:
            wrong_pairs[(r["true_class"], r["pred_class"])] += 1

    report_path = os.path.join(report_dir, "report_summary.txt")

    with open(report_path, "w") as f:
        f.write("Grad-CAM Export and Prediction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint used: {checkpoint_path}\n")
        f.write(f"Number of classes: {n_classes}\n")
        f.write(f"Total test samples evaluated: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Wrong predictions: {wrong}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write("Class name mapping\n")
        f.write("-" * 25 + "\n")
        for k in sorted(CLASS_NAME_MAP.keys()):
            f.write(f"Class {k}: {CLASS_NAME_MAP[k]}\n")

        f.write("\nExport settings\n")
        f.write("-" * 25 + "\n")
        f.write(f"Correct samples saved per pair: {correct_per_pair}\n")
        f.write(f"Wrong samples saved per pair: {wrong_per_pair}\n")
        f.write(f"Total Grad-CAM figures saved: {len(selected_entries)}\n\n")

        f.write("Top wrong confusion pairs\n")
        f.write("-" * 25 + "\n")
        if wrong_pairs:
            for (t, p), count in wrong_pairs.most_common(10):
                true_name = CLASS_NAME_MAP.get(t, "Unknown")
                pred_name = CLASS_NAME_MAP.get(p, "Unknown")
                f.write(f"True {t} ({true_name}) -> Pred {p} ({pred_name}): {count}\n")
        else:
            f.write("No wrong confusion pairs found.\n")

    return report_path


# ======================================================
# Plotting functions
# ======================================================

def plot_confusion_matrix(all_records, n_classes, report_dir):
    """
    Save a prettier confusion matrix with real class names.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for r in all_records:
        cm[r["true_class"], r["pred_class"]] += 1

    class_names = [CLASS_NAME_MAP.get(i, str(i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix", fontsize=15, pad=12)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    threshold = cm.max() * 0.5 if cm.max() > 0 else 0

    for i in range(n_classes):
        for j in range(n_classes):
            value = cm[i, j]
            color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    out_path = os.path.join(report_dir, "confusion_matrix_pretty.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_correct_wrong_counts(all_records, report_dir):
    """
    Save a prettier correct-vs-wrong bar chart.
    """
    correct_count = sum(1 for r in all_records if r["true_class"] == r["pred_class"])
    wrong_count = len(all_records) - correct_count

    labels = ["Correct", "Wrong"]
    values = [correct_count, wrong_count]
    colors = ["#5B8E7D", "#C06C84"]

    fig, ax = plt.subplots(figsize=(6, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_title("Correct vs Wrong Predictions", fontsize=14, pad=10)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    out_path = os.path.join(report_dir, "correct_vs_wrong_counts_pretty.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_top_confusion_pairs(all_records, report_dir, top_k=10):
    """
    Plot the most frequent wrong confusion pairs.
    This is very useful for deciding which Grad-CAM cases to inspect first.
    """
    pair_counter = Counter()

    for r in all_records:
        if r["true_class"] != r["pred_class"]:
            pair_counter[(r["true_class"], r["pred_class"])] += 1

    top_pairs = pair_counter.most_common(top_k)

    fig, ax = plt.subplots(figsize=(12, 6))

    if top_pairs:
        labels = [
            f"{CLASS_NAME_MAP.get(t, t)} → {CLASS_NAME_MAP.get(p, p)}"
            for (t, p), _ in top_pairs
        ]
        values = [count for _, count in top_pairs]

        bars = ax.bar(labels, values, color="#4C78A8", width=0.7)

        ax.set_title("Top Wrong Confusion Pairs", fontsize=15, pad=12)
        ax.set_xlabel("True class → Predicted class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="x", labelrotation=30, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    else:
        ax.text(0.5, 0.5, "No wrong predictions found.", ha="center", va="center", fontsize=13)
        ax.axis("off")

    plt.tight_layout()

    out_path = os.path.join(report_dir, "top_confusion_pairs_pretty.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ======================================================
# Main workflow
# ======================================================

def main():
    print("RUNNING THE FULL REPORT VERSION OF run_gradcam.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional manual checkpoint:
    # manual_checkpoint_path = "model_weights/model_03_11_21_03.txt"
    manual_checkpoint_path = None

    n_classes = 6

    # Export limits
    correct_samples_per_pair = 2
    wrong_samples_per_pair = 3

    # Optional: clean old outputs before running
    clean_previous_outputs = False

    # Folder setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_root = os.path.join(script_dir, "gradcam_results")
    report_dir = os.path.join(save_root, "reports")

    if clean_previous_outputs:
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    checkpoint_path = get_checkpoint_path(manual_path=manual_checkpoint_path)

    print(f"[run_gradcam] Final checkpoint used: {checkpoint_path}")
    print(f"Grad-CAM output folder: {save_root}")
    print(f"Report output folder: {report_dir}")

    # Load model
    model = Net(n_classes=n_classes).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load dataset
    test_dataset = load_test_dataset()

    # Grad-CAM target layer
    target_layer = model.cnn_layers[10]
    gradcam = GradCAM(model, target_layer)

    # --------------------------------------------------
    # First pass: inference over the whole test set
    # --------------------------------------------------
    all_records = []
    selected_records = []

    saved_correct_count_per_pair = defaultdict(int)
    saved_wrong_count_per_pair = defaultdict(int)

    total_samples = len(test_dataset)
    print(f"Scanning {total_samples} test samples...")

    with torch.no_grad():
        for sample_index in range(total_samples):
            input_image, label = test_dataset[sample_index]

            if torch.is_tensor(label):
                true_class = int(label.item())
            else:
                true_class = int(label)

            batched = input_image.unsqueeze(0).to(device)
            logits = model(batched)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_class = int(np.argmax(probs))
            pred_prob = float(probs[pred_class])
            true_prob = float(probs[true_class])

            record = {
                "sample_index": sample_index,
                "true_class": true_class,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
                "true_prob": true_prob,
            }
            all_records.append(record)

            pair_key = (true_class, pred_class)

            if true_class == pred_class:
                if saved_correct_count_per_pair[pair_key] < correct_samples_per_pair:
                    selected_records.append(record)
                    saved_correct_count_per_pair[pair_key] += 1
            else:
                if saved_wrong_count_per_pair[pair_key] < wrong_samples_per_pair:
                    selected_records.append(record)
                    saved_wrong_count_per_pair[pair_key] += 1

    print(f"Selected {len(selected_records)} samples for Grad-CAM export.")

    # --------------------------------------------------
    # Second pass: generate Grad-CAM images
    # --------------------------------------------------
    selected_entries = []

    for r in selected_records:
        sample_index = r["sample_index"]
        true_class = r["true_class"]
        pred_class = r["pred_class"]
        pred_prob = r["pred_prob"]
        true_prob = r["true_prob"]

        input_image, _ = test_dataset[sample_index]
        input_image = input_image.unsqueeze(0).to(device)

        result = gradcam.generate_pred_true(input_image, true_class_idx=true_class)
        pred_cam = result["pred_cam"]
        true_cam = result["true_cam"]

        image_np = tensor_to_display_image(input_image[0])

        output_path = save_gradcam_figure(
            image_np=image_np,
            pred_cam=pred_cam,
            true_cam=true_cam,
            true_class=true_class,
            pred_class=pred_class,
            pred_prob=pred_prob,
            true_prob=true_prob,
            sample_index=sample_index,
            save_root=save_root,
        )

        case_type = get_case_type(true_class, pred_class)

        selected_entries.append(
            {
                "output_path": output_path,
                "case_type": case_type,
                "sample_index": sample_index,
                "true_class": true_class,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
                "true_prob": true_prob,
            }
        )

        print(f"Saved: {output_path}")

    gradcam.remove_hooks()

    # --------------------------------------------------
    # Save reports and plots
    # --------------------------------------------------
    prediction_csv = save_prediction_summary_csv(all_records, report_dir)
    selection_csv = save_selection_summary_csv(selected_entries, report_dir)
    annotation_csv = save_annotation_table(selected_entries, report_dir)

    cm_png = plot_confusion_matrix(all_records, n_classes, report_dir)
    counts_png = plot_correct_wrong_counts(all_records, report_dir)
    top_pairs_png = plot_top_confusion_pairs(all_records, report_dir, top_k=10)

    summary_txt = save_text_report(
        checkpoint_path=checkpoint_path,
        n_classes=n_classes,
        all_records=all_records,
        selected_entries=selected_entries,
        correct_per_pair=correct_samples_per_pair,
        wrong_per_pair=wrong_samples_per_pair,
        report_dir=report_dir,
    )

    print("\nDone.")
    print(f"Grad-CAM results saved in: {save_root}")
    print(f"Reports saved in: {report_dir}")
    print(f"Prediction summary CSV: {prediction_csv}")
    print(f"Selection summary CSV: {selection_csv}")
    print(f"Annotation table CSV: {annotation_csv}")
    print(f"Confusion matrix: {cm_png}")
    print(f"Correct vs wrong plot: {counts_png}")
    print(f"Top confusion pairs plot: {top_pairs_png}")
    print(f"Text summary: {summary_txt}")

    required_outputs = [
        prediction_csv,
        selection_csv,
        annotation_csv,
        cm_png,
        counts_png,
        top_pairs_png,
        summary_txt,
    ]

    missing_outputs = [p for p in required_outputs if not os.path.exists(p)]

    if missing_outputs:
        print("\nWARNING: Some expected report files were not created:")
        for p in missing_outputs:
            print(f"  Missing: {p}")
    else:
        print("\nAll expected report files were created successfully.")


if __name__ == "__main__":
    main()