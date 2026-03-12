# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import random
# Evaluation imports (NEW)
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

def deterministic_mode(seed=42):

    # Set seeds for reproducibility
    # Ensuring the order of classes cannot change in the future
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Force cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Force PyTorch operations to be deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    # Candidate severity-based weight sets
    weight_sets = {
        "1st run": [2.0, 2.0, 2.0, 1.0, 1.25, 4.0],
        "2nd run": [2.5, 2.5, 2.5, 1.0, 1.25, 5.0],
        "3rd run": [3.0, 3.0, 3.0, 1.0, 1.25, 6.0],
        "4th run": [3.5, 3.5, 3.5, 1.0, 1.25, 7.0],
        "5th run": [4.0, 4.0, 4.0, 1.0, 1.25, 8.0],
    }

    class_names = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Nodule", "Pneumothorax"]

    active_weight_set_name = "5th run"
    active_weight_values = weight_sets[active_weight_set_name]

    # Load the Neural Net. NOTE: set number of distinct labels here
    n_classes = 6
    model = Net(n_classes=n_classes)

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size
    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif torch.backends.mps.is_available() and not DEBUG:
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Create class weights AFTER device is known
    class_weights = torch.tensor(active_weight_values, dtype=torch.float32).to(device)

    print(f"Using weight set: {active_weight_set_name}")
    for name, weight in zip(class_names, class_weights):
        print(f"{name}: {weight.item():.2f}")

    # Weighted loss
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=False
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=False
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    for e in range(n_epochs):
        if activeloop:

            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            losses = test_model(model, test_sampler, loss_function, device)
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")
            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])
            plotext.show()

    # ========= NEW: Evaluation (Confusion Matrix + Macro-F1) =========
    # Collect predictions on the test set
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in test_sampler:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    print("\n================ Weight experiment ================")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro-F1: {macro_f1:.6f}")
    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)

    # If your label mapping is known, replace these with the exact class names in order 0..5
    class_names = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Nodule", "Pneumothorax"]

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    # ===============================================================

    # retrieve current time to label artifacts
    now = datetime.now()

    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saving the model
    torch.save(
        model.state_dict(),
        f"model_weights/model_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}.txt",
    )

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(
        range(1, 1 + n_epochs),
        [x.detach().cpu() for x in mean_losses_train],
        label="Train",
        color="blue",
    )
    ax2.plot(
        range(1, 1 + n_epochs),
        [x.detach().cpu() for x in mean_losses_test],
        label="Test",
        color="red",
    )
    fig.legend()

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(
        Path("artifacts") / f"session_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}_loss.png"
    )

    # ========= NEW: Save Confusion Matrix plot + metrics to artifacts =========
    # Save confusion matrix figure
    fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax_cm, cmap=None, values_format="d")  # cmap=None => matplotlib default
    ax_cm.set_title("Confusion Matrix (weight_experiment)")
    fig_cm.tight_layout()
    fig_cm.savefig(
        Path("artifacts") / f"session_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}_cm.png"
    )

    # Save text metrics
    metrics_path = Path("artifacts") / f"session_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Weight experiment EVALUATION\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    # ========================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()
    deterministic_mode(seed=42)
    main(args)