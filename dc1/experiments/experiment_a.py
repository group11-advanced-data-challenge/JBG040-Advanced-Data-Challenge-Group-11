from __future__ import annotations

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
import numpy as np
import os
import argparse
import random
import copy
from datetime import datetime
from pathlib import Path
from typing import List


class SubsetDataset:
    """
    Subset of ImageDataset that keeps the same interface as the original:
    - .targets as numpy array
    - .imgs as numpy array
    - __getitem__ returning (image, label)
    """

    def __init__(self, base_dataset: ImageDataset, indices) -> None:
        self.base_dataset = base_dataset
        self.indices = np.array(indices)

        self.targets = self.base_dataset.targets[self.indices]
        self.imgs = self.base_dataset.imgs[self.indices]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label


def split_dataset(
    dataset: ImageDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[SubsetDataset, SubsetDataset]:
    """
    Split the original training set into train and validation subsets.
    """
    n = len(dataset)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(n * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = SubsetDataset(dataset, train_indices)
    val_subset = SubsetDataset(dataset, val_indices)

    return train_subset, val_subset


def main(args: argparse.Namespace) -> None:
    # experiment_a.py is inside dc1/experiments/, so go one level up to dc1
    base_dir = Path(__file__).resolve().parent.parent  # .../dc1

    # Load full training and test sets
    full_train_dataset = ImageDataset(
        base_dir / "data" / "X_train.npy",
        base_dir / "data" / "Y_train.npy",
    )
    test_dataset = ImageDataset(
        base_dir / "data" / "X_test.npy",
        base_dir / "data" / "Y_test.npy",
    )

    # Split the original training set into train + validation
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Full train set size:      {len(full_train_dataset)}")
    print(f"Train subset size:        {len(train_dataset)}")
    print(f"Validation subset size:   {len(val_dataset)}")
    print(f"Test set size:            {len(test_dataset)}")

    # Load model
    model = Net(n_classes=6)

    # Improved pipeline: Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    n_epochs = args.nb_epochs
    batch_size = args.batch_size
    patience = args.patience

    DEBUG = False

    # Device selection
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        summary(model, (1, 128, 128), device=device)
    elif torch.backends.mps.is_available() and not DEBUG:
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        summary(model, (1, 128, 128), device=device)

    # Samplers
    train_sampler = BatchSampler(
        batch_size=batch_size,
        dataset=train_dataset,
        balanced=args.balanced_batches,
    )

    val_sampler = BatchSampler(
        batch_size=100,
        dataset=val_dataset,
        balanced=False,
    )

    test_sampler = BatchSampler(
        batch_size=100,
        dataset=test_dataset,
        balanced=False,
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_val: List[torch.Tensor] = []

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_state = copy.deepcopy(model.state_dict())

    print("\n@@@ Starting Experiment A training...\n")

    for e in range(n_epochs):
        # Training
        train_losses = train_model(model, train_sampler, optimizer, loss_function, device)
        mean_train_loss = sum(train_losses) / len(train_losses)
        mean_losses_train.append(mean_train_loss)

        # Validation
        val_losses = test_model(model, val_sampler, loss_function, device)
        mean_val_loss = sum(val_losses) / len(val_losses)
        mean_losses_val.append(mean_val_loss)

        print(f"Epoch {e + 1}/{n_epochs}")
        print(f"  Train loss: {mean_train_loss.item():.6f}")
        print(f"  Val loss:   {mean_val_loss.item():.6f}")

        # Early stopping
        current_val_loss = mean_val_loss.item()
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = e + 1
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print("  -> New best model saved")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"\n@@@ Early stopping triggered at epoch {e + 1}")
            break

    print(f"\n@@@ Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Load best model before final test evaluation
    model.load_state_dict(best_model_state)

    # Final test evaluation
    test_losses = test_model(model, test_sampler, loss_function, device)
    mean_test_loss = sum(test_losses) / len(test_losses)
    print(f"@@@ Final test loss (best model): {mean_test_loss.item():.6f}")

    # Create output folders inside dc1/
    now = datetime.now()

    weights_dir = base_dir / "model_weights"
    artifacts_dir = base_dir / "artifacts"

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save best model
    model_path = (
        weights_dir
        / f"experimentA_adam_best_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"@@@ Saved best model to: {model_path}")

    # Save loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(mean_losses_train) + 1),
        [x.detach().cpu().item() for x in mean_losses_train],
        label="Train loss",
    )
    plt.plot(
        range(1, len(mean_losses_val) + 1),
        [x.detach().cpu().item() for x in mean_losses_val],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Experiment A: Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    plot_path = (
        artifacts_dir
        / f"experimentA_loss_curve_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}.png"
    )
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"@@@ Saved loss curve to: {plot_path}")

    # Save readable results to text file
    results_path = (
        artifacts_dir
        / f"experimentA_results_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}.txt"
    )

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Experiment A results\n")
        f.write("====================\n")
        f.write(f"Epochs requested: {n_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Validation fraction: {args.val_fraction}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Balanced training batches: {args.balanced_batches}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Final test loss: {mean_test_loss.item():.6f}\n")
        f.write(f"Saved model path: {model_path}\n")
        f.write(f"Saved plot path: {plot_path}\n\n")

        f.write("Train loss per epoch:\n")
        for i, loss in enumerate(mean_losses_train, start=1):
            f.write(f"Epoch {i}: {loss.detach().cpu().item():.6f}\n")

        f.write("\nValidation loss per epoch:\n")
        for i, loss in enumerate(mean_losses_val, start=1):
            f.write(f"Epoch {i}: {loss.detach().cpu().item():.6f}\n")

    print(f"@@@ Saved results to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nb_epochs", help="maximum number of epochs", default=50, type=int)
    parser.add_argument("--batch_size", help="batch size", default=25, type=int)
    parser.add_argument("--learning_rate", help="Adam learning rate", default=0.001, type=float)
    parser.add_argument("--val_fraction", help="fraction of training data used for validation", default=0.2, type=float)
    parser.add_argument("--patience", help="early stopping patience", default=5, type=int)
    parser.add_argument("--seed", help="random seed for train/validation split", default=42, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance training batches for class labels",
        default=True,
        type=bool,
    )

    args = parser.parse_args()
    main(args)