"""
Performance analysis tools.

Tools for analyzing model training performance including loss curves,
decoding accuracy, and convergence analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Optional, Tuple


def plot_loss_curves(
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
    epochs: Optional[List[int]] = None,
    loss_file: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    title: str = "Training and Validation Loss",
    log_x: bool = True,
    log_y: bool = True,
    show_improvement: bool = True,
) -> None:
    """
    Plot training and validation loss curves.

    Parameters:
    -----------
    train_losses : list, optional
        Training loss values
    val_losses : list, optional
        Validation loss values
    epochs : list, optional
        Epoch numbers. If None, uses range
    loss_file : str, optional
        Path to JSON file containing loss data. If provided, overrides other parameters
    figsize : tuple
        Figure size
    title : str
        Plot title
    log_x : bool
        Use log scale on x-axis
    log_y : bool
        Use log scale on y-axis
    show_improvement : bool
        Show improvement statistics
    """

    # Load from file if provided
    if loss_file is not None:
        if not os.path.exists(loss_file):
            raise FileNotFoundError(f"Loss file not found: {loss_file}")

        with open(loss_file, "r") as f:
            loss_data = json.load(f)

        # Extract data from JSON
        epochs = loss_data["epochs"][:-1]  # Remove last epoch if incomplete
        train_losses = loss_data["train_losses_epoch"]
        val_losses = loss_data["val_losses_epoch"][:-1]  # Align with epochs

        print(f"Loaded loss data from: {loss_file}")
        print(f"Training epochs: {len(epochs)}")

    # Validate that we have data
    if train_losses is None or val_losses is None:
        raise ValueError("Must provide either (train_losses, val_losses) or loss_file")

    # Default epochs if not provided
    if epochs is None:
        epochs = list(range(len(train_losses)))

    # Create the plot
    plt.figure(figsize=figsize)

    plt.plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2, markersize=4)
    plt.plot(epochs, val_losses, "s-", label="Val Loss", linewidth=2, markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Apply log scales
    if log_y:
        plt.yscale("log")
    if log_x:
        plt.xscale("log")

    # Add final loss values as text
    plt.text(
        0.02,
        0.98,
        f"Final train loss: {train_losses[-1]:.6f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )
    plt.text(
        0.02,
        0.85,
        f"Final val loss: {val_losses[-1]:.6f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()

    if show_improvement:
        print(f"Initial train loss: {train_losses[0]:.4f}")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Initial val loss: {val_losses[0]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        print(
            f"Best val loss: {min(val_losses):.6f} at epoch {epochs[np.argmin(val_losses)]}"
        )


def load_training_metrics(run_dir: str) -> dict:
    """
    Load training metrics from a run directory.

    Parameters:
    -----------
    run_dir : str
        Path to run directory containing training_losses.json

    Returns:
    --------
    dict : Training metrics data
    """
    loss_file = os.path.join(run_dir, "training_losses.json")

    if not os.path.exists(loss_file):
        raise FileNotFoundError(f"No training_losses.json found in {run_dir}")

    with open(loss_file, "r") as f:
        return json.load(f)
