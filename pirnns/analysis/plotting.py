

import matplotlib.pyplot as plt
import numpy as np

from pirnns.analysis.sweep_evaluator import SweepResult


def plot_sweep_results_ood_trajectory_length_decoding_error(
    sweep_result: SweepResult,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
    log_x: bool = True,
    log_y: bool = False,
) -> None:
    """
    Plot sweep evaluation results.

    Args:
        sweep_result: SweepResult object from evaluator.evaluate()
        figsize: Figure size tuple
        save_path: Optional path to save figure
        log_x: Use log scale on x-axis
        log_y: Use log scale on y-axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(sweep_result.experiment_results)))

    for i, (exp_name, exp_result) in enumerate(sweep_result.experiment_results.items()):
        x = exp_result.test_conditions
        y_mean = exp_result.mean_measurements
        y_std = exp_result.std_measurements

        # Plot mean line
        ax.plot(
            x,
            y_mean,
            "-o",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=exp_name,
        )

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        # Plot shaded error region (std)
        ax.fill_between(
            x,
            np.array(y_mean) - np.array(y_std),
            np.array(y_mean) + np.array(y_std),
            color=colors[i],
            alpha=0.2,
        )

    # Get training length from metadata (if available)
    first_exp = list(sweep_result.experiment_results.values())[0]
    if "training_length" in first_exp.metadata:
        training_length = first_exp.metadata["training_length"]
        ax.axvline(
            training_length,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"Training length ({training_length})",
        )

    ax.set_xlabel("Trajectory Length (time steps)", fontsize=12)
    ax.set_ylabel("Position Decoding Error (m)", fontsize=12)
    ax.set_title("OOD Generalization: Decoding Error vs Trajectory Length", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()