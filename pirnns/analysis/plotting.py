"""
Plotting utilities for analysis results and training curves.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from .sweep_evaluator import SweepResult


def plot_sweep_results(
    sweep_result: SweepResult,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_training_length: bool = True,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """
    Plot sweep evaluation results (e.g., OOD generalization).
    
    Args:
        sweep_result: SweepResult object from evaluator.evaluate()
        figsize: Figure size tuple
        save_path: Optional path to save figure
        show_training_length: Whether to show vertical line at training length
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
        ax.plot(x, y_mean, '-o', color=colors[i], linewidth=2, 
                markersize=6, label=exp_name)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        # Plot shaded error region (std)
        ax.fill_between(
            x,
            np.array(y_mean) - np.array(y_std),
            np.array(y_mean) + np.array(y_std),
            color=colors[i],
            alpha=0.2
        )
    
    # Get training length from metadata (if available)
    if show_training_length:
        first_exp = list(sweep_result.experiment_results.values())[0]
        if "training_length" in first_exp.metadata:
            training_length = first_exp.metadata["training_length"]
            ax.axvline(training_length, color='red', linestyle='--', 
                       alpha=0.7, linewidth=1.5, 
                       label=f'Training length ({training_length})')
    
    # Get labels from result
    condition_name = list(sweep_result.experiment_results.values())[0].condition_name
    
    ax.set_xlabel(condition_name.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Measurement Value', fontsize=12)
    ax.set_title(f'{sweep_result.analysis_type}: {sweep_result.measurement_type}', 
                 fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_training_curves_sweep(
    sweep: dict,
    metric: str = "val_loss",
    figsize: Tuple[int, int] = (12, 7),
    log_x: bool = False,
    log_y: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training curves for an entire sweep.
    
    Args:
        sweep: Sweep dictionary loaded from load_experiment_sweep
        metric: What to plot - "train_loss", "val_loss", or "decoding_error"
        figsize: Figure size
        log_x: Use log scale on x-axis
        log_y: Use log scale on y-axis
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sweep)))
    
    for i, (exp_name, seeds) in enumerate(sweep.items()):
        # Collect data from all seeds
        all_curves = []
        epochs = None
        
        for seed_key, seed_data in seeds.items():
            # Navigate nested structure based on metric
            if metric == "train_loss":
                if "training_losses" not in seed_data or seed_data["training_losses"] is None:
                    continue
                losses_dict = seed_data["training_losses"]
                if "train_losses" not in losses_dict:
                    continue
                curve = losses_dict["train_losses"]
                if epochs is None and "epochs" in losses_dict:
                    epochs = losses_dict["epochs"]
                    
            elif metric == "val_loss":
                if "training_losses" not in seed_data or seed_data["training_losses"] is None:
                    continue
                losses_dict = seed_data["training_losses"]
                if "val_losses" not in losses_dict:
                    continue
                curve = losses_dict["val_losses"]
                if epochs is None and "epochs" in losses_dict:
                    epochs = losses_dict["epochs"]
                    
            elif metric == "decoding_error":
                if "position_decoding_errors" not in seed_data or seed_data["position_decoding_errors"] is None:
                    continue
                errors_dict = seed_data["position_decoding_errors"]
                if "position_errors_epoch" not in errors_dict:
                    continue
                curve = errors_dict["position_errors_epoch"]
                if epochs is None and "epochs" in errors_dict:
                    epochs = errors_dict["epochs"]
            else:
                raise ValueError(f"Unknown metric: {metric}. Choose from ['train_loss', 'val_loss', 'decoding_error']")
            
            if len(curve) == 0:
                continue
                
            all_curves.append(curve)
        
        if not all_curves:
            print(f"Warning: No {metric} data found for {exp_name}")
            continue
        
        # Ensure all curves have same length (use minimum)
        min_len = min(len(c) for c in all_curves)
        all_curves = [c[:min_len] for c in all_curves]
        
        # Convert to array [n_seeds, n_epochs]
        curves_array = np.array(all_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        
        # Create epochs array if not available
        if epochs is None or len(epochs) < min_len:
            epochs = np.arange(min_len)
        else:
            epochs = np.array(epochs[:min_len])
        
        # Plot mean with shaded std
        ax.plot(epochs, mean_curve, '-', color=colors[i], 
                linewidth=2, label=f"{exp_name} (n={len(all_curves)})")
        ax.fill_between(
            epochs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=colors[i],
            alpha=0.2
        )
    
    # Labels and formatting
    metric_labels = {
        "train_loss": "Training Loss",
        "val_loss": "Validation Loss",
        "decoding_error": "Decoding Error (m)",
    }
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_labels[metric], fontsize=12)
    ax.set_title(f"Training Curves: {metric_labels[metric]}", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_final_performance_comparison(
    sweep: dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Bar plot comparing final validation loss and decoding error across experiments.
    
    Args:
        sweep: Sweep dictionary loaded from load_experiment_sweep
        figsize: Figure size
        save_path: Optional path to save figure
    """
    exp_names = []
    val_losses_mean = []
    val_losses_std = []
    decoding_errors_mean = []
    decoding_errors_std = []
    
    for exp_name, seeds in sweep.items():
        exp_names.append(exp_name)
        
        # Collect final values from all seeds
        final_val_losses = []
        final_decoding_errors = []
        
        for seed_data in seeds.values():
            # Get final validation loss
            if "training_losses" in seed_data and seed_data["training_losses"]:
                losses_dict = seed_data["training_losses"]
                if "val_losses" in losses_dict and losses_dict["val_losses"]:
                    final_val_losses.append(losses_dict["val_losses"][-1])
            
            # Get final decoding error
            if "position_decoding_errors" in seed_data and seed_data["position_decoding_errors"]:
                errors_dict = seed_data["position_decoding_errors"]
                if "position_errors_epoch" in errors_dict and errors_dict["position_errors_epoch"]:
                    final_decoding_errors.append(errors_dict["position_errors_epoch"][-1])
        
        val_losses_mean.append(np.mean(final_val_losses) if final_val_losses else np.nan)
        val_losses_std.append(np.std(final_val_losses) if final_val_losses else 0)
        decoding_errors_mean.append(np.mean(final_decoding_errors) if final_decoding_errors else np.nan)
        decoding_errors_std.append(np.std(final_decoding_errors) if final_decoding_errors else 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(exp_names))
    width = 0.6
    
    # Validation loss
    ax1.bar(x, val_losses_mean, width, yerr=val_losses_std, 
            capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Experiment', fontsize=11)
    ax1.set_ylabel('Final Validation Loss', fontsize=11)
    ax1.set_title('Final Validation Loss', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Decoding error
    ax2.bar(x, decoding_errors_mean, width, yerr=decoding_errors_std,
            capsize=5, alpha=0.7, color='coral')
    ax2.set_xlabel('Experiment', fontsize=11)
    ax2.set_ylabel('Final Decoding Error (m)', fontsize=11)
    ax2.set_title('Final Decoding Error', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()