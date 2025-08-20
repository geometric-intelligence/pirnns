#!/usr/bin/env python3
"""
Standalone script to visualize trajectory data generation
Run with: python debug_trajectories.py
"""

import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import datetime
import os
import sys

# Add the parent directory to path so we can import datamodule
sys.path.append("/home/facosta/pirnns/pirnns")
from datamodule import PathIntegrationDataModule


def create_trajectory_animation(
    trained_model,
    untrained_model,
    eval_data,  # (inputs, targets) tuple
    pca_trained,
    pca_untrained,
    device,
    trajectory_idx=0,
    num_frames=100,
    fps=2,
    output_dir=".",
    run_id=None,
    figsize=(16, 12),
):
    """
    Create an animated visualization of trajectory prediction.

    Parameters:
    -----------
    trained_model : PathIntRNN
        Trained model
    untrained_model : PathIntRNN
        Untrained model for comparison
    eval_data : tuple
        (inputs, targets) tuple where inputs/targets are torch tensors
    pca_trained, pca_untrained : sklearn.PCA
        Fitted PCA objects for dimensionality reduction
    device : str
        Device to run models on
    trajectory_idx : int, default=0
        Which trajectory to visualize
    num_frames : int, default=100
        Number of time steps to animate
    fps : int, default=2
        Frames per second for video
    output_dir : str, default="."
        Directory to save video
    run_id : str, optional
        Run identifier for filename
    figsize : tuple, default=(16, 12)
        Figure size

    Returns:
    --------
    str : Path to saved video file
    """

    # Generate run_id if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Creating trajectory animation for run: {run_id}")

    # ===== PREPARE DATA =====
    inputs, targets = eval_data

    # Extract single trajectory data
    single_inputs = inputs[trajectory_idx : trajectory_idx + 1]
    single_targets = targets[trajectory_idx : trajectory_idx + 1]

    # Get model predictions
    with torch.no_grad():
        single_inputs_gpu = single_inputs.to(device)
        single_targets_gpu = single_targets.to(device)

        trained_hidden, trained_output = trained_model(
            inputs=single_inputs_gpu, pos_0=single_targets_gpu[:, 0, :]
        )
        untrained_hidden, untrained_output = untrained_model(
            inputs=single_inputs_gpu, pos_0=single_targets_gpu[:, 0, :]
        )

    # Convert to numpy
    trajectory_true = single_targets[0].cpu().numpy()
    trajectory_predicted = trained_output[0].cpu().numpy()
    trajectory_hidden_trained = trained_hidden[0].cpu().numpy()
    trajectory_hidden_untrained = untrained_hidden[0].cpu().numpy()
    velocity_inputs = single_inputs[0].cpu().numpy()  # [heading, speed]

    # Project to PCA space
    trajectory_pca_trained = pca_trained.transform(trajectory_hidden_trained)
    trajectory_pca_untrained = pca_untrained.transform(trajectory_hidden_untrained)

    # Extract heading and speed
    headings = velocity_inputs[:, 0]  # In radians
    speeds = velocity_inputs[:, 1]  # Magnitude

    # Calculate prediction error
    pred_error = np.linalg.norm(trajectory_predicted - trajectory_true, axis=1)

    # Limit frames to available data
    num_frames = min(num_frames, len(trajectory_true))

    print(f"Trajectory data prepared: {len(trajectory_true)} time steps")
    print(f"Animating first {num_frames} steps")
    print(f"Speed range: {speeds.min():.3f} to {speeds.max():.3f}")

    # ===== SETUP PLOTS =====
    plt.ioff()  # Turn off interactive mode
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Setup velocity plot (top-left)
    ax_vel = axes[0]
    ax_vel.remove()
    ax_vel = fig.add_subplot(2, 2, 1, projection="polar")
    ax_vel.set_ylim(0, speeds.max() * 1.1)
    ax_vel.set_title(
        "Input: Velocity (Heading & Speed)", fontsize=14, fontweight="bold"
    )
    ax_vel.set_theta_zero_location("E")
    ax_vel.set_theta_direction(1)

    # Setup PCA plot (top-right)
    ax_pca = axes[1]
    ax_pca.set_xlim(
        trajectory_pca_trained[:, 0].min() - 1, trajectory_pca_trained[:, 0].max() + 1
    )
    ax_pca.set_ylim(
        trajectory_pca_trained[:, 1].min() - 1, trajectory_pca_trained[:, 1].max() + 1
    )
    ax_pca.set_xlabel(f"PC1 ({pca_trained.explained_variance_ratio_[0]*100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({pca_trained.explained_variance_ratio_[1]*100:.1f}%)")
    ax_pca.set_title("Computation: Hidden State PCA", fontsize=14, fontweight="bold")
    ax_pca.grid(True, alpha=0.3)

    # Setup arena plot (bottom-left)
    ax_arena = axes[2]
    ax_arena.set_xlim(
        trajectory_true[:, 0].min() - 0.5, trajectory_true[:, 0].max() + 0.5
    )
    ax_arena.set_ylim(
        trajectory_true[:, 1].min() - 0.5, trajectory_true[:, 1].max() + 0.5
    )
    ax_arena.set_xlabel("X Position")
    ax_arena.set_ylabel("Y Position")
    ax_arena.set_title("Output: Arena View", fontsize=14, fontweight="bold")
    ax_arena.grid(True, alpha=0.3)
    ax_arena.set_aspect("equal")

    # Plot start/end markers
    start_pos = trajectory_true[0]
    end_pos = trajectory_true[-1]
    ax_arena.plot(start_pos[0], start_pos[1], "go", markersize=10, label="Start")
    ax_arena.plot(end_pos[0], end_pos[1], "ro", markersize=10, label="End")

    # Setup error plot (bottom-right)
    ax_error = axes[3]
    ax_error.set_xlim(0, num_frames - 1)
    ax_error.set_ylim(0, pred_error[:num_frames].max() * 1.1)
    ax_error.set_xlabel("Time Step")
    ax_error.set_ylabel("Prediction Error")
    ax_error.set_title("Error: Prediction vs Truth", fontsize=14, fontweight="bold")
    ax_error.grid(True, alpha=0.3)

    # Initialize line objects
    (line_true,) = ax_arena.plot([], [], "b-", linewidth=2, alpha=0.7, label="True")
    (line_pred,) = ax_arena.plot(
        [], [], "hotpink", linewidth=2, alpha=0.7, label="Predicted"
    )
    (point_current,) = ax_arena.plot([], [], "ko", markersize=8)

    (line_pca_trained,) = ax_pca.plot(
        [], [], "b-", linewidth=2, alpha=0.7, label="Trained"
    )
    (line_pca_untrained,) = ax_pca.plot(
        [], [], "gray", linewidth=2, alpha=0.7, label="Untrained"
    )
    (point_pca_current,) = ax_pca.plot([], [], "ko", markersize=8)

    (line_error,) = ax_error.plot([], [], "red", linewidth=2, alpha=0.8, label="Error")

    # Add legends
    ax_arena.legend()
    ax_pca.legend()
    ax_error.legend()

    def animate(frame):
        # Arena
        line_true.set_data(
            trajectory_true[: frame + 1, 0], trajectory_true[: frame + 1, 1]
        )
        line_pred.set_data(
            trajectory_predicted[: frame + 1, 0], trajectory_predicted[: frame + 1, 1]
        )
        point_current.set_data([trajectory_true[frame, 0]], [trajectory_true[frame, 1]])

        # PCA
        line_pca_trained.set_data(
            trajectory_pca_trained[: frame + 1, 0],
            trajectory_pca_trained[: frame + 1, 1],
        )
        line_pca_untrained.set_data(
            trajectory_pca_untrained[: frame + 1, 0],
            trajectory_pca_untrained[: frame + 1, 1],
        )
        point_pca_current.set_data(
            [trajectory_pca_trained[frame, 0]], [trajectory_pca_trained[frame, 1]]
        )

        # Velocity - clear and redraw
        ax_vel.clear()
        ax_vel.set_ylim(0, speeds.max() * 1.1)
        ax_vel.set_title(
            "Input: Velocity (Heading & Speed)", fontsize=14, fontweight="bold"
        )
        ax_vel.set_theta_zero_location("E")
        ax_vel.set_theta_direction(1)

        # Plot velocity history
        theta_history = headings[: frame + 1]
        r_history = speeds[: frame + 1]
        ax_vel.plot(
            theta_history,
            r_history,
            "lightblue",
            linewidth=1,
            alpha=0.6,
            label="History",
        )

        # Add arrow for current velocity
        current_heading = headings[frame]
        current_speed = speeds[frame]
        ax_vel.annotate(
            "",
            xy=(current_heading, current_speed),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )
        ax_vel.legend()

        # Error plot
        error_history = pred_error[: frame + 1]
        time_steps = np.arange(frame + 1)
        line_error.set_data(time_steps, error_history)

        fig.suptitle(f"PIRNN - Step {frame}/{num_frames-1}", fontsize=16)

        return (
            line_true,
            line_pred,
            point_current,
            line_pca_trained,
            line_pca_untrained,
            point_pca_current,
            line_error,
        )

    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=100, blit=False, repeat=True
    )

    # Save video
    output_filename = os.path.join(output_dir, f"trajectory_analysis_run_{run_id}.mp4")
    print(f"Saving video as {output_filename}...")

    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(
            fps=fps, metadata=dict(artist="Path Integration Analysis"), bitrate=1800
        )
        anim.save(output_filename, writer=writer)
        print(f"Video saved as {output_filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        return None
    finally:
        plt.close(fig)

    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"Duration: {len(trajectory_true)} time steps")
    print(f"Mean prediction error: {pred_error.mean():.3f}")
    print(f"Final prediction error: {pred_error[-1]:.3f}")
    print(f"Speed range: {speeds.min():.3f} to {speeds.max():.3f}")

    return output_filename


def debug_trajectories(config_path="configs/vanilla_config.yaml", num_samples=3):
    """Debug trajectory generation and visualization"""

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=== CONFIG ANALYSIS ===")
    print(f"Trajectory duration: {config['trajectory_duration']} seconds")
    print(f"Time steps: {config['num_time_steps']}")
    print(
        f"dt: {config['trajectory_duration'] / config['num_time_steps']:.3f} seconds per step"
    )
    print(f"Speed: μ={config['mu_speed']}, σ={config['sigma_speed']}")
    print(f"Arena size: {config['arena_size']}")
    print(
        f"Max possible distance (at max speed for full duration): {config['mu_speed'] * config['trajectory_duration']:.3f}"
    )
    print(f"Arena diagonal: {config['arena_size'] * np.sqrt(2):.3f}")
    print()

    # Create datamodule with small number of trajectories for testing
    datamodule = PathIntegrationDataModule(
        num_trajectories=num_samples,
        batch_size=num_samples,
        num_workers=0,  # No multiprocessing for debugging
        train_val_split=1.0,  # All data for testing
        trajectory_duration=config["trajectory_duration"],
        num_time_steps=config["num_time_steps"],
        arena_size=config["arena_size"],
        mu_speed=config["mu_speed"],
        sigma_speed=config["sigma_speed"],
        tau_vel=config["tau_vel"],
        num_place_cells=config["num_place_cells"],
        place_cell_rf=config["place_cell_rf"],
        surround_scale=config["surround_scale"],
        DoG=config["DoG"],
        trajectory_type=config["trajectory_type"],
    )

    # Generate trajectories
    print("=== GENERATING TRAJECTORIES ===")
    inputs, positions, place_cell_activations = datamodule.simulate_trajectories(
        device="cpu"
    )

    print(f"Generated {inputs.shape[0]} trajectories")
    print(f"Input shape: {inputs.shape} (batch, time, features)")
    print(f"Position shape: {positions.shape} (batch, time, 2)")
    print(f"Place cell shape: {place_cell_activations.shape}")

    # Analyze trajectory statistics
    print("\n=== TRAJECTORY STATISTICS ===")
    distances = torch.norm(positions[:, 1:] - positions[:, :-1], dim=-1).sum(dim=1)
    speeds = inputs[..., 1].mean(dim=1)  # inputs[..., 1] contains speeds

    print("Total distances traveled:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Std: {distances.std():.4f}")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")

    print("Average speeds:")
    print(f"  Mean: {speeds.mean():.4f}")
    print(f"  Std: {speeds.std():.4f}")

    print("\nPosition ranges:")
    print(f"  X: [{positions[..., 0].min():.3f}, {positions[..., 0].max():.3f}]")
    print(f"  Y: [{positions[..., 1].min():.3f}, {positions[..., 1].max():.3f}]")

    xlim = [-config["arena_size"] / 2, config["arena_size"] / 2]
    ylim = [-config["arena_size"] / 2, config["arena_size"] / 2]

    print(f"\nUsing axis limits: x={xlim}, y={ylim}")

    # Create visualization: Each trajectory gets its own row with 3 plots
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array even for single trajectory

    place_centers = datamodule.place_cell_centers.numpy()
    print("\nPlace cell centers range:")
    print(f"  X: [{place_centers[:, 0].min():.3f}, {place_centers[:, 0].max():.3f}]")
    print(f"  Y: [{place_centers[:, 1].min():.3f}, {place_centers[:, 1].max():.3f}]")

    for i in range(num_samples):
        traj = positions[i].numpy()
        print(f"\nTrajectory {i+1} data:")
        print(f"  Start: ({traj[0, 0]:.3f}, {traj[0, 1]:.3f})")
        print(f"  End: ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})")
        print(f"  X range: [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")
        print(f"  Y range: [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]")

        # === TRAJECTORY PLOT ===
        ax_traj = axes[i, 0]

        # Plot trajectory
        ax_traj.plot(
            traj[:, 0], traj[:, 1], "b-", linewidth=3, label="Trajectory", alpha=0.8
        )

        # Mark start and end
        ax_traj.scatter(
            traj[0, 0],
            traj[0, 1],
            c="green",
            s=150,
            marker="o",
            label="Start",
            zorder=5,
            edgecolors="black",
            linewidth=2,
        )
        ax_traj.scatter(
            traj[-1, 0],
            traj[-1, 1],
            c="red",
            s=150,
            marker="s",
            label="End",
            zorder=5,
            edgecolors="black",
            linewidth=2,
        )

        # Mark all points to see trajectory detail
        ax_traj.scatter(
            traj[:, 0],
            traj[:, 1],
            c=range(len(traj)),
            cmap="plasma",
            s=30,
            alpha=0.7,
            zorder=3,
        )

        ax_traj.set_xlim(xlim)
        ax_traj.set_ylim(ylim)
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_title(f"Trajectory {i+1}\nDistance: {distances[i]:.4f}")
        ax_traj.legend()
        ax_traj.set_xlabel("X Position")
        ax_traj.set_ylabel("Y Position")

        # Add trajectory info
        speed_info = f"Avg Speed: {speeds[i]:.3f}"
        ax_traj.text(
            0.02,
            0.98,
            speed_info,
            transform=ax_traj.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # === PLACE CELL ACTIVATIONS AT START ===
        ax_start = axes[i, 1]

        start_activations = place_cell_activations[i, 0, :].numpy()  # t=0
        start_pos = positions[i, 0].numpy()

        # Plot place cell activations
        scatter_start = ax_start.scatter(
            place_centers[:, 0],
            place_centers[:, 1],
            c=start_activations,
            cmap="viridis",
            s=30,
            alpha=0.8,
        )

        # Mark agent position
        ax_start.scatter(
            start_pos[0],
            start_pos[1],
            c="green",
            s=200,
            marker="*",
            label="Start Position",
            zorder=5,
            edgecolors="black",
            linewidth=2,
        )

        ax_start.set_xlim(xlim)
        ax_start.set_ylim(ylim)
        ax_start.set_aspect("equal")
        ax_start.grid(True, alpha=0.3)
        ax_start.set_title(
            f"Place Cells at START\n({start_pos[0]:.3f}, {start_pos[1]:.3f})"
        )
        ax_start.legend()
        ax_start.set_xlabel("X Position")
        ax_start.set_ylabel("Y Position")

        # Add colorbar
        cbar_start = plt.colorbar(scatter_start, ax=ax_start)
        cbar_start.set_label("Activation")

        # === PLACE CELL ACTIVATIONS AT END ===
        ax_end = axes[i, 2]

        end_activations = place_cell_activations[i, -1, :].numpy()  # t=final
        end_pos = positions[i, -1].numpy()

        # Plot place cell activations
        scatter_end = ax_end.scatter(
            place_centers[:, 0],
            place_centers[:, 1],
            c=end_activations,
            cmap="viridis",
            s=30,
            alpha=0.8,
        )

        # Mark agent position
        ax_end.scatter(
            end_pos[0],
            end_pos[1],
            c="red",
            s=200,
            marker="*",
            label="End Position",
            zorder=5,
            edgecolors="black",
            linewidth=2,
        )

        ax_end.set_xlim(xlim)
        ax_end.set_ylim(ylim)
        ax_end.set_aspect("equal")
        ax_end.grid(True, alpha=0.3)
        ax_end.set_title(f"Place Cells at END\n({end_pos[0]:.3f}, {end_pos[1]:.3f})")
        ax_end.legend()
        ax_end.set_xlabel("X Position")
        ax_end.set_ylabel("Y Position")

        # Add colorbar
        cbar_end = plt.colorbar(scatter_end, ax=ax_end)
        cbar_end.set_label("Activation")

    plt.tight_layout()
    plt.suptitle(f"Trajectory Debug Analysis (n={num_samples})", fontsize=16, y=0.98)

    # Save figure
    output_path = "debug_trajectories.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("\n=== VISUALIZATION SAVED ===")
    print(f"Saved to: {output_path}")

    return distances, speeds, positions


if __name__ == "__main__":
    # Change to the script directory
    os.chdir("/home/facosta/pirnns/pirnns")

    print("Starting trajectory debugging...")
    distances, speeds, positions = debug_trajectories(num_samples=3)

    print("\n=== RECOMMENDATIONS ===")
    if distances.mean() < 0.1:
        print("❌ Trajectories are very short! Consider:")
        print("   - Increasing trajectory_duration (e.g., 10-20 seconds)")
        print("   - Increasing mu_speed (e.g., 0.5-1.0)")
        print("   - Increasing num_time_steps for more detail")
    elif distances.mean() > 1.0:
        print("✅ Trajectories look reasonable length")
    else:
        print("⚠️  Trajectories are short but might be visible")

    print(
        f"\nCurrent trajectory length / arena size ratio: {distances.mean() / 2.2:.3f}"
    )
    print("Good ratio should be > 0.1 for visible trajectories")
