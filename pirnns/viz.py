import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from datetime import datetime
import os


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
