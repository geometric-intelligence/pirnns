import lightning as L
import json
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
from typing import Any
import matplotlib.pyplot as plt
import wandb
from pirnns.rnns.coupled_rnn import CoupledRNN
from pirnns.rnns.rnn import RNN


class LossLoggerCallback(L.Callback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

        self.train_losses_epoch: list[float] = []
        self.val_losses_epoch: list[float] = []
        self.epochs: list[int] = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics.get("train_loss_epoch", None)
        if train_loss is not None:
            self.train_losses_epoch.append(float(train_loss))

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.sanity_checking:
            print("Sanity checking, skipping validation loss logging")
            return

        val_loss = trainer.logged_metrics.get("val_loss", None)
        if val_loss is not None:
            self.val_losses_epoch.append(float(val_loss))
            self.epochs.append(trainer.current_epoch)

        self._save_losses()

    @rank_zero_only
    def _save_losses(self):
        os.makedirs(self.save_dir, exist_ok=True)

        loss_data = {
            "epochs": self.epochs,
            "train_losses_epoch": self.train_losses_epoch,
            "val_losses_epoch": self.val_losses_epoch,
        }

        with open(os.path.join(self.save_dir, "training_losses.json"), "w") as f:
            json.dump(loss_data, f, indent=2)


class PositionDecodingCallback(L.Callback):
    """Callback to compute and log position decoding error during validation."""

    def __init__(
        self,
        place_cell_centers: torch.Tensor,
        decode_k: int = 3,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.place_cell_centers = place_cell_centers
        self.decode_k = decode_k
        self.log_every_n_epochs = log_every_n_epochs

    def decode_position_from_place_cells(
        self, activation: torch.Tensor
    ) -> torch.Tensor:
        """Decode position from place cell activations using top-k method."""
        # Move centers to same device as activation
        centers = self.place_cell_centers.to(activation.device)
        _, idxs = torch.topk(activation, k=self.decode_k, dim=-1)  # [B, T, k]
        pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
        return pred_pos

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute position decoding error for each validation batch."""

        # Only run every N epochs to reduce computation
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        inputs, target_positions, target_place_cells = batch

        # Get model outputs
        with torch.no_grad():
            if isinstance(pl_module.model, RNN):
                _, outputs = pl_module.model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )
            elif isinstance(pl_module.model, CoupledRNN):
                _, _, outputs = pl_module.model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )

            # Convert to probabilities and decode positions
            place_cell_probs = torch.softmax(outputs, dim=-1)
            predicted_positions = self.decode_position_from_place_cells(
                place_cell_probs
            )
            position_error = torch.sqrt(
                ((target_positions - predicted_positions) ** 2).sum(-1)
            ).mean()

            # Log the error
            pl_module.log(
                "val_position_error",
                position_error,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )


class TrajectoryVisualizationCallback(L.Callback):
    """Callback to visualize and log trajectory predictions to wandb."""

    def __init__(
        self,
        place_cell_centers: torch.Tensor,
        arena_size: float,
        decode_k: int = 3,
        log_every_n_epochs: int = 5,
        num_trajectories_to_plot: int = 3,
    ):
        super().__init__()
        self.place_cell_centers = place_cell_centers
        self.arena_size = arena_size
        self.decode_k = decode_k
        self.log_every_n_epochs = log_every_n_epochs
        self.num_trajectories_to_plot = num_trajectories_to_plot

    def decode_position_from_place_cells(
        self, activation: torch.Tensor
    ) -> torch.Tensor:
        """Decode position from place cell activations using top-k method."""
        centers = self.place_cell_centers.to(activation.device)
        _, idxs = torch.topk(activation, k=self.decode_k, dim=-1)  # [B, T, k]
        pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
        return pred_pos

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Only run every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get a validation batch
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        # Get first batch from validation set
        batch = next(iter(val_dataloader))
        inputs, target_positions, target_place_cells = batch

        # Move to correct device
        inputs = inputs.to(pl_module.device)
        target_positions = target_positions.to(pl_module.device)
        target_place_cells = target_place_cells.to(pl_module.device)

        # Get model predictions
        with torch.no_grad():
            if isinstance(pl_module.model, RNN):
                _, outputs = pl_module.model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )
            elif isinstance(pl_module.model, CoupledRNN):
                _, _, outputs = pl_module.model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )

            # Convert to probabilities and decode positions
            place_cell_probs = torch.softmax(outputs, dim=-1)
            predicted_positions = self.decode_position_from_place_cells(
                place_cell_probs
            )

        # Set consistent axis limits based on arena size (centered at origin)
        lim = self.arena_size / 2
        xlim = [-lim, lim]
        ylim = [-lim, lim]

        # Create plots for first few trajectories
        figs = []
        for i in range(min(self.num_trajectories_to_plot, inputs.shape[0])):
            # Create figure with 5 subplots: trajectory + 4 place cell plots
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            # === TRAJECTORY PLOT ===
            ax_traj = axes[0]

            # Ground truth trajectory
            gt_traj = target_positions[i].cpu().numpy()
            pred_traj = predicted_positions[i].cpu().numpy()

            # Plot trajectories
            ax_traj.plot(
                gt_traj[:, 0],
                gt_traj[:, 1],
                "b-",
                linewidth=3,
                label="Ground Truth",
                alpha=0.8,
            )
            ax_traj.plot(
                pred_traj[:, 0],
                pred_traj[:, 1],
                "r--",
                linewidth=3,
                label="Predicted",
                alpha=0.8,
            )

            # Mark start and end points
            ax_traj.scatter(
                gt_traj[0, 0],
                gt_traj[0, 1],
                c="green",
                s=150,
                marker="o",
                label="Start",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )
            ax_traj.scatter(
                gt_traj[-1, 0],
                gt_traj[-1, 1],
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
                gt_traj[:, 0],
                gt_traj[:, 1],
                c=range(len(gt_traj)),
                cmap="plasma",
                s=30,
                alpha=0.7,
                zorder=3,
            )

            # Mark predicted trajectory points with different colormap
            ax_traj.scatter(
                pred_traj[:, 0],
                pred_traj[:, 1],
                c=range(len(pred_traj)),
                cmap="spring",
                s=30,
                alpha=0.7,
                zorder=3,
                marker="^",
            )

            ax_traj.set_xlabel("X Position")
            ax_traj.set_ylabel("Y Position")
            ax_traj.set_title("Trajectory Comparison")
            ax_traj.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=4)
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_aspect("equal")
            ax_traj.set_xlim(xlim)
            ax_traj.set_ylim(ylim)

            # Calculate and display error
            error = torch.sqrt(
                ((target_positions[i] - predicted_positions[i]) ** 2).sum(-1)
            ).mean()
            ax_traj.text(
                0.02,
                0.98,
                f"Mean Error: {error:.4f}",
                transform=ax_traj.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # === PLACE CELL ACTIVATION PLOTS ===
            place_centers = self.place_cell_centers.cpu().numpy()

            # START TIME - Ground Truth
            ax_start_gt = axes[1]
            start_gt_activations = target_place_cells[i, 0, :].cpu().numpy()  # t=0
            start_pos = target_positions[i, 0].cpu().numpy()

            scatter_start_gt = ax_start_gt.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=start_gt_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark agent position
            ax_start_gt.scatter(
                start_pos[0],
                start_pos[1],
                c="green",
                s=200,
                marker="*",
                label="Agent",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_start_gt.set_xlabel("X Position")
            ax_start_gt.set_ylabel("Y Position")
            ax_start_gt.set_title(
                f"START: GT Place Cells\n({start_pos[0]:.3f}, {start_pos[1]:.3f})"
            )
            ax_start_gt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_start_gt.grid(True, alpha=0.3)
            ax_start_gt.set_aspect("equal")
            ax_start_gt.set_xlim(xlim)
            ax_start_gt.set_ylim(ylim)
            cbar_start_gt = plt.colorbar(scatter_start_gt, ax=ax_start_gt)
            cbar_start_gt.set_label("Activation")

            # START TIME - Predicted
            ax_start_pred = axes[2]
            start_pred_activations = place_cell_probs[i, 0, :].cpu().numpy()  # t=0

            # Decode position from predicted activations
            start_decoded_pos = (
                self.decode_position_from_place_cells(
                    place_cell_probs[i : i + 1, 0:1, :]  # Keep batch and time dims
                )[0, 0]
                .cpu()
                .numpy()
            )  # Extract single position

            scatter_start_pred = ax_start_pred.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=start_pred_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark DECODED agent position
            ax_start_pred.scatter(
                start_decoded_pos[0],
                start_decoded_pos[1],
                c="green",
                s=200,
                marker="*",
                label="Decoded Pos",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_start_pred.set_xlabel("X Position")
            ax_start_pred.set_ylabel("Y Position")
            ax_start_pred.set_title(
                f"START: Predicted Place Cells\n({start_decoded_pos[0]:.3f}, {start_decoded_pos[1]:.3f})"
            )
            ax_start_pred.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_start_pred.grid(True, alpha=0.3)
            ax_start_pred.set_aspect("equal")
            ax_start_pred.set_xlim(xlim)
            ax_start_pred.set_ylim(ylim)
            cbar_start_pred = plt.colorbar(scatter_start_pred, ax=ax_start_pred)
            cbar_start_pred.set_label("Activation")

            # END TIME - Ground Truth
            ax_end_gt = axes[3]
            end_gt_activations = target_place_cells[i, -1, :].cpu().numpy()  # t=final
            end_pos = target_positions[i, -1].cpu().numpy()

            scatter_end_gt = ax_end_gt.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=end_gt_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark agent position
            ax_end_gt.scatter(
                end_pos[0],
                end_pos[1],
                c="red",
                s=200,
                marker="*",
                label="Agent",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_end_gt.set_xlabel("X Position")
            ax_end_gt.set_ylabel("Y Position")
            ax_end_gt.set_title(
                f"END: GT Place Cells\n({end_pos[0]:.3f}, {end_pos[1]:.3f})"
            )
            ax_end_gt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_end_gt.grid(True, alpha=0.3)
            ax_end_gt.set_aspect("equal")
            ax_end_gt.set_xlim(xlim)
            ax_end_gt.set_ylim(ylim)
            cbar_end_gt = plt.colorbar(scatter_end_gt, ax=ax_end_gt)
            cbar_end_gt.set_label("Activation")

            # END TIME - Predicted
            ax_end_pred = axes[4]
            end_pred_activations = place_cell_probs[i, -1, :].cpu().numpy()  # t=final

            # Decode position from predicted activations
            end_decoded_pos = (
                self.decode_position_from_place_cells(
                    place_cell_probs[i : i + 1, -1:, :]  # Keep batch and time dims
                )[0, 0]
                .cpu()
                .numpy()
            )  # Extract single position

            scatter_end_pred = ax_end_pred.scatter(
                place_centers[:, 0],
                place_centers[:, 1],
                c=end_pred_activations,
                cmap="viridis",
                s=30,
                alpha=0.8,
            )
            # Mark DECODED agent position
            ax_end_pred.scatter(
                end_decoded_pos[0],
                end_decoded_pos[1],
                c="red",
                s=200,
                marker="*",
                label="Decoded Pos",
                zorder=5,
                edgecolors="black",
                linewidth=2,
            )

            ax_end_pred.set_xlabel("X Position")
            ax_end_pred.set_ylabel("Y Position")
            ax_end_pred.set_title(
                f"END: Predicted Place Cells\n({end_decoded_pos[0]:.3f}, {end_decoded_pos[1]:.3f})"
            )
            ax_end_pred.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")
            ax_end_pred.grid(True, alpha=0.3)
            ax_end_pred.set_aspect("equal")
            ax_end_pred.set_xlim(xlim)
            ax_end_pred.set_ylim(ylim)
            cbar_end_pred = plt.colorbar(scatter_end_pred, ax=ax_end_pred)
            cbar_end_pred.set_label("Activation")

            # Add overall title
            fig.suptitle(
                f"Epoch {trainer.current_epoch} - Trajectory {i+1}", fontsize=16
            )

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for legends below

            figs.append(fig)

        # Log to wandb
        if trainer.logger is not None and hasattr(
            trainer.logger, "experiment"
        ):  # Check if wandb logger
            for i, fig in enumerate(figs):
                trainer.logger.experiment.log(
                    {
                        f"trajectory_analysis_{i+1}": wandb.Image(fig),
                    }
                )

        # Close figures to free memory
        for fig in figs:
            plt.close(fig)
