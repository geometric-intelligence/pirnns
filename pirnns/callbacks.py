import lightning as L
import json
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
from typing import Any


class LossLoggerCallback(L.Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.train_losses_epoch = []
        self.val_losses_epoch = []
        self.epochs = []

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
            hidden_states, outputs = pl_module.model(  # type: ignore
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
