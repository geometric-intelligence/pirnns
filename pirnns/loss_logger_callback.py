import lightning as L
import json
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only


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
