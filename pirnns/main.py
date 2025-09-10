import torch.nn as nn
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import yaml
import torch
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from callbacks import (
    LossLoggerCallback,
    PositionDecodingCallback,
    TrajectoryVisualizationCallback,
    TimescaleVisualizationCallback,
)

from datamodule import PathIntegrationDataModule

from pirnns.rnns.rnn import RNN, RNNLightning
from pirnns.rnns.multitimescale_rnn import MultiTimescaleRNN, MultiTimescaleRNNLightning

import datetime

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
print("Log directory:", log_dir)


def create_vanilla_rnn_model(config: dict):
    """Create vanilla PathIntRNN model and lightning module."""

    model = RNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["num_place_cells"],
        alpha=config["alpha"],
        activation=getattr(nn, config["activation"]),
    )

    lightning_module = RNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=config["step_size"],
        gamma=config["gamma"],
    )

    return model, lightning_module


def create_multitimescale_rnn_model(config: dict):
    """Create MultiTimescaleRNN model and lightning module."""
    model = MultiTimescaleRNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["num_place_cells"],
        dt=config["dt"],
        timescales_config=config["timescales_config"],
        activation=getattr(nn, config["activation"]),
    )

    lightning_module = MultiTimescaleRNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        step_size=config["step_size"],
        gamma=config["gamma"],
    )

    return model, lightning_module


def main(config: dict):
    # Set global seed - this handles all randomness sources
    seed_everything(config["seed"], workers=True)
    print(f"Global seed set to: {config['seed']}")

    model_type = config.get("model_type", "vanilla").lower()

    # Generate unique run identifier
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting training run: {run_id}")
    print(f"Model type: {model_type}")

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=f"{config['project_name']}_{model_type}_{run_id}",
        dir=log_dir,
        save_dir=log_dir,
        config=config,
    )
    print("Wandb initialized. Find logs at: ", log_dir)
    print(f"Wandb run name: {config['project_name']}_{model_type}_{run_id}")

    datamodule = PathIntegrationDataModule(
        num_trajectories=config["num_trajectories"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_val_split=config["train_val_split"],
        velocity_representation=config["velocity_representation"],
        dt=config["dt"],
        num_time_steps=config["num_time_steps"],
        arena_size=config["arena_size"],
        speed_scale=config["speed_scale"],
        sigma_speed=config["sigma_speed"],
        tau_vel=config["tau_vel"],
        sigma_rotation=config["sigma_rotation"],
        border_region=config["border_region"],
        num_place_cells=config["num_place_cells"],
        place_cell_rf=config["place_cell_rf"],
        surround_scale=config["surround_scale"],
        DoG=config["DoG"],
        trajectory_type=config["trajectory_type"],
        place_cell_layout=config["place_cell_layout"],
    )

    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("Data prepared")

    # Create model based on type
    if model_type == "vanilla":
        model, lightning_module = create_vanilla_rnn_model(config)
        print("Vanilla PathIntRNN initialized")
    elif model_type == "multitimescale":
        model, lightning_module = create_multitimescale_rnn_model(config)
        print("MultiTimescaleRNN initialized")
        timescale_stats = model.get_timescale_stats()
        print(f"Timescale statistics: {timescale_stats}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"{model_type.capitalize()} Lightning module initialized")

    run_dir = os.path.join(log_dir, "checkpoints", f"{model_type}_{run_id}")

    @rank_zero_only
    def create_directories():
        os.makedirs(run_dir, exist_ok=True)

    create_directories()

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="best-model-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    loss_logger = LossLoggerCallback(save_dir=run_dir)

    position_decoding_callback = PositionDecodingCallback(
        place_cell_centers=datamodule.place_cell_centers,
        decode_k=config["decode_k"],
        log_every_n_epochs=config["log_every_n_epochs"],
    )

    trajectory_viz_callback = TrajectoryVisualizationCallback(
        place_cell_centers=datamodule.place_cell_centers,
        arena_size=config["arena_size"],
        decode_k=config["decode_k"],
        log_every_n_epochs=config["viz_log_every_n_epochs"],
        num_trajectories_to_plot=3,
    )

    callbacks = [
        checkpoint_callback,
        loss_logger,
        position_decoding_callback,
        trajectory_viz_callback,
    ]

    if model_type == "multitimescale":
        timescale_viz_callback = TimescaleVisualizationCallback()
        callbacks.append(timescale_viz_callback)

    device_str = config["device"]
    if device_str.startswith("cuda:"):
        device_id = int(device_str.split(":")[1])
        devices = [device_id]
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "auto"

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        default_root_dir=log_dir,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
        strategy="auto",
    )

    print("Trainer initialized")
    print("Training...")

    trainer.fit(lightning_module, train_loader, val_loader)

    print("Training complete!")

    # Only save on rank 0
    @rank_zero_only
    def save_additional_artifacts():
        model_path = os.path.join(run_dir, f"final_model_{run_id}.pth")
        torch.save(lightning_module.model.state_dict(), model_path)

        config_path = os.path.join(run_dir, f"config_{run_id}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        print(f"All artifacts saved to: {run_dir}")

    save_additional_artifacts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RNN training (vanilla, multitimescale)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path containing the config file",
    )
    args = parser.parse_args()

    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    main(config)
