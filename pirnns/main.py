import torch.nn as nn
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import yaml
import torch
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loss_logger_callback import LossLoggerCallback

from datamodule import PathIntegrationDataModule

from pirnns.rnns.rnn import RNN, RNNLightning
from pirnns.topornns.hypergraph_rnn import (
    Hypergraph,
    HypergraphRNN,
    HypergraphRNNLightning,
)

import datetime

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
print("Log directory:", log_dir)


def create_vanilla_rnn_model(config: dict):
    """Create vanilla PathIntRNN model and lightning module."""

    model = RNN(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        alpha=config["alpha"],
        activation=getattr(nn, config.get("activation", "Tanh")),
    )

    lightning_module = RNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
        step_size=config.get("step_size", 100),
        gamma=config.get("gamma", 0.5),
    )

    return model, lightning_module


def create_hypergraph_rnn_model(config: dict):
    """Create HypergraphRNN model and lightning module."""
    # Import hypergraph RNN components

    # Create hypergraph structure
    if config.get("hypergraph_structure") == "random":
        hypergraph = Hypergraph.create_random_hypergraph(
            num_nodes=config["num_nodes"],
            num_hyperedges=config["num_hyperedges"],
            max_hyperedge_size=config.get("max_hyperedge_size", 3),
        )
    else:
        raise ValueError(
            f"Unknown hypergraph structure: {config.get('hypergraph_structure')}"
        )

    print(
        f"Hypergraph created with {hypergraph.num_nodes} nodes and {hypergraph.num_hyperedges} hyperedges"
    )

    hypergraph.to_device(config["device"])

    model = HypergraphRNN(
        input_size=config["input_size"],
        output_size=config["output_size"],
        hypergraph=hypergraph,
        alpha_node=config["alpha_node"],
        alpha_hyperedge=config["alpha_hyperedge"],
        activation=getattr(nn, config.get("activation", "Tanh")),
    )

    lightning_module = HypergraphRNNLightning(
        model=model,
        learning_rate=config["learning_rate"],
        step_size=config.get("step_size", 100),
        gamma=config.get("gamma", 0.5),
        weight_decay=config.get("weight_decay", 0.0),
    )

    return model, lightning_module


def main(config: dict):
    model_type = config.get("model_type", "vanilla").lower()

    # Generate unique run identifier
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting training run: {run_id}")
    print(f"Model type: {model_type}")

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=f"{config['name']}_{model_type}_{run_id}",
        dir=log_dir,
        save_dir=log_dir,
    )
    print("Wandb initialized. Find logs at: ", log_dir)
    print(f"Wandb run name: {config['name']}_{model_type}_{run_id}")

    datamodule = PathIntegrationDataModule(
        num_trajectories=config["num_trajectories"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_val_split=config["train_val_split"],
        start_time=config["start_time"],
        end_time=config["end_time"],
        num_time_steps=config["num_time_steps"],
        arena_L=config["arena_L"],
        mu_speed=config["mu_speed"],
        sigma_speed=config["sigma_speed"],
        tau_vel=config["tau_vel"],
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
    else:  # hypergraph
        model, lightning_module = create_hypergraph_rnn_model(config)
        print("HypergraphRNN initialized")

    lightning_module.to(config["device"])
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

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, loss_logger],
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
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
        description="Run RNN training (vanilla or hypergraph)"
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
