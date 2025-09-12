"""
Out-of-distribution (OOD) generalization analysis for trained RNN models.

Evaluates how well models generalize to longer sequence lengths than 
they were trained on by computing position decoding error.
"""

import os
import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
# Import from parent directories
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datamodule import PathIntegrationDataModule
from main import create_vanilla_rnn_model, create_multitimescale_rnn_model
from pirnns.rnns.rnn import RNN
from pirnns.rnns.multitimescale_rnn import MultiTimescaleRNN


def load_experiment_metadata(
    experiment_dir: str,
) -> Dict:
    """
    Load experiment metadata and configuration.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary containing experiment metadata and base config
    """
    # Load experiment metadata
    metadata_path = os.path.join(experiment_dir, "experiment_metadata.yaml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No experiment metadata found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Load base config
    config_path = os.path.join(experiment_dir, "configs", "base_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No base config found at {config_path}")
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    return {
        'metadata': metadata,
        'config': base_config,
        'seeds': metadata['seeds']
    }


def load_trained_model(
    seed_dir: str,
    config: Dict,
) -> Tuple[torch.nn.Module, torch.Tensor]:
    """
    Load trained model from best checkpoint.
    
    Args:
        seed_dir: Path to seed directory
        config: Model configuration
        
    Returns:
        Tuple of (model, place_cell_centers)
    """
    checkpoints_dir = os.path.join(seed_dir, "checkpoints")
    best_ckpt_pattern = os.path.join(checkpoints_dir, "best-model-*.ckpt")
    best_ckpt_files = glob.glob(best_ckpt_pattern)
    
    if not best_ckpt_files:
        raise FileNotFoundError(f"No best checkpoint found in {checkpoints_dir}")
    
    best_ckpt_path = best_ckpt_files[0]  # Should only be one best checkpoint
    
    # Create model architecture
    model_type = config["model_type"]
    
    if model_type == "vanilla":
        model, lightning_module = create_vanilla_rnn_model(config)
    elif model_type == "multitimescale":
        model, lightning_module = create_multitimescale_rnn_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location='cpu')
    lightning_module.load_state_dict(checkpoint['state_dict'])
    model = lightning_module.model
    
    # Create datamodule to get place cell centers
    datamodule = PathIntegrationDataModule(
        num_trajectories=config["num_trajectories"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_val_split=config["train_val_split"],
        velocity_representation=config["velocity_representation"],
        dt=config["dt"],
        num_time_steps=config["num_time_steps"],  # Original training length
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
    
    return model, datamodule.place_cell_centers


def create_test_dataloader(
    config: Dict,
    test_length: int,
    num_test_trajectories: int = 100,
) -> DataLoader:
    """
    Create test dataloader with longer sequences.
    
    Args:
        config: Base configuration
        test_length: Number of time steps for test sequences
        num_test_trajectories: Number of test trajectories to generate
        
    Returns:
        Test dataloader
    """

    test_config = config.copy()
    test_config["num_time_steps"] = test_length
    test_config["num_trajectories"] = num_test_trajectories
    test_config["batch_size"] = min(config["batch_size"], num_test_trajectories)
    
    test_datamodule = PathIntegrationDataModule(
        num_trajectories=test_config["num_trajectories"],
        batch_size=test_config["batch_size"],
        num_workers=1,  # Reduce workers for test
        train_val_split=1.0,  # Use all data for testing
        velocity_representation=test_config["velocity_representation"],
        dt=test_config["dt"],
        num_time_steps=test_config["num_time_steps"],
        arena_size=test_config["arena_size"],
        speed_scale=test_config["speed_scale"],
        sigma_speed=test_config["sigma_speed"],
        tau_vel=test_config["tau_vel"],
        sigma_rotation=test_config["sigma_rotation"],
        border_region=test_config["border_region"],
        num_place_cells=test_config["num_place_cells"],
        place_cell_rf=test_config["place_cell_rf"],
        surround_scale=test_config["surround_scale"],
        DoG=test_config["DoG"],
        trajectory_type=test_config["trajectory_type"],
        place_cell_layout=test_config["place_cell_layout"],
    )
    
    test_datamodule.setup()
    return test_datamodule.train_dataloader()


def decode_position_from_place_cells(
    activation: torch.Tensor,
    place_cell_centers: torch.Tensor,
    decode_k: int = 3,
) -> torch.Tensor:
    """
    Decode position from place cell activations using top-k method.
    
    Same implementation as PositionDecodingCallback.
    """
    centers = place_cell_centers.to(activation.device)
    _, idxs = torch.topk(activation, k=decode_k, dim=-1)  # [B, T, k]
    pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
    return pred_pos


def compute_decoding_error(
    model: torch.nn.Module,
    dataloader,
    place_cell_centers: torch.Tensor,
    decode_k: int,
) -> float:
    """
    Compute position decoding error on test data.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        place_cell_centers: Place cell center positions
        decode_k: Number of top place cells for decoding
        
    Returns:
        Mean position decoding error (L2 distance)
    """
    model.eval()
    total_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, target_positions, target_place_cells = batch
            
            # Get model outputs
            if isinstance(model, RNN):
                _, outputs = model(inputs=inputs, place_cells_0=target_place_cells[:, 0, :])
            elif isinstance(model, MultiTimescaleRNN):
                _, outputs = model(inputs=inputs, place_cells_0=target_place_cells[:, 0, :])
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            # Convert to probabilities and decode positions
            place_cell_probs = torch.softmax(outputs, dim=-1)
            predicted_positions = decode_position_from_place_cells(
                place_cell_probs, place_cell_centers, decode_k
            )
            
            
            position_error = torch.sqrt(
                ((target_positions - predicted_positions) ** 2).sum(-1)
            )
            
            total_error += position_error.sum().item()
            total_samples += position_error.numel()
    
    return total_error / total_samples


def evaluate_ood_generalization(
    experiment_dir: str, 
    test_lengths: List[int],
    num_test_trajectories: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Evaluate OOD generalization across different sequence lengths.
    
    Args:
        experiment_dir: Path to experiment directory
        test_lengths: List of sequence lengths to test
        num_test_trajectories: Number of test trajectories per length
        device: Device to run evaluation on
        
    Returns:
        Dictionary with results for plotting
    """
    print(f"Evaluating OOD generalization for experiment: {experiment_dir}")
    print(f"Test lengths: {test_lengths}")
    
    # Load experiment metadata and config
    exp_data = load_experiment_metadata(experiment_dir)
    config = exp_data['config']
    seeds = exp_data['seeds']
    
    print(f"Found {len(seeds)} seeds: {seeds}")
    print(f"Original training length: {config['num_time_steps']}")
    
    # Store results for each seed and test length
    results = {}
    
    for seed in seeds:
        print(f"\nProcessing seed {seed}...")
        seed_dir = os.path.join(experiment_dir, f"seed_{seed}")
        
        if not os.path.exists(seed_dir):
            print(f"Warning: Seed directory {seed_dir} not found, skipping")
            continue
        
        try:
            model, place_cell_centers = load_trained_model(seed_dir, config)
            model.to(device)
            place_cell_centers = place_cell_centers.to(device)
            
            seed_results = []
            for test_length in test_lengths:
                print(f"  Testing length {test_length}...")
                
                test_dataloader = create_test_dataloader(config, test_length, num_test_trajectories)
                
                error = compute_decoding_error(
                    model, test_dataloader, place_cell_centers, config["decode_k"]
                )
                seed_results.append(error)
                print(f"    Error: {error:.4f}")
            
            results[f"seed_{seed}"] = seed_results
            
        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
            continue
    
    if results:
        errors_by_length = np.array(list(results.values()))  # [n_seeds, n_lengths]
        mean_errors = np.mean(errors_by_length, axis=0)
        std_errors = np.std(errors_by_length, axis=0)
    else:
        mean_errors = [np.nan] * len(test_lengths)
        std_errors = [np.nan] * len(test_lengths)
    
    return {
        'experiment_dir': experiment_dir,
        'test_lengths': test_lengths,
        'training_length': config['num_time_steps'],
        'results': results,
        'mean_errors': mean_errors.tolist(),
        'std_errors': std_errors.tolist(),
        'config': config
    }


def plot_ood_results(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot OOD generalization results with error bars.
    
    Args:
        results: Results dictionary from evaluate_ood_generalization
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    test_lengths = results['test_lengths']
    mean_errors = results['mean_errors']
    std_errors = results['std_errors']
    training_length = results['training_length']
    
    # Plot individual seed results
    for seed_name, seed_errors in results['results'].items():
        plt.plot(test_lengths, seed_errors, 'o-', alpha=0.3, linewidth=1, markersize=4)
    
    # Plot mean with error bars
    plt.errorbar(test_lengths, mean_errors, yerr=std_errors, 
                 color='k', linestyle='-', linewidth=2, markersize=6, capsize=5, capthick=2,
                 label=f'Mean ± std (n={len(results["results"])} seeds)')
    
    # Mark training length
    plt.axvline(training_length, color='red', linestyle='--', alpha=0.7,
                label=f'Training length ({training_length})')
    
    plt.xlabel('Sequence Length (time steps)')
    plt.ylabel('Position Decoding Error (m)')
    plt.title('Out-of-Distribution Generalization: Longer Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    exp_dir_name = os.path.basename(results['experiment_dir'])
    model_type = results['config']['model_type']
    plt.text(0.02, 0.98, f'Experiment: {exp_dir_name}\nModel: {model_type}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example: Evaluate OOD generalization for an experiment
    experiment_dir = "logs/experiments/expt_20250912_023600"  # Replace with your experiment
    test_lengths = [150, 200, 250, 300, 400, 500]  # Assuming training was on 100 steps
    
    results = evaluate_ood_generalization(experiment_dir, test_lengths)
    save_path = os.path.join(experiment_dir, "ood_generalization_results.png")
    plot_ood_results(results, save_path=save_path)
