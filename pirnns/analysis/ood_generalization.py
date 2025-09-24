"""
Minimal OOD generalization utilities - only the essential functions.
"""

import os
import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
from datamodule import PathIntegrationDataModule
from pirnns.rnns.rnn import RNN
from pirnns.rnns.multitimescale_rnn import MultiTimescaleRNN


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
    
    # Get device from model
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            inputs, target_positions, target_place_cells = batch
            
            # Move batch to model device
            inputs = inputs.to(model_device)
            target_positions = target_positions.to(model_device)
            target_place_cells = target_place_cells.to(model_device)

            # Get model outputs
            if isinstance(model, RNN):
                _, outputs = model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )
            elif isinstance(model, MultiTimescaleRNN):
                _, outputs = model(
                    inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                )
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

def evaluate_sweep_ood_generalization(
    models: Dict,
    test_lengths: List[int],
    num_test_trajectories: int = 100,
    device: str = "cpu"
) -> Dict:
    """
    Evaluate OOD generalization for all experiments in a sweep using pre-loaded models.
    
    Args:
        models: Dictionary from load_experiment_sweep
        test_lengths: List of sequence lengths to test
        num_test_trajectories: Number of test trajectories per length
        device: Device to run evaluation on
        
    Returns:
        Dictionary with results for each experiment
    """
    from analysis.ood_generalization import create_test_dataloader, compute_decoding_error
    from datamodule import PathIntegrationDataModule
    
    print(f"Evaluating OOD generalization for {len(models)} experiments")
    print(f"Test lengths: {test_lengths}")
    print(f"Device: {device}")
    
    # Store results for each experiment
    sweep_results = {}
    
    for exp_name, seeds in models.items():
        print(f"\nProcessing experiment: {exp_name}")
        
        # Get config from first seed
        sample_config = list(seeds.values())[0]['config']
        training_length = sample_config['num_time_steps']
        
        print(f"  Training length: {training_length}")
        print(f"  Testing {len(seeds)} seeds on lengths: {test_lengths}")
        
        # Store results for each seed in this experiment
        exp_results = {}
        
        for seed, seed_data in seeds.items():
            print(f"    Processing seed {seed}...")
            model = seed_data['model']
            config = seed_data['config']
            
            try:
                # Create datamodule to get place cell centers (using training config)
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
                place_cell_centers = datamodule.place_cell_centers.to(device)
                
                # Test on different lengths
                seed_results = []
                for test_length in test_lengths:
                    print(f"      Testing length {test_length}...")
                    
                    test_dataloader = create_test_dataloader(
                        config, test_length, num_test_trajectories
                    )
                    
                    error = compute_decoding_error(
                        model, test_dataloader, place_cell_centers, config["decode_k"]
                    )
                    seed_results.append(error)
                    print(f"        Error: {error:.4f}")
                
                exp_results[f"seed_{seed}"] = seed_results
                
            except Exception as e:
                print(f"    Error processing seed {seed}: {str(e)}")
                continue
        
        # Compute statistics for this experiment - FIX THE .tolist() ERROR
        if exp_results:
            errors_by_length = np.array(list(exp_results.values()))  # [n_seeds, n_lengths]
            mean_errors = np.mean(errors_by_length, axis=0)
            std_errors = np.std(errors_by_length, axis=0)
            
            # Convert to lists safely
            mean_errors = mean_errors.tolist() if hasattr(mean_errors, 'tolist') else list(mean_errors)
            std_errors = std_errors.tolist() if hasattr(std_errors, 'tolist') else list(std_errors)
        else:
            mean_errors = [np.nan] * len(test_lengths)
            std_errors = [np.nan] * len(test_lengths)
        
        sweep_results[exp_name] = {
            "test_lengths": test_lengths,
            "training_length": training_length,
            "results": exp_results,
            "mean_errors": mean_errors,
            "std_errors": std_errors,
            "config": sample_config,
            "timescales_config": sample_config["timescales_config"]
        }
    
    return sweep_results

def analyze_sweep_ood(models, test_lengths=None, save_path=None, paper_ready=False, 
                     num_test_trajectories=100, device="cpu"):
    """
    One-liner function for your notebook using pre-loaded models.
    """
    if test_lengths is None:
        test_lengths = [25, 30, 35, 40, 50, 60, 80, 100]  # Default test lengths
    
    print("Evaluating OOD generalization for loaded models...")
    results = evaluate_sweep_ood_generalization(
        models, test_lengths, num_test_trajectories=num_test_trajectories, device=device
    )
    
    print("\nPlotting results...")
    plot_sweep_ood_results(results, save_path=save_path, paper_ready=paper_ready)
    
    return results