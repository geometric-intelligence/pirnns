import os
import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple



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
    import sys
    import os
    sys.path.append('/home/facosta/pirnns/pirnns')  # Adjust path as needed
    
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
        
        # Compute statistics for this experiment
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

def plot_sweep_ood_results(
    sweep_results: Dict,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    paper_ready: bool = False
) -> None:
    """
    Plot OOD generalization results for all experiments in sweep.
    
    Args:
        sweep_results: Results from evaluate_sweep_ood_generalization
        figsize: Figure size
        save_path: Optional path to save figure
        paper_ready: If True, create cleaner version for publication
    """
    plt.figure(figsize=figsize)
    
    # Color palette for experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(sweep_results)))
    
    training_length = None
    
    for i, (exp_name, results) in enumerate(sweep_results.items()):
        test_lengths = results["test_lengths"]
        mean_errors = results["mean_errors"]
        std_errors = results["std_errors"]
        
        if training_length is None:
            training_length = results["training_length"]
        
        # Create label with timescales info
        timescales_config = results["timescales_config"]
        if paper_ready:
            # Simplified labels for paper
            if "discrete_single" in exp_name:
                tau_val = timescales_config["values"][0]
                label = f"Single τ={tau_val:.3f}"
            elif "discrete_two" in exp_name:
                label = "Two timescales"
            elif "discrete_four" in exp_name:
                label = "Four timescales"
            elif "continuous" in exp_name:
                label = f"Continuous ({timescales_config.get('distribution', 'powerlaw')})"
            else:
                label = exp_name
        else:
            # Detailed labels for exploration
            if timescales_config["type"] == "discrete":
                if len(timescales_config["values"]) == 1:
                    label = f"{exp_name} (τ={timescales_config['values'][0]:.3f})"
                else:
                    label = f"{exp_name} ({len(timescales_config['values'])} timescales)"
            else:
                label = f"{exp_name} ({timescales_config['type']})"
        
        # Plot individual seeds (very transparent) only if not paper ready
        if not paper_ready:
            for seed_name, seed_errors in results["results"].items():
                plt.plot(test_lengths, seed_errors, color=colors[i], 
                        alpha=0.2, linewidth=1, markersize=2)
        
        # Plot mean with error bars
        plt.errorbar(
            test_lengths, mean_errors, yerr=std_errors,
            color=colors[i], linestyle="-", linewidth=2, markersize=4,
            capsize=3, capthick=1, label=label
        )
    
    # Mark training length
    if training_length and not paper_ready:
        plt.axvline(
            training_length, color="red", linestyle="--", alpha=0.7,
            label=f"Training length ({training_length})"
        )
    
    plt.xlabel("Sequence Length (time steps)", fontsize=12)
    plt.ylabel("Position Decoding Error (m)", fontsize=12)
    
    if paper_ready:
        plt.title("OOD Generalization Performance", fontsize=14)
    else:
        plt.title("OOD Generalization by Timescale Configuration", fontsize=14)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    
    plt.show()

# Convenience function for your notebook
def analyze_sweep_ood(
    models,
    test_lengths=None,
    save_path=None,
    paper_ready=False,
    num_test_trajectories=100,
    device="cuda",
):
    """
    One-liner function for your notebook using pre-loaded models.
    """
    if test_lengths is None:
        test_lengths = [25, 30, 35, 40, 50, 60, 80, 100]  # Default test lengths
    
    print("Evaluating OOD generalization for loaded models...")
    results = evaluate_sweep_ood_generalization(
        models,
        test_lengths,
        num_test_trajectories=num_test_trajectories,
        device=device,
    )
    
    print("\nPlotting results...")
    plot_sweep_ood_results(results, save_path=save_path, paper_ready=paper_ready)
    
    return results


