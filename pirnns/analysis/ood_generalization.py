"""
Comprehensive OOD (Out-of-Distribution) Generalization Analysis

This module provides an object-oriented API for evaluating how well trained models 
generalize to longer sequence lengths than seen during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from dataclasses import dataclass

from datamodule import PathIntegrationDataModule
from pirnns.rnns.rnn import RNN
from pirnns.rnns.multitimescale_rnn import MultiTimescaleRNN


@dataclass
class OODResults:
    """Container for OOD evaluation results."""
    test_lengths: List[int]
    training_length: int
    experiment_results: Dict[str, Dict[str, List[float]]]  # exp_name -> {seed -> errors}
    mean_errors: Dict[str, List[float]]  # exp_name -> mean_errors_per_length
    std_errors: Dict[str, List[float]]   # exp_name -> std_errors_per_length
    configs: Dict[str, Dict]             # exp_name -> config


class OODEvaluator:
    """
    Object-oriented interface for OOD generalization evaluation.
    
    This class encapsulates models, configurations, and provides methods for
    evaluation and visualization of out-of-distribution generalization performance.
    """
    
    def __init__(
        self, 
        models: Dict, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
    ):
        """
        Initialize the OOD evaluator.
        
        Args:
            models: Dictionary from load_experiment_sweep with structure:
                   {exp_name: {seed: {'model': model, 'config': config}}}
            device: Device to run evaluation on
            verbose: Whether to print progress information
        """
        self.models = models
        self.device = device
        self.verbose = verbose
        self._place_cell_centers_cache = {}
        
        if self.verbose:
            print(f"Initialized OODEvaluator with {len(models)} experiments on {device}")
    
    def _get_place_cell_centers(self, config: Dict) -> torch.Tensor:
        """Get place cell centers for a config, with caching."""
        config_key = str(sorted(config.items()))
        
        if config_key not in self._place_cell_centers_cache:
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
            self._place_cell_centers_cache[config_key] = datamodule.place_cell_centers
        
        return self._place_cell_centers_cache[config_key].to(self.device)
    
    def _create_test_dataloader(
        self,
        config: Dict,
        test_length: int,
        num_test_trajectories: int = 100,
    ) -> DataLoader:
        """Create test dataloader with specified sequence length."""
        test_config = config.copy()
        test_config.update({
            "num_time_steps": test_length,
            "num_trajectories": num_test_trajectories,
            "batch_size": min(config["batch_size"], num_test_trajectories)
        })

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
    
    def _decode_position_from_place_cells(
        self,
        activation: torch.Tensor,
        place_cell_centers: torch.Tensor,
        decode_k: int = 3,
    ) -> torch.Tensor:
        """Decode position from place cell activations using top-k method."""
        centers = place_cell_centers.to(activation.device)
        _, idxs = torch.topk(activation, k=decode_k, dim=-1)  # [B, T, k]
        pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
        return pred_pos
    
    def _compute_decoding_error(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        place_cell_centers: torch.Tensor,
        decode_k: int,
    ) -> float:
        """Compute position decoding error on test data."""
        model.eval()
        total_error = 0.0
        total_samples = 0
        
        model_device = next(model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                inputs, target_positions, target_place_cells = batch
                
                # Move batch to model device
                inputs = inputs.to(model_device)
                target_positions = target_positions.to(model_device)
                target_place_cells = target_place_cells.to(model_device)

                # Get model outputs
                if isinstance(model, (RNN, MultiTimescaleRNN)):
                    _, outputs = model(
                        inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
                    )
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")

                # Convert to probabilities and decode positions
                place_cell_probs = torch.softmax(outputs, dim=-1)
                predicted_positions = self._decode_position_from_place_cells(
                    place_cell_probs, place_cell_centers, decode_k
                )

                position_error = torch.sqrt(
                    ((target_positions - predicted_positions) ** 2).sum(-1)
                )

                total_error += position_error.sum().item()
                total_samples += position_error.numel()

        return total_error / total_samples
    
    def evaluate(
        self,
        test_lengths: List[int],
        num_test_trajectories: int = 100,
        experiments: Optional[List[str]] = None
    ) -> OODResults:
        """
        Evaluate OOD generalization for experiments.
        
        Args:
            test_lengths: List of sequence lengths to test
            num_test_trajectories: Number of test trajectories per length
            experiments: List of experiment names to evaluate (None = all)
            
        Returns:
            OODResults object containing all evaluation results
        """
        if experiments is None:
            experiments = list(self.models.keys())
        
        if self.verbose:
            print(f"Evaluating OOD generalization for {len(experiments)} experiments")
            print(f"Test lengths: {test_lengths}")
            print(f"Device: {self.device}")
        
        experiment_results = {}
        mean_errors = {}
        std_errors = {}
        configs = {}
        
        for exp_name in experiments:
            if exp_name not in self.models:
                print(f"Warning: Experiment '{exp_name}' not found in models")
                continue
                
            seeds = self.models[exp_name]
            if self.verbose:
                print(f"\nProcessing experiment: {exp_name}")
            
            # Get config from first seed
            sample_config = list(seeds.values())[0]['config']
            training_length = sample_config['num_time_steps']
            configs[exp_name] = sample_config
            
            if self.verbose:
                print(f"  Training length: {training_length}")
                print(f"  Testing {len(seeds)} seeds on lengths: {test_lengths}")
            
            # Get place cell centers for this config
            place_cell_centers = self._get_place_cell_centers(sample_config)
            
            # Store results for each seed in this experiment
            exp_results = {}
            
            for seed, seed_data in seeds.items():
                if self.verbose:
                    print(f"    Processing seed {seed}...")
                
                model = seed_data['model']
                config = seed_data['config']
                
                try:
                    # Test on different lengths
                    seed_results = []
                    for test_length in test_lengths:
                        if self.verbose:
                            print(f"      Testing length {test_length}...")
                        
                        test_dataloader = self._create_test_dataloader(
                            config, test_length, num_test_trajectories
                        )
                        
                        error = self._compute_decoding_error(
                            model, test_dataloader, place_cell_centers, config["decode_k"]
                        )
                        seed_results.append(error)
                        
                        if self.verbose:
                            print(f"        Error: {error:.4f}")
                    
                    exp_results[f"seed_{seed}"] = seed_results
                    
                except Exception as e:
                    print(f"    Error processing seed {seed}: {str(e)}")
                    continue
            
            # Compute statistics for this experiment
            if exp_results:
                errors_by_length = np.array(list(exp_results.values()))  # [n_seeds, n_lengths]
                mean_errs = np.mean(errors_by_length, axis=0)
                std_errs = np.std(errors_by_length, axis=0)
                
                # Convert to lists safely
                mean_errors[exp_name] = (
                    mean_errs.tolist() if hasattr(mean_errs, 'tolist') else list(mean_errs)
                )
                std_errors[exp_name] = (
                    std_errs.tolist() if hasattr(std_errs, 'tolist') else list(std_errs)
                )
            else:
                mean_errors[exp_name] = [np.nan] * len(test_lengths)
                std_errors[exp_name] = [np.nan] * len(test_lengths)
            
            experiment_results[exp_name] = exp_results
        
        # Get training length from first experiment
        training_length = list(configs.values())[0]['num_time_steps'] if configs else None
        
        return OODResults(
            test_lengths=test_lengths,
            training_length=training_length,
            experiment_results=experiment_results,
            mean_errors=mean_errors,
            std_errors=std_errors,
            configs=configs
        )
    
    def plot_results(
        self,
        results: OODResults,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        paper_ready: bool = False,
        show_individual_seeds: bool = None,
        show_training_length: bool = None
    ) -> None:
        """
        Plot OOD generalization results.
        
        Args:
            results: Results from evaluate()
            figsize: Figure size
            save_path: Optional path to save figure
            paper_ready: If True, create cleaner version for publication
            show_individual_seeds: Whether to show individual seed trajectories
            show_training_length: Whether to show vertical line at training length
        """
        if show_individual_seeds is None:
            show_individual_seeds = not paper_ready
        if show_training_length is None:
            show_training_length = not paper_ready
            
        plt.figure(figsize=figsize)
        
        # Color palette for experiments
        colors = plt.cm.tab10(np.linspace(0, 1, len(results.experiment_results)))
        
        for i, exp_name in enumerate(results.experiment_results.keys()):
            test_lengths = results.test_lengths
            mean_errs = results.mean_errors[exp_name]
            std_errs = results.std_errors[exp_name]
            
            # Create label with timescales info
            config = results.configs[exp_name]
            timescales_config = config["timescales_config"]
            
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
            
            # Plot individual seeds (very transparent)
            if show_individual_seeds:
                for seed_name, seed_errors in results.experiment_results[exp_name].items():
                    plt.plot(test_lengths, seed_errors, color=colors[i], 
                            alpha=0.2, linewidth=1, markersize=2)
            
            # Plot mean with error bars
            plt.errorbar(
                test_lengths, mean_errs, yerr=std_errs,
                color=colors[i], linestyle="-", linewidth=2, markersize=4,
                capsize=3, capthick=1, label=label
            )
        
        # Mark training length
        if show_training_length and results.training_length:
            plt.axvline(
                results.training_length, color="red", linestyle="--", alpha=0.7,
                label=f"Training length ({results.training_length})"
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
            if self.verbose:
                print(f"Figure saved to: {save_path}")
        
        plt.show()


def analyze_sweep_ood(
    models: Dict,
    test_lengths: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    paper_ready: bool = False,
    num_test_trajectories: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> OODResults:
    """
    One-liner function for comprehensive OOD analysis.
    
    Args:
        models: Dictionary from load_experiment_sweep
        test_lengths: List of sequence lengths to test
        save_path: Optional path to save figure
        paper_ready: If True, create cleaner plots for publication
        num_test_trajectories: Number of test trajectories per length
        device: Device to run evaluation on
        **kwargs: Additional arguments passed to plot_results()
        
    Returns:
        OODResults object with all evaluation results
    """
    if test_lengths is None:
        test_lengths = [25, 30, 35, 40, 50, 60, 80, 100]
    
    evaluator = OODEvaluator(models, device=device, verbose=True)
    
    print("Evaluating OOD generalization for loaded models...")
    results = evaluator.evaluate(test_lengths, num_test_trajectories)
    
    print("\nPlotting results...")
    evaluator.plot_results(results, save_path=save_path, paper_ready=paper_ready, **kwargs)
    
    return results