#!/usr/bin/env python3
"""
Multi-seed experiment runner for RNN training.

This script runs multiple training runs with different random seeds to assess
result uncertainty and provides organized directory structure for analysis.
"""

import os
import sys
import yaml
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Any


def create_experiment_directory(experiment_name: str, base_log_dir: str) -> str:
    """Create experiment directory structure."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_log_dir, "experiments", f"expt_{timestamp}")
    
    # Just create the main directory - seed subdirectories will be created by main.py
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def save_experiment_config(config: Dict[Any, Any], experiment_dir: str, 
                          seeds: List[int], experiment_name: str) -> None:
    """Save the base config and experiment metadata."""
    
    # Save base config
    base_config_path = os.path.join(experiment_dir, "configs", "base_config.yaml")
    with open(base_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Save experiment metadata
    experiment_meta = {
        "experiment_name": experiment_name,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "model_type": config.get("model_type", "vanilla"),
        "created_at": datetime.datetime.now().isoformat(),
        "base_config_file": "base_config.yaml"
    }
    
    meta_path = os.path.join(experiment_dir, "experiment_metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(experiment_meta, f, default_flow_style=False, indent=2)
    
    print(f"Experiment metadata saved to: {meta_path}")


def run_single_seed(config: Dict[Any, Any], seed: int, experiment_name: str, 
                   seed_idx: int, total_seeds: int) -> Dict[str, Any]:
    """Run training for a single seed."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING SEED {seed_idx + 1}/{total_seeds}: seed={seed}")
    print(f"{'='*60}")
    
    # Create config for this seed
    seed_config = config.copy()
    seed_config["seed"] = seed
    seed_config["experiment_name"] = experiment_name  # Pass experiment name, not full path
    
    # The seed-specific directory will be: experiment_dir/seed_{seed}/
    seed_dir = os.path.join(experiment_name, f"seed_{seed}")
    
    # Import and run main (avoid subprocess for better error handling)
    try:
        from main import main_single_seed
        result = main_single_seed(seed_config)
        
        # Save run summary in the seed directory
        run_summary = {
            "seed": seed,
            "status": "completed",
            "seed_dir": seed_dir,
            "final_val_loss": result.get("final_val_loss", None),
            "completed_at": datetime.datetime.now().isoformat()
        }
        
        summary_path = os.path.join(seed_dir, "run_summary.yaml")
        with open(summary_path, "w") as f:
            yaml.dump(run_summary, f, default_flow_style=False, indent=2)
            
        print(f"✓ Seed {seed} completed successfully")
        return run_summary
        
    except Exception as e:
        print(f"✗ Seed {seed} failed with error: {str(e)}")
        
        # Save error summary
        error_summary = {
            "seed": seed,
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.datetime.now().isoformat()
        }
        
        # Create seed directory if it doesn't exist and save error
        os.makedirs(seed_dir, exist_ok=True)
        error_path = os.path.join(seed_dir, "error_summary.yaml")
        with open(error_path, "w") as f:
            yaml.dump(error_summary, f, default_flow_style=False, indent=2)
            
        return error_summary


def generate_experiment_summary(experiment_dir: str, run_results: List[Dict[str, Any]]) -> None:
    """Generate overall experiment summary."""
    
    successful_runs = [r for r in run_results if r["status"] == "completed"]
    failed_runs = [r for r in run_results if r["status"] == "failed"]
    
    # Calculate statistics for successful runs
    val_losses = [r["final_val_loss"] for r in successful_runs if r.get("final_val_loss") is not None]
    
    summary = {
        "experiment_completed_at": datetime.datetime.now().isoformat(),
        "total_seeds": len(run_results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "success_rate": len(successful_runs) / len(run_results) if run_results else 0,
    }
    
    if val_losses:
        import numpy as np
        summary["validation_loss_stats"] = {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses)),
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses)),
            "median": float(np.median(val_losses))
        }
    
    # Add individual run results
    summary["run_details"] = run_results
    
    # Save summary
    summary_path = os.path.join(experiment_dir, "experiment_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Successful runs: {len(successful_runs)}/{len(run_results)}")
    if val_losses:
        print(f"Validation loss: {summary['validation_loss_stats']['mean']:.4f} ± {summary['validation_loss_stats']['std']:.4f}")
    print(f"Results saved to: {experiment_dir}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed RNN training experiment")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to config file")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name for this experiment")
    parser.add_argument("--n_seeds", type=int, default=5,
                       help="Number of random seeds to run")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                       help="Specific seeds to use (if not provided, will use range(n_seeds))")
    parser.add_argument("--base_seed", type=int, default=0,
                       help="Starting seed when using n_seeds (seeds will be base_seed + range(n_seeds))")
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Determine seeds to use
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    
    print(f"Running experiment '{args.experiment_name}' with seeds: {seeds}")
    
    # Create experiment directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    experiment_dir = create_experiment_directory(args.experiment_name, log_dir)
    
    # Save experiment config and metadata
    save_experiment_config(config, experiment_dir, seeds, args.experiment_name)
    
    # Run all seeds
    run_results = []
    for seed_idx, seed in enumerate(seeds):
        result = run_single_seed(config, seed, args.experiment_name, seed_idx, len(seeds))
        run_results.append(result)
    
    # Generate experiment summary
    generate_experiment_summary(experiment_dir, run_results)


if __name__ == "__main__":
    main()
