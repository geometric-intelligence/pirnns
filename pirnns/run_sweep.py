#!/usr/bin/env python3
"""
Parameter sweep experiment runner.

Takes an experiment configuration file that defines parameter sweeps
over a base configuration and runs all combinations with multiple seeds.
"""

import os
import yaml
import argparse
import datetime
import copy
from typing import List, Dict, Any, Tuple
from main import main_single_seed
import numpy as np
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def deep_merge_dict(base: Dict, override: Dict) -> Dict:
    """Deep merge override dictionary into base dictionary."""
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_experiment_config(experiment_file: str) -> Dict:
    """
    Load experiment configuration with base config and parameter sweeps.

    Args:
        experiment_file: Path to experiment YAML file

    Returns:
        Dictionary containing experiment configuration
    """
    if not os.path.exists(experiment_file):
        raise FileNotFoundError(f"Experiment file not found: {experiment_file}")

    with open(experiment_file, "r") as f:
        experiment_config = yaml.safe_load(f)

    # Load base configuration
    base_config_path = experiment_config["base_config"]
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    experiment_config["_base_config"] = base_config
    return experiment_config


def generate_experiment_configs(experiment_config: Dict) -> List[Tuple[str, Dict]]:
    """
    Generate all individual experiment configurations from sweep.

    Args:
        experiment_config: Loaded experiment configuration

    Returns:
        List of (experiment_name, config) tuples
    """
    base_config = experiment_config["_base_config"]
    experiments = experiment_config["experiments"]

    experiment_configs = []

    for exp in experiments:
        exp_name = exp["name"]
        overrides = exp.get("overrides", {})

        # Merge base config with overrides
        merged_config = deep_merge_dict(base_config, overrides)

        experiment_configs.append((exp_name, merged_config))

    return experiment_configs


@rank_zero_only
def create_sweep_directory(base_log_dir: str, sweep_name: str) -> str:
    """Create sweep directory structure."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(base_log_dir, "sweeps", f"{sweep_name}_{timestamp}")

    os.makedirs(sweep_dir, exist_ok=True)
    return sweep_dir


@rank_zero_only
def save_sweep_metadata(
    sweep_dir: str, experiment_config: Dict, experiment_configs: List[Tuple[str, Dict]]
) -> None:
    """Save sweep metadata and configurations."""

    # Save sweep metadata
    sweep_metadata = {
        "sweep_name": os.path.basename(sweep_dir),
        "base_config_file": experiment_config["base_config"],
        "n_seeds": experiment_config["n_seeds"],
        "n_experiments": len(experiment_configs),
        "total_runs": len(experiment_configs) * experiment_config["n_seeds"],
        "created_at": datetime.datetime.now().isoformat(),
        "experiments": [name for name, _ in experiment_configs],
    }

    metadata_path = os.path.join(sweep_dir, "sweep_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(sweep_metadata, f, default_flow_style=False, indent=2)

    # Save individual experiment configs
    configs_dir = os.path.join(sweep_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    for exp_name, config in experiment_configs:
        config_path = os.path.join(configs_dir, f"{exp_name}_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sweep metadata saved to: {metadata_path}")


def run_single_experiment(
    exp_name: str, config: Dict, seeds: List[int], sweep_dir: str
) -> List[Dict[str, Any]]:
    """
    Run a single experiment configuration with multiple seeds.

    Args:
        exp_name: Name of the experiment
        config: Configuration for this experiment
        seeds: List of seeds to run
        sweep_dir: Base sweep directory

    Returns:
        List of run results for each seed
    """
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}")

    # Create experiment directory
    exp_dir = os.path.join(sweep_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save experiment metadata
    exp_metadata = {
        "experiment_name": exp_name,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "created_at": datetime.datetime.now().isoformat(),
    }

    metadata_path = os.path.join(exp_dir, "experiment_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(exp_metadata, f, default_flow_style=False, indent=2)

    # Run each seed
    run_results = []
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'-'*60}")
        print(f"EXPERIMENT {exp_name} - SEED {seed_idx + 1}/{len(seeds)}: seed={seed}")
        print(f"{'-'*60}")

        # Create seed config
        seed_config = config.copy()
        seed_config["seed"] = seed
        seed_config["experiment_dir"] = exp_name  # Use experiment name as directory

        # Seed directory will be: sweep_dir/exp_name/seed_{seed}/
        seed_dir = os.path.join(exp_dir, f"seed_{seed}")

        try:
            result = main_single_seed(seed_config)

            # Save run summary
            run_summary = {
                "experiment_name": exp_name,
                "seed": seed,
                "status": "completed",
                "seed_dir": seed_dir,
                "final_val_loss": result.get("final_val_loss", None),
                "completed_at": datetime.datetime.now().isoformat(),
            }

            summary_path = os.path.join(seed_dir, "run_summary.yaml")
            with open(summary_path, "w") as f:
                yaml.dump(run_summary, f, default_flow_style=False, indent=2)

            print(f"✓ {exp_name} seed {seed} completed successfully")
            run_results.append(run_summary)

        except Exception as e:
            print(f"✗ {exp_name} seed {seed} failed with error: {str(e)}")

            # Save error summary
            error_summary = {
                "experiment_name": exp_name,
                "seed": seed,
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.datetime.now().isoformat(),
            }

            os.makedirs(seed_dir, exist_ok=True)
            error_path = os.path.join(seed_dir, "error_summary.yaml")
            with open(error_path, "w") as f:
                yaml.dump(error_summary, f, default_flow_style=False, indent=2)

            run_results.append(error_summary)

    # Generate experiment summary
    successful_runs = [r for r in run_results if r["status"] == "completed"]
    failed_runs = [r for r in run_results if r["status"] == "failed"]

    val_losses = [
        r["final_val_loss"]
        for r in successful_runs
        if r.get("final_val_loss") is not None
    ]

    exp_summary = {
        "experiment_name": exp_name,
        "experiment_completed_at": datetime.datetime.now().isoformat(),
        "total_seeds": len(run_results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "success_rate": len(successful_runs) / len(run_results) if run_results else 0,
    }

    if val_losses:
        exp_summary["validation_loss_stats"] = {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses)),
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses)),
            "median": float(np.median(val_losses)),
        }

    exp_summary["run_details"] = run_results

    # Save experiment summary
    summary_path = os.path.join(exp_dir, "experiment_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(exp_summary, f, default_flow_style=False, indent=2)

    print(f"\nExperiment {exp_name} complete!")
    print(f"Successful runs: {len(successful_runs)}/{len(run_results)}")
    if val_losses:
        print(
            f"Validation loss: {exp_summary['validation_loss_stats']['mean']:.4f} ± {exp_summary['validation_loss_stats']['std']:.4f}"
        )

    return run_results


@rank_zero_only
def generate_sweep_summary(
    sweep_dir: str, all_results: Dict[str, List[Dict[str, Any]]]
) -> None:
    """Generate overall sweep summary."""

    # Aggregate statistics across all experiments
    total_runs = sum(len(results) for results in all_results.values())
    total_successful = sum(
        len([r for r in results if r["status"] == "completed"])
        for results in all_results.values()
    )
    total_failed = total_runs - total_successful

    # Per-experiment statistics
    experiment_stats = {}
    for exp_name, results in all_results.items():
        successful = [r for r in results if r["status"] == "completed"]
        val_losses = [
            r["final_val_loss"]
            for r in successful
            if r.get("final_val_loss") is not None
        ]

        stats = {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
        }

        if val_losses:
            stats["validation_loss_stats"] = {
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "median": float(np.median(val_losses)),
            }

        experiment_stats[exp_name] = stats

    # Overall sweep summary
    sweep_summary = {
        "sweep_completed_at": datetime.datetime.now().isoformat(),
        "total_experiments": len(all_results),
        "total_runs": total_runs,
        "total_successful_runs": total_successful,
        "total_failed_runs": total_failed,
        "overall_success_rate": total_successful / total_runs if total_runs else 0,
        "experiment_statistics": experiment_stats,
    }

    # Save sweep summary
    summary_path = os.path.join(sweep_dir, "sweep_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, indent=2)

    print(f"\n{'='*80}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {total_successful}/{total_runs}")
    print(f"Overall success rate: {sweep_summary['overall_success_rate']:.2%}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_path}")

    # Print per-experiment summary
    print("\nPer-experiment results:")
    for exp_name, stats in experiment_stats.items():
        print(
            f"  {exp_name}: {stats['successful_runs']}/{stats['total_runs']} successful",
            end="",
        )
        if "validation_loss_stats" in stats:
            print(
                f" (val_loss: {stats['validation_loss_stats']['mean']:.4f} ± {stats['validation_loss_stats']['std']:.4f})"
            )
        else:
            print()


def run_parameter_sweep(experiment_file: str):
    """
    Run full parameter sweep experiment.

    Args:
        experiment_file: Path to experiment configuration file
    """
    print(f"Loading parameter sweep configuration: {experiment_file}")

    # Load experiment configuration
    experiment_config = load_experiment_config(experiment_file)
    n_seeds = experiment_config["n_seeds"]

    # Generate individual experiment configurations
    experiment_configs = generate_experiment_configs(experiment_config)

    print("Parameter sweep configuration:")
    print(f"  Base config: {experiment_config['base_config']}")
    print(f"  Number of experiments: {len(experiment_configs)}")
    print(f"  Seeds per experiment: {n_seeds}")
    print(f"  Total runs: {len(experiment_configs) * n_seeds}")
    print(f"  Experiments: {[name for name, _ in experiment_configs]}")

    # Create sweep directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    sweep_name = os.path.splitext(os.path.basename(experiment_file))[0]

    # Generate consistent sweep directory path across all ranks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(log_dir, "sweeps", f"{sweep_name}_{timestamp}")

    # Create directory only on rank 0
    create_sweep_directory_only(sweep_dir)
    save_sweep_metadata(sweep_dir, experiment_config, experiment_configs)

    # Generate seeds
    seeds = list(range(n_seeds))

    # Run all experiments
    all_results = {}
    for exp_name, config in experiment_configs:
        results = run_single_experiment(exp_name, config, seeds, sweep_dir)
        all_results[exp_name] = results

    # Generate overall summary
    generate_sweep_summary(sweep_dir, all_results)


@rank_zero_only
def create_sweep_directory_only(sweep_dir: str) -> None:
    """Create sweep directory structure (rank 0 only)."""
    os.makedirs(sweep_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment configuration file",
    )

    args = parser.parse_args()

    run_parameter_sweep(args.experiment)


if __name__ == "__main__":
    main()
