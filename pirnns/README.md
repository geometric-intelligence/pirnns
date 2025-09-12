# Training Models

This project provides three ways to train models:

### 1. Single Training Run (`main.py`)

For testing a configuration or single model training, use:

```bash
# Train a vanilla RNN
python main.py --config configs/vanilla_config.yaml

# Train a multitimescale RNN  
python main.py --config configs/mts_config.yaml
```

**Output:** `logs/single_runs/{model_type}_{timestamp}/`

### 2. Multi-Seed Experiments (`run_multiseed.py`)

For assessing variability across random seeds with one configuration:

```bash
python run_multiseed.py --config configs/mts_config.yaml --n_seeds 5
```

**Output:** `logs/experiments/expt_{timestamp}/seed_{0,1,2...}/`

### 3. Parameter Sweeps (`run_sweep.py`)

For comparing multiple configurations systematically:

```bash
python run_sweep.py --experiment experiments/timescales_sweep.yaml
```

**Use when:**
- Hyperparameter tuning
- Comparing different model architectures
- Systematic ablation studies
- You want to compare multiple configurations, each with multiple seeds

**Output:** `logs/sweeps/{sweep_name}_{timestamp}/{config_name}/seed_{0,1,2...}/`

#### Creating Parameter Sweep Experiments

Create an experiment file (e.g., `experiments/timescales_sweep.yaml`):

```yaml
# Base configuration to inherit from
base_config: "configs/mts_config.yaml"

# Number of seeds per configuration
n_seeds: 3

# Parameter sweep configurations
experiments:
  - name: "discrete_single"
    overrides:
      timescales_config:
        type: "discrete"
        values: [0.1443]
  
  - name: "discrete_dual" 
    overrides:
      timescales_config:
        type: "discrete"
        values: [0.1, 0.5]
      max_epochs: 25  # Can override multiple parameters
```

This will run 2 configurations × 3 seeds = 6 total training runs.