# Train a path-integrating RNN

To train a path-integrating RNN, take the following steps:

1. Set the appropriate parameters in your `config.yaml` file in the `configs` directory.

2. To run the vanilla RNN, use the following command:

```bash
python main.py --config configs/vanilla_config.yaml
```

or to run the coupled RNN, use the following command:

```bash
python main.py --config configs/coupled_config.yaml
```

or to run the multitimescale RNN, use the following command:

```bash
python main.py --config configs/mts_config.yaml
```



3. The script will create a run ID (based on the time of the run) and save the trained model in `logs/checkpoints/<run_id>/`.


## Analyze the trained model

The notebook `analyze.ipynb` shows how to load the trained model, and provides some visualizations.








