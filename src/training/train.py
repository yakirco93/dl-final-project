"""Training entry point — implemented once a model (baseline or SigLIP2) exists.

Usage (once implemented):
    python -m src.training.train --config configs/exp01_frozen_head.yaml

Responsibilities:
  - load_config + set_seed (src/utils.py) at the very start, for reproducibility
  - build the Dataset/DataLoader (src/data/dataset.py)
  - build the model per config["model"]["type"]
  - BCE-with-class-weight (or focal loss) per config["training"]["loss"]
  - log metrics every config["logging"]["log_every_n_steps"] to
    experiments/<experiment_name>/
  - early stopping on validation PR-AUC (not accuracy — see proposal's
    metric-choice rationale for the class-imbalance reasoning)
"""
from src.utils import load_config, set_seed


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    raise NotImplementedError("Implement once a baseline or SigLIP2 model class exists.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
