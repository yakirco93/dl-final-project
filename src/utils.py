"""Shared utilities: config loading and reproducibility.

Every script in this project should start with:

    from src.utils import load_config, set_seed
    cfg = load_config("configs/base_config.yaml")
    set_seed(cfg["seed"])

Never hardcode hyperparameters directly in training/model code — add them to
the relevant config YAML instead, so every run is fully reproducible from its
config file alone.
"""
import random
import yaml

import numpy as np


def load_config(path: str) -> dict:
    """Load a YAML experiment config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds across python, numpy, and torch (if available) for
    reproducibility. Call this once at the start of every script."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
