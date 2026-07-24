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

try:
    # Use the OS trust store for HTTPS instead of certifi's bundled CAs.
    # Needed on networks with an SSL-inspecting proxy (e.g. a corporate
    # Fortinet firewall) whose root CA is trusted by the OS but not by
    # certifi -- without this, huggingface_hub downloads fail with
    # CERTIFICATE_VERIFY_FAILED on such networks. No-op / harmless elsewhere
    # (e.g. Colab), so safe to always enable.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


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


def get_device() -> str:
    """Pick the best available torch device: CUDA (Colab) > MPS (Apple
    Silicon) > CPU. Lets the same training code run locally on an M-series
    Mac and on a Colab GPU without changes."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
