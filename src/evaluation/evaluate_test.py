"""Final, one-time evaluation on the held-out test set (June 2026 -- never
used for anything else in this project, only train/val throughout).

Usage:
    python -m src.evaluation.evaluate_test --config configs/exp07_model2_hybrid_tuned.yaml

Loads the checkpoint from experiments/<experiment_name>/best_model.pt (the
same model type/config that produced it) and reports the same metrics used
throughout training, on test.csv instead of val.csv.
"""
import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ArticleDataset
from src.training.train import _build_model_and_collate_fns, evaluate
from src.utils import get_device, load_config, set_seed


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    k_values = tuple(cfg["evaluation"]["precision_at_k_values"])

    model, _, eval_collate_fn = _build_model_and_collate_fns(cfg, device)

    run_dir = os.path.join(cfg["logging"]["output_dir"], cfg["experiment_name"])
    checkpoint_path = os.path.join(run_dir, "best_model.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # strict=False: checkpoint only has requires_grad=True params (see
    # train.py's _trainable_state_dict), not the frozen backbone.
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    test_ds = ArticleDataset(f"{data_cfg['processed_dir']}/test.csv", data_cfg["images_dir"])
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                              collate_fn=eval_collate_fn)

    metrics = evaluate(model, test_loader, device, k_values)
    print(f"\nTEST SET results ({len(test_ds)} rows, held out, first use) -- {checkpoint_path}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
