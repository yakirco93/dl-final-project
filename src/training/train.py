"""Training entry point for the main SigLIP2 model.

Usage:
    python -m src.training.train --config configs/base_config.yaml

Responsibilities:
  - load_config + set_seed (src/utils.py) at the very start, for reproducibility
  - build ArticleDataset/DataLoader (src/data/dataset.py), with SigLIP2-specific
    batching from src/models/siglip2_model.py's make_collate_fn
  - class-weighted BCE loss per config["training"]["class_weight_positive"]
    (only "bce_with_class_weight" is implemented; "focal_loss" is not)
  - log training loss every config["logging"]["log_every_n_steps"] steps, and
    validation metrics every epoch, to experiments/<experiment_name>/
  - early stopping on validation PR-AUC (not accuracy -- see proposal's
    metric-choice rationale for the class-imbalance reasoning), saving the
    best checkpoint to experiments/<experiment_name>/best_model.pt
"""
import csv
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ArticleDataset
from src.evaluation.metrics import compute_metrics
from src.models.siglip2_model import SigLIP2Classifier, make_collate_fn
from src.utils import get_device, load_config, set_seed


def _move_to_device(batch, device):
    inputs = {
        "image_inputs": {k: v.to(device) for k, v in batch["image_inputs"].items()},
        "text_inputs": {k: v.to(device) for k, v in batch["text_inputs"].items()},
        "site_idx": batch["site_idx"].to(device),
    }
    return inputs, batch["labels"].to(device)


def _trainable_state_dict(model):
    """Only the params actually being trained (e.g. ~400K for a frozen-backbone
    Stage A run, not the full ~375M-param backbone) -- keeps checkpoints small."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable_names}


def evaluate(model, val_loader, device, k_values):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = _move_to_device(batch, device)
            probs = torch.sigmoid(model(inputs))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return compute_metrics(all_labels, all_probs, k_values=k_values)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()

    assert cfg["training"]["loss"] == "bce_with_class_weight", \
        f"Only 'bce_with_class_weight' is implemented, got {cfg['training']['loss']!r}"

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    k_values = tuple(cfg["evaluation"]["precision_at_k_values"])

    model = SigLIP2Classifier(cfg).to(device)
    collate_fn = make_collate_fn(model.processor, model.site_to_idx)

    train_ds = ArticleDataset(f"{data_cfg['processed_dir']}/train.csv", data_cfg["images_dir"])
    val_ds = ArticleDataset(f"{data_cfg['processed_dir']}/val.csv", data_cfg["images_dir"])

    # Loading + decoding images from disk one-by-one on the main process is the
    # bottleneck on a cold OS file cache (frozen-backbone forward/backward is
    # cheap on MPS/CUDA) -- worker processes parallelize that. But train+val
    # each get their own pool, so total worker processes are double this
    # number; on a loaded 24GB machine, num_workers=6 (12 total) caused heavy
    # swapping that looked like a hang but was actually swap thrashing.
    # num_workers=0 turned out fine once the image cache was warm (after
    # Stage A/B already read the full dataset), so that's the default now --
    # bump it back up via config if starting from a cold cache.
    num_workers = min(train_cfg.get("num_workers", 0), os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0)

    pos_weight = torch.tensor(train_cfg["class_weight_positive"], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Discriminative fine-tuning: any unfrozen backbone params (Stage B) or
    # LoRA params (Stage C) train at a lower LR than the head -- they're
    # pretrained and shouldn't move as fast as the randomly-initialized head.
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    param_groups = [{"params": head_params, "lr": train_cfg["learning_rate"]}]
    if backbone_params:
        param_groups.append({"params": backbone_params,
                              "lr": train_cfg["learning_rate"] * train_cfg["backbone_lr_multiplier"]})
        print(f"Discriminative LR: head={train_cfg['learning_rate']:.2e}, "
              f"backbone={train_cfg['learning_rate'] * train_cfg['backbone_lr_multiplier']:.2e} "
              f"({len(backbone_params)} backbone param tensors unfrozen)")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])

    num_epochs = train_cfg["num_epochs"]
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    run_dir = os.path.join(cfg["logging"]["output_dir"], cfg["experiment_name"])
    os.makedirs(run_dir, exist_ok=True)

    # Persist the training/val history to disk as it happens (not just stdout)
    # -- stdout is lost if the terminal closes or scrolls; these CSVs survive
    # even a killed process, and are what notebooks/03_results.ipynb reads to
    # render loss curves and the metrics table for the presentation.
    step_log_path = os.path.join(run_dir, "train_log.csv")
    epoch_log_path = os.path.join(run_dir, "val_log.csv")
    step_log_file = open(step_log_path, "w", newline="")
    step_log_writer = csv.writer(step_log_file)
    step_log_writer.writerow(["step", "epoch", "loss"])
    epoch_log_file = open(epoch_log_path, "w", newline="")
    epoch_log_writer = csv.writer(epoch_log_file)
    epoch_log_writer.writerow(["epoch"] + list(compute_metrics([0, 1], [0.1, 0.9], k_values=k_values).keys()))

    best_pr_auc = -1.0
    epochs_without_improvement = 0
    step = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = _move_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % cfg["logging"]["log_every_n_steps"] == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
                step_log_writer.writerow([step, epoch, loss.item()])
                step_log_file.flush()

        metrics = evaluate(model, val_loader, device, k_values)
        print(f"epoch {epoch} val: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
        epoch_log_writer.writerow([epoch] + list(metrics.values()))
        epoch_log_file.flush()

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            epochs_without_improvement = 0
            # Reload with: model.load_state_dict(torch.load(path), strict=False)
            # -- strict=False because this excludes the untouched frozen backbone.
            torch.save(_trainable_state_dict(model), os.path.join(run_dir, "best_model.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val PR-AUC improvement for {train_cfg['early_stopping_patience']} epochs)")
                break

    step_log_file.close()
    epoch_log_file.close()
    print(f"Best val PR-AUC: {best_pr_auc:.4f}. Checkpoint: {run_dir}/best_model.pt")
    print(f"Logs: {step_log_path}, {epoch_log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
