"""Time-based train/val/test split.

Why time-based (not random): the model predicts success *before* publication
using only pre-publication signals. A random split would leak
same-period information across train/val/test (shared trending topics,
near-duplicate coverage of the same event) and overstate real-world
performance. Splitting by calendar month, with val/test strictly after
train, mirrors how the model would actually be deployed. Month boundaries
come from configs/base_config.yaml -- the most recent (partial) month is
deliberately left out of every split by not being covered by any boundary.

Usage:
    python -m src.data.split --config configs/base_config.yaml
"""
import argparse
import hashlib
import os

import pandas as pd
import yaml

from src.data.extract import validate_extract


def _cache_path(url: str, cache_dir: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{h}.jpg")


def split_by_month(df: pd.DataFrame, train_end_month: str, val_month: str, test_month: str):
    df = df.copy()
    df["display_time"] = pd.to_datetime(df["display_time"])
    df["publish_month"] = df["display_time"].dt.to_period("M").astype(str)

    train = df[df["publish_month"] <= train_end_month]
    val = df[df["publish_month"] == val_month]
    test = df[df["publish_month"] == test_month]
    return train, val, test


def _summarize(name: str, split_df: pd.DataFrame) -> None:
    if len(split_df) == 0:
        print(f"{name}: 0 rows -- check month boundaries in configs/base_config.yaml")
        return
    print(f"{name}: {len(split_df):,} rows, "
          f"{split_df['publish_month'].min()}..{split_df['publish_month'].max()}, "
          f"positive rate {split_df['target'].mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]

    df = validate_extract(data_cfg["raw_csv"])

    n_before = len(df)
    df = df.dropna(subset=["teaser_title"]).reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with null teaser_title.")

    n_before = len(df)
    has_image = df["pic_furl"].apply(lambda u: os.path.exists(_cache_path(u, data_cfg["images_dir"])))
    df = df[has_image].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with no cached image "
          f"(failed download -- see src/data/download_images.py failure breakdown).")

    split_cfg = data_cfg["split"]
    train, val, test = split_by_month(
        df, split_cfg["train_end_month"], split_cfg["val_month"], split_cfg["test_month"]
    )

    os.makedirs(data_cfg["processed_dir"], exist_ok=True)
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(data_cfg["processed_dir"], f"{name}.csv")
        split_df.to_csv(path, index=False)
        _summarize(name, split_df)
