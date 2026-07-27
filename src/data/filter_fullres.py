"""Filter processed train/val/test splits down to rows whose cached thumbnail
is the full-resolution original -- not one of the two fixed-size CDN
fallback tiers (82x62 or 232x175px) that src/data/download_images.py falls
back to when the original photo is unavailable (~39% of the full dataset;
see project_explanation_HE.md section 11). Used for a model variant that
tests whether image signal looks stronger once resolution is guaranteed
consistent, instead of a mix of full-res/thumbnail/tiny.

Cached files carry no metadata about which tier they came from, but the two
fallback tiers are always exactly 82x62 or 232x175px (fixed CDN output
sizes), while the full-res original -- resized down to at most 512px on the
long side, per download_images.py -- essentially never lands on exactly one
of those two sizes. So exact pixel-size match is a reliable, cheap filter.

Usage:
    python -m src.data.filter_fullres --config configs/base_config.yaml \
        --out-dir data/processed/fullres
"""
import argparse
import os

import pandas as pd
from PIL import Image

from src.data.download_images import _cache_path

_FALLBACK_SIZES = {(82, 62), (232, 175)}


def is_full_res(pic_furl: str, images_dir: str) -> bool:
    path = _cache_path(pic_furl, images_dir)
    if not os.path.exists(path):
        return False
    with Image.open(path) as img:
        return img.size not in _FALLBACK_SIZES


if __name__ == "__main__":
    from src.utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    os.makedirs(args.out_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        df = pd.read_csv(f"{data_cfg['processed_dir']}/{split_name}.csv")
        mask = df["pic_furl"].apply(lambda u: is_full_res(u, data_cfg["images_dir"]))
        filtered = df[mask].reset_index(drop=True)
        filtered.to_csv(f"{args.out_dir}/{split_name}.csv", index=False)
        print(f"{split_name}: kept {len(filtered):,}/{len(df):,} ({mask.mean():.1%}) full-res rows")
