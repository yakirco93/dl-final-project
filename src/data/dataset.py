"""PyTorch Dataset for (title, tags, image, site) -> label.

Returns raw fields only (title string, tags string, a loaded PIL image, site
string, int label) -- no tokenization or image transforms here. That stays
model-specific: SigLIP2Processor / a separate tokenizer + torchvision
transform belongs in the caller's collate_fn, so this one Dataset class is
reusable across baselines and the main model.
"""
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from src.data.download_images import _cache_path


class ArticleDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = _cache_path(row["pic_furl"], self.images_dir)
        image = Image.open(image_path).convert("RGB")
        return {
            "title": row["teaser_title"],
            "tags": row["tags"] if pd.notna(row["tags"]) else "",
            "image": image,
            "site": row["site"],
            "label": int(row["target"]),
        }
