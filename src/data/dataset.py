"""PyTorch Dataset for (title, tags, image, site) -> label.

TODO (Step 1, after extract.py + download_images.py are wired up):
  - Load data/processed/{train,val,test}.csv (produced by a time-based split
    of the extracted articles — see README "Workflow").
  - __getitem__ should return the raw text fields + a loaded PIL image from
    the local cache (src/data/download_images.py's cache_dir), letting the
    model-specific processor (SigLIP2Processor, or separate tokenizer +
    torchvision transform for the fallback model) handle featurization.
  - Keep this Dataset model-agnostic; put SigLIP2-specific preprocessing in
    src/models/siglip2_model.py's collate_fn instead, so baselines can reuse
    the same Dataset class.
"""
from torch.utils.data import Dataset


class ArticleDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str):
        raise NotImplementedError("Step 1: implement after the data pipeline is in place.")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
