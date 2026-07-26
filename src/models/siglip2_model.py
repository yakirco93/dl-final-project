"""Main model: SigLIP2 fine-tuned for binary top-10% classification.

Implemented in Steps 4-5, staged per configs/base_config.yaml -> model.siglip2:

  Stage A (freeze_backbone: true):
      Load AutoModel.from_pretrained(checkpoint), freeze all backbone params,
      add a small trainable classification head on top of the pooled
      text+image embeddings (concat) plus a learned site embedding.

  Stage B (unfreeze_last_n_layers > 0):
      Unfreeze the top N transformer layers of both the vision and text
      towers (`backbone.vision_model.encoder.layers`,
      `backbone.text_model.encoder.layers`).

  Stage C (use_lora: true):
      Wrap the backbone with peft.LoraConfig instead of full unfreezing —
      cheaper and lower overfitting risk with a ~90K-article train set.

Stage A learned something the baselines already hinted at (see
src/models/baselines.py's title_image_frozen result: naive concatenation of
frozen embeddings did NOT beat text_only_bert alone). This class still uses
a frozen backbone in Stage A, but with a proper trainable (small MLP, not
sklearn LogisticRegression) head trained end-to-end with the same
class-weighted BCE loss as the rest of the project -- Stage B/C exist
specifically because Stage A's frozen features may again prove insufficient,
same lesson as the baseline.

This class intentionally does NOT implement the fallback hybrid model
(AlephBERT/HeBERT + EfficientNet/ResNet + fusion head) — that lives in
src/models/hybrid_fallback.py and is only built if this model underperforms
on Hebrew text, per the contingency plan in the proposal.
"""
import torch
from PIL import Image
from torch import nn


class SigLIP2Classifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.siglip_cfg = config["model"]["siglip2"]
        checkpoint = self.siglip_cfg["checkpoint"]

        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.backbone = AutoModel.from_pretrained(checkpoint)
        self._apply_freeze_strategy()

        self.site_to_idx = {s: i for i, s in enumerate(config["data"]["sites"])}
        site_embed_dim = 8
        self.site_embedding = nn.Embedding(len(self.site_to_idx), site_embed_dim)

        image_dim, text_dim = self._infer_embed_dims()
        head_hidden_dim = 256
        self.head = nn.Sequential(
            nn.Linear(image_dim + text_dim + site_embed_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1),
        )

    def _apply_freeze_strategy(self):
        cfg = self.siglip_cfg
        if cfg.get("freeze_backbone", True):
            for p in self.backbone.parameters():
                p.requires_grad = False

        n = cfg.get("unfreeze_last_n_layers", 0)
        if n > 0:
            for tower in (self.backbone.vision_model, self.backbone.text_model):
                for layer in tower.encoder.layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        if cfg.get("use_lora", False):
            from peft import LoraConfig, get_peft_model

            lora_cfg = cfg["lora"]
            peft_config = LoraConfig(
                r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                target_modules=["q_proj", "v_proj"], bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

    def _infer_embed_dims(self):
        """Determine get_image_features/get_text_features output dims empirically
        (varies by checkpoint) rather than guessing which config attribute holds it."""
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_inputs = self.processor(images=[dummy_image], return_tensors="pt")
        txt_inputs = self.processor(text=["x"], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            image_dim = self._pooled(self.backbone.get_image_features(**img_inputs)).shape[-1]
            text_dim = self._pooled(self.backbone.get_text_features(**txt_inputs)).shape[-1]
        return image_dim, text_dim

    @staticmethod
    def _pooled(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def forward(self, batch):
        """batch: {"image_inputs": {...}, "text_inputs": {...}, "site_idx": LongTensor}
        as produced by make_collate_fn. Returns raw logits (apply sigmoid for probabilities)."""
        img_feat = self._pooled(self.backbone.get_image_features(**batch["image_inputs"]))
        txt_feat = self._pooled(self.backbone.get_text_features(**batch["text_inputs"]))
        site_feat = self.site_embedding(batch["site_idx"])
        combined = torch.cat([img_feat, txt_feat, site_feat], dim=-1)
        return self.head(combined).squeeze(-1)


class _Siglip2CollateFn:
    """SigLIP2-specific batching: raw ArticleDataset items -> model-ready tensors.
    Kept here (not in ArticleDataset) so the Dataset stays model-agnostic.

    A module-level class rather than a closure -- DataLoader(num_workers>0) on
    macOS uses the 'spawn' start method, which pickles the collate_fn to send
    it to worker processes; a nested closure function can't be pickled."""

    def __init__(self, processor, site_to_idx: dict):
        self.processor = processor
        self.site_to_idx = site_to_idx

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        titles = [item["title"] for item in batch]
        sites = [self.site_to_idx[item["site"]] for item in batch]
        labels = [float(item["label"]) for item in batch]

        image_inputs = dict(self.processor(images=images, return_tensors="pt"))
        text_inputs = dict(self.processor(text=titles, return_tensors="pt", padding=True, truncation=True))

        return {
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
            "site_idx": torch.tensor(sites, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def make_collate_fn(processor, site_to_idx: dict):
    return _Siglip2CollateFn(processor, site_to_idx)
