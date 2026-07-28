"""Model 2 of the "improvement phase" (see project_explanation_HE.md section
11): the contingency plan from the original proposal, built now because
three different SigLIP2 strategies (frozen/unfreeze/LoRA) and a
cross-attention fusion upgrade all failed to beat text_only_bert.

Different architecture entirely from SigLIP2 -- two backbones that were
NEVER jointly pretrained (unlike SigLIP2's contrastively-aligned text/image
towers), fused only here:

  - Text: AlephBERT (Hebrew-specific, not SigLIP2's multilingual-but-
    English-heavy encoder) -- CLS token pooling (the checkpoint's pooler
    layer is untrained/randomly initialized, so CLS from last_hidden_state
    is used directly rather than routing through it).
  - Image: ResNet50 (ImageNet-pretrained) -- global-average-pooled features
    (2048-dim), classification head removed.
  - Site: learned embedding, same as SigLIP2Classifier.
  - Fusion: concatenation + small MLP (config.model.hybrid_fallback.
    fusion_hidden_dim). Not cross-attention -- the controlled ablation in
    exp05 showed cross-attention didn't help when isolated from the
    overfitting confound, and here the backbones aren't even in a shared
    embedding space to begin with, so there's less reason to expect
    attention-based fusion to pay off immediately; concat is the simpler
    first attempt.

Both encoders are frozen by default; unfreeze_text_layers /
unfreeze_image_blocks (config) unfreeze the top N layers/blocks of each --
same discriminative-LR pattern as SigLIP2Classifier (config.training.
backbone_lr_multiplier applies to these params too). Kept conservative by
default (Stage B's lesson: too many trainable params on ~87K rows overfits
fast) -- tune up if early stopping isn't triggering.
"""
import torch
from torch import nn
from torchvision import transforms as T


class HybridClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        import torchvision.models as tv_models

        self.hybrid_cfg = config["model"]["hybrid_fallback"]

        text_checkpoint = self.hybrid_cfg["text_encoder"]
        self.tokenizer = AutoTokenizer.from_pretrained(text_checkpoint)
        self.text_encoder = AutoModel.from_pretrained(text_checkpoint)

        image_encoder_name = self.hybrid_cfg.get("image_encoder", "resnet50")
        self.image_encoder = getattr(tv_models, image_encoder_name)(weights="DEFAULT")
        image_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()  # drop the ImageNet classification head

        self._apply_freeze_strategy()

        self.site_to_idx = {s: i for i, s in enumerate(config["data"]["sites"])}
        site_embed_dim = 8
        self.site_embedding = nn.Embedding(len(self.site_to_idx), site_embed_dim)

        text_dim = self.text_encoder.config.hidden_size
        fusion_hidden_dim = self.hybrid_cfg.get("fusion_hidden_dim", 256)
        self.head = nn.Sequential(
            nn.Linear(image_dim + text_dim + site_embed_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def _apply_freeze_strategy(self):
        cfg = self.hybrid_cfg
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        n_text = cfg.get("unfreeze_text_layers", 1)
        if n_text > 0:
            for layer in self.text_encoder.encoder.layer[-n_text:]:
                for p in layer.parameters():
                    p.requires_grad = True

        n_image_blocks = cfg.get("unfreeze_image_blocks", 1)
        if n_image_blocks > 0:
            blocks = [self.image_encoder.layer1, self.image_encoder.layer2,
                      self.image_encoder.layer3, self.image_encoder.layer4]
            for block in blocks[-n_image_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, batch):
        """batch: {"input_ids": ..., "attention_mask": ..., "pixel_values": ...,
        "site_idx": LongTensor} as produced by make_hybrid_collate_fn.
        Returns raw logits (apply sigmoid for probabilities)."""
        text_out = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_feat = text_out.last_hidden_state[:, 0]  # CLS token (pooler is untrained -- see module docstring)
        image_feat = self.image_encoder(batch["pixel_values"])
        site_feat = self.site_embedding(batch["site_idx"])
        combined = torch.cat([image_feat, text_feat, site_feat], dim=-1)
        return self.head(combined).squeeze(-1)


_IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Deterministic -- used for val/test so metrics aren't affected by random
# augmentation draws.
_EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    _IMAGENET_NORMALIZE,
])

# Random crop/flip/color-jitter -- only meaningful once the vision backbone
# (or part of it, e.g. layer4) is actually being fine-tuned; augmenting
# inputs to a fully-frozen ResNet just adds noise to features nothing
# downstream of the backbone can adapt to compensate for. Intended to fight
# overfitting more directly than freezing capacity away entirely (see
# project_explanation_HE.md section 11).
_TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    _IMAGENET_NORMALIZE,
])


class _HybridCollateFn:
    """Module-level class (not a closure) -- see _Siglip2CollateFn in
    siglip2_model.py for why: DataLoader(num_workers>0) on macOS pickles the
    collate_fn to send to worker processes."""

    def __init__(self, tokenizer, site_to_idx: dict, max_length: int = 64, train: bool = True):
        self.tokenizer = tokenizer
        self.site_to_idx = site_to_idx
        self.max_length = max_length
        self.transform = _TRAIN_TRANSFORM if train else _EVAL_TRANSFORM

    def __call__(self, batch):
        titles = [item["title"] for item in batch]
        sites = [self.site_to_idx[item["site"]] for item in batch]
        labels = [float(item["label"]) for item in batch]
        pixel_values = torch.stack([self.transform(item["image"]) for item in batch])

        text_inputs = self.tokenizer(titles, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.max_length)

        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
            "site_idx": torch.tensor(sites, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def make_hybrid_collate_fn(tokenizer, site_to_idx: dict, train: bool = True):
    return _HybridCollateFn(tokenizer, site_to_idx, train=train)
