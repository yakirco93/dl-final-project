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

  unfreeze_last_n_layers and use_lora can be combined (LoRA is applied
  first, then the top-N layers' original weights are re-unfrozen on top --
  order matters, since get_peft_model() otherwise freezes everything that
  isn't a LoRA param, undoing an earlier unfreeze).

Results so far (see project_explanation_HE.md section 11): Stage A
(0.651/0.167 ROC-AUC/PR-AUC) and Stage C/LoRA (0.659/0.184, best of the
three) never beat the text_only_bert baseline (0.704/0.212); Stage B
(full unfreeze, 57M params) overfit almost immediately. The naive
concatenation fusion (also what src/models/baselines.py's
title_image_frozen does) is a suspected cause -- it doesn't let text and
image actually condition on each other. `fusion: "cross_attention"` (vs.
the original `"concat"`) replaces the concat+MLP head with two
cross-attention passes (text pools over image patches, image pools over
text tokens) before the final MLP, on the theory that real cross-modal
interaction might extract signal concatenation can't.

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

        self.fusion_type = self.siglip_cfg.get("fusion", "concat")
        if self.fusion_type == "cross_attention":
            self._build_cross_attention_head(image_dim, text_dim, site_embed_dim)
        elif self.fusion_type == "concat":
            self._build_concat_head(image_dim, text_dim, site_embed_dim)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type!r}")

    def _build_concat_head(self, image_dim, text_dim, site_embed_dim):
        head_hidden_dim = 256
        self.head = nn.Sequential(
            nn.Linear(image_dim + text_dim + site_embed_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1),
        )

    def _build_cross_attention_head(self, image_dim, text_dim, site_embed_dim):
        fusion_dim = self.siglip_cfg.get("fusion_dim", 512)
        num_heads = self.siglip_cfg.get("fusion_num_heads", 8)

        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        # Two directions: the pooled text embedding queries over image patch
        # tokens ("what in this image matches the headline?"), and the pooled
        # image embedding queries over text tokens ("which words does the
        # photo support?"). Each is a single-query attention (query len 1).
        self.text_queries_image = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        self.image_queries_text = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)

        head_hidden_dim = 256
        # [image-conditioned-by-text, text-conditioned-by-image, pooled image,
        #  pooled text, site] -- attended contexts plus the plain pooled
        # vectors, so the head still has direct access to unconditioned
        # embeddings alongside the cross-attended ones.
        head_input_dim = fusion_dim * 4 + site_embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1),
        )

    def _apply_freeze_strategy(self):
        cfg = self.siglip_cfg
        if cfg.get("freeze_backbone", True):
            for p in self.backbone.parameters():
                p.requires_grad = False

        if cfg.get("use_lora", False):
            from peft import LoraConfig, get_peft_model

            lora_cfg = cfg["lora"]
            peft_config = LoraConfig(
                r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                target_modules=["q_proj", "v_proj"], bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        # Applied *after* LoRA wrapping (if any): get_peft_model() freezes
        # every non-LoRA base param, which would silently undo an earlier
        # unfreeze. Doing this last lets "unfreeze top N" and "LoRA
        # everywhere" combine -- top-N layers get both their original
        # weights unfrozen AND LoRA adapters, everything below only LoRA.
        n = cfg.get("unfreeze_last_n_layers", 0)
        if n > 0:
            for tower in (self.backbone.vision_model, self.backbone.text_model):
                for layer in tower.encoder.layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

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
        if self.fusion_type == "cross_attention":
            return self._forward_cross_attention(batch)
        return self._forward_concat(batch)

    def _forward_concat(self, batch):
        img_feat = self._pooled(self.backbone.get_image_features(**batch["image_inputs"]))
        txt_feat = self._pooled(self.backbone.get_text_features(**batch["text_inputs"]))
        site_feat = self.site_embedding(batch["site_idx"])
        combined = torch.cat([img_feat, txt_feat, site_feat], dim=-1)
        return self.head(combined).squeeze(-1)

    def _forward_cross_attention(self, batch):
        # Full (unpooled) hidden states -- patch tokens / word tokens -- not
        # just the pooled get_image_features()/get_text_features() outputs,
        # since attention needs something to attend *over*. SigLIP2's image
        # processor always resizes to a fixed 224x224 (196 patches, no
        # padding); its tokenizer pads text to a fixed length without
        # returning an attention_mask (no masking needed either -- see
        # investigation notes in project_explanation_HE.md).
        vision_out = self.backbone.vision_model(**batch["image_inputs"])
        text_out = self.backbone.text_model(**batch["text_inputs"])

        img_tokens = self.image_proj(vision_out.last_hidden_state)   # (B, n_patches, fusion_dim)
        txt_tokens = self.text_proj(text_out.last_hidden_state)      # (B, seq_len, fusion_dim)
        img_pooled = self.image_proj(vision_out.pooler_output)       # (B, fusion_dim)
        txt_pooled = self.text_proj(text_out.pooler_output)          # (B, fusion_dim)

        img_context, _ = self.text_queries_image(
            query=txt_pooled.unsqueeze(1), key=img_tokens, value=img_tokens
        )
        txt_context, _ = self.image_queries_text(
            query=img_pooled.unsqueeze(1), key=txt_tokens, value=txt_tokens
        )

        site_feat = self.site_embedding(batch["site_idx"])
        combined = torch.cat(
            [img_context.squeeze(1), txt_context.squeeze(1), img_pooled, txt_pooled, site_feat], dim=-1
        )
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
