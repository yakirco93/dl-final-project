"""Main model: SigLIP2 fine-tuned for binary top-10% classification.

Implemented in Steps 4-5, staged per configs/base_config.yaml -> model.siglip2:

  Stage A (freeze_backbone: true):
      Load AutoModel.from_pretrained(checkpoint), freeze all backbone params,
      add a small trainable classification head on top of the pooled
      text+image embeddings (concat or learned gating).

  Stage B (unfreeze_last_n_layers > 0):
      Unfreeze the top N transformer layers of the backbone; lower the
      learning rate for backbone params relative to the head (discriminative
      fine-tuning).

  Stage C (use_lora: true):
      Wrap the backbone with peft.LoraConfig instead of full unfreezing —
      cheaper and lower overfitting risk with a ~150K-article dataset.

This class intentionally does NOT implement the fallback hybrid model
(AlephBERT/HeBERT + EfficientNet/ResNet + fusion head) — that lives in
src/models/hybrid_fallback.py and is only built if this model underperforms
on Hebrew text, per the contingency plan in the proposal.
"""


class SigLIP2Classifier:
    def __init__(self, config: dict):
        raise NotImplementedError("Step 4: build after baselines (Step 2) establish the floor.")

    def forward(self, batch):
        raise NotImplementedError
