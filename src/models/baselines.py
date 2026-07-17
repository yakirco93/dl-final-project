"""Baseline models — implemented in Step 2, after the data pipeline (Step 1).

Three baselines, each isolating a different input modality so the ablation
study (see configs/base_config.yaml -> evaluation.ablation) can attribute
performance gains correctly:

  1. tfidf_logreg(...)   - TF-IDF on title text + sklearn LogisticRegression.
                            The classical-ML floor every DL model must beat.
  2. text_only_bert(...)  - Hebrew BERT (AlephBERT/HeBERT) fine-tuned on
                            title+tags only, no image.
  3. image_only(...)     - Frozen pretrained vision encoder on the thumbnail
                            only, no text.
"""


def tfidf_logreg(train_df, val_df):
    raise NotImplementedError("Step 2")


def text_only_bert(train_df, val_df, checkpoint="onlplab/alephbert-base"):
    raise NotImplementedError("Step 2")


def image_only(train_df, val_df, checkpoint="google/siglip2-base-patch16-224"):
    raise NotImplementedError("Step 2")
