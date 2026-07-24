"""Baseline models — implemented in Step 2, after the data pipeline (Step 1).

Each isolates a different input modality so the ablation study (see
configs/base_config.yaml -> evaluation.ablation) can attribute performance
gains correctly -- see that file's docstring for why this matters:

  1. tfidf_logreg(...)      - TF-IDF on title text + sklearn LogisticRegression.
                               The classical-ML floor every DL model must beat.
  2. text_only_bert(...)     - Hebrew BERT (AlephBERT) fine-tuned on title+tags
                               only, no image. Class-weighted loss (same
                               imbalance-handling approach as the main model's
                               config.training.class_weight_positive).
  3. image_only(...)        - Frozen SigLIP2 vision encoder on the thumbnail
                               only (no text), pooled image embeddings feeding
                               a LogisticRegression head -- "frozen" means only
                               a linear head is fit, same spirit as baseline 1.
  4. title_image_frozen(...) - First combined-modality check: frozen SigLIP2
                               image + text embeddings concatenated, one
                               LogisticRegression head. Cheap way to see
                               whether title+image beats either alone, before
                               investing in the main model's staged
                               fine-tuning (which adds cross-modal attention
                               and actual gradient updates through the
                               backbone -- this baseline has neither).

Runs on CUDA (Colab) or Apple Silicon MPS automatically (src.utils.get_device).

Usage:
    python -m src.models.baselines --config configs/base_config.yaml --model tfidf_logreg
    python -m src.models.baselines --config configs/base_config.yaml --model text_only_bert
    python -m src.models.baselines --config configs/base_config.yaml --model image_only
    python -m src.models.baselines --config configs/base_config.yaml --model title_image_frozen
"""
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.download_images import _cache_path
from src.evaluation.metrics import compute_metrics
from src.utils import get_device


def tfidf_logreg(train_df, val_df, k_values=(50, 100, 200)):
    """TF-IDF (title only) + LogisticRegression. Returns (metrics, fitted_pipeline)."""
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["teaser_title"])
    X_val = vectorizer.transform(val_df["teaser_title"])

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, train_df["target"])

    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(val_df["target"].values, y_pred_proba, k_values=k_values)
    return metrics, {"vectorizer": vectorizer, "model": clf}


def text_only_bert(train_df, val_df, checkpoint="onlplab/alephbert-base",
                    k_values=(50, 100, 200), num_epochs=3, batch_size=16, max_length=64,
                    learning_rate=3e-5, warmup_ratio=0.1,
                    output_dir="experiments/baseline_text_bert"):
    """Fine-tune AlephBERT on title+tags text. Returns (metrics, {tokenizer, model, trainer})."""
    import torch
    from torch import nn
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                               Trainer, TrainingArguments)

    class _TextDataset(TorchDataset):
        def __init__(self, df, tokenizer, max_length):
            texts = (df["teaser_title"].fillna("") + " " + df["tags"].fillna("")).tolist()
            self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
            self.labels = df["target"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    train_ds = _TextDataset(train_df, tokenizer, max_length)
    val_ds = _TextDataset(val_df, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer)

    # Class-weighted CE loss for the ~10% positive rate -- mirrors
    # configs/base_config.yaml -> training.class_weight_positive.
    pos_rate = train_df["target"].mean()
    class_weights = torch.tensor([1.0, (1 - pos_rate) / pos_rate], dtype=torch.float)

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )

    trainer = _WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator
    )
    trainer.train()

    preds = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
    metrics = compute_metrics(val_df["target"].values, probs, k_values=k_values)
    return metrics, {"tokenizer": tokenizer, "model": model, "trainer": trainer}


def image_only(train_df, val_df, checkpoint="google/siglip2-base-patch16-224",
               images_dir="data/images/", k_values=(50, 100, 200), batch_size=32, device=None):
    """Frozen SigLIP2 vision encoder -> pooled image embeddings -> LogisticRegression head."""
    import torch
    from transformers import AutoModel, AutoProcessor

    device = device or get_device()
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()

    def embed(df):
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(df), batch_size):
                urls = df["pic_furl"].iloc[start:start + batch_size]
                images = [Image.open(_cache_path(u, images_dir)).convert("RGB") for u in urls]
                inputs = processor(images=images, return_tensors="pt").to(device)
                out = model.get_image_features(**inputs)
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
                embeddings.append(feats.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    X_train = embed(train_df)
    X_val = embed(val_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, train_df["target"])

    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(val_df["target"].values, y_pred_proba, k_values=k_values)
    return metrics, {"processor": processor, "vision_model": model, "scaler": scaler, "classifier": clf}


def title_image_frozen(train_df, val_df, checkpoint="google/siglip2-base-patch16-224",
                        images_dir="data/images/", k_values=(50, 100, 200), batch_size=32, device=None):
    """First look at combining modalities: frozen SigLIP2 image embeddings +
    frozen SigLIP2 text embeddings (of the title), concatenated, feeding one
    LogisticRegression head. Not the main model (no cross-modal attention,
    no fine-tuning) -- just the cheapest possible answer to "does having
    both beat either alone?" before investing in full staged fine-tuning."""
    import torch
    from transformers import AutoModel, AutoProcessor

    device = device or get_device()
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()

    def embed(df):
        image_feats, text_feats = [], []
        with torch.no_grad():
            for start in range(0, len(df), batch_size):
                batch = df.iloc[start:start + batch_size]
                images = [Image.open(_cache_path(u, images_dir)).convert("RGB") for u in batch["pic_furl"]]
                titles = batch["teaser_title"].fillna("").tolist()

                img_inputs = processor(images=images, return_tensors="pt").to(device)
                img_out = model.get_image_features(**img_inputs)
                img_out = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
                image_feats.append(img_out.cpu().numpy())

                txt_inputs = processor(text=titles, return_tensors="pt", padding=True, truncation=True).to(device)
                txt_out = model.get_text_features(**txt_inputs)
                txt_out = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
                text_feats.append(txt_out.cpu().numpy())
        return np.concatenate(image_feats, axis=0), np.concatenate(text_feats, axis=0)

    img_train, txt_train = embed(train_df)
    img_val, txt_val = embed(val_df)
    X_train = np.concatenate([img_train, txt_train], axis=1)
    X_val = np.concatenate([img_val, txt_val], axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, train_df["target"])

    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(val_df["target"].values, y_pred_proba, k_values=k_values)
    return metrics, {"processor": processor, "model": model, "scaler": scaler, "classifier": clf}


if __name__ == "__main__":
    import argparse

    import pandas as pd

    from src.utils import load_config, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model",
                         choices=["tfidf_logreg", "text_only_bert", "image_only", "title_image_frozen"],
                         default="tfidf_logreg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    k_values = tuple(cfg["evaluation"]["precision_at_k_values"])

    train_df = pd.read_csv(f"{data_cfg['processed_dir']}/train.csv")
    val_df = pd.read_csv(f"{data_cfg['processed_dir']}/val.csv")

    if args.model == "tfidf_logreg":
        metrics, _ = tfidf_logreg(train_df, val_df, k_values=k_values)
    elif args.model == "text_only_bert":
        metrics, _ = text_only_bert(train_df, val_df, k_values=k_values)
    elif args.model == "image_only":
        metrics, _ = image_only(train_df, val_df, images_dir=data_cfg["images_dir"], k_values=k_values)
    else:
        metrics, _ = title_image_frozen(train_df, val_df, images_dir=data_cfg["images_dir"], k_values=k_values)

    print(f"{args.model} on val:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
