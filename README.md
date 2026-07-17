# Predicting Relative Article Popularity Using Multimodal Transfer Learning

Deep Learning Final Project (Technion DS 26-529625). Predicts whether a newly published
mako.co.il / n12.co.il article will land in the top 10% of page views (first 48h)
among articles on the same site in the same month, using only pre-publication
signals: headline, tags, thumbnail image, and site.

## Project Structure

```
.
├── configs/            # YAML experiment configs (one per run — no hardcoded params in code)
├── data/
│   ├── raw/             # raw SQL extracts (CSV), git-ignored
│   ├── processed/        # cleaned/split datasets, git-ignored
│   └── images/           # downloaded + cached thumbnails, git-ignored
├── notebooks/           # EDA and one-off diagnostics (not production code)
├── src/
│   ├── data/             # extraction, image download/cache, Dataset classes
│   ├── models/           # baseline models, SigLIP2 wrapper, fallback fusion model
│   ├── training/          # training loops, LoRA setup, schedulers
│   └── evaluation/        # metrics, ablation runner
├── experiments/         # saved checkpoints + logs per run, git-ignored
├── report/               # final PDF + figures
├── requirements.txt
└── README.md
```

## Workflow

1. **Data pipeline** (`src/data/`): extract from Snowflake → download/cache images
   (run locally, not from Colab — see note below) → time-based train/val/test split.
2. **Baselines** (`src/models/baselines.py`): TF-IDF+LR, text-only BERT, image-only.
3. **Main model** (`src/models/siglip2_model.py`): staged fine-tuning of SigLIP2.
4. **Evaluation** (`src/evaluation/`): metrics + ablation study.
5. **Report**: `report/`, following the course's 4-6 page conference format.

## Known constraint: image downloads and Google Colab

The mako.co.il image CDN blocks Google Cloud IP ranges (confirmed via diagnostic:
~60% failure rate from Colab vs. 96.5% success from a local network). Run
`src/data/download_images.py` from a local machine, not from a Colab cell, then
sync the resulting `data/images/` cache (e.g. via Drive or git-lfs) for use in
Colab training.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

## Reproducibility

All experiment parameters (model name, learning rate, batch size, freeze strategy,
random seed, etc.) live in `configs/*.yaml` — never hardcoded in `src/`. Every
training run is launched with an explicit config path and logs to `experiments/<run_name>/`.
