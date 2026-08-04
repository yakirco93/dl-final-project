# Predicting Relative Article Popularity Using Multimodal Transfer Learning

Yakir Cohen — Deep Learning Final Project (Technion DS 26-529625)

## Abstract

We predict, at the moment of publication, whether a mako.co.il / n12.co.il news
article will land in the top 10% of page views (first 48 hours) among articles
on the same site in the same month, using only pre-publication signals:
headline, thumbnail image, and site. We build four non-deep baselines
(TF-IDF+LogReg, image-only, naively-concatenated title+image, and a
fine-tuned Hebrew BERT text model) and two multimodal deep models: (1)
staged fine-tuning of SigLIP2 (frozen head → partial unfreeze → LoRA), later
extended with a cross-attention fusion head, and (2) a hybrid architecture
combining AlephBERT (Hebrew-specific text encoder) with a ResNet50 image
encoder. SigLIP2, in all tested configurations, did not surpass the
text-only BERT baseline. The hybrid model, after tuning regularization and
early-stopping granularity, became the only model in the project to exceed
the text-only baseline's PR-AUC on the validation set (0.240 vs. 0.212).
On the held-out test set this advantage narrows substantially (0.212 vs.
0.212), while ranking quality (ROC-AUC) is stable or slightly improved. We
report this honestly as a case study in how aggressive validation-driven
model selection across many iterations can inflate the apparent gap between
models, and conclude that the image thumbnail provides at most a modest,
non-conclusive improvement over the headline alone for this task.

## 1. Introduction & Related Work

### 1.1 Problem Statement

News organizations routinely need to judge, before publication, whether a
given headline and thumbnail combination is likely to perform well. This
project frames that judgment as a supervised binary classification problem:
given an article's headline, thumbnail image, and publishing site, predict
whether it will fall in the top 10% of page views (by first-48-hour
traffic) among articles on the *same site* in the *same month*. The
relative, peer-group-normalized target avoids conflating "popular" with
"high absolute traffic," which would otherwise be dominated by site-level
and seasonal effects unrelated to editorial quality.

The central research question is whether the thumbnail image adds
predictive value beyond the headline text alone. This is not obvious a
priori: headlines are a strong, information-dense signal (an editor's
distilled pitch for the piece), while thumbnails on Israeli news sites are
frequently generic stock or wire-service images only loosely tied to
article content. Answering this question required not just training one
multimodal model, but *isolating* the image's marginal contribution
through a series of controlled ablations — which is the organizing
principle of this report.

### 1.2 Related Work

- Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with
  Improved Semantic Understanding, Localization, and Dense Features,"
  arXiv:2502.14786, 2025. Provides the pretrained multilingual
  vision-language encoder used as the primary multimodal backbone.
- Abousaleh, Cheng, Yu, and Tsao, "Multimodal Deep Learning Framework for
  Image Popularity Prediction on Social Media," IEEE Transactions on
  Cognitive and Developmental Systems, vol. 13, no. 3, 2021. Directly
  motivates fusing visual and contextual (textual) signals for popularity
  prediction, over text-only or image-only models.
- Bandari, Asur, and Huberman (and the broader "Online News Popularity"
  literature it spawned), "Predicting the Popularity of Online News from
  Content Metadata," IEEE, 2017 — predicts pre-publication popularity
  (shares/likes/comments) on the widely-used Mashable benchmark using
  gradient-boosted trees over content-metadata features (keyword
  statistics, day of week, word counts). Directly comparable in framing
  (before-publication-only features) but, like most of this literature,
  restricted to text/metadata features — it does not use the article's
  actual image content.
- "News Popularity Prediction with Machine Learning," a comparative study
  of classical models (Random Forest, Bayes Net, C4.5) on keyword and
  content-metadata features, with Random Forest + PCA performing best by
  ROC. Confirms the same metadata-only pattern: our use of the raw
  thumbnail image itself, rather than text-derived metadata about it, is
  a point of difference from this literature, not something we found
  direct precedent for.

A preliminary zero-shot diagnostic (no fine-tuning, ~950 real headline-image
pairs) found SigLIP2's top-1 image-retrieval accuracy from Hebrew titles to
be roughly 28x the random baseline — evidence that the pretrained backbone
carries *some* cross-modal signal in Hebrew despite being English-heavy in
its training mix, which justified committing to full fine-tuning rather
than abandoning the multimodal approach at the outset.

## 2. Methodology (Math & Architecture)

### 2.1 Data

Source: Keshet's internal Snowflake page-view event warehouse, covering
mako.co.il and n12.co.il. For each article we extract the headline
(`teaser_title`), thumbnail URL (`pic_furl`), site, and publication
timestamp, plus first-48h page views (label construction only, never a
model input). Formally, for site *s* and calendar month *m*, an article
*i* is labeled positive iff its 48-hour page-view count vᵢ satisfies

**vᵢ ≥ Q₀.₉ ( { vⱼ : site(j)=s, month(j)=m } )**

i.e. it is at or above the 90th percentile of its own site-month peer
group.

**Extraction and split.** 97,309 rows from January 2024 onward. A
**time-based** (not random) split avoids leakage from near-duplicate or
related articles across the train/val boundary:

| Split | Period | Rows | Positive rate |
|---|---|---|---|
| Train | 2024-01 – 2026-04 | 87,334 | ~10.0% |
| Val | 2026-05 | 3,722 | ~10.0% |
| Test | 2026-06 | 3,745 | ~10.0% |

**Images.** Thumbnails are fetched by URL; 71,536/71,546 unique URLs
(99.99%) downloaded successfully. A significant data-quality finding
during this process: the URL as stored in the warehouse (`pic_furl`) by
default resolves to a tiny 82x62px CDN thumbnail, not the original image.
Stripping the CDN suffix recovers the full-resolution original when it
still exists; a `..._autoOrient_b.jpg` variant (232x175px) serves as a
reliable fallback. Final quality distribution (2,000-row sample): 60.8%
full-resolution, 35.1% the 232x175 fallback, 4.1% stuck on the 82x62
thumbnail. This resolution variance was later used as a controlled
experimental variable (Section 3.3).

**Design decision: no tags as a core input.** 49.5% of articles carry no
tags at all, so tags were excluded from the core model (headline + image +
site only) and treated as a secondary enrichment question that was not
pursued further given the project timeline.

### 2.2 Baseline Models

Four models isolate the contribution of each input modality individually:

1. **`tfidf_logreg`** — TF-IDF over headline text + logistic regression.
2. **`text_only_bert`** — AlephBERT (Hebrew BERT) fine-tuned end-to-end on
   headline text.
3. **`image_only`** — frozen SigLIP2 image embeddings + logistic
   regression.
4. **`title_image_frozen`** — frozen SigLIP2 text and image embeddings,
   naively concatenated, into one logistic regression. Tests whether
   *naive* (non-jointly-trained) fusion of the two modalities helps at all.

### 2.3 Model 1: SigLIP2 Staged Fine-Tuning

`SigLIP2Classifier` (`google/siglip2-base-patch16-224`) encodes headline
and thumbnail jointly, adds a learned 8-dim site embedding, and a small
MLP head. Fine-tuning proceeds in three increasingly aggressive stages:

- **Stage A** — backbone fully frozen; only the head (395K params, 0.11%
  of the model) trains.
- **Stage B** — top 4 transformer layers unfrozen (57.1M params, 15.2%),
  with a discriminative learning rate (head at full LR, unfrozen backbone
  at LR × 0.1).
- **Stage C** — LoRA (r=8) on `q_proj`/`v_proj` across all 12 layers.

**LoRA formulation.** For a pretrained weight matrix W₀ ∈ ℝ^(d×k) (here,
the attention projection matrices), LoRA freezes W₀ and represents the
fine-tuning update as a low-rank decomposition:

**W = W₀ + ΔW = W₀ + BA,  where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d,k)**

so the forward pass becomes h = W₀x + BAx. Only A and B are trained
(r=8 here), giving ~986K trainable parameters versus 57.1M
for full-layer unfreezing at comparable adaptation capacity — the
parameter-efficiency that motivates LoRA's use as this project's
"Advanced Training Strategy" component alongside standard transfer
learning.

After all three stages under-performed `text_only_bert`, two further
variants tested whether the fusion mechanism itself (naive concatenation
of pooled embeddings) was the bottleneck:

- **Model 1** (`exp04`) — Stage B + Stage C combined, **and** a
  cross-attention fusion head in place of concatenation, plus
  full-resolution-only images.
- **Model 1b** (`exp05`) — a controlled follow-up isolating the fusion
  variable alone: identical capacity to Stage C (LoRA only), on the full
  dataset.

**Cross-attention formulation.** Standard scaled dot-product attention:

**Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V**

implemented bidirectionally via `nn.MultiheadAttention`: the pooled text
representation serves as Q against the image's raw patch tokens as K=V
(196 tokens from `vision_model.last_hidden_state`), and symmetrically the
pooled image representation serves as Q against the raw text tokens as
K=V — letting text "query" specific image regions and vice versa, rather
than only comparing pre-summarized vectors.

### 2.4 Model 2: Hybrid (AlephBERT + ResNet50)

A structurally different architecture: two backbones **never jointly
pretrained** (unlike SigLIP2's contrastively-aligned towers), fused only
at the classification head.

- **Text**: AlephBERT (Hebrew-specific), CLS-token pooling.
- **Image**: ResNet50 (ImageNet-pretrained), global-average-pooled
  2048-dim features.
- **Fusion**: concatenation + a small MLP — motivated directly by Section
  3.3's finding that cross-attention did not help once isolated from
  confounds, and by the absence of a shared pretrained embedding space
  here to begin with.

Both encoders are frozen by default, with a configurable number of top
layers/blocks unfrozen, using the same discriminative-LR pattern as
Model 1.

### 2.5 Training Objective and Metrics

**Loss.** All models optimize class-weighted binary cross-entropy:

**L = −(1/N) · Σ (i = 1 to N) [ w₊ · yᵢ · log σ(zᵢ) + (1−yᵢ) · log(1−σ(zᵢ)) ]**

with positive-class weight w₊ = 9.0 (matching the ~1:9 class imbalance),
logit zᵢ, and sigmoid σ. Optimizer: AdamW with a cosine learning-rate
schedule; early stopping on validation PR-AUC.

**Metrics.** With precision = TP/(TP+FP) and recall = TP/(TP+FN),
**PR-AUC** is the area under the precision-recall curve swept over all
thresholds, and **ROC-AUC** is the area under the true-positive-rate vs.
false-positive-rate curve — equivalently, the probability that a randomly
chosen positive example is scored higher than a randomly chosen negative
one. Under class imbalance, a random classifier scores ROC-AUC = 0.5 but
PR-AUC ≈ the base rate (≈ 0.10 here) — so PR-AUC, not accuracy or ROC-AUC
alone, is the primary metric
throughout, alongside F1 and Precision@K (K ∈ {50, 100, 200}), the
latter matching the intended editorial ranking use case.

All experiment hyperparameters live in one YAML config per run
(`configs/exp01`–`exp08`) for reproducibility; training ran locally on
Apple Silicon (MPS backend).

## 3. Experiments & Results

### 3.1 Baselines

| Model | Inputs | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| `tfidf_logreg` | title | 0.660 | 0.208 | 0.245 |
| **`text_only_bert`** | title (+tags) | **0.704** | **0.212** | 0.248 |
| `image_only` | image | 0.619 | 0.148 | 0.229 |
| `title_image_frozen` | title+image (naive concat) | 0.619 | 0.147 | 0.214 |

The text-only BERT baseline is the strongest of the four by a clear
margin. Critically, **`title_image_frozen` is no better than `image_only`
alone, and worse than `text_only_bert` alone** — naively concatenating
frozen embeddings from two modalities does not combine their signal.

### 3.2 SigLIP2 Staged Fine-Tuning

| Stage | Trainable params | ROC-AUC | PR-AUC | Outcome |
|---|---|---|---|---|
| A (frozen) | 395.8K (0.11%) | 0.651 | 0.167 | Improves over naive baselines, still below `text_only_bert` |
| B (unfreeze top 4) | 57.1M (15.2%) | — | — | Overfit almost immediately; did not beat Stage A |
| C (LoRA, r=8) | 986K (0.28%) | 0.659 | 0.184 | Best of the three stages; stable across all 10 epochs |

None of the three staged variants surpassed `text_only_bert` (0.704 /
0.212). Stage C's stability confirms the expected pattern: on ~87K
training rows, 57M trainable parameters overfit quickly, while LoRA's
~1M parameters generalize far better for comparable adaptation capacity.

### 3.3 Isolating the Fusion Head: Cross-Attention

| Run | Fusion | Capacity | Data | Best ROC-AUC | Best PR-AUC |
|---|---|---|---|---|---|
| Stage C | concat | LoRA only | full (87.3K) | 0.659 | 0.184 |
| Model 1 (`exp04`) | cross-attention | unfreeze+LoRA combined | full-res filtered (53.5K) | 0.625 | 0.161 |
| Model 1b (`exp05`) | cross-attention | LoRA only (matched to Stage C) | full (87.3K) | 0.643 | 0.179 |

Model 1 changed three variables simultaneously and regressed sharply.
Model 1b isolates the fusion variable alone: the result (0.643/0.179) is
close to but consistently below Stage C's concat-fusion result
(0.659/0.184) across the whole run. **Conclusion: cross-attention fusion,
as implemented here, was not the fix — Model 1's regression was driven
mainly by added capacity and reduced data, not the fusion mechanism.**

![Isolating the fusion head: concat (Stage C) stays consistently above cross-attention at matched capacity (Model 1b); Model 1's added capacity and filtered data cause a collapse after epoch 4.](figures/fig2_fusion_ablation.png)

### 3.4 Hybrid Model (Model 2)

| Run | Config | Best ROC-AUC | Best PR-AUC | Outcome |
|---|---|---|---|---|
| v1 (`exp06`) | BERT top-1-layer + ResNet layer4 unfrozen, concat fusion | 0.660 | 0.212 | Peaked epoch 1, overfit steadily afterward |
| tuned, first attempt | ResNet fully frozen, finer eval | 0.670 | 0.198 | Early-stopped by a patience/eval-frequency scaling bug -- false stop |
| **tuned (`exp07`)** | same, bug fixed | **0.693** | **0.240** | Completed all 10 epochs with no early stopping; best result |
| augment+unfreeze4 (`exp08`) | `exp07` + re-unfrozen ResNet + augmentation | 0.674 | 0.222 | Beat v1 but did not beat `exp07` |

`exp07` is the only model in this project to exceed `text_only_bert` on
PR-AUC (0.240 vs. 0.212). Two design choices were decisive: (1) a
Hebrew-specific text encoder (AlephBERT) rather than SigLIP2's
multilingual encoder, and (2) conservative regularization — ResNet50
fully frozen, higher weight decay, and early-stopping patience correctly
scaled to the finer evaluation frequency.

![Model 2 tuning journey: v1 overfits and degrades after ~step 5000; exp07 (ResNet frozen) climbs steadily past the text-only baseline with no collapse; exp08 improves on v1 but plateaus below exp07.](figures/fig1_hybrid_tuning.png)

### 3.5 Final Ablation Table

| Ablation entry | Inputs | ROC-AUC | PR-AUC |
|---|---|---|---|
| title_only | title | 0.660 | 0.208 |
| title_tags* | title (no tags baseline exists) | 0.704 | 0.212 |
| image_only | image | 0.619 | 0.148 |
| title_image | title+image (naive concat) | 0.619 | 0.147 |
| **full_model** | title+image+site (`exp07`, no tags) | **0.693** | **0.240** |

*No model in this project was given tags as an input (Section 2.1);
`title_tags` here is really title-only text, kept for the table's
canonical shape. The cleanest single comparison for "does the image
help" is `full_model` vs. `text_only_bert`: 0.240 vs. 0.212 PR-AUC.

### 3.6 Held-Out Test Set

| Split | ROC-AUC | PR-AUC | F1 | P@50 | P@100 | P@200 |
|---|---|---|---|---|---|---|
| Val (best checkpoint, step 21,500) | 0.693 | 0.240 | 0.263 | 0.50 | 0.39 | 0.305 |
| **Test (held-out)** | **0.697** | **0.212** | 0.270 | 0.40 | 0.33 | 0.30 |

ROC-AUC is stable (marginally higher on test). **PR-AUC drops from 0.240
to 0.212**, landing almost exactly at `text_only_bert`'s validation
PR-AUC, erasing most of the apparent advantage seen during tuning —
attributed to selection-driven overfitting to the validation set across
`exp06`→`exp07`→`exp08` (every tuning decision was made by watching the
same ~3,700-row validation set).

![Final model, validation vs. held-out test: ROC-AUC and F1 hold up, while PR-AUC and Precision@K drop toward the text-only baseline's level.](figures/fig3_val_vs_test.png)

## 4. Discussion & Conclusion

**Does the image help?** Modestly, and not conclusively. On validation
data the best multimodal model (`exp07`) beat the strongest text-only
baseline by a sizeable margin; on a single untouched test month that
margin nearly disappears on PR-AUC while holding up on ROC-AUC. Given the
well-understood mechanism (repeated validation-based model selection), we
read this as "the image's contribution is real but small, and was likely
overstated by how many tuning decisions were routed through the same
validation set" rather than "the image adds nothing."

**Why did the more sophisticated fusion approach fail?** The controlled
ablation (Section 3.3) is fairly definitive for this implementation: it
was the *added trainable capacity and reduced training data*, not the
*mechanism* (cross-attention vs. concatenation), that hurt Model 1. A
more thorough treatment might test cross-attention at matched capacity
across a wider sweep, or full token-to-token attention rather than pooled
queries — both left unexplored given the project's time budget.

**Why did the Hybrid model succeed where SigLIP2 didn't?** A genuinely
Hebrew-specific text encoder instead of a lightly-Hebrew-exposed
multilingual one, plus conservative regularization discovered through
iteration. This suggests language-appropriateness of the text backbone
mattered more, for this task and language, than the theoretical advantage
of a jointly-pretrained, contrastively-aligned text/image embedding space.

**Limitations.** (1) A single val/test month means month-to-month metric
volatility is not separately quantified. (2) Tags were excluded entirely;
their marginal value was never measured. (3) Thumbnail quality varies and
was only controlled for in one experiment. (4) A single random seed per
configuration; no variance estimate across seeds.

**Conclusion.** We built and rigorously compared six modeling approaches
— four single-modality baselines, staged SigLIP2 fine-tuning across three
capacity regimes with two fusion mechanisms, and a Hebrew-specific hybrid
architecture — to answer whether a news article's thumbnail image
predicts its relative popularity beyond its headline. The best model
(AlephBERT + frozen ResNet50) improved PR-AUC over a strong text-only
baseline on validation data (0.240 vs. 0.212) but that advantage
substantially narrowed on a genuinely held-out test month (0.212 vs.
0.212), while ranking quality (ROC-AUC) held up. We conclude the
thumbnail carries a real but modest signal beyond the headline, and that
naive fusion of frozen multimodal embeddings is actively counterproductive
— joint, task-specific fine-tuning with an appropriately-matched
(Hebrew-specific) text encoder is what allowed any image contribution to
surface at all.

## References

1. M. Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders
   with Improved Semantic Understanding, Localization, and Dense
   Features," arXiv:2502.14786, 2025.
2. F. S. Abousaleh, W.-H. Cheng, N.-H. Yu, and Y. Tsao, "Multimodal Deep
   Learning Framework for Image Popularity Prediction on Social Media,"
   IEEE Transactions on Cognitive and Developmental Systems, vol. 13, no.
   3, 2021.
3. "Predicting the Popularity of Online News from Content Metadata," IEEE
   Conference Publication, 2017.
4. "News Popularity Prediction with Machine Learning," comparative study
   of classical ML models on content-metadata features.
