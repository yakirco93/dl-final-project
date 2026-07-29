# Predicting Relative Article Popularity Using Multimodal Transfer Learning

Yakir Cohen — Deep Learning Final Project (Technion DS 26-529625)

## What this project is about

News editors need to decide, before a story goes live, whether it's likely
to do well. This project builds a model that looks at a draft article's
headline, thumbnail image, and which site it's published on (mako.co.il or
n12.co.il), and predicts whether it will end up among the top 10% most-read
articles on that site that month.

We defined "did well" relative to other articles on the same site in the
same month, rather than using a fixed number of page views. That matters
because mako and n12 get very different amounts of traffic, and traffic
also swings with the news cycle and the season — an absolute threshold
would mostly just be measuring which site or which month an article
happened to run in, not whether it was actually a strong piece.

The real question we wanted to answer: **does the thumbnail image actually
help predict success, or is the headline basically all that matters?**
That question shaped almost everything about how we ran this project — not
just building one final model, but running a series of step-by-step
comparisons designed to isolate exactly what the image was or wasn't
contributing.

## The data

The data comes from Keshet's internal analytics for mako.co.il and n12.co.il.
For each article we have: the headline, the thumbnail image, the site, when
it was published, and how many page views it got in its first 48 hours
(used only to build the label — the model never sees view counts).

We split the data by time, not randomly, so the model is always tested on
articles published *after* everything it was trained on — this is closer
to how it would actually be used, and avoids the model accidentally
learning from near-duplicate articles that leaked across the split.

- **Training data:** ~87,300 articles, January 2024 through April 2026
- **Validation data:** ~3,700 articles, May 2026
- **Test data:** ~3,700 articles, June 2026 (used only once, at the very end)

About 10% of articles are "top performers" in every split, by design.

**A data quality issue worth mentioning:** the image URL stored in the
database points to a tiny 82×62 pixel thumbnail by default, not the real
image. We found that stripping a suffix from the URL recovers the
full-size original about 61% of the time, with a decent 232×175 fallback
covering most of the rest. We only discovered this partway through the
project, and it mattered enough that we treated image resolution as
something to explicitly control for in one of our experiments (more on
that below).

**One deliberate simplification:** about half of all articles have no tags
at all, so we left tags out of the core model entirely. It's headline +
image + site only.

## What we built and tried

We didn't jump straight to one final model. We built four simple
baselines first, specifically so we could isolate what each *type* of
input (text alone, image alone, or a naive combination of both)
contributes on its own:

- **Headline text only**, two ways: a classic TF-IDF + logistic regression
  model (no deep learning at all), and a fine-tuned Hebrew BERT model.
- **Image only** — a pretrained image model's embeddings fed into a simple
  classifier.
- **Headline + image, naively combined** — just gluing together frozen
  text and image embeddings from a pretrained model, with no real training
  of how they interact.

Then we built two real multimodal models — models that see the headline
and the image together and are actually trained to combine them:

**Model 1 (SigLIP2)** — a general-purpose pretrained model that understands
both text and images together. We tried fine-tuning it three different
ways, from "barely touch it" to "adjust it quite a lot," to find the right
balance — too little adjustment and it can't specialize to our task, too
much and it starts memorizing the training set instead of learning general
patterns. We also tried a smarter way of combining the text and image
signals ("cross-attention," letting the model actively relate specific
words to specific parts of the image) instead of just gluing the two
together.

**Model 2 (Hybrid)** — after Model 1 consistently fell short of the
plain-text baseline, we tried a different combination: a Hebrew-specific
text model (AlephBERT) paired with a standard image model (ResNet50),
instead of one general multilingual model trying to do both. The idea was
that a text encoder actually built for Hebrew might matter more than a
model that was jointly pretrained on text and images but mostly in other
languages.

## What we found

**Gluing frozen embeddings together doesn't work.** The naive
"headline+image" baseline was no better than image-only, and worse than
headline-only. Two signals that aren't actually trained together don't
combine — if anything the weaker one drags the stronger one down. This
told us early on that a real, jointly-trained model was necessary if the
image was going to help at all.

**The general-purpose model (SigLIP2) never beat plain headline text**, no
matter how we fine-tuned it or fused it. We tested this carefully: when we
isolated the "smarter fusion" idea on its own (same model size, same data,
only the fusion method changed), it didn't help either. The real bottleneck
was overfitting — when we gave the model more room to adjust its image
understanding, it started memorizing quirks of the training set rather
than learning something generalizable, on a dataset of this size.

**The Hebrew-specific model (Model 2) was the one that worked.** After
some tuning — training with the image model mostly frozen, being more
careful about how often we checked for overfitting, adding some
regularization — this was the only version of the multimodal model that
beat the plain-headline-text baseline, on our validation data:

| Model | How well it ranks articles (ROC-AUC) | How well it finds the actual top performers (PR-AUC) |
|---|---|---|
| Headline text only (best baseline) | 0.704 | 0.212 |
| **Hybrid model (final, tuned)** | 0.693 | **0.240** |

PR-AUC is the more important number here — it's specifically about how
well the model identifies the relatively rare "top 10%" articles, which is
the actual goal, rather than just ranking articles in general.

We also tried pushing this model further afterward (letting the image
model adjust a bit more, adding some image augmentation during training)
— it improved over our very first attempt at this model, but didn't beat
the version above, so we kept the earlier, simpler one as final.

![The three attempts at Model 2, tracked over training. The first attempt (blue) improves quickly, then overfits and gets worse. The final tuned version (orange) climbs steadily past the text-only baseline with no such collapse. The augmented version (green) is more stable than the first attempt but doesn't reach the tuned version.](figures/fig1_hybrid_tuning.png)

## The honest catch: test set results

Everything above was measured on the validation set — the month of data
we used throughout the project to decide which version of the model was
best. To get an unbiased read, we ran the final model, exactly once, on
June 2026 data that had never been touched before:

| | ROC-AUC | PR-AUC | Precision @ 50 |
|---|---|---|---|
| Validation (what we used while tuning) | 0.693 | 0.240 | 0.50 |
| **Test (untouched, first use)** | **0.697** | **0.212** | 0.40 |

The general ranking ability (ROC-AUC) held up fine — it was actually
slightly better on the untouched data. But the headline number we were
excited about (PR-AUC 0.240, clearly ahead of the text-only baseline)
dropped to 0.212 on test — landing almost exactly even with the text-only
baseline, not ahead of it.

![Validation vs. test, across every metric we tracked. ROC-AUC and F1 hold up on untouched data; PR-AUC and the precision-at-K scores drop noticeably.](figures/fig3_val_vs_test.png)

We want to be upfront about why this happened rather than just report the
better-looking number: over several rounds of tuning, every decision about
which version of the model to keep was made by watching validation
performance. That process tends to quietly favor whichever run got a
slightly lucky validation score, even without anyone intending it to. The
test set — used only this one time, at the end — is what corrects for
that, and it's telling us the model's real advantage over plain text is
smaller than it looked mid-project.

## Bottom line

Does the thumbnail image help predict whether an article will do well?
**A little, but not by much, and we can't claim it conclusively.** The
image-plus-text model is at least as good as headline text alone, and
possibly modestly better — but the clear win we saw during development
mostly reflects how much tuning we did against the same validation data,
not a large, reliable effect from the image itself. Given how loosely tied
these thumbnails often are to actual article content, that's a reasonable
outcome, not a disappointing one.

What we're more confident about: naively combining a headline and an
image without training them together doesn't work at all, and using a
text model actually built for Hebrew mattered more than we initially
expected — more than the fancier "understands text and images jointly"
model we tried first.

## What we'd do differently with more time

- Test across more than one validation/test month, so we're not drawing
  conclusions from a single month's worth of noise.
- Actually measure what tags add, instead of leaving them out entirely.
- Try a few different random seeds per model, to know how much of any
  result is just run-to-run variance.
- Be more disciplined about limiting how many times we look at the
  validation set before finalizing a model — the test-set drop above is a
  pretty direct illustration of why that discipline matters.

## Sources

- Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with
  Improved Semantic Understanding, Localization, and Dense Features,"
  2025 (arXiv:2502.14786) — the pretrained model used for the first
  multimodal approach.
- Abousaleh, Cheng, Yu, and Tsao, "Multimodal Deep Learning Framework for
  Image Popularity Prediction on Social Media," IEEE Transactions on
  Cognitive and Developmental Systems, 2021 — prior work motivating the
  idea of combining image and text signals for popularity prediction.
