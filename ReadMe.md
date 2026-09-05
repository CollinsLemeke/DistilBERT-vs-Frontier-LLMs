# Carbon Aware Sentiment Analysis at Scale

> **Can a 67M-parameter fine-tuned model match frontier LLMs on mental health app review classification — at a fraction of the carbon cost?**
>
> **Short answer: on macro F1 it beats five of the seven, including GPT-4, for 13.3 grams of CO₂ and five and a half minutes of training.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers)
[![DistilBERT](https://img.shields.io/badge/Model-distilbert--base--uncased-FFD21E)](https://huggingface.co/distilbert/distilbert-base-uncased)
[![CodeCarbon](https://img.shields.io/badge/🌱-CodeCarbon-green)](https://codecarbon.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MHARD-blueviolet)](https://github.com/Sensify-Lab/MHARD)
[![Task B Accuracy](https://img.shields.io/badge/3--class%20Accuracy-88.35%25-success)](#results)
[![Macro F1](https://img.shields.io/badge/3--class%20Macro%20F1-0.734%20(2nd%20of%208)-success)](#the-ranking-flips-depending-on-the-metric)
[![Training CO2](https://img.shields.io/badge/Training%20CO₂-13.3%20g-brightgreen)](#carbon-efficiency-analysis)
[![License](https://img.shields.io/badge/Code-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Motivation and Research Question](#motivation-and-research-question)
- [Headline Results](#headline-results)
- [Dataset: MHARD](#dataset-mhard)
- [Model: DistilBERT](#model-distilbert)
- [Experimental Design](#experimental-design)
- [Project Architecture and Pipeline](#project-architecture-and-pipeline)
- [Key Design Decisions and Justifications](#key-design-decisions-and-justifications)
- [Results](#results)
  - [Head to Head, Task A](#head-to-head-task-a--5-class-rating)
  - [Head to Head, Task B](#head-to-head-task-b--3-class-sentiment)
  - [The Ranking Flips Depending on the Metric](#the-ranking-flips-depending-on-the-metric)
  - [Per-Class Performance](#per-class-performance)
  - [Where It Fails](#where-it-fails)
- [Carbon Efficiency Analysis](#carbon-efficiency-analysis)
- [Limitations](#limitations)
- [How to Reproduce](#how-to-reproduce)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Citations and Acknowledgements](#citations-and-acknowledgements)
- [Author](#author)
- [License](#license)

---

## Overview

This project fine-tunes **DistilBERT** ([Sanh et al., 2019](#model-and-tooling)) — specifically `distilbert-base-uncased`, 66,955,779 parameters — for automated sentiment analysis of mental health app reviews, and benchmarks it against **seven frontier Large Language Models** whose predictions ship with the **MHARD** dataset ([Wang et al., ICWSM 2025](#dataset)): GPT-3.5 Instruct, GPT-3.5 Turbo, GPT-4, Gemini 1.5 Flash, Gemini 1.5 Pro, LLaMA 3.1 8B, and LLaMA 3.3 70B.

Two classification heads are trained from the same encoder:

- **Task A** — 5-class ordinal rating prediction (predict the exact 1–5 star rating)
- **Task B** — 3-class sentiment classification (negative / neutral / positive)

Carbon emissions during both training and inference are measured with **CodeCarbon** ([Courty et al.](#model-and-tooling)), which turns the environmental argument from a claim into a measurement.

Every model is evaluated on **the identical 20,082 held-out reviews**. Because the LLM predictions come bundled with the dataset, this is a genuine head-to-head rather than a comparison against published numbers from different test sets.

This notebook constitutes the empirical core of the MSc dissertation *"Reducing AI Carbon Footprint: A Study of DistilBERT for Mental Health Sentiment Analysis"*.

---

## Motivation and Research Question

The AI industry is consuming more energy than ever. Training and running frontier LLMs requires significant compute and therefore significant carbon. Yet many real-world NLP tasks — such as classifying the sentiment of user reviews — are well-defined, structured problems where a smaller, purpose-built model may be just as effective.

> *Can a fine-tuned DistilBERT (67M parameters) match or closely approach the predictive performance of frontier LLMs (8B to 1.76T parameters) on mental health app review classification, while emitting orders of magnitude less CO₂?*

This is not merely academic. Mental health app developers, researchers, and regulators increasingly need to understand how users feel about these tools at scale. Automating that with an energy-efficient local model rather than paying for GPT-4 API calls per prediction has economic, environmental **and privacy** implications — a locally-run model means review text never leaves your infrastructure.

---

## Headline Results

| | DistilBERT | Best LLM | Gap |
|---|---|---|---|
| **3-class accuracy** | 88.35% | 90.73% (LLaMA 3.3 70B) | −2.38 pp |
| **3-class macro F1** | **0.7338 — 2nd of 8** | 0.7473 (Gemini 1.5 Pro) | −0.0135 |
| **5-class accuracy** | 71.90% | 74.94% (GPT-4) | −3.04 pp |
| **5-class macro F1** | **0.5664 — 3rd of 8** | 0.5791 (Gemini 1.5 Pro) | −0.0127 |
| **5-class MAE (stars)** | 0.3479 | 0.3130 (GPT-4) | +0.0349 |
| **Parameters** | **67 million** | 1.76 trillion (GPT-4, est.) | **26,000× smaller** |
| **Training CO₂** | **13.3 g, both models** | not disclosed | — |
| **Inference CO₂** | **0.0041 mg / prediction** | not disclosed | — |
| **Throughput** | **12,300 reviews/sec** | API rate-limited | — |
| **Failed predictions** | **0 of 20,082** | up to 1,039 (LLaMA 3.1 8B) | — |

On the 5-class task, DistilBERT's macro F1 of **0.5664** sits **0.0012 below GPT-4's 0.5676**. That is a difference of roughly one prediction in a thousand, between a 67M-parameter model trained in three minutes and a system three orders of magnitude larger.

---

## Dataset: MHARD

**MHARD — Mental Health App Reviews Dataset**
**Authors:** Wang, Erqsous, Khatiwada, Karwankar, Alhassan, Chandrasekaran, Abraham, Lovell, Ngo & Mauriello — University of Delaware
**Paper:** *Leveraging Large Language Models for Review Classification and Rating Estimation of Mental Health Applications*, ICWSM 2025, 19(1), 2017–2029. [DOI: 10.1609/icwsm.v19i1.35916](https://doi.org/10.1609/icwsm.v19i1.35916)
**Repository:** [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD) (MIT licence)
**Size:** 200,973 reviews across 73 mental health apps, March 2011 to July 2023

Full credit for collection, annotation and the LLM prediction runs belongs to the MHARD authors. This project contributes only the fine-tuned DistilBERT models, the carbon measurement, and the comparative analysis. **Cite MHARD if you use it** — see [Citations](#citations-and-acknowledgements).

Each row contains the review text, the ground-truth star rating given by the user, and predicted ratings from seven LLMs — which is what makes the head-to-head possible without re-running any API calls.

> **A note on row counts.** The MHARD paper reports 200,973 reviews; the CSV read into this notebook returned 200,972. A single-row discrepancy, most likely a header or export artefact. It has no bearing on any result here.

### The MHARD authors' own finding

The original paper reports that their best supervised learning method achieved an F1-score of **0.79** while requiring significantly more human effort, whereas GPT-4 and Gemini 1.5 Pro delivered strong out-of-the-box performance at an overall F1-score of **0.76**.

That framing — supervised fine-tuning wins on quality but costs human effort — is the starting point this project pushes on. The question here is what that fine-tuning costs in *carbon and compute*, and whether the effort is justified once you measure it. (Metric definitions differ between the two papers, so treat the 0.79 as context rather than a directly comparable number to the figures below.)

### Class Distribution

The dataset has a strong positive skew, a structural feature of app review data known as the J-curve, where satisfied and very dissatisfied users are most motivated to write:

| Rating | Test-set count | Share |
|--------|---------------|-------|
| 1-star | 3,457 | 17.2% |
| 2-star | 892 | 4.4% |
| 3-star | 1,110 | 5.5% |
| 4-star | 2,509 | 12.5% |
| 5-star | 12,114 | 60.3% |

These proportions match those reported in the MHARD paper for the full corpus, confirming the split preserved the distribution.

Collapsed to three classes: negative 4,349 (21.7%), neutral 1,110 (5.5%), positive 14,623 (72.8%).

**Class imbalance ratio:** 13.59× on the 5-class task, 13.17× on the 3-class task.

**The majority-class baseline on Task B is 72.82%.** Always predicting "positive" gets you nearly 73% accuracy without a model at all. Any accuracy figure on this dataset has to be read against that floor — which is precisely why macro F1 is the primary metric here.

### After cleaning

| Split | Size |
|---|---|
| Train | 160,649 |
| Validation | 20,081 |
| Test | **20,082** |
| **Total** | **200,812** |

Roughly 160 rows were removed from the raw file: null reviews, empty-after-normalisation rows, and reviews under three words.

### Missing LLM predictions

The MHARD repository documents this directly: some LLM predictions are missing because the models returned unexpected outputs during the original experiments — bracketed numbers, lengthy explanations, or error messages instead of a parseable rating.

That is not a flaw in the dataset; it is an honest record of what running frontier models at scale actually looks like, and it becomes a finding in its own right in the [results](#where-it-fails).

---

## Model: DistilBERT

**`distilbert-base-uncased`** — 66,955,779 parameters, six transformer layers, from the Hugging Face Hub.

DistilBERT ([Sanh et al., 2019](#model-and-tooling)) is a distilled version of BERT ([Devlin et al., 2019](#model-and-tooling)): roughly 40% smaller and 60% faster, retaining approximately 97% of BERT's language understanding on the GLUE benchmark. It was produced through knowledge distillation during pretraining, with the smaller student network trained to reproduce the teacher's output distribution.

That design goal is the reason this project uses it. The model was built to answer the same question this dissertation asks — how much capability survives compression — and it is a natural candidate when the argument is about efficiency rather than raw capability.

All training uses the Hugging Face `transformers` library ([Wolf et al., 2020](#model-and-tooling)), with `DistilBertForSequenceClassification` and a custom `Trainer` subclass for the class-weighted loss.

- Model card: [huggingface.co/distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- Licence: Apache 2.0

---

## Experimental Design

Both tasks use the **same** stratified split, the **same** tokenisation, and the **same** hyperparameters except epoch count. This makes Task A and Task B metrics directly comparable, and it means the LLM baselines are evaluated on exactly the same rows.

**Stratification is performed on the 5-class rating**, not the 3-class collapse. Stratifying on the finer grid automatically preserves the coarser distribution; the reverse is not true.

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Standard overall correctness — but see the majority baseline above |
| Weighted F1 | Class-size-weighted F1 |
| **Macro F1** | Unweighted mean F1 across classes. **The primary metric**, because it treats minority classes equally and cannot be gamed by ignoring them |
| MAE (stars) | Ordinal error magnitude, Task A only. Predicting 4 for a true 5 is a smaller mistake than predicting 1 |
| Cohen's κ | Agreement between DistilBERT and each LLM |
| CO₂eq | Emissions for training and inference, via CodeCarbon |
| Latency / throughput | Operational efficiency |

All classification metrics computed with scikit-learn ([Pedregosa et al., 2011](#model-and-tooling)).

---

## Project Architecture and Pipeline

The notebook runs 17 sequential steps. Condensed walkthrough:

**Steps 1–2 — Setup and reproducibility.** Mounts Drive, installs dependencies, fixes every source of randomness to seed 42 (Python `random`, NumPy, PyTorch, CUDA, HuggingFace `set_seed`), and checks the device.

**Steps 3–5 — Load and profile.** Reads 200,972 rows × 17 columns. Analyses missingness (`response` is ~74% missing and dropped; `review` and `rating` are near-complete with ~21 nulls). Quantifies class imbalance and computes balanced inverse-frequency class weights.

**Step 6 — `max_length` decision.** DistilBERT supports 512 tokens, but compute scales with sequence length. Word-count statistics plus a 2,000-review empirical tokenisation check gave **`MAX_LENGTH = 128`**, covering **99.45% of reviews without truncation**, with a median tokenised length of just **30 tokens**. This is a carbon decision made with evidence rather than a default.

**Step 7 — Minimal cleaning.** Only four operations: strip whitespace, replace URLs, collapse 3+ character repetitions to 2, collapse whitespace runs. Classical NLP cleaning (stemming, stopword removal, lowercasing) would destroy signal the pretrained tokeniser was built to interpret.

**Step 8 — EDA and LLM preview.** Word clouds per rating, log-odds distinctiveness analysis, and a pairwise agreement heatmap between LLMs. That heatmap shows the top-tier models agree with **each other** more than with ground truth, which suggests shared systematic bias in how LLMs read star ratings.

**Steps 9–11 — Labels, split, tokenisation.** 0-indexed labels for both schemes, LLM predictions remapped to the same schema, stratified 80/10/10 split with UID-overlap assertions, then tokenisation with `padding=False` and `DataCollatorWithPadding` for **dynamic per-batch padding** — which eliminates roughly 75% of wasted compute on padding tokens, given a 30-token median.

**Step 12 — Model, weighted loss, metrics.** A `WeightedLossTrainer` subclass overrides `compute_loss()` to inject class-weighted cross-entropy. A dry-run forward pass verifies the weighted loss actually differs from unweighted before any GPU time is committed.

**Steps 13–16 — Training and evaluation.** Task A for 3 epochs, Task B for 2. Full CodeCarbon tracking on both training and inference, inside `try/finally` so the tracker always stops cleanly.

**Step 17 — Head-to-head.** DistilBERT against all seven LLMs on the identical test rows, with Cohen's κ.

---

## Key Design Decisions and Justifications

| Decision | Choice | Why |
|----------|--------|-----|
| Input text | Raw `review` column | DistilBERT's tokeniser handles casing and punctuation; cleaning destroys signal |
| `max_length` | 128 tokens | Covers 99.45% of reviews; ~4× less compute than 512 for 0.55% truncation cost |
| Padding | Dynamic per-batch | Median review is 30 tokens; static padding to 128 wastes ~75% of compute |
| Class imbalance | Weighted CrossEntropyLoss | 5-star is 13.59× more frequent than 2-star; unweighted, the model predicts "5" by default |
| Best-model metric | Macro F1 | Accuracy and weighted F1 can be gamed by ignoring minority classes |
| Precision | fp16 | ~40% speedup, ~50% memory saving, negligible quality cost = direct carbon reduction |
| **Task B epochs** | **2 rather than 3** | Task A's validation macro F1 was 0.5698 at epoch 1 and 0.5716 at epoch 3 — flat. Cutting the third epoch is an **evidence-based carbon saving**, not a compromise |
| Stratification column | 5-class rating | Preserves both label schemes simultaneously |
| Single split | Same rows for both tasks | Required for comparable metrics and a fair LLM comparison |

---

## Results

All figures read from the stored outputs of the notebook committed to this repository.

### Task A — 5-class rating

**Training:** 3 epochs, 15,060 steps, wall time **3.23 minutes**, **7.973 g CO₂**. Final training loss 0.8879, best validation macro F1 0.5809.

| Metric | Value |
|---|---|
| Accuracy | **71.90%** |
| Weighted F1 | 0.7436 |
| **Macro F1** | **0.5664** |
| MAE (stars) | 0.3479 |
| Eval loss | 0.9983 |
| Throughput | 12,303 samples/sec |
| Latency | 0.08 ms/sample |
| Inference CO₂ | 0.0829 g total, **0.0041 mg per prediction** |

### Task B — 3-class sentiment

**Training:** 2 epochs, 10,040 steps, wall time **2.16 minutes**, **5.334 g CO₂**. Final training loss 0.5072, best validation macro F1 0.7390.

| Metric | Value |
|---|---|
| Accuracy | **88.35%** |
| Weighted F1 | 0.8981 |
| **Macro F1** | **0.7338** |
| Eval loss | 0.5557 |
| Majority-class baseline | 72.82% |
| **Uplift over baseline** | **+15.54 pp** |
| Throughput | 12,324 samples/sec |
| Latency | 0.08 ms/sample |
| Inference CO₂ | 0.0820 g total, **0.0041 mg per prediction** |

### Head to Head, Task A — 5-class rating

All models on the identical 20,082 test rows. LLM predictions from MHARD. Sorted by accuracy.

| Model | n evaluated | n missing | Accuracy | Weighted F1 | Macro F1 | MAE |
|---|---|---|---|---|---|---|
| GPT-4 | 20,030 | 52 | **0.7494** | 0.7634 | 0.5676 | **0.3130** |
| LLaMA 3.3 70B | 19,995 | 87 | 0.7411 | 0.7524 | 0.5542 | 0.3309 |
| Gemini 1.5 Pro | 20,080 | 2 | 0.7386 | 0.7566 | **0.5791** | 0.3244 |
| **DistilBERT (67M)** | **20,082** | **0** | **0.7190** | **0.7436** | **0.5664** | **0.3479** |
| GPT-3.5 Instruct | 20,045 | 37 | 0.6910 | 0.7230 | 0.5479 | 0.3726 |
| GPT-3.5 Turbo | 20,064 | 18 | 0.6661 | 0.7027 | 0.5162 | 0.4000 |
| Gemini 1.5 Flash | 19,859 | 223 | 0.6080 | 0.6473 | 0.4876 | 0.5125 |
| LLaMA 3.1 8B | 19,043 | 1,039 | 0.6072 | 0.6037 | 0.3551 | 0.6204 |

DistilBERT is 4th on accuracy but **3rd on macro F1**, and its 0.5664 sits **0.0012 below GPT-4's 0.5676**. It comfortably beats both GPT-3.5 variants, Gemini 1.5 Flash, and LLaMA 3.1 8B on every metric.

### Head to Head, Task B — 3-class sentiment

| Model | n evaluated | n missing | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|
| LLaMA 3.3 70B | 19,995 | 87 | **0.9073** | 0.9030 | 0.7065 |
| Gemini 1.5 Pro | 20,080 | 2 | 0.9064 | **0.9116** | **0.7473** |
| GPT-3.5 Instruct | 20,045 | 37 | 0.8982 | 0.9038 | 0.7299 |
| GPT-4 | 20,030 | 52 | 0.8961 | 0.9035 | 0.7302 |
| GPT-3.5 Turbo | 20,064 | 18 | 0.8942 | 0.8974 | 0.7099 |
| **DistilBERT (67M)** | **20,082** | **0** | **0.8835** | **0.8981** | **0.7338** |
| Gemini 1.5 Flash | 19,859 | 223 | 0.8629 | 0.8711 | 0.6646 |
| LLaMA 3.1 8B | 19,043 | 1,039 | 0.7832 | 0.7959 | 0.5690 |

### The Ranking Flips Depending on the Metric

This is the most interesting result in the project, and it is easy to miss if you only read the accuracy column.

**Task B ranked by accuracy — DistilBERT is 6th of 8.**
**Task B ranked by macro F1 — DistilBERT is 2nd of 8.**

| Rank | By accuracy | By macro F1 |
|---|---|---|
| 1 | LLaMA 3.3 70B (0.9073) | Gemini 1.5 Pro (0.7473) |
| 2 | Gemini 1.5 Pro (0.9064) | **DistilBERT (0.7338)** |
| 3 | GPT-3.5 Instruct (0.8982) | GPT-4 (0.7302) |
| 4 | GPT-4 (0.8961) | GPT-3.5 Instruct (0.7299) |
| 5 | GPT-3.5 Turbo (0.8942) | GPT-3.5 Turbo (0.7099) |
| 6 | **DistilBERT (0.8835)** | LLaMA 3.3 70B (0.7065) |
| 7 | Gemini 1.5 Flash (0.8629) | Gemini 1.5 Flash (0.6646) |
| 8 | LLaMA 3.1 8B (0.7832) | LLaMA 3.1 8B (0.5690) |

**Same predictions. Same test rows. Opposite conclusion.**

LLaMA 3.3 70B tops the accuracy table and finishes 6th on macro F1. DistilBERT does the reverse. The explanation is the class distribution: 72.8% of the test set is positive, so a model that leans positive scores well on accuracy while performing poorly on the 5.5% neutral class that accuracy barely registers.

The weighted-loss training explicitly optimised for the balanced view, and the ranking reflects that choice. **Which model is "best" here depends entirely on whether you care about the average review or about every class of review** — and for a mental health application, the neutral and negative reviews are the ones that matter operationally.

### Per-Class Performance

**Task A — 5-class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1-star | 0.8660 | 0.7368 | **0.7962** | 3,457 |
| 2-star | 0.3149 | 0.4854 | **0.3820** | 892 |
| 3-star | 0.3077 | 0.4730 | **0.3729** | 1,110 |
| 4-star | 0.3580 | 0.5249 | 0.4257 | 2,509 |
| 5-star | 0.9264 | 0.7939 | **0.8550** | 12,114 |
| **Macro avg** | 0.5546 | 0.6028 | **0.5664** | 20,082 |
| Weighted avg | 0.7836 | 0.7190 | 0.7436 | 20,082 |

**Task B — 3-class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.8837 | 0.8558 | **0.8695** | 4,349 |
| neutral | 0.2919 | 0.5712 | **0.3863** | 1,110 |
| positive | 0.9773 | 0.9155 | **0.9454** | 14,623 |
| **Macro avg** | 0.7176 | 0.7808 | **0.7338** | 20,082 |
| Weighted avg | 0.9191 | 0.8835 | 0.8981 | 20,082 |

### Where It Fails

**The middle of the scale.** On Task B, neutral scores F1 **0.3863** against 0.8695 for negative and 0.9454 for positive. The pattern is precision 0.2919 with recall 0.5712 — the model finds most neutral reviews but is wrong about two-thirds of the ones it labels neutral. That is the weighted loss doing exactly what it was told: over-predict the rare class to catch it, at the cost of precision.

The same shape appears on Task A, where 2-star (0.3820) and 3-star (0.3729) are the two weakest classes while 1-star and 5-star both clear 0.79. The extremes are easy; the middle is genuinely ambiguous, and a review saying "it has helped a little but nothing groundbreaking" sits somewhere between 3 and 4 for a human annotator too.

**Qualitative check.** On 50 random test reviews (seed 42): 36 correct (72.0%), 9 off by one (18.0%), **45 within ±1 (90.0%)**. The errors are near-misses on an ordinal scale, not category collapses.

**The operational point about missing predictions.** DistilBERT returned a prediction for all 20,082 rows. The API models did not — LLaMA 3.1 8B failed on 1,039 rows (5.2%), Gemini 1.5 Flash on 223, GPT-4 on 52.

The MHARD authors document the cause: the models returned bracketed numbers, lengthy explanations, or error messages rather than a parseable rating. In a production pipeline every one of those is a row needing a retry, a fallback, or manual review — and the failures are not random, they cluster on the inputs the model found hardest to categorise. A local classification head has no such failure mode: it always returns a distribution over the label set.

---

## Carbon Efficiency Analysis

Measured with CodeCarbon, which tracks CPU, GPU and RAM energy and converts using regional grid carbon intensity.

| Phase | Task | Duration | CO₂ |
|---|---|---|---|
| Training | Task A (5-class, 3 epochs) | 3.23 min | **7.973 g** |
| Training | Task B (3-class, 2 epochs) | 2.16 min | **5.334 g** |
| Inference | Task A test set (20,082 samples) | 1.63 s | 0.0829 g |
| Inference | Task B test set (20,082 samples) | 1.63 s | 0.0820 g |
| **Total** | both models, end to end | **~5.4 min** | **~13.5 g** |

### What that means per prediction

**0.0041 mg CO₂ per prediction.** At that rate:

- Classifying the entire 200,972-review MHARD corpus costs about **0.82 g CO₂**
- Classifying **one million** reviews costs about **4.1 g CO₂** — less than a third of what it took to train the model
- Training both models cost **13.3 g**, roughly the emissions of boiling a small cup of water

The training cost is a one-off. Every prediction after that is close to free, both in carbon and in money, and it runs on your own hardware.

### The comparison, honestly stated

The seven LLM baselines were run by the MHARD authors, and **their inference emissions are not disclosed**. Frontier providers do not publish per-token energy figures, so an exact multiplier is not available from this data.

What can be said precisely: each of those predictions triggered inference through a model between 8 billion and an estimated 1.76 trillion parameters, running in a remote data centre, with the review text travelling over the network in both directions. DistilBERT is 67 million parameters running locally at 12,300 predictions per second. The order-of-magnitude difference is not in dispute; the exact figure is simply not measurable from the public side.

**The privacy argument is arguably stronger than the carbon one.** Mental health app reviews are sensitive user text. A locally-run classifier means that data never leaves your infrastructure. No API contract, no data-processing agreement, no third-party retention policy to reason about.

---

## Limitations

- **Neutral class performance is poor.** F1 0.3863, precision 0.2919. If your application depends on identifying ambivalent users specifically, this model is not adequate as-is. Threshold tuning, focal loss, or a dedicated ordinal-regression head are the obvious next steps
- **Single run, single seed.** Seed 42 throughout, so the run is reproducible — but reproducible is not the same as representative. A mean and standard deviation across several seeds would be the stronger claim
- **LLM emissions are unmeasured**, for the reasons above. The carbon comparison is directional, not a precise ratio
- **The LLM predictions are pre-computed** by the MHARD authors. Their prompting strategy, temperature and parsing are fixed and not controlled by this work. Different prompts could produce different LLM results
- **Single domain.** Mental health app reviews from Google Play, English only, 2011–2023. Transfer to other review domains or other languages is untested
- **Label noise.** The "ground truth" is the star rating the user selected, which is itself a noisy proxy for sentiment. Users routinely write positive text and leave 3 stars, or vice versa. Part of the residual error on every model here is irreducible
- **Hardware variance.** The wall times logged (3.23 and 2.16 minutes) are considerably faster than a free-tier T4 would deliver on 160k samples. Emissions scale with the hardware actually used, so reproduce on your own setup before quoting the figures

---

## How to Reproduce

### Prerequisites

- Google Colab account (GPU runtime)
- Google Drive with at least 2 GB free
- MHARD dataset CSV — available from [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD)

### Steps

**1. Upload the dataset to Drive** at `My Drive/Colab Notebooks/DistilBERT/MHARD_dataset.csv`

**2. Open the notebook in Colab**

**3. Enable GPU:** `Runtime → Change runtime type → GPU`

**4. Run all cells in order.** Steps 13 and 15 are the training runs.

**5. Retrieve artifacts from Drive:**

```
My Drive/Colab Notebooks/DistilBERT/
├── distilbert_5class_final/          # 255 MB model.safetensors + tokenizer + metrics
├── distilbert_3class_final/          # same structure
├── emissions/
│   ├── emissions_5class_train.csv
│   ├── emissions_5class_test.csv
│   ├── emissions_3class_train.csv
│   └── emissions_3class_test.csv
└── comparison/
    ├── task_a_comparison.csv
    ├── task_b_comparison.csv
    ├── kappa_matrix_5class.csv
    ├── kappa_vs_truth.json
    └── test_set_predictions_all_models.csv   # every model's prediction on every test row
```

That last file is the one to open if you want to check any number in this README yourself.

### Changing the dataset path

Update in Step 3:
```python
CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/DistilBERT/MHARD_dataset.csv"
```

---

## Repository Structure

```
.
├── README.md                              # This file
├── Carbon_Aware_Sentiment_Analysis_at_Scale.ipynb
│                                          # Full pipeline, 17 steps, outputs included
├── requirements.txt                       # Python dependencies
└── emissions/                             # CodeCarbon CSV logs
```

Trained weights are **not included** (255 MB per model). They are written to Drive during training.

---

## Dependencies

```
transformers>=4.46.0
datasets>=2.18.0
evaluate>=0.4.0
accelerate>=0.27.0
torch>=2.0.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
wordcloud>=1.9.0
codecarbon>=2.3.0
```

**Note on `transformers` version:** the notebook uses `processing_class=tokenizer` in `Trainer`, the updated argument name from `transformers>=4.46`. On older versions, replace with `tokenizer=` in Steps 13 and 15.

---

## Citations and Acknowledgements

This project stands on three pieces of other people's work: the MHARD dataset and its LLM prediction runs, the DistilBERT model, and the Hugging Face and CodeCarbon tooling. Please cite them.

### Dataset

**MHARD** — Mental Health App Reviews Dataset, University of Delaware.
Repository: [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD) · Contact: Kyle Wang (kylewang@udel.edu), Moath Erqsous (merqsous@udel.edu)

```bibtex
@article{wang2025mhard,
  title     = {Leveraging Large Language Models for Review Classification
               and Rating Estimation of Mental Health Applications},
  author    = {Wang, Qiaoyu and Erqsous, Moath and Khatiwada, Pallav and
               Karwankar, Ashutosh and Alhassan, Fatimah Mohammed and
               Chandrasekaran, Aravind and Abraham, Bettina and
               Lovell, Fiona and Ngo, An Ai and Mauriello, Matthew Louis},
  journal   = {Proceedings of the International AAAI Conference on
               Web and Social Media},
  volume    = {19},
  number    = {1},
  pages     = {2017--2029},
  year      = {2025},
  doi       = {10.1609/icwsm.v19i1.35916}
}
```

### Model and tooling

**DistilBERT** — the model fine-tuned throughout this project.
Model card: [huggingface.co/distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) · Apache 2.0

```bibtex
@article{sanh2019distilbert,
  title   = {DistilBERT, a distilled version of BERT: smaller, faster,
             cheaper and lighter},
  author  = {Sanh, Victor and Debut, Lysandre and Chaumond, Julien and
             Wolf, Thomas},
  journal = {arXiv preprint arXiv:1910.01108},
  year    = {2019},
  note    = {5th Workshop on Energy Efficient Machine Learning and
             Cognitive Computing, NeurIPS 2019}
}
```

**BERT** — the teacher model DistilBERT was distilled from.

```bibtex
@inproceedings{devlin2019bert,
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers
               for Language Understanding},
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and
               Toutanova, Kristina},
  booktitle = {Proceedings of NAACL-HLT 2019},
  pages     = {4171--4186},
  year      = {2019}
}
```

**Hugging Face Transformers** — the library used for tokenisation, training and inference.

```bibtex
@inproceedings{wolf2020transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and
               Chaumond, Julien and Delangue, Clement and Moi, Anthony and
               Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and
               Funtowicz, Morgan and Davison, Joe and Shleifer, Sam and
               von Platen, Patrick and Ma, Clara and Jernite, Yacine and
               Plu, Julien and Xu, Canwen and Le Scao, Teven and
               Gugger, Sylvain and Drame, Mariama and Lhoest, Quentin and
               Rush, Alexander M.},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in
               Natural Language Processing: System Demonstrations},
  pages     = {38--45},
  year      = {2020},
  publisher = {Association for Computational Linguistics}
}
```

**CodeCarbon** — emissions tracking.
Repository: [github.com/mlco2/codecarbon](https://github.com/mlco2/codecarbon)

```bibtex
@software{codecarbon,
  title  = {CodeCarbon: Estimate and Track Carbon Emissions from
            Machine Learning Computing},
  author = {Courty, Beno{\^i}t and Schmidt, Victor and
            Luccioni, Sasha and others},
  url    = {https://github.com/mlco2/codecarbon}
}
```

**scikit-learn** — all classification metrics and class-weight computation.

```bibtex
@article{pedregosa2011scikit,
  title   = {Scikit-learn: Machine Learning in {P}ython},
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and
             Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and
             Prettenhofer, P. and Weiss, R. and Dubourg, V. and
             Vanderplas, J. and Passos, A. and Cournapeau, D. and
             Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}
```

### This work

```bibtex
@misc{lemeke2026distilbert_mhard,
  title  = {Reducing AI Carbon Footprint: A Study of DistilBERT for
            Mental Health Sentiment Analysis},
  author = {Lemeke, Collins},
  year   = {2026},
  note   = {MSc Artificial Intelligence Dissertation,
            University of Greater Manchester.
            Centre of Intelligence of Things (CIoTh).
            Supervisor: Prof. Celestine Iwendi.},
  url    = {https://github.com/CollinsLemeke/DistilBERT-vs-Frontier-LLMs}
}
```

---

## Author

**Collins Lemeke** — full pipeline design, implementation, training, evaluation and analysis.

MSc Artificial Intelligence (Distinction), University of Greater Manchester. AI Research Engineer, Centre of Intelligence of Things. Co-founder, AI Nexus Society. IEEE Member.

This is part of a wider research programme on reading internal state from observable signals — across facial expression, physiological sensing, gait and language:

- [Facial Expression Recognition with CNN](https://github.com/CollinsLemeke/Facial-Expression-Recognition-Model) — imbalance-aware evaluation on FER2013
- [Autism Facial Emotion Classification](https://github.com/CollinsLemeke/Autism-Facial-Emotion-Classification) — VGG16 transfer learning, published at IEEE IC3ECSBHI 2026
- [Detecting Cognitive Decline, Falls and Frailty](https://github.com/CollinsLemeke/Detecting-Cognitive-Decline-Falls-and-Frailty) — interpretable screening from gait sensor data

The thread connecting them is the same one running through this project: a headline number is not a result until you know which class it is hiding.

- [GitHub](https://github.com/CollinsLemeke)
- [Kaggle](https://www.kaggle.com/collinslemeke/code)

For questions, open an issue.

---

## License

**Code: MIT.** Free to use, modify, and distribute. See [LICENSE](LICENSE).

**Dataset:** MHARD is released by its authors under the MIT licence and is **not redistributed here**. Obtain it from [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD) and cite Wang et al. (2025).

**Model:** `distilbert-base-uncased` is released under Apache 2.0 by Hugging Face. Fine-tuned weights derived from it inherit that licence.

---

> *67 million parameters. Five and a half minutes. Thirteen grams of CO₂. Second place on macro F1, ahead of GPT-4.*
