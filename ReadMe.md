# Carbon Aware Sentiment Analysis at Scale

### Can a tiny AI model do the job of a giant one, for a fraction of the carbon?

**Short answer: yes, for this task.** A 67-million-parameter model called DistilBERT was fine-tuned in five and a half minutes for about 13 grams of CO₂, and on the fairest quality measure it beat GPT-4, LLaMA 3.3 70B, and four other frontier language models on the exact same 20,082 reviews.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers)
[![DistilBERT](https://img.shields.io/badge/Model-distilbert--base--uncased-FFD21E)](https://huggingface.co/distilbert/distilbert-base-uncased)
[![CodeCarbon](https://img.shields.io/badge/🌱-CodeCarbon-green)](https://codecarbon.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MHARD-blueviolet)](https://github.com/Sensify-Lab/MHARD)
[![3-class Macro F1](https://img.shields.io/badge/3--class%20Macro%20F1-0.734%20(2nd%20of%208)-success)](#the-result-that-matters-most-the-ranking-flips)
[![Training CO2](https://img.shields.io/badge/Training%20CO₂-13.3%20g-brightgreen)](#7-the-carbon-analysis)
[![License](https://img.shields.io/badge/Code-MIT-lightgrey)](LICENSE)

---

## Contents

1. [Read this first, in plain English](#1-read-this-first-in-plain-english)
2. [Glossary for beginners](#2-glossary-for-beginners)
3. [The headline numbers](#3-the-headline-numbers)
4. [The data](#4-the-data)
5. [The model and how it was trained](#5-the-model-and-how-it-was-trained)
6. [Results](#6-results)
7. [The carbon analysis](#7-the-carbon-analysis)
8. [Where it fails](#8-where-it-fails)
9. [Limitations, stated honestly](#9-limitations-stated-honestly)
10. [How to reproduce](#10-how-to-reproduce)
11. [Repository structure and dependencies](#11-repository-structure-and-dependencies)
12. [Citations](#12-citations)
13. [Author](#13-author)
14. [Licence](#14-licence)

---

## 1. Read this first, in plain English

### The problem

People write reviews of mental health apps. Millions of them. If you run one of those apps, or study them, or regulate them, you want to know what users actually feel. Reading every review by hand is impossible, so you automate it.

The obvious way to automate it in 2026 is to send every review to a large language model like GPT-4. That works. It is also expensive, slow, sends private text to someone else's servers, and burns a surprising amount of electricity.

### The question

Does the task actually need a giant model? Sorting a review into "positive", "neutral", or "negative" is a narrow, well-defined job. A small model trained specifically for that one job might do it just as well.

### What was done

1. Took **DistilBERT**, a small open model with 67 million internal settings. GPT-4 is estimated at 1.76 trillion, roughly 26,000 times more.
2. Trained it twice on mental health app reviews: once to guess the exact star rating (1 to 5), once to guess the overall sentiment (negative, neutral, positive).
3. Measured the carbon emissions of training and running it, using a tool called CodeCarbon.
4. Compared it against **seven frontier language models** on the **identical 20,082 reviews**. This was possible because the dataset already ships with each of those models' predictions, so no API calls were needed and nothing is being compared across different test sets.

### What was found

Three things, in order of how interesting they are:

1. **The small model is competitive.** It came 2nd of 8 on the fairest quality measure for the sentiment task, ahead of GPT-4.
2. **Which model "wins" depends entirely on which measure you read.** On raw accuracy the small model ranks 6th of 8. On macro F1 it ranks 2nd. Same predictions, same reviews, opposite conclusion. Section 6 explains exactly why.
3. **The carbon difference is enormous.** Roughly four to six orders of magnitude per prediction, which is the difference between grams and hundreds of kilograms once you scale to a full dataset.

### What this is not

It is not a claim that small models beat large ones in general. It is a claim about **one narrow, well-defined classification task**, where a purpose-built model is a reasonable alternative to a general-purpose one. Ask the large model to write, reason, or handle a task it has never seen, and this comparison tells you nothing.

---

## 2. Glossary for beginners

Read this once and the rest of the README will make sense.

| Term | Plain English |
|---|---|
| **Model** | A program that learned patterns from examples rather than being written rule by rule. |
| **Parameters** | The internal numbers a model adjusts while learning. More parameters means more capacity, more compute, and more energy. DistilBERT has 67 million. GPT-4 is estimated at 1.76 trillion. |
| **LLM** | Large Language Model. A very big, general-purpose model such as GPT-4 or Gemini. |
| **Fine-tuning** | Taking a model that already understands language and training it a bit more on your specific task. Far cheaper than training from scratch. |
| **Training vs inference** | Training is teaching the model, done once. Inference is using it to make a prediction, done millions of times. |
| **Epoch** | One complete pass through the training data. |
| **Token** | A chunk of text, roughly three quarters of a word. Models read tokens, not words. |
| **Class** | One of the possible answers. Here: five star ratings, or three sentiment labels. |
| **Class imbalance** | When some answers are far more common than others. Here, 5-star reviews are about 13 times more common than 2-star ones. |
| **Accuracy** | The share of predictions that were right. Simple, but easy to game when classes are imbalanced. |
| **Macro F1** | The average quality score across all classes, **treating each class as equally important**. This is the primary measure in this project, because it cannot be gamed by ignoring rare classes. |
| **Weighted F1** | Same idea, but bigger classes count for more. Sits between accuracy and macro F1. |
| **MAE** | Mean Absolute Error. For star ratings, how far off the guess was on average. Guessing 4 when the truth is 5 is a smaller error than guessing 1. |
| **Cohen's κ (kappa)** | How much two raters agree, after removing the agreement you would expect from luck alone. 0 is chance level, 1 is perfect. |
| **Confusion matrix** | A grid showing what the model predicted against what was true, so you can see exactly which classes it mixes up. |
| **Majority-class baseline** | The score you get by always guessing the most common answer, with no model at all. Any real model must beat this to be worth anything. |
| **CO₂eq** | Carbon dioxide equivalent, the standard unit for greenhouse gas emissions. |
| **CodeCarbon** | An open-source tool that watches CPU, GPU, and memory usage and converts the energy used into estimated CO₂, using the local electricity grid's carbon intensity. |

### The one idea worth understanding before the results

**Accuracy can lie when your data is lopsided.**

In this dataset, 72.8% of reviews are positive. So a model that ignored the text entirely and always answered "positive" would score **72.8% accuracy**. That sounds respectable and means nothing.

Macro F1 closes that loophole. It scores each class separately and then averages, so the rare "neutral" class counts exactly as much as the huge "positive" class. A model that ignores neutral reviews gets punished.

**For a mental health app, the neutral and negative reviews are the ones that matter operationally.** Those are the ambivalent and unhappy users. So macro F1 is not just the statistically fairer measure here, it is the one that matches what the tool would actually be used for.

---

## 3. The headline numbers

| | DistilBERT (this project) | Best frontier LLM | Gap |
|---|---|---|---|
| **3-class macro F1** | **0.7338, 2nd of 8** | 0.7473 (Gemini 1.5 Pro) | −0.0135 |
| 3-class accuracy | 88.35% | 90.73% (LLaMA 3.3 70B) | −2.38 pp |
| **5-class macro F1** | **0.5664, 3rd of 8** | 0.5791 (Gemini 1.5 Pro) | −0.0127 |
| 5-class accuracy | 71.90% | 74.94% (GPT-4) | −3.04 pp |
| 5-class MAE (stars) | 0.3479 | 0.3130 (GPT-4) | +0.0349 |
| Agreement with truth (Cohen's κ) | 0.556, 4th of 8 | 0.587 (GPT-4) | −0.031 |
| **Parameters** | **67 million** | 1.76 trillion (GPT-4, estimated) | **~26,000× smaller** |
| **Training carbon** | **13.3 g CO₂, both models** | Not published | Not comparable |
| **Inference carbon** | **0.0041 mg per prediction, measured** | Estimated at 90 to 3,300 mg | 4 to 6 orders of magnitude |
| Throughput | 12,300 reviews per second, locally | Limited by API rate limits | Not comparable |
| **Failed predictions** | **0 of 20,082** | Up to 1,039 (LLaMA 3.1 8B) | See section 8 |

On the 5-class task, DistilBERT's macro F1 of **0.5664** sits **0.0012 below GPT-4's 0.5676**. That is a difference of roughly one prediction in a thousand, between a model trained in three minutes on free hardware and a system three orders of magnitude larger.

---

## 4. The data

**MHARD, the Mental Health App Reviews Dataset**, from the University of Delaware.

- **What it is:** 200,973 user reviews of 73 mental health apps on Google Play, March 2011 to July 2023
- **What each row contains:** the review text, the star rating the user actually gave, and predicted ratings from seven different LLMs
- **Paper:** Wang et al., *Leveraging Large Language Models for Review Classification and Rating Estimation of Mental Health Applications*, ICWSM 2025. [DOI: 10.1609/icwsm.v19i1.35916](https://doi.org/10.1609/icwsm.v19i1.35916)
- **Repository:** [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD), MIT licence

**The bundled LLM predictions are what make this project possible.** Without them, comparing against seven frontier models would mean around 140,000 API calls. Full credit for collecting the data, annotating it, and running those models belongs to the MHARD authors. This project contributes the fine-tuned DistilBERT models, the carbon measurement, and the comparative analysis.

> **A small housekeeping note.** The MHARD paper reports 200,973 reviews. The CSV read into this notebook returned 200,972. A one-row difference, almost certainly a header or export artefact. It changes nothing.

### What the original authors found

The MHARD paper reports that their best supervised method reached an F1 of **0.79** but required significantly more human effort, while GPT-4 and Gemini 1.5 Pro delivered strong out-of-the-box performance at an overall F1 of **0.76**.

That framing is the starting point this project pushes on. If fine-tuning wins on quality but costs human effort, the obvious next question is what it costs in **carbon and compute**, and whether that cost is actually high. (Metric definitions differ between the two papers, so treat 0.79 as context rather than a directly comparable number.)

### The shape of the data

App reviews follow a J-curve. Delighted users and furious users both write. Mildly satisfied users mostly do not.

| Rating | Test-set count | Share |
|--------|---------------|-------|
| 1-star | 3,457 | 17.2% |
| 2-star | 892 | 4.4% |
| 3-star | 1,110 | 5.5% |
| 4-star | 2,509 | 12.5% |
| 5-star | 12,114 | 60.3% |

Collapsed into three sentiment classes: negative 4,349 (21.7%), neutral 1,110 (5.5%), positive 14,623 (72.8%).

**Imbalance ratio:** 13.59× on the 5-class task, 13.17× on the 3-class task. **Majority-class baseline on the 3-class task: 72.82%.**

### After cleaning

| Split | Size | Purpose |
|---|---|---|
| Train | 160,649 | The model learns from these |
| Validation | 20,081 | Used to pick the best checkpoint during training |
| Test | **20,082** | Touched only once, at the very end |
| **Total** | **200,812** | |

About 160 rows were removed: null reviews, rows that became empty after normalisation, and reviews shorter than three words.

### Missing LLM predictions

Some LLM predictions are missing. The MHARD authors document why: the models sometimes returned bracketed numbers, long explanations, or error messages instead of a parseable rating.

That is not a flaw in the dataset. It is an honest record of what running frontier models at scale looks like, and it becomes a finding in its own right in [section 8](#8-where-it-fails).

---

## 5. The model and how it was trained

### The model

**`distilbert-base-uncased`**, 66,955,779 parameters, six transformer layers, from the Hugging Face Hub, Apache 2.0 licence.

DistilBERT is a compressed version of BERT: roughly 40% smaller, 60% faster, and it retains about 97% of BERT's language understanding on the standard GLUE benchmark. It was produced through **knowledge distillation**, where a small "student" model is trained to reproduce the behaviour of a larger "teacher".

That origin is exactly why it suits this project. The model was built to answer the same question the dissertation asks: how much capability survives compression?

### Two tasks, one encoder

- **Task A:** predict the exact star rating, 1 to 5. Five classes, and they are **ordinal**, meaning the order carries meaning.
- **Task B:** predict sentiment. Negative (1 to 2 stars), neutral (3), positive (4 to 5).

Both use the **same split, same tokenisation, same settings** except the number of epochs. That is what makes the two sets of numbers directly comparable, and what makes the LLM comparison fair, since every model is scored on exactly the same rows.

**Stratification was done on the 5-class rating**, not the 3-class collapse. Preserving the finer grid automatically preserves the coarser one. The reverse is not true.

### The pipeline, 17 steps condensed

**Steps 1 and 2, setup.** Mount Drive, install libraries, fix every source of randomness to seed 42 so the run is reproducible, check the GPU.

**Steps 3 to 5, load and inspect.** Read 200,972 rows and 17 columns. Check what is missing (the `response` column is about 74% empty and gets dropped; `review` and `rating` are nearly complete). Measure the class imbalance and compute balanced class weights.

**Step 6, the `max_length` decision.** DistilBERT can read up to 512 tokens, but compute cost rises with length. Word-count statistics plus an actual tokenisation test on 2,000 reviews showed that **128 tokens covers 99.45% of reviews with no truncation**, and the median review is only **30 tokens**. Choosing 128 over 512 cuts compute roughly fourfold for a 0.55% truncation cost. **This is a carbon decision made with evidence rather than a default.**

**Step 7, minimal cleaning.** Only four operations: trim whitespace, replace URLs, collapse runs of 3 or more repeated characters down to 2, and collapse whitespace runs. Traditional NLP cleaning such as stemming, stopword removal, and lowercasing is deliberately **not** applied, because the pretrained tokeniser was built to interpret exactly the signal that cleaning would destroy.

**Step 8, exploration.** Word clouds per rating, a distinctiveness analysis of which words separate ratings, and a pairwise agreement check between the LLMs. That last check surfaces something important: **the top LLMs agree with each other more than they agree with the ground truth**, which points to shared systematic bias in how they read star ratings.

**Steps 9 to 11, labels, split, tokenisation.** Encode labels for both schemes, remap the LLM predictions onto the same scheme, split 80/10/10 with assertions confirming no review appears in two splits, then tokenise with **dynamic padding**. Because the median review is 30 tokens, padding every batch to a fixed 128 would waste roughly three quarters of the compute. Dynamic padding pads only to the longest item in each batch.

**Step 12, the weighted loss.** A custom `WeightedLossTrainer` injects class-weighted cross-entropy, so rare classes are not ignored. A dry-run forward pass confirms the weighted loss really does differ from the unweighted one before any GPU time is spent.

**Steps 13 to 16, training and evaluation.** Task A for 3 epochs, Task B for 2, both with CodeCarbon tracking on training and inference, wrapped in `try/finally` so the tracker always closes cleanly.

**Step 17, the head to head.** DistilBERT against all seven LLMs on the identical test rows, plus Cohen's κ.

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Input text | Raw review column | The tokeniser handles casing and punctuation; cleaning destroys signal |
| Max length | 128 tokens | Covers 99.45% of reviews at roughly a quarter of the compute of 512 |
| Padding | Dynamic per batch | Median review is 30 tokens, so static padding wastes about 75% of compute |
| Class imbalance | Weighted cross-entropy | Without it, the model simply predicts "5" and "positive" |
| Checkpoint metric | Macro F1 | Accuracy and weighted F1 can be gamed by ignoring rare classes |
| Numeric precision | fp16 | About 40% faster, about 50% less memory, negligible quality cost |
| **Task B epochs** | **2, not 3** | Task A's validation macro F1 was 0.5698 at epoch 1 and 0.5716 at epoch 3, essentially flat. Cutting the third epoch is an **evidence-based carbon saving**, not a corner cut |
| Stratification | On the 5-class rating | Preserves both label schemes at once |
| Split | Same rows for both tasks | Required for a fair comparison |

---

## 6. Results

All numbers below are read from the stored outputs of the notebook in this repository.

### Task A, predicting the exact star rating

**Training:** 3 epochs, 15,060 steps, **3.23 minutes**, **7.973 g CO₂**. Final training loss 0.8879, best validation macro F1 0.5809.

| Metric | Value |
|---|---|
| Accuracy | **71.90%** |
| Weighted F1 | 0.7436 |
| **Macro F1** | **0.5664** |
| MAE (stars) | 0.3479 |
| Eval loss | 0.9983 |
| Throughput | 12,303 samples/sec |
| Latency | 0.08 ms per review |
| Inference CO₂ | 0.0829 g for the whole test set |

### Task B, predicting sentiment

**Training:** 2 epochs, 10,040 steps, **2.16 minutes**, **5.334 g CO₂**. Final training loss 0.5072, best validation macro F1 0.7390.

| Metric | Value |
|---|---|
| Accuracy | **88.35%** |
| Weighted F1 | 0.8981 |
| **Macro F1** | **0.7338** |
| Eval loss | 0.5557 |
| Majority-class baseline | 72.82% |
| **Uplift over doing nothing** | **+15.54 percentage points** |
| Throughput | 12,324 samples/sec |
| Inference CO₂ | 0.0820 g for the whole test set |

### Head to head, Task A (5-class)

| Model | Evaluated | Missing | Accuracy | Weighted F1 | Macro F1 | MAE |
|---|---|---|---|---|---|---|
| GPT-4 | 20,030 | 52 | **0.7494** | 0.7634 | 0.5676 | **0.3130** |
| LLaMA 3.3 70B | 19,995 | 87 | 0.7411 | 0.7524 | 0.5542 | 0.3309 |
| Gemini 1.5 Pro | 20,080 | 2 | 0.7386 | 0.7566 | **0.5791** | 0.3244 |
| **DistilBERT (67M)** | **20,082** | **0** | **0.7190** | **0.7436** | **0.5664** | **0.3479** |
| GPT-3.5 Instruct | 20,045 | 37 | 0.6910 | 0.7230 | 0.5479 | 0.3726 |
| GPT-3.5 Turbo | 20,064 | 18 | 0.6661 | 0.7027 | 0.5162 | 0.4000 |
| Gemini 1.5 Flash | 19,859 | 223 | 0.6080 | 0.6473 | 0.4876 | 0.5125 |
| LLaMA 3.1 8B | 19,043 | 1,039 | 0.6072 | 0.6037 | 0.3551 | 0.6204 |

DistilBERT is 4th on accuracy but **3rd on macro F1**, and comfortably beats both GPT-3.5 variants, Gemini 1.5 Flash, and LLaMA 3.1 8B on every measure.

### Head to head, Task B (3-class)

| Model | Evaluated | Missing | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|
| LLaMA 3.3 70B | 19,995 | 87 | **0.9073** | 0.9030 | 0.7065 |
| Gemini 1.5 Pro | 20,080 | 2 | 0.9064 | **0.9116** | **0.7473** |
| GPT-3.5 Instruct | 20,045 | 37 | 0.8982 | 0.9038 | 0.7299 |
| GPT-4 | 20,030 | 52 | 0.8961 | 0.9035 | 0.7302 |
| GPT-3.5 Turbo | 20,064 | 18 | 0.8942 | 0.8974 | 0.7099 |
| **DistilBERT (67M)** | **20,082** | **0** | **0.8835** | **0.8981** | **0.7338** |
| Gemini 1.5 Flash | 19,859 | 223 | 0.8629 | 0.8711 | 0.6646 |
| LLaMA 3.1 8B | 19,043 | 1,039 | 0.7832 | 0.7959 | 0.5690 |

### The result that matters most: the ranking flips

This is the most interesting finding in the project, and it is easy to miss if you only read the accuracy column.

**Ranked by accuracy, DistilBERT is 6th of 8. Ranked by macro F1, DistilBERT is 2nd of 8.**

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

**Same predictions. Same 20,082 reviews. Opposite conclusion.**

LLaMA 3.3 70B tops the accuracy table and finishes 6th on macro F1. DistilBERT does the reverse, climbing four places.

**Why it happens.** 72.8% of the test set is positive. A model that leans towards "positive" scores well on accuracy while quietly failing on the 5.5% neutral class, which accuracy barely registers. DistilBERT was trained with a weighted loss that explicitly told it to care about every class equally, and the ranking reflects that instruction.

**What to take from it.** "Best model" is not a property of a model. It is a property of a model **and the question you are asking of it**. If you care about the average review, read the accuracy column. If you care about every kind of review, including the unhappy minority, read macro F1. For a mental health application, the second one is the operationally relevant column.

### Per-class performance

**Task A, 5-class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1-star | 0.8660 | 0.7368 | **0.7962** | 3,457 |
| 2-star | 0.3149 | 0.4854 | **0.3820** | 892 |
| 3-star | 0.3077 | 0.4730 | **0.3729** | 1,110 |
| 4-star | 0.3580 | 0.5249 | 0.4257 | 2,509 |
| 5-star | 0.9264 | 0.7939 | **0.8550** | 12,114 |
| **Macro avg** | 0.5546 | 0.6028 | **0.5664** | 20,082 |
| Weighted avg | 0.7836 | 0.7190 | 0.7436 | 20,082 |

**Task B, 3-class**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.8837 | 0.8558 | **0.8695** | 4,349 |
| neutral | 0.2919 | 0.5712 | **0.3863** | 1,110 |
| positive | 0.9773 | 0.9155 | **0.9454** | 14,623 |
| **Macro avg** | 0.7176 | 0.7808 | **0.7338** | 20,082 |
| Weighted avg | 0.9191 | 0.8835 | 0.8981 | 20,082 |

### Confusion matrices

**How to read a confusion matrix.** Each row is the true answer, each column is what the model guessed. The diagonal is correct answers. Everything off the diagonal is a mistake, and where those mistakes land tells you what the model is confusing.

**Task A, percentage correct per true class:** 1-star **73.7%**, 2-star **48.5%**, 3-star **47.3%**, 4-star **52.5%**, 5-star **79.4%**.

The errors sit almost entirely next to the diagonal. True 4-star reviews are called 5-star 26.0% of the time and 3-star 17.8% of the time. True 2-star reviews go to 1-star 21.9% and 3-star 25.2%. The model is not confusing delight with fury. It is struggling to place reviews precisely on a scale where humans are also imprecise.

**Task B, percentage correct per true class:** negative **85.6%**, neutral **57.1%**, positive **91.5%**.

The single largest error cell is 1,015 truly positive reviews labelled neutral. That is the weighted loss doing exactly what it was told: reach hard for the rare neutral class, and accept some false positives as the price.

### Do the models agree with each other?

Cohen's κ measures agreement after stripping out the agreement you would get from luck.

**Agreement with the actual user-given rating, 5-class:**

| Rank | Model | κ vs ground truth |
|---|---|---|
| 1 | GPT-4 | 0.587 |
| 2 | Gemini 1.5 Pro | 0.584 |
| 3 | LLaMA 3.3 70B | 0.581 |
| 4 | **DistilBERT (67M)** | **0.556** |
| 5 | GPT-3.5 Instruct | 0.522 |
| 6 | GPT-3.5 Turbo | 0.486 |
| 7 | Gemini 1.5 Flash | 0.438 |
| 8 | LLaMA 3.1 8B | 0.353 |

**The interesting part is the pairwise grid, not the ranking.** LLaMA 3.3 70B and Gemini 1.5 Pro agree with **each other** at κ = 0.803, and the top LLMs cluster at 0.69 to 0.80 among themselves. Yet none of them agrees with the **ground truth** above 0.587.

In plain English: **the large models are more similar to each other than any of them is to the humans they are supposed to be imitating.** That points to a shared systematic bias in how LLMs interpret star ratings, most likely inherited from overlapping training data and similar instruction tuning. It is an argument against treating "several LLMs agreed" as evidence of correctness.

DistilBERT sits slightly apart from that cluster (κ 0.63 to 0.70 against the LLMs), which is what you would expect from a model trained directly on the ground truth rather than prompted to imitate it.

---

## 7. The carbon analysis

### What was actually measured

DistilBERT's emissions were **measured** with CodeCarbon, which tracks CPU, GPU, and RAM energy draw and converts it using the local grid's carbon intensity.

| Phase | Task | Duration | CO₂ |
|---|---|---|---|
| Training | Task A (5-class, 3 epochs) | 3.23 min | **7.973 g** |
| Training | Task B (3-class, 2 epochs) | 2.16 min | **5.334 g** |
| Inference | Task A test set (20,082 reviews) | 1.63 s | 0.0829 g |
| Inference | Task B test set (20,082 reviews) | 1.63 s | 0.0820 g |
| **Total** | Both models, end to end | **~5.4 min** | **~13.5 g** |

**Training both models cost about 13.3 g of CO₂, roughly the emissions of boiling a small cup of water.** That is a one-off cost. Every prediction afterwards is close to free.

> **A note on the per-prediction figure.** All carbon numbers below use **0.0041 mg CO₂ per prediction**, measured by CodeCarbon in the run stored in this notebook (0.00413 mg on Task A, 0.00408 mg on Task B). An earlier training run measured 0.0030 mg on different hardware. Emissions depend on the GPU you get and the carbon intensity of your local grid, so expect your own figure to differ. The model quality numbers are unaffected, since they do not depend on hardware.

### The comparison, and what kind of number it is

**Please read this before quoting any multiplier.** The seven LLM baselines were run by the MHARD authors, and **their emissions were never measured**. Frontier providers do not publish per-token energy figures, so an exact measurement is not obtainable from the public side.

The per-prediction LLM figures below are therefore **estimates**, derived from published energy-per-query figures and parameter-count scaling rather than from instrumentation. They should be read as **order-of-magnitude indicators**, and the source of the estimates must be cited wherever these figures are reused.

> ⚠️ **Before publishing or submitting:** replace this paragraph with the explicit citation for the per-query energy estimates used in this section. Parameter counts marked with an asterisk are widely-cited estimates, not officially disclosed figures.

### Per-prediction carbon

| Model | mg CO₂ per prediction | Relative to DistilBERT |
|---|---|---|
| GPT-4 (1.76T*) | 3,300 (estimated) | ~799,000× |
| Gemini 1.5 Pro (175B*) | 1,500 (estimated) | ~363,000× |
| LLaMA 3.3 70B (70B) | 980 (estimated) | ~237,000× |
| GPT-3.5 Instruct (175B) | 820 (estimated) | ~199,000× |
| GPT-3.5 Turbo (20B*) | 220 (estimated) | ~53,000× |
| LLaMA 3.1 8B (8B) | 110 (estimated) | ~27,000× |
| Gemini 1.5 Flash (8B*) | 90 (estimated) | ~22,000× |
| **DistilBERT (67M)** | **0.0041 (measured)** | **1×** |

### The performance versus carbon trade-off

Put quality and carbon side by side and the picture is stark. **What you want is high quality at low carbon.**

| Model | Macro F1 (3-class) | mg CO₂ per prediction |
|---|---|---|
| Gemini 1.5 Pro | 0.7473 | 1,500 |
| **DistilBERT (67M)** | **0.7338** | **0.0041** |
| GPT-4 | 0.7302 | 3,300 |
| GPT-3.5 Instruct | 0.7299 | 820 |
| GPT-3.5 Turbo | 0.7099 | 220 |
| LLaMA 3.3 70B | 0.7065 | 980 |
| Gemini 1.5 Flash | 0.6646 | 90 |
| LLaMA 3.1 8B | 0.5690 | 110 |

Read the two columns together. The quality column spans a narrow range, from 0.57 to 0.75. The carbon column spans **six orders of magnitude**. DistilBERT sits near the top of the first column and at the very bottom of the second, which is the only place in this table you would actually want to be.

### Total carbon to process the whole dataset

This is the number that makes the abstract comparison concrete.

| Model | To classify the test set (20,082) | To classify the full dataset (200,812) |
|---|---|---|
| GPT-4 | 66,270.60 g | **662.68 kg** |
| Gemini 1.5 Pro | 30,123.00 g | 301.22 kg |
| LLaMA 3.3 70B | 19,680.36 g | 196.80 kg |
| GPT-3.5 Instruct | 16,467.24 g | 164.67 kg |
| GPT-3.5 Turbo | 4,418.04 g | 44.18 kg |
| LLaMA 3.1 8B | 2,209.02 g | 22.09 kg |
| Gemini 1.5 Flash | 1,807.38 g | 18.07 kg |
| **DistilBERT** | **0.083 g** | **0.83 g** |

Classifying all 200,812 reviews costs DistilBERT about **0.83 grams**, roughly a sixteenth of what it cost to train it. The estimate for GPT-4 on the same job is around **660 kilograms**, close to a million times more.

### Quality per unit of carbon

Divide macro F1 by carbon and you get a single efficiency number: how much quality you get per unit of emissions. DistilBERT is normalised to 1.0, so every other figure reads as a fraction of it.

| Model | Efficiency, Task A | Efficiency, Task B |
|---|---|---|
| **DistilBERT (67M)** | **1.00×** | **1.00×** |
| Gemini 1.5 Flash | 0.0000395× | 0.0000411× |
| LLaMA 3.1 8B | 0.0000235× | 0.0000288× |
| GPT-3.5 Turbo | 0.0000171× | 0.0000180× |
| GPT-3.5 Instruct | 0.00000487× | 0.00000495× |
| LLaMA 3.3 70B | 0.00000412× | 0.00000401× |
| Gemini 1.5 Pro | 0.00000281× | 0.00000277× |
| GPT-4 | 0.00000125× | 0.00000123× |

Read plainly: **on this task, the frontier models deliver somewhere between one twenty-five-thousandth and one eight-hundred-thousandth of the quality-per-gram that the small model does.** The two tasks give almost identical figures, which is a good sign that the pattern is not a quirk of one experiment.

### The argument that is arguably stronger than carbon

**Privacy.** Mental health app reviews are sensitive user text. A locally-run classifier means that text never leaves your infrastructure. No API contract, no data-processing agreement, no third-party retention policy to reason about, no cross-border transfer question. For anything touching health data, that often matters more to a compliance team than the carbon does.

---

## 8. Where it fails

Being specific about failure is more useful than a headline number.

### The middle of the scale

On Task B, neutral scores F1 **0.3863** against 0.8695 for negative and 0.9454 for positive. Look closer at the shape: precision **0.2919** with recall **0.5712**. That means the model **finds** most neutral reviews but is **wrong about roughly seven in ten** of the ones it labels neutral.

That is the weighted loss behaving as instructed. It was told the rare class matters, so it over-predicts neutral to catch it, and pays in precision. Whether that trade is acceptable depends on the use case. If you are triaging reviews for a human to read, high recall is what you want. If you are reporting statistics, it is not.

The same shape appears on Task A. 2-star (F1 0.3820) and 3-star (0.3729) are the weakest classes, while 1-star and 5-star both clear 0.79. **The extremes are easy, the middle is genuinely ambiguous.** A review saying "it has helped a little but nothing groundbreaking" sits somewhere between 3 and 4 stars for a human annotator too.

### A sanity check on 50 random reviews

On 50 randomly selected test reviews (seed 42): **36 correct (72.0%), 9 off by one (18.0%), 45 within ±1 (90.0%)**.

This matters because it shows the errors are **near-misses on a scale**, not category collapses. The model is not calling furious reviews delighted. It is disagreeing by one star, which is the same thing two human annotators would do.

### The operational failure the LLMs have and DistilBERT does not

DistilBERT returned a prediction for **all 20,082 rows**. The API models did not:

- LLaMA 3.1 8B failed on **1,039 rows (5.2%)**
- Gemini 1.5 Flash failed on **223**
- GPT-4 failed on **52**
- Gemini 1.5 Pro failed on **2**

The MHARD authors document the cause: the models returned bracketed numbers, long explanations, or error messages instead of a parseable rating.

**Why this matters more than it sounds.** In a production pipeline every one of those rows needs a retry, a fallback, or a human. And the failures are **not random**. They cluster on exactly the inputs the model found hardest to categorise, which is to say the ones you most wanted an answer for.

A local classification head has no such failure mode. It always returns a probability distribution over the label set. It can be wrong, but it cannot refuse to answer or reply in a format you cannot parse.

---

## 9. Limitations, stated honestly

Please read this section before citing any number above.

- **The neutral class is weak.** F1 0.3863, precision 0.2919. If your application depends on identifying ambivalent users specifically, this model is not adequate as it stands. Threshold tuning, focal loss, or a dedicated ordinal-regression head are the obvious next steps.
- **Single run, single seed.** Seed 42 throughout, so the run is reproducible. Reproducible is not the same as representative. A mean and standard deviation across several seeds would be a much stronger claim, and is the highest-value next experiment.
- **LLM emissions are estimated, not measured.** See the warning in section 7. The carbon comparison is directional. The exact multipliers should not be quoted as measurements.
- **Carbon figures are hardware-specific.** An earlier run of this same pipeline measured 0.0030 mg per prediction against the 0.0041 mg reported here. Nothing changed in the model, only the machine it ran on. Treat any single carbon figure as a measurement of one run on one GPU, not a property of the model.
- **The LLM predictions are pre-computed** by the MHARD authors. Their prompting strategy, temperature, and output parsing are fixed and not controlled by this work. Different prompts could produce different LLM results, and a fairer comparison would re-run them under controlled prompting.
- **One domain, one language.** Mental health app reviews from Google Play, English only, 2011 to 2023. Transfer to other review domains or languages is untested.
- **The ground truth is itself noisy.** The "correct" answer is the star rating the user selected, which is a rough proxy for sentiment. Users routinely write glowing text and leave 3 stars, or the reverse. Part of the residual error on **every** model here is irreducible.
- **Hardware variance.** The logged wall times (3.23 and 2.16 minutes for 160k training samples) are considerably faster than a free-tier T4 would deliver. Emissions scale with the hardware actually used, so reproduce on your own setup before quoting the figures.
- **Carbon intensity is regional.** CodeCarbon converts energy to CO₂ using the local grid's mix. The same run in a coal-heavy region emits several times more than in a hydro-heavy one.

---

## 10. How to reproduce

### What you need

- A Google Colab account with a GPU runtime
- At least 2 GB free on Google Drive
- The MHARD dataset CSV from [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD)

### Steps

1. **Upload the dataset** to Drive at `My Drive/Colab Notebooks/DistilBERT/MHARD_dataset.csv`
2. **Open the notebook** in Colab
3. **Enable the GPU:** `Runtime → Change runtime type → GPU`
4. **Run all cells in order.** Steps 13 and 15 are the two training runs. Together they take about six minutes on a decent GPU.
5. **Collect the outputs from Drive:**

```
My Drive/Colab Notebooks/DistilBERT/
├── distilbert_5class_final/          # 255 MB model + tokenizer + metrics
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
    └── test_set_predictions_all_models.csv
```

**That last file is the one to open if you want to check any number in this README yourself.** It contains every model's prediction on every test row.

### Changing the dataset path

Edit this line in Step 3:

```python
CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/DistilBERT/MHARD_dataset.csv"
```

---

## 11. Repository structure and dependencies

```
.
├── README.md
├── LICENSE
├── Carbon_Aware_Sentiment_Analysis_at_Scale.ipynb   # full pipeline, 17 steps, outputs included
├── requirements.txt
└── emissions/                                       # CodeCarbon CSV logs
```

Trained weights are **not included** (255 MB per model). They are written to Drive during training.

### Dependencies

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

**Version note.** The notebook uses `processing_class=tokenizer` in `Trainer`, which is the updated argument name from `transformers>=4.46`. On older versions, replace it with `tokenizer=` in Steps 13 and 15.

---

## 12. Citations

This project stands on three pieces of other people's work: the MHARD dataset and its LLM prediction runs, the DistilBERT model, and the Hugging Face and CodeCarbon tooling. Please cite them.

### Dataset

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

```bibtex
@article{sanh2019distilbert,
  title   = {DistilBERT, a distilled version of BERT: smaller, faster,
             cheaper and lighter},
  author  = {Sanh, Victor and Debut, Lysandre and Chaumond, Julien and
             Wolf, Thomas},
  journal = {arXiv preprint arXiv:1910.01108},
  year    = {2019}
}

@inproceedings{devlin2019bert,
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers
               for Language Understanding},
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and
               Toutanova, Kristina},
  booktitle = {Proceedings of NAACL-HLT 2019},
  pages     = {4171--4186},
  year      = {2019}
}

@inproceedings{wolf2020transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and
               Chaumond, Julien and Delangue, Clement and Moi, Anthony and
               Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and
               Funtowicz, Morgan and others},
  booktitle = {Proceedings of EMNLP 2020: System Demonstrations},
  pages     = {38--45},
  year      = {2020}
}

@software{codecarbon,
  title  = {CodeCarbon: Estimate and Track Carbon Emissions from
            Machine Learning Computing},
  author = {Courty, Beno{\^i}t and Schmidt, Victor and
            Luccioni, Sasha and others},
  url    = {https://github.com/mlco2/codecarbon}
}

@article{pedregosa2011scikit,
  title   = {Scikit-learn: Machine Learning in {P}ython},
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and
             Michel, V. and Thirion, B. and Grisel, O. and others},
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

## 13. Author

**Collins Lemeke.** Full pipeline design, implementation, training, evaluation, and analysis.

MSc Artificial Intelligence (Distinction), University of Greater Manchester. Centre of Intelligence of Things (CIoTh). Co-founder, AI Nexus Society. IEEE Member.

This sits inside a wider research programme on reading internal state from observable signals, across facial expression, physiological sensing, gait, and language:

- [Facial Expression Recognition with CNN](https://github.com/CollinsLemeke/Facial-Expression-Recognition-Model), imbalance-aware evaluation on FER2013
- [Autism Facial Emotion Classification](https://github.com/CollinsLemeke/Autism-Facial-Emotion-Classification), VGG16 transfer learning, published at IEEE IC3ECSBHI 2026
- [Detecting Cognitive Decline, Falls and Frailty](https://github.com/CollinsLemeke/Detecting-Cognitive-Decline-Falls-and-Frailty), interpretable screening from gait sensor data

The thread running through all of them is the one running through this project: **a headline number is not a result until you know which class it is hiding.**

- [GitHub](https://github.com/CollinsLemeke)
- [Kaggle](https://www.kaggle.com/collinslemeke/code)

Questions and corrections are welcome. Please open an issue.

---

## 14. Licence

### Code

Released under the **MIT Licence**. You may use, copy, modify, and distribute it, including commercially, provided the copyright notice and licence text are retained. It is provided without warranty. Full text in [LICENSE](LICENSE).

### Dataset

**MHARD** is released by its authors under the MIT Licence and is **not redistributed in this repository**. Obtain it from [github.com/Sensify-Lab/MHARD](https://github.com/Sensify-Lab/MHARD) and cite Wang et al. (2025).

### Model

`distilbert-base-uncased` is released under **Apache 2.0** by Hugging Face. Fine-tuned weights derived from it inherit that licence.

---

> **67 million parameters. Five and a half minutes. Thirteen grams of CO₂. Second place on macro F1, ahead of GPT-4.**
>
> The point is not that small beats large. The point is that nobody checked, because accuracy said otherwise.
