# Reducing AI Carbon Footprint: A Study of DistilBERT for Mental Health Sentiment Analysis

> **Can a 66M-parameter fine-tuned model match frontier LLMs on mental health app review classification — at a fraction of the carbon cost?**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers)
[![CodeCarbon](https://img.shields.io/badge/🌱-CodeCarbon-green)](https://codecarbon.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MHARD%20200k%2B-blueviolet)](https://huggingface.co/datasets)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00)](https://colab.research.google.com/)

---

## Table of Contents

- [Overview](#overview)
- [Motivation and Research Question](#motivation-and-research-question)
- [Dataset: MHARD](#dataset-mhard)
- [Experimental Design](#experimental-design)
- [Project Architecture and Pipeline](#project-architecture-and-pipeline)
  - [Step 1 – Environment Setup](#step-1--environment-setup)
  - [Step 2 – Imports, Seeds, and Device Check](#step-2--imports-seeds-and-device-check)
  - [Step 3 – Load the MHARD Dataset](#step-3--load-the-mhard-dataset)
  - [Step 4 – Missing Value Analysis](#step-4--missing-value-analysis)
  - [Step 5 – Rating Distribution and Class Imbalance](#step-5--rating-distribution-and-class-imbalance)
  - [Step 6 – Review Length Analysis and max_length Decision](#step-6--review-length-analysis-and-max_length-decision)
  - [Step 7 – Data Cleaning Implementation](#step-7--data-cleaning-implementation)
  - [Step 8 – Word Clouds and LLM Agreement Preview](#step-8--word-clouds-and-llm-agreement-preview)
  - [Step 9 – Label Encoding for Both Tasks](#step-9--label-encoding-for-both-tasks)
  - [Step 10 – Stratified 80/10/10 Split](#step-10--stratified-801010-split)
  - [Step 11 – Tokenization with DistilBERT](#step-11--tokenization-with-distilbert)
  - [Step 12 – Model Architecture, Weighted Loss, and Metrics](#step-12--model-architecture-weighted-loss-and-metrics)
  - [Step 13 – Task A Training (5-class, 3 Epochs)](#step-13--task-a-training-5-class-3-epochs)
  - [Step 14 – Task A Test Evaluation](#step-14--task-a-test-evaluation)
  - [Step 15 – Task B Training (3-class, 2 Epochs)](#step-15--task-b-training-3-class-2-epochs)
  - [Step 16 – Task B Test Evaluation](#step-16--task-b-test-evaluation)
  - [Step 17 – LLM Baseline Comparison](#step-17--llm-baseline-comparison)
- [Key Design Decisions and Justifications](#key-design-decisions-and-justifications)
- [Results and Findings](#results-and-findings)
- [Carbon Efficiency Analysis](#carbon-efficiency-analysis)
- [How to Reproduce](#how-to-reproduce)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [Author](#author)

---

## Overview

This project fine-tunes **DistilBERT** (`distilbert-base-uncased`, 66M parameters) for automated sentiment analysis of mental health app reviews, and benchmarks it against **seven frontier Large Language Models (LLMs)** whose predictions come bundled in the MHARD dataset. The models benchmarked are: GPT-3.5 Instruct, GPT-3.5 Turbo, GPT-4, Gemini 1.5 Flash, Gemini 1.5 Pro, LLaMA 3.1 8B, and LLaMA 3.3 70B.

The work trains two classification heads from the same `distilbert-base-uncased` encoder:

- **Task A** — 5-class ordinal rating prediction (predict the exact 1–5 star rating)
- **Task B** — 3-class sentiment classification (negative / neutral / positive)

Carbon emissions during both training and inference are measured using **CodeCarbon**, enabling a concrete, quantitative argument about the environmental cost-efficiency of small fine-tuned models versus large general-purpose LLMs.

This notebook constitutes the empirical core of the MSc dissertation *"Reducing AI Carbon Footprint: A Study of DistilBERT for Mental Health Sentiment Analysis"* and extends the **NetZeroNLP** research narrative to a real-world application domain.

---

## Motivation and Research Question

The AI industry is consuming more energy than ever. Training and running frontier LLMs like GPT-4 requires significant compute and therefore significant carbon emissions. Yet many real-world NLP tasks — such as classifying the sentiment of user reviews — are well-defined, structured problems where a smaller, purpose-built model may be just as effective.

**The central research question is:**

> *Can a fine-tuned DistilBERT (66M parameters) match or closely approach the predictive performance of frontier LLMs (ranging from 8B to 100B+ parameters) on mental health app review classification, while emitting orders of magnitude less CO₂?*

This is not merely an academic comparison. Mental health app developers, researchers, and regulators increasingly need to understand how users feel about these tools at scale. Automating this with an energy-efficient model rather than paying for GPT-4 API calls per prediction has both economic and environmental implications.

---

## Dataset: MHARD

**Name:** MHARD — Mental Health App Reviews Dataset  
**Source:** Wang et al., ICWSM 2025  
**Size:** 200,972 user reviews  
**Coverage:** 73 mental health applications on the Google Play Store  
**Time Range:** Reviews scraped between 2011 and 2023  

Each row in the dataset contains:

| Column | Description |
|--------|-------------|
| `UID` | Unique review identifier |
| `app_name` | Name of the mental health app |
| `date` | Date the review was posted |
| `review` | Raw review text (primary input) |
| `review_cleaned` | Pre-tokenised, stopword-removed version (used for EDA only) |
| `rating` | Ground truth star rating (1–5), given by the app user |
| `likes` | Number of helpful votes the review received |
| `response` | Developer's response to the review (~74% missing) |
| `pred_gpt3.5instruct` | GPT-3.5 Instruct predicted rating |
| `pred_gpt3.5turbo` | GPT-3.5 Turbo predicted rating |
| `pred_gpt4` | GPT-4 predicted rating |
| `pred_gemini1.5flash` | Gemini 1.5 Flash predicted rating |
| `pred_gemini1.5pro` | Gemini 1.5 Pro predicted rating |
| `pred_llama3.1_8b` | LLaMA 3.1 8B predicted rating |
| `pred_llama3.3_70b` | LLaMA 3.3 70B predicted rating |

### Class Distribution

The dataset exhibits a strong positive skew — a structural feature common to app review data where users are more motivated to write reviews when satisfied:

| Rating | Count | Percentage |
|--------|-------|------------|
| 1-star | ~24,000 | ~12% |
| 2-star | ~11,000 | ~5% |
| 3-star | ~18,000 | ~9% |
| 4-star | ~32,000 | ~16% |
| 5-star | ~116,000 | ~58% |

Collapsed into 3 classes: **~17% negative, ~9% neutral, ~73% positive**. This means a majority-class baseline (always predict "positive") would achieve ~73% accuracy on Task B — setting a meaningful minimum bar that any model must comfortably exceed.

---

## Experimental Design

The experiment is structured around two parallel fine-tuning tasks, each evaluated on the same held-out test set and compared against LLM baselines on the same rows.

**Task A — 5-class ordinal classification:** Predict the exact star rating (1–5). This is the harder, more granular problem. Mean Absolute Error (MAE) is meaningful here because a prediction of 4 for a true 5-star review is a smaller mistake than predicting 1.

**Task B — 3-class sentiment classification:** Collapse ratings into Negative (1–2), Neutral (3), and Positive (4–5). This is the standard simplified view used in review analytics.

Both tasks use the **same** stratified train/validation/test split, the **same** tokenisation, and the **same** training hyperparameters (except for the number of epochs). This ensures the two task metrics are directly comparable and that LLM baselines are evaluated on exactly the same test rows.

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Standard overall correctness |
| Weighted F1 | Class-size-weighted F1 — useful summary for imbalanced data |
| Macro F1 | Unweighted mean F1 across all classes — this is the **primary metric** because it treats minority classes equally and cannot be gamed by ignoring them |
| MAE (stars) | Ordinal error magnitude — meaningful only for Task A |
| Cohen's κ | Agreement between DistilBERT predictions and each LLM |
| CO₂eq (kg) | Carbon emissions for both training and inference phases |
| Inference latency (ms/sample) | Operational efficiency |
| Throughput (samples/sec) | Scalability indicator |

---

## Project Architecture and Pipeline

The notebook is structured into 17 sequential steps, each building on the last. Below is a detailed walkthrough of every step — what it does, why, and what it produces.

---

### Step 1 – Environment Setup

**What it does:** Mounts Google Drive for persistent storage and installs all required packages.

```bash
pip install transformers datasets evaluate accelerate
pip install scikit-learn wordcloud codecarbon
```

Google Drive is used throughout to persist model checkpoints, emissions logs, and evaluation artifacts. Training directly to Drive would be slow due to sync latency, so a local-to-Drive pattern is used: train locally on Colab's SSD, then copy final artifacts to Drive at the end.

---

### Step 2 – Imports, Seeds, and Device Check

**What it does:** Imports all libraries, fixes every source of randomness to seed `42`, and detects the available hardware.

Reproducibility is enforced at every level — Python's `random`, NumPy, PyTorch, CUDA, and HuggingFace's `set_seed()` are all fixed. This ensures that a reviewer or collaborator can re-run the notebook and obtain identical splits, identical initialisation weights, and identical metric values.

The device check confirms whether a GPU is available. Training DistilBERT on 200k+ rows on CPU is impractical; a T4 GPU (free-tier Colab) is the minimum recommended hardware.

---

### Step 3 – Load the MHARD Dataset

**What it does:** Reads the MHARD CSV into a pandas DataFrame and performs an initial structural inspection.

Key outputs:
- Dataset shape: **200,972 rows × 17 columns**
- Memory footprint reported (approximately 150–200 MB in pandas)
- Column-by-column dtype and non-null count summary
- Side-by-side preview of raw vs. cleaned review text for the same rows — this makes the downstream tokenisation choice (use raw, not pre-cleaned) concrete and visible

---

### Step 4 – Missing Value Analysis

**What it does:** Analyses and visualises missing values across all columns, then categorises each column by its role in the pipeline.

Key findings:
- `review` and `rating`: near-complete (only ~21 null reviews — less than 0.01%)
- `response` and `response_date`: ~74% missing — dropped from the pipeline
- LLM prediction columns: varying degrees of missingness, handled pairwise at evaluation time

A horizontal bar chart visualises missingness percentage per column, and columns are classified into four roles: Training, LLM Baseline Evaluation, Metadata Only, and Optional.

---

### Step 5 – Rating Distribution and Class Imbalance

**What it does:** Quantifies and visualises the class imbalance for both the 5-class and 3-class label schemes, then computes class weights for the weighted loss function.

**Class imbalance ratio (5-class):** The majority class (5-star) is approximately 10–14× larger than the minority class (2-star). If left unaddressed, the model would learn to predict "5-star" by default.

**Solution — balanced inverse-frequency class weights:** Using `sklearn.utils.class_weight.compute_class_weight`, weights are computed such that the loss function penalises misclassifications on rare classes proportionally more. These weights are passed to a custom `WeightedLossTrainer` (see Step 12).

Visualisations include dual bar/pie charts for both 5-class and 3-class distributions, and a stacked horizontal bar chart showing rating distribution per app (top 15 apps by review count) — confirming the positive skew is a dataset-wide structural pattern, not driven by one outlier app.

---

### Step 6 – Review Length Analysis and max_length Decision

**What it does:** Analyses the distribution of review lengths in words and characters, then makes a principled, emissions-aware choice of `max_length` for the DistilBERT tokeniser.

**The problem:** DistilBERT supports up to 512 tokens, but compute cost scales with sequence length. Padding every review to 512 tokens when the median review is ~30 words is massively wasteful.

**The methodology:**
1. Compute word count statistics across all reviews
2. Apply the rule-of-thumb conversion: 1 English word ≈ 1.3 WordPiece tokens
3. Identify the candidate `max_length` that covers 95%+ of reviews without truncation
4. **Empirically verify** by actually tokenising a 2,000-review sample and measuring real token lengths

**Result:** `MAX_LENGTH = 128` was selected. It covers **99.45% of reviews without truncation**, and the median tokenised review is only **30 tokens** — meaning most reviews are padded from 30 to (at most) 128, a fraction of the 512-token maximum. This directly reduces training time and carbon emissions while sacrificing only 0.55% of reviews to truncation.

---

### Step 7 – Data Cleaning Implementation

**What it does:** Applies a minimal, targeted text cleaning pipeline and audits each step with exact row counts.

**Philosophy:** Do as little as possible. DistilBERT's WordPiece tokeniser already handles casing, punctuation, contractions, and subword decomposition. Classical NLP cleaning (stemming, stopword removal, lowercasing) would **destroy signal** that the pretrained model was specifically trained to interpret.

The normalisation function applies only four operations:
1. Strip leading/trailing whitespace
2. Replace URLs with a space (URL tokens are long junk sequences with no review sentiment)
3. Collapse character repetitions of 3+ to 2 (e.g., `"sooooo"` → `"soo"`)
4. Collapse runs of whitespace to a single space

**Cleaning audit results:**

| Step | Action | Rows Removed |
|------|--------|-------------|
| Initial | — | 200,972 |
| Drop null reviews | Remove 21 rows | ~21 |
| Drop empty-after-normalisation | Remove whitespace-only rows | ~0 |
| Drop reviews < 3 words | Too short to carry signal | ~100–200 |
| **Final** | — | **~200,750** |

A before/after distribution plot confirms the rating proportions are unchanged after cleaning (delta < 0.02% per class).

---

### Step 8 – Word Clouds and LLM Agreement Preview

**What it does:** Performs two key EDA analyses — qualitative text patterns per rating, and a quantitative preview of LLM baseline accuracy.

**Word clouds per rating:** Built from the pre-cleaned column (stopwords already removed), using colour-maps that match sentiment (red for 1-star, green for 5-star). These reveal the vocabulary that most distinctively characterises each star rating.

**Log-odds distinctiveness analysis:** Rather than raw frequency (dominated by "app" across all classes), a log-odds metric identifies words that appear disproportionately in each class vs. all others. This surfaces genuinely discriminative vocabulary.

**LLM Baseline Preview (full cleaned dataset):**

This is the most important EDA cell for the thesis narrative. For each LLM, exact-match accuracy and MAE are computed across all available predictions:

| LLM | Accuracy | MAE |
|-----|----------|-----|
| GPT-4 | **0.752** | **0.307** |
| Gemini 1.5 Pro | ~0.748 | ~0.315 |
| LLaMA 3.3 70B | ~0.742 | ~0.320 |
| GPT-3.5 Turbo | ~0.680 | ~0.390 |
| GPT-3.5 Instruct | ~0.660 | ~0.410 |
| Gemini 1.5 Flash | ~0.612 | ~0.450 |
| LLaMA 3.1 8B | ~0.610 | ~0.460 |

*Note: Exact figures are computed at runtime from the MHARD dataset.*

The frontier LLMs (GPT-4, Gemini 1.5 Pro, LLaMA 3.3 70B) cluster at ~74–75% accuracy. This is the performance bar DistilBERT needs to approach or match.

A pairwise agreement heatmap between LLMs reveals that the top-tier models agree with each other more than they agree with ground truth — suggesting shared systematic biases in how LLMs interpret star ratings.

---

### Step 9 – Label Encoding for Both Tasks

**What it does:** Creates the integer label columns expected by PyTorch, and remaps LLM predictions to the same schema for fair comparison.

**Task A (5-class):** Ratings 1–5 are shifted to 0–4 (PyTorch classification heads expect 0-indexed labels).

```
rating 1 → label 0 (1-star)
rating 2 → label 1 (2-star)
rating 3 → label 2 (3-star)
rating 4 → label 3 (4-star)
rating 5 → label 4 (5-star)
```

**Task B (3-class):**
```
rating 1, 2 → label 0 (negative)
rating 3    → label 1 (neutral)
rating 4, 5 → label 2 (positive)
```

All seven LLM prediction columns are remapped to the 3-class schema in new columns (`pred_*_3class`), with NaN values preserved for pairwise evaluation. A programmatic consistency assertion confirms that `label_3class` is always the correct collapse of `label_5class`, catching any label assignment bugs before training.

---

### Step 10 – Stratified 80/10/10 Split

**What it does:** Splits the cleaned dataset into train (80%), validation (10%), and test (10%) with stratification, and verifies split integrity.

**Critical design decision:** Stratification is performed on the **5-class rating** column, not 3-class. Stratifying on the finer grid automatically preserves the 3-class distribution too (since 3-class is a deterministic collapse), but not vice versa. This single split is used for both tasks — using different splits would make Task A and Task B test metrics incomparable.

**Split sizes (~200,750 rows):**
| Split | Size | Percentage |
|-------|------|-----------|
| Train | ~160,600 | 80% |
| Validation | ~20,075 | 10% |
| Test | ~20,075 | 10% |

**Integrity checks performed:**
- No UID overlap between any two splits (train/val, train/test, val/test)
- All 5 rating classes present in every split
- All 3 sentiment classes present in every split
- Split sizes sum to total cleaned row count

A `test_llm_preds` DataFrame is extracted immediately, containing the LLM predictions for test-set rows only. This is the slice used in Step 17 for a fair head-to-head comparison.

---

### Step 11 – Tokenization with DistilBERT

**What it does:** Converts the three pandas splits to HuggingFace `DatasetDict` objects, tokenises all reviews with the DistilBERT tokeniser, and creates task-specific dataset views.

**Why HuggingFace Datasets?** Apache Arrow (the underlying format) is memory-mapped, which keeps RAM usage manageable for 200k+ rows. Batched `.map()` tokenisation is ~10× faster than row-wise pandas apply, and the Trainer API expects Dataset objects directly.

**Tokenisation parameters:**
- `tokenizer`: `DistilBertTokenizerFast` from `distilbert-base-uncased`
- `truncation=True` (handles the 0.55% of reviews > 128 tokens)
- `padding=False` (dynamic padding at batch time — see below)
- `max_length=128`
- `batch_size=1000` for the map operation

**Dynamic padding:** Instead of padding every review to 128 tokens statically, `DataCollatorWithPadding` pads each training batch to the length of its longest sample. Since the median review is only 30 tokens, this eliminates ~75% of wasted compute on padding tokens — a direct carbon saving with no quality trade-off.

After tokenisation, two task-specific views are created by renaming the appropriate label column to `"label"` (required by HuggingFace Trainer) and dropping the unused one. Both views use the **identical encoded text** — only the label column differs.

---

### Step 12 – Model Architecture, Weighted Loss, and Metrics

**What it does:** Defines the three shared components used by both task trainers: the model factory, the custom weighted-loss trainer, and the compute-metrics functions.

**Model factory (`build_distilbert`):** A function that loads `DistilBertForSequenceClassification` with the correct `num_labels`, `id2label`, and `label2id` for each task, and moves the model to the target device. Each task gets its own freshly initialised model — there is no weight sharing between Task A and Task B.

**`WeightedLossTrainer`:** A subclass of HuggingFace's `Trainer` that overrides `compute_loss()` to inject class-weighted `CrossEntropyLoss`. The class weights (computed in Step 5) are passed as a `torch.Tensor` at construction time and automatically moved to the model's device. This is the idiomatic HuggingFace way to handle class imbalance — monkey-patching or callbacks cannot reach the loss function.

**Metrics functions:**
- `compute_metrics_5class`: returns accuracy, weighted F1, macro F1, and MAE in stars
- `compute_metrics_3class`: returns accuracy, weighted F1, and macro F1

Macro F1 is used as the `metric_for_best_model` in TrainingArguments. This is intentional — macro F1 treats minority classes equally and cannot be artificially inflated by a model that ignores neutral or 2-star reviews. A dry-run forward pass verifies that the weighted loss actually differs from unweighted loss before any GPU time is committed.

---

### Step 13 – Task A Training (5-class, 3 Epochs)

**What it does:** Configures and launches fine-tuning of the 5-class DistilBERT head with full CodeCarbon emissions tracking.

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 32 | Standard for DistilBERT on T4; fits fp16 with headroom |
| Epochs | 3 | Standard for fine-tuning a pretrained model on large datasets |
| Learning rate | 2e-5 | Canonical DistilBERT fine-tune LR; stable under class-weighted loss |
| Warmup ratio | 0.1 | 10% warmup ramps LR from 0 to peak — improves early training stability |
| Weight decay | 0.01 | Light L2 regularisation (AdamW default) |
| Precision | fp16 | ~40% speed improvement and ~50% memory reduction vs fp32, with negligible accuracy cost |
| Eval strategy | epoch | One eval per epoch — enough granularity without wasting compute |
| Best-model metric | f1_macro | Imbalance-aware metric; selected checkpoint is restored at end of training |
| Early stopping | patience=2 | Stops training if validation f1_macro doesn't improve by 0.001 across 2 epochs |

Training is run inside a `try/finally` block with `EmissionsTracker` so that CodeCarbon always stops cleanly even if training is interrupted. Emissions are logged to Drive as `emissions_5class_train.csv`.

**Expected training time:** 25–40 minutes on a T4 GPU (fp16).

---

### Step 14 – Task A Test Evaluation

**What it does:** Evaluates the best-checkpoint model on the held-out test set, generates confusion matrices, produces a per-class classification report, and saves all artifacts to Drive.

**Inference emissions** are tracked separately from training using a second `EmissionsTracker` instance, enabling the per-prediction CO₂ figure that is the key carbon efficiency metric.

**Outputs produced:**
- Test set accuracy, weighted F1, macro F1, MAE
- Inference throughput (samples/sec) and latency (ms/sample)
- CO₂ emissions in grams total and micrograms per prediction
- Two confusion matrices (raw counts + row-normalised percentages)
- Full per-class `classification_report` (precision, recall, F1, support)
- Qualitative inference on 7 hand-crafted example reviews
- `metrics_summary.json` and `label_mapping.json` saved to Drive alongside the model weights

---

### Step 15 – Task B Training (3-class, 2 Epochs)

**What it does:** Configures and launches fine-tuning of the 3-class DistilBERT head. Structurally identical to Step 13 with one deliberate difference.

**KEY DIFFERENCE — Epochs reduced from 3 to 2:**

Task A's training logs showed that validation f1_macro at epoch 1 was 0.5698 and at epoch 3 was 0.5716 — essentially flat. The 3-class problem has fewer decision boundaries to learn (3 classes vs. 5), so convergence is faster. Training for a third epoch would burn additional GPU time and therefore additional emissions without meaningfully improving performance. **Reducing to 2 epochs is a carbon-aware methodology decision**, not a compromise — it saves approximately one-third of Task B's training emissions while the model is already converged.

This is directly relevant to the thesis narrative: even within a single small model, thoughtful hyperparameter choices driven by observed convergence can reduce emissions without sacrificing quality.

---

### Step 16 – Task B Test Evaluation

**What it does:** Mirrors Step 14 for the 3-class task. Generates test metrics, confusion matrices, and classification report, and saves all Task B artifacts to Drive.

---

### Step 17 – LLM Baseline Comparison

**What it does:** Produces a comprehensive head-to-head comparison table of DistilBERT (both tasks) against all seven LLMs, on the **exact same test rows**, with Cohen's κ agreement statistics.

The comparison covers:
- 5-class accuracy and MAE for all models
- 3-class accuracy for all models
- Macro F1 for all models
- Cohen's κ between DistilBERT's predictions and each LLM's predictions
- Carbon emissions comparison (training + inference CO₂ for DistilBERT vs. estimated API call costs for LLMs)

This is the central results table that answers the research question.

---

## Key Design Decisions and Justifications

| Decision | Choice | Why |
|----------|--------|-----|
| Input text | Raw `review` column | DistilBERT's tokeniser handles casing and punctuation; cleaning would destroy signal |
| `max_length` | 128 tokens | Covers 99.45% of reviews; reduces compute ~4× vs. 512 with 0.55% truncation cost |
| Padding strategy | Dynamic per-batch | Median review is 30 tokens; static padding to 128 wastes 75% of compute on padding tokens |
| Class imbalance | Weighted CrossEntropyLoss | 5-star class is ~14× more frequent than 2-star; without weights the model predicts "5" by default |
| Best-model metric | Macro F1 | Accuracy and weighted F1 can be gamed by ignoring minority classes; macro F1 cannot |
| Precision | fp16 | ~40% training speedup and ~50% memory saving with negligible quality cost = direct carbon reduction |
| Task B epochs | 2 (vs. 3 for Task A) | Task A validation f1_macro was flat from epoch 1 to 3; reducing epochs is an evidence-based carbon saving |
| Stratification column | 5-class rating | Preserves both 5-class and 3-class distributions simultaneously; 3-class stratification alone would not |
| Single split | Same rows for Task A and B | Required for comparable headline numbers and for the LLM baseline comparison to be meaningful |

---

## Results and Findings

> *Note: The exact numerical results below are from the DistilBERT EDA baseline preview and training design. Final training metrics are populated at runtime. Update this section after running the notebook.*

### LLM Baselines (5-class, from dataset EDA preview on full cleaned set)

| Model | Parameters | Accuracy | MAE |
|-------|-----------|----------|-----|
| GPT-4 | ~1.76T (est.) | 0.752 | 0.307 |
| Gemini 1.5 Pro | Large | ~0.748 | ~0.315 |
| LLaMA 3.3 70B | 70B | ~0.742 | ~0.320 |
| GPT-3.5 Turbo | ~175B | ~0.680 | ~0.390 |
| GPT-3.5 Instruct | ~175B | ~0.660 | ~0.410 |
| Gemini 1.5 Flash | Medium | ~0.612 | ~0.450 |
| LLaMA 3.1 8B | 8B | ~0.610 | ~0.460 |

The frontier LLMs (GPT-4, Gemini 1.5 Pro, LLaMA 3.3 70B) cluster at ~74–75% accuracy with MAE around 0.31 stars — these form the performance ceiling that DistilBERT targets.

### DistilBERT Results (to be filled after training run)

| Task | Accuracy | Weighted F1 | Macro F1 | MAE |
|------|----------|-------------|----------|-----|
| Task A (5-class) | *(run notebook)* | *(run notebook)* | *(run notebook)* | *(run notebook)* |
| Task B (3-class) | *(run notebook)* | *(run notebook)* | *(run notebook)* | — |

### Key Narrative Finding

GPT-4 achieves 75.2% accuracy on the 5-class task. The fine-tuned DistilBERT — with 66M parameters versus GPT-4's estimated 1.76T, trained on a T4 GPU in under an hour — is designed to land in the competitive range of this top cluster, at an order-of-magnitude lower carbon and compute cost. The thesis argument is not that DistilBERT necessarily *exceeds* GPT-4, but that approaching frontier performance at drastically lower cost makes task-specific fine-tuning the environmentally responsible choice for well-defined classification tasks.

---

## Carbon Efficiency Analysis

One of the central contributions of this project is the concrete, measurable carbon comparison between a fine-tuned small model and frontier LLMs.

### DistilBERT Carbon Footprint (tracked with CodeCarbon)

| Phase | Task | Duration | CO₂ emitted |
|-------|------|----------|-------------|
| Training | Task A (5-class, 3 epochs) | ~30–40 min on T4 | *(logged to `emissions_5class_train.csv`)* |
| Training | Task B (3-class, 2 epochs) | ~20–28 min on T4 | *(logged to `emissions_3class_train.csv`)* |
| Inference | Task A test set (20,075 samples) | *(seconds)* | *(logged to `emissions_5class_test.csv`)* |
| Inference | Task B test set (20,075 samples) | *(seconds)* | *(logged to `emissions_3class_test.csv`)* |

CodeCarbon measures energy consumption by CPU, GPU, and RAM, then converts to kg CO₂eq using the regional grid carbon intensity (auto-detected; UK grid ≈ 250 g CO₂/kWh).

### Why This Matters

Frontier LLM API calls (e.g., GPT-4 via OpenAI API) each trigger inference through a multi-hundred-billion-parameter model. Classifying 200,000 reviews through the GPT-4 API generates orders of magnitude more CO₂ than a single fine-tuning run of DistilBERT — and the fine-tuned model can then be reused for millions of predictions at a per-sample energy cost that is a tiny fraction of each API call.

The per-prediction CO₂ figure (measured in **milligrams** for DistilBERT inference) is the key number that makes this argument concrete and defensible.

---

## How to Reproduce

### Prerequisites

- Google Colab account (free tier with T4 GPU, or Colab Pro for L4/A100)
- Google Drive with at least 2 GB free space
- MHARD dataset CSV (`MHARD_dataset.csv`) — available from the authors of Wang et al. ICWSM 2025 or via the HuggingFace datasets hub

### Steps

**1. Upload the dataset to Google Drive**

Place `MHARD_dataset.csv` at:
```
My Drive/Colab Notebooks/DistilBERT/MHARD_dataset.csv
```

**2. Open the notebook in Google Colab**

Either upload the `.ipynb` file directly to Colab, or open it from GitHub using the Colab badge (if configured).

**3. Enable GPU runtime**

In Colab: `Runtime → Change runtime type → GPU (T4 or better)`

**4. Run all cells in order**

Execute cells sequentially from Step 1 through Step 17. Each step prints a summary of its outputs. Steps 13 and 15 (training) will take 20–40 minutes each.

**5. Retrieve artifacts from Drive**

After the notebook completes, the following artifacts will be in Drive:
```
My Drive/Colab Notebooks/DistilBERT/
├── distilbert_5class_final/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── special_tokens_map.json
│   ├── metrics_summary.json
│   └── label_mapping.json
├── distilbert_3class_final/
│   └── (same structure as above)
└── emissions/
    ├── emissions_5class_train.csv
    ├── emissions_5class_test.csv
    ├── emissions_3class_train.csv
    └── emissions_3class_test.csv
```

### Changing the Dataset Path

If your CSV is in a different Drive location, update this variable in Step 3:
```python
CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/DistilBERT/MHARD_dataset.csv"
```

---

## Repository Structure

```
.
├── README.md                                           # This file
├── Reducing_AI_Carbon_Footprint_A_Study_of_DistilBERT_for_Mental_Health_Sentiment_Analysis.ipynb
│                                                       # Full training and evaluation notebook
├── requirements.txt                                    # Python dependencies
└── emissions/                                          # (generated) CodeCarbon CSV logs
    ├── emissions_5class_train.csv
    ├── emissions_5class_test.csv
    ├── emissions_3class_train.csv
    └── emissions_3class_test.csv
```

The trained model weights are **not included** in this repository due to file size (~250 MB per model). They are saved to Google Drive during training. If you wish to use the pre-trained models without retraining, they can be hosted on HuggingFace Hub — contact the author.

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

Install all dependencies with:
```bash
pip install transformers datasets evaluate accelerate scikit-learn wordcloud codecarbon
```

Or use the full pinned requirements file:
```bash
pip install -r requirements.txt
```

**Note on `transformers` version:** This notebook uses `processing_class=tokenizer` in Trainer (the updated argument name from `transformers>=4.46`). If you are using an older version, replace `processing_class=` with `tokenizer=` in Steps 13 and 15.

---

## Citation

If you use this work, the notebook, or any part of the methodology, please cite:

```bibtex
@misc{lemeke2025distilbert_mhard,
  title  = {Reducing AI Carbon Footprint: A Study of DistilBERT for 
             Mental Health Sentiment Analysis},
  author = {Lemeke, Collins},
  year   = {2025},
  note   = {MSc Artificial Intelligence Dissertation, 
            University of Greater Manchester.
            Centre for Intelligence of Things (CIoTh).
            Supervisor: Prof. Celestine Iwendi.},
  url    = {https://github.com/[your-github-username]/[repo-name]}
}
```

The MHARD dataset should be cited as:
```bibtex
@inproceedings{wang2025mhard,
  title     = {MHARD: Mental Health App Reviews Dataset},
  author    = {Wang et al.},
  booktitle = {Proceedings of ICWSM 2025},
  year      = {2025}
}
```

---

## Author

**Collins Lemeke**  
MSc Artificial Intelligence, University of Greater Manchester  
AI Research @ Centre for Intelligence of Things (CIoTh)
Co-founder, AI Nexus Society @ UGM  
IEEE Member

📧 For questions about this work, feel free to open an issue or reach out via GitHub.

---
