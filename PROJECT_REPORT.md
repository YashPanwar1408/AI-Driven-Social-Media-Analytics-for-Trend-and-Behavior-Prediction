# Project Report

## Title
**AI-Driven Social Media Analytics for Trend and Behavior Prediction**

---

## Abstract
Social media platforms generate high‑volume, high‑velocity text data that can be used to understand public opinion and emerging trends. This project builds an end‑to‑end analytics system that performs (1) sentiment analysis on short posts, (2) hashtag‑based trend detection over time, and (3) time‑series forecasting to predict the future popularity of a trend. A complete execution pipeline is provided so the project can be run from raw data to outputs with a single command. The solution includes a TF‑IDF + Logistic Regression baseline model, an optional BERT fine‑tuning module, a PyTorch LSTM forecasting model for hashtag frequency prediction, and a FastAPI backend for serving results through REST endpoints.

---

## Introduction
Social media text reflects user opinions, reactions to events, and community behavior. Automated analysis of this data is valuable for multiple use cases, including brand monitoring, event detection, community management, and early warning systems. However, social media data is noisy: it includes informal language, spelling variations, emojis, URLs, mentions, and hashtags.

This project addresses three key analytics goals:
1. **Sentiment classification**: predict whether a post is negative or positive.
2. **Trend detection**: identify which hashtags are most frequent overall and in recent time windows.
3. **Trend forecasting**: predict future hashtag frequency using time‑series modeling.

The project is designed to be beginner‑friendly while still demonstrating a full ML lifecycle: preprocessing, modeling, evaluation, artifact saving/loading, visualization, and API deployment.

---

## Methodology

### 1) Dataset
- **Source format**: Sentiment140‑style CSV (no header, 6 columns)
- **Key columns used**: `sentiment`, `timestamp`, `text`
- **Label convention**: typically `0 = negative`, `4 = positive`

Important dataset note: the Sentiment140 training file is often **ordered** (negatives first, then positives). For model training/evaluation, this project uses **balanced sampling** to avoid accidentally training on only one class.

### 2) Data Preprocessing
A lightweight NLP cleaning pipeline is applied to create a `clean_text` feature:
- Lowercasing
- URL removal
- Emoji removal
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization

NLTK resources (tokenizers, stopwords, WordNet) are automatically downloaded if missing.

### 3) Sentiment Modeling
Two sentiment modeling approaches are provided:

**A. TF‑IDF + Logistic Regression (Baseline)**
- Convert text into TF‑IDF vectors
- Train Logistic Regression classifier
- Fast and interpretable baseline

**B. BERT Fine‑Tuning (Optional)**
- Fine‑tune a pretrained Transformer on the sentiment task
- Uses a small BERT checkpoint to make CPU training feasible
- This stage is optional and can be skipped using the pipeline fast mode

### 4) Trend Detection
Trends are detected using hashtags:
- Extract hashtags with regex (e.g., `#AI`, `#fail`)
- Normalize to lowercase
- Aggregate counts by time bucket (daily/weekly)
- Identify:
  - top hashtags overall
  - top hashtags in the latest time period

### 5) Trend Forecasting (LSTM)
A time series is constructed for a target hashtag:
- Hashtag counts per day/week
- Missing time periods are filled with zero counts

An LSTM is trained to predict the next value from a sliding window of past counts:
- Input: previous `lookback` periods
- Output: next period count

The trained artifact stores:
- hashtag name
- time frequency
- lookback
- scaling parameters (min/max)
- network hyperparameters

### 6) Visualizations
Plots are generated using Matplotlib/Seaborn and saved as PNG files (non‑interactive backend):
- Sentiment distribution (bar chart)
- Hashtag trend over time (line chart)
- Model comparison (bar chart)

### 7) Deployment (FastAPI)
A production‑style FastAPI backend is provided:
- Loads trained artifacts from `./artifacts/`
- Provides endpoints:
  - `POST /sentiment`
  - `GET /trends`
  - `POST /predict`

Artifacts are reused (loaded if already trained) to avoid unnecessary retraining.

---

## Algorithms Used

### A) TF‑IDF (Term Frequency–Inverse Document Frequency)
- Represents each post as a sparse vector
- Higher weights for words that are frequent in a document but rare across the corpus

### B) Logistic Regression
- Linear classifier trained on TF‑IDF features
- Outputs class probabilities via a sigmoid/softmax depending on the number of classes

### C) BERT (Bidirectional Encoder Representations from Transformers)
- Transformer architecture using self‑attention
- Produces contextual embeddings (word meaning depends on surrounding words)
- Fine‑tuning adapts a pretrained language model to the sentiment classification task

### D) Hashtag Trend Aggregation
- Regex extraction + grouping counts by time bucket
- Produces interpretable time series for each hashtag

### E) LSTM (Long Short‑Term Memory)
- Recurrent neural network designed for sequential data
- Learns temporal dependencies and patterns in hashtag frequency
- Used for multi‑step recursive forecasting

---

## Results
Results below are taken from the pipeline outputs generated in this repository.

### 1) Sentiment Model Metrics
From `outputs/metrics/model_metrics.json`:

- **TF‑IDF + Logistic Regression**
  - Accuracy: **0.8278**
  - Precision: **0.8260**
  - Recall: **0.8305**
  - F1‑score: **0.8282**

- **BERT (small model, 1 epoch fine‑tuning)**
  - Accuracy: **0.6305**
  - Precision: **0.6766**
  - Recall: **0.5000**
  - F1‑score: **0.5750**

**Observation:** In this CPU‑friendly configuration (small BERT + 1 epoch), TF‑IDF outperformed BERT. This is a realistic outcome when transformer training is under‑tuned or compute‑constrained. With a GPU, a larger checkpoint (e.g., `bert-base-uncased`) and more epochs, BERT often improves.

### 2) Trend Detection
From `outputs/trends/trends_D.json` (daily grouping on the loaded sample):

Top hashtags overall included:
- `#fb` (218)
- `#asot400` (137)
- `#fail` (74)
- `#followfriday` (66)
- `#f1` (58)

The latest detected day in the processed window showed top hashtags like:
- `#marsiscoming` (40)
- `#myweakness` (27)
- `#asylm` (15)

### 3) Forecasting Output (LSTM)
From `outputs/forecast/forecast.json` (trained hashtag: `#fb`, daily frequency, lookback = 14):

Example future predictions (counts per day) were approximately:
- 2009‑05‑31: **3.63**
- 2009‑06‑01: **3.58**
- 2009‑06‑02: **3.56**

This demonstrates the end‑to‑end ability to forecast trend strength into the future.

### 4) Visual Evidence
Generated plots saved under `outputs/plots/`:
- Sentiment distribution bar chart
- Hashtag trend line chart
- Model comparison bar chart

---

## Conclusion
This project successfully implements an end‑to‑end pipeline for AI‑driven social media analytics. It integrates preprocessing, sentiment classification, trend detection, and forecasting, and provides a FastAPI backend for serving predictions. The TF‑IDF baseline achieved strong performance and serves as a reliable benchmark. Trend detection provided interpretable insights into the most frequent hashtags, and the LSTM forecaster demonstrated how trend behavior can be predicted using time‑series modeling.

---

## Future Scope
1. **Improve transformer performance**: use a GPU, increase epochs, tune learning rate, and experiment with larger models.
2. **Multi‑hashtag forecasting**: train models for multiple hashtags or build a single model that forecasts multiple series.
3. **Richer trend features**: incorporate retweets, mentions, topic modeling, and burst detection algorithms.
4. **Advanced preprocessing**: handle negation more explicitly, preserve emojis (often sentiment‑bearing), and use domain‑specific tokenization.
5. **Better evaluation and monitoring**: add cross‑validation, calibration, and drift detection for real deployments.
6. **Deployment hardening**: add authentication, rate limiting, and structured logging; containerize with Docker for portable deployment.

---

## How to Run (Summary)
- Fast pipeline: `python run_pipeline.py --fast`
- Full pipeline: `python run_pipeline.py --full`
- API: `python -m uvicorn api.main:app --reload`
- API test: `python test_api.py`
