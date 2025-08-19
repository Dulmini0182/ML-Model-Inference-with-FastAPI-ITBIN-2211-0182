# Sentiment Analysis API (FastAPI)

A minimal ML inference service using **TF-IDF + Logistic Regression** to classify text sentiment (positive/negative).

## Problem
- **Task:** Binary text sentiment classification
- **Input:** `text` (string)
- **Output:** `prediction` (positive/negative), `confidence` (probability)

## Model
- **Vectorizer:** TfidfVectorizer(1-2 grams, English stopwords)
- **Classifier:** LogisticRegression (max_iter=1000)
- **Why:** Simple, fast, and strong baseline for sentiment tasks

## Data
Place a CSV at `data/sentiment.csv` with columns:
- `text` — the sentence/review
- `label` — `positive` or `negative` (or 1/0)

> If not provided, a tiny fallback dataset is used (for demo only).

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
