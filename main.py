from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import os
import json

APP_TITLE = "ML Model API - Sentiment"
APP_DESC = "API for sentiment analysis (TF-IDF + LogisticRegression)"
MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"

app = FastAPI(title=APP_TITLE, description=APP_DESC)

# Global holders (loaded at startup)
model = None
model_meta = {}

class PredictionInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text")

class BatchPredictionInput(BaseModel):
    texts: List[str] = Field(..., description="List of input texts")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

@app.on_event("startup")
def load_model():
    global model, model_meta
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Train the model first.")
    model = joblib.load(MODEL_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            model_meta = json.load(f)
    else:
        model_meta = {
            "problem_type": "binary_classification",
            "model_type": "TFIDF + LogisticRegression",
            "note": "No meta file found; using defaults."
        }

@app.get("/", tags=["health"])
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput, tags=["inference"])
def predict(input_data: PredictionInput):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        # Predict
        proba_supported = hasattr(model, "predict_proba")
        pred = model.predict([input_data.text])[0]
        conf = None
        if proba_supported:
            probs = model.predict_proba([input_data.text])[0]
            conf = float(max(probs))
        return PredictionOutput(prediction=str(pred), confidence=conf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["inference"])
def predict_batch(batch: BatchPredictionInput):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if not batch.texts or not isinstance(batch.texts, list):
            raise HTTPException(status_code=422, detail="Field 'texts' must be a non-empty list of strings.")

        texts = [t if isinstance(t, str) and t.strip() else "" for t in batch.texts]
        preds = model.predict(texts)
        outputs: List[PredictionOutput] = []
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(texts)
            for label, p in zip(preds, probas):
                outputs.append(PredictionOutput(prediction=str(label), confidence=float(max(p))))
        else:
            for label in preds:
                outputs.append(PredictionOutput(prediction=str(label), confidence=None))
        return BatchPredictionOutput(predictions=outputs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {e}")

@app.get("/model-info", tags=["info"])
def model_info():
    features = ["text"]
    return {
        "model_type": model_meta.get("model_type", "Unknown"),
        "problem_type": model_meta.get("problem_type", "Unknown"),
        "features": features,
        "meta": model_meta,
    }
