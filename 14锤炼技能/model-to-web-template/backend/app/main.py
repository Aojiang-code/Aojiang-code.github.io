from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import json

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "iris_rf.pkl"))
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
TARGET_NAMES_PATH = os.path.join(MODEL_DIR, "target_names.json")

app = FastAPI(title="Model-to-Web Backend", version="0.1.0")

# CORS for local dev and common docker-compose hostnames
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://frontend:3000",
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    X: List[List[float]] = Field(..., description="2D array-like input features")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature names matching columns")

class PredictResponse(BaseModel):
    y_pred: List[int]
    y_proba: Optional[List[List[float]]] = None
    target_names: Optional[List[str]] = None

def _load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run scripts/train.py first.")
    model = joblib.load(MODEL_PATH)
    feature_names = None
    target_names = None
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            feature_names = json.load(f)
    if os.path.exists(TARGET_NAMES_PATH):
        with open(TARGET_NAMES_PATH) as f:
            target_names = json.load(f)
    return model, feature_names, target_names

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, model_feature_names, target_names = _load_artifacts()
    X = np.array(req.X, dtype=float)
    # Optional: check feature alignment
    if req.feature_names and model_feature_names:
        if list(req.feature_names) != list(model_feature_names):
            raise HTTPException(status_code=400, detail={
                "error": "Feature names mismatch",
                "expected": model_feature_names,
                "got": req.feature_names,
            })
    y_pred = model.predict(X).tolist()
    y_proba = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None
    return {"y_pred": y_pred, "y_proba": y_proba, "target_names": target_names}

@app.post("/predict_csv", response_model=PredictResponse)
async def predict_csv(file: UploadFile = File(...)):
    model, model_feature_names, target_names = _load_artifacts()
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
    if model_feature_names:
        missing = [c for c in model_feature_names if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing}. Expected: {model_feature_names}")
        X = df[model_feature_names].values
    else:
        X = df.values
    y_pred = model.predict(X).tolist()
    y_proba = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None
    return {"y_pred": y_pred, "y_proba": y_proba, "target_names": target_names}
