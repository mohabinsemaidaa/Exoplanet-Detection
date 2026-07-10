"""Small FastAPI app serving exoplanet disposition predictions."""

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import CORE_FEATURES
from src.train import MODEL_PATH

app = FastAPI(title="Exoplanet Detection API")
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_model_snr: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float
    koi_kepmag: float
    koi_impact: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/")
def read_root():
    return {"message": "Exoplanet Detection API. See /docs for usage."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    input_df = pd.DataFrame([request.model_dump()])[CORE_FEATURES]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(max(probabilities))

    return PredictionResponse(prediction=prediction, confidence=confidence)
