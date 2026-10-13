"""
Model-serving layer: a FastAPI app that loads the current "Production"
model version from the MLflow Model Registry and exposes it as a REST
API. Designed to run as a standalone container behind Docker/K8s.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import List

import joblib
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "wine-classifier-prod")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
SCALER_PATH = os.getenv("SCALER_PATH", "data/processed/scaler.joblib")
N_FEATURES = int(os.getenv("N_FEATURES", 13))  # wine dataset has 13 features

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

model_state = {"model": None, "scaler": None, "version": None}


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    model_state["model"] = model
    model_state["scaler"] = scaler
    logger.info("Model loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        # Don't crash the container on boot if the registry is briefly
        # unavailable; /health will report not-ready and k8s can retry.
        logger.error("Model load failed at startup: %s", e)
    yield


app = FastAPI(title="MLOps Model Serving API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Feature vector of length {N_FEATURES}",
    )


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int
    probabilities: List[float]
    model_stage: str



@app.get("/health")
def health():
    ready = model_state["model"] is not None
    return {"status": "ok" if ready else "loading", "model_loaded": ready}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model():
    """Hot-reload the latest Production model without restarting the pod."""
    load_model()
    return {"status": "reloaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model_state["model"] is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    with REQUEST_LATENCY.time():
        try:
            X = np.array(req.features).reshape(1, -1)
            if model_state["scaler"] is not None:
                X = model_state["scaler"].transform(X)

            pred = model_state["model"].predict(X)[0]
            proba = model_state["model"].predict_proba(X)[0].tolist()

            REQUEST_COUNT.labels(status="success").inc()
            return PredictionResponse(
                prediction=int(pred), probabilities=proba, model_stage=MODEL_STAGE
            )
        except Exception as e:
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
