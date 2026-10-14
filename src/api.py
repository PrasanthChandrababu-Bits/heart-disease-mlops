"""FastAPI service for heart-disease predictions."""
from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import time
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from joblib import load
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from src.data_processing import FEATURES

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("heart_disease_api")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"

Binary = Annotated[int, Field(ge=0, le=1)]


class PredictionRequest(BaseModel):
    """Raw UCI feature values accepted by the model."""

    model_config = ConfigDict(extra="forbid")

    age: Annotated[float, Field(gt=0, le=120)]
    sex: Binary
    cp: Annotated[int, Field(ge=1, le=4)]
    trestbps: Annotated[float, Field(gt=0, le=300)]
    chol: Annotated[float, Field(gt=0, le=700)]
    fbs: Binary
    restecg: Annotated[int, Field(ge=0, le=2)]
    thalach: Annotated[float, Field(gt=0, le=250)]
    exang: Binary
    oldpeak: Annotated[float, Field(ge=0, le=10)]
    slope: Annotated[int, Field(ge=1, le=3)]
    ca: Annotated[int | None, Field(ge=0, le=3)] = None
    thal: Literal[3, 6, 7] | None = None


class PredictionResponse(BaseModel):
    """Prediction and model probability-based confidence."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    disease_present: bool
    confidence: float
    probability_disease: float
    model_name: str


class HealthResponse(BaseModel):
    """Service health and loaded-model status."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str


def model_path() -> Path:
    """Return configured model artifact path."""
    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def metadata_path() -> Path:
    """Return configured model metadata path."""
    return Path(os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH)))


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Load and cache the packaged sklearn pipeline."""
    path = model_path()
    if not path.is_file():
        raise RuntimeError(
            f"Model artifact not found at {path}. Run scripts/train.py first."
        )
    return load(path)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """Load and cache packaged model metadata."""
    path = metadata_path()
    if not path.is_file():
        raise RuntimeError(
            f"Model metadata not found at {path}. Run scripts/train.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Binary risk prediction using the UCI Cleveland dataset.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, status, and duration."""
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    LOGGER.info(
        "method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verify that the service can load its packaged model."""
    load_model()
    metadata = load_metadata()
    return HealthResponse(status="healthy", model_name=metadata["model_name"])


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict disease presence and return the chosen class confidence."""
    model = load_model()
    metadata = load_metadata()
    frame = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(metadata["decision_threshold"])
    prediction = int(probability >= threshold)
    confidence = probability if prediction else 1 - probability
    return PredictionResponse(
        prediction=prediction,
        disease_present=bool(prediction),
        confidence=round(confidence, 6),
        probability_disease=round(probability, 6),
        model_name=metadata["model_name"],
    )


Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)
