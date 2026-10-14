"""Tests for the prediction API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_health_loads_packaged_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_name"]


def test_predict_returns_prediction_and_confidence() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {0, 1}
    assert body["disease_present"] == bool(body["prediction"])
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


def test_predict_rejects_out_of_range_category() -> None:
    response = client.post(
        "/predict",
        json={**VALID_PAYLOAD, "cp": 9},
    )

    assert response.status_code == 422


def test_predict_rejects_undocumented_thal_category() -> None:
    response = client.post(
        "/predict",
        json={**VALID_PAYLOAD, "thal": 5},
    )

    assert response.status_code == 422


def test_metrics_are_exposed() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
