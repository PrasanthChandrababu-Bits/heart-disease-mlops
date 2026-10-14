"""Tests for training, selection, and packaging."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.data_processing import load_clean_split
from src.modeling import train_all


def test_train_all_packages_complete_pipeline(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    report_dir = tmp_path / "reports"
    tracking_uri = (tmp_path / "mlruns").as_uri()

    selected, results = train_all(
        model_path=model_path,
        metadata_path=metadata_path,
        report_dir=report_dir,
        tracking_uri=tracking_uri,
        cross_validation_splits=2,
    )

    assert {result.name for result in results} == {
        "logistic_regression",
        "random_forest",
    }
    assert 0.0 <= selected.test_metrics.roc_auc <= 1.0
    assert model_path.is_file()
    assert metadata_path.is_file()
    assert (report_dir / "model_comparison.json").is_file()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["model_name"] == selected.name
    assert metadata["decision_threshold"] == 0.5
    assert metadata["selection_rule"] == (
        "highest cross-validation ROC-AUC"
    )

    model = joblib.load(model_path)
    features, _ = load_clean_split()
    prediction = model.predict(features.head(1))
    probability = model.predict_proba(features.head(1))
    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)
