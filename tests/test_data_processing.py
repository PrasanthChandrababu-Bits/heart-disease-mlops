"""Tests for dataset validation and preprocessing."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_processing import (
    FEATURES,
    RAW_COLUMNS,
    clean_data,
    get_features_and_target,
    load_data,
)


def make_valid_row() -> dict[str, float]:
    """Return one valid source record."""
    return {
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
        "num": 0,
    }


def test_load_data_validates_real_dataset() -> None:
    raw = load_data()

    assert list(raw.columns) == RAW_COLUMNS
    assert raw.shape == (303, 14)
    assert raw.isna().sum().loc[["ca", "thal"]].to_dict() == {
        "ca": 4,
        "thal": 2,
    }


def test_clean_data_creates_binary_target() -> None:
    first = make_valid_row()
    second = {**first, "num": 4}

    clean = clean_data(pd.DataFrame([first, second]))
    features, target = get_features_and_target(clean)

    assert features.columns.tolist() == FEATURES
    assert target.tolist() == [0, 1]
    assert "num" not in clean.columns


def test_validation_rejects_missing_column() -> None:
    invalid = pd.DataFrame([make_valid_row()]).drop(columns=["age"])

    with pytest.raises(ValueError, match="Missing: \\['age'\\]"):
        clean_data(invalid)


def test_validation_rejects_unexpected_category() -> None:
    invalid = pd.DataFrame([{**make_valid_row(), "sex": 7}])

    with pytest.raises(ValueError, match="Unexpected values in 'sex'"):
        clean_data(invalid)


def test_validation_rejects_non_numeric_value() -> None:
    invalid = pd.DataFrame([{**make_valid_row(), "chol": "unknown"}])

    with pytest.raises(ValueError, match="Non-numeric values.*chol"):
        clean_data(invalid)
