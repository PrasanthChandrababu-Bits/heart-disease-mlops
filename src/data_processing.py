"""Data loading, cleaning and preprocessing for the Heart Disease dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "heart_disease.csv"
)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"
RAW_TARGET = "num"
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    RAW_TARGET,
]
NULLABLE_COLUMNS: Final = {"ca", "thal"}
ALLOWED_VALUES: Final = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
    RAW_TARGET: {0, 1, 2, 3, 4},
}


def validate_raw_data(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` when the source dataset violates its schema."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = sorted(set(RAW_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(RAW_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(
            "Dataset columns do not match the expected schema. "
            f"Missing: {missing_columns or 'none'}; "
            f"unexpected: {extra_columns or 'none'}."
        )

    numeric = df[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna() & df[RAW_COLUMNS].notna()
    invalid_columns = invalid_numeric.any()
    if invalid_columns.any():
        names = invalid_columns[invalid_columns].index.tolist()
        raise ValueError(f"Non-numeric values found in columns: {names}.")

    required_columns = [
        column for column in RAW_COLUMNS if column not in NULLABLE_COLUMNS
    ]
    missing_required = numeric[required_columns].isna().any()
    if missing_required.any():
        names = missing_required[missing_required].index.tolist()
        raise ValueError(f"Missing values found in required columns: {names}.")

    for column, allowed in ALLOWED_VALUES.items():
        observed = set(numeric[column].dropna().unique())
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise ValueError(
                f"Unexpected values in '{column}': {unexpected}; "
                f"allowed values are {sorted(allowed)}."
            )


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV produced by scripts/download_data.py."""
    df = pd.read_csv(path)
    validate_raw_data(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame: binarise the target and coerce numeric types.

    The source target column `num` ranges 0-4 (0 = no disease). The problem is
    framed as binary, so any value > 0 is treated as disease present.
    """
    validate_raw_data(df)
    df = df.copy()
    df[TARGET] = (df[RAW_TARGET] > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET])

    # `ca` and `thal` arrive as floats with some missing values; keep them
    # numeric so the imputer in the pipeline can fill them.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_preprocessor() -> ColumnTransformer:
    """Return the preprocessing ColumnTransformer (impute + scale + encode)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into the feature matrix X and target vector y."""
    return df[FEATURES], df[TARGET]


def load_clean_split() -> tuple[pd.DataFrame, pd.Series]:
    """Convenience helper: load, clean and return (X, y)."""
    df = clean_data(load_data())
    return get_features_and_target(df)
