"""Download the UCI Heart Disease (Cleveland) dataset and save it as a CSV.

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/heart+Disease

The processed Cleveland file has 13 features plus a target column (`num`).
Missing values are encoded as "?" in the source and converted to empty cells.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column order as documented by the UCI repository for the processed files.
COLUMNS = [
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
    "num",
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "heart_disease.csv"
)


def download() -> pd.DataFrame:
    """Fetch the raw Cleveland data and return it as a labelled DataFrame."""
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "processed.cleveland.data").write_text(response.text)

    df = pd.read_csv(
        io.StringIO(response.text),
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    return df


def main() -> None:
    df = download()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows and {df.shape[1]} columns to {OUT_PATH}")
    missing = df.isna().sum()
    print(f"Missing values per column:\n{missing[missing > 0]}")


if __name__ == "__main__":
    main()
