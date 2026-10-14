"""Tests for reproducible EDA artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eda import run_eda
from src.data_processing import DATA_PATH


def test_run_eda_writes_expected_artifacts(tmp_path: Path) -> None:
    summary = run_eda(DATA_PATH, tmp_path)

    assert summary["rows"] == 303
    assert summary["missing_values"] == {"ca": 4, "thal": 2}
    assert summary["binary_target_counts"] == {"0": 164, "1": 139}

    expected_files = {
        "class_balance.png",
        "correlation_heatmap.png",
        "numeric_histograms.png",
        "summary.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
