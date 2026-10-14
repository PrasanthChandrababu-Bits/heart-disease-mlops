"""Generate reproducible EDA artifacts for the Heart Disease dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (  # noqa: E402
    DATA_PATH,
    NUMERIC_FEATURES,
    RAW_TARGET,
    TARGET,
    clean_data,
    load_data,
)

plt.switch_backend("Agg")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"


def build_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Return dataset characteristics in a JSON-serializable structure."""
    target_counts = clean[TARGET].value_counts().sort_index()
    return {
        "rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "feature_columns": int(clean.shape[1] - 1),
        "duplicate_rows": int(raw.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in raw.isna().sum().items()
            if count
        },
        "binary_target_counts": {
            str(int(label)): int(count)
            for label, count in target_counts.items()
        },
        "binary_target_percentages": {
            str(int(label)): round(float(count / len(clean) * 100), 2)
            for label, count in target_counts.items()
        },
        "source_target_counts": {
            str(int(label)): int(count)
            for label, count in raw[RAW_TARGET]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def save_numeric_histograms(clean: pd.DataFrame, output_dir: Path) -> None:
    """Save distributions of continuous numeric features."""
    axes = clean[NUMERIC_FEATURES].hist(
        bins=15,
        figsize=(12, 8),
        color="#2878B5",
        edgecolor="white",
    )
    figure = axes[0, 0].figure
    figure.suptitle("Numeric Feature Distributions", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "numeric_histograms.png", dpi=150)
    plt.close(figure)


def save_class_balance(clean: pd.DataFrame, output_dir: Path) -> None:
    """Save the binary target class distribution."""
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.countplot(
        data=clean,
        x=TARGET,
        hue=TARGET,
        palette=["#4C72B0", "#DD8452"],
        legend=False,
        ax=axis,
    )
    axis.set(
        title="Heart Disease Class Balance",
        xlabel="Heart disease (0 = absent, 1 = present)",
        ylabel="Patients",
    )
    for container in axis.containers:
        axis.bar_label(container)
    figure.tight_layout()
    figure.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close(figure)


def save_correlation_heatmap(clean: pd.DataFrame, output_dir: Path) -> None:
    """Save a correlation heatmap for encoded source features and target."""
    correlation = clean.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        ax=axis,
    )
    axis.set_title("Feature Correlation Heatmap")
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(figure)


def run_eda(data_path: Path, output_dir: Path) -> dict:
    """Validate data and write the complete Phase 1 EDA artifact set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_data(data_path)
    clean = clean_data(raw)
    summary = build_summary(raw, clean)

    save_numeric_histograms(clean, output_dir)
    save_class_balance(clean, output_dir)
    save_correlation_heatmap(clean, output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run EDA and print the generated artifact location."""
    args = parse_args()
    summary = run_eda(args.data, args.output_dir)
    print(
        f"Generated EDA for {summary['rows']} rows in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
