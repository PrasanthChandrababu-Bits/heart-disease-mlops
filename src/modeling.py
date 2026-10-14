"""Model training, evaluation, experiment tracking, and packaging."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data_processing import FEATURES, build_preprocessor, load_clean_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
RANDOM_STATE = 42


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics reported for a fitted binary classifier."""

    accuracy: float
    precision: float
    recall: float
    roc_auc: float


@dataclass
class ModelResult:
    """Evaluation result and fitted estimator for one model family."""

    name: str
    estimator: Pipeline
    best_params: dict[str, Any]
    cross_validation_roc_auc: float
    test_metrics: EvaluationMetrics
    confusion_matrix_path: Path
    roc_curve_path: Path
    input_example: pd.DataFrame


def build_candidates(
) -> dict[str, tuple[ClassifierMixin, dict[str, list]]]:
    """Return model families and small reproducible search grids."""
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
    }


def evaluate(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> EvaluationMetrics:
    """Calculate required assignment metrics on held-out data."""
    predictions = estimator.predict(features)
    probabilities = estimator.predict_proba(features)[:, 1]
    return EvaluationMetrics(
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions)),
        recall=float(recall_score(target, predictions)),
        roc_auc=float(roc_auc_score(target, probabilities)),
    )


def save_evaluation_plots(
    name: str,
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    report_dir: Path,
) -> tuple[Path, Path]:
    """Write confusion-matrix and ROC-curve plots for one estimator."""
    report_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = report_dir / f"{name}_confusion_matrix.png"
    roc_path = report_dir / f"{name}_roc_curve.png"

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        estimator,
        features,
        target,
        cmap="Blues",
        ax=axis,
    )
    axis.set_title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(estimator, features, target, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_title(f"{name.replace('_', ' ').title()} ROC Curve")
    figure.tight_layout()
    figure.savefig(roc_path, dpi=150)
    plt.close(figure)

    return confusion_path, roc_path


def train_candidate(
    name: str,
    classifier: ClassifierMixin,
    parameter_grid: dict[str, list],
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    report_dir: Path,
    cross_validation_splits: int = 5,
) -> ModelResult:
    """Tune and evaluate one model family."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
    cross_validation = StratifiedKFold(
        n_splits=cross_validation_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
    )
    search.fit(train_features, train_target)
    metrics = evaluate(search.best_estimator_, test_features, test_target)
    confusion_path, roc_path = save_evaluation_plots(
        name,
        search.best_estimator_,
        test_features,
        test_target,
        report_dir,
    )
    return ModelResult(
        name=name,
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cross_validation_roc_auc=float(search.best_score_),
        test_metrics=metrics,
        confusion_matrix_path=confusion_path,
        roc_curve_path=roc_path,
        input_example=test_features.head(3),
    )


def log_result(result: ModelResult) -> None:
    """Log one completed experiment to MLflow."""
    with mlflow.start_run(run_name=result.name):
        mlflow.set_tag("model_family", result.name)
        mlflow.set_tag("dataset", "UCI Cleveland Heart Disease")
        mlflow.log_params(result.best_params)
        mlflow.log_metric(
            "cross_validation_roc_auc",
            result.cross_validation_roc_auc,
        )
        mlflow.log_metrics(asdict(result.test_metrics))
        mlflow.log_artifact(str(result.confusion_matrix_path), "plots")
        mlflow.log_artifact(str(result.roc_curve_path), "plots")
        signature = infer_signature(
            result.input_example,
            result.estimator.predict_proba(result.input_example),
        )
        mlflow.sklearn.log_model(
            result.estimator,
            "model",
            signature=signature,
            input_example=result.input_example,
        )


def select_best(results: list[ModelResult]) -> ModelResult:
    """Select the model with the highest cross-validation ROC-AUC."""
    return max(
        results,
        key=lambda result: result.cross_validation_roc_auc,
    )


def save_comparison(results: list[ModelResult], report_dir: Path) -> Path:
    """Persist a machine-readable comparison of all candidate models."""
    comparison_path = report_dir / "model_comparison.json"
    payload = {
        result.name: {
            "best_params": result.best_params,
            "cross_validation_roc_auc": result.cross_validation_roc_auc,
            "test_metrics": asdict(result.test_metrics),
        }
        for result in results
    }
    comparison_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison_path


def train_all(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cross_validation_splits: int = 5,
) -> tuple[ModelResult, list[ModelResult]]:
    """Train all candidates, track runs, and package the selected pipeline."""
    features, target = load_clean_split()
    train_features, test_features, train_target, test_target = (
        train_test_split(
            features,
            target,
            test_size=0.2,
            stratify=target,
            random_state=RANDOM_STATE,
        )
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-classification")

    results = []
    for name, (classifier, parameter_grid) in build_candidates().items():
        result = train_candidate(
            name,
            classifier,
            parameter_grid,
            train_features,
            train_target,
            test_features,
            test_target,
            report_dir,
            cross_validation_splits,
        )
        log_result(result)
        results.append(result)

    selected = select_best(results)
    comparison_path = save_comparison(results, report_dir)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected.estimator, model_path)
    metadata = {
        "model_name": selected.name,
        "features": FEATURES,
        "decision_threshold": 0.5,
        "selection_rule": "highest cross-validation ROC-AUC",
        "cross_validation_splits": cross_validation_splits,
        "cross_validation_roc_auc": selected.cross_validation_roc_auc,
        "test_metrics": asdict(selected.test_metrics),
        "comparison_artifact": os.path.relpath(
            comparison_path,
            metadata_path.parent,
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="selected-model-package"):
        mlflow.set_tag("selected_model", selected.name)
        mlflow.log_metrics(asdict(selected.test_metrics))
        mlflow.log_artifact(str(comparison_path), "evaluation")
        mlflow.log_artifact(str(metadata_path), "package")
        signature = infer_signature(
            selected.input_example,
            selected.estimator.predict_proba(selected.input_example),
        )
        mlflow.sklearn.log_model(
            selected.estimator,
            "model",
            signature=signature,
            input_example=selected.input_example,
        )

    return selected, results
