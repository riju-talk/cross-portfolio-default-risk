from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from .diagnostics import (
        build_validation_metrics_map,
        model_slug,
        record_output_path,
        save_calibration_plot,
        save_class_distribution_plot,
        save_confusion_matrix_plot,
        save_feature_importance_plot,
        save_metric_comparison_plot,
        save_missingness_plot,
        save_numeric_feature_grid,
        save_precision_recall_plot,
        save_prediction_table,
        save_roc_curve_plot,
        save_score_distribution_plot,
        save_train_vs_validation_plot,
    )
except ImportError:
    from diagnostics import (
        build_validation_metrics_map,
        model_slug,
        record_output_path,
        save_calibration_plot,
        save_class_distribution_plot,
        save_confusion_matrix_plot,
        save_feature_importance_plot,
        save_metric_comparison_plot,
        save_missingness_plot,
        save_numeric_feature_grid,
        save_precision_recall_plot,
        save_prediction_table,
        save_roc_curve_plot,
        save_score_distribution_plot,
        save_train_vs_validation_plot,
    )


SEED = 42
DEFAULT_DATA_PATH = Path("data/default_of_credit_card_clients.csv")
DEFAULT_RESULTS_DIR = Path("results/card")
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "metrics.json"
CARD_CATEGORICAL_COLUMNS = {"SEX", "EDUCATION", "MARRIAGE"}


@dataclass
class ModelResult:
    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    top_decile_capture_rate: float


def _top_decile_capture(y_true: pd.Series, y_prob: np.ndarray) -> float:
    top_n = max(1, int(np.ceil(0.10 * len(y_prob))))
    ranked_indices = np.argsort(y_prob)[::-1][:top_n]
    captured = int(y_true.iloc[ranked_indices].sum())
    positives = int(y_true.sum())
    return float(captured / positives) if positives > 0 else 0.0


def load_card_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find credit-card dataset at {csv_path}. Run the download script first."
        )

    df = pd.read_csv(csv_path)

    target_candidates = [
        "Y",
        "default payment next month",
        "default.payment.next.month",
        "default_payment_next_month",
        "target",
    ]
    target_col = next((col for col in target_candidates if col in df.columns), None)
    if target_col is None:
        raise ValueError(
            "No recognized target column found. Expected one of: "
            + ", ".join(target_candidates)
        )

    columns_to_drop = [target_col] + (["ID"] if "ID" in df.columns else [])
    features = df.drop(columns=columns_to_drop)
    target = df[target_col].astype(int)
    return features, target, target_col


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [
        c
        for c in features.columns
        if c in CARD_CATEGORICAL_COLUMNS
        or pd.api.types.is_object_dtype(features[c])
        or pd.api.types.is_string_dtype(features[c])
        or pd.api.types.is_categorical_dtype(features[c])
    ]
    numeric_features = [c for c in features.columns if c not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=0.8,
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=20,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
        )

    return models


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "top_decile_capture_rate": _top_decile_capture(y_true, y_prob),
    }


def _save_dataset_level_plots(
    X: pd.DataFrame,
    y: pd.Series,
    plots_dir: Path,
    generated_output_files: list[str],
) -> None:
    dataset_plot_dir = plots_dir / "dataset"
    dataset_plot_dir.mkdir(parents=True, exist_ok=True)

    target_plot_path = dataset_plot_dir / "target_distribution.png"
    save_class_distribution_plot(
        y=y,
        output_path=target_plot_path,
        title="Credit Card Default Class Distribution",
    )
    record_output_path(target_plot_path, generated_output_files)

    missingness_plot_path = dataset_plot_dir / "missing_values.png"
    if save_missingness_plot(
        features=X,
        output_path=missingness_plot_path,
        title="Top Missing-Value Rates (Credit Card Dataset)",
    ):
        record_output_path(missingness_plot_path, generated_output_files)

    numeric_grid_path = dataset_plot_dir / "numeric_feature_distributions.png"
    if save_numeric_feature_grid(
        features=X,
        output_path=numeric_grid_path,
        title="Representative Numeric Feature Distributions",
    ):
        record_output_path(numeric_grid_path, generated_output_files)


def run_card_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    output_path = Path(output_path)
    results_dir = output_path.parent
    plots_dir = results_dir / "plots"
    predictions_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    X, y, target_col = load_card_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    preprocessor = build_preprocessor(X)

    results: list[ModelResult] = []
    reports: dict[str, dict[str, Any]] = {}
    train_validation_metrics: dict[str, dict[str, dict[str, float]]] = {}
    generated_output_files: list[str] = []

    _save_dataset_level_plots(
        X=X,
        y=y,
        plots_dir=plots_dir,
        generated_output_files=generated_output_files,
    )

    for name, estimator in make_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_train_prob = pipeline.predict_proba(X_train)[:, 1]
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        train_metrics = evaluate_predictions(y_train, y_train_prob)
        train_validation_metrics[name] = {
            "train": train_metrics,
            "validation": evaluate_predictions(y_test, y_prob),
        }

        metrics = train_validation_metrics[name]["validation"]
        results.append(ModelResult(name=name, **metrics))
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

        model_plots_dir = plots_dir / model_slug(name)
        model_plots_dir.mkdir(parents=True, exist_ok=True)

        prediction_path = predictions_dir / f"{model_slug(name)}_validation_predictions.csv"
        save_prediction_table(
            y_true=y_test,
            y_prob=y_prob,
            y_pred=y_pred,
            output_path=prediction_path,
        )
        record_output_path(prediction_path, generated_output_files)

        roc_path = model_plots_dir / "roc_curve.png"
        save_roc_curve_plot(
            y_true=y_test,
            y_prob=y_prob,
            output_path=roc_path,
            title=f"ROC Curve - {name}",
        )
        record_output_path(roc_path, generated_output_files)

        pr_path = model_plots_dir / "precision_recall_curve.png"
        save_precision_recall_plot(
            y_true=y_test,
            y_prob=y_prob,
            output_path=pr_path,
            title=f"Precision-Recall Curve - {name}",
        )
        record_output_path(pr_path, generated_output_files)

        cm_path = model_plots_dir / "confusion_matrix.png"
        save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=y_pred,
            output_path=cm_path,
            title=f"Confusion Matrix - {name}",
        )
        record_output_path(cm_path, generated_output_files)

        distribution_path = model_plots_dir / "prediction_score_distribution.png"
        save_score_distribution_plot(
            y_true=y_test,
            y_prob=y_prob,
            output_path=distribution_path,
            title=f"Prediction Score Distribution - {name}",
        )
        record_output_path(distribution_path, generated_output_files)

        calibration_path = model_plots_dir / "calibration_curve.png"
        save_calibration_plot(
            y_true=y_test,
            y_prob=y_prob,
            output_path=calibration_path,
            title=f"Calibration Curve - {name}",
        )
        record_output_path(calibration_path, generated_output_files)

        train_validation_path = model_plots_dir / "train_vs_validation_metrics.png"
        save_train_vs_validation_plot(
            train_metrics=train_validation_metrics[name]["train"],
            validation_metrics=train_validation_metrics[name]["validation"],
            output_path=train_validation_path,
            title=f"Train vs Validation Metrics - {name}",
        )
        record_output_path(train_validation_path, generated_output_files)

        feature_importance_path = model_plots_dir / "feature_importance.png"
        if save_feature_importance_plot(
            trained_pipeline=pipeline,
            output_path=feature_importance_path,
            title=f"Top Feature Importance - {name}",
        ):
            record_output_path(feature_importance_path, generated_output_files)

    ranked_results = sorted(results, key=lambda row: (row.pr_auc, row.roc_auc), reverse=True)
    best_model = ranked_results[0]

    comparison_plot_path = plots_dir / "model_validation_comparison.png"
    save_metric_comparison_plot(
        model_metrics=build_validation_metrics_map([asdict(row) for row in ranked_results]),
        output_path=comparison_plot_path,
        title="Validation Metrics Comparison (Credit Card Models)",
    )
    record_output_path(comparison_plot_path, generated_output_files)

    record_output_path(output_path, generated_output_files)

    payload = {
        "problem": "credit_card_default",
        "dataset": str(data_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "target_column": target_col,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {
            "non_default_0": int((y == 0).sum()),
            "default_1": int((y == 1).sum()),
            "default_rate": float((y == 1).mean()),
        },
        "results": [asdict(r) for r in ranked_results],
        "best_model": asdict(best_model),
        "classification_reports": reports,
        "train_validation_metrics": train_validation_metrics,
        "output_root": str(results_dir),
        "plots_dir": str(plots_dir),
        "predictions_dir": str(predictions_dir),
        "generated_files": sorted(set(generated_output_files)),
        "tuned_hyperparameters_in_script": {
            "logistic_regression": {
                "C": 0.8,
                "max_iter": 3000,
                "class_weight": "balanced",
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_leaf": 10,
                "min_samples_split": 20,
                "class_weight": "balanced_subsample",
            },
            "xgboost": {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 5,
                "reg_lambda": 1.5,
            },
        },
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


if __name__ == "__main__":
    metrics = run_card_pipeline()
    print("Card-default experiment complete.")
    print(f"Best model: {metrics['best_model']['name']}")
    print(f"Metrics written to: {DEFAULT_OUTPUT_PATH}")
