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


SEED = 42
DEFAULT_DATA_PATH = Path("data/default_of_credit_card_clients.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/card_metrics.json")
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
        if c in CARD_CATEGORICAL_COLUMNS or pd.api.types.is_object_dtype(features[c])
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


def run_card_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
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

    for name, estimator in make_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_predictions(y_test, y_prob)
        results.append(ModelResult(name=name, **metrics))
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

    ranked_results = sorted(results, key=lambda row: (row.pr_auc, row.roc_auc), reverse=True)
    best_model = ranked_results[0]

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


if __name__ == "__main__":
    metrics = run_card_pipeline()
    print("Card-default experiment complete.")
    print(f"Best model: {metrics['best_model']['name']}")
    print(f"Metrics written to: {DEFAULT_OUTPUT_PATH}")
