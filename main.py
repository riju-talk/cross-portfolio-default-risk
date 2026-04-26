from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
except Exception:  # optional dependency in some environments
    XGBClassifier = None


SEED = 42
DEFAULT_DATA_PATH = Path("data/default_of_credit_card_clients.csv")
OUTPUT_DIR = Path("artifacts")


@dataclass
class ModelResult:
    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    top_decile_capture_rate: float


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find data file at {csv_path}. Run data download scripts first."
        )

    df = pd.read_csv(csv_path)

    # common names across UCI export variants
    target_candidates = [
        "Y",
        "default payment next month",
        "default.payment.next.month",
        "default_payment_next_month",
    ]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("No recognized target column found in dataset.")

    # ID is non-informative and can leak applicant identity
    feature_df = df.drop(columns=[target_col] + (["ID"] if "ID" in df.columns else []))
    target = df[target_col].astype(int)
    return feature_df, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical_features = [c for c in features.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
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
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
        )

    return models


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)

    # percentage of true defaulters captured in highest-risk decile
    top_n = max(1, int(0.10 * len(y_prob)))
    ranked_indices = np.argsort(y_prob)[::-1][:top_n]
    captured_positives = y_true.iloc[ranked_indices].sum()
    total_positives = y_true.sum()
    capture_rate = float(captured_positives / total_positives) if total_positives > 0 else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "top_decile_capture_rate": capture_rate,
    }


def run_experiment(data_path: Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    X, y = load_data(data_path)
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
                ("prep", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_predictions(y_test, y_prob)
        results.append(ModelResult(name=name, **metrics))
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

    ranked = sorted(results, key=lambda x: (x.pr_auc, x.roc_auc), reverse=True)
    best = ranked[0]

    payload = {
        "dataset": str(data_path),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            "non_default_0": int((y == 0).sum()),
            "default_1": int((y == 1).sum()),
            "default_rate": float((y == 1).mean()),
        },
        "results": [asdict(r) for r in ranked],
        "best_model": asdict(best),
        "classification_reports": reports,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "metrics.json"
    out_file.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    payload = run_experiment()
    print("Loan default prediction experiment complete.")
    print(f"Metrics written to: {OUTPUT_DIR / 'metrics.json'}")
    print("Best model:", payload["best_model"])


if __name__ == "__main__":
    main()
