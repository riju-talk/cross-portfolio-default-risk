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
DEFAULT_ACCEPTED_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv")
DEFAULT_FALLBACK_PATH = Path("data/processed/loan_sample_processed.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/loan_metrics.json")

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
NON_DEFAULT_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
TERMINAL_STATUSES = DEFAULT_STATUSES | NON_DEFAULT_STATUSES

LOAN_FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]


@dataclass
class ModelResult:
    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    top_decile_capture_rate: float


def _parse_percent_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def _top_decile_capture(y_true: pd.Series, y_prob: np.ndarray) -> float:
    top_n = max(1, int(np.ceil(0.10 * len(y_prob))))
    ranked_indices = np.argsort(y_prob)[::-1][:top_n]
    captured = int(y_true.iloc[ranked_indices].sum())
    positives = int(y_true.sum())
    return float(captured / positives) if positives > 0 else 0.0


def _clean_loan_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_hint_columns = {
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "fico_range_low",
        "fico_range_high",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
    }

    if "int_rate" in df.columns:
        df["int_rate"] = _parse_percent_series(df["int_rate"])
    if "revol_util" in df.columns:
        df["revol_util"] = _parse_percent_series(df["revol_util"])

    for col in numeric_hint_columns:
        if col in df.columns and col not in {"int_rate", "revol_util"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_null_columns = [col for col in df.columns if col != "target" and df[col].isna().all()]
    if all_null_columns:
        df = df.drop(columns=all_null_columns)

    return df


def load_loan_data(
    accepted_path: Path,
    max_rows: int = 180_000,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.Series, str]:
    if accepted_path.exists():
        required_columns = {"loan_status", *LOAN_FEATURE_COLUMNS}

        frames: list[pd.DataFrame] = []
        collected_rows = 0

        for chunk in pd.read_csv(
            accepted_path,
            usecols=lambda c: c in required_columns,
            chunksize=chunksize,
            low_memory=False,
        ):
            if "loan_status" not in chunk.columns:
                continue

            chunk = chunk[chunk["loan_status"].isin(TERMINAL_STATUSES)].copy()
            if chunk.empty:
                continue

            chunk["target"] = chunk["loan_status"].apply(
                lambda status: 1 if status in DEFAULT_STATUSES else 0
            )
            chunk = chunk.drop(columns=["loan_status"])
            chunk = _clean_loan_frame(chunk)

            frames.append(chunk)
            collected_rows += len(chunk)
            if collected_rows >= max_rows:
                break

        if not frames:
            raise ValueError(
                "No terminal-status rows found in accepted loans. "
                "Check that the raw LendingClub CSV is valid."
            )

        df = pd.concat(frames, ignore_index=True).head(max_rows)
        y = df.pop("target").astype(int)
        return df, y, "accepted_loans_chunked"

    if DEFAULT_FALLBACK_PATH.exists():
        fallback_df = pd.read_csv(DEFAULT_FALLBACK_PATH)
        if "target" not in fallback_df.columns:
            raise ValueError(
                f"Fallback dataset exists at {DEFAULT_FALLBACK_PATH} but has no 'target' column."
            )
        fallback_df = _clean_loan_frame(fallback_df)
        y = fallback_df.pop("target").astype(int)
        return fallback_df, y, "processed_fallback"

    raise FileNotFoundError(
        "No loan dataset found. Expected either raw accepted loans file or "
        f"fallback file at {DEFAULT_FALLBACK_PATH}."
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [
        col
        for col in features.columns
        if pd.api.types.is_object_dtype(features[col])
        or pd.api.types.is_string_dtype(features[col])
        or pd.api.types.is_categorical_dtype(features[col])
    ]
    numeric_features = [col for col in features.columns if col not in categorical_features]

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


def make_models(y_train: pd.Series) -> dict[str, Any]:
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))
    scale_pos_weight = negative / positive

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=40,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
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


def run_loan_pipeline(
    accepted_path: Path = DEFAULT_ACCEPTED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_rows: int = 180_000,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    X, y, source_type = load_loan_data(
        accepted_path=accepted_path,
        max_rows=max_rows,
        chunksize=chunksize,
    )

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

    for name, estimator in make_models(y_train).items():
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
        "problem": "lendingclub_loan_default",
        "dataset": str(accepted_path if accepted_path.exists() else DEFAULT_FALLBACK_PATH),
        "data_source_mode": source_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "max_rows_requested": max_rows,
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
        "selected_feature_count": int(X.shape[1]),
        "selected_features": list(X.columns),
        "tuned_hyperparameters_in_script": {
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 3000,
                "class_weight": "balanced",
            },
            "random_forest": {
                "n_estimators": 350,
                "max_depth": 14,
                "min_samples_split": 40,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_lambda": 2.0,
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


if __name__ == "__main__":
    metrics = run_loan_pipeline()
    print("Loan-default experiment complete.")
    print(f"Best model: {metrics['best_model']['name']}")
    print(f"Metrics written to: {DEFAULT_OUTPUT_PATH}")
