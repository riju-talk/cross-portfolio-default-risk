from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

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
DEFAULT_ACCEPTED_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv")
DEFAULT_FALLBACK_PATH = Path("data/processed/loan_sample_processed.csv")
DEFAULT_RESULTS_DIR = Path("results/loan")
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "metrics.json"

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
    """Container for model evaluation metrics.

    Brier score measures probability calibration (lower is better) — critical in
    credit risk where predicted probabilities must reflect true default rates for
    regulatory capital computation and pricing decisions.

    KS statistic quantifies the maximum separability between default and
    non-default score distributions — the industry-standard measure of
    discriminative power in credit scoring (higher = better separation).

    Calibration error (ECE) measures systematic bias in probability estimates
    across the risk spectrum — essential for validating that model outputs are
    trustworthy for risk-tier pricing.
    """
    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    top_decile_capture_rate: float
    brier_score: float = 0.0
    ks_statistic: float = 0.0
    calibration_error: float = 0.0


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


def _ks_statistic(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for credit scoring discrimination.

    Measures the maximum separation between the cumulative distribution
    functions of predicted scores for defaulters vs non-defaulters. In
    credit scoring, a KS > 0.3 is considered acceptable, > 0.5 excellent.
    """
    y = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(prob)
    y_sorted = y[order]
    n_neg = int((y_sorted == 0).sum())
    n_pos = int((y_sorted == 1).sum())
    if n_neg == 0 or n_pos == 0:
        return 0.0
    cum_neg = np.cumsum(y_sorted == 0) / n_neg
    cum_pos = np.cumsum(y_sorted == 1) / n_pos
    return float(np.max(np.abs(cum_neg - cum_pos)))


def _calibration_error(y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (ECE) — mean absolute deviation between
    predicted probabilities and observed event rates across bins.

    Low calibration error means the model's probability outputs are
    trustworthy for risk-tier pricing and loan-loss provisioning.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return float(np.mean(np.abs(frac_pos - mean_pred)))


def _engineer_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-relevant features for credit risk modeling.

    Rationale:
      - loan_to_income / payment_to_income: debt burden ratios — primary
        drivers of default probability in consumer lending.
      - fico_avg: single-point credit score estimate from LendingClub's
        reported range (low/high).
      - has_delinq / has_bankruptcy: binary flags capture non-linear default
        risk jumps that continuous counts may miss.
      - acc_open_ratio: credit utilization breadth — thin-file borrowers
        behave differently.
      - inquiry_per_acc: recent credit-seeking intensity, a well-known
        delinquency predictor.
    """
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["payment_to_income"] = (df["installment"] * 12) / (df["annual_inc"] + 1)
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    if "delinq_2yrs" in df.columns:
        df["has_delinq"] = (df["delinq_2yrs"] > 0).astype(int)
    if "pub_rec_bankruptcies" in df.columns:
        df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["acc_open_ratio"] = df["open_acc"] / (df["total_acc"] + 1)
    if "inq_last_6mths" in df.columns and "open_acc" in df.columns:
        df["inquiry_per_acc"] = df["inq_last_6mths"] / (df["open_acc"] + 1)
    return df


def calculate_information_value(
    df: pd.DataFrame,
    target: pd.Series,
    feature: str,
    n_bins: int = 10,
) -> float:
    """Weight of Evidence (WoE) Information Value for a single feature.

    IV is the standard variable-selection metric in credit scoring:
      < 0.02  — not useful
      0.02–0.1 — weak predictor
      0.1–0.3  — medium predictor
      > 0.3    — strong predictor

    Uses quantile-based binning to handle skewed distributions common in
    loan features (e.g. income, loan amount).
    """
    data = pd.DataFrame({"feature": df[feature], "target": target}).dropna()
    if data.empty or data["target"].nunique() < 2:
        return 0.0
    n_events = data["target"].sum()
    n_nonevents = len(data) - n_events
    if n_events == 0 or n_nonevents == 0:
        return 0.0
    try:
        data["bin"] = pd.qcut(data["feature"].rank(method="first"), q=n_bins, duplicates="drop")
    except ValueError:
        return 0.0
    grouped = data.groupby("bin", observed=False)["target"].agg(
        events="sum",
        count="count",
    )
    grouped["non_events"] = grouped["count"] - grouped["events"]
    grouped["event_rate"] = grouped["events"] / n_events
    grouped["non_event_rate"] = grouped["non_events"] / n_nonevents
    grouped["woe"] = np.log(
        np.clip(grouped["event_rate"] / grouped["non_event_rate"], 1e-10, 1e10)
    )
    grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]
    return float(grouped["iv"].sum())


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

    df = _engineer_loan_features(df)

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
    """Compute classification, ranking, and calibration metrics.

    Returns both ranking metrics (ROC-AUC, PR-AUC) and calibration metrics
    (Brier score, ECE) because a well-calibrated probability is essential in
    credit risk — regulators and pricing models need accurate probabilities,
    not just correct rank-ordering.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "top_decile_capture_rate": _top_decile_capture(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ks_statistic": _ks_statistic(y_true, y_prob),
        "calibration_error": _calibration_error(y_true, y_prob),
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
        title="LendingClub Default Class Distribution",
    )
    record_output_path(target_plot_path, generated_output_files)

    missingness_plot_path = dataset_plot_dir / "missing_values.png"
    if save_missingness_plot(
        features=X,
        output_path=missingness_plot_path,
        title="Top Missing-Value Rates (LendingClub Features)",
    ):
        record_output_path(missingness_plot_path, generated_output_files)

    numeric_grid_path = dataset_plot_dir / "numeric_feature_distributions.png"
    if save_numeric_feature_grid(
        features=X,
        output_path=numeric_grid_path,
        title="Representative Numeric Feature Distributions",
    ):
        record_output_path(numeric_grid_path, generated_output_files)


def run_loan_pipeline(
    accepted_path: Path = DEFAULT_ACCEPTED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_rows: int = 180_000,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    output_path = Path(output_path)
    results_dir = output_path.parent
    plots_dir = results_dir / "plots"
    predictions_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    X, y, source_type = load_loan_data(
        accepted_path=accepted_path,
        max_rows=max_rows,
        chunksize=chunksize,
    )

    logger.info(
        "Loaded %s rows (%s) — default rate: %.2f%% — from %s",
        len(X), source_type, 100.0 * y.mean(), accepted_path if accepted_path.exists() else DEFAULT_FALLBACK_PATH,
    )

    engineered_features = [
        "loan_to_income", "payment_to_income", "fico_avg",
        "has_delinq", "has_bankruptcy", "acc_open_ratio", "inquiry_per_acc",
    ]
    present_engineered = [c for c in engineered_features if c in X.columns]
    if present_engineered:
        logger.info(
            "Engineered %d features: %s — these capture debt burden, credit quality, "
            "and credit-seeking intensity beyond raw LendingClub fields.",
            len(present_engineered), present_engineered,
        )

    logger.info(
        "Computing Information Value (IV) for each engineered feature — "
        "IV > 0.1 indicates medium predictive power, > 0.3 is strong."
    )
    information_values: dict[str, float] = {}
    for feat in present_engineered:
        iv = calculate_information_value(X, y, feat)
        information_values[feat] = iv
        if iv > 0.1:
            logger.info("  %s: IV = %.4f — useful predictor", feat, iv)
        else:
            logger.info("  %s: IV = %.4f — weak or negligible", feat, iv)

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

    for name, estimator in make_models(y_train).items():
        logger.info("Training %s... (n_train=%d, default_rate=%.2f%%)", name, len(X_train), 100.0 * y_train.mean())
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
        title="Validation Metrics Comparison (LendingClub Models)",
    )
    record_output_path(comparison_plot_path, generated_output_files)

    record_output_path(output_path, generated_output_files)

    logger.info(
        "Best model: %s (PR-AUC=%.4f, KS=%.4f, Brier=%.4f) — "
        "KS > 0.3 is acceptable in credit scoring, Brier < 0.1 indicates good calibration.",
        best_model.name, best_model.pr_auc, best_model.ks_statistic, best_model.brier_score,
    )

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
        "train_validation_metrics": train_validation_metrics,
        "engineered_features": present_engineered,
        "information_values": {
            feat: round(iv, 6) for feat, iv in sorted(information_values.items(), key=lambda x: -x[1])
        },
        "output_root": str(results_dir),
        "plots_dir": str(plots_dir),
        "predictions_dir": str(predictions_dir),
        "generated_files": sorted(set(generated_output_files)),
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

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


if __name__ == "__main__":
    metrics = run_loan_pipeline()
    print("Loan-default experiment complete.")
    print(f"Best model: {metrics['best_model']['name']}")
    print(f"Metrics written to: {DEFAULT_OUTPUT_PATH}")
