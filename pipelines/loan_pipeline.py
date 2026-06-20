from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


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
    if "pub_rec" in df.columns:
        df["has_pub_rec"] = (df["pub_rec"] > 0).astype(int)
    if "dti" in df.columns:
        df["dti_bin"] = pd.qcut(df["dti"].rank(method="first"), q=5, labels=False, duplicates="drop")
    if "int_rate" in df.columns:
        df["int_rate_bin"] = pd.qcut(df["int_rate"].rank(method="first"), q=5, labels=False, duplicates="drop")
    if "fico_avg" in df.columns:
        df["fico_score_bucket"] = pd.qcut(df["fico_avg"].rank(method="first"), q=5, labels=False, duplicates="drop")
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

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=40,
            min_child_samples=20,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
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


def _apply_isotonic_calibration(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit model on 80% of training data, calibrate isotonic on 20%, return calibrated test probs."""
    X_sub, X_cal, y_sub, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    model.fit(X_sub, y_sub)
    cal_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_probs, y_cal)
    return iso.transform(model.predict_proba(X_test)[:, 1])


def _compute_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """Mutual Information scores for all numeric features (metric only, no plot)."""
    numeric = X_train.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.empty or numeric.shape[1] < 2:
        return {}
    numeric = numeric.fillna(numeric.median())
    mi = mutual_info_classif(numeric, y_train, random_state=SEED)
    return dict(sorted(zip(numeric.columns, mi), key=lambda x: -x[1]))


def _compute_cost_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_fn: float = 12_000.0,
    cost_fp: float = 100.0,
    avg_loan: float = 15_000.0,
    recovery_rate: float = 0.20,
) -> dict[str, Any]:
    """Cost-sensitive threshold optimization + business impact.

    Scans thresholds 0.01–0.99 to find minimum-cost operating point given
    asymmetric FN:FP costs (default FN penalty = LGD ≈ 80% of avg loan).
    """
    loss_per_default = avg_loan * (1 - recovery_rate)
    thresholds = np.linspace(0.01, 0.99, 99)

    best_thresh = 0.5
    min_cost = float("inf")
    cost_curve = []

    for t in thresholds:
        y_bin = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_bin)
        tn, fp, fn, tp = cm.ravel()
        total = fn * loss_per_default + fp * cost_fp
        cost_curve.append({"threshold": round(t, 3), "total_cost": total, "fn": int(fn), "fp": int(fp)})
        if total < min_cost:
            min_cost = total
            best_thresh = t

    n = len(y_true)
    n_defaults = int(y_true.sum())
    loss_no_model = n_defaults * loss_per_default
    y_opt = (y_prob >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y_true, y_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    loss_with_model = fn_opt * loss_per_default + fp_opt * cost_fp
    savings = loss_no_model - loss_with_model
    savings_pct = (savings / loss_no_model * 100) if loss_no_model > 0 else 0.0
    roi = (savings / (fp_opt * cost_fp)) if fp_opt > 0 else float("inf")

    return {
        "optimal_threshold": round(float(best_thresh), 4),
        "min_cost": float(min_cost),
        "loss_no_model": float(loss_no_model),
        "loss_with_model": float(loss_with_model),
        "savings": float(savings),
        "savings_pct": round(float(savings_pct), 2),
        "roi": round(float(roi), 2),
        "fn_at_optimal": int(fn_opt),
        "fp_at_optimal": int(fp_opt),
        "tp_at_optimal": int(tp_opt),
        "tn_at_optimal": int(tn_opt),
        "cost_curve": cost_curve,
    }


def run_loan_pipeline(
    accepted_path: Path = DEFAULT_ACCEPTED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_rows: int = 180_000,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    output_path = Path(output_path)
    results_dir = output_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, source_type = load_loan_data(
        accepted_path=accepted_path,
        max_rows=max_rows,
        chunksize=chunksize,
    )

    default_rate = float(y.mean())
    logger.info(
        "Loaded %s rows (%s) — default rate: %.2f%% — from %s",
        len(X), source_type, 100.0 * default_rate,
        accepted_path if accepted_path.exists() else DEFAULT_FALLBACK_PATH,
    )

    engineered_features = [
        "loan_to_income", "payment_to_income", "fico_avg",
        "has_delinq", "has_bankruptcy", "has_pub_rec",
        "acc_open_ratio", "inquiry_per_acc",
        "dti_bin", "int_rate_bin", "fico_score_bucket",
    ]
    present_engineered = [c for c in engineered_features if c in X.columns]
    if present_engineered:
        logger.info("Engineered %d features: %s", len(present_engineered), present_engineered)

    logger.info("Computing Information Value (IV) for engineered features...")
    information_values: dict[str, float] = {}
    for feat in present_engineered:
        iv = calculate_information_value(X, y, feat)
        information_values[feat] = iv
        logger.info("  %s: IV = %.4f %s", feat, iv,
                    "— useful predictor (>0.1)" if iv > 0.1 else "— weak or negligible")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

    preprocessor = build_preprocessor(X)

    # --- Mutual information (metric only) ---
    logger.info("Computing mutual information...")
    mi_scores = _compute_mutual_information(X_train, y_train)

    results: list[ModelResult] = []
    calibrated_results: list[ModelResult] = []
    reports: dict[str, dict[str, Any]] = {}
    train_validation_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for name, estimator in make_models(y_train).items():
        logger.info("Training %s... (n_train=%d, default_rate=%.2f%%)",
                     name, len(X_train), 100.0 * y_train.mean())
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        y_train_prob = pipeline.predict_proba(X_train)[:, 1]
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        train_metrics = evaluate_predictions(y_train, y_train_prob)
        val_metrics = evaluate_predictions(y_test, y_prob)
        train_validation_metrics[name] = {"train": train_metrics, "validation": val_metrics}

        results.append(ModelResult(name=name, **val_metrics))
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

        logger.info("  Raw → ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | KS: %.4f",
                     val_metrics["roc_auc"], val_metrics["pr_auc"],
                     val_metrics["brier_score"], val_metrics["ks_statistic"])

        # --- Isotonic calibration ---
        logger.info("  Calibrating %s with isotonic regression...", name)
        y_prob_cal = _apply_isotonic_calibration(estimator, X_train, y_train, X_test)
        cal_metrics = evaluate_predictions(y_test, y_prob_cal)
        calibrated_results.append(ModelResult(name=f"{name}_calibrated", **cal_metrics))

        logger.info("  Cal → ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | KS: %.4f",
                     cal_metrics["roc_auc"], cal_metrics["pr_auc"],
                     cal_metrics["brier_score"], cal_metrics["ks_statistic"])

    # --- Rank raw models ---
    ranked_raw = sorted(results, key=lambda r: (r.pr_auc, r.roc_auc), reverse=True)
    best_raw = ranked_raw[0]

    # --- Rank calibrated models ---
    ranked_cal = sorted(calibrated_results, key=lambda r: (r.pr_auc, r.roc_auc), reverse=True)
    best_cal = ranked_cal[0]

    logger.info("Best raw model:      %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
                best_raw.name, best_raw.pr_auc, best_raw.roc_auc)
    logger.info("Best calibrated model: %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
                best_cal.name, best_cal.pr_auc, best_cal.roc_auc)

    # --- Cost-sensitive threshold analysis on best calibrated model ---
    best_cal_name = best_cal.name.replace("_calibrated", "")
    logger.info("Computing cost-sensitive metrics for best model: %s", best_cal_name)
    best_cal_idx = next(i for i, r in enumerate(calibrated_results) if r.name == best_cal.name)
    y_best_prob = None
    # Re-run calibration for best model to get final probs for cost analysis
    for name, estimator in make_models(y_train).items():
        if name == best_cal_name:
            y_best_prob = _apply_isotonic_calibration(estimator, X_train, y_train, X_test)
            break

    cost_metrics = _compute_cost_metrics(y_test, y_best_prob) if y_best_prob is not None else {}

    payload = {
        "problem": "lendingclub_loan_default",
        "dataset": str(accepted_path if accepted_path.exists() else DEFAULT_FALLBACK_PATH),
        "data_source_mode": source_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "max_rows_requested": max_rows,
        "raw_model_count": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {
            "non_default_0": int((y == 0).sum()),
            "default_1": int((y == 1).sum()),
            "default_rate": round(default_rate, 4),
        },
        "mutual_information_top_20": dict(list(mi_scores.items())[:20]),
        "information_values": {
            feat: round(iv, 6) for feat, iv in sorted(information_values.items(), key=lambda x: -x[1])
        },
        "raw_results": [asdict(r) for r in ranked_raw],
        "calibrated_results": [asdict(r) for r in ranked_cal],
        "best_raw_model": asdict(best_raw),
        "best_calibrated_model": asdict(best_cal),
        "cost_analysis": cost_metrics,
        "classification_reports": reports,
        "train_validation_metrics": train_validation_metrics,
        "engineered_features": present_engineered,
        "selected_feature_count": int(X.shape[1]),
        "selected_features": list(X.columns),
        "tuned_hyperparameters_in_script": {
            "logistic_regression": {"C": 1.0, "max_iter": 3000, "class_weight": "balanced"},
            "random_forest": {"n_estimators": 350, "max_depth": 14, "min_samples_split": 40,
                              "min_samples_leaf": 20, "max_features": "sqrt",
                              "class_weight": "balanced_subsample"},
            "xgboost": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
                        "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 5,
                        "reg_lambda": 2.0, "objective": "binary:logistic", "eval_metric": "aucpr"},
            "lightgbm": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                         "num_leaves": 40, "min_child_samples": 20, "class_weight": "balanced"},
        },
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    return payload


if __name__ == "__main__":
    metrics = run_loan_pipeline()
    print("Loan-default experiment complete.")
    print(f"Best model: {metrics['best_model']['name']}")
    print(f"Metrics written to: {DEFAULT_OUTPUT_PATH}")
