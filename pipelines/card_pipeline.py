from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.calibration import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


SEED = 42
DEFAULT_DATA_PATH = Path("data/default_of_credit_card_clients.csv")
DEFAULT_RESULTS_DIR = Path("results/card")
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "metrics.json"
CARD_CATEGORICAL_COLUMNS = {"SEX", "EDUCATION", "MARRIAGE"}

# PAY columns indicating delayed payment (>0 means delayed)
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


@dataclass
class ModelResult:
    """Evaluation results for a single model.

    Captures rank-based (ROC-AUC, PR-AUC), threshold-based (precision,
    recall, F1), business-value (top-decile capture), and calibration
    (Brier, KS) metrics.  In credit risk, calibration error and KS
    are as important as AUC because a well-separated, well-calibrated
    score supports sound lending decisions.
    """

    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    top_decile_capture_rate: float
    brier_score: float
    ks_statistic: float
    calibration_error: float


def _top_decile_capture(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Fraction of all positives captured in the highest-risk decile.

    A standard business metric for credit scoring: if we rank applicants
    by predicted default probability and approve only the top 10 %, how
    many of the eventual defaults do we correctly flag?
    """
    top_n = max(1, int(np.ceil(0.10 * len(y_prob))))
    ranked_indices = np.argsort(y_prob)[::-1][:top_n]
    captured = int(y_true.iloc[ranked_indices].sum())
    positives = int(y_true.sum())
    return float(captured / positives) if positives > 0 else 0.0


def _compute_ks_statistic(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for credit score separation.

    KS measures the maximum difference between the cumulative
    distribution of predicted probabilities for defaulters vs.
    non-defaulters.  In credit risk, KS > 0.3 is generally
    considered good separation; values above 0.5 indicate strong
    discriminative power.  Calculated as the D-statistic from a
    two-sample KS test.
    """
    scores_default = y_prob[y_true == 1]
    scores_non_default = y_prob[y_true == 0]
    if len(scores_default) == 0 or len(scores_non_default) == 0:
        return 0.0
    d_stat, _ = stats.ks_2samp(scores_non_default, scores_default)
    return float(d_stat)


def _compute_information_values(
    X: pd.DataFrame, y: pd.Series, bins: int = 10
) -> pd.Series:
    """Information Value (IV) for each feature — standard in credit scoring.

    IV quantifies the predictive power of a feature by measuring the
    divergence between the distributions of defaulters and non-defaulters
    across binned feature values.

    In the credit industry:
        IV < 0.02   → not useful
        0.02–0.10   → weak
        0.10–0.30   → medium
        0.30–0.50   → strong
        IV > 0.50   → suspicious (possible overfit / too predictive)

    Note: This implementation bins numeric features into deciles for a
    weight-of-evidence style calculation.  It works best with at least
    a few hundred rows.
    """
    iv_values: dict[str, float] = {}
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return pd.Series(dtype=float)

    for col in X.columns:
        col_data = X[col].dropna()
        # align
        mask = col_data.index
        col_y = y.loc[mask]

        n_unique = col_data.nunique()
        if n_unique <= 1:
            iv_values[col] = 0.0
            continue

        if pd.api.types.is_numeric_dtype(col_data) and n_unique > bins:
            # discretize into decile bins
            col_binned = pd.qcut(col_data, q=bins, duplicates="drop")
        else:
            col_binned = col_data.astype(str)

        grouped = pd.DataFrame({"col": col_binned, "y": col_y})
        woe_table = grouped.groupby("col", observed=True)["y"].agg(
            n_events="sum",  # number of 1's (default)
            count="count",
        )
        woe_table["n_non_events"] = woe_table["count"] - woe_table["n_events"]
        woe_table["pct_events"] = woe_table["n_events"] / n_pos
        woe_table["pct_non_events"] = woe_table["n_non_events"] / n_neg

        # avoid log(0) with small smoothing
        eps = 1e-10
        woe_table["woe"] = np.log(
            (woe_table["pct_events"] + eps) / (woe_table["pct_non_events"] + eps)
        )
        woe_table["iv_contrib"] = (
            woe_table["pct_events"] - woe_table["pct_non_events"]
        ) * woe_table["woe"]
        iv_values[col] = float(woe_table["iv_contrib"].sum())

    return pd.Series(iv_values).sort_values(ascending=False)


def _engineer_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 12 domain-driven features replicating Yeh & Lien (2009) feature engineering.

    Adds utilization ratios, payment ratios, delinquency features, averages, and trends.
    """
    # --- Utilization ratios: BILL_AMT / LIMIT_BAL per month ---
    if "LIMIT_BAL" in df.columns:
        for i, col in enumerate(BILL_COLS, 1):
            if col in df.columns:
                df[f"UTIL_{i}"] = df[col] / (df["LIMIT_BAL"] + 1)
        # Average utilization
        util_cols = [c for c in df.columns if c.startswith("UTIL_") and c != "UTIL_DECILE"]
        if util_cols:
            df["AVG_UTIL"] = df[util_cols].mean(axis=1)

    # --- Payment ratios: PAY_AMT / BILL_AMT per month ---
    for i, (pay_col, bill_col) in enumerate(zip(PAY_AMT_COLS, BILL_COLS), 1):
        if pay_col in df.columns and bill_col in df.columns:
            df[f"PAY_RATIO_{i}"] = df[pay_col] / (df[bill_col] + 1)

    # --- Delinquency features ---
    present_pay = [c for c in PAY_COLS if c in df.columns]
    if present_pay:
        df["NUM_DELAYED"] = (df[present_pay] > 0).sum(axis=1)
        df["MAX_DELAY"] = df[present_pay].max(axis=1)
        # Recency: most recent non-zero PAY value
        delayed_mask = df[present_pay] > 0
        df["RECENCY_DELAY"] = df[present_pay].where(delayed_mask).bfill(axis=1).iloc[:, 0].fillna(0)

    # --- Averages ---
    present_bill = [c for c in BILL_COLS if c in df.columns]
    if present_bill:
        df["AVG_BILL_AMT"] = df[present_bill].mean(axis=1)
    present_pay_amt = [c for c in PAY_AMT_COLS if c in df.columns]
    if present_pay_amt:
        df["AVG_PAY_AMT"] = df[present_pay_amt].mean(axis=1)

    pay_ratio_cols = [c for c in df.columns if c.startswith("PAY_RATIO_")]
    if pay_ratio_cols:
        df["AVG_PAY_RATIO"] = df[pay_ratio_cols].mean(axis=1)

    # --- Log transform of credit limit ---
    if "LIMIT_BAL" in df.columns:
        df["LIMIT_BAL_LOG"] = np.log1p(df["LIMIT_BAL"])

    # --- Trends (linear slope over 6 months) ---
    def _slope(series_row: pd.Series) -> float:
        vals = series_row.values.astype(float)
        x = np.arange(len(vals))
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            return 0.0
        return float(np.polyfit(x[mask], vals[mask], 1)[0])

    if present_bill:
        df["BILL_TREND"] = df[present_bill].apply(_slope, axis=1)
    if present_pay_amt:
        df["PAY_TREND"] = df[present_pay_amt].apply(_slope, axis=1)

    # --- Binning ---
    if "AGE" in df.columns:
        df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(float)
    if "LIMIT_BAL" in df.columns:
        df["LIMIT_BIN"] = pd.qcut(df["LIMIT_BAL"].rank(method="first"), q=5, labels=False, duplicates="drop")
    if "AVG_UTIL" in df.columns:
        df["UTIL_DECILE"] = pd.qcut(df["AVG_UTIL"].rank(method="first"), q=10, labels=False, duplicates="drop")
        df["UTIL_BIN"] = pd.qcut(df["AVG_UTIL"].rank(method="first"), q=5, labels=False, duplicates="drop")

    return df


def load_card_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the UCI credit-card default dataset and split features/target.

    The dataset records whether a Taiwanese credit-card holder defaulted
    in October 2005, along with demographic attributes, payment history,
    and bill statement amounts.  The default rate is ~22 %, making this
    an imbalanced classification problem where calibration and separation
    metrics are especially important.
    """
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
    features = _engineer_card_features(features)
    return features, target, target_col


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the credit-card dataset.

    Numeric columns are median-imputed and standard-scaled (robust to
    outliers in bill amounts).  Categorical columns (SEX, EDUCATION,
    MARRIAGE, AGE_GROUP, UTIL_DECILE, UTIL_BIN, LIMIT_BIN) are
    mode-imputed and one-hot encoded.  Using median rather than mean
    avoids distortion from the heavy right tail in bill-amount distributions.
    """
    known_categorical = CARD_CATEGORICAL_COLUMNS | {"AGE_GROUP", "LIMIT_BIN", "UTIL_DECILE", "UTIL_BIN"}
    categorical_features = [
        c
        for c in features.columns
        if c in known_categorical
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
    """Instantiate 8 candidate models for credit-default prediction.

    Replicates the Yeh & Lien (2009) benchmark zoo — LR, LDA, GNB,
    k-NN, DT, RF, XGB, MLP — all with class-weight balancing where
    available to handle the ~22 % default rate.
    """
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=0.8, max_iter=3000, class_weight="balanced", random_state=SEED,
        ),
        "lda": LinearDiscriminantAnalysis(),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10, min_samples_split=20,
            class_weight="balanced", random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            min_samples_split=20, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
            random_state=SEED,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
            reg_lambda=1.5, objective="binary:logistic", eval_metric="aucpr",
            random_state=SEED, n_jobs=-1,
        )

    return models


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Compute a comprehensive set of evaluation metrics.

    Returns both rank-based metrics (ROC-AUC, PR-AUC) and calibration/
    separation metrics (Brier, KS, calibration error) so the data
    scientist can assess *discrimination* and *calibration* separately.
    In credit risk, a model with high AUC can still be poorly calibrated,
    leading to mis-priced loans or regulatory disapproval.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    brier = brier_score_loss(y_true, y_prob)
    ks = _compute_ks_statistic(y_true, y_prob)
    calib_error = abs(float(y_prob.mean()) - float(y_true.mean()))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "top_decile_capture_rate": _top_decile_capture(y_true, y_prob),
        "brier_score": float(brier),
        "ks_statistic": float(ks),
        "calibration_error": float(calib_error),
    }


def _apply_isotonic_calibration(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit model on 80% of training data, isotonic calibrate on 20%, return calibrated probs."""
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


def run_card_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run the full credit-card default prediction pipeline.

    INGEST → PREPROCESS (12 engineered features) → TRAIN (8 models)
    → EVALUATE (raw + calibrated metrics + cost analysis) → SAVE_OUTPUTS.

    No figures — only metrics.
    """
    output_path = Path(output_path)
    results_dir = output_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s ...", data_path)
    X, y, target_col = load_card_data(data_path)
    default_rate = float(y.mean())
    logger.info("Dataset loaded: %d rows, %d features, default rate = %.2f%%",
                len(X), X.shape[1], default_rate * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

    preprocessor = build_preprocessor(X)

    # --- Mutual information (metric only) ---
    logger.info("Computing mutual information...")
    mi_scores = _compute_mutual_information(X_train, y_train)

    # --- Information values ---
    iv_scores = _compute_information_values(X, y)
    top_iv_features = iv_scores.head(15).to_dict()

    results_raw: list[ModelResult] = []
    results_cal: list[ModelResult] = []
    reports: dict[str, dict[str, Any]] = {}
    train_validation_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for name, estimator in make_models().items():
        logger.info("Training %s ...", name)
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
        results_raw.append(ModelResult(name=name, **val_metrics))
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

        logger.info("  Raw  → ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | KS: %.4f",
                    val_metrics["roc_auc"], val_metrics["pr_auc"],
                    val_metrics["brier_score"], val_metrics["ks_statistic"])

        # --- Isotonic calibration ---
        logger.info("  Calibrating %s with isotonic regression...", name)
        try:
            y_prob_cal = _apply_isotonic_calibration(estimator, X_train, y_train, X_test)
            cal_metrics = evaluate_predictions(y_test, y_prob_cal)
            results_cal.append(ModelResult(name=f"{name}_calibrated", **cal_metrics))
            logger.info("  Cal → ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | KS: %.4f",
                        cal_metrics["roc_auc"], cal_metrics["pr_auc"],
                        cal_metrics["brier_score"], cal_metrics["ks_statistic"])
        except Exception as e:
            logger.warning("  Calibration failed for %s: %s — skipping.", name, e)

    ranked_raw = sorted(results_raw, key=lambda r: (r.pr_auc, r.roc_auc), reverse=True)
    ranked_cal = sorted(results_cal, key=lambda r: (r.pr_auc, r.roc_auc), reverse=True)
    best_raw = ranked_raw[0]
    best_cal = ranked_cal[0] if ranked_cal else ranked_raw[0]

    # --- Cost analysis on best calibrated model ---
    logger.info("Computing cost-sensitive metrics for best model: %s ...", best_cal.name)
    best_cal_name = best_cal.name.replace("_calibrated", "")
    y_best_prob = None
    for name, estimator in make_models().items():
        if name == best_cal_name:
            try:
                y_best_prob = _apply_isotonic_calibration(estimator, X_train, y_train, X_test)
            except Exception:
                pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
                pipeline.fit(X_train, y_train)
                y_best_prob = pipeline.predict_proba(X_test)[:, 1]
            break
    cost_metrics = _compute_cost_metrics(y_test, y_best_prob) if y_best_prob is not None else {}

    logger.info("Best raw model:      %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
                best_raw.name, best_raw.pr_auc, best_raw.roc_auc)
    logger.info("Best calibrated model: %s (PR-AUC=%.4f, ROC-AUC=%.4f)",
                best_cal.name, best_cal.pr_auc, best_cal.roc_auc)

    payload = {
        "problem": "credit_card_default",
        "dataset": str(data_path),
        "target_column": target_col,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {
            "non_default_0": int((y == 0).sum()),
            "default_1": int((y == 1).sum()),
            "default_rate": round(default_rate, 4),
        },
        "mutual_information_top_20": dict(list(mi_scores.items())[:20]),
        "information_values_top_15": top_iv_features,
        "engineered_features": [
            "UTIL_1..6", "PAY_RATIO_1..6", "AVG_UTIL", "AVG_BILL_AMT", "AVG_PAY_AMT",
            "AVG_PAY_RATIO", "NUM_DELAYED", "MAX_DELAY", "RECENCY_DELAY",
            "LIMIT_BAL_LOG", "BILL_TREND", "PAY_TREND",
            "AGE_GROUP", "LIMIT_BIN", "UTIL_DECILE", "UTIL_BIN",
        ],
        "raw_results": [asdict(r) for r in ranked_raw],
        "calibrated_results": [asdict(r) for r in ranked_cal],
        "best_raw_model": asdict(best_raw),
        "best_calibrated_model": asdict(best_cal),
        "cost_analysis": cost_metrics,
        "classification_reports": reports,
        "train_validation_metrics": train_validation_metrics,
        "tuned_hyperparameters_in_script": {
            "logistic_regression": {"C": 0.8, "max_iter": 3000, "class_weight": "balanced"},
            "lda": {"solver": "svd"},
            "knn": {"n_neighbors": 15, "weights": "distance"},
            "decision_tree": {"max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 20, "class_weight": "balanced"},
            "random_forest": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 10,
                              "min_samples_split": 20, "class_weight": "balanced_subsample"},
            "xgboost": {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
                        "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 5,
                        "reg_lambda": 1.5},
            "mlp": {"hidden_layer_sizes": (64, 32), "max_iter": 500, "early_stopping": True},
        },
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    return payload


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    metrics = run_card_pipeline()
    print("Card-default experiment complete.")
    print(f"Best model: {metrics['best_model']['name']}")
    print(f"Metrics written to: {DEFAULT_OUTPUT_PATH}")
