from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
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
DEFAULT_DATA_PATH = Path("data/default_of_credit_card_clients.csv")
DEFAULT_RESULTS_DIR = Path("results/card")
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "metrics.json"
CARD_CATEGORICAL_COLUMNS = {"SEX", "EDUCATION", "MARRIAGE"}


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
    return features, target, target_col


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the credit-card dataset.

    Numeric columns are median-imputed and standard-scaled (robust to
    outliers in bill amounts).  Categorical columns (SEX, EDUCATION,
    MARRIAGE) are mode-imputed and one-hot encoded.  Using median
    rather than mean avoids distortion from the heavy right tail in
    bill-amount distributions.
    """
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
    """Instantiate candidate models for credit-default prediction.

    Includes a regularised logistic regression (interpretable, strong
    baseline for credit scoring), a random forest (non-linear effects,
    interaction detection), and XGBoost (SOTA gradient boosting).
    All use class-weight or scale_pos_weight to handle the ~22 %
    default rate — a model that simply predicts "no default" every
    time would achieve 78 % accuracy but zero business value.
    """
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


def _save_dataset_level_plots(
    X: pd.DataFrame,
    y: pd.Series,
    plots_dir: Path,
    generated_output_files: list[str],
) -> None:
    """Exploratory plots that characterise the dataset before modelling.

    In a credit-risk workflow, understanding class imbalance, missingness,
    feature distributions, collinearity, and each feature's marginal
    predictive power (mutual information) guides feature engineering
    and model selection.
    """
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

    # --- Feature collinearity heatmap ---
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_path = dataset_plot_dir / "correlation_heatmap.png"
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = X[numeric_cols].corr(method="spearman")
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            ax=ax,
        )
        ax.set_title("Spearman Correlation Heatmap (Numeric Features)")
        fig.tight_layout()
        fig.savefig(corr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        record_output_path(corr_path, generated_output_files)
        logger.info("  Saved correlation heatmap (%d numeric features)", len(numeric_cols))

    # --- Mutual Information with the target ---
    if len(numeric_cols) > 1:
        mi_path = dataset_plot_dir / "mutual_information.png"
        mi_scores = mutual_info_classif(X[numeric_cols].fillna(X[numeric_cols].median()), y)
        mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(numeric_cols) * 0.35)))
        mi_series.plot(kind="barh", ax=ax, color="steelblue", edgecolor="none")
        ax.set_xlabel("Mutual Information (nats)")
        ax.set_title("Feature Predictive Power — Mutual Information with Default")
        fig.tight_layout()
        fig.savefig(mi_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        record_output_path(mi_path, generated_output_files)
        logger.info("  Saved mutual information plot")

    # --- Information Value (IV) summary ---
    iv_path = dataset_plot_dir / "information_value.png"
    iv_scores = _compute_information_values(X, y)
    top_iv = iv_scores.head(15)

    if len(top_iv) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(top_iv) * 0.35)))
        colors = ["#2ecc71" if v >= 0.1 else "#f39c12" if v >= 0.02 else "#e74c3c" for v in top_iv.values]
        top_iv.sort_values().plot(kind="barh", ax=ax, color=colors, edgecolor="none")
        ax.axvline(0.02, ls="--", color="gray", alpha=0.5, label="Weak threshold (0.02)")
        ax.axvline(0.1, ls="--", color="orange", alpha=0.5, label="Medium threshold (0.10)")
        ax.set_xlabel("Information Value (IV)")
        ax.set_title("Top Features by Information Value")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(iv_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        record_output_path(iv_path, generated_output_files)
        logger.info("  Saved information value plot (top %d features)", len(top_iv))


def run_card_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run the full credit-card default prediction experiment.

    Pipeline steps:
      1. Load and split the UCI credit-card dataset (80/20 stratified).
      2. Generate exploratory plots (distribution, missingness,
         correlation, mutual information, information value).
      3. Train logistic regression, random forest, and XGBoost
         with class-weight adjustments for imbalance.
      4. Evaluate each model on rank, threshold, calibration, and
         separation metrics (ROC-AUC, PR-AUC, F1, precision, recall,
         top-decile capture, Brier score, KS statistic, calibration
         error).
      5. Record diagnostics (confusion matrix, calibration curve,
         score distribution, feature importance, train/val comparison).
      6. Rank models by (PR-AUC, ROC-AUC) and persist results as JSON.

    In a production credit-risk setting, the Brier score and KS
    statistic are as critical as AUC — a well-calibrated score with
    strong default/non-default separation ensures sound lending
    decisions and regulatory compliance.
    """
    output_path = Path(output_path)
    results_dir = output_path.parent
    plots_dir = results_dir / "plots"
    predictions_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s ...", data_path)
    X, y, target_col = load_card_data(data_path)
    default_rate = float(y.mean())
    logger.info(
        "Dataset loaded: %d rows, %d features, default rate = %.2f%%",
        len(X), X.shape[1], default_rate * 100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    logger.info(
        "Train/test split: %d / %d (stratified, test=20%%)",
        len(X_train), len(X_test),
    )

    preprocessor = build_preprocessor(X)

    results: list[ModelResult] = []
    reports: dict[str, dict[str, Any]] = {}
    train_validation_metrics: dict[str, dict[str, dict[str, float]]] = {}
    generated_output_files: list[str] = []

    logger.info("Generating dataset-level exploratory plots ...")
    _save_dataset_level_plots(
        X=X,
        y=y,
        plots_dir=plots_dir,
        generated_output_files=generated_output_files,
    )

    # Compute IV once for the payload (on full dataset for stability)
    logger.info("Computing information values for all features ...")
    iv_scores = _compute_information_values(X, y)
    top_iv_features = iv_scores.head(10).to_dict()

    for name, estimator in make_models().items():
        logger.info("Training %s ...", name)
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
        val_metrics = evaluate_predictions(y_test, y_prob)
        train_validation_metrics[name] = {
            "train": train_metrics,
            "validation": val_metrics,
        }

        metrics = val_metrics
        results.append(ModelResult(name=name, **metrics))
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

        logger.info(
            "  %s — ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | KS: %.4f | CalibErr: %.4f | TopDec: %.2f%%",
            name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["brier_score"],
            metrics["ks_statistic"],
            metrics["calibration_error"],
            metrics["top_decile_capture_rate"] * 100,
        )

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

    logger.info(
        "Best model: %s (PR-AUC=%.4f, ROC-AUC=%.4f, Brier=%.4f, KS=%.4f)",
        best_model.name,
        best_model.pr_auc,
        best_model.roc_auc,
        best_model.brier_score,
        best_model.ks_statistic,
    )

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
        "information_values": top_iv_features,
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
