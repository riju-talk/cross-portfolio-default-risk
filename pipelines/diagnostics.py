from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_slug(name: str) -> str:
    """Return a filesystem-safe slug for model names."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return "_".join(part for part in slug.split("_") if part)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution_plot(y: pd.Series, output_path: Path, title: str) -> None:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    total = max(1, int(counts.sum()))

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Non-default (0)", "Default (1)"]
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518"])
    ax.set_title(title)
    ax.set_ylabel("Rows")

    for bar, count in zip(bars, counts.values):
        pct = 100.0 * float(count) / float(total)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
        )

    _save_figure(fig, output_path)


def save_missingness_plot(
    features: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> bool:
    missing_ratio = features.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0].head(top_n)

    if missing_ratio.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(missing_ratio) + 1)))
    ax.barh(missing_ratio.index.astype(str), missing_ratio.values * 100.0, color="#E45756")
    ax.set_title(title)
    ax.set_xlabel("Missing values (%)")
    ax.invert_yaxis()

    _save_figure(fig, output_path)
    return True


def save_numeric_feature_grid(
    features: pd.DataFrame,
    output_path: Path,
    title: str,
    max_features: int = 6,
) -> bool:
    numeric_df = features.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return False

    variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
    selected = variances.head(max_features).index.tolist()
    if not selected:
        return False

    n_cols = 3
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows))
    axes_arr = np.array(axes, dtype=object).reshape(-1)

    for idx, feature in enumerate(selected):
        ax = axes_arr[idx]
        values = pd.to_numeric(numeric_df[feature], errors="coerce").dropna()
        if values.empty:
            ax.text(0.5, 0.5, "No non-null values", ha="center", va="center")
            ax.set_title(str(feature))
            continue
        ax.hist(values, bins=30, color="#54A24B", alpha=0.85)
        ax.set_title(str(feature))
        ax.set_ylabel("Count")

    for ax in axes_arr[len(selected) :]:
        ax.axis("off")

    fig.suptitle(title)
    _save_figure(fig, output_path)
    return True


def save_metric_comparison_plot(
    model_metrics: Mapping[str, Mapping[str, float]],
    output_path: Path,
    title: str,
) -> None:
    if not model_metrics:
        return

    metric_specs = [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("ks_statistic", "KS Statistic"),
        ("f1", "F1"),
        ("top_decile_capture_rate", "Top-Decile Capture"),
        ("brier_score", "Brier Score"),
    ]
    model_names = list(model_metrics.keys())
    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, (metric_key, label) in enumerate(metric_specs):
        values = [float(model_metrics[m].get(metric_key, 0.0)) for m in model_names]
        offset = (idx - (len(metric_specs) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    _save_figure(fig, output_path)


def save_train_vs_validation_plot(
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    output_path: Path,
    title: str,
) -> None:
    metric_keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier_score", "ks_statistic", "calibration_error"]
    labels = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "Brier", "KS", "Calib Err"]
    train_values = [float(train_metrics.get(k, 0.0)) for k in metric_keys]
    validation_values = [float(validation_metrics.get(k, 0.0)) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, train_values, width=width, label="Train", color="#4C78A8")
    ax.bar(
        x + width / 2,
        validation_values,
        width=width,
        label="Validation",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    _save_figure(fig, output_path)


def save_roc_curve_plot(y_true: pd.Series, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, name="Validation", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#808080")
    ax.set_title(title)
    _save_figure(fig, output_path)


def save_precision_recall_plot(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, name="Validation", ax=ax)
    ax.set_title(title)
    _save_figure(fig, output_path)


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Non-default", "Default"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    _save_figure(fig, output_path)


def save_score_distribution_plot(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    y_arr = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, 21)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        y_prob[y_arr == 0],
        bins=bins,
        alpha=0.7,
        label="True class: Non-default (0)",
        color="#4C78A8",
    )
    ax.hist(
        y_prob[y_arr == 1],
        bins=bins,
        alpha=0.7,
        label="True class: Default (1)",
        color="#F58518",
    )
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    _save_figure(fig, output_path)


def save_calibration_plot(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="#54A24B", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#808080", label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend()

    _save_figure(fig, output_path)


def save_feature_importance_plot(
    trained_pipeline: Pipeline,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> bool:
    model = trained_pipeline.named_steps.get("model")
    preprocessor = trained_pipeline.named_steps.get("preprocessor")
    if model is None or preprocessor is None:
        return False

    importances: np.ndarray
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    else:
        return False

    try:
        feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    except Exception:
        feature_names = np.asarray(
            [f"feature_{idx}" for idx in range(importances.shape[0])],
            dtype=str,
        )

    size = min(len(feature_names), importances.shape[0])
    if size == 0:
        return False

    feature_names = feature_names[:size]
    importances = importances[:size]

    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_features = feature_names[top_idx][::-1]
    top_values = importances[top_idx][::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top_features) + 1)))
    ax.barh(top_features, top_values, color="#72B7B2")
    ax.set_xlabel("Importance")
    ax.set_title(title)

    _save_figure(fig, output_path)
    return True


def save_prediction_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(y_true, pd.Series):
        row_ids = y_true.index.to_list()
    else:
        row_ids = list(range(len(y_prob)))

    table = pd.DataFrame(
        {
            "row_id": row_ids,
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
            "y_prob_default": np.asarray(y_prob, dtype=float),
        }
    )
    table["risk_percentile"] = table["y_prob_default"].rank(method="average", pct=True)
    table.sort_values("y_prob_default", ascending=False, inplace=True)
    table.to_csv(output_path, index=False)


def record_output_path(path: Path, sink: list[str]) -> None:
    try:
        sink.append(str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/"))
    except Exception:
        sink.append(str(path).replace("\\", "/"))


def build_validation_metrics_map(results: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for row in results:
        name = str(row.get("name", "model"))
        payload[name] = {
            "roc_auc": float(row.get("roc_auc", 0.0)),
            "pr_auc": float(row.get("pr_auc", 0.0)),
            "f1": float(row.get("f1", 0.0)),
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "top_decile_capture_rate": float(row.get("top_decile_capture_rate", 0.0)),
            "brier_score": float(row.get("brier_score", 0.0)),
            "ks_statistic": float(row.get("ks_statistic", 0.0)),
            "calibration_error": float(row.get("calibration_error", 0.0)),
        }
    return payload
