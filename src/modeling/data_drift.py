"""Utility helpers to simulate and measure data drift throughout the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, ks_2samp

from src.modeling.data_processor import calculate_metrics


DRIFT_THRESHOLDS = {
    "r2_drop": 0.05,
    "rmse_increase": 2.0,
    "mae_increase": 1.0,
}


@dataclass
class PerformanceDriftReport:
    baseline_metrics: Dict[str, float]
    monitoring_metrics: Dict[str, float]
    deltas: Dict[str, float]
    alert_triggered: bool
    recommended_action: str


def _column_available(df_a: pd.DataFrame, df_b: pd.DataFrame, column: str) -> bool:
    return column in df_a.columns and column in df_b.columns


def detect_drift(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_cols: Iterable[str],
    categorical_cols: Iterable[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    """Detects distribution shift for numerical and categorical columns."""

    drift_results: Dict[str, Dict[str, object]] = {}

    for col in numerical_cols:
        if not _column_available(baseline_data, current_data, col):
            drift_results[col] = {"error": "Column not found in both datasets"}
            continue

        base_values = baseline_data[col].dropna()
        current_values = current_data[col].dropna()
        if base_values.empty or current_values.empty:
            drift_results[col] = {"error": "Insufficient data for KS test"}
            continue

        stat, p_value = ks_2samp(base_values, current_values)
        drift_results[col] = {
            "drift_detected": p_value < alpha,
            "p_value": float(p_value),
            "statistic": float(stat),
            "test": "Kolmogorov-Smirnov",
        }

    for col in categorical_cols:
        if not _column_available(baseline_data, current_data, col):
            drift_results[col] = {"error": "Column not found in both datasets"}
            continue

        baseline_counts = baseline_data[col].value_counts()
        current_counts = current_data[col].value_counts()
        contingency = (
            pd.concat([baseline_counts, current_counts], axis=1, keys=["baseline", "current"])
            .fillna(0)
        )

        # Drop categories or datasets that contribute zero information
        contingency = contingency.loc[contingency.sum(axis=1) > 0]
        contingency = contingency.loc[:, contingency.sum(axis=0) > 0]

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            drift_results[col] = {"error": "Insufficient data for Chi-squared test"}
            continue

        try:
            chi2, p_value, _, _ = chi2_contingency(contingency)
        except ValueError as err:
            drift_results[col] = {"error": f"Chi-squared failed: {err}"}
            continue
        drift_results[col] = {
            "drift_detected": p_value < alpha,
            "p_value": float(p_value),
            "statistic": float(chi2),
            "test": "Chi-squared",
        }

    return drift_results


def drift_results_as_dataframe(results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Transforms drift results into a tabular representation."""
    records: List[Dict[str, object]] = []
    for feature, stats in results.items():
        record = {"feature": feature}
        record.update(stats)
        records.append(record)
    return pd.DataFrame(records)


def log_drift_summary(results: Dict[str, Dict[str, object]]) -> None:
    """Logs a compact summary table for quick inspection."""
    report_df = drift_results_as_dataframe(results)
    if report_df.empty:
        logger.warning("No drift evaluation results to report.")
        return

    # Sort so that detected drifts bubble to the top
    if "drift_detected" in report_df.columns:
        report_df = report_df.sort_values(by="drift_detected", ascending=False)

    logger.info("Data drift detection summary:\n{}", report_df.to_string(index=False))


def _infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int", "float", "number"]).columns.tolist()
    return num_cols, cat_cols


def generate_monitoring_dataset(
    reference_df: pd.DataFrame,
    mean_shift_std: float = 2.0,
    missing_rate: float = 0.1,
    seasonal_col: str = "season",
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulates a monitoring dataset by distorting the validation distribution.

    Numeric columns are shifted by ``mean_shift_std`` standard deviations to emulate the
    "two means" displacement described in the monitoring requirements.
    """

    rng = np.random.default_rng(random_state)
    monitoring_df = reference_df.copy(deep=True)
    numeric_cols, categorical_cols = _infer_feature_types(monitoring_df)

    for col in numeric_cols:
        col_std = monitoring_df[col].std(ddof=0)
        shift_value = mean_shift_std * (col_std if not np.isnan(col_std) else 0)
        monitoring_df[col] = monitoring_df[col] + shift_value

        missing_mask = rng.random(len(monitoring_df)) < missing_rate
        monitoring_df.loc[missing_mask, col] = np.nan

    if categorical_cols:
        drift_col = seasonal_col if seasonal_col in monitoring_df.columns else categorical_cols[0]
        shuffled = monitoring_df[drift_col].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        monitoring_df[drift_col] = shuffled

        missing_mask = rng.random(len(monitoring_df)) < missing_rate
        monitoring_df.loc[missing_mask, drift_col] = np.nan

    return monitoring_df


def evaluate_performance_drift(
    model,
    baseline_X: pd.DataFrame,
    baseline_y: pd.Series,
    monitoring_X: pd.DataFrame,
    monitoring_y: pd.Series,
) -> PerformanceDriftReport:
    """Calculates metric deltas between baseline and monitoring datasets."""

    baseline_pred = model.predict(baseline_X)
    monitoring_pred = model.predict(monitoring_X)

    baseline_metrics = calculate_metrics(baseline_y, baseline_pred)
    monitoring_metrics = calculate_metrics(monitoring_y, monitoring_pred)

    deltas = {
        "r2_drop": baseline_metrics["r2"] - monitoring_metrics["r2"],
        "rmse_increase": monitoring_metrics["rmse"] - baseline_metrics["rmse"],
        "mae_increase": monitoring_metrics["mae"] - baseline_metrics["mae"],
    }

    alert_triggered = any(
        deltas[key] > threshold for key, threshold in DRIFT_THRESHOLDS.items() if key in deltas
    )

    if alert_triggered:
        recommended_action = "Revisar pipeline de features y considerar retrain"
    else:
        recommended_action = "Sin acción inmediata, continuar monitoreo"

    return PerformanceDriftReport(
        baseline_metrics=baseline_metrics,
        monitoring_metrics=monitoring_metrics,
        deltas=deltas,
        alert_triggered=alert_triggered,
        recommended_action=recommended_action,
    )


def save_drift_report(report: PerformanceDriftReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": report.baseline_metrics,
        "monitoring": report.monitoring_metrics,
        "deltas": report.deltas,
        "alert_triggered": report.alert_triggered,
        "recommended_action": report.recommended_action,
        "thresholds": DRIFT_THRESHOLDS,
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    logger.info("Drift report stored at {}", output_path)


def plot_metric_comparison(
    report: PerformanceDriftReport,
    output_path: Path,
) -> None:
    """Creates a bar plot comparing baseline vs monitoring metrics."""

    metrics = ["rmse", "mae", "r2"]
    baseline_values = [report.baseline_metrics[m] for m in metrics]
    monitoring_values = [report.monitoring_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, baseline_values, width, label="Baseline")
    plt.bar(x + width / 2, monitoring_values, width, label="Monitoring")
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylabel("Metric value")
    plt.title("Performance comparison: baseline vs monitoring")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Metric comparison plot saved to {}", output_path)


def persist_and_log_drift_outputs(
    *,
    performance_report: PerformanceDriftReport,
    monitoring_features: pd.DataFrame,
    monitoring_target: pd.Series,
    pre_drift_df: Optional[pd.DataFrame],
    post_drift_df: Optional[pd.DataFrame],
    experiment_name: str,
    drift_dir: Path,
    timestamp: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """Stores drift artifacts and records a monitoring run in MLflow."""

    drift_dir = Path(drift_dir)
    drift_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    monitoring_path = drift_dir / f"monitoring_batch_{timestamp}.csv"
    monitoring_features.assign(cnt=monitoring_target.values).to_csv(monitoring_path, index=False)

    summary_path = drift_dir / f"drift_report_{timestamp}.json"
    figure_path = drift_dir / f"drift_metrics_{timestamp}.png"
    pre_path_candidate = drift_dir / f"feature_drift_pre_{timestamp}.csv"
    post_path_candidate = drift_dir / f"feature_drift_post_{timestamp}.csv"

    def _write_df(df: Optional[pd.DataFrame], dst: Path) -> Optional[Path]:
        if df is not None and not df.empty:
            df.to_csv(dst, index=False)
            return dst
        return None

    feature_drift_pre_path = _write_df(pre_drift_df, pre_path_candidate)
    feature_drift_post_path = _write_df(post_drift_df, post_path_candidate)

    save_drift_report(performance_report, summary_path)
    plot_metric_comparison(performance_report, figure_path)

    print(
        f"Alert triggered: {performance_report.alert_triggered}. Acción sugerida: {performance_report.recommended_action}"
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="DataDriftMonitoring"):
        mlflow.set_tag("stage", "Data Drift Monitoring")
        mlflow.log_params({
            "alert_triggered": str(performance_report.alert_triggered),
            "recommended_action": performance_report.recommended_action,
        })
        mlflow.log_params({f"threshold_{k}": v for k, v in DRIFT_THRESHOLDS.items()})

        mlflow.log_metrics({f"baseline_{k}": v for k, v in performance_report.baseline_metrics.items()})
        mlflow.log_metrics({f"monitoring_{k}": v for k, v in performance_report.monitoring_metrics.items()})
        mlflow.log_metrics({f"delta_{k}": v for k, v in performance_report.deltas.items()})

        for artifact_path in (summary_path, figure_path, monitoring_path, feature_drift_pre_path, feature_drift_post_path):
            if artifact_path is not None:
                mlflow.log_artifact(artifact_path, artifact_path="drift")

    return {
        "monitoring_path": monitoring_path,
        "summary_path": summary_path,
        "figure_path": figure_path,
        "feature_drift_pre_path": feature_drift_pre_path,
        "feature_drift_post_path": feature_drift_post_path,
    }
