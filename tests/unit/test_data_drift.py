import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.modeling.data_drift import (
    PerformanceDriftReport,
    detect_drift,
    drift_results_as_dataframe,
    evaluate_performance_drift,
    generate_monitoring_dataset,
)


def test_detect_drift_flags_numerical_shift():
    rng = np.random.default_rng(42)
    baseline = pd.DataFrame({"temp": rng.normal(loc=0.0, scale=1.0, size=500)})
    current = pd.DataFrame({"temp": rng.normal(loc=1.0, scale=1.0, size=500)})

    results = detect_drift(baseline, current, numerical_cols=["temp"], categorical_cols=[])

    assert bool(results["temp"]["drift_detected"]) is True
    assert results["temp"]["test"] == "Kolmogorov-Smirnov"


def test_detect_drift_handles_categorical_counts():
    baseline = pd.DataFrame({"season": [1] * 300 + [2] * 200})
    current = pd.DataFrame({"season": [1] * 100 + [3] * 400})

    results = detect_drift(baseline, current, numerical_cols=[], categorical_cols=["season"])

    assert bool(results["season"]["drift_detected"]) is True
    assert results["season"]["test"] == "Chi-squared"


def test_drift_results_as_dataframe_produces_table():
    baseline = pd.DataFrame({"windspeed": [0.1, 0.2, 0.3]})
    current = pd.DataFrame({"windspeed": [0.1, 0.2, 0.4]})

    results = detect_drift(baseline, current, numerical_cols=["windspeed"], categorical_cols=[])
    report = drift_results_as_dataframe(results)

    assert not report.empty
    assert set(report.columns) >= {"feature", "test"}


def test_generate_monitoring_dataset_introduces_changes():
    season_values = np.resize(np.array([1, 2, 3, 4]), 50)
    df = pd.DataFrame(
        {
            "temp": np.linspace(0, 1, 50),
            "hum": np.linspace(0.5, 0.9, 50),
            "season": pd.Series(season_values).astype("category"),
        }
    )

    monitoring = generate_monitoring_dataset(df, random_state=7)

    assert monitoring.shape == df.shape
    assert monitoring.isnull().sum().sum() > 0
    assert not monitoring.equals(df)

    expected_shift = 2.0 * df["temp"].std(ddof=0)
    diff = (monitoring["temp"] - df["temp"]).dropna()
    assert np.isclose(diff.mean(), expected_shift)


def test_detect_drift_handles_missing_categories_without_error():
    baseline = pd.DataFrame({"season": [1] * 50 + [2] * 50})
    current = pd.DataFrame({"season": [np.nan] * 100})

    results = detect_drift(baseline, current, numerical_cols=[], categorical_cols=["season"])

    assert "error" in results["season"]


def test_evaluate_performance_drift_returns_report():
    X = pd.DataFrame({"temp": np.linspace(0, 1, 30), "hum": np.linspace(1, 2, 30)})
    y = pd.Series(np.linspace(10, 20, 30))
    model = LinearRegression()
    model.fit(X, y)

    monitoring_X = X.copy()
    monitoring_X["temp"] += 2

    report = evaluate_performance_drift(model, X, y, monitoring_X, y)

    assert isinstance(report, PerformanceDriftReport)
    assert set(report.deltas.keys()) == {"r2_drop", "rmse_increase", "mae_increase"}
