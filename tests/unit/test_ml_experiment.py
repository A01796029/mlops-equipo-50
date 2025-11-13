# tests/unit/test_ml_experiment.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.modeling.ml_experiment import MLExperiment

# ===============================
# Fixtures de ejemplo
# ===============================
@pytest.fixture
def sample_data():
    X_train = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'c'])
    X_val = pd.DataFrame(np.random.rand(5, 3), columns=['a', 'b', 'c'])
    y_train = pd.Series(np.random.rand(10))
    y_val = pd.Series(np.random.rand(5))
    return X_train, X_val, y_train, y_val

# ===============================
# Test del run_lazypredict
# ===============================
@patch("src.modeling.ml_experiment.mlflow.log_artifact")
@patch("src.modeling.ml_experiment.mlflow.start_run")
@patch("src.modeling.ml_experiment.LazyRegressor")
def test_run_lazypredict(mock_lazy, mock_mlflow, mock_log_artifact, sample_data):
    X_train, X_val, y_train, y_val = sample_data

    # Mock de MLflow
    mock_run = MagicMock()
    mock_run.info.run_id = "12345"
    mock_mlflow.return_value.__enter__.return_value = mock_run

    # Mock LazyRegressor
    mock_reg = MagicMock()
    mock_reg.fit.return_value = (
        pd.DataFrame({
            'Model': ['ModelA', 'ModelB'],
            'R-Squared': [0.8, 0.7],
            'RMSE': [0.1, 0.2]
        }),
        None
    )
    mock_lazy.return_value = mock_reg

    experiment = MLExperiment("TestExperiment")
    top_models = experiment.run_lazypredict(X_train, X_val, y_train, y_val, top_n=2)

    assert top_models == ['ModelA', 'ModelB']
    mock_lazy.assert_called_once()
    mock_log_artifact.assert_called_once_with("lazy_regressor_top10.csv", "results_data", run_id="12345")

# ===============================
# Test de _log_plots
# ===============================
@patch("src.modeling.ml_experiment.mlflow.log_artifact")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_log_plots(mock_close, mock_save, mock_log_artifact):
    y_true = [1, 2, 3]
    y_pred = [1.1, 1.9, 3.05]
    
    experiment = MLExperiment("TestExperiment")
    experiment._log_plots(y_true, y_pred, "ModelX", "TRAIN", "run123")
    
    mock_save.assert_called_once()
    mock_log_artifact.assert_called_once_with("ModelX_TRAIN_vs_pred.png", "plots", run_id="run123")
    mock_close.assert_called()

# ===============================
# Test de run_grid_search con mock
# ===============================
@patch("src.modeling.ml_experiment.GridSearchCV")
@patch("src.modeling.ml_experiment.mlflow.start_run")
@patch("src.modeling.ml_experiment.mlflow.sklearn.log_model")
@patch("src.modeling.ml_experiment.calculate_metrics")
@patch("src.modeling.ml_experiment.MLExperiment._log_plots")
def test_run_grid_search(
    mock_log_plots,
    mock_calc_metrics,
    mock_log_model,
    mock_mlflow_start,
    mock_gridsearch
):
    X_trainval = pd.DataFrame(np.random.rand(5, 2), columns=['x1', 'x2'])
    y_trainval = pd.Series(np.random.rand(5))
    X_test = pd.DataFrame(np.random.rand(2, 2), columns=['x1', 'x2'])
    y_test = pd.Series(np.random.rand(2))
    
    # Mock MLflow
    mock_run = MagicMock()
    mock_run.info.run_id = "run999"
    mock_mlflow_start.return_value.__enter__.return_value = mock_run
    
    # Mock GridSearchCV
    mock_gs = MagicMock()
    mock_gs.best_estimator_ = MagicMock()
    mock_gs.best_estimator_.predict.side_effect = [y_trainval, y_test]
    mock_gs.best_estimator_.get_params.return_value = {"param1": 1}
    mock_gs.best_params_ = {"param1": 1}
    mock_gs.best_score_ = 0.95
    mock_gridsearch.return_value = mock_gs
    
    # Mock calculate_metrics
    mock_calc_metrics.side_effect = lambda y_true, y_pred: {"r2": 0.9, "rmse": 0.1, "mae": 0.05}
    
    experiment = MLExperiment("TestExperiment")
    experiment.run_grid_search(
        "ModelX",
        model_instance=MagicMock(),
        param_grid={"param1": [1,2]},
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        X_test=X_test,
        y_test=y_test
    )
    
    mock_mlflow_start.assert_called()
    mock_gridsearch.assert_called()
    mock_log_model.assert_called()
    mock_log_plots.assert_called()
