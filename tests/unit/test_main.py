# tests/unit/test_main.py
import pytest
from unittest.mock import patch, MagicMock
import builtins

# Importamos el módulo completo, no la función directamente,
# para poder recargarlo si es necesario
import src.modeling.main as main_mod


@pytest.fixture
def mock_data_split():
    """Fixture que devuelve datos ficticios simulando load_and_split()."""
    import pandas as pd
    import numpy as np
    X_trainval = pd.DataFrame(np.random.rand(5, 2))
    X_test = pd.DataFrame(np.random.rand(2, 2))
    y_trainval = pd.Series(np.random.rand(5))
    y_test = pd.Series(np.random.rand(2))
    X_train = pd.DataFrame(np.random.rand(3, 2))
    X_val = pd.DataFrame(np.random.rand(2, 2))
    y_train = pd.Series(np.random.rand(3))
    y_val = pd.Series(np.random.rand(2))
    return X_trainval, X_test, y_trainval, y_test, X_train, X_val, y_train, y_val


@patch("src.modeling.main.MLExperiment")
@patch("src.modeling.main.build_pipeline")
@patch("src.modeling.main.DataProcessor")
def test_main_success(mock_data_proc, mock_build_pipeline, mock_ml_exp, mock_data_split, capsys):
    """Caso exitoso: se ejecuta el flujo completo sin errores."""

    # Mock del DataProcessor y su retorno
    mock_data_instance = MagicMock()
    mock_data_instance.load_and_split.return_value = mock_data_split
    mock_data_proc.return_value = mock_data_instance

    # Mock del experimento y su comportamiento
    mock_exp_instance = MagicMock()
    mock_exp_instance.run_lazypredict.return_value = ["ModelA", "ModelB"]
    mock_ml_exp.return_value = mock_exp_instance

    # Mock del pipeline builder
    mock_build_pipeline.return_value = "mock_pipeline"

    # Mock de MODEL_MAP y PARAM_GRIDS
    with patch.object(main_mod, "MODEL_MAP", {"ModelA": "mock_model", "ModelB": "mock_model"}), \
         patch.object(main_mod, "PARAM_GRIDS", {"ModelA": {"param1": [1, 2]}}):

        main_mod.main()

    captured = capsys.readouterr()
    assert "Preparando Datos" in captured.out
    assert "LazyRegressor" in captured.out
    assert "Optimizando Modelos" in captured.out

    # Verifica llamadas clave
    mock_data_proc.assert_called_once()
    mock_exp_instance.run_lazypredict.assert_called_once()
    mock_build_pipeline.assert_called_once_with("mock_model", mock_data_split[0])
    mock_exp_instance.run_grid_search.assert_called_once()


@patch("src.modeling.main.DataProcessor")
def test_main_load_error(mock_data_proc, capsys):
    """Simula un error al cargar datos para cubrir el bloque except."""
    mock_data_instance = MagicMock()
    mock_data_instance.load_and_split.side_effect = Exception("Archivo corrupto")
    mock_data_proc.return_value = mock_data_instance

    main_mod.main()

    captured = capsys.readouterr()
    assert "No se pudo cargar" in captured.out
    assert "Error: Archivo corrupto" in captured.out


@patch("src.modeling.main.MLExperiment")
@patch("src.modeling.main.build_pipeline")
@patch("src.modeling.main.DataProcessor")
def test_main_model_without_param_grid(mock_data_proc, mock_build_pipeline, mock_ml_exp, mock_data_split, capsys):
    """Cubre el caso en que un modelo no tiene PARAM_GRIDS definidos."""
    mock_data_instance = MagicMock()
    mock_data_instance.load_and_split.return_value = mock_data_split
    mock_data_proc.return_value = mock_data_instance

    mock_exp_instance = MagicMock()
    mock_exp_instance.run_lazypredict.return_value = ["ModelX"]
    mock_ml_exp.return_value = mock_exp_instance

    with patch.object(main_mod, "MODEL_MAP", {"ModelX": "mock_model"}), \
         patch.object(main_mod, "PARAM_GRIDS", {}):
        main_mod.main()

    captured = capsys.readouterr()
    assert "Skipping optimization for ModelX" in captured.out
