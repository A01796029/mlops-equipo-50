import pandas as pd
import pytest
from unittest.mock import patch
from src.data.make_dataset import load_dataset, run_cleaning_pipeline, FINAL_COLUMNS

@patch("src.data.make_dataset.run_cleaning_pipeline")
def test_load_dataset_returns_dataframe(mock_run_pipeline):
    """Verifica que load_dataset devuelve un DataFrame limpio con columnas esperadas."""
    # Simula un DataFrame limpio
    mock_df = pd.DataFrame(columns=FINAL_COLUMNS)
    mock_run_pipeline.return_value = mock_df

    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FINAL_COLUMNS
    mock_run_pipeline.assert_called_once()

@patch("src.data.make_dataset.DataCleaningPipeline")
def test_run_cleaning_pipeline_returns_clean_df(mock_pipeline_class):
    """Prueba que run_cleaning_pipeline devuelve el DataFrame esperado."""
    mock_instance = mock_pipeline_class.return_value
    mock_instance.ejecutar_pipeline.return_value = pd.DataFrame(columns=FINAL_COLUMNS)

    df = run_cleaning_pipeline(raw_data_path="fake_path.csv", expected_columns=FINAL_COLUMNS)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FINAL_COLUMNS
    mock_instance.ejecutar_pipeline.assert_called_once()