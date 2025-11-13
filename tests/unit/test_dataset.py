# tests/unit/test_dataset.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner

from src import dataset

runner = CliRunner()


@patch("src.dataset.simulate_progress")
@patch("src.dataset.logger")
def test_process_dataset(mock_logger, mock_progress):
    """Prueba la función principal de procesamiento."""
    input_path = Path("data/raw/sample.csv")
    output_path = Path("data/processed/output.csv")

    result = dataset.process_dataset(input_path, output_path)

    # Validaciones básicas
    assert result == output_path
    mock_logger.info.assert_called_once_with(f"Processing dataset from {input_path}")
    mock_logger.success.assert_called_once_with(
        f"Processing dataset complete. Saved to {output_path}"
    )
    mock_progress.assert_called_once_with("Processing dataset")


@patch("src.dataset.process_dataset")
def test_main_cli(mock_process):
    """Prueba el comando Typer CLI."""
    mock_process.return_value = Path("fake.csv")

    result = runner.invoke(dataset.app, [])

    # CLI debe ejecutarse sin errores
    assert result.exit_code == 0
    mock_process.assert_called_once()
