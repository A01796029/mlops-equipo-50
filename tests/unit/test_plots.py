# tests/unit/test_plots.py
import pytest
from unittest.mock import patch
from pathlib import Path
from typer.testing import CliRunner

from src import plots

runner = CliRunner()


@patch("src.plots.simulate_progress")
@patch("src.plots.logger")
def test_generate_plot(mock_logger, mock_progress):
    """Prueba la generación de gráficos."""
    input_path = Path("data/processed/sample.csv")
    output_path = Path("data/figures/plot.png")

    result = plots.generate_plot(input_path, output_path)

    assert result == output_path
    mock_logger.info.assert_called_once_with(f"Generating plot from {input_path}")
    mock_logger.success.assert_called_once_with(
        f"Plot generation complete. Saved to {output_path}"
    )
    mock_progress.assert_called_once_with("Generating plot")


@patch("src.plots.generate_plot")
def test_main_cli(mock_generate):
    """Prueba el CLI de generación de gráficos."""
    mock_generate.return_value = Path("fake_plot.png")

    result = runner.invoke(plots.app, [])

    assert result.exit_code == 0
    mock_generate.assert_called_once()
