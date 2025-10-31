# src/plots.py
from pathlib import Path
from typing import Optional

from loguru import logger
import typer

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from src.utils import simulate_progress

app = typer.Typer()

def generate_plot(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Genera y guarda un plot a partir de input_path (placeholder).
    Reemplazar por lógica real de visualización.
    """
    logger.info(f"Generating plot from {input_path}")
    simulate_progress("Generating plot")
    # Ejemplo real:
    # df = pd.read_csv(input_path)
    # fig = plot_some_figure(df)
    # fig.savefig(output_path)
    logger.success(f"Plot generation complete. Saved to {output_path}")
    return output_path


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
) -> None:
    generate_plot(input_path, output_path)


if __name__ == "__main__":
    app()