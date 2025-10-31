# src/dataset.py
from pathlib import Path
from typing import Optional

from loguru import logger
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import simulate_progress

app = typer.Typer()

def process_dataset(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Función principal para procesar el dataset crudo a procesado.
    Actualmente placeholder; reemplace con transformaciones reales.
    """
    logger.info(f"Processing dataset from {input_path}")
    simulate_progress("Processing dataset")
    # Ejemplo real:
    # df = pd.read_csv(input_path)
    # processed_df = cleaning_and_transformations(df)
    # save_dataframe_to_csv(processed_df, output_path)
    logger.success(f"Processing dataset complete. Saved to {output_path}")
    return output_path


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    process_dataset(input_path, output_path)


if __name__ == "__main__":
    app()
