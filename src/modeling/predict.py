# src/modeling/predict.py
from pathlib import Path
from typing import Optional

from loguru import logger
import typer
import pandas as pd 

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import simulate_progress

app = typer.Typer()

def run_inference(features_path: Path, model_path: Path, predictions_path: Path) -> Optional[Path]:
    """
    Ejecuta la inferencia de un modelo sobre features y guarda las predicciones.
    (Placeholder; implementar carga de modelo y predicción real.)
    """
    logger.info("Performing inference for model...")
    simulate_progress("Inference")

    # Simulación: crear archivo vacío de predicciones para pruebas
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pred": [0.5, 0.7, 0.9]}).to_csv(predictions_path, index=False)

    logger.success(f"Inference complete. Predictions saved to {predictions_path}")
    return predictions_path


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    run_inference(features_path, model_path, predictions_path)


def generate_predictions(features_path: Path, model_path: Path, predictions_path: Path) -> Optional[Path]:
    """
    Alias para compatibilidad con pruebas de integración.
    Internamente llama a run_inference.
    """
    return run_inference(features_path, model_path, predictions_path)


if __name__ == "__main__":
    app()
