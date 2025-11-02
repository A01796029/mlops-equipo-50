# src/modeling/train.py
from pathlib import Path
from typing import Optional

from loguru import logger
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import simulate_progress

app = typer.Typer()

def train_model(features_path: Path, labels_path: Path, model_path: Path) -> Optional[Path]:
    """
    Entrena el modelo y guarda el artefacto en model_path.
    (Placeholder; implementar training real con sklearn/xgboost/etc.)
    """
    logger.info("Starting model training...")
    simulate_progress("Training model")
    # Ejemplo real:
    # X = pd.read_csv(features_path)
    # y = pd.read_csv(labels_path)
    # model = SomeModel().fit(X, y)
    # joblib.dump(model, model_path)
    logger.success(f"Modeling training complete. Model saved to {model_path}")
    return model_path


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
) -> None:
    train_model(features_path, labels_path, model_path)


if __name__ == "__main__":
    app()
