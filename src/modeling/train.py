# src/modeling/train.py
from pathlib import Path
from typing import Optional, Union
from loguru import logger
import pandas as pd
import joblib
import numpy as np
import typer
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import simulate_progress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = typer.Typer()


def train_model(df: pd.DataFrame, model_path=None):
    logger.info("Starting model training...")
    simulate_progress("Training model")

    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot train model.")

    logger.debug("Using in-memory DataFrame for training.")

    X = df[["temp", "hum"]]
    y = df["cnt"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    results = {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "mse": float(mse),
    }

    # Guarda el modelo si se especifica un path
    if model_path:
        import joblib
        joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}")

    return results


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
) -> None:
    train_model(features_path, labels_path, model_path)


if __name__ == "__main__":
    app()