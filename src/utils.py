from pathlib import Path
from typing import Any
from loguru import logger
from tqdm import tqdm
import pandas as pd
import os

def simulate_progress(task_name: str, steps: int = 10) -> None:
    """
    Simula un progreso para tareas dummy (útil para ejemplos / placeholders).

    Args:
        task_name: nombre de la tarea a mostrar en logs.
        steps: número de pasos para el progress bar.
    """
    logger.info(f"Starting: {task_name}")
    for i in tqdm(range(steps), total=steps):
        if i == steps // 2:
            logger.warning(f"Midway event in {task_name}")
    logger.success(f"{task_name} complete.")


def save_dataframe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Guarda un DataFrame en CSV asegurando que la carpeta destino exista.

    Args:
        df: DataFrame a guardar.
        path: ruta destino.
        index: si se guarda o no el índice.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Saved DataFrame to {path}")


def ensure_path(path: Path) -> None:
    """Crea el directorio padre de la ruta si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
