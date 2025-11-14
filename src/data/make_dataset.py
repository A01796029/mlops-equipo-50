# src/data/make_dataset.py
# =============================================================================
# IMPORTS
# =============================================================================
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
import typer  # <--- 1. Importar Typer

from src.data.cleaning import DataCleaningPipeline
from src.utils import save_dataframe_to_csv
# Importamos las rutas desde la configuración central
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# Usamos las rutas de config.py para definir los *defaults*
DEFAULT_RAW_PATH: Path = RAW_DATA_DIR / "bike_sharing_modified.csv"
DEFAULT_CLEANED_PATH: Path = INTERIM_DATA_DIR / "bike_sharing_cleaned.csv"

FINAL_COLUMNS: List[str] = [
    'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
    'casual', 'registered', 'cnt'
]

# --- Crear la App de Typer ---
app = typer.Typer()

# =============================================================================
# FUNCIONES DE ORQUESTACIÓN
# =============================================================================

def run_cleaning_pipeline(raw_data_path: Path, expected_columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Orquesta y ejecuta el pipeline de limpieza de datos utilizando la clase POO.
    (Esta función interna no cambia)
    """
    logger.info("=" * 63)
    logger.info("INICIANDO ORQUESTACIÓN DEL PIPELINE DE LIMPIEZA")
    logger.info("=" * 63)

    pipeline = DataCleaningPipeline(raw_data_path=str(raw_data_path))
    cleaned_df = pipeline.ejecutar_pipeline()

    if cleaned_df is None:
        logger.error("El pipeline no generó un resultado debido a un error en la carga de datos.")
        return None

    # Verificación final de columnas
    missing = [col for col in expected_columns if col not in cleaned_df.columns]
    if missing:
        logger.warning("Faltan columnas esperadas después de la limpieza: {}", missing)
    else:
        logger.info("Todas las columnas esperadas están presentes.")

    logger.info("Orquestación de limpieza completada. DataFrame listo para guardado.")
    return cleaned_df


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

# --- Se convierte main() en un comando de Typer ---
@app.command()
def main(
    input_path: Path = typer.Option(
        DEFAULT_RAW_PATH, 
        "--input-path", "-i",
        help="Ruta al archivo CSV crudo (raw)."
    ),
    output_path: Path = typer.Option(
        DEFAULT_CLEANED_PATH, 
        "--output-path", "-o",
        help="Ruta para guardar el archivo CSV limpio (interim)."
    )
) -> None:
    """
    Función principal para ejecutar el script de orquestación de limpieza.
    Toma datos de 'input_path' y guarda el resultado en 'output_path'.
    """
    cleaned_df = run_cleaning_pipeline(
        raw_data_path=input_path, # <-- Usamos el argumento
        expected_columns=FINAL_COLUMNS
    )

    if cleaned_df is not None:
        try:
            save_dataframe_to_csv(cleaned_df, output_path) # <-- Usamos el argumento
            logger.success("Datos limpios guardados exitosamente en: {}", output_path)
        except Exception as e:
            logger.exception("Error al intentar guardar el DataFrame: {}", e)

    logger.info("-" * 63)
    logger.info("FIN DEL PROCESO make_dataset")
    logger.info("-" * 63)


def load_dataset() -> Optional[pd.DataFrame]:
    """
    Carga el dataset crudo, ejecuta la limpieza y devuelve el DataFrame limpio.
    Se mantiene por compatibilidad con pruebas de integración.
    """
    return run_cleaning_pipeline(raw_data_path=DEFAULT_RAW_PATH, expected_columns=FINAL_COLUMNS)


if __name__ == "__main__":
    app()