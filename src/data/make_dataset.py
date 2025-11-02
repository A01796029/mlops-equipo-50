# src/data/make_dataset.py

from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.data.cleaning import DataCleaningPipeline
from src.utils import save_dataframe_to_csv

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH: Path = ROOT_DIR / "data" / "raw" / "hour.csv"
CLEANED_DATA_PATH: Path = ROOT_DIR / "data" / "interim" / "bike_sharing_cleaned.csv"

FINAL_COLUMNS: List[str] = [
    'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
    'casual', 'registered', 'cnt'
]

# =============================================================================
# FUNCIONES DE ORQUESTACIÓN
# =============================================================================

def run_cleaning_pipeline(raw_data_path: Path, expected_columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Orquesta y ejecuta el pipeline de limpieza de datos utilizando la clase POO.

    Args:
        raw_data_path: ruta al CSV crudo (Path).
        expected_columns: lista de columnas finales esperadas para verificación.

    Returns:
        DataFrame limpio o None si ocurre un fallo en la carga.
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

def main() -> None:
    """Función principal para ejecutar el script de orquestación."""
    cleaned_df = run_cleaning_pipeline(
        raw_data_path=RAW_DATA_PATH,
        expected_columns=FINAL_COLUMNS
    )

    if cleaned_df is not None:
        try:
            save_dataframe_to_csv(cleaned_df, CLEANED_DATA_PATH)
            logger.success("Datos limpios guardados exitosamente en: {}", CLEANED_DATA_PATH)
        except Exception as e:
            logger.exception("Error al intentar guardar el DataFrame: {}", e)

    logger.info("-" * 63)
    logger.info("FIN DEL PROCESO make_dataset")
    logger.info("-" * 63)


if __name__ == "__main__":
    main()
