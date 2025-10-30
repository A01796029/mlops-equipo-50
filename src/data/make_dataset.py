# src/data/make_dataset.py

import os
import pandas as pd
# Importamos la clase de limpieza que creamos en el paso anterior
from src.data.cleaning import DataCleaningPipeline

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# Definiciones de ruta basadas en la estructura Cookiecutter (asumiendo que este
# script se ejecuta desde la raíz del proyecto o desde la carpeta 'src/data')
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'hour.csv') # Ajusta el nombre del archivo si es diferente
CLEANED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'interim', 'bike_sharing_cleaned.csv')

# Columnas finales esperadas para verificación (puedes ajustar esta lista)
FINAL_COLUMNS = [
    'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 
    'casual', 'registered', 'cnt'
]

# =============================================================================
# FUNCIONES DE ORQUESTACIÓN
# =============================================================================

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """Guarda un DataFrame en un archivo CSV."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"\n✅ Datos limpios guardados exitosamente en: {file_path}")
    except Exception as e:
        print(f"\n❌ Error al intentar guardar el DataFrame: {e}")

def run_cleaning_pipeline(raw_data_path: str, expected_columns: list) -> pd.DataFrame | None:
    """
    Orquesta y ejecuta el pipeline de limpieza de datos utilizando la clase POO.

    Args:
        raw_data_path (str): La ruta al archivo CSV crudo.
        expected_columns (list): La lista de columnas finales esperadas (para verificación).

    Returns:
        pd.DataFrame | None: Un DataFrame limpio y procesado, o None si la carga inicial falla.
    """
    print("===============================================================")
    print("       INICIANDO ORQUESTACIÓN DEL PIPELINE DE LIMPIEZA         ")
    print("===============================================================")

    # 1. Instanciar la clase de limpieza
    pipeline = DataCleaningPipeline(raw_data_path=raw_data_path)
    
    # 2. Ejecutar el método principal del pipeline (contiene todos los pasos)
    cleaned_df = pipeline.ejecutar_pipeline()
    
    if cleaned_df is None:
        print("\n❌ El pipeline no generó un resultado debido a un error en la carga de datos.")
        return None
        
    # 3. Verificación final de columnas
    if not all(col in cleaned_df.columns for col in expected_columns):
        missing = [col for col in expected_columns if col not in cleaned_df.columns]
        print(f"\n⚠️ Advertencia: Faltan columnas esperadas después de la limpieza: {missing}")

    print("\n✅ Orquestación de limpieza completada. DataFrame listo para guardado.")
    return cleaned_df

# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    """Función principal para ejecutar el script de orquestación."""
    
    # Ejecutar la limpieza
    cleaned_df = run_cleaning_pipeline(
        raw_data_path=RAW_DATA_PATH,
        expected_columns=FINAL_COLUMNS
    )

    if cleaned_df is not None:
        # Guardar el resultado en la carpeta 'interim'
        save_dataframe_to_csv(df=cleaned_df, file_path=CLEANED_DATA_PATH)
    
    print("\n---------------------------------------------------------------")
    print("                   FIN DEL PROCESO make_dataset                  ")
    print("---------------------------------------------------------------")

if __name__ == '__main__':
    main()