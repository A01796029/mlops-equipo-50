# src/modeling/main.py

import mlflow
import warnings
import typer
from pathlib import Path
from datetime import datetime

from src.modeling.data_processor import DataProcessor
from src.modeling.pipeline import build_pipeline
from src.modeling.ml_experiment import MLExperiment, MODEL_MAP, PARAM_GRIDS
from src.config import INTERIM_DATA_DIR, RANDOM_SEEDS

# --- Configuración General ---
DEFAULT_DATA_PATH = INTERIM_DATA_DIR / "bike_sharing_cleaned.csv"
# Add timestamp to experiment name for unique experiments each run
EXPERIMENT_NAME = f"Bike_Sharing_MLOps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Suprimir las advertencias para una salida más limpia
warnings.filterwarnings("ignore")

# Crear la App de Typer
app = typer.Typer()

@app.command()
def main(
    input_path: Path = DEFAULT_DATA_PATH,
    models: str = "auto"
):
    """
    Ejecuta el pipeline de Machine Learning, incluyendo:
    1. Carga y preprocesamiento de datos.
    2. Comparación inicial con LazyRegressor (Top 3).
    3. Optimización de hiperparámetros con GridSearchCV y registro en MLflow.
    """
    
    # Configurar todas las semillas para reproducibilidad total
    RANDOM_SEEDS.set_all_seeds()
    
    print("--- 1. Preparando Datos ---")
    data_proc = DataProcessor(str(input_path))
    
    try:
        X_trainval, X_test, y_trainval, y_test, X_train, X_val, y_train, y_val = data_proc.load_and_split()
    except Exception as e:
        print(f"No se pudo cargar y dividir los datos. Terminando. Error: {e}")
        return

    ml_exp = MLExperiment(EXPERIMENT_NAME)

    # --- 2. Selección de Modelos (usando modelos optimizados) ---
    if models == "auto":
        print("\n--- 2. Usando modelos optimizados para entrenamiento rápido ---")
        # Using best models from previous experiments + Ridge (lightweight)
        models_to_train = ['MLPRegressor', 'LGBMRegressor', 'ExtraTreesRegressor', 'Ridge']
        print(f"Modelos a entrenar: {models_to_train}")
        print("(Skipping LazyRegressor para reducir tiempo de entrenamiento)")
    else:
        print("\n--- 2. Usando modelos especificados ---")
        models_to_train = [m.strip() for m in models.split(',')]
        print(f"Modelos a entrenar: {models_to_train}")

    # --- 3. Optimización de Modelos ---
    print("\n--- 3. Optimizando Modelos Seleccionados con GridSearchCV ---")

    for model_name in models_to_train:
        if model_name in MODEL_MAP:
            print(f"\nProcesando modelo: {model_name}")

            if model_name in PARAM_GRIDS:
                base_model = MODEL_MAP[model_name]
                # 3.1 Envolver el modelo en el pipeline de preprocesamiento
                wrapped_model = build_pipeline(base_model, X_trainval)

                # 3.2 Prefijar los hiperparámetros con 'model__' para el paso del modelo
                original_grid = PARAM_GRIDS[model_name]
                param_grid_prefixed = {f"model__{k}": v for k, v in original_grid.items()}

                # 3.3 Ejecutar GridSearch con el pipeline
                ml_exp.run_grid_search(
                    model_name,
                    wrapped_model,
                    param_grid_prefixed,
                    X_trainval,
                    y_trainval,
                    X_test,
                    y_test
                )
            else:
                print(f"Skipping optimization for {model_name}: No PARAM_GRIDS defined.")

    print("\n--- Pipeline de MLOps Completado ---")
    print("Ejecuta 'mlflow ui' para visualizar y comparar los resultados de todos los experimentos.")


if __name__ == "__main__":
    app()