# src/main.py

import mlflow
import warnings
from src.modeling.data_processor import DataProcessor
from src.modeling.ml_experiment import MLExperiment, MODEL_MAP, PARAM_GRIDS

# --- Configuración General ---
DATA_PATH = './data/interim/bike_sharing_cleaned.csv'
EXPERIMENT_NAME = "Bike_Sharing_MLOps_Project"
RANDOM_STATE = 42

# Suprimir las advertencias para una salida más limpia
warnings.filterwarnings("ignore")

def main():
    """
    Ejecuta el pipeline de Machine Learning, incluyendo:
    1. Carga y preprocesamiento de datos.
    2. Comparación inicial con LazyRegressor (Top 3).
    3. Optimización de hiperparámetros con GridSearchCV y registro en MLflow.
    """
    
    print("--- 1. Preparando Datos ---")
    data_proc = DataProcessor(DATA_PATH, random_state=RANDOM_STATE)
    
    # X_trainval/y_trainval es el 80% (usado para entrenamiento en GS)
    # X_test/y_test es el 20% (usado para la evaluación final)
    # X_train/X_val/y_train/y_val es la división 72/8 (usada solo en LazyRegressor)
    try:
        X_trainval, X_test, y_trainval, y_test, X_train, X_val, y_train, y_val = data_proc.load_and_split()
    except Exception as e:
        print(f"No se pudo cargar y dividir los datos. Terminando. Error: {e}")
        return

    ml_exp = MLExperiment(EXPERIMENT_NAME, random_state=RANDOM_STATE)

    # --- 2. Comparación Inicial con LazyRegressor ---
    print("\n--- 2. Ejecutando LazyRegressor para Screeening Inicial ---")
    top_models = ml_exp.run_lazypredict(X_train, X_val, y_train, y_val, top_n=3)

    # --- 3. Optimización de los 3 Mejores Modelos ---
    print("\n--- 3. Optimizando Modelos Seleccionados con GridSearchCV ---")
    for model_name in top_models:
        if model_name in MODEL_MAP:
            print(f"\nProcesando modelo: {model_name}")
            
            # Si LazyPredict devuelve otro modelo, solo lo optimizaremos si tenemos un grid definido.
            if model_name in PARAM_GRIDS:
                ml_exp.run_grid_search(
                    model_name, 
                    MODEL_MAP[model_name], 
                    PARAM_GRIDS[model_name], 
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
    main()