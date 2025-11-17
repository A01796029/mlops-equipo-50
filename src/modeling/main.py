# src/modeling/main.py

from datetime import datetime
from pathlib import Path

import mlflow
import warnings
import typer

from src.modeling.data_processor import DataProcessor
from src.modeling.pipeline import build_pipeline, infer_feature_types
from src.modeling.ml_experiment import MLExperiment, MODEL_MAP, PARAM_GRIDS
from src.modeling.data_drift import (
    DRIFT_THRESHOLDS,
    detect_drift,
    evaluate_performance_drift,
    generate_monitoring_dataset,
    drift_results_as_dataframe,
    log_drift_summary,
    plot_metric_comparison,
    save_drift_report,
)
from src.config import INTERIM_DATA_DIR, REPORTS_DIR

# --- Configuración General ---
DEFAULT_DATA_PATH = INTERIM_DATA_DIR / "bike_sharing_cleaned.csv"
EXPERIMENT_NAME = "Bike_Sharing_MLOps_Project"
RANDOM_STATE = 42

# Suprimir las advertencias para una salida más limpia
warnings.filterwarnings("ignore")

# Crear la App de Typer
app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context, # Argumento de contexto requerido para callback
    input_path: Path = typer.Option(
        DEFAULT_DATA_PATH, 
        "--input-path", "-i",
        help="Ruta al archivo CSV limpio (de 'interim')."
    )
):
    """
    Ejecuta el pipeline de Machine Learning, incluyendo:
    1. Carga y preprocesamiento de datos.
    2. Comparación inicial con LazyRegressor (Top 3).
    3. Optimización de hiperparámetros con GridSearchCV y registro en MLflow.
    """
    
    print("--- 1. Preparando Datos ---")
    data_proc = DataProcessor(str(input_path), random_state=RANDOM_STATE)
    
    try:
        X_trainval, X_test, y_trainval, y_test, X_train, X_val, y_train, y_val = data_proc.load_and_split()
    except Exception as e:
        print(f"No se pudo cargar y dividir los datos. Terminando. Error: {e}")
        return

    print("\n--- 1b. Analizando Data Drift entre Train/Validation y Test ---")
    num_cols, cat_cols = infer_feature_types(X_trainval)
    pre_drift_results = detect_drift(X_trainval, X_test, num_cols, cat_cols)
    log_drift_summary(pre_drift_results)
    pre_drift_df = drift_results_as_dataframe(pre_drift_results)

    print("\n--- 1c. Simulando data drift y evaluando impacto en performance ---")
    monitoring_features = generate_monitoring_dataset(X_val, random_state=RANDOM_STATE)
    monitoring_target = y_val.copy()

    print("\n--- 1d. Verificando drift entre Validación y lote simulado ---")
    post_drift_results = detect_drift(X_val, monitoring_features, num_cols, cat_cols)
    log_drift_summary(post_drift_results)
    post_drift_df = drift_results_as_dataframe(post_drift_results)

    baseline_model = build_pipeline(MODEL_MAP['RandomForestRegressor'], X_trainval)
    baseline_model.fit(X_trainval, y_trainval)

    performance_report = evaluate_performance_drift(
        model=baseline_model,
        baseline_X=X_val,
        baseline_y=y_val,
        monitoring_X=monitoring_features,
        monitoring_y=monitoring_target,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_dir = REPORTS_DIR / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    monitoring_path = drift_dir / f"monitoring_batch_{timestamp}.csv"
    monitoring_features.assign(cnt=monitoring_target.values).to_csv(monitoring_path, index=False)

    summary_path = drift_dir / f"drift_report_{timestamp}.json"
    figure_path = drift_dir / f"drift_metrics_{timestamp}.png"
    feature_drift_pre_path = drift_dir / f"feature_drift_pre_{timestamp}.csv"
    feature_drift_post_path = drift_dir / f"feature_drift_post_{timestamp}.csv"
    if not pre_drift_df.empty:
        pre_drift_df.to_csv(feature_drift_pre_path, index=False)
    else:
        feature_drift_pre_path = None

    if not post_drift_df.empty:
        post_drift_df.to_csv(feature_drift_post_path, index=False)
    else:
        feature_drift_post_path = None
    save_drift_report(performance_report, summary_path)
    plot_metric_comparison(performance_report, figure_path)

    print(
        f"Alert triggered: {performance_report.alert_triggered}. Acción sugerida: {performance_report.recommended_action}"
    )

    # Registrar resultados del monitoreo en MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="DataDriftMonitoring"):
        mlflow.set_tag("stage", "Data Drift Monitoring")
        mlflow.log_params({
            "alert_triggered": str(performance_report.alert_triggered),
            "recommended_action": performance_report.recommended_action,
        })
        mlflow.log_params({f"threshold_{k}": v for k, v in DRIFT_THRESHOLDS.items()})

        mlflow.log_metrics({f"baseline_{k}": v for k, v in performance_report.baseline_metrics.items()})
        mlflow.log_metrics({f"monitoring_{k}": v for k, v in performance_report.monitoring_metrics.items()})
        mlflow.log_metrics({f"delta_{k}": v for k, v in performance_report.deltas.items()})

        mlflow.log_artifact(summary_path, artifact_path="drift")
        mlflow.log_artifact(figure_path, artifact_path="drift")
        mlflow.log_artifact(monitoring_path, artifact_path="drift")
        for path in (feature_drift_pre_path, feature_drift_post_path):
            if path:
                mlflow.log_artifact(path, artifact_path="drift")

    ml_exp = MLExperiment(EXPERIMENT_NAME, random_state=RANDOM_STATE)

    # # --- 2. Comparación Inicial con LazyRegressor ---
    # print("\n--- 2. Ejecutando LazyRegressor para Screeening Inicial ---")
    # top_models = ml_exp.run_lazypredict(X_train, X_val, y_train, y_val, top_n=3)

    # --- 3. Optimización de Modelos ---
    print("\n--- 3. Optimizando Modelos Seleccionados con GridSearchCV ---")

    models_to_train = ['XGBRegressor', 'RandomForestRegressor'] 
    print(f"Modelos a entrenar: {models_to_train}")

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