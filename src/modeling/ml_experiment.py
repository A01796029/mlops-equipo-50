# src/ml_experiment.py

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import r2_score

from src.modeling.data_processor import calculate_rmse, calculate_metrics

class MLExperiment:
    """
    Orquesta los experimentos de Machine Learning, registrando todo con MLflow.
    """
    
    def __init__(self, experiment_name: str, random_state: int = 117):
        self.random_state = random_state
        mlflow.set_experiment(experiment_name)
        
    def _log_plots(self, y_true, y_pred, model_name, subset, run_id):
        """Genera y registra las gráficas de Actual vs. Predicted como artefactos."""
        
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, marker='o', alpha=0.6)
        # Línea de y=x para referencia perfecta
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--') 
        plt.title(f'Actual vs. Predicted ({subset}) - {model_name}')
        plt.xlabel('Actual Values (y_true)')
        plt.ylabel('Predicted Values (y_pred)')
        
        # Guarda y registra la gráfica
        plot_path = f"{model_name}_{subset}_vs_pred.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, "plots", run_id=run_id)
        plt.close()

    def run_lazypredict(self, X_train, X_val, y_train, y_val, top_n: int = 10):
        """
        Ejecuta LazyRegressor para una comparación inicial y registra los resultados.
        Devuelve el nombre de los top_n modelos.
        """
        run_name = "LazyRegressor_Initial_Screening"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tag("stage", "Initial Screening")
            
            try:
                # Convert to numpy arrays and ensure no mixed types
                import numpy as np
                
                # Handle mixed type columns and convert to numeric
                for col in X_train.columns:
                    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                        # Convert to string first, then to category codes
                        X_train[col] = pd.Categorical(X_train[col].astype(str)).codes
                        X_val[col] = pd.Categorical(X_val[col].astype(str)).codes
                
                # Ensure y is numeric
                y_train = pd.to_numeric(y_train, errors='coerce')
                y_val = pd.to_numeric(y_val, errors='coerce')
                
                # Drop any remaining NaN values
                train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
                
                X_train_clean = X_train[train_mask].copy()
                y_train_clean = y_train[train_mask].copy()
                X_val_clean = X_val[val_mask].copy()
                y_val_clean = y_val[val_mask].copy()
                
                print(f"  → Datos limpios: Train={X_train_clean.shape}, Val={X_val_clean.shape}")
                
                reg = LazyRegressor(ignore_warnings=True, random_state=self.random_state, verbose=0)
                print(f"Iniciando LazyRegressor. Evaluando en set de Validación...")
                
                # Entrenar en X_train y evaluar en X_val
                models_df, _ = reg.fit(X_train_clean, X_val_clean, y_train_clean, y_val_clean)
                top_models_df = models_df.head(top_n).reset_index()
                
                # 1. Registrar Parámetros y Resultados
                mlflow.log_param("dataset_split", "Train (90%) vs. Validation (10%)")
                
                # Registrar métricas individuales para los top_n modelos
                for index, row in top_models_df.iterrows():
                    model_name = row['Model']
                    mlflow.log_metric(f"R2_VAL_{model_name}", row['R-Squared'])
                    mlflow.log_metric(f"RMSE_VAL_{model_name}", row['RMSE'])
                    mlflow.log_param(f"TOP_{index+1}_MODEL", model_name)
                
                # 2. Registrar el DataFrame de resultados como artefacto
                top_models_df.to_csv("lazy_regressor_top10.csv", index=False)
                mlflow.log_artifact("lazy_regressor_top10.csv", "results_data", run_id=run_id)
                
                print(f"LazyRegressor Run ID: {run_id}")
                print(f"Top {top_n} modelos encontrados: {top_models_df['Model'].tolist()}")
                
                return top_models_df['Model'].tolist()[:top_n]
                
            except Exception as e:
                print(f"Error en LazyRegressor: {e}")
                mlflow.log_param("status", f"FAILED: {str(e)}")
                # Return empty list if LazyRegressor fails
                return []

    def run_grid_search(self, model_name: str, model_instance, param_grid: dict, X_trainval, y_trainval, X_test, y_test):
        """
        Ejecuta GridSearchCV para un modelo, registra el mejor estimador y sus métricas finales
        en el set de prueba.
        """
        run_name = f"GridSearch_{model_name}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tag("stage", "Hyperparameter Tuning")
            mlflow.set_tag("base_model", model_name)

            print(f"\n--- Optimizando {model_name} con GridSearchCV ---")
            
            # Registrar el grid de parámetros
            mlflow.log_param("param_grid", str(param_grid))
            
            grid_search = GridSearchCV(
                estimator=model_instance, 
                param_grid=param_grid, 
                scoring='neg_mean_squared_error',
                cv=2,  # Reduced from 3 to 2 for faster training
                verbose=1, 
                n_jobs=-1
            )
            
            # Entrenamiento usando el set combinado Train+Validation (X_trainval)
            try:
                grid_search.fit(X_trainval, y_trainval)
            except ValueError as e:
                print(f"Error al entrenar {model_name}: {e}. Saltando...")
                mlflow.log_param("status", f"FAILED - {e}")
                return

            best_estimator = grid_search.best_estimator_
            
            # --- 1. Registrar Parámetros y Métricas del Mejor Estimador (CV Score) ---
            mlflow.log_param("status", "SUCCESS")
            mlflow.log_param("best_params", str(grid_search.best_params_))
            mlflow.log_metric("best_neg_mean_squared_error_cv", grid_search.best_score_)
            
            # Registrar todos los parámetros del mejor estimador
            best_params_full = best_estimator.get_params()
            mlflow.log_params(best_params_full)

            # --- 2. Evaluación Final en el Set de Prueba (X_test) ---
            y_train_pred = best_estimator.predict(X_trainval)
            y_test_pred = best_estimator.predict(X_test)
            
            train_metrics = calculate_metrics(y_trainval, y_train_pred)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            
            # Registrar métricas finales del set de PRUEBA
            mlflow.log_metrics({
                "final_r2_train": train_metrics['r2'],
                "final_rmse_test": test_metrics['rmse'],
                "final_mae_test": test_metrics['mae'],
                "final_r2_test": test_metrics['r2']
            })

            # --- 3. Registrar Gráficos ---
            self._log_plots(y_trainval, y_train_pred, model_name, "TRAINVAL", run_id)
            self._log_plots(y_test, y_test_pred, model_name, "TEST", run_id)

            # --- 4. Registrar el Mejor Modelo ---
            mlflow.sklearn.log_model(best_estimator, "best_model_gs")
            
            print(f"Optimización y evaluación de {model_name} registrada. R2 Test: {test_metrics['r2']:.4f}")
            print(f"Run ID: {run_id}")


# ----------------------------------------------------------------------
# Modelos y Grids para la Optimización
# ----------------------------------------------------------------------

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODEL_MAP = {
    'MLPRegressor': MLPRegressor(random_state=117, max_iter=300),
    'LGBMRegressor': LGBMRegressor(random_state=117, verbose=-1),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=117),
    'Ridge': Ridge(random_state=117)  # Lighter model
}

PARAM_GRIDS = {
    'MLPRegressor': {
        'activation': ['tanh'],  # Best from your results
        'hidden_layer_sizes': [(100,)],  # Best from your results
    },
    'LGBMRegressor': {
        'learning_rate': [0.1],  # Best from your results
        'max_depth': [-1],  # Best from your results (unlimited)
        'n_estimators': [100],  # Best from your results
    },
    'ExtraTreesRegressor': {
        'n_estimators': [100],  # From your results
        'max_depth': [10],  # From your results
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0],  # Simple linear model, very fast
    }
}