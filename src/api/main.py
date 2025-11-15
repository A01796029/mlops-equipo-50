# src/api/main.py

import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List

# Importamos nuestros schemas de Pydantic
from .schemas import PredictionInput, PredictionOutput

# --- Configuración de la Carga del Modelo ---

# Usamos variables de entorno para configurar MLflow
# Esto nos permitirá cambiarlo en Docker
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Esta es la URI del modelo en el Registro de MLflow
MODEL_URI = "models:/Bike_Sharing_MLOps_Project/1"

# Columnas que deben ser tratadas como categóricas
# (Basado en tu 'data_processor.py' y 'pipeline.py')
CATEGORICAL_COLS = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 
    'weekday', 'workingday', 'weathersit'
]

# Un diccionario global para mantener el modelo cargado
model_registry = {}

# --- 1. Ciclo de Vida de la App: Cargar el Modelo al Inicio ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Función de 'lifespan' para cargar el modelo de MLflow al iniciar
    la aplicación y mantenerlo en memoria.
    """
    print("Iniciando aplicación y cargando modelo...")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Cargamos el pipeline completo (preprocesador + modelo)
        loaded_model = mlflow.pyfunc.load_model(model_uri=MODEL_URI)
        
        model_registry["model"] = loaded_model
        model_registry["version"] = MODEL_URI # Guardamos la URI para logging
        
        print(f"Modelo {MODEL_URI} cargado exitosamente.")
        
    except Exception as e:
        print(f"Error crítico al cargar el modelo: {e}")
        # Si el modelo no carga, la API no debe iniciar
        raise RuntimeError(f"No se pudo cargar el modelo al inicio: {e}")

    yield # Aquí es donde la aplicación se ejecuta
    
    # Limpiar al apagar (si es necesario)
    print("Apagando aplicación...")
    model_registry.clear()

# --- 2. Inicialización de la App FastAPI ---

app = FastAPI(
    title="API de Predicción de Renta de Bicicletas",
    description="Proyecto MLOps: Sirve un modelo para predecir 'cnt' (conteo de bicis).",
    version="1.0.0",
    lifespan=lifespan # Asocia el ciclo de vida
)

# --- 3. Endpoints de la API ---

@app.get("/", tags=["Health Check"])
async def root():
    """
    Endpoint de 'Health Check' para verificar que la API está viva.
    """
    return {
        "status": "ok", 
        "model_version": model_registry.get("version", "loading...")
    }

@app.post("/predict", 
          response_model=PredictionOutput, # Usa el schema de salida
          tags=["Predictions"])
async def predict(input_data: PredictionInput): # Usa el schema de entrada
    """
    Recibe datos de entrada, los procesa con el pipeline de MLflow
    y devuelve una predicción.
    """
    
    # 3.1. Validar que el modelo esté cargado
    if "model" not in model_registry:
        raise HTTPException(status_code=503, detail="Modelo no cargado. La API aún está iniciando.")
        
    try:
        # 3.2. Convertir la entrada Pydantic a un DataFrame de Pandas
        # MLflow pyfunc espera un DataFrame como entrada
        input_df = pd.DataFrame([input_data.dict()])

        # 3.3. ¡Paso Crítico de Dtypes!
        # El pipeline (ColumnTransformer) espera dtypes 'category'
        # para aplicar el OneHotEncoder correctamente.
        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')
        
        # 3.4. Realizar la predicción
        # 'model.predict()' aquí ejecuta el pipeline COMPLETO:
        # (Imputer -> Scaler -> OHE -> Modelo)
        prediction = model_registry["model"].predict(input_df)
        
        # 3.5. Formatear la salida
        # El resultado de 'prediction' es un array (ej. [250.7]), 
        # extraemos el primer valor.
        result = prediction[0]
        
        # Devolvemos el JSON usando nuestro schema de Pydantic
        return PredictionOutput(
            prediction=float(result),
            model_version=model_registry.get("version")
        )

    except Exception as e:
        # Manejo de errores durante la predicción
        print(f"Error durante la predicción: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno al procesar la predicción: {str(e)}"
        )