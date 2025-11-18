# src/api/main.py

import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from loguru import logger
import sys

# Importamos nuestros schemas de Pydantic
from .schemas import PredictionInput, PredictionOutput

# Configurar logging
logger.remove()  # Remover handler por defecto
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

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
    
    Este contexto manager se ejecuta al inicio y al final del ciclo de vida
    de la aplicación FastAPI, asegurando la correcta gestión de recursos.
    """
    logger.info("🚀 Iniciando aplicación y cargando modelo...")
    logger.info(f"📍 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"🎯 Model URI: {MODEL_URI}")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("✅ MLflow tracking URI configurado correctamente")
        
        # Cargamos el pipeline completo (preprocesador + modelo)
        logger.info("⏳ Cargando modelo desde MLflow...")
        loaded_model = mlflow.pyfunc.load_model(model_uri=MODEL_URI)
        
        model_registry["model"] = loaded_model
        model_registry["version"] = MODEL_URI # Guardamos la URI para logging
        
        logger.success(f"✅ Modelo {MODEL_URI} cargado exitosamente en memoria")
        logger.info(f"📊 Columnas categóricas configuradas: {CATEGORICAL_COLS}")
        
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar el modelo: {e}")
        logger.exception("Stack trace completo:")
        # Si el modelo no carga, la API no debe iniciar
        raise RuntimeError(f"No se pudo cargar el modelo al inicio: {e}")

    yield # Aquí es donde la aplicación se ejecuta
    
    # Limpiar al apagar (si es necesario)
    logger.info("🛑 Apagando aplicación y liberando recursos...")
    model_registry.clear()
    logger.success("✅ Aplicación apagada correctamente")

# --- 2. Inicialización de la App FastAPI ---

app = FastAPI(
    title="API de Predicción de Renta de Bicicletas",
    description="Proyecto MLOps: Sirve un modelo para predecir 'cnt' (conteo de bicis).",
    version="1.0.0",
    lifespan=lifespan # Asocia el ciclo de vida
)

# --- 3. Endpoints de la API ---

@app.get("/", tags=["Health Check"])
async def root() -> Dict[str, Any]:
    """
    Endpoint de 'Health Check' para verificar que la API está viva.
    
    Returns:
        Dict con el estado de la API y la versión del modelo cargado.
    """
    model_loaded = "model" in model_registry
    logger.debug(f"Health check solicitado - Modelo cargado: {model_loaded}")
    
    return {
        "status": "ok" if model_loaded else "starting",
        "model_version": model_registry.get("version", "loading..."),
        "model_loaded": model_loaded,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }

@app.post("/predict", 
          response_model=PredictionOutput, # Usa el schema de salida
          tags=["Predictions"])
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Recibe datos de entrada, los procesa con el pipeline de MLflow
    y devuelve una predicción.
    
    Args:
        input_data: Datos de entrada validados por Pydantic (PredictionInput)
    
    Returns:
        PredictionOutput con la predicción y versión del modelo
        
    Raises:
        HTTPException: Si el modelo no está cargado (503) o si hay error en predicción (500)
    """
    
    # 3.1. Validar que el modelo esté cargado
    if "model" not in model_registry:
        logger.error("❌ Intento de predicción con modelo no cargado")
        raise HTTPException(
            status_code=503, 
            detail="Modelo no cargado. La API aún está iniciando."
        )
    
    logger.info(f"📥 Nueva solicitud de predicción recibida")
    logger.debug(f"Datos de entrada: {input_data.dict()}")
        
    try:
        # 3.2. Convertir la entrada Pydantic a un DataFrame de Pandas
        # MLflow pyfunc espera un DataFrame como entrada
        input_df = pd.DataFrame([input_data.dict()])
        logger.debug(f"DataFrame creado con shape: {input_df.shape}")

        # 3.3. ¡Paso Crítico de Dtypes!
        # El pipeline (ColumnTransformer) espera dtypes 'category'
        # para aplicar el OneHotEncoder correctamente.
        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')
        
        logger.debug("✅ Dtypes categóricos aplicados correctamente")
        
        # 3.4. Realizar la predicción
        # 'model.predict()' aquí ejecuta el pipeline COMPLETO:
        # (Imputer -> Scaler -> OHE -> Modelo)
        logger.debug("🔮 Ejecutando predicción del modelo...")
        prediction = model_registry["model"].predict(input_df)
        
        # 3.5. Formatear la salida
        # El resultado de 'prediction' es un array (ej. [250.7]), 
        # extraemos el primer valor.
        result = prediction[0]
        logger.info(f"✅ Predicción exitosa: {result:.2f}")
        
        # Devolvemos el JSON usando nuestro schema de Pydantic
        return PredictionOutput(
            prediction=float(result),
            model_version=model_registry.get("version")
        )

    except ValueError as e:
        # Errores de validación de datos
        logger.error(f"❌ Error de validación en los datos de entrada: {e}")
        logger.exception("Stack trace:")
        raise HTTPException(
            status_code=422, 
            detail=f"Error de validación en los datos: {str(e)}"
        )
    except Exception as e:
        # Manejo de errores durante la predicción
        logger.error(f"❌ Error inesperado durante la predicción: {e}")
        logger.exception("Stack trace completo:")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno al procesar la predicción: {str(e)}"
        )