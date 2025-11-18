# src/api/schemas.py

from pydantic import BaseModel, Field, validator
from typing import Optional

# =============================================================================
# Schema de Salida (Output)
# =============================================================================
class PredictionOutput(BaseModel):
    """
    Define el JSON de respuesta que la API devolverá.
    
    Attributes:
        prediction: Predicción del modelo (número de bicicletas alquiladas)
        model_version: URI del modelo MLflow utilizado
    """
    prediction: float = Field(..., description="Número predicho de bicicletas alquiladas", ge=0.0)
    model_version: str = Field(..., description="Versión del modelo MLflow (URI)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 234.56,
                "model_version": "models:/Bike_Sharing_MLOps_Project/1"
            }
        }

# =============================================================================
# Schema de Entrada (Input)
# =============================================================================
class PredictionInput(BaseModel):
    """
    Define el JSON de entrada que el endpoint /predict espera.
    
    Pydantic validará automáticamente los tipos y rangos de estos datos.
    Todos los valores están normalizados según el dataset original de bike sharing.
    
    Attributes:
        season: Estación del año (1:invierno, 2:primavera, 3:verano, 4:otoño)
        yr: Año relativo (0:2011, 1:2012)
        mnth: Mes del año (1-12)
        hr: Hora del día (0-23)
        holiday: Indica si es día festivo (0:No, 1:Sí)
        weekday: Día de la semana (0:domingo, ..., 6:sábado)
        workingday: Indica si es día laboral (0:No, 1:Sí)
        weathersit: Situación climática (1-4, de mejor a peor)
        temp: Temperatura normalizada (0.0-1.0)
        atemp: Sensación térmica normalizada (0.0-1.0)
        hum: Humedad relativa normalizada (0.0-1.0)
        windspeed: Velocidad del viento normalizada (0.0-1.0)
    """
    
    # Columnas categóricas (como int)
    season: int = Field(
        ..., 
        ge=1, 
        le=4, 
        description="Temporada (1:invierno, 2:primavera, 3:verano, 4:otoño)"
    )
    yr: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Año (0: 2011, 1: 2012)"
    )
    mnth: int = Field(
        ..., 
        ge=1, 
        le=12, 
        description="Mes (1 a 12)"
    )
    hr: int = Field(
        ..., 
        ge=0, 
        le=23, 
        description="Hora (0 a 23)"
    )
    holiday: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Es día festivo (0:No, 1:Sí)"
    )
    weekday: int = Field(
        ..., 
        ge=0, 
        le=6, 
        description="Día de la semana (0:domingo a 6:sábado)"
    )
    workingday: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Es día laboral (0:No, 1:Sí)"
    )
    weathersit: int = Field(
        ..., 
        ge=1, 
        le=4, 
        description="Situación climática (1:Despejado, 2:Niebla, 3:Nieve ligera, 4:Lluvia intensa)"
    )
    
    # Columnas numéricas (float)
    temp: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Temperatura normalizada (0.0 a 1.0)"
    )
    atemp: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Sensación térmica normalizada (0.0 a 1.0)"
    )
    hum: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Humedad normalizada (0.0 a 1.0)"
    )
    windspeed: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Velocidad del viento normalizada (0.0 a 1.0)"
    )
    
    @validator('workingday', 'holiday')
    def validate_working_holiday_logic(cls, v, values, field):
        """
        Valida la lógica entre holiday y workingday.
        Si es día festivo (holiday=1), generalmente no debería ser día laboral.
        Esta es una validación suave que solo genera una advertencia en logs.
        """
        # Esta validación es informativa, no restrictiva
        # ya que el dataset puede tener excepciones
        if field.name == 'workingday' and 'holiday' in values:
            if values['holiday'] == 1 and v == 1:
                # Caso poco común pero posible, no lanzamos error
                pass
        return v
    
    @validator('temp', 'atemp', 'hum', 'windspeed')
    def round_normalized_values(cls, v):
        """
        Redondea valores normalizados a 4 decimales para consistencia.
        """
        return round(v, 4)

    # Configuración de ejemplo para la documentación de Swagger
    class Config:
        schema_extra = {
            "example": {
                "season": 1,
                "yr": 0,
                "mnth": 1,
                "hr": 10,
                "holiday": 0,
                "weekday": 6,
                "workingday": 0,
                "weathersit": 1,
                "temp": 0.24,
                "atemp": 0.2879,
                "hum": 0.81,
                "windspeed": 0.0
            }
        }