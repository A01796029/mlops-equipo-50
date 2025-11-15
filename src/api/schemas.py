# src/api/schemas.py

from pydantic import BaseModel, Field

# =============================================================================
# Schema de Salida (Output)
# =============================================================================
class PredictionOutput(BaseModel):
    """
    Define el JSON de respuesta que la API devolverá.
    """
    prediction: float
    model_version: str

# =============================================================================
# Schema de Entrada (Input)
# =============================================================================
class PredictionInput(BaseModel):
    """
    Define el JSON de entrada que el endpoint /predict espera.
    
    Pydantic validará automáticamente los tipos y rangos de estos datos.
    """
    
    # Columnas categóricas (como int)
    season: int = Field(..., ge=1, le=4, description="Temporada (1:invierno, 2:primavera, 3:verano, 4:otoño)")
    yr: int = Field(..., ge=0, le=1, description="Año (0: 2011, 1: 2012)")
    mnth: int = Field(..., ge=1, le=12, description="Mes (1 a 12)")
    hr: int = Field(..., ge=0, le=23, description="Hora (0 a 23)")
    holiday: int = Field(..., ge=0, le=1, description="Es día festivo (0:No, 1:Sí)")
    weekday: int = Field(..., ge=0, le=6, description="Día de la semana (0 a 6)")
    workingday: int = Field(..., ge=0, le=1, description="Es día laboral (0:No, 1:Sí)")
    weathersit: int = Field(..., ge=1, le=4, description="Situación climática (1:Despejado, 2:Niebla, 3:Nieve ligera, 4:Lluvia intensa)")
    
    # Columnas numéricas (float)
    temp: float = Field(..., ge=0.0, le=1.0, description="Temperatura normalizada")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Sensación térmica normalizada")
    hum: float = Field(..., ge=0.0, le=1.0, description="Humedad normalizada")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Velocidad del viento normalizada")

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