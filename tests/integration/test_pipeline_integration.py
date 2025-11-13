# tests/integration/test_pipeline_integration.py
import pytest
from src.data.make_dataset import load_dataset
from src.data.cleaning import DataCleaningPipeline
from src.modeling.train import train_model
from src.modeling.predict import generate_predictions

@pytest.mark.integration
def test_full_pipeline(tmp_path):
    """Prueba el flujo completo del pipeline: carga → limpieza → entrenamiento → predicción"""

    # 1. Cargar datos
    df = load_dataset()
    assert not df.empty, "El dataset cargado está vacío"

    # 2. Preprocesar / limpiar datos
    cleaner = DataCleaningPipeline()
    df_clean = cleaner.clean(df)
    assert df_clean.isnull().sum().sum() == 0, "El dataset limpio contiene valores nulos"

    # 3. Entrenar modelo (usa un directorio temporal)
    model_path = tmp_path / "model.pkl"
    model = train_model(df_clean, model_path=model_path)
    assert model_path.exists(), "No se guardó el modelo entrenado"

    # 4. Generar predicciones
    predictions_path = tmp_path / "predictions.csv"
    preds = generate_predictions(df_clean, model_path, predictions_path)
    assert predictions_path.exists(), "No se generó el archivo de predicciones"
