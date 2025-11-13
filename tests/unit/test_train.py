import pytest
import pandas as pd
from src.modeling.train import train_model


def test_train_model_returns_expected_keys():
    """Ensure the trained model output contains the expected keys."""
    # Datos sintéticos simples
    data = {
        "temp": [0.3, 0.6, 0.9],
        "hum": [0.7, 0.4, 0.2],
        "cnt": [100, 200, 300],
    }
    df = pd.DataFrame(data)

    model = train_model(df)

    # Verifica que se devuelvan las claves esperadas
    expected_keys = {"coef", "intercept", "mse"}
    assert expected_keys.issubset(model.keys())
    assert isinstance(model["coef"], float)
    assert isinstance(model["intercept"], float)
    assert isinstance(model["mse"], float)


def test_train_model_raises_with_empty_dataframe():
    """Ensure that training fails when dataframe is empty."""
    empty_df = pd.DataFrame(columns=["temp", "hum", "cnt"])

    with pytest.raises(ValueError):
        train_model(empty_df)