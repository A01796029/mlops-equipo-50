import pandas as pd
from src.data.cleaning import DataCleaningPipeline

def test_date_to_season_winter():
    # 21 de diciembre → Invierno (1)
    date = pd.Timestamp("2011-12-21")
    result = DataCleaningPipeline._date_to_season(date)
    assert result == 1

def test_date_to_season_spring():
    # 1 de abril → Primavera (2)
    date = pd.Timestamp("2011-04-01")
    result = DataCleaningPipeline._date_to_season(date)
    assert result == 2

def test_date_to_season_summer():
    # 10 de julio → Verano (3)
    date = pd.Timestamp("2011-07-10")
    result = DataCleaningPipeline._date_to_season(date)
    assert result == 3

def test_date_to_season_autumn():
    # 10 de octubre → Otoño (4)
    date = pd.Timestamp("2011-10-10")
    result = DataCleaningPipeline._date_to_season(date)
    assert result == 4

def test_date_to_season_nan():
    # Valor nulo → np.nan
    result = DataCleaningPipeline._date_to_season(pd.NaT)
    assert pd.isna(result)