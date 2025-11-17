# src/data_processor.py
##Cambio ligero
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    Clase para cargar, preprocesar y dividir el dataset.
    """
    
    def __init__(self, file_path, target_col='cnt', test_size=0.2, val_size=0.10, random_state=42):
        self.file_path = file_path
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_pipeline = True  # Indica si se usará el pipeline de preprocesamiento

    def _one_hot_encoder(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Aplica One-Hot Encoding a una columna."""
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
        data = data.drop(columns=[column], axis=1)
        return data

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la preparación inicial y One-Hot Encoding."""
        
        # 1. Replicar la corrección de dtypes para columnas categóricas (como en el notebook)
        cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # 2. Definir X e y y eliminar columnas no utilizadas/filtradas
        X = df.drop(columns=[self.target_col, "casual", "registered", "dteday"], errors='ignore')
        
        # 3. Aplicar One-Hot Encoding
        if self.use_pipeline:
            # No OHE aquí: el ColumnTransformer del pipeline lo hará
            return X
        else:
            # OHE manual (comportamiento previo definido por Edmundo en el notebook)
            for col in X.select_dtypes(include=["category"]).columns:
                X = self._one_hot_encoder(X, col)
            return X

    def load_and_split(self):
        """Carga los datos y los divide en Train, Validation y Test sets."""
        
        try:
            df = pd.read_csv(self.file_path)
            
            # Clean target variable: convert to numeric and drop invalid rows
            df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
            initial_rows = len(df)
            df = df.dropna(subset=[self.target_col])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                print(f"  → Eliminadas {dropped_rows} filas con valores inválidos en '{self.target_col}'")
            
            X = self._prepare_data(df)
            y = df[self.target_col]
        except Exception as e:
            print(f"Error al cargar o preparar los datos: {e}")
            raise

        # 1. División Train/Validation + Test (Hold-out test)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 2. División Train + Validation -> Train y Validation (para LazyPredict)
        # Nota: Usaremos el 10% del 80% (trainval) para validación.
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=self.val_size, random_state=117
        )

        return X_trainval, X_test, y_trainval, y_test, X_train, X_val, y_train, y_val

# ----------------------------------------------------------------------
# Helper functions (se mantienen como funciones simples)
# ----------------------------------------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calculate_rmse(y_true, y_pred):
    """Calcula el Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_metrics(y_true, y_pred):
    """Calcula y devuelve un diccionario de métricas."""
    from sklearn.metrics import r2_score
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred)
    }