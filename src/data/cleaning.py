import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# =============================================================================
# Clase de Pipeline de Limpieza de Datos
# =============================================================================

class DataCleaningPipeline:
    """
    Clase para encapsular y ejecutar el pipeline completo de limpieza
    de datos de 'bike sharing' utilizando Programación Orientada a Objetos (POO).
    """
    # Constantes definidas en base a la lógica del notebook original
    _VALID_COLUMNS: List[str] = [
        'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
        'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 
        'casual', 'registered', 'cnt'
    ]
    
    # Reglas de validación para manejo de inválidos
    _VALIDATION_RULES: Dict[str, List | Tuple] = {
        'dteday': (pd.to_datetime('2011-01-01'), pd.to_datetime('2012-12-31')),
        'season': [1, 2, 3, 4],
        'yr': [0, 1],
        'mnth': list(range(1, 13)),
        'hr': list(range(0, 24)),
        'holiday': [0, 1],
        'weekday': list(range(0, 7)),
        'workingday': [0, 1],
        'weathersit': [1, 2, 3, 4],
        'hum': (0.0, 1.0),
        'windspeed': (0.0, 1.0),
        'cnt': (0, float('inf')) 
    }

    def __init__(self, raw_data_path: str):
        """
        Inicializa el pipeline de limpieza con la ruta del archivo crudo.
        
        Args:
            raw_data_path (str): Ruta al archivo CSV crudo.
        """
        self.raw_data_path = raw_data_path
        self.df: Optional[pd.DataFrame] = None
        print(f"Pipeline inicializado para la ruta: {self.raw_data_path}")

    @staticmethod
    def _date_to_season(date_obj: pd.Timestamp) -> int:
        """
        [Método Estático de Ayuda] Convierte una fecha a la estación del año.
        1:invierno, 2:primavera, 3:verano, 4:otoño.
        """
        if pd.isna(date_obj):
            return np.nan
            
        month = date_obj.month
        day = date_obj.day

        # Lógica de solsticios/equinoccios
        if (month == 12 and day >= 21) or (month in [1, 2]) or (month == 3 and day < 21):
            return 1 # Invierno
        elif (month == 3 and day >= 21) or (month in [4, 5]) or (month == 6 and day < 21):
            return 2 # Primavera
        elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day < 23):
            return 3 # Verano
        else:
            return 4 # Otoño
    
    def cargar_datos(self) -> bool:
        """
        Carga el conjunto de datos desde la ruta y lo asigna a self.df.
        
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario.
        """
        print("\n--- PASO 1: CARGA DEL DATASET CRUDO ---")
        try:
            self.df = pd.read_csv(self.raw_data_path)
            print(f"Datos cargados exitosamente. Filas: {len(self.df)}")
            return True
        except FileNotFoundError:
            print(f"Error: El archivo no fue encontrado en la ruta: {self.raw_data_path}")
            self.df = None
            return False
        except Exception as e:
            print(f"Ocurrió un error inesperado al cargar el archivo: {e}")
            self.df = None
            return False

    def diagnostico_inicial(self, df_name: str = "DataFrame") -> None:
        """Imprime un resumen completo y diagnóstico del DataFrame actual."""
        if self.df is None:
            print("No hay datos para diagnosticar.")
            return

        print(f"\n=============== DIAGNÓSTICO PARA: '{df_name}' ===============")
        print(f"\n**Dimensiones:** {self.df.shape[0]} filas y {self.df.shape[1]} columnas.")
        print("\n**Tipos de Datos y Valores No Nulos:**")
        self.df.info()
        print("\n**Estadísticas Descriptivas (Numéricas):**")
        print(self.df.describe().T)

    def eliminar_columnas_innecesarias(self) -> None:
        """Elimina columnas que no están en la lista de columnas válidas."""
        if self.df is None: return

        current_columns = self.df.columns.tolist()
        cols_to_drop = [col for col in current_columns if col not in self._VALID_COLUMNS]

        if cols_to_drop:
            print(f"\n  -> Eliminando columnas: {cols_to_drop}")
            self.df = self.df.drop(columns=cols_to_drop).copy()
            print("Columnas innecesarias eliminadas.")
        else:
            print("  -> No se encontraron columnas innecesarias. El esquema es correcto.")

    def corregir_tipos_datos_iniciales(self) -> None:
        """Corrige tipos de datos: numéricos con coerce y la columna de fecha."""
        if self.df is None: return

        print("\n  -> Iniciando corrección de tipos de datos semánticos...")
        
        # 1. Procesa columnas numéricas/ordinales (incluyendo categóricas como int para limpieza)
        numeric_cols = [col for col in self._VALID_COLUMNS if col in self.df.columns and col != 'dteday']
        for col in numeric_cols:
             self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 2. Procesa columna de fecha
        if 'dteday' in self.df.columns:
            self.df['dteday'] = pd.to_datetime(self.df['dteday'], errors='coerce', format='mixed')
            
        print("  -> Tipos de datos iniciales corregidos.")

    def manejar_valores_invalidos(self) -> None:
        """Valida los datos contra las reglas definidas y convierte inválidos a NaN."""
        if self.df is None: return

        print("\n  -> Iniciando validación de valores lógicos...")
        
        for column, rule in self._VALIDATION_RULES.items():
            if column in self.df.columns:
                initial_nulls = self.df[column].isnull().sum()

                if isinstance(rule, list):
                    # Validación para variables categóricas
                    invalid_mask = ~self.df[column].isin(rule)
                elif isinstance(rule, tuple):
                    # Validación para variables numéricas (rango)
                    min_val, max_val = rule
                    invalid_mask = (self.df[column] < min_val) | (self.df[column] > max_val)
                
                # Reemplaza los valores que no cumplen la regla con NaN
                self.df.loc[invalid_mask, column] = np.nan
                
                final_nulls = self.df[column].isnull().sum()
                newly_invalid = final_nulls - initial_nulls
                if newly_invalid > 0:
                    print(f"    - Columna '{column}': {int(newly_invalid)} valores inválidos convertidos a NaN.")

        print("  -> Validación de valores lógicos completada.")

    def imputar_valores_faltantes(self) -> None:
        """Elimina filas críticas y luego imputa nulos de forma contextual y estadística."""
        if self.df is None: return

        print("\n  -> Iniciando manejo e imputación de valores faltantes...")
        initial_rows = len(self.df)
        
        # Estrategia 1: Eliminación de filas críticas
        critical_cols = ['dteday', 'hr', 'holiday', 'workingday', 'casual', 'registered', 'cnt']
        self.df.dropna(subset=critical_cols, inplace=True)
        rows_deleted = initial_rows - len(self.df)
        if rows_deleted > 0:
            print(f"    - Se eliminaron {rows_deleted} filas con datos críticos faltantes.")

        # Estrategia 2: Imputación Contextual (Derivación por fecha)
        # Se asume que las columnas son numéricas (flotantes) en este punto
        mask_yr = self.df['yr'].isna()
        if 'yr' in self.df.columns:
            self.df.loc[mask_yr, 'yr'] = self.df.loc[mask_yr, 'dteday'].dt.year - 2011
        
        mask_mnth = self.df['mnth'].isna()
        if 'mnth' in self.df.columns:
            self.df.loc[mask_mnth, 'mnth'] = self.df.loc[mask_mnth, 'dteday'].dt.month
        
        mask_weekday = self.df['weekday'].isna()
        if 'weekday' in self.df.columns:
            self.df.loc[mask_weekday, 'weekday'] = (self.df.loc[mask_weekday, 'dteday'].dt.weekday + 1) % 7
        
        mask_season = self.df['season'].isna()
        if 'season' in self.df.columns:
            self.df.loc[mask_season, 'season'] = self.df.loc[mask_season, 'dteday'].apply(self._date_to_season)

        # Estrategia 3: Imputación estadística (mediana) para variables de clima
        weather_cols = ['weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        for column in weather_cols:
            if column in self.df.columns and self.df[column].isnull().any():
                median_val = self.df[column].median()
                self.df[column] = self.df[column].fillna(median_val)
        
        print("  -> Proceso de imputación de nulos finalizado.")

    def validar_inconsistencias(self) -> None:
        """Verifica y corrige inconsistencias lógicas como fecha vs. variables y conteo."""
        if self.df is None: return

        print("\n  -> Verificando y corrigiendo inconsistencias lógicas...")

        # Regla 1: Consistencia de variables de tiempo vs. 'dteday' (Corrección)
        self.df['yr'] = self.df['dteday'].dt.year - 2011
        self.df['mnth'] = self.df['dteday'].dt.month
        self.df['weekday'] = (self.df['dteday'].dt.weekday + 1) % 7
        self.df['season'] = self.df['dteday'].apply(self._date_to_season)

        print("    - Consistencia de fecha (yr, mnth, season, weekday) corregida.")

        # Regla 2: Consistencia de 'workingday' (Corrección)
        # 1 si es día laboral (no fin de semana AND no feriado), 0 en caso contrario
        correct_workingday = ((self.df['weekday'].isin([0, 6])) | (self.df['holiday'] == 1)).apply(lambda x: 0 if x else 1)
        self.df['workingday'] = correct_workingday
        print("    - Consistencia de 'workingday' corregida.")

        # Regla 3: Consistencia de conteo (Eliminación)
        # Identifica las filas donde cnt != casual + registered
        inconsistent_sum_mask = self.df['cnt'] != (self.df['casual'] + self.df['registered'])
        rows_to_drop = inconsistent_sum_mask.sum()
        
        if rows_to_drop > 0:
            self.df = self.df[~inconsistent_sum_mask].copy()
            print(f"    - Se eliminaron {rows_to_drop} filas por inconsistencia en la suma de conteos.")
        else:
            print("    - No se encontraron inconsistencias en la suma de conteos.")

        print("  -> Verificación de inconsistencias completada.")


    def eliminar_duplicados(self) -> None:
        """Encuentra y elimina filas completamente duplicadas."""
        if self.df is None: return

        print("\n  -> Verificando filas duplicadas...")
        initial_rows = len(self.df)
        
        self.df = self.df.drop_duplicates().copy()
        
        rows_dropped = initial_rows - len(self.df)
        
        if rows_dropped > 0:
            print(f"    - Se eliminaron {rows_dropped} filas duplicadas.")
        else:
            print("    - No se encontraron filas duplicadas.")

    def finalizar_tipos_datos(self) -> None:
        """Convierte las columnas a sus tipos semánticos finales (category, int)."""
        if self.df is None: return
        
        print("\n  -> Puliendo los tipos de datos finales...")

        categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        count_cols = ['casual', 'registered', 'cnt']

        # Conversión a categóricas
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # Conversión a entero (deben estar libres de NaN para esto)
        for col in count_cols:
            if col in self.df.columns:
                # Se usa 'Int64' para manejar nulos como NaN y aún tener tipo entero
                # Si se eliminaron todos los nulos antes, 'int' normal es suficiente.
                # Asumiendo que handle_missing_values limpió los nulos en estas columnas.
                self.df[col] = self.df[col].astype(int)

        self.df = self.df.reset_index(drop=True)
        print("  -> Tipos de datos finalizados.")

    def ejecutar_pipeline(self) -> Optional[pd.DataFrame]:
        """
        Método principal que ejecuta la secuencia completa de limpieza.
        
        Returns:
            Optional[pd.DataFrame]: El DataFrame limpio o None si la carga falló.
        """
        print("\n=========================================================")
        print("INICIANDO EJECUCIÓN DEL PIPELINE DE LIMPIEZA DE DATOS")
        print("=========================================================")
        
        if not self.cargar_datos():
            print("El pipeline se detuvo debido a un error de carga.")
            return None

        # --- FASE 1: Diagnóstico (Opcional) ---
        self.diagnostico_inicial(df_name="DATOS CRUDOS")
        
        # --- FASE 2: Transformaciones y Limpieza ---
        print("\n--- PASO 2: APLICANDO TRANSFORMACIONES DE LIMPIEZA ---")
        self.eliminar_columnas_innecesarias()
        self.corregir_tipos_datos_iniciales()
        self.manejar_valores_invalidos()
        self.imputar_valores_faltantes()
        self.validar_inconsistencias()
        self.eliminar_duplicados()
        self.finalizar_tipos_datos()
        
        # --- FASE 3: Diagnóstico Final (Opcional) ---
        print("\n--- PASO 3: DIAGNÓSTICO FINAL ---")
        self.diagnostico_inicial(df_name="DATOS LIMPIOS Y PROCESADOS")
        
        return self.df

# =============================================================================
# Ejemplo de Uso del Módulo (Simulando make_dataset.py)
# =============================================================================
if __name__ == '__main__':
    # Aquí usarías rutas relativas dentro de tu estructura Cookiecutter
    # Ejemplo: RAW_DATA_PATH = '../../data/raw/bike_sharing_raw.csv'
    RAW_DATA_PATH = 'hour.csv' # RUTA EJEMPLO: REEMPLAZA CON TU RUTA
    
    # 1. Crea una instancia de la clase
    pipeline = DataCleaningPipeline(raw_data_path=RAW_DATA_PATH)
    
    # 2. Ejecuta el pipeline completo
    cleaned_df = pipeline.ejecutar_pipeline()
    
    if cleaned_df is not None:
        print("\nPipeline completado exitosamente. DataFrame listo para modelado.")
        
        # Guardado del resultado (simulación, la ruta sería en la carpeta 'interim')
        # cleaned_df.to_csv('../../data/interim/bike_sharing_cleaned.csv', index=False)