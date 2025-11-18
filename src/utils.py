from pathlib import Path
from typing import Any, Optional, Union
from loguru import logger
from tqdm import tqdm
import pandas as pd
import os


def simulate_progress(task_name: str, steps: int = 10) -> None:
    """
    Simula un progreso para tareas dummy (útil para ejemplos / placeholders).

    Args:
        task_name: Nombre de la tarea a mostrar en logs.
        steps: Número de pasos para el progress bar (default: 10).
        
    Raises:
        ValueError: Si steps es menor o igual a 0.
    """
    if steps <= 0:
        raise ValueError(f"El número de pasos debe ser mayor a 0, recibido: {steps}")
    
    logger.info(f"Starting: {task_name}")
    for i in tqdm(range(steps), total=steps, desc=task_name):
        if i == steps // 2:
            logger.warning(f"Midway event in {task_name}")
    logger.success(f"{task_name} complete.")


def save_dataframe_to_csv(
    df: pd.DataFrame, 
    path: Union[Path, str], 
    index: bool = False,
    validate_not_empty: bool = True
) -> Path:
    """
    Guarda un DataFrame en CSV asegurando que la carpeta destino exista.

    Args:
        df: DataFrame a guardar.
        path: Ruta destino (Path o str).
        index: Si se guarda o no el índice (default: False).
        validate_not_empty: Si True, valida que el DataFrame no esté vacío (default: True).
        
    Returns:
        Path: Ruta donde se guardó el archivo.
        
    Raises:
        ValueError: Si el DataFrame está vacío y validate_not_empty es True.
        TypeError: Si df no es un DataFrame.
        OSError: Si hay problemas al crear directorios o escribir archivo.
    """
    # Validaciones
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Se esperaba pd.DataFrame, se recibió {type(df)}")
    
    if validate_not_empty and df.empty:
        raise ValueError("El DataFrame está vacío y no se puede guardar")
    
    # Convertir a Path si es necesario
    path = Path(path) if isinstance(path, str) else path
    
    try:
        # Crear directorio padre si no existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar CSV
        df.to_csv(path, index=index)
        
        logger.info(f"✅ DataFrame guardado exitosamente en {path}")
        logger.debug(f"   Shape: {df.shape}, Columnas: {list(df.columns)}")
        
        return path
        
    except PermissionError as e:
        logger.error(f"❌ Permisos insuficientes para escribir en {path}")
        raise OSError(f"No se tienen permisos para escribir en {path}: {e}")
    except Exception as e:
        logger.error(f"❌ Error al guardar DataFrame en {path}: {e}")
        raise


def ensure_path(path: Union[Path, str], is_file: bool = True) -> Path:
    """
    Crea el directorio padre de la ruta si no existe.
    
    Args:
        path: Ruta del archivo o directorio (Path o str).
        is_file: Si True, crea el directorio padre. Si False, crea el directorio mismo.
        
    Returns:
        Path: Ruta procesada como objeto Path.
        
    Raises:
        ValueError: Si la ruta está vacía.
    """
    if not path:
        raise ValueError("La ruta no puede estar vacía")
    
    # Convertir a Path si es necesario
    path = Path(path) if isinstance(path, str) else path
    
    try:
        if is_file:
            # Crear directorio padre para archivos
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Directorio asegurado: {path.parent}")
        else:
            # Crear el directorio mismo
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Directorio creado: {path}")
            
        return path
        
    except Exception as e:
        logger.error(f"❌ Error al crear directorio para {path}: {e}")
        raise


def load_dataframe_from_csv(
    path: Union[Path, str],
    validate_not_empty: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo CSV con validaciones.
    
    Args:
        path: Ruta del archivo CSV.
        validate_not_empty: Si True, valida que el DataFrame cargado no esté vacío.
        **kwargs: Argumentos adicionales para pd.read_csv.
        
    Returns:
        pd.DataFrame: DataFrame cargado desde el CSV.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el DataFrame está vacío y validate_not_empty es True.
        pd.errors.EmptyDataError: Si el CSV está vacío.
    """
    path = Path(path) if isinstance(path, str) else path
    
    if not path.exists():
        logger.error(f"❌ Archivo no encontrado: {path}")
        raise FileNotFoundError(f"El archivo {path} no existe")
    
    try:
        logger.info(f"📂 Cargando DataFrame desde {path}")
        df = pd.read_csv(path, **kwargs)
        
        if validate_not_empty and df.empty:
            raise ValueError(f"El DataFrame cargado desde {path} está vacío")
        
        logger.success(f"✅ DataFrame cargado exitosamente")
        logger.debug(f"   Shape: {df.shape}, Columnas: {list(df.columns)}")
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"❌ El archivo CSV está vacío: {path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error al cargar CSV desde {path}: {e}")
        raise
