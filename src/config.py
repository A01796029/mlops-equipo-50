"""
Configuración de rutas y variables del proyecto.

Se usa por los módulos CLI y scripts para localizar data, modelos y reportes.
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()


# =============================================================================
# Random Seed Management
# =============================================================================

class RandomSeedManager:
    """
    Clase centralizada para gestionar todas las semillas aleatorias del proyecto.
    
    Asegura reproducibilidad consistente en todo el pipeline de ML, desde la
    división de datos hasta el entrenamiento de modelos.
    
    Attributes:
        MASTER_SEED: Semilla principal del proyecto
        DATA_SPLIT_SEED: Para división train/validation/test
        VALIDATION_SPLIT_SEED: Para división train/validation
        MODEL_TRAINING_SEED: Para inicialización de modelos
        LAZY_PREDICT_SEED: Para LazyRegressor
    """
    
    # Semilla maestra del proyecto
    MASTER_SEED = 42
    
    # Semillas específicas para diferentes componentes
    # (derivadas de la semilla maestra para mantener reproducibilidad)
    DATA_SPLIT_SEED = MASTER_SEED  # Para train_test_split principal
    VALIDATION_SPLIT_SEED = MASTER_SEED + 1  # Para split de validación
    MODEL_TRAINING_SEED = MASTER_SEED + 75  # 117 (valor previo usado)
    LAZY_PREDICT_SEED = MODEL_TRAINING_SEED  # Mismo que modelos
    
    @classmethod
    def get_master_seed(cls) -> int:
        """Retorna la semilla maestra del proyecto."""
        return cls.MASTER_SEED
    
    @classmethod
    def get_data_split_seed(cls) -> int:
        """Retorna la semilla para división de datos (train/test)."""
        return cls.DATA_SPLIT_SEED
    
    @classmethod
    def get_validation_split_seed(cls) -> int:
        """Retorna la semilla para división de validación."""
        return cls.VALIDATION_SPLIT_SEED
    
    @classmethod
    def get_model_seed(cls) -> int:
        """Retorna la semilla para entrenamiento de modelos."""
        return cls.MODEL_TRAINING_SEED
    
    @classmethod
    def get_lazy_predict_seed(cls) -> int:
        """Retorna la semilla para LazyRegressor."""
        return cls.LAZY_PREDICT_SEED
    
    @classmethod
    def set_all_seeds(cls, seed: int = None):
        """
        Configura todas las librerías de random para reproducibilidad total.
        
        Args:
            seed: Semilla a usar. Si es None, usa MASTER_SEED.
        """
        if seed is None:
            seed = cls.MASTER_SEED
        
        # Python random
        import random
        random.seed(seed)
        
        # NumPy
        import numpy as np
        np.random.seed(seed)
        
        # Si hay PyTorch instalado
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        # Si hay TensorFlow instalado
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        logger.info(f"🌱 Todas las semillas aleatorias configuradas a: {seed}")


# Instancia global para fácil acceso
RANDOM_SEEDS = RandomSeedManager()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
