# src/__main__.py
# =============================================================================
# IMPORTS
# =============================================================================
import typer

from src import config

# Importar los scripts desde sus ubicaciones correctas
from src.data import make_dataset as data_mod      # -> Apunta al script de limpieza real
from src.modeling import features as features_mod  # -> Arregla el error de importación
from src import plots                              # -> Apunta al placeholder de plots
from src.modeling import main as train_mod         # -> Apunta al script de train real
from src.modeling import predict as predict_mod    # -> Apunta al placeholder de predict

# =============================================================================
# CREACIÓN DE LA APP CLI PRINCIPAL
# =============================================================================
app = typer.Typer()

# Conectar los comandos
app.add_typer(data_mod.app, name="data")
app.add_typer(features_mod.app, name="features")
app.add_typer(plots.app, name="plots")
app.add_typer(train_mod.app, name="train")
app.add_typer(predict_mod.app, name="predict")

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    app()
