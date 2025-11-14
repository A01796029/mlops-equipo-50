# Permite ejecutar subcomandos desde python -m src.
# por ejemplo
# python -m src data
# python -m src features
# python -m src train
import typer

from src import config
# from src import dataset
from src.data import make_dataset as data_mod
from src import features
from src import plots
# from src.modeling import train as train_mod
from src.modeling import main as train_mod
from src.modeling import predict as predict_mod

app = typer.Typer()
app.add_typer(data_mod.app, name="data") # Antes: dataset.app
app.add_typer(features.app, name="features")
app.add_typer(plots.app, name="plots")
app.add_typer(train_mod.app, name="train")
app.add_typer(predict_mod.app, name="predict")

if __name__ == "__main__":
    app()
