# Equipo 50

## Integrantes 
* Edmundo Carmona Galindo
* Aarón Cortés García
* César Alejandro Cruz Salas
* Arturo González Corona
* Luis Santiago Vázquez Mancilla

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fase 1 del Proyecto Bike Sharing: https://docs.google.com/presentation/d/1LZJk0EhUpKuh7kQsRSAzbQCTJ0b0cwNmPyQ3AUO4_EY/edit?usp=sharing

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_equipo_50 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_equipo_50 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── main.py                    <- Este es el script de ejecución que junta el procesador de datos y el orquestador de experimentos
    │   ├── data_processor.py          <- Este archivo contendrá la clase DataProcessor que maneja la carga, el encoding y la división de los datos, siguiendo los pasos de tu notebook original.         
    │   └── ml_experiment.py           <- Esta clase contendrá la lógica principal de MLOps: registro de experimentos (MLflow), visualización (artefactos) y entrenamiento (incluyendo Grid Search).
    │
    └── plots.py                <- Code to create visualizations
```

--------



Fase 1 del Proyecto Bike Sharing:

## Modelo Desplegado (API)

El servicio de inferencia de FastAPI (ubicado en `src/api/main.py`) está diseñado para cargar un modelo directamente desde el **Registro de Modelos de MLflow**.

Para garantizar la reproducibilidad, la API está **"fijada" (pinned)** a una versión de modelo específica.

### URI del Artefacto

La URI del modelo que la API está configurada para cargar es: 
- models:/Bike_Sharing_MLOps_Project/1

**Desglose de la URI:**
* **`models:/`**: Es el prefijo de MLflow para un modelo registrado.
* **`Bike_Sharing_MLOps_Project`**: Es el nombre del modelo registrado (el que creaste en la pestaña "Models").
* **`1`**: Es el número de **Versión** específico que se está sirviendo.

### Flujo de trabajo para Actualizaciones

1.  El pipeline de entrenamiento se ejecuta (`python -m src train`) y se registra una nueva versión (ej. `Version 2`) en MLflow.
2.  Para desplegar este nuevo modelo, el archivo `src/api/main.py` debe ser actualizado, cambiando la variable `MODEL_URI` para que apunte a la nueva versión (ej. `.../2`).
3.  La API debe ser reiniciada (o la imagen de Docker reconstruida y redesplegada) para cargar el nuevo modelo.