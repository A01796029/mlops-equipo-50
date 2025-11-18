FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para lightgbm y otras librerías científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./data/interim/bike_sharing_cleaned.csv /app/data/interim/bike_sharing_cleaned.csv

COPY ./src /app/src

CMD ["python", "-m", "src.modeling.main"]