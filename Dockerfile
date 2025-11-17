FROM python:3.10-slim

WORKDIR /src/modeling

COPY ./requirements.txt /src/modeling/requirements.txt

COPY ./data/interim/bike_sharing_cleaned.csv /data/interim/bike_sharing_cleaned.csv

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./src/modeling /src/modeling
COPY ./src/config.py /src/config.py

CMD ["python", "src/modeling/main.py"]