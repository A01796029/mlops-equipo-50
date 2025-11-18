FROM python:3.10-slim

WORKDIR /app

# El contexto de build debe ser la raíz del proyecto
COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src /app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]