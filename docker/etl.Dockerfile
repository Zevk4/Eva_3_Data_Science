FROM python:3.11

WORKDIR /app

# Copia los requerimientos del ETL desde la raíz del proyecto
COPY ./etl/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia el script train.py y lo que esté dentro de la carpeta /etl
COPY ./etl/ .

# Corre el entrenamiento y el contenedor finaliza su ciclo de vida
CMD ["python", "train.py"]