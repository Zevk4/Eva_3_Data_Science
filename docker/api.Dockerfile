FROM python:3.11

WORKDIR /app

# Copia los requerimientos de la API
COPY ./api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia el codigo de app.py
COPY ./api/ .

EXPOSE 8000

CMD uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000