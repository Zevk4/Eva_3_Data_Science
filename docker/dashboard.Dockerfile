# version de python
FROM python:3.11

# Se establece el directorio de trabajo
WORKDIR /app

# Copia del archivo de requerimientos
COPY ./dashboards/requirements.txt .

# Instalación de librerías sin guardar caché para optimizar el espacio
RUN pip install --no-cache-dir -r requirements.txt

# Copia del código fuente de la aplicación
COPY ./dashboards/ .

# Exposición del puerto estándar de Streamlit
EXPOSE 8501

# Comando de ejecución 
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]