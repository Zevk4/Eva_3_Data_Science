import logging
import os
import json 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Configuración del Monitoreo con Logging (#10)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API REST de Segmentación de Usuarios - Streaming",
    description="Servicio backend para exponer predicciones de clusters KMeans y analítica de datos.",
    version="1.0.0"
)

# Rutas relativas para los volúmenes compartidos de Docker
DATA_PATH = "./data/processed/clientes_segmentados.csv"
CENTROIDES_PATH = "./data/processed/centroides.csv"
MODEL_PATH = "./models/modelo_kmeans.pkl"
METRICAS_PATH = "./models/metricas.json"
SCALER_PATH = "./models/scaler.pkl"

clientes_df = None
model_kmeans = None
scaler = None

@app.on_event("startup")
def load_assets():
    """Carga de Artefactos al iniciar el servicio Docker (#10)"""
    global clientes_df, model_kmeans, scaler
    logger.info("Iniciando la carga de artefactos del sistema analítico...")
    
    if os.path.exists(DATA_PATH):
        clientes_df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset cargado con éxito. Registros: {len(clientes_df)}")
    else:
        logger.warning(f"No se encontró el archivo de datos en {DATA_PATH}. Esperando pipeline.")

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model_kmeans = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info("Modelo KMeans y Scaler cargados exitosamente en memoria.")
        else:
            logger.warning("Archivos .pkl del modelo no encontrados temporalmente.")
    except Exception as e:
        logger.error(f"Error al cargar los archivos binarios del modelo: {str(e)}")

# Endpoint Base (/) (#10)
@app.get("/")
def read_root():
    logger.info("Petición recibida en Endpoint Base (/)")
    return {
        "status": "online",
        "componente": "FastAPI Backend Service",
        "nota": "Eva 3 - Programación para la Ciencia de Datos"
    }

# Endpoint de Datos para Visualización 
@app.get("/dashboard-data")
def get_dashboard_data():
    global clientes_df
    logger.info("Petición recibida en /dashboard-data")
    
    # 1. Intentar cargar el dataset de clientes si no está en memoria
    if clientes_df is None:
        if os.path.exists(DATA_PATH):
            clientes_df = pd.read_csv(DATA_PATH)
            logger.info("Dataset de clientes cargado en caliente de forma exitosa.")
        else:
            raise HTTPException(status_code=404, detail="Datos analíticos no disponibles en el volumen.")
            
    # 2. Leer los centroides en caliente
    centroides_list = []
    if os.path.exists(CENTROIDES_PATH):
        centroides_df = pd.read_csv(CENTROIDES_PATH)
        centroides_list = centroides_df.to_dict(orient="records")
        logger.info("Centroides cargados de forma exitosa.")
    else:
        logger.warning("Archivo de centroides no encontrado todavía.")
        
    # 3. Leer las métricas reales directo desde el archivo JSON
    metricas_dict = {}
    if os.path.exists(METRICAS_PATH):
        try:
            with open(METRICAS_PATH, "r") as f:
                metricas_dict = json.load(f)
            logger.info("Métricas reales cargadas desde JSON con éxito.")
        except Exception as e:
            logger.error(f"Error al leer el archivo JSON de métricas: {str(e)}")
            raise HTTPException(status_code=500, detail="El archivo de métricas JSON está corrupto o mal formateado.")
    else:
        logger.error(f"Archivo crítico no encontrado en: {METRICAS_PATH}")
        raise HTTPException(status_code=404, detail="Las métricas en formato JSON aún no han sido generadas por el ETL.")

    # 4. Retorno final unificado
    return {
        "total_usuarios": int(len(clientes_df)),
        "usuarios": clientes_df.to_dict(orient="records"),
        "centroides": centroides_list,
        "metricas": metricas_dict 
    }
# Esquema de validación estricta usando Pydantic (#11)
class UsuarioInput(BaseModel):
    horas_consumo_mensual: float = Field(..., example=55.5)
    gasto_mensual: float = Field(..., example=250.0)
    cantidad_contenidos_vistos: int = Field(..., example=12)
    sesiones_semana: int = Field(..., example=5)
    porcentaje_finalizacion: float = Field(..., example=75.0)
    tiempo_promedio_sesion_min: float = Field(..., example=120.0)
    cantidad_generos_consumidos: int = Field(..., example=4)
    porcentaje_uso_promociones: float = Field(..., example=0.15)
    antiguedad_cliente_meses: int = Field(..., example=24)
    edad: int = Field(..., example=34)
    dispositivos_registrados: int = Field(..., example=3)
    porcentaje_uso_app_movil: float = Field(..., example=0.6)
    cantidad_perfiles_creados: int = Field(..., example=2)
    interacciones_mensuales_soporte: int = Field(..., example=1)
    distancia_promedio_red_km: float = Field(..., example=12.4)

# Endpoint de Predicción en Caliente (/predict) (#10)
@app.post("/predict")
def predict_cluster(usuario: UsuarioInput):
    logger.info("Procesando petición de predicción en caliente...")
    if model_kmeans is None or scaler is None:
        raise HTTPException(status_code=503, detail="El modelo predictivo no está disponible.")
    
    try:
        input_data = pd.DataFrame([usuario.dict()])
        scaled_data = scaler.transform(input_data)
        cluster_asignado = int(model_kmeans.predict(scaled_data)[0])
        
        logger.info(f"Predicción exitosa. Cluster: {cluster_asignado}")
        return {
            "cluster_asignado": cluster_asignado
        }
    except Exception as e:
        logger.error(f"Error en la inferencia de KMeans: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))