import os
import json
import logging
import pandas as pd
import pickle

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA

# Configuración de Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline_ETL_Train")

# Definimos Rutas Absolutas
os.makedirs("/app/models", exist_ok=True)
os.makedirs("/app/data/processed", exist_ok=True)

try:
    logger.info("Cargando fuentes de datos...")
    # Archivo CSV
    clientes = pd.read_csv("data/raw/usuarios_streaming.csv")

    # Fuente desde la BD
    engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_db")
    perfil = pd.read_sql("SELECT * FROM perfil_usuarios", engine)

    # Integración
    data = clientes.merge(perfil, on="id_cliente")
    
    # Guarda el archivo con la data integrada
    data.to_csv("data/processed/data_clientes.csv", index=False)

    logger.info("Filtrando variables numéricas para el modelo...")
    # Filtro para evitar textos/categóricos (Esta de más pero nunca viene mal)
    X = data.drop(columns=["id_cliente"])
    X = X.select_dtypes(include=["int64", "float64"])

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Evaluando número óptimo de clusters")
    inertias = []
    for k in range(2, 11):
        modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
        modelo.fit(X_scaled)
        inertias.append(modelo.inertia_)

    kl = KneeLocator(
        range(2, 11),
        inertias,
        curve='convex',
        direction='decreasing'
    )

    # Modelo    
    k_optimo = kl.elbow

    logger.info(f"Entrenando modelo final KMeans con K={k_optimo}")

    kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    data["cluster"] = clusters

    # Reducción dimensional para gráficos
    pca = PCA(n_components=2)

    componentes = pca.fit_transform(X_scaled)

    data["pc1"] = componentes[:, 0]
    data["pc2"] = componentes[:, 1]

    # Guarda data con los cluster y dos componentes principales
    data.to_csv("data/processed/clientes_segmentados.csv", index=False)

    # Guarda las métricas
    metricas = {
        "k_optimo": int(k_optimo),
        "silhouette_score": float(silhouette_score(X_scaled, data["cluster"])),
        "n_clientes": int(len(data)),
        "n_clusters": int(k_optimo),
        "varianza_pca": float(pca.explained_variance_ratio_.sum())
    }

    with open("/app/models/metricas.json", "w") as f:
        json.dump(metricas, f, indent=4)

    # Guarda los centroides desescalados
    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

    centroides_df = pd.DataFrame(
        centroides_original, 
        columns=X.columns
    )

    centroides_df.to_csv("data/processed/centroides.csv", index=False)


    # Guarda Modelo y data escalada
    pickle.dump(kmeans, open("/app/models/modelo_kmeans.pkl", "wb"))
    pickle.dump(scaler, open("/app/models/scaler.pkl", "wb"))
    pickle.dump(pca, open("/app/models/pca.pkl", "wb"))

    logger.info("Modelo Guardado!!!")

except Exception as e:
    logger.error(f"Error crítico durante la ejecución del pipeline: {str(e)}")
    raise e