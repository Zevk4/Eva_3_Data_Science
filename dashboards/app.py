import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard - Segmentación Streaming", layout="wide")

st.title("Dashboard de Segmentación de Usuarios (Streaming)")
st.markdown("---")

# Obtiene los datos para la visualización desde el microservicio enriquecido
try:
    respuesta = requests.get("http://ml-service:8000/dashboard-data")
    payload = respuesta.json()
    data = pd.DataFrame(payload["usuarios"])
    centroides = pd.DataFrame(payload["centroides"])
    metricas = payload["metricas"]
    
except Exception as e:
    st.error(f"Error al conectar con el servicio de datos o procesar el JSON: {e}")
    st.stop()



# SECCIÓN 1: MÉTRICAS COMPLETA DEL MODELO
st.subheader("Resumen del Sistema Analítico")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Usamos la métrica real del archivo JSON de entrenamiento
    st.metric(
        label="Número de Clusters (K)",
        value=metricas.get("k_optimo", data["cluster"].nunique() if "cluster" in data.columns else 0)
    )

with col2:
    # Mostramos el Silhouette Score directo provisto por la API
    sil_score = metricas.get("silhouette_score", 0.0)
    st.metric(
        label="Silhouette Score",
        value=f"{sil_score:.3f}" if sil_score else "N/A"
    )

with col3:
    # Total de registros proveniente de la métrica oficial de la API
    total_usuarios = payload.get("total_usuarios", len(data))
    st.metric(
        label="Total Usuarios Procesados",
        value=f"{total_usuarios:,}".replace(",", ".")
    )

with col4:
    # Agregamos la varianza explicada del PCA 
    var_pca = metricas.get("varianza_pca", 0.0)
    st.metric(
        label="Varianza Explicada (PCA)",
        value=f"{var_pca * 100:.1f}%" if var_pca else "N/A"
    )

st.markdown("---")



# SECCIÓN 2: VISUALIZACIÓN DE DATASET
st.subheader("Dataset de Usuarios Segmentados")
st.dataframe(data, use_container_width=True)

st.markdown("---")



#  SECCIÓN 3: DISTRIBUCIÓN Y PERFILES REALES
col_graf, col_perf = st.columns([1, 2])

with col_graf:
    st.subheader("Distribución por Segmento")
    if "cluster" in data.columns:
        counts = data["cluster"].value_counts().sort_index()
        st.bar_chart(counts)
    else:
        st.warning("La columna 'cluster' no está disponible en los datos.")

with col_perf:
    st.subheader("Perfil Promedio de Streaming por Grupo")
    # Identifica automáticamente las variables de streaming excluyendo IDs o componentes de gráficos
    columnas_excluir = ["id_cliente", "cliente_id", "pc1", "pc2", "cluster"]
    columnas_analisis = [c for c in data.select_dtypes(include=["int64", "float64"]).columns if c not in columnas_excluir]
    
    if "cluster" in data.columns and len(columnas_analisis) > 0:
        perfil_segmentos = data.groupby("cluster")[columnas_analisis].mean().round(2)
        st.dataframe(perfil_segmentos, use_container_width=True)
    else:
        st.info("No hay suficientes variables numéricas para calcular los perfiles.")

st.markdown("---")



#  SECCIÓN 4: COMPONENTES PRINCIPALES (PCA)
if "pc1" in data.columns and "pc2" in data.columns:
    st.subheader("Espacio de Distribución PCA (2D)")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for cluster in sorted(data["cluster"].unique()):
        subset = data[data["cluster"] == cluster]
        ax.scatter(subset["pc1"], subset["pc2"], label=f"Segmento {cluster}", alpha=0.7, edgecolors='w', s=50)
    
    ax.set_xlabel("Componente Principal 1 (PC1)", fontweight="bold")
    ax.set_ylabel("Componente Principal 2 (PC2)", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    st.markdown("---")



# SECCIÓN 5: COMPARADOR INTERACTIVO CON CENTROIDES REALES
st.subheader("Comparador Interactivo de Comportamiento")
st.markdown("Selecciona dos métricas reales del negocio para analizar cómo interactúan los usuarios y ubicar sus respectivos centros estratégicos.")

# Variables del esquema de streaming listas para ser graficadas
opciones_variables = [c for c in data.columns if c not in ["id_cliente", "cliente_id", "cluster", "pc1", "pc2"]]

if len(opciones_variables) >= 2:
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        columna_x = st.selectbox("Eje X (Métrica 1)", opciones_variables, index=0)
    with col_sel2:
        columna_y = st.selectbox("Eje Y (Métrica 2)", opciones_variables, index=min(1, len(opciones_variables)-1))
        
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # 1. Graficar dispersión de los usuarios individuales
    scatter = ax2.scatter(
        data[columna_x],
        data[columna_y],
        c=data["cluster"],
        cmap="tab10",
        alpha=0.6,
        s=45
    )

    # 2. Pintar las marcas de los Centroides 
    if not centroides.empty and columna_x in centroides.columns and columna_y in centroides.columns:
        ax2.scatter(
            centroides[columna_x],
            centroides[columna_y],
            marker="X",
            s=250,
            color="red",
            edgecolor="black",
            linewidth=2,
            label="Centroides del Grupo"
        )

    ax2.set_xlabel(columna_x.replace('_', ' ').title(), fontsize=11, fontweight="bold")
    ax2.set_ylabel(columna_y.replace('_', ' ').title(), fontsize=11, fontweight="bold")
    ax2.set_title(f"Dispersión Estructural: {columna_x.replace('_', ' ').title()} vs {columna_y.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Agregamos la leyenda 
    ax2.legend()

    st.pyplot(fig2)
else:
    st.info("Cargando variables estructuradas...")