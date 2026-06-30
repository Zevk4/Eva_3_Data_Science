import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard - Segmentación Streaming", layout="wide")

st.title("Dashboard de Segmentación de Usuarios (Streaming)")
st.markdown("---")

# Obtiene los datos para la visualización
try:
    respuesta = requests.get("http://ml-service:8000/dashboard-data")
    payload = respuesta.json()
    data = pd.DataFrame(payload["usuarios"])
    centroides = pd.DataFrame(payload["centroides"])
    metricas = payload["metricas"]
    
except Exception as e:
    st.error(f"Error al conectar con el servicio de datos o procesar el JSON: {e}")
    st.stop()


# DIFERENCIACIÓN POR AUDIENCIA
# Se crean dos entornos de navegación separados según el perfil del usuario final
tab_negocio, tab_tecnica = st.tabs(["Vista de Negocio y Operaciones", "Vista Técnica de Machine Learning"])



# PESTAÑA 1: VISTA DE NEGOCIO
with tab_negocio:
    st.header("Análisis de Segmentos de Mercado")
    
    # Métricas ejecutivas clave
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.metric(
            label="Total Usuarios en Base de Datos",
            value=f"{payload.get('total_usuarios', len(data)):,}".replace(",", ".")
        )
    with col_n2:
        st.metric(
            label="Segmentos de Mercado Identificados",
            value=metricas.get("k_optimo", data["cluster"].nunique() if "cluster" in data.columns else 0)
        )
        
    st.markdown("---")
    
    # Distribución y Perfiles 
    col_graf, col_perf = st.columns([1, 2])

    with col_graf:
        st.subheader("Distribución Volumétrica")
        if "cluster" in data.columns:
            counts = data["cluster"].value_counts().sort_index()
            # Ajuste de nombre para negocio
            counts.index = [f"Segmento {i}" for i in counts.index]
            st.bar_chart(counts)
        else:
            st.warning("La columna 'cluster' no está disponible en los datos.")

    with col_perf:
        st.subheader("Perfil Promedio por Segmento")
        columnas_excluir = ["id_cliente", "cliente_id", "pc1", "pc2", "cluster"]
        columnas_analisis = [c for c in data.select_dtypes(include=["int64", "float64"]).columns if c not in columnas_excluir]
        
        if "cluster" in data.columns and len(columnas_analisis) > 0:
            perfil_segmentos = data.groupby("cluster")[columnas_analisis].mean().round(2)
            # Limpieza de nombres de columnas para lectura corporativa
            perfil_segmentos.columns = [c.replace("_", " ").title() for c in perfil_segmentos.columns]
            perfil_segmentos.index = [f"Segmento {i}" for i in perfil_segmentos.index]
            
            st.dataframe(perfil_segmentos, use_container_width=True)
        else:
            st.info("No hay suficientes variables numéricas para calcular los perfiles.")

    st.markdown("---")
    
    # Dataset Crudo
    st.subheader("Directorio de Usuarios Consolidados")
    st.dataframe(data.drop(columns=["pc1", "pc2"], errors="ignore"), use_container_width=True)



# PESTAÑA 2: VISTA TÉCNICA
with tab_tecnica:
    st.header("Métricas de Rendimiento y Análisis")
    
    # Métricas científicas
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        sil_score = metricas.get("silhouette_score", 0.0)
        st.metric(
            label="Coeficiente Silhouette (Cohesión de Clusters)",
            value=f"{sil_score:.3f}" if sil_score else "N/A"
        )
    with col_t2:
        var_pca = metricas.get("varianza_pca", 0.0)
        st.metric(
            label="Varianza Retenida (Reducción PCA)",
            value=f"{var_pca * 100:.1f}%" if var_pca else "N/A"
        )

    st.markdown("---")

    # Gráfico PCA
    if "pc1" in data.columns and "pc2" in data.columns:
        st.subheader("Proyección Ortogonal PCA (2D)")
        st.markdown("Visualización de la varianza explicada máxima para evaluar la separación.")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        for cluster in sorted(data["cluster"].unique()):
            subset = data[data["cluster"] == cluster]
            ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster}", alpha=0.7, edgecolors='w', s=50)
        
        ax.set_xlabel("Componente Principal 1 (PC1)", fontweight="bold")
        ax.set_ylabel("Componente Principal 2 (PC2)", fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        st.markdown("---")

    # Comparador Dinámico 
    st.subheader("Matriz de Dispersión Dinámica (Bivariada)")
    
    opciones_variables = [c for c in data.columns if c not in ["id_cliente", "cliente_id", "cluster", "pc1", "pc2"]]

    if len(opciones_variables) >= 2:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            columna_x = st.selectbox("Variable Eje X", opciones_variables, index=0)
        with col_sel2:
            columna_y = st.selectbox("Variable Eje Y", opciones_variables, index=min(1, len(opciones_variables)-1))
            
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        scatter = ax2.scatter(
            data[columna_x], data[columna_y],
            c=data["cluster"], cmap="tab10", alpha=0.6, s=45
        )

        if not centroides.empty and columna_x in centroides.columns and columna_y in centroides.columns:
            ax2.scatter(
                centroides[columna_x], centroides[columna_y],
                marker="X", s=250, color="red", edgecolor="black", linewidth=2,
                label="Centroide Calculado"
            )

        ax2.set_xlabel(columna_x.replace('_', ' ').title(), fontsize=11, fontweight="bold")
        ax2.set_ylabel(columna_y.replace('_', ' ').title(), fontsize=11, fontweight="bold")
        ax2.set_title(f"Dispersión Estructural: {columna_x.replace('_', ' ').title()} vs {columna_y.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        
        st.pyplot(fig2)
    else:
        st.info("Cargando matriz dimensional...")