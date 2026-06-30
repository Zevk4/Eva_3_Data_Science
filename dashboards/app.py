import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard - Segmentación Streaming", layout="wide")

st.title("Dashboard de Segmentación de Usuarios (Streaming)")
st.markdown("---")

# Obtiene los datos para la visualización desde el microservicio
try:
    respuesta = requests.get("http://ml-service:8000/dashboard-data")
    payload = respuesta.json()
    data = pd.DataFrame(payload["usuarios"])
    centroides = pd.DataFrame(payload["centroides"])
    metricas = payload["metricas"]
    
except Exception as e:
    st.error(f"Error al conectar con el servicio de datos: {e}")
    st.stop()


# DIFERENCIACIÓN POR AUDIENCIA Y SIMULADOR
tab_negocio, tab_tecnica, tab_simulador = st.tabs([
    "Vista de Negocio y Operaciones", 
    "Vista Técnica de Machine Learning",
    "Simulador de Predicción (En Vivo)"
])



# PESTAÑA 1: VISTA DE NEGOCIO (Con Plotly y Filtros)
with tab_negocio:
    st.header("Análisis de Segmentos de Mercado")
    
    # 1. Métricas Generales
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.metric("Total Usuarios Procesados", f"{payload.get('total_usuarios', len(data)):,}".replace(",", "."))
    with col_n2:
        st.metric("Segmentos de Mercado Identificados", metricas.get("k_optimo", data["cluster"].nunique() if "cluster" in data.columns else 0))
        
    st.markdown("---")
    
    # 2. Gráfico de Anillo Interactivo (Plotly)
    st.subheader("Distribución Porcentual del Mercado")
    if "cluster" in data.columns:
        counts = data["cluster"].value_counts().reset_index()
        counts.columns = ["Segmento", "Cantidad"]
        counts["Segmento"] = counts["Segmento"].astype(str).apply(lambda x: f"Segmento {x}")
        
        fig_donut = px.pie(
            counts, 
            names="Segmento", 
            values="Cantidad", 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.warning("La columna 'cluster' no está disponible.")

    st.markdown("---")

    # 3. Filtro Dinámico y Gráfico Radial (Spider Chart)
    st.subheader("Perfilamiento por Segmento")
    
    columnas_excluir = ["id_cliente", "cliente_id", "pc1", "pc2", "cluster"]
    columnas_analisis = [c for c in data.select_dtypes(include=["int64", "float64"]).columns if c not in columnas_excluir]
    
    if "cluster" in data.columns and len(columnas_analisis) > 0:
        perfil_segmentos = data.groupby("cluster")[columnas_analisis].mean()
        perfil_segmentos.columns = [c.replace("_", " ").title() for c in perfil_segmentos.columns]
        
        # Selector para interactividad de negocio
        opciones_segmentos = [f"Segmento {i}" for i in perfil_segmentos.index]
        segmento_seleccionado = st.selectbox("Seleccione un segmento para analizar su comportamiento:", opciones_segmentos)
        
        # Extraer índice numérico
        idx_segmento = int(segmento_seleccionado.split(" ")[1])
        
        col_radar, col_tabla = st.columns([1, 1])
        
        with col_radar:
            # Normalizamos los datos al 100% para que el gráfico radial no se distorsione por diferencias de escala
            maximos = perfil_segmentos.max()
            valores_normalizados = (perfil_segmentos.loc[idx_segmento] / maximos).tolist()
            valores_normalizados.append(valores_normalizados[0]) 
            
            categorias = perfil_segmentos.columns.tolist()
            categorias.append(categorias[0])
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=valores_normalizados,
                theta=categorias,
                fill='toself',
                name=segmento_seleccionado,
                line_color='blue'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_tabla:
            st.markdown(f"**Promedios Reales del {segmento_seleccionado}**")
            df_mostrar = perfil_segmentos.loc[[idx_segmento]].T
            df_mostrar.columns = ["Valor Promedio"]
            st.dataframe(df_mostrar.round(2), use_container_width=True)
            
    st.markdown("---")
    st.subheader("Directorio de Usuarios")
    st.dataframe(data.drop(columns=["pc1", "pc2"], errors="ignore"), use_container_width=True)


# PESTAÑA 2: VISTA TÉCNICA
with tab_tecnica:
    st.header("Métricas de Rendimiento y Análisis")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        sil_score = metricas.get("silhouette_score", 0.0)
        st.metric("Coeficiente Silhouette (Cohesión de Clusters)", f"{sil_score:.3f}" if sil_score else "N/A")
    with col_t2:
        var_pca = metricas.get("varianza_pca", 0.0)
        st.metric("Varianza Retenida (Reducción PCA)", f"{var_pca * 100:.1f}%" if var_pca else "N/A")

    st.markdown("---")

    if "pc1" in data.columns and "pc2" in data.columns:
        st.subheader("Proyección Rectangular PCA (2D)")
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

    st.subheader("Matriz de Dispersión Dinámica (Bivariada)")
    opciones_variables_tec = [c for c in data.columns if c not in ["id_cliente", "cliente_id", "cluster", "pc1", "pc2"]]

    if len(opciones_variables_tec) >= 2:
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            columna_x = st.selectbox("Variable Eje X", opciones_variables_tec, index=0)
        with col_sel2:
            columna_y = st.selectbox("Variable Eje Y", opciones_variables_tec, index=min(1, len(opciones_variables_tec)-1))
            
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
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)


# PESTAÑA 3: SIMULADOR DE PREDICCIÓN
with tab_simulador:
    st.header("Clasificación de Nuevos Usuarios")
    st.markdown("Ingresa las métricas de comportamiento de un cliente para predecir a qué segmento estratégico pertenece.")

    with st.form("form_prediccion"):
        st.subheader("Parámetros de Comportamiento")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            horas = st.number_input("Horas Consumo Mensual", min_value=0.0, value=50.0, step=1.0)
            gasto = st.number_input("Gasto Mensual ($)", min_value=0.0, value=250.0, step=10.0)
            contenidos = st.number_input("Cantidad Contenidos Vistos", min_value=0, value=15, step=1)
            sesiones = st.number_input("Sesiones por Semana", min_value=0, value=4, step=1)
            porcentaje_fin = st.slider("Porcentaje Finalización (%)", 0.0, 100.0, 75.0)

        with col_f2:
            tiempo_sesion = st.number_input("Tiempo Prom. Sesión (min)", min_value=0.0, value=120.0, step=5.0)
            generos = st.number_input("Cantidad Géneros Consumidos", min_value=1, value=3, step=1)
            promociones = st.slider("Uso de Promociones (%)", 0.0, 1.0, 0.15)
            antiguedad = st.number_input("Antigüedad Cliente (meses)", min_value=0, value=12, step=1)
            edad = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)

        with col_f3:
            dispositivos = st.number_input("Dispositivos Registrados", min_value=1, value=2, step=1)
            app_movil = st.slider("Uso App Móvil (%)", 0.0, 1.0, 0.50)
            perfiles = st.number_input("Perfiles Creados", min_value=1, value=2, step=1)
            soporte = st.number_input("Interacciones Soporte", min_value=0, value=1, step=1)
            distancia = st.number_input("Distancia Prom. Red (km)", min_value=0.0, value=15.0, step=1.0)

        submitted = st.form_submit_button("Ejecutar Predicción del Modelo", type="primary")

    if submitted:
        payload_prediccion = {
            "horas_consumo_mensual": horas,
            "gasto_mensual": gasto,
            "cantidad_contenidos_vistos": contenidos,
            "sesiones_semana": sesiones,
            "porcentaje_finalizacion": porcentaje_fin,
            "tiempo_promedio_sesion_min": tiempo_sesion,
            "cantidad_generos_consumidos": generos,
            "porcentaje_uso_promociones": promociones,
            "antiguedad_cliente_meses": antiguedad,
            "edad": edad,
            "dispositivos_registrados": dispositivos,
            "porcentaje_uso_app_movil": app_movil,
            "cantidad_perfiles_creados": perfiles,
            "interacciones_mensuales_soporte": soporte,
            "distancia_promedio_red_km": distancia
        }

        try:
            respuesta_pred = requests.post("http://ml-service:8000/predict", json=payload_prediccion)
            if respuesta_pred.status_code == 200:
                resultado = respuesta_pred.json()
                st.success(f"Predicción Exitosa: El usuario ha sido clasificado en el **Segmento {resultado.get('cluster_asignado')}**")
            else:
                st.error(f"Error en la predicción. Código: {respuesta_pred.status_code}")
        except Exception as e:
            st.error(f"No se pudo conectar con el servicio de inferencia: {e}")