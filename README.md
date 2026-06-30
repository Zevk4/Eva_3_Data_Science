# Plataforma de Segmentación de Usuarios Streaming

## Título y Propósito

**Plataforma de Segmentación de Usuarios Streaming - Evaluacion 3**

Sistema de segmentación de usuarios basado en machine learning para servicios de streaming. La plataforma implementa un pipeline de datos end-to-end que procesa información de comportamiento de clientes, entrena modelos de clustering KMeans y proporciona una interfaz interactiva para análisis de segmentos de mercado y predicción en tiempo real.

### Valor de Negocio:

*   Identificación automática de segmentos de mercado basados en patrones de consumo
*   Optimización de estrategias de marketing mediante perfilamiento de usuarios
*   Capacidades de predicción en tiempo real para clasificación de nuevos clientes
*   Dashboard unificado para equipos de negocio y técnicos

## Arquitectura Técnica

El sistema implementa una arquitectura de microservicios orquestada mediante Docker Compose, compuesta por cuatro servicios principales que siguen una cadena de dependencia lineal.

### Integración de Fuentes de Datos

El sistema integra múltiples fuentes de datos mediante un proceso ETL:

*   **PostgreSQL (SQL):** Base de datos relacional que almacena perfiles de usuarios. Se inicializa mediante scripts SQL y carga de CSV durante el arranque del contenedor.

*   **CSV (Archivos Locales):**
    *   `perfil_usuarios.csv`: Datos de perfil cargados directamente en la base de datos
    *   `data_clientes.csv`: Dataset unificado procesado por el ETL
    *   `clientes_segmentados.csv`: Resultado del proceso de segmentación

*   **JSON (Métricas):** Archivo `metricas.json` que almacena indicadores de rendimiento del modelo (Silhouette Score, varianza PCA) generado por el proceso ETL.

### Arquitectura de Contenedores Docker

El sistema utiliza cuatro servicios Docker con la siguiente cadena de dependencia:

**Servicios:**

*   **postgres:** Capa de base de datos PostgreSQL 16. Persiste datos mediante volumen `postgres_data` y monta scripts de inicialización.

*   **etl:** Servicio de procesamiento que ejecuta `train.py`, entrena el modelo KMeans y genera artefactos. Es un contenedor "run-to-completion" que finaliza tras el entrenamiento.

*   **ml-service:** API REST FastAPI que sirve predicciones y datos procesados. Expone puerto 8000 y depende de `postgres` y `etl`.

*   **dashboard:** Interfaz Streamlit para visualización. Expone puerto 8501 y depende de `ml-service`.

**Volúmenes Compartidos:**

*   `postgres_data`: Persistencia de base de datos
*   `modelos_vol`: Almacena modelos `.pkl` (KMeans, Scaler, PCA) compartido entre `etl`, `ml-service` y `dashboard`
*   `./data`: Bind mount para datasets compartidos entre servicios.

# Tecnologías utilizadas

## Lenguaje

- Python 3.11

## Machine Learning

- Scikit-learn
- KMeans
- StandardScaler
- Silhouette Score
- Método del codo mediante KneeLocator


## Datos

- Pandas
- PostgreSQL
- SQLAlchemy


## Backend

- FastAPI
- Uvicorn


## Visualización

- Streamlit


## Infraestructura

- Docker
- Docker Compose




# Estructura del proyecto


```

Eva_3_Data_Science/

│
├── docker-compose.yml
│
├── database/
│ ├── init.sql
│ └── perfil_usuarios.csv
│
├── data/
│ ├── raw/
│ │ └── usuarios_streaming.csv
│ └── processed/
│ ├── data_clientes.csv
│ ├── clientes_segmentados.csv
│ └── centroides.csv
│
├── docker/
│ ├── etl.Dockerfile
│ ├── api.Dockerfile
│ └── dashboard.Dockerfile
│
├── etl/
│ ├── train.py
│ └── requirements.txt
│
├── docs/
│ ├── diagramas/
│    ├── arquitectura.mmd
│    └── flujo_etl.mmd
├── api/
│ ├── app.py
│ └── requirements.txt
│
└── dashboards/
├── app.py
└── requirements.txt


```



## Guía de Instalación Reproducible

### Prerrequisitos

*   Docker Engine 20.10+
*   Docker Compose 2.0+
*   Git

### Pasos de Instalación

1.  **Clonar el Repositorio**

    ```bash
    git clone <repository-url>
    cd Eva_3_Data_Science
    ```

2.  **Configurar Variables de Entorno**

    Crear archivo `.env` en la raíz del proyecto con las siguientes variables:

    ```bash
    DB_USER=postgres
    DB_PASS=your_secure_password
    DB_NAME=streaming_db
    ```
    Estas variables son referenciadas en `docker-compose.yml` para la configuración de PostgreSQL.

3.  **Verificar Estructura de Archivos**

    Asegurar que existen los siguientes directorios y archivos:

    *   `./database/init.sql`
    *   `./database/perfil_usuarios.csv`
    *   `./data/` (para datasets procesados)
    *   `./docker/` (contiene Dockerfiles)
    *   `./etl/`, `./api/`, `./dashboards/` (código fuente)

4.  **Construir y Levantar Servicios**

    El orden de encendido es crítico y está gestionado automáticamente por Docker Compose mediante la directiva `depends_on`:

    ```bash
    docker-compose up --build
    ```

    **Secuencia de Arranque:**

    *   `postgres` se inicia primero y ejecuta `init.sql`
    *   `etl` espera a `postgres` y ejecuta el entrenamiento
    *   `ml-service` espera a `etl` y `postgres` para iniciar la API
    *   `dashboard` espera a `ml-service` para levantar la interfaz web.

5.  **Verificar Estado de Servicios**

    ```bash
    docker-compose ps
    ```
    Los servicios `postgres`, `ml-service`, y `dashboard` deben estar activos. El servicio `etl` habrá finalizado exitosamente (`exit code 0`).


### Integración de Fuentes (ETL)

El proceso ETL se ejecuta en el contenedor `etl` mediante el script `train.py`. Este servicio:

*   Extrae datos de PostgreSQL y archivos CSV locales
*   Aplica transformaciones (escalado, PCA)
*   Entrena modelo KMeans
*   Guarda artefactos en volumen compartido `modelos_vol`
*   Genera `clientes_segmentados.csv` y `metricas.json` (`etl.Dockerfile:13-14`).

### Arquitectura de Contenedores Docker

La arquitectura sigue el patrón de microservicios con separación de responsabilidades:

*   Cada servicio tiene su propio Dockerfile especializado
*   Comunicación mediante volúmenes compartidos (no archivos locales)
*   Orquestación mediante Docker Compose con gestión de dependencias
*   Aislamiento de entornos mediante contenedores Python 3.11

### Descripción del Dashboard

El dashboard es una aplicación Streamlit que se divide en tres pestañas orientadas a diferentes audiencias:

1.  **Vista de Negocio y Operaciones (Product Managers / Marketing):**
    *   Análisis de cuota de mercado mediante gráficos de anillo
    *   Perfilamiento de segmentos con gráficos radiales (spider charts)
    *   Directorio de usuarios segmentados
    *   Métricas: total de usuarios, número de segmentos identificados.

2.  **Vista Técnica de Machine Learning (Data Scientists / ML Engineers):**
    *   Coeficiente Silhouette para evaluación de calidad de clustering
    *   Varianza retenida por PCA
    *   Proyección 2D de clusters
    *   Matriz de dispersión dinámica bivariada con centroides.

3.  **Simulador de Predicción en Vivo (Sales / Customer Support):**
    *   Formulario interactivo con 15 variables de comportamiento
    *   Clasificación en tiempo real mediante endpoint `/predict`
    *   Feedback inmediato del segmento asignado.

## Instrucciones de Uso

### Acceso a la Interfaz

Una vez levantados los servicios, acceder a:

*   **Dashboard:** `http://localhost:8501`
*   **API Documentation:** `http://localhost:8000/docs` (FastAPI auto-docs)
*   **API Health Check:** `http://localhost:8000/`

### Prueba del Endpoint de Predicción

**Opción 1: Mediante el Dashboard**

*   Navegar a la pestaña "Simulador de Predicción (En Vivo)"
*   Completar el formulario con los 15 parámetros de comportamiento del usuario
*   Clic en "Ejecutar Predicción del Modelo"
*   Visualizar el segmento asignado en el mensaje de éxito

**Opción 2: Mediante cURL (API directa)**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "horas_consumo_mensual": 55.5,
    "gasto_mensual": 250.0,
    "cantidad_contenidos_vistos": 12,
    "sesiones_semana": 5,
    "porcentaje_finalizacion": 75.0,
    "tiempo_promedio_sesion_min": 120.0,
    "cantidad_generos_consumidos": 4,
    "porcentaje_uso_promociones": 0.15,
    "antiguedad_cliente_meses": 24,
    "edad": 34,
    "dispositivos_registrados": 3,
    "porcentaje_uso_app_movil": 0.6,
    "cantidad_perfiles_creados": 2,
    "interacciones_mensuales_soporte": 1,
    "distancia_promedio_red_km": 12.4
  }'
```
El endpoint valida los 15 campos requeridos mediante Pydantic antes de procesar la predicción.

## Créditos y Gestión

### Equipo de Desarrollo:

*   [Yaii Selti] - Data Engineer
*   [Gino Andrades] - Data Scientist

### Prácticas de Gestión:

**Ramas (Branching):**

*   `main`: Rama principal para producción
*   `Gino_Andrades`: Rama de desarrollo personal de Gino Andrades
*   `feature/api-fastapi`: Ramas de desarrolo personal de Yaii Selti
*   `yaiis/cambios-funciones`: Ramas de desarrolo personal de Yaii Selti

**Issues:**

*   Utilizar plantillas para reportar bugs y feature requests
*   Incluir etiquetas de prioridad y tipo
*   Referenciar commits relacionados

**Pull Requests:**

*   Solicitar revisión mínima de 1 aprobación
*   Asegurar que todos los tests pasen
*   Actualizar documentación si es necesario
*   Seguir convenciones de commits (Conventional Commits)
