# Sistema de Segmentación Analítica de Usuarios de Streaming
### Programación para la Ciencia de Datos | Evaluación 3

---

## Descripción del Proyecto
Este proyecto implementa una solución tecnológica de extremo a extremo (*end-to-end*) diseñada para el sector analítico de una plataforma corporativa de contenido en streaming. El objetivo de negocio consiste en **segmentar de forma automatizada la base de clientes** en función de sus patrones e indicadores de comportamiento (horas de consumo mensual, gasto, sesiones semanales, antigüedad, etc.). 

A través de esta segmentación, el equipo de operaciones puede identificar clústeres clave de usuarios (por ejemplo, clientes fidelizados, usuarios en riesgo de abandono o consumidores ocasionales) para diseñar estrategias personalizadas de retención y marketing analítico.

---

## Arquitectura del Sistema
El ecosistema completo está diseñado bajo una arquitectura de microservicios contenerizados independientes que se coordinan y comunican de manera unificada a través de una red virtual compartida:

1. **Base de Datos (`postgres`):** Motor relacional PostgreSQL 16 encargado de almacenar de manera persistente los registros de perfil de los usuarios. Inicializa automáticamente su esquema DDL mediante scripts SQL y cargas de archivos base.
2. **Pipeline de Datos (`etl`):** Contenedor efímero que se enciende de forma síncrona tras la disponibilidad de la base de datos. Se encarga de la extracción de registros, preprocesamiento técnico, escalamiento y el entrenamiento del modelo de aprendizaje no supervisado **KMeans**. Al finalizar, exporta los artefactos de Machine Learning (.pkl y .json) hacia un volumen compartido y se apaga automáticamente de forma limpia.
3. **Servicio Backend Restful (`ml-service`):** API construida sobre el framework **FastAPI** que corre en el puerto `8000`. Carga en memoria los artefactos generados por el ETL, implementa registros de logs para auditoría, valida estrictamente los datos entrantes mediante esquemas Pydantic y expone endpoints clave para predicciones en vivo (`/predict`) y consultas unificadas de métricas estadísticas (`/dashboard-data`).
4. **Dashboard Interactivo (`dashboard`):** Interfaz gráfica interactiva desarrollada con **Streamlit** que se expone en el puerto `8501`. Funciona como la capa frontend del sistema, consumiendo los endpoints analíticos de la API para renderizar visualizaciones de mercado, métricas ejecutivas clave y dispersión de clústeres de manera intuitiva para los usuarios finales.

---

## Estructura del Repositorio
El repositorio sigue una organización modular limpia, separando las configuraciones de infraestructura del código fuente de cada microservicio:

```text
eva_3_data_science/
├── docker-compose.yml              # Orquestador multi-contenedor global
├── .gitignore                      # Exclusión de archivos temporales y locales
├── .env                            # Variables de entorno locales (Excluido de internet)
├── api/                            # Backend FastAPI
│   ├── app.py                      # Código principal del servidor REST
│   └── requirements.txt            # Dependencias del backend
├── dashboards/                     # Frontend Streamlit
│   ├── app.py                      # Código de la interfaz interactiva gráfica
│   └── requirements.txt            # Dependencias del frontend
├── docker/                         # Recetas de compilación de imágenes Docker
│   ├── api.Dockerfile
│   ├── dashboard.Dockerfile
│   └── etl.Dockerfile
├── etl/                            # Pipeline de Machine Learning
│   ├── train.py                    # Script de entrenamiento KMeans y extracción
│   └── requirements.txt            # Librerías científicas (scikit-learn, pandas)
├── data/                           # Almacenamiento persistente compartida
│   ├── raw/                        # Datos crudos
│   └── processed/                  # Datos limpios (centroides.csv, clientes_segmentados.csv)
└── database/                       # Scripts de la Base de Datos Relacional
    ├── init.sql                    # Script DDL de inicialización de tablas
    └── perfil_usuarios.csv         # Datos iniciales para poblar PostgreSQL
