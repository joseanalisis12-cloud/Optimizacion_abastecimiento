# Optimización de Abastecimiento

Pipeline de forecasting cuantílico con LightGBM + optimización modelo "Newsvendor" (vendedor de periódicos para productos perecederos), servido mediante FastAPI y trackeado con MLflow. Dockerizado para despliegue reproducible.

---

## Arquitectura de la solución

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRENAMIENTO  (local · notebooks/OD.ipynb)                    │
│                                                                  │
│  dataset_retail_unificado.csv                                   │
│       │                                                          │
│       ▼                                                          │
│  Feature Engineering  ──►  LightGBM Quantile Regression        │
│  (lags, rolling, CV)        P10 · P50 · P90                     │
│                                  │                               │
│                                  ▼                               │
│                         MLflow Tracking                          │
│                         (métricas + artefactos)                  │
│                                  │                               │
│                    notebooks/mlruns/   +   mlflow_run_ids.json  │
└──────────────────────────────────┬──────────────────────────────┘
                                   │  volumen compartido
┌──────────────────────────────────▼──────────────────────────────┐
│  PRODUCCIÓN  (Docker Compose)                                    │
│                                                                  │
│  ┌─────────────────────┐     ┌──────────────────────────────┐  │
│  │  mlflow_server      │     │  forecast_api                │  │
│  │  python:3.11-slim   │     │  Dockerfile (python:3.11)    │  │
│  │  puerto 5000        │◄────│  puerto 8000                 │  │
│  │                     │     │                              │  │
│  │  UI de experimentos │     │  POST /predict               │  │
│  │  Model Registry     │     │  → P10 / P50 / P90           │  │
│  └─────────────────────┘     │  → Q* Newsvendor             │  │
│                               │  → pedido recomendado       │  │
│                               └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes principales

| Componente : Descripción |
|
| `notebooks/OD.ipynb` : Pipeline completo: EDA, feature engineering, entrenamiento, validación |
| `api.py` : Servidor FastAPI que carga los modelos y expone `/predict` |
| `Dockerfile` : Imagen del servicio `forecast_api` |
| `docker-compose.yml` : Orquesta `mlflow_server` + `forecast_api` |
| `requirements.txt` : Dependencias Python del contenedor de la API |
| `notebooks/mlruns/` : Artefactos y métricas generados por MLflow |
| `notebooks/mlflow_run_ids.json` : IDs de los runs de P10, P50 y P90 |

### Flujo de datos

1. El notebook entrena 3 modelos LightGBM (uno por cuantil) y los registra en MLflow
2. MLflow guarda métricas, parámetros e importancia de features en `mlruns/`
3. La API carga los modelos desde `mlruns/` al arrancar
4. El endpoint `/predict` recibe las features de un SKU-Tienda, obtiene P10/P50/P90 y calcula el pedido óptimo con el modelo Newsvendor

---

## Requisitos previos

| Herramienta | Versión mínima | Verificar |
|---|---|---|
| Python | 3.11 | `python --version` |
| Docker Desktop | 26.x | `docker --version` |
| WSL2 | Habilitado | `wsl --status` |
| Git | Cualquiera | `git --version` |

> **Windows:** WSL2 y la virtualización deben estar habilitados en el BIOS.  
> **Docker Desktop:** debe estar corriendo con el motor Linux activo antes de ejecutar cualquier comando Docker.

---

## Estructura de carpetas

```
FORECAST/
├── notebooks/
│   ├── OD.ipynb                  # pipeline principal
│   ├── mlflow_run_ids.json       # generado al entrenar
│   └── mlruns/                   # generado al entrenar
├── api.py                        # servidor FastAPI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md
```

---

## Instalación y ejecución

### Opción A — Ejecución local (sin Docker)

#### 1. Clonar o ubicarse en la carpeta del proyecto

```bash
cd "C:\Users\USUARIO\Documents\KONRAD LORENZ\MAESTRÍA\MLOPS\FORECAST"
```

#### 2. Crear y activar el entorno virtual

```bash
python -m venv venv
```

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Entrenar los modelos

Abrir `notebooks/OD.ipynb` en VSCode, seleccionar el kernel del entorno virtual y ejecutar todas las celdas con `Ctrl+Alt+R`. Al finalizar se generan:

```
notebooks/mlruns/
notebooks/mlflow_run_ids.json
```

#### 5. Explorar experimentos en MLflow UI

```bash
mlflow ui --backend-store-uri "file:///ruta/absoluta/FORECAST/notebooks/mlruns"
```

Abrir `http://localhost:5000` en el navegador.

#### 6. Levantar la API

```bash
uvicorn api:app --reload
```

Abrir `http://localhost:8000/docs` para acceder al Swagger UI.

---

### Opción B — Ejecución con Docker (recomendado)

#### 1. Verificar que Docker Desktop está corriendo

```bash
docker info
docker ps
```

#### 2. Entrenar primero los modelos localmente

Completar el paso 4 de la Opción A antes de continuar. Los archivos `mlruns/` y `mlflow_run_ids.json` deben existir antes de levantar Docker.

#### 3. Construir y levantar los servicios

```bash
cd "C:\Users\USUARIO\Documents\KONRAD LORENZ\MAESTRÍA\MLOPS\FORECAST"
docker compose up --build
```

La primera vez tarda 3-5 minutos mientras descarga las imágenes base e instala dependencias.

#### 4. Verificar que los servicios están activos

```bash
docker compose ps
```

Resultado esperado:

```
NAME             STATUS    PORTS
mlflow_server    running   0.0.0.0:5000->5000/tcp
forecast_api     running   0.0.0.0:8000->8000/tcp
```

#### 5. Acceder a los servicios

| Servicio | URL |
|---|---|
| FastAPI — Swagger UI | http://localhost:8000/docs |
| FastAPI — Health check | http://localhost:8000/health |
| MLflow — UI | http://localhost:5000 |

#### 6. Detener los servicios

```bash
docker compose down
```

---

## Uso de la API

### GET /health

Verifica que la API está activa y cuántos modelos están cargados.

```bash
curl http://localhost:8000/health
```

Respuesta:

```json
{"status": "healthy", "n_modelos": 3}
```

### POST /predict

Recibe las features de un SKU-Tienda y retorna el pronóstico cuantílico más el pedido óptimo Newsvendor.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lag_1w": 45, "lag_2w": 42, "lag_3w": 40, "lag_4w": 38,
    "rolling_mean_2w": 43.5, "rolling_mean_4w": 41.25,
    "rolling_std_2w": 2.1, "rolling_std_4w": 2.9,
    "cv": 0.07, "delta_lag1_lag2": 3.0,
    "ratio_tienda_vs_media": 1.05,
    "semana_num": 14, "proporcion_finde": 0.28,
    "costo_unitario": 800, "precio_venta": 1500,
    "costo_almacenamiento_semanal": 25,
    "margen_unitario": 700, "margen_pct": 0.46,
    "tamaño_m2": 120,
    "trend_enc": 2, "cat_enc": 0,
    "tienda_enc": 0, "producto_enc": 4,
    "stock_actual": 10
  }'
```

Respuesta:

```json
{
  "P10": 38.5,
  "P50": 45.2,
  "P90": 53.8,
  "sigma_hat": 5.97,
  "critical_ratio": 0.9654,
  "Q_optimo": 56.8,
  "pedido_recomendado": 47,
  "nivel_servicio_pct": 96.5,
  "agresividad": "MUY AGRESIVO"
}
```

### POST /predict/batch

Mismo esquema pero acepta una lista de SKUs en un solo request.

---

## Descripción de los campos de respuesta

| Campo | Descripción |
|---|---|
| `P10` | Límite inferior del pronóstico (cuantil 10%) |
| `P50` | Pronóstico central — mediana de la demanda |
| `P90` | Límite superior del pronóstico (cuantil 90%) |
| `sigma_hat` | Incertidumbre estimada: `(P90 - P10) / 2.56` |
| `critical_ratio` | `Cu / (Cu + Co)` — ratio que determina agresividad del pedido |
| `Q_optimo` | Cantidad óptima según el modelo Newsvendor |
| `pedido_recomendado` | `max(0, Q* - stock_actual)` — unidades a pedir |
| `nivel_servicio_pct` | Probabilidad de no tener stockout (%) |
| `agresividad` | Perfil del pedido: MUY CONSERVADOR → MUY AGRESIVO |

---

## Solución de problemas comunes (manejo de errores)

| Error | Causa | Solución |
|---|---|---|
| `No such file or directory: mlflow_run_ids.json` | El notebook no ha corrido | Ejecutar todas las celdas de `OD.ipynb` primero |
| `Run not found` | MLflow apunta a la carpeta incorrecta | Verificar que `MLFLOW_URI` apunta a `notebooks/mlruns/` |
| `Connection refused` al cargar modelos | La API arrancó antes que MLflow | Esperar el reintento automático (hasta 10 intentos × 10 s) |
| `uvicorn: executable not found` | Falta en `requirements.txt` | Verificar que `uvicorn` está listado y reconstruir con `--build` |
| `WSL2 not compatible` | Virtualización deshabilitada en BIOS | Habilitar `VT-x` o `AMD-V` en la configuración del BIOS |
| `500 Internal Server Error — float division by zero` | `margen_unitario` o `costo_almacenamiento_semanal` es 0 | Verificar que los campos tienen valores positivos en el payload |

 {
  "lag_1w": 45.0,
  "lag_2w": 39.0,
  "lag_3w": 38.0,
  "lag_4w": 57.0,
  "rolling_mean_2w": 42.0,
  "rolling_mean_4w": 44.75,
  "rolling_std_2w": 4.242640687119285,
  "rolling_std_4w": 8.73212459828649,
  "cv": 0.19513127593500243,
  "delta_lag1_lag2": 6.0,
  "ratio_tienda_vs_media": 0.864406779646366,
  "semana_num": 13,
  "proporcion_finde": 0.42857142857142855,
  "costo_unitario": 800.0,
  "precio_venta": 2500.0,
  "costo_almacenamiento_semanal": 10.0,
  "margen_unitario": 1700.0,
  "margen_pct": 0.68,
  "tama\u00f1o_m2": 28.0,
  "trend_enc": 3,
  "cat_enc": 1,
  "tienda_enc": 0,
  "producto_enc": 0,
  "stock_actual": 1
}

---

## Tecnologías utilizadas

| Tecnología | Rol |
|---|---|
| LightGBM | Modelo de forecasting cuantílico (P10/P50/P90) |
| MLflow | Tracking de experimentos y model registry |
| FastAPI | Servidor REST para inferencia |
| Uvicorn | Servidor ASGI para FastAPI |
| Docker Compose | Orquestación de servicios |
| Pandas / NumPy | Procesamiento de datos |
| SciPy | Función de distribución normal para Newsvendor |
| Scikit-learn | Métricas de evaluación del modelo |
