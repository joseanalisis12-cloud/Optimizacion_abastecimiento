# api.py
# Ejecutar con: uvicorn api:app --reload
# Endpoint: POST http://localhost:8000/predict

import os
import json
import numpy as np
import pandas as pd
import mlflow.lightgbm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from scipy.stats import norm

# ── Rutas base ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# MLflow apunta a notebooks/mlruns/ donde se entrenó el modelo
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"file:///{NOTEBOOKS_DIR}/mlruns".replace("\\", "/")
)
mlflow.set_tracking_uri(MLFLOW_URI)
print(f"  MLflow URI: {MLFLOW_URI}")

# Run IDs desde notebooks/mlflow_run_ids.json
RUN_IDS_PATH = os.path.join(NOTEBOOKS_DIR, "mlflow_run_ids.json")

FEATURES = [
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "rolling_mean_2w", "rolling_mean_4w",
    "rolling_std_2w",  "rolling_std_4w",
    "cv", "delta_lag1_lag2",
    "ratio_tienda_vs_media",
    "semana_num", "proporcion_finde",
    "costo_unitario", "precio_venta",
    "costo_almacenamiento_semanal",
    "margen_unitario", "margen_pct",
    "tamaño_m2",
    "trend_enc", "cat_enc",
    "tienda_enc", "producto_enc",
]

# ── Carga de modelos al arrancar la API ────────────────────────────────────────
print("Cargando modelos desde MLflow...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_IDS_PATH = os.path.join(BASE_DIR, "notebooks", "mlflow_run_ids.json")

with open(RUN_IDS_PATH) as f:
    run_ids = json.load(f)[0]   # {"P10": "...", "P50": "...", "P90": "..."}

modelos = {}
for nombre_q, run_id in run_ids.items():
    uri = f"runs:/{run_id}/model_{nombre_q}"
    modelos[nombre_q] = mlflow.lightgbm.load_model(uri)
    print(f"  ✓ {nombre_q} cargado desde run {run_id[:8]}...")

app = FastAPI(
    title       = "API de Forecast y Pedido Óptimo",
    description = "LightGBM Quantile Regression + Newsvendor — Hospital / Retail",
    version     = "1.0.0",
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class FeaturesInput(BaseModel):
    lag_1w:                      float
    lag_2w:                      float
    lag_3w:                      float
    lag_4w:                      float
    rolling_mean_2w:             float
    rolling_mean_4w:             float
    rolling_std_2w:              float
    rolling_std_4w:              float
    cv:                          float
    delta_lag1_lag2:             float
    ratio_tienda_vs_media:       float
    semana_num:                  int
    proporcion_finde:            float
    costo_unitario:              float
    precio_venta:                float
    costo_almacenamiento_semanal:float
    margen_unitario:             float
    margen_pct:                  float
    tamaño_m2:                   float
    trend_enc:                   int
    cat_enc:                     int
    tienda_enc:                  int
    producto_enc:                int
    stock_actual:                Optional[int] = 0

class PredictionResponse(BaseModel):
    P10:                  float
    P50:                  float
    P90:                  float
    sigma_hat:            float
    critical_ratio:       float
    Q_optimo:             float
    pedido_recomendado:   int
    nivel_servicio_pct:   float
    agresividad:          str

# ── Lógica Newsvendor ──────────────────────────────────────────────────────────
def newsvendor(mu, sigma, cu, co):
    sigma = max(float(sigma), 1e-6)
    cu    = max(float(cu), 0.0)
    co    = max(float(co), 1e-6)   # evita división por cero

    if (cu + co) == 0:
        raise ValueError(f"cu ({cu}) + co ({co}) no pueden ser ambos 0")

    cr    = cu / (cu + co)
    z     = norm.ppf(cr)
    q_opt = max(0.0, float(mu) + z * sigma)
    ns    = norm.cdf((q_opt - float(mu)) / sigma) * 100

    if   cr >= 0.90: agresividad = "MUY AGRESIVO"
    elif cr >= 0.75: agresividad = "AGRESIVO"
    elif cr >= 0.55: agresividad = "MODERADO"
    elif cr >= 0.35: agresividad = "CONSERVADOR"
    else:            agresividad = "MUY CONSERVADOR"

    return cr, q_opt, ns, agresividad

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "modelos_cargados": list(modelos.keys())}

@app.get("/health")
def health():
    return {"status": "healthy", "n_modelos": len(modelos)}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: FeaturesInput):
    try:
        # Construir DataFrame con el orden exacto de features
        row = pd.DataFrame([data.model_dump()])[FEATURES]

        p10 = float(np.maximum(0, modelos["P10"].predict(row))[0])
        p50 = float(np.maximum(0, modelos["P50"].predict(row))[0])
        p90 = float(np.maximum(0, modelos["P90"].predict(row))[0])

        sigma = (p90 - p10) / (2 * 1.2816)
        cu    = data.margen_unitario
        co    = data.costo_almacenamiento_semanal

        cr, q_opt, ns, agresividad = newsvendor(p50, sigma, cu, co)
        pedido = max(0, int(round(q_opt - data.stock_actual)))

        return PredictionResponse(
            P10                = round(p10, 2),
            P50                = round(p50, 2),
            P90                = round(p90, 2),
            sigma_hat          = round(sigma, 2),
            critical_ratio     = round(cr, 4),
            Q_optimo           = round(q_opt, 2),
            pedido_recomendado = pedido,
            nivel_servicio_pct = round(ns, 1),
            agresividad        = agresividad,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(records: list[FeaturesInput]):
    return [predict(r) for r in records]