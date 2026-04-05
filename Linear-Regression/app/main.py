from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "saved_model" / "model_values.json"
DATA_FILE = BASE_DIR / "data" / "data.csv"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(title="Delivery Time Predictor")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PredictionInput(BaseModel):
    distance_km: float = Field(..., gt=0, description="Delivery distance in kilometers")


class PredictionResponse(BaseModel):
    distance_km: float
    predicted_delivery_time_min: float
    equation: str


MODEL_VALUES = None


def load_model_values() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_FILE}. Run model/train.py first."
        )

    with MODEL_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "m" not in data or "b" not in data:
        raise ValueError("model_values.json must contain both 'm' and 'b'.")

    return data


def load_csv_points() -> list[dict]:
    points: list[dict] = []

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"CSV file not found at {DATA_FILE}")

    with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            points.append(
                {
                    "distance_km": float(row["DistanceKm"]),
                    "delivery_time_min": float(row["DeliveryTimeMin"]),
                }
            )

    return points


def mean_squared_error(points: list[dict], m: float, b: float) -> float:
    if not points:
        return 0.0

    total = 0.0
    for point in points:
        predicted = m * point["distance_km"] + b
        total += (predicted - point["delivery_time_min"]) ** 2
    return total / len(points)


@app.on_event("startup")
def startup_event() -> None:
    global MODEL_VALUES
    MODEL_VALUES = load_model_values()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"model_ready": MODEL_VALUES is not None},
    )


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={"model_ready": MODEL_VALUES is not None},
    )


@app.get("/api/model")
def get_model():
    if MODEL_VALUES is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    return MODEL_VALUES


@app.get("/api/analytics")
def get_analytics_model():
    if MODEL_VALUES is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    points = load_csv_points()
    m = float(MODEL_VALUES["m"])
    b = float(MODEL_VALUES["b"])

    distances = [point["distance_km"] for point in points]
    times = [point["delivery_time_min"] for point in points]

    payload = dict(MODEL_VALUES)
    payload["data_points"] = points
    payload["sample_count"] = len(points)
    payload["distance_min"] = min(distances) if distances else None
    payload["distance_max"] = max(distances) if distances else None
    payload["time_min"] = min(times) if times else None
    payload["time_max"] = max(times) if times else None
    payload["final_mse"] = mean_squared_error(points, m, b)
    return payload


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    if MODEL_VALUES is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    m = float(MODEL_VALUES["m"])
    b = float(MODEL_VALUES["b"])
    predicted_time = m * data.distance_km + b

    return PredictionResponse(
        distance_km=data.distance_km,
        predicted_delivery_time_min=round(predicted_time, 2),
        equation=f"y = {m:.4f}x + {b:.4f}",
    )


@app.post("/api/predict")
def predict_api(data: PredictionInput):
    if MODEL_VALUES is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    m = float(MODEL_VALUES["m"])
    b = float(MODEL_VALUES["b"])
    predicted_time = m * data.distance_km + b

    return JSONResponse(
        {
            "distance_km": data.distance_km,
            "predicted_delivery_time_min": round(predicted_time, 2),
            "model": {"m": m, "b": b},
        }
    )
