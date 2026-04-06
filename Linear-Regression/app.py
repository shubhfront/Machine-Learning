from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import linear

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")


class DeliveryFeatures(BaseModel):
    distance_km: float = Field(..., ge=0)
    traffic_level: str
    weather_condition: str
    road_condition: str
    vehicle_type: str
    is_peak_hour: int = Field(..., ge=0, le=1)
    stops: int = Field(..., ge=0)
    parcel_weight_kg: float = Field(..., ge=0)
    pickup_delay_min: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    predicted_delivery_time_min: float


@dataclass
class AppState:
    training: bool = False
    trained: bool = False
    error: str | None = None


state = AppState()
app = FastAPI(title="Delivery Time Predictor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


async def train_in_background() -> None:
    state.training = True
    state.error = None
    try:
        await asyncio.to_thread(linear.train_model)
        state.trained = True
    except Exception as exc:
        state.error = str(exc)
        state.trained = False
    finally:
        state.training = False


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(train_in_background())


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "title": "Delivery Time Predictor"},
    )


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    categories = linear.get_category_options()
    metrics = linear.get_metrics()
    return {
        "training": state.training,
        "trained": state.trained,
        "error": state.error,
        "metrics": metrics,
        "categories": categories,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def api_predict(payload: DeliveryFeatures) -> PredictResponse:
    if state.training:
        raise HTTPException(status_code=503, detail="Model is still training. Please try again shortly.")

    if not state.trained:
        message = "Model is not ready yet."
        if state.error:
            message = f"Model training failed: {state.error}"
        raise HTTPException(status_code=503, detail=message)

    prediction = linear.predict_delivery_time(
        payload.distance_km,
        payload.traffic_level,
        payload.weather_condition,
        payload.road_condition,
        payload.vehicle_type,
        payload.is_peak_hour,
        payload.stops,
        payload.parcel_weight_kg,
        payload.pickup_delay_min
    )
    return PredictResponse(predicted_delivery_time_min=round(prediction, 2))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
