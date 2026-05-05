from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import warnings

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import uvicorn

from .services.classifier import MODEL_PATH, load_model, preprocess_image


BASE_DIR = Path(__file__).resolve().parent


class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float | None = None


app = FastAPI(title="Perceptron Digit Classifier")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

model: Any | None = None
model_error: str | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global model, model_error
    try:
        model = load_model(MODEL_PATH)
        model_error = None
    except Exception as exc:
        model = None
        model_error = str(exc)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Perceptron Digit Classifier"},
    )


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "ready": model is not None,
        "model_path": str(MODEL_PATH),
        "error": model_error,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if model is None:
        message = "Model is not loaded."
        if model_error:
            message = f"{message} {model_error}"
        raise HTTPException(status_code=503, detail=message)

    if file.content_type != "image/png":
        raise HTTPException(status_code=400, detail="Please upload a PNG image.")

    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
        with Image.open(io.BytesIO(contents)) as image:
            img_flat = preprocess_image(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from exc

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        prediction = int(model.predict(img_flat)[0])

        confidence = None
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(img_flat)[0])
            confidence = round(1 / (1 + pow(2.718281828, -abs(score))), 4)

    return PredictResponse(
        prediction=prediction,
        label=str(prediction),
        confidence=confidence,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
