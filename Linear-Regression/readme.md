# Machine-Learning

A simple machine-learning project that trains and serves a **linear regression model from scratch** to predict delivery time from distance.

The repository currently includes a complete FastAPI app under `Linear-Regression/` with:
- a gradient-descent training script,
- persisted model values,
- a prediction UI,
- and an analytics dashboard.

## Project structure

```text
Machine-Learning/
├── README.md
├── requirements.txt
└── Linear-Regression/
    ├── app/
    │   ├── main.py
    │   ├── static/style.css
    │   └── templates/
    │       ├── index.html
    │       └── analytics.html
    ├── data/data.csv
    ├── model/train.py
    ├── requirements.txt
    └── saved_model/model_values.json
```

## What this project does

- Trains a linear model `y = mx + b` using manual gradient descent.
- Saves learned parameters (`m`, `b`) and training metadata to JSON.
- Loads the saved model in a FastAPI service.
- Provides:
  - an interactive predictor page (`/`),
  - an analytics page (`/analytics`) with Chart.js visualizations,
  - API endpoints for predictions and model analytics.

## Requirements

- Python 3.10+
- pip

Python dependencies:
- `fastapi`
- `uvicorn[standard]`
- `jinja2`
- `pydantic`
- `numpy`
- `matplotlib`

> Note: `numpy` and `matplotlib` are required by `model/train.py`, while FastAPI-related packages are required by the web app.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install numpy matplotlib
```

(Alternatively, install from `Linear-Regression/requirements.txt` plus `numpy` and `matplotlib`.)

## Train the model

```bash
python Linear-Regression/model/train.py
```

This reads `Linear-Regression/data/data.csv` and writes model outputs to:

`Linear-Regression/saved_model/model_values.json`

## Run the web app

```bash
uvicorn Linear-Regression.app.main:app --reload
```

Open:
- Predictor UI: <http://127.0.0.1:8000/>
- Analytics UI: <http://127.0.0.1:8000/analytics>
- OpenAPI docs: <http://127.0.0.1:8000/docs>

## API endpoints

- `GET /api/model` — returns saved model values.
- `GET /api/analytics` — returns model values + data points + training history + MSE.
- `POST /api/predict` — returns prediction JSON for `{ "distance_km": number }`.
- `POST /predict` — same prediction payload, typed with Pydantic response model.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 7.5}'
```

## Notes

- The app expects `saved_model/model_values.json` to exist at startup.
- If the model file is missing, run the training script first.
- The training script displays a matplotlib plot when it finishes.
