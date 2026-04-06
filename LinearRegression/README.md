# Linear Regression

This project demonstrates a simple linear regression workflow for predicting delivery time from delivery distance.

It includes:

- a training script built with gradient descent
- a saved model in JSON format
- a FastAPI web app for predictions
- an analytics page to inspect the fitted line and training loss

## Project Overview

The model learns a relationship between:

- `DistanceKm` as the input feature
- `DeliveryTimeMin` as the target value

The trained values are stored in `saved_model/model_values.json` and are loaded by the FastAPI app at startup.

## Folder Structure

```text
LinearRegression/
├── app/
│   ├── main.py
│   ├── static/
│   └── templates/
├── data/
│   └── data.csv
├── model/
│   └── train.py
├── saved_model/
│   └── model_values.json
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to run the training script, make sure `matplotlib` and `numpy` are available in your environment.

## Run the App

Start the FastAPI server with:

```bash
uvicorn app.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/` for the prediction page
- `http://127.0.0.1:8000/analytics` for the analytics dashboard

## Train the Model

Run the training script from the project root:

```bash
python model/train.py
```

This script:

- reads the dataset from `data/data.csv`
- trains a linear regression model using gradient descent
- stores the learned slope and intercept in `saved_model/model_values.json`
- saves training checkpoints for the analytics page

## API Endpoints

- `GET /` renders the main prediction page
- `GET /analytics` renders the analytics page
- `GET /api/model` returns the saved model values
- `GET /api/analytics` returns model details, dataset points, and training history
- `POST /api/predict` predicts delivery time from a distance value
- `POST /predict` returns a typed prediction response

Example request:

```json
{
  "distance_km": 7.5
}
```

## Notes

- The app expects `saved_model/model_values.json` to exist before startup.
- The current dataset is focused on delivery distance and delivery time examples.
- This project is a good starting point for learning how linear regression, model serving, and simple analytics fit together.
