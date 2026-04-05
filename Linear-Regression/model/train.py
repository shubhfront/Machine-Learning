import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "data.csv"
MODEL_DIR = ROOT_DIR / "saved_model"
MODEL_FILE = MODEL_DIR / "model_values.json"

x = []
y = []

with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

print(x[:5])
print(y[:5])

m, b = 0, 0
al = 0.0001
epochs = 1000
history = []

for epoch in range(epochs):

    erY = []
    for i in range(len(x)):
        predY = m * x[i] + b
        erY.append(predY - y[i])

    sumer = 0
    for i in range(len(x)):
        sumer += (erY[i]) ** 2

    mse = (1 / len(x)) * sumer

    sumgr = 0
    for i in range(len(x)):
        sumgr += x[i] * erY[i]
    grad_m = (2 / len(x)) * sumgr

    sumbr = 0
    for i in range(len(x)):
        sumbr += erY[i]
    grad_b = (2 / len(x)) * sumbr

    m = m - (al * grad_m)
    b = b - (al * grad_b)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {mse}, m: {m}, b: {b}")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(mse, 6),
                "m": round(m, 6),
                "b": round(b, 6),
            }
        )

print("Final m:", m)
print("Final b:", b)

new_x = 10
pred = m * new_x + b
print("Prediction for x=10:", pred)

plt.scatter(x, y, color="blue")

line_y = []
for i in x:
    line_y.append(m * i + b)

plt.plot(x, line_y, color="red")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE.write_text(
    json.dumps(
        {
            "m": m,
            "b": b,
            "feature_name": "DistanceKm",
            "target_name": "DeliveryTimeMin",
            "sample_count": len(x),
            "distance_min": min(x),
            "distance_max": max(x),
            "time_min": min(y),
            "time_max": max(y),
            "data_points": [
                {"distance_km": distance, "delivery_time_min": time}
                for distance, time in zip(x, y)
            ],
            "training_history": history,
            "final_mse": mse,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Saved model to: {MODEL_FILE}")

plt.show()
