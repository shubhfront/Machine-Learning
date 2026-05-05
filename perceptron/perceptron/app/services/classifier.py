from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_DIR / "models" / "model.pkl"
SAMPLES_DIR = APP_DIR / "samples"
IMAGE_SIZE = (28, 28)


def train_and_save_model(model_path: Path = MODEL_PATH) -> tuple[Perceptron, float]:
    mnist = fetch_openml("mnist_784", parser="auto")

    X = mnist.data
    y = mnist.target.astype(int)

    mask = (y == 0) | (y == 1)
    X = X[mask] / 255.0
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, model_path)
    return model, accuracy


def load_model(model_path: Path = MODEL_PATH) -> Perceptron:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("L").resize(IMAGE_SIZE)
    img_array = np.array(img)

    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0
    return img_array.flatten().reshape(1, -1)


def predict_image(image: Image.Image, model: Perceptron | None = None) -> int:
    active_model = model or load_model()
    img_flat = preprocess_image(image)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        prediction = active_model.predict(img_flat)
    return int(prediction[0])


def predict_image_file(image_path: str | Path, model: Perceptron | None = None) -> int:
    with Image.open(image_path) as image:
        return predict_image(image, model)


if __name__ == "__main__":
    model, accuracy = train_and_save_model()
    print("Accuracy:", accuracy)

    sample_path = SAMPLES_DIR / "images.png"
    if sample_path.exists():
        print("Prediction:", predict_image_file(sample_path, model))
