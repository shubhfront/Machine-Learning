from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import os
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'delivery_data.csv')

_model = None
_le_traffic = None
_le_weather = None
_le_road = None
_le_vehicle = None
_metrics = None
# Make matplotlib safe for server environments


# Make matplotlib safe for server environments
# Make matplotlib safe for server environments
# Make matplotlib safe for server environments
def train_model():
    global _model, _le_traffic, _le_weather, _le_road, _le_vehicle, _metrics
    
    # Load CSV
    with open(DATA_PATH, mode='r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Read CSV with pandas
    df = pd.read_csv(DATA_PATH)
    df_label = df.copy()

    # Encode categorical features (EXACT same as original)
    _le_traffic = LabelEncoder()
    _le_weather = LabelEncoder()
    _le_road = LabelEncoder()
    _le_vehicle = LabelEncoder()
    
    df_label["traffic_level_encoded"] = _le_traffic.fit_transform(df_label["traffic_level"])
    df_label["weather_condition_encoded"] = _le_weather.fit_transform(df_label["weather_condition"])
    df_label["road_condition_encoded"] = _le_road.fit_transform(df_label["road_condition"])
    df_label["vehicle_type_encoded"] = _le_vehicle.fit_transform(df_label["vehicle_type"])

    # Build features exactly as original
    X = df_label[[
        "distance_km",
        "traffic_level_encoded",
        "weather_condition_encoded",
        "road_condition_encoded",
        "vehicle_type_encoded",
        "is_peak_hour",
        "stops",
        "parcel_weight_kg",
        "pickup_delay_min"
    ]]

    y = df_label["delivery_time_min"]

    # Train/test split exactly as original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model exactly as original
    _model = LinearRegression()
    _model.fit(X_train, y_train)

    # Get predictions
    train_prediction = _model.predict(X_train)
    test_prediction = _model.predict(X_test)

    # Calculate metrics for API
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, test_prediction)
    rmse = mean_squared_error(y_test, test_prediction) ** 0.5
    r2 = r2_score(y_test, test_prediction)
    train_r2 = _model.score(X_train, y_train)
    
    _metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "train_r2": round(float(train_r2), 4)
    }

    print(f"Model trained successfully!")
    print(f"Slope: {_model.coef_}")
    print(f"Intercept: {_model.intercept_}")
    print(f"Training Accuracy: {_model.score(X_train, y_train)}")
    print(f"Testing Accuracy: {_model.score(X_test, y_test)}")


def predict_delivery_time(distance_km, traffic_level, weather_condition, road_condition, 
                         vehicle_type, is_peak_hour, stops, parcel_weight_kg, pickup_delay_min):
    global _model, _le_traffic, _le_weather, _le_road, _le_vehicle
    
    if _model is None:
        raise ValueError("Model not trained yet")
    
    # Encode categorical inputs
    traffic_encoded = _le_traffic.transform([traffic_level])[0]
    weather_encoded = _le_weather.transform([weather_condition])[0]
    road_encoded = _le_road.transform([road_condition])[0]
    vehicle_encoded = _le_vehicle.transform([vehicle_type])[0]
    
    # Build feature array exactly as original
    new_val = pd.DataFrame(
        [[distance_km, traffic_encoded, weather_encoded, road_encoded, 
          vehicle_encoded, is_peak_hour, stops, parcel_weight_kg, pickup_delay_min]],
        columns=[
            "distance_km",
            "traffic_level_encoded",
            "weather_condition_encoded",
            "road_condition_encoded",
            "vehicle_type_encoded",
            "is_peak_hour",
            "stops",
            "parcel_weight_kg",
            "pickup_delay_min"
        ]
    )
    
    return float(_model.predict(new_val)[0])


def get_category_options():
    """Get unique values for categorical fields"""
    df = pd.read_csv(DATA_PATH)
    return {
        "traffic_level": sorted(df["traffic_level"].unique().tolist()),
        "weather_condition": sorted(df["weather_condition"].unique().tolist()),
        "road_condition": sorted(df["road_condition"].unique().tolist()),
        "vehicle_type": sorted(df["vehicle_type"].unique().tolist())
    }


def get_metrics():
    """Return training metrics"""
    return _metrics


# Test the model when run directly
if __name__ == "__main__":
    train_model()

    # Test prediction as in original
    new_val = pd.DataFrame(
        [[11, 0, 1, 0, 1, 0, 4, 30, 0]],
        columns=[
            "distance_km",
            "traffic_level_encoded",
            "weather_condition_encoded",
            "road_condition_encoded",
            "vehicle_type_encoded",
            "is_peak_hour",
            "stops",
            "parcel_weight_kg",
            "pickup_delay_min"
        ]
    )
    print(f"\nTest Prediction: {_model.predict(new_val)[0]}")
