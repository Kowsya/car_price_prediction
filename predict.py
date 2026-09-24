"""
predict.py
Automobile Price Prediction — Inference Application

Usage (command line):
    python predict.py

Or import and call predict_price(input_dict) directly.
"""

import sys
import joblib
import pandas as pd

MODEL_PATH = "model/model.pkl"

REQUIRED_FEATURES = [
    "symboling", "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "wheel-base",
    "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
]


def load_model(model_path: str = MODEL_PATH):
    """Load the trained model pipeline from disk."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{model_path}'. "
                                "Run train.py first.")


def predict_price(input_dict: dict, model_path: str = MODEL_PATH) -> float:
    """
    Predict car price from a dictionary of feature values.

    Parameters
    ----------
    input_dict : dict
        Must contain all keys listed in REQUIRED_FEATURES.
    model_path : str
        Path to the saved model .pkl file.

    Returns
    -------
    float
        Predicted price in USD.

    Raises
    ------
    ValueError
        If any required feature is missing from input_dict.
    """
    # Validate input features
    missing = [f for f in REQUIRED_FEATURES if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    model = load_model(model_path)
    df = pd.DataFrame([input_dict])[REQUIRED_FEATURES]
    prediction = model.predict(df)
    return float(prediction[0])


# ── Example usage when run as a script ───────────────────────────────────────
if __name__ == "__main__":
    sample_car = {
        "symboling": 1,
        "make": "toyota",
        "fuel-type": "gas",
        "aspiration": "std",
        "num-of-doors": "four",
        "body-style": "sedan",
        "drive-wheels": "fwd",
        "engine-location": "front",
        "wheel-base": 98.4,
        "length": 170.7,
        "width": 64.0,
        "height": 54.0,
        "curb-weight": 2350,
        "engine-type": "ohc",
        "num-of-cylinders": "four",
        "engine-size": 97,
        "fuel-system": "2bbl",
        "bore": 3.19,
        "stroke": 3.03,
        "compression-ratio": 9.0,
        "horsepower": 70.0,
        "peak-rpm": 4800.0,
        "city-mpg": 30,
        "highway-mpg": 37,
    }

    try:
        price = predict_price(sample_car)
        print(f"Predicted Price: ${price:,.2f}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
