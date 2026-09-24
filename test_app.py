"""
test_app.py
Automated tests for the automobile price prediction application.

Tests:
  1. Model loads successfully from disk.
  2. A valid sample input produces a float prediction of the correct shape.
  3. An input missing a required feature is rejected with a clear ValueError.
"""

import pytest
import joblib
from predict import predict_price, load_model, MODEL_PATH, REQUIRED_FEATURES

# ── Shared valid sample ───────────────────────────────────────────────────────
VALID_INPUT = {
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


# ── Test 1: Model loads ───────────────────────────────────────────────────────
def test_model_loads():
    """The saved model pipeline can be loaded from disk without errors."""
    model = load_model(MODEL_PATH)
    assert model is not None, "Model should not be None after loading."
    # Check it has a predict method (i.e. it's a scikit-learn pipeline)
    assert hasattr(model, "predict"), "Loaded object must have a predict() method."


# ── Test 2: Valid input → prediction of correct type/shape ───────────────────
def test_valid_prediction():
    """A complete valid input dict produces a single float prediction."""
    price = predict_price(VALID_INPUT.copy())
    assert isinstance(price, float), f"Prediction should be float, got {type(price)}."
    assert price > 0, f"Predicted price should be positive, got {price}."
    # Sanity bounds: dataset range is $5,118 – $45,400; allow some extrapolation
    assert 1000 < price < 100_000, (
        f"Predicted price ${price:,.2f} is outside the plausible range."
    )


# ── Test 3: Missing feature → clear error ─────────────────────────────────────
def test_missing_feature_rejected():
    """Input missing a required feature must raise ValueError with a clear message."""
    incomplete = VALID_INPUT.copy()
    del incomplete["horsepower"]          # remove a required feature

    with pytest.raises(ValueError) as exc_info:
        predict_price(incomplete)

    error_msg = str(exc_info.value)
    assert "horsepower" in error_msg, (
        f"Error message should name the missing feature 'horsepower'. Got: {error_msg}"
    )


# ── Test 4: Multiple missing features all reported ───────────────────────────
def test_multiple_missing_features_reported():
    """Multiple missing features should all be listed in the error message."""
    incomplete = VALID_INPUT.copy()
    del incomplete["make"]
    del incomplete["city-mpg"]

    with pytest.raises(ValueError) as exc_info:
        predict_price(incomplete)

    error_msg = str(exc_info.value)
    assert "make" in error_msg, "Error should mention missing feature 'make'."
    assert "city-mpg" in error_msg, "Error should mention missing feature 'city-mpg'."
