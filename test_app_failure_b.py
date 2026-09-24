"""
test_app_failure_b.py  ── FAILURE DEMO B  (DO NOT USE IN PRODUCTION)
─────────────────────────────────────────────────────────────────────
Purpose : Demonstrate that a broken application test stops the pipeline
          and prevents artifact upload.
Change   : test_valid_prediction() asserts price < 0, which is always False
           for a car price prediction, causing the test to fail.
Restore  : Switch back to test_app.py after capturing the failed run link.
"""

import pytest
from predict import predict_price, load_model, MODEL_PATH

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


def test_model_loads():
    model = load_model(MODEL_PATH)
    assert model is not None


def test_valid_prediction():
    price = predict_price(VALID_INPUT.copy())
    # ── BUG INTRODUCED: price can never be negative ────────────────────────
    assert price < 0, f"[INTENTIONAL BUG] Expected negative price, got {price}"


def test_missing_feature_rejected():
    incomplete = VALID_INPUT.copy()
    del incomplete["horsepower"]
    with pytest.raises(ValueError) as exc_info:
        predict_price(incomplete)
    assert "horsepower" in str(exc_info.value)
