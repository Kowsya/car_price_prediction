"""
train.py
Automobile Price Prediction — Training & Evaluation
Dataset: UCI Automobile Dataset (imports-85)
Target: price (regression)
Model: Random Forest Regressor vs DummyRegressor baseline
Metric: Mean Absolute Error (MAE) — lower is better
"""

import sys
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/automobile.csv"
MODEL_PATH  = "model/model.pkl"
METRICS_PATH = "model/metrics.json"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
MARGIN       = 500.0   # $500 improvement over baseline MAE  (units: USD)

# Required columns (validation check)
REQUIRED_FEATURES = [
    "symboling", "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "wheel-base",
    "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
]
TARGET = "price"


# ── 1. Load & validate ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and validating dataset")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Column validation
missing_cols = [c for c in REQUIRED_FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    print(f"[ERROR] Missing required columns: {missing_cols}")
    sys.exit(1)
print(f"[OK] All required columns present.")

# Drop rows where target is missing
df = df.dropna(subset=[TARGET])
print(f"Rows after dropping missing target: {df.shape[0]}")

if df.shape[0] < 50:
    print("[ERROR] Not enough rows after cleaning.")
    sys.exit(1)


# ── 2. Feature engineering ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preparing features")
print("=" * 60)

X = df[REQUIRED_FEATURES].copy()
y = df[TARGET].copy()

# Drop normalized-losses (too many missing) — already excluded from REQUIRED_FEATURES

categorical_cols = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "engine-location", "engine-type", "num-of-cylinders",
    "fuel-system",
]
numerical_cols = [c for c in REQUIRED_FEATURES if c not in categorical_cols]

print(f"Numerical features  : {numerical_cols}")
print(f"Categorical features: {categorical_cols}")

# Train/validation split — fixed seed for reproducibility
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")


# ── 3. Preprocessing pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])


# ── 4. Train baseline (DummyRegressor) ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training baseline (DummyRegressor)")
print("=" * 60)

baseline_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DummyRegressor(strategy="mean")),
])
baseline_pipeline.fit(X_train, y_train)
baseline_preds = baseline_pipeline.predict(X_val)
baseline_mae = mean_absolute_error(y_val, baseline_preds)
print(f"Baseline MAE : ${baseline_mae:,.2f}")


# ── 5. Train candidate model (RandomForestRegressor) ─────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training Random Forest Regressor")
print("=" * 60)

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_val)
model_mae = mean_absolute_error(y_val, rf_preds)
print(f"Random Forest MAE : ${model_mae:,.2f}")


# ── 6. Quality gate ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Quality gate")
print("=" * 60)

required_max = baseline_mae - MARGIN
gate_passed = model_mae <= required_max

print(f"Baseline MAE      : ${baseline_mae:,.2f}")
print(f"Model MAE         : ${model_mae:,.2f}")
print(f"Margin            : ${MARGIN:,.2f}")
print(f"Required max MAE  : ${required_max:,.2f}  (baseline - margin)")
print(f"Gate condition    : {model_mae:.2f} <= {required_max:.2f} → {'PASSED ✅' if gate_passed else 'FAILED ❌'}")

metrics = {
    "baseline_mae": round(baseline_mae, 2),
    "model_mae": round(model_mae, 2),
    "margin": MARGIN,
    "required_max_mae": round(required_max, 2),
    "gate_passed": gate_passed,
    "train_size": int(X_train.shape[0]),
    "val_size": int(X_val.shape[0]),
}

if not gate_passed:
    os.makedirs("model", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[ERROR] Quality gate FAILED. Model MAE ${model_mae:,.2f} does not beat "
          f"baseline MAE ${baseline_mae:,.2f} by at least ${MARGIN:,.2f}.")
    sys.exit(1)


# ── 7. Save model & metrics ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving model and metrics")
print("=" * 60)

os.makedirs("model", exist_ok=True)
joblib.dump(rf_pipeline, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {METRICS_PATH}")
print("\n Training pipeline completed successfully.")
