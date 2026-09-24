"""
train_failure_a.py  ── FAILURE DEMO A  (DO NOT USE IN PRODUCTION)
─────────────────────────────────────────────────────────────────
Purpose : Demonstrate that the quality gate fails and blocks artifact upload.
Change   : Train RandomForest on only 10 % of the training data, which makes
           the model too weak to beat the DummyRegressor by the required margin.
Restore  : Switch back to train.py after capturing the failed run link.
"""

import sys, json, os
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH    = "data/automobile.csv"
MODEL_PATH   = "model/model.pkl"
METRICS_PATH = "model/metrics.json"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
MARGIN       = 500.0

REQUIRED_FEATURES = [
    "symboling", "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "wheel-base",
    "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
]
TARGET = "price"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET])

X = df[REQUIRED_FEATURES].copy()
y = df[TARGET].copy()

categorical_cols = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system",
]
numerical_cols = [c for c in REQUIRED_FEATURES if c not in categorical_cols]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ── DELIBERATE WEAKNESS: use only 10 % of training data ──────────────────────
X_train_tiny, _, y_train_tiny, _ = train_test_split(
    X_train, y_train, test_size=0.99, random_state=RANDOM_STATE
)
print(f"[FAILURE A] Training on only {X_train_tiny.shape[0]} rows instead of {X_train.shape[0]}")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])

baseline = Pipeline([("pre", preprocessor), ("reg", DummyRegressor(strategy="mean"))])
baseline.fit(X_train, y_train)
baseline_mae = mean_absolute_error(y_val, baseline.predict(X_val))

rf = Pipeline([("pre", preprocessor),
               ("reg", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))])
rf.fit(X_train_tiny, y_train_tiny)          # ← tiny dataset
model_mae = mean_absolute_error(y_val, rf.predict(X_val))

print(f"Baseline MAE : ${baseline_mae:,.2f}")
print(f"Model MAE    : ${model_mae:,.2f}  (expected to be worse)")

gate_passed = model_mae <= (baseline_mae - MARGIN)
metrics = {
    "baseline_mae": round(baseline_mae, 2),
    "model_mae": round(model_mae, 2),
    "margin": MARGIN,
    "required_max_mae": round(baseline_mae - MARGIN, 2),
    "gate_passed": gate_passed,
}
os.makedirs("model", exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

if not gate_passed:
    print(f"\n[QUALITY GATE ❌] Model MAE ${model_mae:,.2f} does NOT beat "
          f"baseline by ${MARGIN:,.2f}. No artifact will be published.")
    sys.exit(1)

joblib.dump(rf, MODEL_PATH)
print("Model saved.")
