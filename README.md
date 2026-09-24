#  Automated Car Price Prediction ML Pipeline

**Lab Assignment 4 — Automating an ML Pipeline with GitHub Actions**

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | [UCI Machine Learning Repository — Automobile Dataset (1985 Ward's)](https://archive.ics.uci.edu/ml/datasets/automobile) |
| **File** | `data/automobile.csv` (converted from `imports-85.data`) |
| **Size** | 205 rows × 26 columns (~15 KB) |
| **Target variable** | `price` — the retail selling price of the car in USD |
| **Input features** | 24 features (see below) |
| **Prediction task** | Regression — predict a continuous dollar price |

### Input Features

**Numerical (14):** `symboling`, `wheel-base`, `length`, `width`, `height`, `curb-weight`, `engine-size`, `bore`, `stroke`, `compression-ratio`, `horsepower`, `peak-rpm`, `city-mpg`, `highway-mpg`

**Categorical (10):** `make`, `fuel-type`, `aspiration`, `num-of-doors`, `body-style`, `drive-wheels`, `engine-location`, `engine-type`, `num-of-cylinders`, `fuel-system`

> `normalized-losses` is excluded — it has 41 missing values (~20 %) and would introduce significant imputation noise for a small dataset.

---

## Model & Baseline

| | Model |
|---|---|
| **Candidate** | `RandomForestRegressor` (100 trees, sklearn default depth) |
| **Baseline** | `DummyRegressor(strategy="mean")` — predicts the training-set mean for every car |

**Train / validation split:** 80 % / 20 %, `random_state=42` (fixed for reproducibility).  
Preprocessing (StandardScaler + OneHotEncoder) is **fit only on training data** and applied to validation data, preventing data leakage.

---

## Evaluation Metric

**Mean Absolute Error (MAE)** — lower is better.

**Why MAE suits this task:**
Car prices span $5,118–$45,400. MAE expresses error in the same units as price (USD), making results directly interpretable ("the model is off by $X on average"). Compared with RMSE, MAE is less sensitive to a handful of extreme luxury cars, which is appropriate here since most of the dataset is mid-range vehicles. For a regression task where we care about average prediction accuracy across all cars, MAE is the natural choice.

---

## Quality Gate

```
gate condition: model_mae <= baseline_mae - margin
```

| Parameter | Value |
|---|---|
| **Margin** | $500 |
| **Typical baseline MAE** | ~$8,262 |
| **Typical model MAE** | ~$1,804 |

**Why $500?**  
The DummyRegressor predicts the mean (~$13,000) for every car. The Random Forest is expected to reduce MAE from ~$8,000 to ~$2,000 — a massive improvement. A $500 margin is large enough to reject a genuinely weak model (e.g., one trained on very little data or with the wrong hyperparameters) while being trivially achievable by a properly trained Random Forest. A margin that is too low would let nearly any model pass; a margin that is too high could reject a good model if the dataset shifts slightly.

The margin, metric, and split are **not changed** during the failure demonstrations.

---

## Repository Structure

```
car-price-ml-pipeline/
│
├── data/
│   └── automobile.csv          # UCI Automobile dataset (included in repo)
│
├── train.py                    # Load, validate, train, evaluate, save model
├── predict.py                  # Prediction application / inference function
├── test_app.py                 # Automated pytest tests (production)
├── requirements.txt            # Python dependencies
│
├── train_failure_a.py          # Failure A demo script (weakened model)
├── test_app_failure_b.py       # Failure B demo script (broken test)
│
├── model/                      # Created at runtime by train.py
│   ├── model.pkl
│   └── metrics.json
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml     # GitHub Actions CI/CD workflow
│
└── README.md
```

---

## Setup Instructions (Local / VS Code)

### Prerequisites

- Python 3.10 or 3.11
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Kowsya/car_price_prediction.git
cd car_price_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

Expected output (last few lines):
```
Gate condition : 1804.01 <= 7762.03 → PASSED ✅
Model saved to: model/model.pkl
Metrics saved to: model/metrics.json
✅ Training pipeline completed successfully.
```

### 5. Run a prediction

```bash
python predict.py
```

Expected output:
```
Predicted Price: $7,XXX.XX
```

### 6. Run the automated tests

```bash
pytest test_app.py -v
```

Expected output — all 4 tests pass:
```
test_app.py::test_model_loads                       PASSED
test_app.py::test_valid_prediction                  PASSED
test_app.py::test_missing_feature_rejected          PASSED
test_app.py::test_multiple_missing_features_reported PASSED
```

---

## Failure & Recovery Demonstrations

### Failure A — Model quality failure

**What was changed:**  
In `train_failure_a.py`, the RandomForest is trained on only **10 % of the training data** (≈16 rows). This makes the model too weak to beat the DummyRegressor by the required $500 margin. The quality gate in `train.py` is unchanged.

**To reproduce:**  
In `.github/workflows/ml-pipeline.yml`, temporarily change `python train.py` → `python train_failure_a.py`, commit, and push.

**Which check prevented publication:**  
The quality gate in the training step exits with `sys.exit(1)`, causing the workflow to fail before the artifact upload step runs.

🔴 **Failed run A link:** *(https://github.com/Kowsya/car_price_prediction/actions/runs/35987138507)

---

### Failure B — Application test failure

**What was changed:**  
In `test_app_failure_b.py`, `test_valid_prediction()` asserts `price < 0`, which is always False for a car price. This causes pytest to fail.

**To reproduce:**  
In `.github/workflows/ml-pipeline.yml`, temporarily change `pytest test_app.py` → `pytest test_app_failure_b.py`, commit, and push.

**Which check prevented publication:**  
The pytest step exits with a non-zero code, and the artifact upload step — which has no `if: always()` — does not run.

🔴 **Failed run B link:** (https://github.com/Kowsya/car_price_prediction/actions/runs/35988120919)

---

### Final successful run

After restoring `ml-pipeline.yml` to use `train.py` and `test_app.py`:

🟢 **Successful run link:** *https://github.com/Kowsya/car_price_prediction/actions/runs/35989161351*

**Downloadable artifact:** `car-price-model-run<N>-<commit-sha>` — available in the Actions tab → successful run → Artifacts section.

---

## Pipeline Questions

### 1. Why does your evaluation metric suit your task?

MAE is expressed in the same units as the target (USD), making it directly interpretable. It treats all prediction errors equally — a $500 error is 5× worse than a $100 error — which matches our goal of consistent accuracy across the price range, without over-penalising outliers like ultra-luxury cars.

### 2. Why did you choose this improvement margin? What would happen if it were too low or too high?

$500 is a substantial but easy-to-achieve bar for a properly trained Random Forest on this data (which beats the baseline by ~$6,000). A margin that is too low (e.g., $10) would allow a nearly-broken model to slip through and get published. A margin that is too high (e.g., $7,000) might block a genuinely good model on a bad random seed or when data is slightly smaller, creating false negatives in the gate.

### 3. What caused each failed run? Which check prevented publication?

- **Failure A:** The model was trained on only 10 % of the training data, resulting in a MAE above the threshold. The quality gate (inside `train.py`) called `sys.exit(1)`, stopping the workflow before the artifact upload.
- **Failure B:** A deliberate bug (`assert price < 0`) was placed in the test file. pytest exited with code 1, and the artifact upload step never ran.

### 4. Which parts of your workflow demonstrate continuous integration and artifact delivery?

**Continuous integration:** Every push to `main` automatically runs data validation, model training, quality evaluation, and automated tests. Any failure blocks progress.  
**Artifact delivery:** The final step packages the trained model, metrics report, prediction script, and requirements into a versioned zip file (named with the run number and commit SHA) and uploads it as a GitHub Actions artifact — downloadable without requiring a deployment environment.

### 5. Which MLOps maturity level best describes your implementation?

**Level 1 — ML Pipeline Automation** (Google MLOps maturity model).

This pipeline automates the entire training → evaluation → testing → packaging sequence on every code push, which goes beyond Level 0 (manual, notebook-based). The quality gate and automated tests enforce reproducibility and catch regressions.

**What remains to reach Level 2 (CI/CD for ML):**
- Automated retraining triggered by new data, not just code changes.
- A model registry (e.g., MLflow, Weights & Biases) tracking experiment history.
- Deployment to a live serving endpoint (API, container).
- Monitoring for data drift or prediction degradation in production.
