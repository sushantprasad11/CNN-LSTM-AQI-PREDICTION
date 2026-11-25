# 🚀 AQI Prediction CNN-LSTM — Complete Project User Guide

> **A Complete, Practical, Step-by-Step Manual for Running, Testing, Predicting, and Extending the AQI Prediction System.**

---

## 📋 Table of Contents

1. [System Requirements & Prerequisites](#1-system-requirements--prerequisites)
2. [Environment Setup & Installation](#2-environment-setup--installation)
3. [Quick Command Cheat Sheet](#3-quick-command-cheat-sheet)
4. [How to Launch the Interactive Web Dashboard](#4-how-to-launch-the-interactive-web-dashboard)
5. [How to Predict AQI for Any Date and Time (Past or Future)](#5-how-to-predict-aqi-for-any-date-and-time-past-or-future)
6. [How to Run the Complete Pipeline from Scratch](#6-how-to-run-the-complete-pipeline-from-scratch)
7. [How to Run Individual Modular Stages](#7-how-to-run-individual-modular-stages)
8. [How to Run the Research Extension Experiments](#8-how-to-run-the-research-extension-experiments)
9. [How to Run the Unit Test Suite](#9-how-to-run-the-unit-test-suite)
10. [Developer Guide: Programmatic Python Usage](#10-developer-guide-programmatic-python-usage)
11. [Configuration Reference (`config/config.yaml`)](#11-configuration-reference-configconfigyaml)
12. [Troubleshooting, Common Issues & FAQs](#12-troubleshooting-common-issues--faqs)

---

## 1. System Requirements & Prerequisites

- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS (12+)
- **Python Version:** Python 3.10, 3.11, or 3.12 (Tested and fully compatible)
- **Memory (RAM):**
  - For inference & web dashboard: 2 GB RAM minimum
  - For full model training: 8 GB RAM recommended
- **Storage:** ~1.5 GB free disk space (covers raw data, engineered features, trained models, and dependencies)
- **GPU:** Optional. TensorFlow will automatically use CUDA-compatible NVIDIA GPUs if configured, but CPU training runs cleanly in 15–30 minutes.
- **Internet Access:** Required only for downloading fresh real-time data from the Open-Meteo APIs. All pre-collected datasets (2022–2025) and pre-trained models are already included locally!

---

## 2. Environment Setup & Installation

### Step 2.1: Open Terminal in Project Root
Open PowerShell (on Windows) or Bash/Zsh (on Linux/macOS) and navigate to the project directory:
```bash
cd D:\AQI-Prediction-CNN-LSTM
```

### Step 2.2: (Recommended) Create and Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 2.3: Install Dependencies
Install all required libraries via `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### What gets installed:
- **`tensorflow` / `keras`**: Neural network modeling, layers, callbacks, and inference
- **`pandas` & `numpy`**: Data manipulation, sliding-window array slicing, time-series operations
- **`scikit-learn`**: Feature scaling (`StandardScaler`), multi-class classification metrics
- **`matplotlib` & `seaborn`**: Training loss/accuracy curves, confusion matrices, ablation charts
- **`flask`**: Interactive web dashboard and REST API
- **`scipy`**: McNemar's chi-squared statistical hypothesis testing
- **`requests`**: Querying Open-Meteo Air Quality and Weather APIs
- **`pyyaml`**: Parsing central configuration file `config/config.yaml`
- **`joblib`**: Serializing the fitted feature scaler

---

## 3. Quick Command Cheat Sheet

| Task | Command | Typical Time | Required Pre-conditions |
|:---|:---|:---:|:---|
| **Web Dashboard** | `python app/app.py` | Instant (< 3s) | Results exist in `results/` |
| **Predict Specific Datetime** | `python predict_any_datetime.py "2026-11-26 12:00"` | 5–10 sec | Internet connection, trained model |
| **Full Pipeline (Data → Training → Eval)** | `python run_full_pipeline.py` | 20–35 min | Raw CSVs present |
| **Ablation Study (6 Steps)** | `python run_ablation_study.py` | 30–50 min | Processed features present |
| **Meteorological Comparison** | `python run_meteorological_comparison.py` | 25–40 min | Processed features present |
| **Generate Visualizations & Metrics** | `python run_visualizations.py` | 1–2 min | Models & sequences present |
| **Execute Unit Tests (34 Tests)** | `python -m unittest discover -s tests` | ~5 sec | None |

---

## 4. How to Launch the Interactive Web Dashboard

The project includes an interactive web dashboard built with Flask and modern CSS/JavaScript.

### Step 4.1: Start the Server
```bash
python app/app.py
```
You will see output similar to:
```
 * Serving Flask app 'app.app'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Step 4.2: Open in Your Browser
Open any web browser and navigate to:
```
http://127.0.0.1:5000
```

### Step 4.3: Features Available in the Dashboard
1. **⚡ Dynamic Live Prediction Feed & Interactive Date/Time Predictor (Top of the Page):**
   - **Interactive Date & Time Selector:** Pick any historical or future date/time directly on the dashboard via the native calendar/time picker or 1-click preset chips (*Current Hour*, *Tomorrow 10:00*, *Day After 14:00*, *Monsoon Peak*, *Winter Peak*).
   - **One-Click AQI Prediction (`🔮 Predict AQI`):** Triggers deep learning model inference for the exact datetime, showing animated progress, health advisories, and category probability distribution bars.
   - **Real-Time Terminal Sync:** Automatically listens for prediction commands executed in PowerShell / Terminal (`predict_any_datetime.py`) and syncs immediately without page reload.
2. **Model Performance Tab:** Side-by-side comparison tables and charts for CNN (77.82%), LSTM (75.01%), and Hybrid CNN-LSTM (80.64%). View interactive confusion matrices and training loss/accuracy curves.
3. **Feature Ablation Tab:** Visualizes the 6-step incremental study proving that Step 3 (+ Rolling Averages) delivers the optimal F1-score of 0.7966.
4. **Meteorological Analysis Tab:** Direct breakdown of pollutant-only vs. pollutant+weather models with accuracy delta metrics and McNemar's p-values.
5. **Early Warning Module Tab:** Health alert evaluation metrics (93.67% alert precision, 90.48% alert F1-score) and sample alert advisories.
6. **Health Risk Advisories Tab:** Detailed medical guidelines for each of the 6 Indian CPCB AQI categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).

---

## 5. How to Predict AQI for Any Date and Time (Past or Future)

The script `predict_any_datetime.py` allows querying any exact hour of the year.

### Step 5.1: Predict for a Past Date (Historical Mode)
Fetches ground-truth historical pollutant and weather records from Open-Meteo's Archive API:
```bash
python predict_any_datetime.py "2025-06-15 14:00"
```

### Step 5.2: Predict for an Upcoming Date (Forecast Mode)
Fetches forecast meteorological and pollutant projections:
```bash
python predict_any_datetime.py "2026-11-26 12:00"
```

### Step 5.3: Default Test Run
```bash
python predict_any_datetime.py
```

### Sample Console Output:
```
======================================================================
  AQI PREDICTION RESULT FOR DELHI
======================================================================
  Target Datetime:    2025-06-15 14:00
  Mode:               HISTORICAL
  Model:              Hybrid CNN-LSTM (models/cnn_lstm/best_model.keras)

  Predicted AQI:      Moderate (Code: 2)
  Confidence:         84.32%

  Class Probabilities:
    [0] Good:          1.20%
    [1] Satisfactory:  8.45%
    [2] Moderate:     84.32%  ◄ PREDICTED
    [3] Poor:          5.11%
    [4] Very Poor:     0.82%
    [5] Severe:        0.10%

  Health Impact:      Moderate — Sensitive people should limit prolonged exertion.
======================================================================
```

---

## 7. How to Run the Complete Pipeline from Scratch

If you wish to re-execute the entire pipeline from raw data to trained models and evaluation:
```bash
python run_full_pipeline.py
```

### What this script executes sequentially:
1. Loads raw pollution and weather CSVs (`data/raw/`).
2. Cleans data with linear interpolation and IQR outlier capping.
3. Computes Indian CPCB AQI sub-indices and category labels.
4. Generates all 82 engineered features (lag, rolling, cyclic, ratio, weather).
5. Fits Z-score `StandardScaler` on training partition only and saves `artifacts/scaler.joblib`.
6. Generates 24-hour sliding window sequences and saves `data/processed/sequences.npz`.
7. Trains CNN, LSTM, and Hybrid CNN-LSTM models with EarlyStopping and ModelCheckpoint.
8. Evaluates all models on the 30% test split, saving metrics and confusion matrix plots to `results/`.

---

## 8. How to Run Individual Modular Stages

If you want fine-grained control over individual stages:

### Step 8.1: Run Preprocessing & Feature Engineering Only
```bash
python run_pipeline.py
```
- Fetches/merges raw data
- Computes AQI
- Generates 82 features
- Creates and saves `data/processed/sequences.npz`

### Step 8.2: Run Model Training & Evaluation Only
```bash
python run_training.py
```
- Loads pre-built `sequences.npz`
- Trains CNN, LSTM, and Hybrid CNN-LSTM models
- Saves checkpoints in `models/cnn/`, `models/lstm/`, `models/cnn_lstm/`
- Computes classification reports and McNemar's tests

---

## 9. How to Run the Research Extension Experiments

### Extension 1: Meteorological Feature Impact Study
Trains all 3 models under two distinct conditions:
- **Condition A:** Pollutant-only features (77 features)
- **Condition B:** Pollutant + Meteorological features (82 features)
```bash
python run_meteorological_comparison.py
```
Outputs saved to: `results/meteorological/weather_impact.csv` and `results/meteorological/weather_impact_comparison.png`.

### Extension 2: Step-Wise Feature Ablation Study
Retrains the Hybrid CNN-LSTM model across 6 incremental steps to isolate feature contributions:
1. Base Pollutants (6 features)
2. + Lag Features (42 features)
3. + Rolling Averages (66 features)
4. + Cyclic Encodings (74 features)
5. + Ratio Features (77 features)
6. + Meteorological Features (82 features)
```bash
python run_ablation_study.py
```
Outputs saved to: `results/ablation/ablation_results.csv` and `results/ablation/ablation_f1_comparison.png`.

### Visualization & Summary Generator
Regenerates all publication-ready plots, McNemar's test results, and `results/reports/full_results_summary.json`:
```bash
python run_visualizations.py
```

---

## 10. How to Run the Unit Test Suite

The project includes an automated test suite with **34 test cases** covering AQI calculations, feature engineering, neural network architectures, and alert logic.

### Running all unit tests:
```bash
# Windows PowerShell
$env:KERAS_HOME = "$((Get-Location).Path)\.keras"
python -m unittest discover -s tests -v

# Linux / macOS
KERAS_HOME="$(pwd)/.keras" python3 -m unittest discover -s tests -v
```

### Expected Output:
```
Ran 34 tests in ~5.0s
OK
```

### Test Coverage Breakdown:
- `tests/test_aqi.py`: Validates Indian CPCB sub-index formulas, breakpoint boundary conditions, and category mapping.
- `tests/test_preprocessing.py`: Validates linear interpolation, IQR outlier capping, cyclic sin/cos encoding, and 24-step sliding window generation.
- `tests/test_models.py`: Validates 1D CNN, LSTM, and Hybrid CNN-LSTM tensor shapes, layer counts, and output dimensions.
- `tests/test_prediction.py`: Validates model loading, single-sample inference, early-warning alert triggers, and health advisory mappings.

---

## 11. Developer Guide: Programmatic Python Usage

### Example 1: Load Trained Model and Predict on Custom Sequence
```python
import numpy as np
import tensorflow as tf
from src.preprocessing.scaler import AQIScaler

# 1. Load fitted scaler and trained model
scaler = AQIScaler()
scaler.load("artifacts/scaler.joblib")
model = tf.keras.models.load_model("models/cnn_lstm/best_model.keras")

# 2. Prepare 24 hours of 82-feature raw data (shape: 24, 82)
# raw_24h_data = np.random.rand(24, 82)
scaled_24h_data = scaler.transform_array(raw_24h_data)

# 3. Add batch dimension -> shape: (1, 24, 82)
input_tensor = np.expand_dims(scaled_24h_data, axis=0)

# 4. Predict
probabilities = model.predict(input_tensor)[0]
predicted_code = int(np.argmax(probabilities))

categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
print(f"Predicted Category: {categories[predicted_code]} ({probabilities[predicted_code]*100:.2f}%)")
```

### Example 2: Run Multi-Model Weighted Ensemble
```python
from src.prediction.ensemble import EnsemblePredictor

predictor = EnsemblePredictor(
    models_dir="models",
    weights={"cnn": 0.20, "lstm": 0.25, "cnn_lstm": 0.55}
)

# Predict on sliding window sequence (shape: 1, 24, 82)
result = predictor.predict_single(input_tensor[0])
print(f"Ensemble Category: {result['category']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Model Agreement: {result['agreement_ratio']*100:.1f}%")
```

### Example 3: Evaluate Early Warning Alerts
```python
from src.prediction.early_warning import generate_alerts, format_alert_message

# Category codes array: 0=Good, 4=Very Poor, 5=Severe
predicted_codes = np.array([1, 2, 4, 5, 3])
alert_flags = generate_alerts(predicted_codes)  # Returns [0, 0, 1, 1, 0]

for code, is_alert in zip(predicted_codes, alert_flags):
    if is_alert:
        msg = format_alert_message(code, aqi_value=350, timestamp="2026-11-14 14:00")
        print(f"ALERT [{msg['level']}]: {msg['headline']}")
        print(f"Advisory: {msg['advisory']}\n")
```

---

## 12. Configuration Reference (`config/config.yaml`)

All global parameters are maintained in a single configuration file:

```yaml
# Target geographic coordinates (Delhi, India)
data:
  city: "Delhi"
  latitude: 28.6139
  longitude: 77.2090
  start_date: "2022-01-01"
  end_date: "2025-12-31"

# 6 Tracked pollutants
  pollutants: ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone"]

# 5 Tracked weather variables
  weather_variables: ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "dew_point_2m"]

# Preprocessing parameters
preprocessing:
  interpolation_limit: 6       # Max consecutive missing hours to fill
  outlier_iqr_factor: 3.0      # Outlier capping threshold
  window_size: 24              # 24-hour sliding sequence length
  train_split: 0.70            # 70% chronological train / 30% test

# Training hyperparameters
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
  reduce_lr_patience: 7
```

To modify dates, coordinates, batch size, or sequence length, edit `config/config.yaml`. All scripts automatically inherit updates from this file!

---

## 13. Troubleshooting, Common Issues & FAQs

### Q1: `PermissionError: [Errno 13] Permission denied: '...\.keras\keras.json'`
- **Cause:** Keras tries to write to the user home directory when running in a sandboxed or restricted environment.
- **Fix:** Set the `KERAS_HOME` environment variable to a folder inside your workspace:
  ```powershell
  # PowerShell
  $env:KERAS_HOME = "$((Get-Location).Path)\.keras"
  ```
  ```bash
  # Linux/macOS
  export KERAS_HOME="$(pwd)/.keras"
  ```

### Q2: What if Open-Meteo API requests fail or time out?
- **Cause:** Temporary network outage or local DNS issue.
- **Fix:** All fetch functions chunk requests into 90-day intervals with automatic retry logic. For local development, pre-fetched raw datasets are already saved in `data/raw/` so API access is not needed for model training or evaluation!

### Q3: `ModuleNotFoundError: No module named 'src'`
- **Cause:** Running a script from a subfolder rather than the project root.
- **Fix:** Always execute commands from the project root (`D:\AQI-Prediction-CNN-LSTM`), or ensure `PYTHONPATH=.` is set:
  ```bash
  # Windows PowerShell
  $env:PYTHONPATH = "."
  ```

### Q4: Why are CNN, LSTM, and Hybrid accuracies slightly different on re-training?
- **Cause:** Stochastic gradient descent, dropout layers, and random weight initialization.
- **Fix:** Seed numbers are set in `config.yaml`, but minor floating-point variations occur across different CPU/GPU instruction sets. Test performance will consistently stay within ±0.5% of published figures.

