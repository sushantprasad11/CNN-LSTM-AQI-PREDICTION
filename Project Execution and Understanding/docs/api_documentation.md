# 🔌 API Documentation

## AQI Prediction using Hybrid CNN–LSTM Deep Learning Model

This document serves as the comprehensive API reference for:
1. **External Third-Party APIs** (Open-Meteo Air Quality & Weather services)
2. **Flask Web Dashboard REST Endpoints**
3. **Internal Python Module APIs** (Data, Preprocessing, Models, Training, Evaluation, and Prediction)

---

## 1. External Third-Party APIs (Open-Meteo)

The project leverages the **Open-Meteo API** suite. It requires no API keys, has no commercial paywalls, and provides global Copernicus Atmosphere Monitoring Service (CAMS) data.

### 1.1 Open-Meteo Air Quality API

- **Endpoint:** `GET https://air-quality-api.open-meteo.com/v1/air-quality`
- **Rate Limit:** ~10,000 requests/day

#### Request Parameters:

| Parameter | Type | Required | Description | Example |
|:---|:---:|:---:|:---|:---|
| `latitude` | float | Yes | Geographic latitude | `28.6139` |
| `longitude` | float | Yes | Geographic longitude | `77.2090` |
| `hourly` | string | Yes | Comma-delimited list of pollutant variables | `pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone` |
| `start_date` | string | Yes | Start date in `YYYY-MM-DD` | `2024-01-01` |
| `end_date` | string | Yes | End date in `YYYY-MM-DD` | `2024-03-31` |
| `timezone` | string | No | Target timezone (default: UTC) | `UTC` |

#### Tracked Variables:
- `pm2_5`: Particulate matter $< 2.5\ \mu\text{m}$ ($\mu\text{g/m}^3$)
- `pm10`: Particulate matter $< 10\ \mu\text{m}$ ($\mu\text{g/m}^3$)
- `nitrogen_dioxide`: $\text{NO}_2$ concentration ($\mu\text{g/m}^3$)
- `sulphur_dioxide`: $\text{SO}_2$ concentration ($\mu\text{g/m}^3$)
- `carbon_monoxide`: $\text{CO}$ concentration ($\mu\text{g/m}^3$)
- `ozone`: $\text{O}_3$ concentration ($\mu\text{g/m}^3$)

---

### 1.2 Open-Meteo Historical Weather Archive API

- **Endpoint:** `GET https://archive-api.open-meteo.com/v1/archive`
- **Rate Limit:** ~10,000 requests/day

#### Request Parameters:

| Parameter | Type | Required | Description | Example |
|:---|:---:|:---:|:---|:---|
| `latitude` | float | Yes | Geographic latitude | `28.6139` |
| `longitude` | float | Yes | Geographic longitude | `77.2090` |
| `hourly` | string | Yes | Comma-delimited weather parameters | `temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,dew_point_2m` |
| `start_date` | string | Yes | Start date in `YYYY-MM-DD` | `2024-01-01` |
| `end_date` | string | Yes | End date in `YYYY-MM-DD` | `2024-03-31` |
| `timezone` | string | No | Target timezone | `UTC` |

#### Tracked Variables:
- `temperature_2m`: Air temperature at 2m above ground (°C)
- `relative_humidity_2m`: Relative humidity at 2m (%)
- `wind_speed_10m`: Wind speed at 10m height ($\text{km/h}$)
- `surface_pressure`: Atmospheric surface pressure ($\text{hPa}$)
- `dew_point_2m`: Dew point temperature (°C)

---

### 1.3 Open-Meteo Weather Forecast API

- **Endpoint:** `GET https://api.open-meteo.com/v1/forecast`
- **Usage:** Used by `predict_any_datetime.py` to retrieve up to 16 days of future meteorological projections and 5 days of air quality projections.

---

## 2. Flask Web Dashboard REST Endpoints

The Flask application (`app/app.py`) provides the web dashboard and JSON endpoints for front-end integration.

### 2.1 Web Interface
- **Route:** `GET /`
- **Returns:** HTML5 interactive dashboard (`app/templates/index.html`)

### 2.2 Results Summary Endpoint
- **Route:** `GET /api/results`
- **Description:** Returns the complete experimental evaluation data including model comparison metrics, ablation study steps, meteorological comparisons, and early warning statistics.
- **Example Response:**
```json
{
  "summary": {
    "project": "AQI Prediction using Hybrid CNN-LSTM",
    "city": "Delhi, India",
    "model_comparison": [
      {
        "Model": "CNN",
        "accuracy": 0.7782,
        "precision": 0.7881,
        "recall": 0.7782,
        "f1_score": 0.7755
      },
      {
        "Model": "LSTM",
        "accuracy": 0.7501,
        "precision": 0.7709,
        "recall": 0.7501,
        "f1_score": 0.7392
      },
      {
        "Model": "Hybrid CNN-LSTM",
        "accuracy": 0.8064,
        "precision": 0.8070,
        "recall": 0.8064,
        "f1_score": 0.8059
      }
    ],
    "early_warning": {
      "alert_precision": 0.9367,
      "alert_recall": 0.8751,
      "alert_f1_score": 0.9048
    }
  }
}
```

### 2.3 AQI Categories Reference Endpoint
- **Route:** `GET /api/aqi-categories`
- **Description:** Returns all 6 Indian CPCB AQI categories with index ranges, visual hex color codes, health risks, and public health advisories.

### 2.4 Project Information Endpoint
- **Route:** `GET /api/project-info`
- **Description:** Returns project metadata, institutional details, and team member information.

### 2.5 Dynamic Live Prediction Endpoint
- **Route:** `GET /api/latest-prediction`
- **Description:** Returns the most recent prediction generated via CLI (`predict_any_datetime.py`) or web UI for real-time frontend polling.
- **Example Response:**
```json
{
  "status": "success",
  "prediction": {
    "type": "single_datetime",
    "command": "python predict_any_datetime.py \"2026-11-26 12:00\"",
    "updated_at": "2026-11-25 21:58:44",
    "summary": {
      "title": "AQI for 2026-11-26 12:00 (FORECAST)",
      "dominant_category": "Very Poor",
      "dominant_code": 4,
      "avg_confidence": 98.8,
      "target_datetime": "2026-11-26 12:00",
      "mode": "FORECAST",
      "risk_level": "Very High",
      "advisory": "Minimize outdoor activities. Use N95 masks. Keep windows closed."
    }
  }
}
```

### 2.6 On-Demand Prediction Runner Endpoint
- **Route:** `POST /api/run-prediction`
- **Headers:** `Content-Type: application/json`
- **Supported Payload:**
  ```json
  {
    "type": "datetime",
    "datetime": "2026-11-26 12:00"
  }
  ```
  *Accepts ISO formats `YYYY-MM-DDTHH:MM`, `YYYY-MM-DD HH:MM`, or `YYYY-MM-DD`.*
- **Description:** Triggers deep learning model inference (Hybrid CNN-LSTM) directly from the web dashboard, saves the result to `results/latest_prediction.json`, and returns the prediction payload for instant UI rendering at the top of the dashboard.
- **Example Response:**
  ```json
  {
    "status": "success",
    "prediction": {
      "type": "single_datetime",
      "command": "Dashboard Run: Predicted for 2026-11-21 16:00",
      "updated_at": "2026-11-21 15:49:38",
      "summary": {
        "title": "AQI for 2026-11-21 16:00:00 (FORECAST)",
        "dominant_category": "Very Poor",
        "dominant_code": 4,
        "avg_confidence": 98.8,
        "risk_level": "Very High",
        "advisory": "Minimize outdoor activities. Use N95 masks."
      }
    }
  }
  ```

### 2.7 Plot Visualizations Endpoint
- **Route:** `GET /results-plots/<filename>`
- **Description:** Serves high-resolution static PNG figures generated in `results/plots/` (e.g., confusion matrices, loss curves).

---

## 3. Internal Python Module API Reference

### 3.1 Data Acquisition (`src.data`)

#### `pollution_fetch.py`
- `fetch_pollution_data(lat, lon, start, end, pollutants, chunk_days=90, save_path=None) -> pd.DataFrame`
  - Fetches air quality data from Open-Meteo in batched chunks.

#### `weather_fetch.py`
- `fetch_weather_data(lat, lon, start, end, variables, chunk_days=90, save_path=None) -> pd.DataFrame`
  - Fetches historical meteorological data in batched chunks.

#### `forecast_fetch.py`
- `fetch_pollution_smart(lat, lon, start_date, end_date, pollutants) -> pd.DataFrame`
- `fetch_weather_smart(lat, lon, start_date, end_date, weather_vars) -> pd.DataFrame`
  - Dynamically routes requests to either the historical archive or forecast endpoints based on date horizons.

#### `data_merger.py`
- `merge_pollution_weather(pollution_df, weather_df, save_path=None) -> pd.DataFrame`
  - Merges pollution and weather DataFrames on their common UTC hourly datetime index.

#### `data_cleaning.py`
- `handle_missing_values(df, limit=6) -> pd.DataFrame`
  - Applies bidirectional linear interpolation.
- `cap_outliers(df, factor=3.0) -> pd.DataFrame`
  - Capping values via IQR winsorization and clamping pollutant values $\ge 0$.
- `validate_data(df) -> dict`
  - Performs data validation checks.

---

### 3.2 Preprocessing & Feature Engineering (`src.preprocessing`)

#### `aqi_calculator.py`
- `compute_sub_index(concentration, breakpoints) -> float`
  - Evaluates linear interpolation across Indian CPCB breakpoint ranges.
- `compute_aqi(row, pollutant_columns=None) -> float`
  - Computes the overall AQI as the maximum of sub-indices.
- `get_aqi_category(aqi_value) -> tuple[str, int]`
  - Maps numerical AQI to `(category_name, category_code)`.
- `calculate_aqi_dataframe(df, save_path=None) -> pd.DataFrame`
  - Enriches the input DataFrame with `aqi`, `aqi_category`, and `aqi_code` columns.

#### `feature_engineering.py`
- `create_lag_features(df, columns, lags=[1, 2, 3, 6, 12, 24]) -> pd.DataFrame`
- `create_rolling_features(df, columns, windows=[3, 6, 12, 24]) -> pd.DataFrame`
- `create_cyclic_features(df) -> pd.DataFrame`
- `create_ratio_features(df) -> pd.DataFrame`
- `engineer_all_features(df, config, save_path=None) -> tuple[pd.DataFrame, dict]`
  - Generates the full 82-feature dataset and returns a dictionary of feature column groups.

#### `scaler.py`
- `class AQIScaler`
  - `fit(df, feature_cols)`: Fits `StandardScaler` on training features.
  - `transform(df) -> pd.DataFrame`: Transforms DataFrame features.
  - `transform_array(array) -> np.ndarray`: Transforms NumPy arrays for inference.
  - `save(path)`: Serializes scaler using `joblib`.
  - `load(path)`: Deserializes scaler.

#### `sequence_generator.py`
- `create_sliding_windows(features_arr, target_arr, window_size=24) -> tuple[np.ndarray, np.ndarray]`
  - Generates 3D tensor sequences `(N, 24, 82)` and 1D target labels `(N,)`.
- `chronological_train_test_split(X, y, train_ratio=0.7) -> tuple`
  - Splits sequences chronologically without shuffling.
- `save_sequences(path, X_train, X_test, y_train, y_test)`: Saves `.npz` archive.
- `load_sequences(path) -> tuple`: Loads `.npz` archive.

---

### 3.3 Neural Network Models (`src.models`)

#### `cnn.py`
- `build_cnn_model(input_shape=(24, 82), num_classes=6, config=None) -> tf.keras.Model`
  - Builds the 1D CNN classifier (~238K parameters).

#### `lstm.py`
- `build_lstm_model(input_shape=(24, 82), num_classes=6, config=None) -> tf.keras.Model`
  - Builds the Stacked LSTM classifier (~166K parameters).

#### `cnn_lstm.py`
- `build_cnn_lstm_model(input_shape=(24, 82), num_classes=6, config=None) -> tf.keras.Model`
  - Builds the Hybrid CNN–LSTM classifier (~231K parameters).

---

### 3.4 Training Pipeline (`src.training`)

#### `callbacks.py`
- `get_callbacks(model_name, output_dir='models', config=None) -> list`
  - Returns `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.

#### `train_cnn.py`, `train_lstm.py`, `train_cnn_lstm.py`
- `train_cnn(X_train, y_train, X_val, y_val, config)`
- `train_lstm(X_train, y_train, X_val, y_val, config)`
- `train_cnn_lstm(X_train, y_train, X_val, y_val, config)`
  - Compiles with Adam optimizer and Sparse Categorical Crossentropy, trains, and saves best model weights.

---

### 3.5 Evaluation & Statistical Hypothesis Testing (`src.evaluation`)

#### `metrics.py`
- `calculate_all_metrics(y_true, y_pred) -> dict`
  - Computes Accuracy, Weighted Precision, Recall, F1, and Macro F1.

#### `confusion_matrix.py`
- `plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path)`
  - Generates and saves a confusion matrix heatmap.

#### `statistical_tests.py`
- `run_mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict`
  - Computes McNemar's $\chi^2$ with Edwards' continuity correction and returns $p$-value.

#### `ablation_analysis.py`
- `run_ablation_experiment(df, config, model_fn)`
  - Executes the 6-step incremental feature ablation study.

---

### 3.6 Prediction & Forecasting Systems (`src.prediction`)

#### `predict.py`
- `load_trained_model(model_path) -> tf.keras.Model`
- `predict_aqi(model, X) -> np.ndarray`
  - Runs batch model prediction and returns class integer labels.

#### `ensemble.py`
- `class EnsemblePredictor(models_dir='models', weights=None)`
  - `predict_probabilities(X_window) -> np.ndarray`: Returns weighted soft-voted probabilities.
  - `predict_single(X_window) -> dict`: Returns category, confidence, and model agreement score.
  - `predict_batch(X_batch) -> list[dict]`: Runs ensemble prediction over multiple samples.

#### `predict_any_datetime.py`
- `predict_for_datetime(target_datetime_str, model_path='models/cnn_lstm/best_model.keras') -> dict`
  - Executes live data acquisition, feature engineering, and Hybrid CNN-LSTM prediction for any target date/time.

#### `early_warning.py`
- `generate_alerts(predicted_categories) -> np.ndarray`
  - Binary alert generation (1 if predicted class $\in \{4, 5\}$, else 0).
- `evaluate_early_warning(y_true, y_pred) -> dict`
  - Evaluates alert precision, recall, F1-score, and false alarm metrics.
- `format_alert_message(aqi_code, aqi_value, timestamp) -> dict`
  - Returns public health advisory text and severity level.
