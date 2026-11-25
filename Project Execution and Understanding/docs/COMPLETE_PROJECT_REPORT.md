# Air Quality Index (AQI) Prediction using Hybrid CNN–LSTM Deep Learning Model
## An Extension with Meteorological Features, Feature Ablation, and an Early-Warning Module

---

**Institution:** G.L. Bajaj Institute of Technology and Management, Greater Noida  
**Department:** Department of Computer Science and Information Technology  
**Project Guide:** Prof. Animesh Malviya  
**Academic Session:** Academic Session: Final Year B.Tech Project  

### Project Team Members & Responsibilities

| Student Name | University Roll Number | Project Role & Technical Focus |
|:---|:---:|:---|
| **Sushant Prasad** | 2301920130192 | Project Lead & Hybrid CNN-LSTM Architecture |
| **Shivam Verma** | 2301920130163 | Stacked LSTM Architecture & Sequence Modeling |
| **Harshit Giri** | 2401920139005 | Full-Stack Web Dashboard & Exploratory Data Analysis |
| **Adarsh Raj** | 2401920139001 | 1D CNN Architecture & Meteorological Extension |

---

## Table of Contents

1. [Executive Summary & Abstract](#1-executive-summary--abstract)
2. [Complete Dataset Catalog & Technical Specifications](#2-complete-dataset-catalog--technical-specifications)
3. [End-to-End Pipeline & Mathematical Formulations](#3-end-to-end-pipeline--mathematical-formulations)
4. [Comprehensive Script-by-Script Breakdown (All 51 Files)](#4-comprehensive-script-by-script-breakdown-all-51-files)
5. [Detailed Walkthrough of All 13 Jupyter Notebooks](#5-detailed-walkthrough-of-all-13-jupyter-notebooks)
6. [Frontend Web Dashboard & RESTful Architecture](#6-frontend-web-dashboard--restful-architecture)
7. [Experimental Results, Benchmarks & Statistical Significance](#7-experimental-results-benchmarks--statistical-significance)
8. [Software Testing, Verification & Viva Preparation Guide](#8-software-testing-verification--viva-preparation-guide)

---

## 1. Executive Summary & Abstract

Air quality degradation in metropolitan environments presents an immediate and acute threat to human respiratory health, atmospheric balance, and urban productivity. In northern India, particularly the National Capital Region (NCR) of Delhi (28.6139°N, 77.2090°E), the interplay of meteorological stagnation, biomass burning, industrial emissions, and vehicular density results in severe seasonal air pollution crises. Traditional forecasting approaches—such as autoregressive integrated moving average (ARIMA) and basic feedforward networks—often fail to capture complex non-linear interactions across atmospheric pollutants and meteorological drivers.

This comprehensive project designs, implements, and benchmarks an end-to-end Hybrid Convolutional Neural Network - Long Short-Term Memory (CNN–LSTM) deep learning architecture for hourly Air Quality Index (AQI) category classification and multi-day forecasting. Built on 4 years (2022–2025, comprising 35,064 continuous hourly observations) of air pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3) and surface meteorological observations (temperature, relative humidity, wind speed, surface pressure, dew point) retrieved from the Open-Meteo APIs, the system adheres strictly to the official Indian Central Pollution Control Board (CPCB) National Air Quality Index (NAQI) sub-index formulation.

A specialized feature engineering pipeline derives 82 distinct temporal, lag, moving-average, cyclic, interaction-ratio, and meteorological features. Time series sequences are generated using a 24-hour sliding window, creating 29,833 continuous multi-dimensional sequence tensors split chronologically into 70% training (20,892 sequences) and 30% testing (8,941 sequences).

Across comprehensive testing, the proposed Hybrid CNN–LSTM model achieved 80.64% classification accuracy (Weighted F1: 0.8059), significantly outperforming baseline 1D CNN (77.82% accuracy, F1: 0.7755) and Stacked LSTM (75.01% accuracy, F1: 0.7392). Pairwise McNemar's statistical significance tests confirmed that the Hybrid model's superiority is statistically significant (p < 0.001).

The research provides three key academic extensions:
1. Meteorological Feature Impact Study: Proving that 1D CNN models gain +0.93% accuracy with meteorological integration, while recurrent models benefit from reduced pollutant-only feature dimensions.
2. Step-wise Feature Ablation Study: Benchmarking 6 feature subsets to demonstrate that rolling averages deliver the highest marginal gain (+1.5% F1) and establishing Step 3 (66 features) as the optimal feature subset.
3. Threshold-Based Early Warning Alert System: Attaining 93.67% alert precision, 87.51% alert recall, and 90.48% alert F1-score for high-risk hazardous conditions (Very Poor and Severe categories).

The system is fully operationalized with an On-Demand Arbitrary Datetime Predictor and a responsive Flask Web Application with dynamic asynchronous dashboard updates.

> **Core Research Innovation:** Combines 1D convolutional feature extraction (local pollutant spikes) with recurrent LSTM temporal modeling (24-hour chronological context), outperforming standalone models and delivering high-precision early warnings.

---

## 2. Complete Dataset Catalog & Technical Specifications

| Dataset / File Path | Category | Dimensions | File Size | Description |
|:---|:---|:---|:---|:---|
| `data/raw/pollution_data.csv` | Raw Ingestion | 35064 rows x 7 cols | 1.74 MB | Hourly atmospheric pollutant concentrations for Delhi (28.6139°N, 77.2090°E) retrieved from Open-Meteo Air Quality API in 90-day historical chunks. |
| `data/raw/weather_data.csv` | Raw Ingestion | 35064 rows x 6 cols | 1.54 MB | Hourly meteorological and atmospheric observations retrieved from Open-Meteo Historical Weather Archive API. |
| `data/processed/merged_data.csv` | Cleaned Preprocessed | 29905 rows x 12 cols | 2.29 MB | Inner join of pollution and weather datasets on datetime index, treated with linear interpolation (limit=6) and 3.0x IQR outlier winsorization. |
| `data/processed/aqi_data.csv` | CPCB Calculated | 29905 rows x 15 cols | 2.82 MB | Merged data enriched with Indian CPCB sub-indices, overall numerical AQI, categorical label, and numerical classification code (0 to 5). |
| `data/processed/featured_data.csv` | Engineered Features | 29881 rows x 86 cols | 26.23 MB | Comprehensive feature-engineered matrix containing 82 predictive features plus metadata and targets. Head dropped by 24 hours to accommodate rolling/lag windows. |
| `data/processed/sequences.npz` | Model Input Tensors | 20892 train, 8941 test | 9.82 MB | Serialized NumPy compressed archive holding 3D sliding-window input sequences and corresponding target integer arrays for model training and evaluation. |
| `artifacts/scaler.joblib` | Fitted Pipeline Artifact | 82 features | 4.97 KB | Serialized scikit-learn StandardScaler instance containing fitted means and variances for all 82 predictive features. Prevents test set data leakage. |
| `artifacts/feature_groups.json` | Configuration Artifact | 6 Feature Groups | 1.74 KB | JSON dictionary mapping each feature group name to its constituent column names. Used dynamically by run_ablation_study.py and run_meteorological_comparison.py. |
| `Project Execution and Understanding/scaler_readable.txt & .xlsx` | Inspection Document | 20 rows x 5 cols | 13.67 KB (txt), 12.65 KB (xlsx) | Human-readable plain text and Excel spreadsheet export of feature scaling parameters (means, standard deviations, variances). |
| `Project Execution and Understanding/X_train.csv & X_test.csv` | Tabular Export | 20892 (train), 8941 (test) | 452.86 MB (train), 193.84 MB (test) | 2D tabular representations of the 3D sequence tensors (24 timesteps * 82 features = 1,968 columns per row). Useful for traditional ML tabular algorithms (Random Forest, XGBoost). |
| `results/latest_prediction.json` | Live Operational Payload | Single JSON object | 1.57 KB | Runtime prediction state payload synchronized to the Flask dashboard on every prediction call from the CLI or web UI. |


### 2.1 Dataset: `data/raw/pollution_data.csv`

- **Category:** Raw Ingestion
- **Size on Disk:** 1.74 MB
- **Temporal Range:** 2022-01-01 00:00 to 2025-12-31 23:00
- **Description:** Hourly atmospheric pollutant concentrations for Delhi (28.6139°N, 77.2090°E) retrieved from Open-Meteo Air Quality API in 90-day historical chunks.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `datetime` | `datetime64[ns]` | ISO 8601 Timestamp | Hourly index timestamp in UTC/local timezone |
| `pm2_5` | `float64` | µg/m³ | Fine particulate matter with aerodynamic diameter <= 2.5 µm |
| `pm10` | `float64` | µg/m³ | Coarse particulate matter with aerodynamic diameter <= 10 µm |
| `no2` | `float64` | µg/m³ | Nitrogen Dioxide concentration |
| `so2` | `float64` | µg/m³ | Sulphur Dioxide concentration |
| `co` | `float64` | mg/m³ | Carbon Monoxide concentration |
| `o3` | `float64` | µg/m³ | Ground-level surface Ozone concentration |


### 2.2 Dataset: `data/raw/weather_data.csv`

- **Category:** Raw Ingestion
- **Size on Disk:** 1.54 MB
- **Temporal Range:** 2022-01-01 00:00 to 2025-12-31 23:00
- **Description:** Hourly meteorological and atmospheric observations retrieved from Open-Meteo Historical Weather Archive API.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `datetime` | `datetime64[ns]` | ISO 8601 Timestamp | Hourly index timestamp |
| `temperature` | `float64` | °C | Ambient 2-meter air temperature |
| `humidity` | `float64` | % | Relative atmospheric humidity percentage |
| `wind_speed` | `float64` | km/h | 10-meter surface wind speed |
| `pressure` | `float64` | hPa | Mean sea-level surface atmospheric pressure |
| `dew_point` | `float64` | °C | 2-meter dew point temperature |


### 2.3 Dataset: `data/processed/merged_data.csv`

- **Category:** Cleaned Preprocessed
- **Size on Disk:** 2.29 MB
- **Temporal Range:** 2022-01-01 to 2025-12-31
- **Description:** Inner join of pollution and weather datasets on datetime index, treated with linear interpolation (limit=6) and 3.0x IQR outlier winsorization.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `datetime` | `datetime64[ns]` | Timestamp | Hourly DatetimeIndex |
| `pm2_5` | `float64` | µg/m³ | Cleaned PM2.5 (negative values clipped, outliers winsorized) |
| `pm10` | `float64` | µg/m³ | Cleaned PM10 concentration |
| `no2` | `float64` | µg/m³ | Cleaned NO2 concentration |
| `so2` | `float64` | µg/m³ | Cleaned SO2 concentration |
| `co` | `float64` | mg/m³ | Cleaned CO concentration |
| `o3` | `float64` | µg/m³ | Cleaned O3 concentration |
| `temperature` | `float64` | °C | Cleaned ambient temperature |
| `humidity` | `float64` | % | Cleaned relative humidity |
| `wind_speed` | `float64` | km/h | Cleaned wind speed |
| `pressure` | `float64` | hPa | Cleaned surface pressure |
| `dew_point` | `float64` | °C | Cleaned dew point temperature |


### 2.4 Dataset: `data/processed/aqi_data.csv`

- **Category:** CPCB Calculated
- **Size on Disk:** 2.82 MB
- **Temporal Range:** 2022-01-01 to 2025-12-31
- **Description:** Merged data enriched with Indian CPCB sub-indices, overall numerical AQI, categorical label, and numerical classification code (0 to 5).

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `datetime` | `datetime64[ns]` | Timestamp | Hourly DatetimeIndex |
| `pm2_5 .. dew_point` | `float64` | Various | 11 base pollutant and meteorological variables |
| `aqi` | `float64` | Index (0-500) | Overall CPCB Air Quality Index (max sub-index across 6 pollutants) |
| `aqi_category` | `object` | String Label | Good, Satisfactory, Moderate, Poor, Very Poor, or Severe |
| `aqi_code` | `int64` | Integer (0-5) | Target classification label for neural network prediction |


### 2.5 Dataset: `data/processed/featured_data.csv`

- **Category:** Engineered Features
- **Size on Disk:** 26.23 MB
- **Temporal Range:** 2022-01-02 00:00 to 2025-12-31 23:00
- **Description:** Comprehensive feature-engineered matrix containing 82 predictive features plus metadata and targets. Head dropped by 24 hours to accommodate rolling/lag windows.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `datetime` | `datetime64[ns]` | Timestamp | Hourly DatetimeIndex |
| `Base Pollutants (6)` | `float64` | µg/m³, mg/m³ | pm2_5, pm10, no2, so2, co, o3 |
| `Lag Features (36)` | `float64` | Concentration | Lags at 1h, 2h, 3h, 6h, 12h, 24h for all 6 pollutants |
| `Rolling Means (24)` | `float64` | Concentration | Moving averages across 3h, 6h, 12h, 24h windows for all 6 pollutants |
| `Cyclic Time (8)` | `float64` | Trigonometric [-1, 1] | sin/cos encodings for hour, day of week, day of year, month |
| `Interaction Ratios (3)` | `float64` | Dimensionless | pm25_pm10_ratio, no2_so2_ratio, co_o3_ratio |
| `Meteorological (5)` | `float64` | Various | temperature, humidity, wind_speed, pressure, dew_point |
| `Targets (3)` | `Various` | Index / String / Int | aqi, aqi_category, aqi_code |


### 2.6 Dataset: `data/processed/sequences.npz`

- **Category:** Model Input Tensors
- **Size on Disk:** 9.82 MB
- **Temporal Range:** Chronological 70/30 Split
- **Description:** Serialized NumPy compressed archive holding 3D sliding-window input sequences and corresponding target integer arrays for model training and evaluation.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `X_train` | `ndarray float32` | Z-score scaled | Shape: (20892, 24, 82) - 20,892 training sequences of 24 hours by 82 features |
| `y_train` | `ndarray int32` | Classes 0-5 | Shape: (20892,) - Ground-truth AQI category code at time t+1 |
| `X_test` | `ndarray float32` | Z-score scaled | Shape: (8941, 24, 82) - 8,941 evaluation sequences |
| `y_test` | `ndarray int32` | Classes 0-5 | Shape: (8941,) - Evaluation target labels |


### 2.7 Dataset: `artifacts/scaler.joblib`

- **Category:** Fitted Pipeline Artifact
- **Size on Disk:** 4.97 KB
- **Temporal Range:** Fitted on Training Split (70%)
- **Description:** Serialized scikit-learn StandardScaler instance containing fitted means and variances for all 82 predictive features. Prevents test set data leakage.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `scaler` | `StandardScaler` | N/A | Fitted StandardScaler object with mean_ and scale_ vectors of length 82 |
| `feature_columns` | `list of str` | Names | Exact ordered list of the 82 feature column identifiers |


### 2.8 Dataset: `artifacts/feature_groups.json`

- **Category:** Configuration Artifact
- **Size on Disk:** 1.74 KB
- **Temporal Range:** N/A
- **Description:** JSON dictionary mapping each feature group name to its constituent column names. Used dynamically by run_ablation_study.py and run_meteorological_comparison.py.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `base_pollutants` | `list` | 6 features | ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3'] |
| `lag_features` | `list` | 36 features | 6 pollutants x 6 lag intervals |
| `rolling_features` | `list` | 24 features | 6 pollutants x 4 rolling windows |
| `cyclic_features` | `list` | 8 features | sin/cos time encodings |
| `ratio_features` | `list` | 3 features | Pollutant emission ratios |
| `weather_features` | `list` | 5 features | Surface meteorological variables |


### 2.9 Dataset: `Project Execution and Understanding/scaler_readable.txt & .xlsx`

- **Category:** Inspection Document
- **Size on Disk:** 13.67 KB (txt), 12.65 KB (xlsx)
- **Temporal Range:** Inspection Export
- **Description:** Human-readable plain text and Excel spreadsheet export of feature scaling parameters (means, standard deviations, variances).

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `Index` | `int` | 1 to 20 | Feature index |
| `Feature Name` | `str` | Identifier | Name of pollutant or weather variable |
| `Training Mean` | `float` | Original units | Mean value observed across training period |
| `Scaling Factor (Std Dev)` | `float` | Original units | Standard deviation used as normalization denominator |
| `Training Variance` | `float` | Units squared | Sample variance across training period |


### 2.10 Dataset: `Project Execution and Understanding/X_train.csv & X_test.csv`

- **Category:** Tabular Export
- **Size on Disk:** 452.86 MB (train), 193.84 MB (test)
- **Temporal Range:** Flattened Tabular Sequences
- **Description:** 2D tabular representations of the 3D sequence tensors (24 timesteps * 82 features = 1,968 columns per row). Useful for traditional ML tabular algorithms (Random Forest, XGBoost).

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `feat_t0_00 .. feat_t23_81` | `float32` | Scaled | 1,968 flattened input dimensions per sequence |


### 2.11 Dataset: `results/latest_prediction.json`

- **Category:** Live Operational Payload
- **Size on Disk:** 1.57 KB
- **Temporal Range:** Dynamically overwritten on prediction
- **Description:** Runtime prediction state payload synchronized to the Flask dashboard on every prediction call from the CLI or web UI.

| Column / Attribute | Data Type | Physical Unit | Description & Value Range |
|:---|:---|:---|:---|
| `timestamp` | `str` | ISO Timestamp | Target timestamp predicted |
| `aqi_category` | `str` | Category Name | Predicted AQI category |
| `aqi_code` | `int` | 0 to 5 | Numerical classification code |
| `confidence` | `float` | Percentage | Softmax model confidence score |
| `health_risk` | `dict` | Advisories | Risk level, precautionary advisories, mask recommendations |
| `class_probabilities` | `dict` | Probabilities | Individual probabilities across all 6 AQI classes |


---

## 3. End-to-End Pipeline & Mathematical Formulations

### 3.1 Data Cleaning & Imputation
1. **Time-Series Linear Interpolation:**
   $$y(t) = y_0 + \frac{t - t_0}{t_1 - t_0} (y_1 - y_0)$$
   Applied for consecutive missing intervals up to 6 hours.

2. **Outlier Winsorization via IQR:**
   $$IQR = Q_3 - Q_1$$
   $$\text{Lower Bound} = \max(0, Q_1 - 3.0 \times IQR), \quad \text{Upper Bound} = Q_3 + 3.0 \times IQR$$

### 3.2 Indian CPCB NAQI Sub-Index Calculation
$$I_p = \frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}} (C_p - B_{lo}) + I_{lo}$$
$$\text{Overall AQI} = \max(I_{\text{PM}_{2.5}}, I_{\text{PM}_{10}}, I_{\text{NO}_2}, I_{\text{SO}_2}, I_{\text{CO}}, I_{\text{O}_3})$$

### 3.3 Cyclic Trigonometric Time Encodings
$$\text{hour}_{\sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$

### 3.4 Feature Standardization
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Fitted strictly on the 70% chronological training split.

---

## 4. Comprehensive Script-by-Script Breakdown (All 51 Files)

### 4.1 Script: `run_full_pipeline.py`

- **File Path:** `run_full_pipeline.py`
- **Lines of Code:** 276
- **Architectural Group:** Root Runners
- **Summary:** Master automated pipeline that executes the entire workflow end-to-end: loads raw data, merges, cleans, computes CPCB AQI, engineers 82 features, scales data, creates 3D sliding-window sequences, trains all 3 deep learning models (CNN, LSTM, CNN-LSTM), benchmarks their performance, runs feature ablation, and evaluates the early warning module.

#### Functions Defined:
- **`main()`**
  - **Parameters:** `None`
  - **Returns:** `None`
  - **Algorithmic Logic:** Step 1: Loads config/config.yaml and reads raw CSVs (pollution_data.csv and weather_data.csv). Step 2: Merges datasets via merge_pollution_weather(), reports missing values via report_missing(), performs linear interpolation via handle_missing_values(limit=6), and applies IQR winsorization via remove_outliers_iqr(factor=3.0). Step 3: Computes Indian CPCB sub-indices via calculate_aqi_dataframe(). Step 4: Derives 82 engineered features via engineer_all_features() and saves artifacts/feature_groups.json. Step 5: Fits AQIScaler on the 70% chronological training split and normalizes the full dataset. Step 6: Generates 24-hour sliding-window sequence tensors via create_sequences() and saves them to data/processed/sequences.npz. Step 7: Trains 1D CNN, Stacked LSTM, and Hybrid CNN-LSTM models via train_cnn(), train_lstm(), and train_cnn_lstm(). Step 8: Generates model comparison tables and saves evaluation metrics to results/metrics/.


> **Key Architectural Role:** Enforces strict chronological sequence splits (70% train / 30% test) to prevent future data leakage into the training phase. Integrates progress logging across all 8 pipeline phases.

---

### 4.2 Script: `run_pipeline.py`

- **File Path:** `run_pipeline.py`
- **Lines of Code:** 162
- **Architectural Group:** Root Runners
- **Summary:** Data Collection and Preprocessing Pipeline Runner. Executes data ingestion from Open-Meteo APIs, cleaning, AQI calculation, feature engineering, and sequence creation without requiring TensorFlow or GPU hardware.

#### Functions Defined:
- **`main()`**
  - **Parameters:** `None`
  - **Returns:** `None`
  - **Algorithmic Logic:** Step 1: Calls fetch_pollution_data() to download historical air quality records for Delhi. Step 2: Calls fetch_weather_data() to download historical meteorological records. Step 3: Merges both datasets on timestamp index and applies data cleaning (interpolation and IQR filtering). Step 4: Calculates CPCB AQI sub-indices and categories. Step 5: Engineers all 82 predictive features and saves featured_data.csv. Step 6: Fits the scaler and exports 3D sequence tensors to data/processed/sequences.npz.


> **Key Architectural Role:** Dedicated lightweight pipeline runner intended for data engineers or team members focusing on data preprocessing who do not require deep learning framework dependencies.

---

### 4.3 Script: `run_training.py`

- **File Path:** `run_training.py`
- **Lines of Code:** 170
- **Architectural Group:** Root Runners
- **Summary:** Model Retraining Runner. Resumes directly from pre-computed sequences (sequences.npz) to train or fine-tune all 3 neural network models without re-fetching or re-processing data.

#### Functions Defined:
- **`main()`**
  - **Parameters:** `None`
  - **Returns:** `None`
  - **Algorithmic Logic:** Loads data/processed/sequences.npz. Verifies tensor shapes: X_train (20892, 24, 82), X_test (8941, 24, 82). Invokes train_cnn(), train_lstm(), and train_cnn_lstm() using hyperparameter configurations from config/config.yaml. Saves best checkpoint weights to models/cnn/, models/lstm/, and models/cnn_lstm/. Computes classification reports and saves evaluation metrics.


> **Key Architectural Role:** Decouples computationally heavy model training from data ingestion, enabling rapid experimentation with neural network hyperparameters, learning rates, and layer architectures.

---

### 4.4 Script: `predict_3days.py` (Retired / Merged into `predict_any_datetime.py`)

- **File Path:** `predict_3days.py`
- **Architectural Status:** Deprecated / Streamlined into single on-demand predictor `predict_any_datetime.py`.
- **Summary:** Previously provided CLI 72-hour rolling forecasts. Streamlined into `predict_any_datetime.py` to give users fine-grained, instantaneous predictions for any chosen date and time without unnecessary continuous 3-day multi-hour execution overhead.

---

### 4.5 Script: `predict_any_datetime.py`

- **File Path:** `predict_any_datetime.py`
- **Lines of Code:** 411
- **Architectural Group:** Root Runners
- **Summary:** On-Demand Arbitrary Datetime AQI Predictor. Accepts any user-specified date and time (past, present, or future), dynamically queries live Open-Meteo APIs (or falls back to local historical datasets), constructs a 24-hour lookback window, scales the feature vector, executes model inference, and syncs to the web dashboard.

#### Functions Defined:
- **`predict_for_datetime(target_dt, ...)`**
  - **Parameters:** `target_dt (datetime or str), model_type (str, default='cnn_lstm'), config_path (str), models_dir (str)`
  - **Returns:** `dict (prediction payload)`
  - **Algorithmic Logic:** Calculates required 24-hour lookback window (target_dt - 24h to target_dt). Attempts to fetch hourly air quality and weather data via forecast_fetch.py from Open-Meteo. If network or rate-limit errors occur, falls back to data/processed/featured_data.csv to locate nearest historical timestamp. Normalizes input tensor using artifacts/scaler.joblib. Loads trained model from models/{model_type}/best_model.keras. Computes softmax probability distribution across 6 classes. Extracts predicted class, confidence score, health risk level, and health advisories. Writes results to results/latest_prediction.json.
- **`predict_range(start_dt, end_dt, ...)`**
  - **Parameters:** `start_dt (str), end_dt (str), step_hours (int=1), ...`
  - **Returns:** `list of dicts`
  - **Algorithmic Logic:** Iterates through a multi-hour interval and runs sequential predictions for every timestep.


> **Key Architectural Role:** Powers the interactive 'Predict for Date & Time' feature on the Flask dashboard. Handles network timeouts gracefully using local dataset fallback.

---

### 4.6 Script: `run_ablation_study.py`

- **File Path:** `run_ablation_study.py`
- **Lines of Code:** 286
- **Architectural Group:** Root Runners
- **Summary:** Step-Wise Feature Ablation Study Runner (Research Extension 2). Evaluates the incremental contribution of each feature group across 6 sequential steps to identify the optimal feature subset.

#### Functions Defined:
- **`main()`**
  - **Parameters:** `None`
  - **Returns:** `None`
  - **Algorithmic Logic:** Loads featured_data.csv and artifacts/feature_groups.json. Defines 6 incremental feature steps: Step 1 (Base Pollutants, 6 feats), Step 2 (+ Lag features, 42 feats), Step 3 (+ Rolling averages, 66 feats), Step 4 (+ Cyclic time, 74 feats), Step 5 (+ Ratios, 77 feats), Step 6 (+ Meteorological, 82 feats). Iterates through all 6 steps: scales subset, creates 24-hour sequence tensors, trains Hybrid CNN-LSTM for each step, evaluates test set accuracy, precision, recall, and F1. Saves models to models/ablation_step_1..6/. Exports results/ablation/ablation_results.csv and renders comparative charts to results/ablation/ablation_chart.png.


> **Key Architectural Role:** Quantifies the exact marginal performance gain of every feature family. Proves that rolling window features provide the single largest performance jump (+1.5% F1).

---

### 4.7 Script: `run_meteorological_comparison.py`

- **File Path:** `run_meteorological_comparison.py`
- **Lines of Code:** 289
- **Architectural Group:** Root Runners
- **Summary:** Meteorological Feature Impact Study Runner (Research Extension 1). Compares performance of all 3 architectures (CNN, LSTM, CNN-LSTM) trained exclusively on pollutant features vs. trained on pollutant + weather features.

#### Functions Defined:
- **`train_and_evaluate(model_name, X_tr, y_tr, X_te, y_te, num_features, ...)`**
  - **Parameters:** `model_name (str), training/testing tensors, num_features (int)`
  - **Returns:** `dict (evaluation metrics)`
  - **Algorithmic Logic:** Constructs and trains specified architecture, evaluates test set performance, logs history, and saves weights.
- **`main()`**
  - **Parameters:** `None`
  - **Returns:** `None`
  - **Algorithmic Logic:** Extracts 77 pollutant-only features vs. 82 pollutant+weather features. Trains CNN, LSTM, and Hybrid models on both subsets (6 total models). Exports comparison metrics to results/meteorological/weather_comparison.csv and results/meteorological/weather_impact.csv. Generates bar chart in results/meteorological/weather_comparison_chart.png.


> **Key Architectural Role:** Reveals critical structural insights: 1D CNN benefits from meteorological context (+0.93% accuracy), whereas recurrent LSTM performs better with lower-dimensional pollutant-only signals.

---

### 4.8 Script: `run_visualizations.py`

- **File Path:** `run_visualizations.py`
- **Lines of Code:** 332
- **Architectural Group:** Root Runners
- **Summary:** Master Visualization, Statistical Testing, and Report Generator. Generates all academic plots, 6x6 confusion matrices, training history loss curves, and McNemar's hypothesis test contingency tables.

#### Functions Defined:
- **`plot_training_histories(metrics_dir, output_path)`**
  - **Parameters:** `metrics_dir (str), output_path (str)`
  - **Returns:** `None`
  - **Algorithmic Logic:** Loads cnn_history.json, lstm_history.json, hybrid_history.json. Plots training vs validation loss and accuracy curves side by side across 50 epochs.
- **`run_statistical_tests(predictions_dict, y_true)`**
  - **Parameters:** `predictions_dict (dict), y_true (array)`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Computes 2x2 contingency matrices for CNN vs Hybrid, LSTM vs Hybrid, and CNN vs LSTM. Computes McNemar's chi-squared statistic with continuity correction and p-values. Exports results/reports/mcnemar_tests.csv.
- **`generate_summary_report(...)`**
  - **Parameters:** `Various metric files`
  - **Returns:** `None`
  - **Algorithmic Logic:** Aggregates all evaluation results into a unified JSON summary at results/reports/full_results_summary.json.
- **`main()`**
  - **Parameters:** `None`
  - **Returns:** `None`
  - **Algorithmic Logic:** Orchestrates all visualization and statistical reporting functions.


> **Key Architectural Role:** Produces publication-grade matplotlib figures and rigorous statistical validation ($p < 0.001$) required for academic defense and research publication.

---

### 4.9 Script: `src/data/api_fetch.py`

- **File Path:** `src/data/api_fetch.py`
- **Lines of Code:** 58
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Unified high-level interface that orchestrates downloading pollution and weather data for Delhi and executing an initial join.

#### Functions Defined:
- **`fetch_all_data(config_path='config/config.yaml')`**
  - **Parameters:** `config_path (str)`
  - **Returns:** `tuple of (DataFrame, DataFrame, DataFrame)`
  - **Algorithmic Logic:** Reads coordinates and date intervals from config. Calls fetch_pollution_data() and fetch_weather_data(). Invokes merge_pollution_weather() and returns dataframes.


> **Key Architectural Role:** Facade pattern giving consumers a one-line function to acquire all raw datasets.

---

### 4.10 Script: `src/data/pollution_fetch.py`

- **File Path:** `src/data/pollution_fetch.py`
- **Lines of Code:** 155
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Low-level API client for Open-Meteo Air Quality Historical API. Handles pagination in 90-day chunks, rate limits, and schema normalization.

#### Functions Defined:
- **`fetch_pollution_chunk(latitude, longitude, start_date, end_date, pollutants)`**
  - **Parameters:** `latitude, longitude, start_date, end_date, pollutants (list)`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Executes HTTP GET request to Open-Meteo API with hourly pollutant parameters. Parses JSON response and creates DatetimeIndex DataFrame.
- **`fetch_pollution_data(latitude, longitude, start_date, end_date, pollutants, chunk_days=90, save_path=None)`**
  - **Parameters:** `latitude, longitude, start_date, end_date, pollutants, chunk_days, save_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Splits 4-year range into 90-day intervals, pauses 1s between calls, concatenates chunks, and writes to data/raw/pollution_data.csv.


> **Key Architectural Role:** Robust chunking algorithm guarantees zero data loss across 35,064 records.

---

### 4.11 Script: `src/data/weather_fetch.py`

- **File Path:** `src/data/weather_fetch.py`
- **Lines of Code:** 152
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Low-level API client for Open-Meteo Historical Weather Archive API. Fetches surface weather variables in 90-day intervals.

#### Functions Defined:
- **`fetch_weather_chunk(latitude, longitude, start_date, end_date, variables)`**
  - **Parameters:** `latitude, longitude, start_date, end_date, variables`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Queries Open-Meteo Weather Archive for temperature, humidity, wind speed, pressure, dew point.
- **`fetch_weather_data(latitude, longitude, start_date, end_date, variables, chunk_days=90, save_path=None)`**
  - **Parameters:** `latitude, longitude, start_date, end_date, variables, chunk_days, save_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Chunks 2022-2025 multi-year range, handles rate limits, merges chronologically, and saves to data/raw/weather_data.csv.


> **Key Architectural Role:** Synchronizes meteorological timestamps identically with air pollutant observations.

---

### 4.12 Script: `src/data/forecast_fetch.py`

- **File Path:** `src/data/forecast_fetch.py`
- **Lines of Code:** 360
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Dual-Mode Smart API Client. Intelligently routes queries between Historical Archive endpoints and Live Forecast endpoints based on whether requested dates fall in the past or future.

#### Functions Defined:
- **`fetch_pollution_smart(latitude, longitude, start_date, end_date, pollutants, ...)`**
  - **Parameters:** `latitude, longitude, start_date, end_date, pollutants, ...`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Routes past dates to historical archive and future dates (up to 7 days) to forecast endpoint. Merges into continuous DatetimeIndex.
- **`fetch_weather_smart(latitude, longitude, start_date, end_date, variables, ...)`**
  - **Parameters:** `latitude, longitude, start_date, end_date, variables, ...`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Routes historical dates to archive-api and future dates to api.open-meteo.com/v1/forecast.


> **Key Architectural Role:** Powers predict_any_datetime.py across both historical queries and future forecasts.

---

### 4.13 Script: `src/data/data_loader.py`

- **File Path:** `src/data/data_loader.py`
- **Lines of Code:** 79
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Centralized YAML configuration loader and standardized CSV data loader utility functions.

#### Functions Defined:
- **`load_config(config_path='config/config.yaml')`**
  - **Parameters:** `config_path (str)`
  - **Returns:** `dict`
  - **Algorithmic Logic:** Loads and parses YAML configuration with validation.
- **`load_csv(path, parse_dates=True, index_col='datetime')`**
  - **Parameters:** `path, parse_dates, index_col`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Generic CSV loader setting DatetimeIndex.
- **`load_pollution_data / load_weather_data / load_merged_data / load_aqi_data / load_featured_data`**
  - **Parameters:** `config_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Convenience loader functions for specific pipeline stages.


> **Key Architectural Role:** Eliminates hardcoded paths across the entire project.

---

### 4.14 Script: `src/data/data_cleaning.py`

- **File Path:** `src/data/data_cleaning.py`
- **Lines of Code:** 202
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Data Preprocessing and Quality Assurance Module. Handles missing value reporting, linear time-series interpolation, and statistical outlier treatment using Interquartile Range (IQR) winsorization.

#### Functions Defined:
- **`report_missing(df, label='Dataset')`**
  - **Parameters:** `df, label`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Computes null counts and percentages per column.
- **`handle_missing_values(df, method='interpolate', limit=6)`**
  - **Parameters:** `df, method, limit`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Linear time-series interpolation along DatetimeIndex for gaps <= 6h, followed by bfill/ffill for edge cases.
- **`remove_outliers_iqr(df, factor=3.0, exclude_cols=None)`**
  - **Parameters:** `df, factor, exclude_cols`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Computes Q1, Q3, and IQR. Sets bounds [max(0, Q1 - 3*IQR), Q3 + 3*IQR] and clamps extreme values (winsorization) to preserve temporal continuity.
- **`validate_data_quality(df, label='Dataset')`**
  - **Parameters:** `df, label`
  - **Returns:** `bool`
  - **Algorithmic Logic:** Validates non-emptiness, monotonic indexing, and physical ranges (no negative pollutants).


> **Key Architectural Role:** 3.0x IQR factor preserves authentic high-pollution winter smog episodes while eliminating sensor malfunctions.

---

### 4.15 Script: `src/data/data_merger.py`

- **File Path:** `src/data/data_merger.py`
- **Lines of Code:** 65
- **Architectural Group:** Data Ingestion (src/data/)
- **Summary:** Temporal Synchronization and Join Module. Executes an exact inner join between pollution and weather DataFrames on DatetimeIndex.

#### Functions Defined:
- **`merge_pollution_weather(pollution_df, weather_df, save_path=None)`**
  - **Parameters:** `pollution_df, weather_df, save_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Executes inner join on DatetimeIndex, removes duplicate timestamps, and saves data/processed/merged_data.csv.


> **Key Architectural Role:** Guarantees exact 1-to-1 temporal synchronization between air pollution and atmospheric conditions.

---

### 4.16 Script: `src/preprocessing/aqi_calculator.py`

- **File Path:** `src/preprocessing/aqi_calculator.py`
- **Lines of Code:** 235
- **Architectural Group:** Preprocessing (src/preprocessing/)
- **Summary:** Indian Central Pollution Control Board (CPCB) National Air Quality Index Calculator. Implements piecewise linear sub-index interpolation across all 6 criteria pollutants and classifies overall AQI into 6 standard categories.

#### Functions Defined:
- **`compute_sub_index(concentration, pollutant)`**
  - **Parameters:** `concentration (float), pollutant (str)`
  - **Returns:** `float`
  - **Algorithmic Logic:** Looks up CPCB breakpoints [B_lo, B_hi] and index range [I_lo, I_hi]. Calculates Ip = ((I_hi - I_lo) / (B_hi - B_lo)) * (Cp - B_lo) + I_lo. Capped at 500.
- **`compute_aqi(row)`**
  - **Parameters:** `row (Series or dict)`
  - **Returns:** `float`
  - **Algorithmic Logic:** Computes sub-indices for all pollutants. Checks CPCB minimum criteria (>=3 pollutants including PM2.5 or PM10). Returns max(sub_indices).
- **`get_aqi_category(aqi_value)`**
  - **Parameters:** `aqi_value (float)`
  - **Returns:** `tuple of (str, int)`
  - **Algorithmic Logic:** Maps AQI: 0-50 ('Good', 0), 51-100 ('Satisfactory', 1), 101-200 ('Moderate', 2), 201-300 ('Poor', 3), 301-400 ('Very Poor', 4), 401-500+ ('Severe', 5).
- **`calculate_aqi_dataframe(df, save_path=None)`**
  - **Parameters:** `df, save_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Calculates sub-indices and appends 'aqi', 'aqi_category', and 'aqi_code'. Saves to data/processed/aqi_data.csv.


> **Key Architectural Role:** Strict compliance with Government of India NAQI regulatory formulation.

---

### 4.17 Script: `src/preprocessing/feature_engineering.py`

- **File Path:** `src/preprocessing/feature_engineering.py`
- **Lines of Code:** 247
- **Architectural Group:** Preprocessing (src/preprocessing/)
- **Summary:** 82-Dimensional Multi-Scale Feature Engineering Module. Constructs lag temporal structures, rolling moving averages, cyclic trigonometric encodings, and emission interaction ratios.

#### Functions Defined:
- **`create_lag_features(df, pollutants, lags=[1, 2, 3, 6, 12, 24])`**
  - **Parameters:** `df, pollutants, lags`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Derives 36 lag columns using df[pollutant].shift(k).
- **`create_rolling_features(df, pollutants, windows=[3, 6, 12, 24])`**
  - **Parameters:** `df, pollutants, windows`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Derives 24 rolling mean features across 3h to 24h horizons.
- **`create_cyclic_features(df)`**
  - **Parameters:** `df`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Encodes sine and cosine for hour (24), weekday (7), day of year (365.25), month (12). 8 columns.
- **`create_ratio_features(df)`**
  - **Parameters:** `df`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Computes pm25_pm10_ratio, no2_so2_ratio, and co_o3_ratio (3 columns).
- **`engineer_all_features(df, config, save_path=None)`**
  - **Parameters:** `df, config, save_path`
  - **Returns:** `tuple of (DataFrame, dict)`
  - **Algorithmic Logic:** Extracts all 82 features, drops first 24 rows, saves data/processed/featured_data.csv, and exports feature group mappings.


> **Key Architectural Role:** Transforms raw pollution readings into an 82-dimensional feature space capturing short-term spikes, long-term trends, circadian cycles, and chemical origin ratios.

---

### 4.18 Script: `src/preprocessing/scaler.py`

- **File Path:** `src/preprocessing/scaler.py`
- **Lines of Code:** 155
- **Architectural Group:** Preprocessing (src/preprocessing/)
- **Summary:** Feature Normalization and Artifact Serialization Class. Encapsulates scikit-learn StandardScaler with training-only fitting to guarantee zero test-set data leakage.

#### Classes Defined:
- **Class `AQIScaler`**
  - `fit(df, feature_cols) -> computes mean_ and scale_ on training split only`
  - `transform(df) -> applies z = (x - mean) / scale`
  - `fit_transform(df, feature_cols)`
  - `save(path) -> serializes via joblib`
  - `load(path) -> deserializes scaler and verifies columns`


> **Key Architectural Role:** Strictly prevents data leakage by standardizing test sequences using training statistics.

---

### 4.19 Script: `src/preprocessing/sequence_generator.py`

- **File Path:** `src/preprocessing/sequence_generator.py`
- **Lines of Code:** 130
- **Architectural Group:** Preprocessing (src/preprocessing/)
- **Summary:** 3D Sliding-Window Sequence Tensor Generator. Transforms 2D scaled feature matrices into 3D tensors (samples, timesteps=24, features=82) for sequential deep learning models.

#### Functions Defined:
- **`create_sequences(df, feature_cols, target_col='aqi_code', sequence_length=24)`**
  - **Parameters:** `df, feature_cols, target_col, sequence_length`
  - **Returns:** `tuple of (X, y)`
  - **Algorithmic Logic:** Extracts sliding windows of shape (24, 82) and corresponding target label y at index i. Returns X shape (N, 24, 82), y shape (N,).
- **`split_sequences(X, y, train_ratio=0.70)`**
  - **Parameters:** `X, y, train_ratio`
  - **Returns:** `tuple of (X_train, X_test, y_train, y_test)`
  - **Algorithmic Logic:** Chronological temporal split at 70%. No random shuffling.
- **`save_sequences / load_sequences`**
  - **Parameters:** `path, tensors`
  - **Returns:** `tensors`
  - **Algorithmic Logic:** Saves and loads compressed data/processed/sequences.npz.


> **Key Architectural Role:** Preserves natural time causality: each 24-hour observation predicts the 25th hour.

---

### 4.20 Script: `src/models/cnn.py`

- **File Path:** `src/models/cnn.py`
- **Lines of Code:** 102
- **Architectural Group:** Model Architectures (src/models/)
- **Summary:** 1D Convolutional Neural Network (1D-CNN) Architecture Builder. Designed for temporal motif and local pattern extraction across sliding time windows.

#### Functions Defined:
- **`build_cnn_model(input_shape=(24, 82), num_classes=6, ...)`**
  - **Parameters:** `input_shape, num_classes, filters=[64, 128], kernel_sizes=[3, 3], dropout_rate=0.3, learning_rate=0.001`
  - **Returns:** `keras.Model`
  - **Algorithmic Logic:** Input(24, 82) -> Conv1D(64, 3, same) -> BatchNorm -> ReLU -> Conv1D(128, 3, same) -> BatchNorm -> ReLU -> MaxPool1D(2) -> Flatten -> Dense(128, relu) -> Dropout(0.3) -> Dense(6, softmax). Trainable params: 224,966.


> **Key Architectural Role:** Extracts high-frequency local changes and sudden pollutant concentration jumps across the 24-hour sequence.

---

### 4.21 Script: `src/models/lstm.py`

- **File Path:** `src/models/lstm.py`
- **Lines of Code:** 93
- **Architectural Group:** Model Architectures (src/models/)
- **Summary:** Stacked Long Short-Term Memory (LSTM) Recurrent Architecture Builder. Specializes in modeling sequential dynamics and long-range dependencies.

#### Functions Defined:
- **`build_lstm_model(input_shape=(24, 82), num_classes=6, units=[128, 64], dropout_rate=0.2, learning_rate=0.001)`**
  - **Parameters:** `input_shape, num_classes, units, dropout_rate, learning_rate`
  - **Returns:** `keras.Model`
  - **Algorithmic Logic:** Input(24, 82) -> LSTM(128, return_sequences=True) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(64, relu) -> Dense(6, softmax). Trainable params: 129,670.


> **Key Architectural Role:** Internal gating mechanisms maintain memory across the full 24-hour lookback window.

---

### 4.22 Script: `src/models/cnn_lstm.py`

- **File Path:** `src/models/cnn_lstm.py`
- **Lines of Code:** 132
- **Architectural Group:** Model Architectures (src/models/)
- **Summary:** Primary Proposed Hybrid CNN–LSTM Architecture Builder. Cascades 1D convolutional feature extraction directly into recurrent LSTM sequence modeling.

#### Functions Defined:
- **`build_cnn_lstm_model(input_shape=(24, 82), num_classes=6, cnn_filters=64, cnn_kernel_size=3, lstm_units=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001)`**
  - **Parameters:** `input_shape, num_classes, ...`
  - **Returns:** `keras.Model`
  - **Algorithmic Logic:** Input(24, 82) -> Conv1D(64, 3, same) -> BatchNorm -> ReLU -> MaxPool1D(2) -> LSTM(64) -> Dropout(0.3) -> Dense(64, relu) -> Dense(6, softmax). Trainable params: 217,542.


> **Key Architectural Role:** Conv1D condenses input features into spatial patterns, passed to LSTM for temporal transition modeling (80.64% accuracy).

---

### 4.23 Script: `src/training/callbacks.py`

- **File Path:** `src/training/callbacks.py`
- **Lines of Code:** 84
- **Architectural Group:** Training Pipelines (src/training/)
- **Summary:** Training Optimization and Callbacks Builder. Configures EarlyStopping, dynamic Learning Rate reduction on plateau, and ModelCheckpoint saving.

#### Functions Defined:
- **`get_callbacks(checkpoint_path, monitor='val_loss', patience=10, reduce_lr_patience=5, reduce_lr_factor=0.5, min_lr=1e-6)`**
  - **Parameters:** `checkpoint_path, monitor, patience, reduce_lr_patience, reduce_lr_factor, min_lr`
  - **Returns:** `list of Callbacks`
  - **Algorithmic Logic:** Returns ModelCheckpoint(save_best_only=True), EarlyStopping(restore_best_weights=True), and ReduceLROnPlateau.


> **Key Architectural Role:** Guarantees that saved checkpoints represent the optimal validation snapshot rather than an overfitted late-epoch model.

---

### 4.24 Script: `src/training/train_cnn.py`

- **File Path:** `src/training/train_cnn.py`
- **Lines of Code:** 80
- **Architectural Group:** Training Pipelines (src/training/)
- **Summary:** 1D-CNN Training Pipeline. Compiles model, connects callbacks, fits on training sequences, and logs training history.

#### Functions Defined:
- **`train_cnn(X_train, y_train, X_test, y_test, config, model_config, checkpoint_dir='models/cnn')`**
  - **Parameters:** `tensors, configs, checkpoint_dir`
  - **Returns:** `tuple of (model, history)`
  - **Algorithmic Logic:** Builds CNN, connects callbacks to save best_model.keras, trains for 50 epochs (batch_size=64), exports results/metrics/cnn_history.json.


> **Key Architectural Role:** Encapsulates end-to-end training of the 1D-CNN baseline model.

---

### 4.25 Script: `src/training/train_lstm.py`

- **File Path:** `src/training/train_lstm.py`
- **Lines of Code:** 80
- **Architectural Group:** Training Pipelines (src/training/)
- **Summary:** Stacked LSTM Training Pipeline. Compiles model, fits on training sequences with early stopping, and serializes training curves.

#### Functions Defined:
- **`train_lstm(X_train, y_train, X_test, y_test, config, model_config, checkpoint_dir='models/lstm')`**
  - **Parameters:** `tensors, configs, checkpoint_dir`
  - **Returns:** `tuple of (model, history)`
  - **Algorithmic Logic:** Builds Stacked LSTM, trains with batch size 64 for 50 epochs, saves best_model.keras and results/metrics/lstm_history.json.


> **Key Architectural Role:** Encapsulates training for the recurrent LSTM baseline.

---

### 4.26 Script: `src/training/train_cnn_lstm.py`

- **File Path:** `src/training/train_cnn_lstm.py`
- **Lines of Code:** 80
- **Architectural Group:** Training Pipelines (src/training/)
- **Summary:** Hybrid CNN–LSTM Training Pipeline. Trains the primary proposed model, records convergence metrics, and saves the production checkpoint.

#### Functions Defined:
- **`train_cnn_lstm(X_train, y_train, X_test, y_test, config, model_config, checkpoint_dir='models/cnn_lstm')`**
  - **Parameters:** `tensors, configs, checkpoint_dir`
  - **Returns:** `tuple of (model, history)`
  - **Algorithmic Logic:** Builds hybrid model, attaches callbacks to save models/cnn_lstm/best_model.keras, trains for 50 epochs, saves hybrid_history.json.


> **Key Architectural Role:** Produces the primary production weights used across all inference tasks and the Flask web dashboard.

---

### 4.27 Script: `src/evaluation/metrics.py`

- **File Path:** `src/evaluation/metrics.py`
- **Lines of Code:** 126
- **Architectural Group:** Model Evaluation (src/evaluation/)
- **Summary:** Multi-Class Classification Metrics Engine. Computes Accuracy, Weighted/Macro Precision, Recall, F1-Score, and Cohen's Kappa.

#### Functions Defined:
- **`compute_metrics(y_true, y_pred, labels=None)`**
  - **Parameters:** `y_true, y_pred, labels`
  - **Returns:** `dict`
  - **Algorithmic Logic:** Computes accuracy, precision_weighted, recall_weighted, f1_weighted, f1_macro, kappa via scikit-learn.
- **`get_classification_report(y_true, y_pred, target_names=None)`**
  - **Parameters:** `y_true, y_pred, target_names`
  - **Returns:** `str and dict`
  - **Algorithmic Logic:** Per-class precision, recall, f1-score, support for all 6 AQI categories.
- **`print_model_metrics(model_name, metrics_dict)`**
  - **Parameters:** `model_name, metrics_dict`
  - **Returns:** `None`
  - **Algorithmic Logic:** Prints formatted tabular summary to console.


> **Key Architectural Role:** Standardized metric evaluation across all 3 models ensuring fair, rigorous comparisons.

---

### 4.28 Script: `src/evaluation/confusion_matrix.py`

- **File Path:** `src/evaluation/confusion_matrix.py`
- **Lines of Code:** 142
- **Architectural Group:** Model Evaluation (src/evaluation/)
- **Summary:** Confusion Matrix Computation and Visualization Module. Generates 6x6 normalized contingency heatmaps for error pattern analysis.

#### Functions Defined:
- **`plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None, normalize='true')`**
  - **Parameters:** `y_true, y_pred, class_names, title, save_path, normalize`
  - **Returns:** `Figure`
  - **Algorithmic Logic:** Computes normalized confusion matrix, plots seaborn heatmap with annotations, saves PNG to results/plots/.
- **`plot_side_by_side_cm(cms_dict, class_names, save_path=None)`**
  - **Parameters:** `cms_dict, class_names, save_path`
  - **Returns:** `Figure`
  - **Algorithmic Logic:** Renders CNN, LSTM, and Hybrid confusion matrices in a 1x3 comparative grid.


> **Key Architectural Role:** Visually demonstrates that the Hybrid model dramatically reduces off-diagonal misclassifications between adjacent categories.

---

### 4.29 Script: `src/evaluation/model_comparison.py`

- **File Path:** `src/evaluation/model_comparison.py`
- **Lines of Code:** 145
- **Architectural Group:** Model Evaluation (src/evaluation/)
- **Summary:** Model Comparison Engine. Aggregates metrics from all 3 architectures into summary tables and generates comparative bar charts.

#### Functions Defined:
- **`compare_models(models_dict, X_test, y_test, save_path=None)`**
  - **Parameters:** `models_dict, X_test, y_test, save_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Evaluates all models, compiles comparison DataFrame, and saves results/metrics/model_comparison.csv.
- **`plot_model_comparison(comparison_df, save_path=None)`**
  - **Parameters:** `comparison_df, save_path`
  - **Returns:** `Figure`
  - **Algorithmic Logic:** Plots grouped bar chart comparing Accuracy and F1 across CNN, LSTM, and Hybrid.
- **`compare_pollutant_vs_weather(results_dict, save_path=None)`**
  - **Parameters:** `results_dict, save_path`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Compares performance deltas for models trained with vs without weather features.


> **Key Architectural Role:** Produces the master comparison tables cited in academic documentation.

---

### 4.30 Script: `src/evaluation/ablation_analysis.py`

- **File Path:** `src/evaluation/ablation_analysis.py`
- **Lines of Code:** 205
- **Architectural Group:** Model Evaluation (src/evaluation/)
- **Summary:** Feature Ablation Analysis Engine. Implements the 6-step incremental feature evaluation algorithm.

#### Functions Defined:
- **`run_ablation_study(df, feature_groups, model_builder_fn, ...)`**
  - **Parameters:** `df, feature_groups, model_builder_fn, ...`
  - **Returns:** `DataFrame`
  - **Algorithmic Logic:** Iterates through 6 feature steps, trains test models, evaluates metrics, records accuracy and F1 improvements.
- **`plot_ablation_results(ablation_df, save_path=None)`**
  - **Parameters:** `ablation_df, save_path`
  - **Returns:** `Figure`
  - **Algorithmic Logic:** Generates dual-axis plot showing Feature Count vs F1-Score and delta gains. Saves to results/ablation/feature_count_vs_performance.png.


> **Key Architectural Role:** Core implementation of Research Extension 2.

---

### 4.31 Script: `src/evaluation/statistical_tests.py`

- **File Path:** `src/evaluation/statistical_tests.py`
- **Lines of Code:** 69
- **Architectural Group:** Model Evaluation (src/evaluation/)
- **Summary:** McNemar's Non-Parametric Hypothesis Test Module. Evaluates whether performance differences between paired classifiers are statistically significant.

#### Functions Defined:
- **`mcnemar_test(y_true, y_pred_a, y_pred_b, model_a_name='Model A', model_b_name='Model B', continuity_correction=True)`**
  - **Parameters:** `y_true, y_pred_a, y_pred_b, model_a_name, model_b_name, continuity_correction`
  - **Returns:** `dict`
  - **Algorithmic Logic:** Builds 2x2 contingency table (b = A correct & B wrong, c = B correct & A wrong). Calculates chi2 = (|b - c| - 1)^2 / (b + c) with p-value.


> **Key Architectural Role:** Statistically proves that the Hybrid model's superiority is not due to random variation (p < 0.001).

---

### 4.32 Script: `src/prediction/predict.py`

- **File Path:** `src/prediction/predict.py`
- **Lines of Code:** 102
- **Architectural Group:** Prediction & Alerts (src/prediction/)
- **Summary:** Core Model Inference Engine. Loads trained checkpoints and executes single-sample or batch forward passes.

#### Functions Defined:
- **`load_trained_model(model_path)`**
  - **Parameters:** `model_path (str)`
  - **Returns:** `keras.Model`
  - **Algorithmic Logic:** Loads Keras model with error handling.
- **`predict_aqi(model, X_input)`**
  - **Parameters:** `model, X_input (N, 24, 82)`
  - **Returns:** `tuple of (codes, categories, probabilities)`
  - **Algorithmic Logic:** Runs forward pass, computes argmax class codes, maps to category names, and returns softmax probability array.
- **`get_best_model_path(models_dir='models', model_type='cnn_lstm')`**
  - **Parameters:** `models_dir, model_type`
  - **Returns:** `str`
  - **Algorithmic Logic:** Resolves path to models/{model_type}/best_model.keras.


> **Key Architectural Role:** Standardized inference entry point used by both CLI predictors and Flask API endpoints.

---

### 4.33 Script: `src/prediction/forecast_3day.py` (Retired / Merged)

- **File Path:** `src/prediction/forecast_3day.py`
- **Architectural Status:** Deprecated / Streamlined into `predict_any_datetime.py`.
- **Summary:** Previously managed 72-hour multi-model ensemble rolling forecasts. Consolidated into `predict_any_datetime.py` to support on-demand single datetime and range predictions directly.

---

### 4.34 Script: `src/prediction/early_warning.py`

- **File Path:** `src/prediction/early_warning.py`
- **Lines of Code:** 168
- **Architectural Group:** Prediction & Alerts (src/prediction/)
- **Summary:** Threshold-Based Early Warning Alert Engine (Research Extension 3). Evaluates predicted AQI categories and generates public health advisories for high-risk conditions.

#### Functions Defined:
- **`generate_alerts(predictions, high_risk_categories=['Very Poor', 'Severe'], confidence_threshold=0.60)`**
  - **Parameters:** `predictions, high_risk_categories, confidence_threshold`
  - **Returns:** `list of dicts`
  - **Algorithmic Logic:** Flags high-risk hours where predicted category is Very Poor or Severe with confidence >= 0.60. Attaches urgency levels and advisories.
- **`evaluate_early_warning(y_true, y_pred, high_risk_classes=[4, 5])`**
  - **Parameters:** `y_true, y_pred, high_risk_classes`
  - **Returns:** `dict`
  - **Algorithmic Logic:** Converts predictions to binary alerts (classes 4, 5). Calculates true positives, false positives, false negatives, precision (93.67%), recall (87.51%), and F1 (90.48%).
- **`format_alert_message(alert)`**
  - **Parameters:** `alert`
  - **Returns:** `str`
  - **Algorithmic Logic:** Formats human-readable warning notice with protective guidance.


> **Key Architectural Role:** Delivers actionable public health warnings with 93.67% precision, minimizing false alarms while protecting sensitive populations.

---

### 4.35 Script: `src/prediction/health_risk.py`

- **File Path:** `src/prediction/health_risk.py`
- **Lines of Code:** 141
- **Architectural Group:** Prediction & Alerts (src/prediction/)
- **Summary:** CPCB Health Advisory and Risk Level Mapping Module. Maps AQI numerical values and categories to clinical and public health guidance.

#### Functions Defined:
- **`map_health_risk(category_or_code)`**
  - **Parameters:** `category_or_code`
  - **Returns:** `dict`
  - **Algorithmic Logic:** Maps AQI category to clinical risk level, sensitive groups, outdoor exercise guidance, and mask requirements (N95 for Very Poor and Severe).
- **`add_health_risk_columns / get_risk_summary`**
  - **Parameters:** `df / predictions`
  - **Returns:** `DataFrame / dict`
  - **Algorithmic Logic:** Attaches structured risk metadata across predictions.


> **Key Architectural Role:** Translates technical deep learning classification outputs into understandable, actionable health recommendations.

---

### 4.36 Script: `src/prediction/ensemble.py`

- **File Path:** `src/prediction/ensemble.py`
- **Lines of Code:** 203
- **Architectural Group:** Prediction & Alerts (src/prediction/)
- **Summary:** Multi-Model Weighted Soft-Voting Ensemble Predictor Class. Combines probability distributions from CNN, LSTM, and Hybrid models to maximize prediction stability.

#### Classes Defined:
- **Class `EnsemblePredictor`**
  - `__init__(models_dict, weights={'cnn_lstm': 0.55, 'lstm': 0.25, 'cnn': 0.20})`
  - `predict_proba(X_input) -> computes weighted average softmax: sum(w_i * P_i)`
  - `predict(X_input) -> argmax(P_ensemble)`
  - `get_model_agreement(X_input) -> proportion of individual models agreeing on predicted class`


> **Key Architectural Role:** Leverages complementary strengths of convolutional and recurrent networks to produce superior, variance-reduced predictions.

---

### 4.37 Script: `src/prediction/latest_result.py`

- **File Path:** `src/prediction/latest_result.py`
- **Lines of Code:** 163
- **Architectural Group:** Prediction & Alerts (src/prediction/)
- **Summary:** Dynamic Dashboard Synchronization Bridge. Serializes latest prediction outcomes to results/latest_prediction.json for asynchronous web rendering.

#### Functions Defined:
- **`save_latest_prediction(prediction_dict, pred_type='single', command_str='')`**
  - **Parameters:** `prediction_dict, pred_type, command_str`
  - **Returns:** `str`
  - **Algorithmic Logic:** Sanitizes data, attaches system timestamp and command metadata, atomically writes results/latest_prediction.json.
- **`load_latest_prediction(file_path='results/latest_prediction.json')`**
  - **Parameters:** `file_path`
  - **Returns:** `dict or None`
  - **Algorithmic Logic:** Loads and deserializes active prediction state.


> **Key Architectural Role:** Decouples background prediction processes from the Flask web server, enabling immediate dynamic dashboard updates.

---

### 4.38 Script: `app/app.py`

- **File Path:** `app/app.py`
- **Lines of Code:** 204
- **Architectural Group:** Web Application (app/)
- **Summary:** Flask Application Server and RESTful API Endpoints. Serves the interactive web dashboard, exposes JSON prediction endpoints, and triggers on-demand background inference.

#### Functions Defined:
- **`index() -> Route: '/'`**
  - **Parameters:** `None`
  - **Returns:** `HTML`
  - **Algorithmic Logic:** Renders index.html with preloaded project overview metadata.
- **`api_latest_prediction() -> Route: '/api/latest-prediction'`**
  - **Parameters:** `None`
  - **Returns:** `JSON`
  - **Algorithmic Logic:** Reads results/latest_prediction.json and returns active prediction state to frontend polling loop.
- **`api_run_prediction() -> Route: '/api/run-prediction' [POST]`**
  - **Parameters:** `datetime (str, target timestamp)`
  - **Returns:** `JSON`
  - **Algorithmic Logic:** Triggers predict_for_datetime(datetime), updates latest_prediction.json, returns prediction payload.
- **`api_predict_datetime() -> Route: '/api/predict-datetime' [POST]`**
  - **Parameters:** `datetime (str, e.g. '2026-11-20 15:00')`
  - **Returns:** `JSON`
  - **Algorithmic Logic:** Calls predict_for_datetime() from predict_any_datetime.py, syncs result, returns prediction.
- **`api_results / api_aqi_categories / api_project_info`**
  - **Parameters:** `None`
  - **Returns:** `JSON`
  - **Algorithmic Logic:** Exposes model benchmarks, CPCB category reference tables, and project metadata.
- **`serve_plot(filename) -> Route: '/plots/<filename>'`**
  - **Parameters:** `filename`
  - **Returns:** `File`
  - **Algorithmic Logic:** Serves saved static evaluation figures from results/plots/.


> **Key Architectural Role:** Central server coordinating user interaction in the browser with backend deep learning prediction pipelines.

---

### 4.39 Script: `tests/test_aqi.py`

- **File Path:** `tests/test_aqi.py`
- **Lines of Code:** 96
- **Architectural Group:** Software Tests (tests/)
- **Summary:** Unit Test Suite for CPCB AQI Calculation Engine. Verifies breakpoint interpolation, category classification, and edge-case clamping.

#### Classes Defined:
- **Class `TestSubIndex`**
  - `test_pm25_sub_index`
  - `test_pm10_sub_index`
  - `test_zero_concentration`
  - `test_extreme_concentration_capping`
- **Class `TestAQICategory`**
  - `test_category_boundaries`
  - `test_category_names_and_codes`
- **Class `TestComputeAQI`**
  - `test_max_sub_index_rule`
  - `test_minimum_pollutants_requirement`


> **Key Architectural Role:** Verifies that sub-index equations match official CPCB tables with mathematical precision.

---

### 4.40 Script: `tests/test_models.py`

- **File Path:** `tests/test_models.py`
- **Lines of Code:** 60
- **Architectural Group:** Software Tests (tests/)
- **Summary:** Unit Test Suite for Neural Network Architectures. Verifies layer builds, tensor input/output shapes, and compilation parameter configurations.

#### Classes Defined:
- **Class `TestCNNModel`**
  - `test_cnn_build_and_output_shape`
  - `test_cnn_parameter_count`
- **Class `TestLSTMModel`**
  - `test_lstm_build_and_output_shape`
- **Class `TestCNNLSTMModel`**
  - `test_hybrid_build_and_output_shape`
  - `test_hybrid_forward_pass`


> **Key Architectural Role:** Guarantees that all 3 deep learning models produce valid (batch_size, 6) output probability tensors.

---

### 4.41 Script: `tests/test_prediction.py`

- **File Path:** `tests/test_prediction.py`
- **Lines of Code:** 89
- **Architectural Group:** Software Tests (tests/)
- **Summary:** Unit Test Suite for Inference and Early Warning Alert Modules. Validates alert triggers, false alarm rates, and health advisory formatting.

#### Classes Defined:
- **Class `TestEarlyWarning`**
  - `test_alert_generation_very_poor`
  - `test_alert_generation_severe`
  - `test_no_alert_for_good_moderate`
- **Class `TestAlertMessage`**
  - `test_message_formatting`
- **Class `TestHealthRisk`**
  - `test_risk_level_mapping`
  - `test_mask_recommendations`


> **Key Architectural Role:** Ensures the early warning alert system triggers reliably for dangerous air quality episodes.

---

### 4.42 Script: `tests/test_preprocessing.py`

- **File Path:** `tests/test_preprocessing.py`
- **Lines of Code:** 98
- **Architectural Group:** Software Tests (tests/)
- **Summary:** Unit Test Suite for Feature Engineering, Scaler, and Sequence Generation. Validates lag shifts, cyclic sine/cosine bounds, and 3D sliding-window tensor creation.

#### Classes Defined:
- **Class `TestLagFeatures`**
  - `test_lag_values_and_shifts`
- **Class `TestCyclicFeatures`**
  - `test_sine_cosine_bounds_between_minus_one_and_one`
- **Class `TestScaler`**
  - `test_scaler_fit_and_transform_dimensions`
  - `test_scaler_persistence`
- **Class `TestSequences`**
  - `test_3d_tensor_sliding_window_shapes`
  - `test_temporal_order_split`


> **Key Architectural Role:** Guarantees feature calculation correctness and temporal integrity of sliding windows.

---

### 4.43 Script: `Project Execution and Understanding/convert_artifact.py`

- **File Path:** `Project Execution and Understanding/convert_artifact.py`
- **Lines of Code:** 105
- **Architectural Group:** Utilities
- **Summary:** Artifact Inspection and Export Utility. Deserializes artifacts/scaler.joblib and exports feature-wise scaling statistics (means, scaling standard deviations, sample variances) to human-readable plain text and Excel formats.

> **Key Architectural Role:** Generates scaler_readable.txt and scaler_readable.xlsx for non-technical users and viva examiners who wish to inspect model normalization parameters without executing Python code.

---

### 4.44 Script: `Project Execution and Understanding/inspect_scaler.py`

- **File Path:** `Project Execution and Understanding/inspect_scaler.py`
- **Lines of Code:** 24
- **Architectural Group:** Utilities
- **Summary:** Quick Console Inspection Script. Reads artifacts/scaler.joblib and prints dictionary keys, scaler attributes, and feature counts directly to the terminal.

> **Key Architectural Role:** Lightweight command-line diagnostic tool for verifying artifact dictionary structure.

---

## 5. Detailed Walkthrough of All 13 Jupyter Notebooks

### Notebook 01: Data Collection from Open-Meteo API (`01_data_collection.ipynb`)

- **Objective:** Verify API connectivity, query Open-Meteo Air Quality and Weather Archive endpoints for Delhi coordinates, and test raw data pagination.
- **Key Steps:**
  - 1. Loads config/config.yaml for coordinates (28.6139°N, 77.2090°E) and pollutant list.
  - 2. Fetches sample 90-day chunk of hourly pollutant data (PM2.5, PM10, NO2, SO2, CO, O3).
  - 3. Fetches sample 90-day chunk of weather data (temperature, humidity, wind speed, pressure, dew point).
  - 4. Verifies timestamp indexing, HTTP 200 responses, and absence of API truncation.
- **Outputs & Visualizations:** Sample raw data tables and verified JSON parsing routines.

### Notebook 02: Data Preprocessing, Cleaning & Merging (`02_data_preprocessing.ipynb`)

- **Objective:** Explore missing data patterns, execute time-series linear interpolation, apply IQR outlier winsorization, and perform timestamp-aligned inner join.
- **Key Steps:**
  - 1. Analyzes missing data percentages per column across raw CSVs.
  - 2. Merges pollution and weather data on DatetimeIndex (29,905 aligned rows).
  - 3. Applies linear interpolation (limit=6) to impute short sensor dropouts.
  - 4. Calculates Q1, Q3, and IQR for each pollutant; winsorizes values above Q3 + 3.0*IQR.
  - 5. Plots before-and-after histograms and boxplots showing outlier treatment.
- **Outputs & Visualizations:** Cleaned merged dataset (data/processed/merged_data.csv) and distribution comparison plots.

### Notebook 03: CPCB Air Quality Index Calculation (`03_aqi_calculation.ipynb`)

- **Objective:** Implement the official Indian CPCB sub-index equations, calculate overall AQI across all timestamps, and classify into 6 standard categories.
- **Key Steps:**
  - 1. Defines breakpoint tables for PM2.5, PM10, NO2, SO2, CO, and O3 according to CPCB NAQI guidelines.
  - 2. Implements piecewise linear interpolation formula for each pollutant.
  - 3. Computes overall AQI as the maximum sub-index value.
  - 4. Maps overall AQI to categories: Good (0-50), Satisfactory (51-100), Moderate (101-200), Poor (201-300), Very Poor (301-400), Severe (401-500+).
  - 5. Generates class distribution bar plots and pie charts showing historical category frequencies.
- **Outputs & Visualizations:** AQI calculated dataset (data/processed/aqi_data.csv) and class frequency distributions.

### Notebook 04: 82-Dimensional Feature Engineering (`04_feature_engineering.ipynb`)

- **Objective:** Construct lag features, rolling moving averages, cyclic trigonometric encodings, emission ratios, and meteorological context variables.
- **Key Steps:**
  - 1. Generates 36 lag features (lags of 1h, 2h, 3h, 6h, 12h, 24h for 6 criteria pollutants).
  - 2. Generates 24 rolling moving averages (windows of 3h, 6h, 12h, 24h).
  - 3. Computes 8 cyclic trigonometric temporal encodings (sine and cosine of hour, day of week, day of year, month).
  - 4. Computes 3 pollutant interaction ratios (PM2.5/PM10, NO2/SO2, CO/O3).
  - 5. Integrates 5 meteorological variables (temperature, humidity, wind speed, pressure, dew point).
  - 6. Exports featured_data.csv (86 total columns: 82 features + 4 targets/metadata).
- **Outputs & Visualizations:** data/processed/featured_data.csv and artifacts/feature_groups.json.

### Notebook 05: Exploratory Data Analysis (EDA) (`05_eda.ipynb`)

- **Objective:** Analyze multi-year atmospheric trends, correlation heatmaps, seasonal pollution spikes, diurnal cycles, and weather impacts.
- **Key Steps:**
  - 1. Plots full correlation heatmap across all pollutants, weather variables, and AQI.
  - 2. Visualizes seasonal pollution patterns, highlighting severe winter spikes (Nov-Jan) vs monsoon dips (Jul-Aug).
  - 3. Computes diurnal (hourly) pollution curves, showing peak traffic/stagnation hours (08:00-10:00 and 20:00-23:00).
  - 4. Analyzes wind speed vs PM2.5 dispersion (higher wind speed strongly correlates with lower surface PM2.5).
  - 5. Displays pairwise scatter plots and pollutant ratio distributions.
- **Outputs & Visualizations:** Publication-quality EDA figures showing correlation heatmaps, seasonal distributions, and diurnal cycles.

### Notebook 06: Sequence Generation & Normalization (`06_sequence_creation.ipynb`)

- **Objective:** Fit standardization scaler on training split and generate 24-hour sliding window 3D tensors for sequential deep learning models.
- **Key Steps:**
  - 1. Identifies 82 feature columns and integer target aqi_code.
  - 2. Splits chronological dataset: 70% training (first 2.8 years) and 30% testing (final 1.2 years).
  - 3. Fits AQIScaler (StandardScaler) strictly on training split and normalizes full dataset.
  - 4. Applies 24-hour sliding window generator to create 3D tensors (N, 24, 82).
  - 5. Verifies split shapes: X_train (20892, 24, 82), y_train (20892,), X_test (8941, 24, 82), y_test (8941,).
  - 6. Saves compressed NumPy archive to data/processed/sequences.npz.
- **Outputs & Visualizations:** data/processed/sequences.npz and artifacts/scaler.joblib.

### Notebook 07: 1D Convolutional Neural Network Training (`07_cnn_model.ipynb`)

- **Objective:** Build, compile, train, and evaluate the baseline 1D-CNN architecture for hourly AQI category prediction.
- **Key Steps:**
  - 1. Loads 3D sequence tensors from sequences.npz.
  - 2. Builds 1D-CNN with Conv1D(64), BatchNorm, ReLU, Conv1D(128), BatchNorm, ReLU, MaxPool1D(2), Dense(128), Dropout(0.3), Dense(6, softmax).
  - 3. Compiles with Adam optimizer and sparse categorical cross-entropy loss.
  - 4. Connects EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
  - 5. Trains for 50 epochs with batch size 64.
  - 6. Evaluates test set metrics (Accuracy: 77.82%, F1: 0.7755) and plots training loss/accuracy curves.
- **Outputs & Visualizations:** models/cnn/best_model.keras, results/metrics/cnn_history.json, and convergence loss plots.

### Notebook 08: Stacked LSTM Model Training (`08_lstm_model.ipynb`)

- **Objective:** Design, train, and evaluate the baseline Stacked Long Short-Term Memory recurrent neural network.
- **Key Steps:**
  - 1. Loads sequence tensors.
  - 2. Builds Stacked LSTM: LSTM(128, return_sequences=True), Dropout(0.2), LSTM(64), Dropout(0.2), Dense(64, relu), Dense(6, softmax).
  - 3. Trains model using Adam optimizer with batch size 64 for 50 epochs.
  - 4. Evaluates performance (Accuracy: 75.01%, F1: 0.7392).
  - 5. Analyzes training dynamics, convergence speed, and gate memory behavior.
- **Outputs & Visualizations:** models/lstm/best_model.keras, results/metrics/lstm_history.json, and training curves.

### Notebook 09: Proposed Hybrid CNN–LSTM Model Training (`09_cnn_lstm_model.ipynb`)

- **Objective:** Design and train the primary proposed Hybrid architecture combining Conv1D feature extraction with recurrent LSTM modeling.
- **Key Steps:**
  - 1. Constructs Hybrid pipeline: Conv1D(64, kernel=3) -> BatchNorm -> ReLU -> MaxPool1D(2) -> LSTM(64) -> Dropout(0.3) -> Dense(64) -> Dense(6, softmax).
  - 2. Evaluates layer dimensions: (24, 82) -> Conv1D (24, 64) -> MaxPool (12, 64) -> LSTM (64,) -> Dense (6,).
  - 3. Fits model on training sequences using callbacks for checkpointing.
  - 4. Achieves peak project performance: Accuracy 80.64%, F1-Score 0.8059.
  - 5. Saves model weights to models/cnn_lstm/best_model.keras.
- **Outputs & Visualizations:** Primary production model checkpoint models/cnn_lstm/best_model.keras and results/metrics/hybrid_history.json.

### Notebook 10: Step-Wise Feature Ablation Study (Extension 2) (`10_ablation_study.ipynb`)

- **Objective:** Empirically evaluate the incremental performance contribution of each feature category across 6 sequential steps.
- **Key Steps:**
  - 1. Step 1: Base Pollutants only (6 features) -> F1: 0.7410.
  - 2. Step 2: Add Lag features (42 features) -> F1: 0.7785 (+3.75%).
  - 3. Step 3: Add Rolling Averages (66 features) -> F1: 0.7966 (+1.81%, largest marginal gain).
  - 4. Step 4: Add Cyclic Time encodings (74 features) -> F1: 0.7980 (+0.14%).
  - 5. Step 5: Add Pollutant Ratios (77 features) -> F1: 0.8012 (+0.32%).
  - 6. Step 6: Add Meteorological variables (82 features) -> F1: 0.8059 (+0.47%).
  - 7. Plots feature count vs F1 score and marginal delta charts.
- **Outputs & Visualizations:** results/ablation/ablation_results.csv, comparative bar charts, and 6 trained ablation checkpoints.

### Notebook 11: Meteorological Feature Impact Study (Extension 1) (`11_meteorological_extension.ipynb`)

- **Objective:** Quantify performance impact of surface weather features across all 3 architectures.
- **Key Steps:**
  - 1. Prepares 77 pollutant-only features vs 82 pollutant+weather features.
  - 2. Trains CNN, LSTM, and Hybrid models on both datasets (6 total models).
  - 3. Demonstrates that 1D-CNN gains +0.93% accuracy and +0.012 F1 when weather features are included.
  - 4. Analyzes why LSTM performs slightly better on lower-dimensional pollutant-only signals (avoiding recurrent state dilution).
  - 5. Exports weather impact summary table and side-by-side comparison charts.
- **Outputs & Visualizations:** results/meteorological/weather_comparison.csv, weather_impact.csv, and comparison charts.

### Notebook 12: Comprehensive Evaluation & Statistical Tests (`12_model_evaluation.ipynb`)

- **Objective:** Execute comprehensive comparative benchmarking, generate 6x6 confusion matrices, and conduct McNemar's hypothesis tests.
- **Key Steps:**
  - 1. Computes full classification reports (accuracy, precision, recall, F1, Cohen's kappa) for CNN, LSTM, and Hybrid models.
  - 2. Renders 6x6 normalized confusion matrices for all 3 models in a 1x3 comparative layout.
  - 3. Performs McNemar's non-parametric paired tests with continuity correction.
  - 4. Proves CNN vs Hybrid (chi2 = 78.4, p < 0.001) and LSTM vs Hybrid (chi2 = 142.1, p < 0.001) are statistically significant.
  - 5. Serializes results to results/reports/mcnemar_tests.csv and model_comparison.csv.
- **Outputs & Visualizations:** Normalized confusion matrices, McNemar test tables, and master benchmark comparison summaries.

### Notebook 13: Threshold-Based Early Warning Alert System (Extension 3) (`13_early_warning.ipynb`)

- **Objective:** Design, calibrate, and validate the high-precision early warning alert module targeting hazardous air quality conditions.
- **Key Steps:**
  - 1. Identifies high-risk conditions: Very Poor (Code 4, AQI 301-400) and Severe (Code 5, AQI 401-500+).
  - 2. Sets probability confidence threshold at 0.60.
  - 3. Evaluates alert triggers across 8,941 test hours.
  - 4. Measures performance: Alert Precision = 93.67%, Alert Recall = 87.51%, Alert F1-Score = 90.48%.
  - 5. Formulates public health advisories and mask guidance for each risk category.
- **Outputs & Visualizations:** results/reports/early_warning_results.json and simulated alert logs.

---

## 6. Frontend Web Dashboard & RESTful Architecture

The web interface is implemented in Flask (`app/app.py`) with responsive HTML5/CSS3/JavaScript (`templates/index.html`, `static/css/style.css`, `static/js/script.js`).

### RESTful Endpoints:
- `GET /`: Serves the primary web dashboard.
- `GET /api/latest-prediction`: Returns the active dynamic prediction state from `results/latest_prediction.json`.
- `POST /api/run-prediction`: Triggers on-demand prediction execution for any target datetime.
- `POST /api/predict-datetime`: Accepts target timestamp (e.g. `{"datetime": "2026-11-20 15:00"}`) and executes real-time prediction.
- `GET /api/results`: Exposes multi-model evaluation benchmarks in JSON format.
- `GET /plots/<filename>`: Serves static evaluation plots (confusion matrices, training loss curves).

---

## 7. Experimental Results, Benchmarks & Statistical Significance

### 7.1 Master Model Benchmark (Test Set: 8,941 samples)

| Model Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Val Loss | Trainable Parameters | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1D CNN** | **77.82%** | 0.7881 | 0.7782 | **0.7755** | 0.5621 | 224,966 | 3.8 min |
| **Stacked LSTM** | **75.01%** | 0.7709 | 0.7501 | **0.7392** | 0.6184 | 129,670 | 6.2 min |
| **Hybrid CNN–LSTM** | **80.64%** | 0.8070 | 0.8064 | **0.8059** | 0.4982 | 217,542 | 4.5 min |


### 7.2 McNemar's Hypothesis Testing (Statistical Significance)

| Pairwise Comparison | Chi-Squared ($\chi^2$) | $p$-value | Significance ($\alpha=0.001$) | Empirical Conclusion |
|:---|:---:|:---:|:---:|:---|
| CNN vs Hybrid CNN–LSTM | 78.42 | < 0.001 | Yes (p < 0.001) | Hybrid model statistically superior to 1D CNN |
| LSTM vs Hybrid CNN–LSTM | 142.15 | < 0.001 | Yes (p < 0.001) | Hybrid model statistically superior to Stacked LSTM |
| CNN vs Stacked LSTM | 41.83 | < 0.001 | Yes (p < 0.001) | 1D CNN statistically superior to Stacked LSTM |


### 7.3 Extension 1: Meteorological Feature Impact

| Model Architecture | Pollutant Acc | Weather Acc | $\Delta$ Accuracy | Pollutant F1 | Weather F1 | $\Delta$ F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1D CNN** | 76.89% | 77.82% | **+0.93%** | 0.7635 | 0.7755 | **+0.0120** |
| **Stacked LSTM** | 75.48% | 75.01% | **-0.47%** | 0.7441 | 0.7392 | **-0.0049** |
| **Hybrid CNN–LSTM** | 80.24% | 80.64% | **+0.40%** | 0.8012 | 0.8059 | **+0.0047** |


### 7.4 Extension 2: Step-Wise Feature Ablation Study

| Ablation Step | Feature Count | Accuracy | F1-Score | Marginal F1 Gain | Feature Set Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Step 1: Base Pollutants** | 6 | 74.82% | **0.7410** | **Baseline** | Raw concentrations (PM2.5, PM10, NO2, SO2, CO, O3) |
| **Step 2: + Lag Features** | 42 | 78.11% | **0.7785** | **+3.75% F1** | Lags at 1h, 2h, 3h, 6h, 12h, 24h |
| **Step 3: + Rolling Means** | 66 | 79.85% | **0.7966** | **+1.81% F1** | Moving averages across 3h, 6h, 12h, 24h (Peak Marginal Gain) |
| **Step 4: + Cyclic Time** | 74 | 80.02% | **0.7980** | **+0.14% F1** | sin/cos encodings for hour, weekday, day of year, month |
| **Step 5: + Ratios** | 77 | 80.24% | **0.8012** | **+0.32% F1** | PM2.5/PM10, NO2/SO2, CO/O3 interaction ratios |
| **Step 6: + Meteorological** | 82 | 80.64% | **0.8059** | **+0.47% F1** | Temperature, humidity, wind speed, pressure, dew point |


### 7.5 Extension 3: Early Warning Alert System Validation

Across 8,941 evaluation test hours:
- **Total Test Hours:** 8941
- **True Hazardous Smog Hours:** 3218
- **Alert Precision:** **93.67%** (Only 6.33% false alarms)
- **Alert Recall:** **87.51%** (Detects 87.5% of hazardous smog episodes)
- **Alert F1-Score:** **90.48%**

---

## 8. Software Testing, Verification & Viva Preparation Guide

### 8.1 Automated Unit Testing
34 automated test cases verify system reliability:
```powershell
python -m unittest discover tests
```
Result: `Ran 34 tests in 4.886s - OK`

### 8.2 Team Member Viva Preparation Guide
- **Harshit Giri (Frontend & EDA):** Defends `app/app.py`, `index.html`, asynchronous API polling, and `notebooks/05_eda.ipynb` seasonal correlation trends.
- **Adarsh Raj (1D CNN & Meteorology):** Defends `src/models/cnn.py`, `run_meteorological_comparison.py`, and Conv1D local pattern extraction (+0.93% weather gain).
- **Shivam Verma (Stacked LSTM & Sequences):** Defends `src/models/lstm.py`, `sequence_generator.py`, 24-hour sliding windows `(24, 82)`, and recurrent gate dynamics.
- **Sushant Prasad (Hybrid CNN-LSTM & Project Lead):** Defends `src/models/cnn_lstm.py` (80.64% accuracy), McNemar's test ($p < 0.001$), 6-step ablation study, and early warning precision (93.67%).
