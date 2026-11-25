# 🌬️ Air Quality Index (AQI) Prediction using Hybrid CNN–LSTM Deep Learning Model

> **An Extension with Meteorological Features, Feature Ablation, and an Early-Warning Module**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📚 Complete Project Documentation Suite

Exhaustive, dedicated documentation is organized inside the [`Project Execution and Understanding/docs/`](Project%20Execution%20and%20Understanding/docs/) directory:

| Document | Purpose |
|:---|:---|
| 📑 [**Complete Master Project Report (DOCX)**](Project%20Execution%20and%20Understanding/docs/COMPLETE_PROJECT_REPORT.docx) / [**(MD)**](Project%20Execution%20and%20Understanding/docs/COMPLETE_PROJECT_REPORT.md) | **Exhaustive Final Project Report** — Complete dataset catalog, 51 script breakdowns, 13 notebooks walkthrough, full benchmarks, and viva defense guide. |
| 📖 [**Project User Guide**](Project%20Execution%20and%20Understanding/docs/PROJECT_USER_GUIDE.md) | **Step-by-step manual** to run web app, arbitrary datetime predictor, pipeline, tests, and API code. |
| 📑 [**Detailed Project Report**](Project%20Execution%20and%20Understanding/docs/DETAILED_PROJECT_REPORT.md) | **26-section master technical and academic report** with full literature review, architecture, tables, and analysis. |
| 🏛️ [**System Architecture & Methodology**](Project%20Execution%20and%20Understanding/docs/SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md) | **Deep mathematical specifications**, CPCB sub-index formulas, 82 feature equations, and CNN-LSTM layer tensor shapes. |
| 📊 [**Results & Evaluation**](Project%20Execution%20and%20Understanding/docs/RESULTS_AND_EVALUATION.md) | **Empirical benchmarks**, McNemar's tests ($p < 0.001$), ablation steps, and meteorological impact comparison. |
| 🔌 [**API Documentation**](Project%20Execution%20and%20Understanding/docs/api_documentation.md) | **Complete reference** for Open-Meteo external APIs, Flask REST endpoints, and internal Python modules. |
| 📄 [**Academic Project Synopsis (DOCX)**](Project%20Execution%20and%20Understanding/docs/AQI_Project_Synopsis.docx) | **Official B.Tech Academic Synopsis** submitted to G.L. Bajaj Institute of Technology and Management. |
| 👥 [**Team Project Distribution & Study Guide**](Project%20Execution%20and%20Understanding/docs/TEAM_PROJECT_DISTRIBUTION.md) | **Individual responsibilities & file study guide** for Harshit (Frontend), Adarsh (CNN), Shivam (LSTM), and Sushant (Hybrid CNN-LSTM & Lead). |
| 📋 [**Quick Run Steps (PDF)**](Project%20Execution%20and%20Understanding/docs/AQI_Simple_Run_Steps.pdf) | **Condensed visual run steps guide** for rapid setup and execution. |

---

## 📋 Project Overview

This project implements an end-to-end **Hybrid CNN–LSTM deep learning framework** for short-term hourly Air Quality Index (AQI) prediction and on-demand date & time forecasting targeting **Delhi, India** (28.6139°N, 77.2090°E).

The system integrates 4 years (2022–2025, 35,064 hourly samples) of pollutant concentrations ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$) and meteorological variables (temperature, humidity, wind speed, pressure, dew point) from the free **Open-Meteo API**. AQI is calculated strictly following the **Indian Central Pollution Control Board (CPCB)** standard across 6 categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).

### Academic Credits
- **Institution:** G.L. Bajaj Institute of Technology and Management, Greater Noida
- **Department:** Computer Science and Information Technology
- **Project Guide:** Prof. Animesh Malviya
- **Team Members:** Sushant Prasad (2301920130192), Shivam Verma (2301920130163), Harshit Giri (2401920139005), Adarsh Raj (2401920139001)

---

## 🏆 Key Experimental Findings

### 1. Model Performance Benchmark (Test Set: 8,941 samples)
| Model Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|:---|:---:|:---:|:---:|:---:|
| **1D CNN** | 77.82% | 0.7881 | 0.7782 | 0.7755 |
| **Stacked LSTM** | 75.01% | 0.7709 | 0.7501 | 0.7392 |
| **Hybrid CNN–LSTM** | **80.64%** | **0.8070** | **80.64%** | **0.8059** |

*All pairwise model differences are statistically significant ($p < 0.001$, McNemar's test).*

### 2. Research Extensions
- **Extension 1 (Meteorological Comparison):** 1D CNN gained +0.93% accuracy with weather features; recurrent models performed better with pure pollutant signals.
- **Extension 2 (6-Step Ablation Study):** Rolling window averages (3h–24h) delivered the single highest performance gain (+1.5% F1), reaching peak performance at Step 3 (0.7966 F1).
- **Extension 3 (Early Warning Alert System):** Threshold-based module achieved **93.67% alert precision**, **87.51% alert recall**, and **90.48% alert F1-score** for dangerous air quality conditions (Very Poor and Severe).
- **On-Demand Live Prediction System:** Multi-model weighted soft-voting ensemble (CNN: 0.20, LSTM: 0.25, Hybrid: 0.55) for any historical or future date and time with 74%–94% prediction confidence.

---

## ⚡ Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/sushantprasad11/AQI-Prediction-CNN-LSTM.git
cd AQI-Prediction-CNN-LSTM
pip install -r requirements.txt
```

### 2. Launch Interactive Web Dashboard
```bash
python app/app.py
```
Open **`http://localhost:5000`** in your browser to view the interactive dashboard, model comparisons, ablation findings, and live forecasts.

### 3. Predict AQI for Any Datetime
```bash
# Historical query
python predict_any_datetime.py "2025-06-15 14:00"

# Future forecast query
python predict_any_datetime.py "2026-11-26 12:00"
```

### 4. Run Automated Unit Tests (34 Tests)
```bash
# Windows PowerShell
$env:KERAS_HOME = "$((Get-Location).Path)\.keras"
python -m unittest discover -s tests -v

# Linux / macOS
KERAS_HOME="$(pwd)/.keras" python3 -m unittest discover -s tests -v
```

---

## 📂 Project Structure

```
AQI-Prediction-CNN-LSTM/
├── app/                               # Flask web dashboard & UI assets
│   ├── app.py                         # Application server & REST APIs
│   ├── templates/index.html           # Modern responsive dashboard
│   └── static/                        # CSS styles and JavaScript
├── artifacts/                         # Serialized pipeline artifacts
│   ├── scaler.joblib                  # Fitted Z-score StandardScaler
│   └── feature_groups.json            # Feature group configurations
├── config/                            # Central configuration
│   └── config.yaml                    # APIs, coordinates, breakpoints, hyperparameters
├── data/                              # Datasets (4 years of hourly records)
│   ├── raw/                           # Raw API CSVs (pollution & weather)
│   └── processed/                     # Merged, AQI-calculated, 82-feature sequences (.npz)
├── "Project Execution and Understanding"/ # MASTER DOCUMENTATION & EXECUTION SUITE
│   ├── In Powershell.txt              # Quick execution commands for PowerShell
│   ├── scaler_readable.txt            # Human-readable feature & scaler statistics
│   ├── scaler_readable.xlsx           # Excel format feature & scaler statistics
│   ├── convert_artifact.py            # Scaler artifact conversion utility
│   └── docs/                          # Comprehensive Documentation Hub
│       ├── README.md                  # Documentation hub & portal
│       ├── PROJECT_USER_GUIDE.md      # Complete operational manual
│       ├── DETAILED_PROJECT_REPORT.md # Comprehensive 26-section project report
│       ├── SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md # Mathematical and deep learning formulation
│       ├── RESULTS_AND_EVALUATION.md  # Experimental benchmark tables & analysis
│       ├── api_documentation.md       # External, Flask, and internal API reference
│       ├── AQI_Project_Synopsis.docx  # Academic college synopsis document
│       ├── TEAM_PROJECT_DISTRIBUTION.md # 4-member project study & responsibility breakdown
│       └── AQI_Simple_Run_Steps.pdf   # Visual setup and run guide
├── models/                            # Saved trained checkpoints (.keras)
│   ├── cnn/best_model.keras           # Trained 1D CNN
│   ├── lstm/best_model.keras          # Trained LSTM
│   ├── cnn_lstm/best_model.keras      # Trained Primary Hybrid CNN-LSTM
│   ├── ablation_step_1 .. step_6/     # Checkpoints for ablation study
│   └── *_pollutant / *_weather/       # Checkpoints for meteorological study
├── notebooks/                         # 13 step-by-step Jupyter notebooks (01 to 13)
├── results/                           # Evaluation metrics, plots, and forecasts
│   ├── metrics/                       # Classification reports CSVs
│   ├── plots/                         # Confusion matrices & loss curves
│   ├── reports/                       # Results JSON summaries & McNemar tests
│   ├── ablation/                      # Ablation study CSV and comparison chart
│   ├── meteorological/                # Weather comparison CSV and chart
│   └── forecasts/                     # Prediction JSON outputs
├── src/                               # Modular Python source code
│   ├── data/                          # Fetching, cleaning, merging, loading
│   ├── preprocessing/                 # AQI calculator, feature engineering, scaler, sequences
│   ├── models/                        # CNN, LSTM, and Hybrid CNN-LSTM architectures
│   ├── training/                      # Training functions & callbacks
│   ├── evaluation/                    # Metrics, confusion matrix, statistical tests
│   └── prediction/                    # Predictors, ensemble, datetime forecast, early warning
├── tests/                             # Unit tests (34 test cases, 100% passing)
├── predict_any_datetime.py            # Single datetime prediction runner
├── run_full_pipeline.py               # End-to-end pipeline runner
├── run_pipeline.py                    # Data preparation runner
├── run_training.py                    # Model training runner
├── run_ablation_study.py              # 6-step ablation runner
├── run_meteorological_comparison.py   # Meteorological comparison runner
├── run_visualizations.py              # Result visualizer & summary generator
├── requirements.txt                   # Python dependencies
└── README.md                          # Repository overview
```

---

## 📜 Academic References

1. Madan, P., et al. (2026). "Hybrid deep learning model for air quality prediction," *Scientific Reports*, Nature.
2. Nadheer, A., & Aal-Nouman, M. (2026). "Hybrid CNN-LSTM for urban air pollution forecasting," *Future Technology*.
3. Çelikten, A. (2026). "Multi-output deep learning for PM10 and AQI forecasting," *Sustainability*, MDPI.
4. Sri Lakshmi, K., & Jyothi, S. (2026). "Predictive analytics for AQI using LSTM," *JAIIS*.
5. Tan, X., et al. (2026). "Deep learning-based prediction of air pollution in China," *Atmosphere*, MDPI.
6. Central Pollution Control Board (CPCB), Government of India. "National Air Quality Index."

---
*Developed for academic evaluation — G.L. Bajaj Institute of Technology and Management*
