# 📘 AQI Prediction CNN-LSTM — Documentation Hub

Welcome to the definitive documentation hub for the **Air Quality Index (AQI) Prediction using Hybrid CNN–LSTM Deep Learning Model** project (with Meteorological Extension, Step-wise Feature Ablation, and Threshold-Based Early Warning Module).

This documentation suite is organized to give researchers, students, evaluators, and developers an exhaustive understanding of the theoretical background, neural network architecture, data engineering pipeline, experimental findings, and practical step-by-step usage of the system.

---

## 🗂️ Documentation Navigation Directory

| Document | Description | Target Audience |
|:---|:---|:---|
| 📑 [**Complete Master Project Report (DOCX)**](COMPLETE_PROJECT_REPORT.docx) / [**(MD)**](COMPLETE_PROJECT_REPORT.md) | **Exhaustive Final Project Report** — Every dataset schema, 51 script breakdowns, 13 notebooks walkthrough, full benchmarks, mathematical formulas, and viva defense guide. | Final B.Tech Submission, College Faculty, Evaluators |
| 📖 [**Project User Guide**](PROJECT_USER_GUIDE.md) | **Complete Step-by-Step Manual** — installation, running the web app, single datetime inference, full pipeline execution, unit testing, and copy-paste code examples. | Developers, Users, Evaluators |
| 📑 [**Detailed Project Report**](DETAILED_PROJECT_REPORT.md) | **26-Section Master Academic & Technical Report** — literature survey, problem statement, full pipeline breakdown, breakpoint tables, layer architectures, training configs, and complete experimental conclusions. | Faculty, Mentors, Academic Reviewers |
| 🏛️ [**System Architecture & Methodology**](SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md) | **Technical & Mathematical Deep Dive** — data pipelines, CPCB interpolation formulas, 82 feature definitions, 1D CNN + LSTM mechanisms, sliding window tensor shapes, and loss formulations. | Machine Learning Engineers, Researchers |
| 📊 [**Results & Evaluation**](RESULTS_AND_EVALUATION.md) | **Exhaustive Experimental Findings** — CNN vs LSTM vs Hybrid metrics, McNemar's statistical tests, Extension 1 (meteorological impact), Extension 2 (ablation study), and Extension 3 (early warning system). | Data Scientists, Evaluators |
| 🔌 [**API Documentation**](api_documentation.md) | **Complete API Reference** — Open-Meteo external data APIs, Flask Web Dashboard REST endpoints, and internal Python class & function interfaces across all modules. | Software Developers, Integrators |
| 📄 [**Academic Project Synopsis (DOCX)**](AQI_Project_Synopsis.docx) | **Original Academic Project Synopsis** submitted for the award of Bachelor of Technology in Computer Science & Information Technology. | College / University Examination |
| 👥 [**Team Project Distribution**](TEAM_PROJECT_DISTRIBUTION.md) | **Individual Work & Study Breakdown** — Detailed responsibility matrix, core concepts, exact files/folders, and mock viva questions for Harshit, Adarsh, Shivam, and Sushant. | Team Members, Examiners, Reviewers |
| 📋 [**Quick Run Steps (PDF)**](AQI_Simple_Run_Steps.pdf) | **Condensed visual run steps guide** for rapid setup and execution. | All Users |

---

## ⚡ Quick Executive Overview

```
                          ┌─────────────────────────────┐
                          │   Open-Meteo APIs (Free)    │
                          │ • Air Quality (6 Pollutants)│
                          │ • Weather (5 Variables)     │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │     Data Preprocessing      │
                          │ • Inner Join on Datetime    │
                          │ • Linear Interpolation      │
                          │ • IQR Winsorization (3.0x)  │
                          │ • Indian CPCB AQI (0 to 5)  │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │   Feature Engineering (82)  │
                          │ • 6 Base Pollutants         │
                          │ • 36 Lag Features (1-24h)   │
                          │ • 24 Rolling Averages       │
                          │ • 8 Cyclic Temporal Encodings│
                          │ • 3 Pollutant Ratios        │
                          │ • 5 Meteorological Variables│
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │     Sequence Generation     │
                          │ • 24-Hour Sliding Windows   │
                          │ • Input Shape: (N, 24, 82)  │
                          │ • Chronological 70/30 Split │
                          └──────────────┬──────────────┘
                                         │
                     ┌───────────────────┼───────────────────┐
                     │                   │                   │
                     ▼                   ▼                   ▼
             ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
             │   1D CNN      │   │     LSTM      │   │Hybrid CNN-LSTM│
             │ Accuracy:77.8%│   │ Accuracy:75.0%│   │ Accuracy:80.6%│
             │ F1: 0.7755    │   │ F1: 0.7392    │   │ F1: 0.8059    │
             └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                     │                   │                   │
                     └───────────────────┼───────────────────┘
                                         │
                                         ▼
             ┌───────────────────────────────────────────────────────┐
             │                  Ensemble Forecast &                  │
             │              Early Warning Alert System               │
             │  • On-Demand Live Forecast (Hybrid CNN-LSTM)          │
             │  • Health Alerts (Very Poor / Severe AQI)             │
             │  • Early Warning F1: 90.48% | Precision: 93.67%       │
             │  • Interactive Flask Web Dashboard (`app/app.py`)     │
             └───────────────────────────────────────────────────────┘
```

---

## 👥 Project Team & Academic Credits

- **Project Title:** Air Quality Index (AQI) Prediction using Hybrid CNN–LSTM Deep Learning Model: An Extension with Meteorological Features and an Early-Warning Module
- **Degree:** Bachelor of Technology in Computer Science and Information Technology
- **Institution:** G.L. Bajaj Institute of Technology and Management, Greater Noida
- **Project Guide:** Prof. Animesh Malviya

**Student Researchers:**
1. **Sushant Prasad** (Roll No: 2301920130192)
2. **Shivam Verma** (Roll No: 2301920130163)
3. **Harshit Giri** (Roll No: 2401920139005)
4. **Adarsh Raj** (Roll No: 2401920139001)

---

## 🎯 How to Use This Documentation

- **For Rapid Execution:** If you just want to run the live dashboard or predict AQI right away, go straight to [**Project User Guide**](PROJECT_USER_GUIDE.md).
- **For Presentation & Viva Voce:** Review [**Detailed Project Report**](DETAILED_PROJECT_REPORT.md) and [**Results & Evaluation**](RESULTS_AND_EVALUATION.md) to understand key statistics, McNemar's tests, and ablation outcomes.
- **For Deep Technical Inquiries:** Check [**System Architecture & Methodology**](SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md) for full mathematical equations and network layer breakdowns.

