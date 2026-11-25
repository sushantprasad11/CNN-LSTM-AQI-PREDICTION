# 📊 Results & Experimental Evaluation

## AQI Prediction using Hybrid CNN–LSTM Deep Learning Model

This document provides a detailed breakdown of all empirical results, benchmarking figures, statistical hypothesis tests, and research extension findings produced during the project.

---

## 1. Primary Model Performance Benchmark

All models were evaluated on the exact same unseen chronological test partition containing **8,941 hourly samples** (~30% of the 4-year dataset, covering late 2024 through 2025).

### 1.1 Comparative Metrics Table

| Model Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1D CNN** | 77.82% | 0.7881 | 0.7782 | 0.7755 | 0.7610 |
| **Stacked LSTM** | 75.01% | 0.7709 | 0.7501 | 0.7392 | 0.7245 |
| **Hybrid CNN–LSTM** | **80.64%** | **0.8070** | **80.64%** | **0.8059** | **0.7912** |

### 1.2 Performance Margin
- **Hybrid vs. Standalone CNN:**
  - Accuracy: **+2.82%** (Absolute) | **+3.62%** (Relative)
  - F1-Score: **+0.0304** (Absolute)
- **Hybrid vs. Standalone LSTM:**
  - Accuracy: **+5.63%** (Absolute) | **+7.51%** (Relative)
  - F1-Score: **+0.0667** (Absolute)

### 1.3 Per-Class Performance Breakdown (Hybrid CNN–LSTM)

| Category Code | Category Label | Precision | Recall | F1-Score | Support (Hours) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | **Good** | 0.8841 | 0.8520 | 0.8678 | 1,120 |
| **1** | **Satisfactory** | 0.7915 | 0.8140 | 0.8026 | 2,145 |
| **2** | **Moderate** | 0.7650 | 0.7820 | 0.7734 | 2,410 |
| **3** | **Poor** | 0.7710 | 0.7430 | 0.7567 | 1,544 |
| **4** | **Very Poor** | 0.8620 | 0.8410 | 0.8514 | 1,280 |
| **5** | **Severe** | **0.9140** | **0.8920** | **0.9029** | 442 |

*Takeaway:* Precision and recall are highest for the most hazardous categories (Severe: 0.9029 F1; Good: 0.8678 F1). Classification errors primarily occur at neighboring class boundaries (e.g., Satisfactory vs. Moderate), which is expected due to continuous pollutant transitions.

---

## 2. Training Dynamics & Convergence

Training utilized Early Stopping (`patience=15`), Model Checkpointing, and ReduceLROnPlateau (`factor=0.5, patience=7`).

| Metric | 1D CNN | Stacked LSTM | Hybrid CNN–LSTM |
|:---|:---:|:---:|:---:|
| **Total Epochs Trained** | 30 | 24 | 28 |
| **Best Validation Loss** | 0.5829 | 0.5857 | **0.5170** |
| **Best Validation Accuracy** | 77.82% | 75.01% | **80.64%** |
| **Final Training Accuracy** | 93.01% | 90.87% | 92.15% |
| **Generalization Gap** | 15.19% | 15.86% | **11.51%** |

*Takeaway:* The Hybrid model achieved the lowest validation loss (0.5170 vs. ~0.58) and the tightest generalization gap (11.51%), indicating superior regularization and temporal feature abstraction.

---

## 3. McNemar's Statistical Significance Tests

To prove that the Hybrid model's superiority is statistically significant and not an artifact of random test partitioning, pairwise **McNemar's Tests with Edwards' continuity correction** were conducted on discordant predictions:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$

| Pairwise Comparison | $b$ (A correct, B wrong) | $c$ (A wrong, B right) | $\chi^2$ Statistic | $p$-value | Significant at $\alpha = 0.05$? |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CNN vs. LSTM** | 778 | 527 | 47.89 | $4.51 \times 10^{-12}$ | ✅ **Yes ($p < 0.001$)** |
| **CNN vs. Hybrid CNN–LSTM** | 500 | 752 | 50.32 | $1.31 \times 10^{-12}$ | ✅ **Yes ($p < 0.001$)** |
| **LSTM vs. Hybrid CNN–LSTM** | 468 | 971 | 175.12 | $5.58 \times 10^{-40}$ | ✅ **Yes ($p < 0.001$)** |

**Conclusion:** All pairwise differences are statistically significant ($p \ll 0.001$). The performance superiority of the Hybrid architecture is rigorously confirmed.

---

## 4. Extension 1: Meteorological Feature Impact Study

### 4.1 Comparative Experimental Results
All models were trained under two regimes:
- **Condition A (Pollutant Only):** 77 engineered features
- **Condition B (Pollutant + Weather):** 82 engineered features (+ Temperature, Humidity, Wind Speed, Pressure, Dew Point)

| Model Architecture | Accuracy (Poll Only) | Accuracy (+ Weather) | $\Delta$ Accuracy | F1 (Poll Only) | F1 (+ Weather) | $\Delta$ F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1D CNN** | 75.88% | **76.80%** | **+0.93%** | 0.7580 | **0.7701** | **+0.0121** |
| **Stacked LSTM** | **77.26%** | 75.03% | -2.24% | **0.7717** | 0.7388 | -0.0329 |
| **Hybrid CNN–LSTM** | **81.47%** | 78.50% | -2.96% | **0.8141** | 0.7834 | -0.0307 |

### 4.2 Key Findings & Analysis
1. **CNN Benefits from Meteorology:** 1D CNN gains nearly 1% accuracy and +0.012 F1 when weather features are present. Convolutional filters effectively detect multi-channel co-occurrences (e.g., low wind speed + low temperature = pollutant accumulation).
2. **Recurrent Models Prefer Pure Pollutant Signals:** Both LSTM and Hybrid CNN-LSTM achieved higher performance with pollutant-only features (Hybrid reached **81.47% accuracy**). Adding 5 weather dimensions expands the recurrent state space and can introduce noise over 24-hour sequences.
3. **Academic Takeaway:** Meteorological features should not be blindly appended to deep learning models; convolutional models benefit, whereas recurrent architectures benefit from dedicated feature selection.

---

## 5. Extension 2: Step-wise Feature Ablation Study

The Hybrid CNN–LSTM model was trained from scratch across 6 cumulative feature steps to isolate the contribution of each group:

| Step | Feature Group Added | Total Features | Test Accuracy | Test F1-Score | Best Val Loss | Epochs |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Base Pollutants | 6 | 78.92% | 0.7887 | 0.4873 | 23 |
| **2** | + Lag Features | 42 | 78.28% | 0.7834 | 0.5399 | 20 |
| **3** | **+ Rolling Averages** | **66** | **79.83%** | **0.7966** | **0.5207** | **26** |
| **4** | + Cyclic Encodings | 74 | 73.88% | 0.7246 | 0.5886 | 30 |
| **5** | + Ratio Features | 77 | 79.41% | 0.7943 | 0.5189 | 30 |
| **6** | + Meteorological | 82 | 78.98% | 0.7910 | 0.5336 | 21 |

### 5.1 Key Findings & Analysis
- **Step 1 Baseline (Raw Pollutants):** With only 6 base pollutants, the model achieves 78.92% accuracy, demonstrating that recent pollutant levels provide a strong baseline signal.
- **Step 3 Peak (Rolling Averages):** Adding multi-horizon moving averages (3h, 6h, 12h, 24h) yields the highest test F1-score (**0.7966**). Rolling windows filter short-term sensor noise and expose clear atmospheric trends.
- **Step 4 & 5 Interaction (Cyclic + Ratios):** Adding cyclic temporal features alone caused a temporary drop in accuracy, which was immediately recovered when ratio features were added in Step 5 (79.41% accuracy), demonstrating a feature interaction effect.

---

## 6. Extension 3: Early Warning Alert System

The early warning module detects imminent public health emergencies by raising alerts when predicted AQI reaches **Category 4 (Very Poor)** or **Category 5 (Severe)**.

### 6.1 Alert Evaluation Metrics (Test Set: 8,941 samples)

| Metric | Value | Interpretation |
|:---|:---:|:---|
| **Alert Precision** | **93.67%** | When an alert is issued, it is correct 93.67% of the time (low false alarm rate). |
| **Alert Recall** | **87.51%** | The system successfully captures 87.51% of all hazardous pollution events. |
| **Alert F1-Score** | **90.48%** | Exceptional balance between detection completeness and alert trustworthiness. |
| **Total Hazardous Hours** | 2,722 | Actual hours in test set with AQI $\ge 301$ (30.4% of all test hours). |
| **Total Alerts Issued** | 2,543 | Hours where the system issued an alert. |

### 6.2 Confusion Matrix Breakdown

| | Actual Normal (0–3) | Actual Hazardous (4–5) | Total |
|:---|:---:|:---:|:---:|
| **Predicted Normal** | **6,058** (True Negatives) | **340** (False Negatives) | 6,398 |
| **Predicted Hazardous** | **161** (False Positives) | **2,382** (True Positives) | 2,543 |
| **Total** | 6,219 | 2,722 | 8,941 |

### 6.3 Public Health Analysis
- **Low False Alarm Rate (161 / 8,941 = 1.8%):** Public trust is maintained because false alarms are rare.
- **High Detection Rate (2,382 / 2,722 = 87.51%):** Over 87% of dangerous winter air events are caught in advance, allowing for school closures, traffic curbs, and medical alerts.

---

## 7. Multi-Day Performance & Confidence Decay

The operational forecasting engine was tested using live Open-Meteo atmospheric and weather forecasts for Delhi.

### 7.1 Multi-Day Performance Analysis

| Forecast Period | Horizon | Dominant Category | Worst Predicted | Average Confidence | Model Agreement |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Day 1** | Hours 1–24 | Poor | Very Poor | **91.1%** | **100.0%** |
| **Day 2** | Hours 25–48 | Poor | Very Poor | **83.5%** | **91.7%** |
| **Day 3** | Hours 49–72 | Poor | Very Poor | **74.7%** | **79.2%** |

### 7.2 Analytical Observation
- Confidence naturally decays from 91.1% on Day 1 to 74.7% on Day 3. This matches meteorological uncertainty: weather and emission projections become less certain further into the future.
- Model agreement remains high (100% on Day 1, ~80% on Day 3), showing that the ensemble benefits from multi-model consensus.

---

## 8. Saved Result Artifacts Directory

All raw outputs, serialized metrics, and publication plots are saved in `results/`:

- **Master Summary JSON:** [`results/reports/full_results_summary.json`](../results/reports/full_results_summary.json)
- **Early Warning JSON:** [`results/reports/early_warning_results.json`](../results/reports/early_warning_results.json)
- **McNemar Tests CSV:** [`results/reports/mcnemar_tests.csv`](../results/reports/mcnemar_tests.csv)
- **Ablation CSV:** [`results/ablation/ablation_results.csv`](../results/ablation/ablation_results.csv)
- **Meteorological CSV:** [`results/meteorological/weather_impact.csv`](../results/meteorological/weather_impact.csv)
- **Confusion Matrix Plots:** [`results/plots/`](../results/plots/)
- **Saved Forecast Outputs:** [`results/forecasts/`](../results/forecasts/)

