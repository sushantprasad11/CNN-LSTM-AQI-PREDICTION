# 👥 Project Work Distribution & Comprehensive Study Guide
## Air Quality Index (AQI) Prediction using Hybrid CNN–LSTM Deep Learning Model
> **Institution:** G.L. Bajaj Institute of Technology and Management, Greater Noida  
> **Department:** Computer Science and Information Technology  
> **Project Guide:** Prof. Animesh Malviya  
> **Academic Session:** 2026–2027  

---

## 📌 Executive Summary & Distribution Matrix

To ensure mastery during academic project reviews, faculty presentations, and external viva voce examinations, the project components, research extensions, source files, and engineering responsibilities are distributed among the four student researchers:

| Member | Primary Responsibility | Core Architecture / Domain | Key Deliverable | Primary Directories & Files |
|:---|:---|:---|:---|:---|
| **Harshit Giri**<br>`2401920139005` | **Frontend & Interactive UI** | Real-Time Dashboard, Dynamic Polling, Responsive CSS, Charts & UX | Live Web UI, Polling System, Visual Forecast Cards | `app/templates/index.html`<br>`app/static/css/style.css`<br>`app/static/js/script.js`<br>`app/app.py` |
| **Adarsh Raj**<br>`2401920139001` | **1D CNN Modeling** | Spatial Feature Extraction, 1D Convolutions, Kernel Optimization | CNN Model Architecture, Weather Extension Impact, CNN Ablation | `src/models/cnn.py`<br>`src/training/train_cnn.py`<br>`notebooks/07_cnn_model.ipynb`<br>`models/cnn/` |
| **Shivam Verma**<br>`2301920130163` | **Stacked LSTM Modeling** | Temporal Dependencies, Recurrent Gates, Cell States, Diurnal Cycles | LSTM Architecture, Sequential Learning, Temporal Memory Analysis | `src/models/lstm.py`<br>`src/training/train_lstm.py`<br>`notebooks/08_lstm_model.ipynb`<br>`models/lstm/` |
| **Sushant Prasad**<br>`2301920130192` | **Hybrid CNN-LSTM & Project Lead** | Spatial-to-Temporal Fusion, Pipeline, On-Demand Predictor, Early Warning | Hybrid CNN-LSTM Model, Prediction Engine, Alert Module, Statistical Tests | `src/models/cnn_lstm.py`<br>`src/preprocessing/`<br>`src/prediction/`<br>`predict_any_datetime.py` |

---

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                  PROJECT ARCHITECTURE                  │
                                  └──────────────────────────┬─────────────────────────────┘
                                                             │
                    ┌─────────────────────────┬──────────────┴───────────────┬─────────────────────────┐
                    │                         │                              │                         │
                    ▼                         ▼                              ▼                         ▼
         ┌─────────────────────┐   ┌─────────────────────┐        ┌─────────────────────┐   ┌─────────────────────┐
         │     HARSHIT GIRI    │   │      ADARSH RAJ     │        │     SHIVAM VERMA    │   │    SUSHANT PRASAD   │
         │  Frontend & Web UI  │   │     1D CNN Model    │        │  Stacked LSTM Model │   │ Hybrid CNN-LSTM & Lead│
         ├─────────────────────┤   ├─────────────────────┤        ├─────────────────────┤   ├─────────────────────┤
         │ • Flask Dashboard   │   │ • 1D Convolutions   │        │ • Recurrent Gates   │   │ • CNN-LSTM Cascading│
         │ • 2.5s Auto-Polling │   │ • Feature Maps      │        │ • Cell State Memory │   │ • CPCB AQI Engine   │
         │ • Responsive CSS    │   │ • Conv1D + MaxPool  │        │ • Diurnal Lag Cycles│   │ • 82-Feature Pipeline│
         │ • Live Prediction   │   │ • Batch Normaliz.   │        │ • Vanishing Gradient│   │ • 3-Day Ensemble    │
         │ • Alert Banners     │   │ • Weather Extension │        │ • Sequential Memory │   │ • Early Warning (90%)│
         │ • REST API Client   │   │ • CNN Ablation Step │        │ • LSTM Ablation Step│   │ • Statistical Tests │
         └─────────────────────┘   └─────────────────────┘        └─────────────────────┘   └─────────────────────┘
```

---

## 1. 🎨 Harshit Giri — Frontend & Interactive Web Dashboard

### 1.1 Scope of Responsibility
Harshit owns the **entire presentation layer**, user experience, and real-time frontend integration of the project. He is responsible for explaining how end users, evaluators, and environmental analysts interact with the deep learning models via the browser interface.

### 1.2 Core Theoretical & Engineering Concepts to Master
1. **Dynamic Real-Time Prediction Polling:**
   - How `app/static/js/script.js` polls `/api/latest-prediction` every 2,500 ms (2.5 seconds) using asynchronous `fetch()`.
   - How the timestamp comparison ensures the DOM is only re-rendered when a fresh prediction is triggered from PowerShell / Terminal.
   - Smooth glow animations (`ring-emerald-500/50`, `animate-pulse`) and toast notifications when new results arrive.
2. **CPCB AQI Category Visual Representation:**
   - 6 color-coded alert classes strictly conforming to Central Pollution Control Board (CPCB) standards:
     - 🟢 **Good** (0–50): `#00E400`
     - 🟢 **Satisfactory** (51–100): `#92D050`
     - 🟡 **Moderate** (101–200): `#FFFF00`
     - 🟠 **Poor** (201–300): `#FF7E00`
     - 🔴 **Very Poor** (301–400): `#FF0000`
     - 🟤 **Severe** (401–500+): `#7E0023`
3. **Responsive UI/UX Layout:**
   - Hero Live Prediction container (`#live-prediction`) at the very top of the page.
   - Real-time AQI prediction card with category badge, confidence score, and risk level.
   - Soft-voting class probability distribution bars (0% to 100%).
4. **Flask Jinja2 Integration:**
   - How data is passed from Python dictionaries in `app/app.py` into `app/templates/index.html`.
   - Handling loading spinners, offline state fallbacks, and REST endpoint triggers (`/api/run-prediction`).

### 1.3 Exact Files & Folders Harshit Must Study

| Path | Purpose & What to Look For |
|:---|:---|
| [`app/templates/index.html`](../app/templates/index.html) | Master HTML layout, Hero `#live-prediction` card, model comparison cards, ablation study charts, alert modals, and Jinja2 syntax. |
| [`app/static/css/style.css`](../app/static/css/style.css) | Custom styling, dark mode glassmorphism effects, status badge color variables, responsive grid breakpoints (`@media`). |
| [`app/static/js/script.js`](../app/static/js/script.js) | Full client-side JavaScript: `pollLatestPrediction()`, `renderPredictionHero()`, `renderProbabilityBars()`, toast notification trigger, event listeners. |
| [`app/app.py`](../app/app.py) | Backend Flask routing: `@app.route('/')`, `@app.route('/api/latest-prediction')`, `@app.route('/api/run-prediction')`, `@app.route('/api/predict')`. |
| [`results/latest_prediction.json`](../results/latest_prediction.json) | The exact schema generated by CLI commands and consumed by the frontend (`status`, `timestamp`, `overall_aqi`, `hourly_timeline`, `probabilities`). |
| [`docs/api_documentation.md`](api_documentation.md) | REST API endpoints, request payloads, response schemas, HTTP status codes. |

### 1.4 How to Run and Demonstrate Harshit's Work
```bash
# 1. Start the Flask server
python app/app.py

# 2. Open browser at http://localhost:5000
# 3. In a separate terminal, run an on-demand prediction:
python predict_any_datetime.py "2026-11-26 12:00"

# 4. Watch the top of the browser page automatically detect and display the new prediction without page reload!
```

### 1.5 Likely Viva Questions for Harshit
1. *How does the frontend know that a prediction script was executed in the terminal without a manual browser refresh?*
   - **Answer:** The frontend runs an asynchronous polling loop (`setInterval` every 2.5s) hitting `/api/latest-prediction`. When the payload's `timestamp` differs from the cached timestamp, it triggers a non-destructive DOM update with a visual pulse badge.
2. *How is mobile responsiveness achieved across smartphones, tablets, and desktops?*
   - **Answer:** Through responsive CSS flexbox and grid layouts in `style.css` combined with standard media queries (`@media (max-width: 768px)`).
3. *How are the multi-class model probabilities visualized on the dashboard?*
   - **Answer:** Normalized softmax probabilities (summing to 1.0) are rendered as percentage width bars styled with the corresponding CPCB category color.

---

## 2. 🔬 Adarsh Raj — 1D CNN Architecture & Spatial Feature Extraction

### 2.1 Scope of Responsibility
Adarsh owns the **1D Convolutional Neural Network (CNN)** branch of the research. He is responsible for explaining how convolutional kernels extract localized spatial relationships across multi-variate pollutant and meteorological feature vectors over a sliding 24-hour temporal window.

### 2.2 Core Theoretical & Engineering Concepts to Master
1. **Mathematical Mechanics of 1D Convolution:**
   - Input tensor shape: $(N, 24, 82)$ where $N = \text{batch size}$, $24 = \text{lookback hours}$, $82 = \text{features}$.
   - The 1D convolution operation slides a kernel of size $k$ across the time dimension:
     $$y[t] = \sum_{i=1}^{82} \sum_{k=0}^{K-1} X[t+k, i] \cdot W[k, i] + b$$
   - Kernel sizes used: $k=3$ (capturing short-term 3-hour micro-trends in emissions).
2. **CNN Layer Architecture:**
   - **Conv1D Layer 1:** 64 filters, kernel size 3, ReLU activation.
   - **Batch Normalization 1:** Normalizes feature activations to prevent internal covariate shift and accelerate convergence.
   - **MaxPooling1D 1:** Pool size 2 (reduces sequence length from 24 to 12).
   - **Conv1D Layer 2:** 128 filters, kernel size 3, ReLU activation.
   - **Batch Normalization 2:** Stabilizes 128-dimensional latent feature maps.
   - **MaxPooling1D 2:** Pool size 2 (reduces sequence length from 12 to 6).
   - **Flatten & Dense Layers:** Flatten to $(N, 768)$ $\rightarrow$ Dense(128, ReLU) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(6, Softmax).
3. **Key Findings for CNN:**
   - **Test Accuracy:** 77.82% | **Macro Precision:** 0.7719 | **Macro Recall:** 0.7483 | **Macro F1-Score:** 0.7578.
   - **Extension 1 (Meteorological Impact):** CNN was the *only* model that gained measurable accuracy (+0.93%) from the inclusion of raw weather features (temperature, humidity, wind speed, pressure, dew point), demonstrating CNN's ability to act as an effective spatial filter over heterogeneous physical variables.

### 2.3 Exact Files & Folders Adarsh Must Study

| Path | Purpose & What to Look For |
|:---|:---|
| [`src/models/cnn.py`](../src/models/cnn.py) | Definition of `build_cnn_model()`: Conv1D, BatchNormalization, MaxPooling1D, Dropout, Dense softmax classification head. |
| [`src/training/train_cnn.py`](../src/training/train_cnn.py) | Standalone CNN training pipeline, Adam optimizer ($lr=0.001$), EarlyStopping, ReduceLROnPlateau. |
| [`notebooks/07_cnn_model.ipynb`](../notebooks/07_cnn_model.ipynb) | Interactive notebook for CNN development, layer parameter breakdown, training curves, classification report. |
| [`models/cnn/best_model.keras`](../models/cnn/best_model.keras) | Pre-trained CNN model weights (2.9 MB). |
| [`models/cnn_pollutant/`](../models/cnn_pollutant/) & [`models/cnn_weather/`](../models/cnn_weather/) | Checkpoints used in the meteorological feature impact study. |
| [`results/plots/cm_cnn.png`](../results/plots/cm_cnn.png) | Normalized confusion matrix for CNN on 8,941 test samples. |
| [`results/metrics/cnn_history.json`](../results/metrics/cnn_history.json) | Epoch-by-epoch loss, val_loss, accuracy, val_accuracy. |
| [`tests/test_models.py`](../tests/test_models.py) | `TestCNNModel` test suite verifying tensor output shapes and parameter counts. |
| [`docs/SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md`](SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md) | Section 6: Full mathematical formulation of 1D CNN for time-series feature extraction. |

### 2.4 How to Run and Demonstrate Adarsh's Work
```bash
# 1. Train the standalone 1D CNN model:
python src/training/train_cnn.py

# 2. Run the meteorological comparison experiment for CNN:
python run_meteorological_comparison.py

# 3. Run unit tests verifying CNN architecture:
python -m unittest tests.test_models.TestCNNModel -v
```

### 2.5 Likely Viva Questions for Adarsh
1. *Why use 1D Convolution instead of 2D Convolution for air quality data?*
   - **Answer:** 2D convolution is designed for image matrices where height and width represent spatial Euclidean dimensions. In multi-variate time-series, the spatial dimension is 1-dimensional (time steps $t=1 \dots 24$). 1D convolution slides along time while convolving across all 82 input feature channels simultaneously.
2. *What is the role of Batch Normalization in your CNN architecture?*
   - **Answer:** It zero-centers and normalizes each mini-batch feature map ($\mu=0, \sigma^2=1$), mitigating vanishing/exploding gradients and allowing higher learning rates.
3. *Why did CNN benefit from weather features (+0.93%) whereas recurrent models did not?*
   - **Answer:** CNN acts as a localized feature filter that extracts joint non-linear relationships (e.g. high humidity + low wind speed traps particulate matter) without being disrupted by long-term seasonal noise.

---

## 3. ⏱️ Shivam Verma — Stacked LSTM Architecture & Temporal Sequence Modeling

### 3.1 Scope of Responsibility
Shivam owns the **Stacked Long Short-Term Memory (LSTM)** recurrent network branch of the research. He is responsible for explaining how recurrent memory cells, gating mechanisms, and hidden state transitions model temporal autocorrelation and diurnal smog cycles across Delhi's air quality data.

### 3.2 Core Theoretical & Engineering Concepts to Master
1. **Mathematical Mechanics of LSTM Gating Units:**
   - For an input vector $x_t$ and previous hidden state $h_{t-1}$:
     - **Forget Gate ($f_t$):** Controls what proportion of past cell memory to discard:
       $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
     - **Input Gate ($i_t$):** Regulates how much new incoming information enters the cell:
       $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
     - **Candidate Cell State ($\tilde{C}_t$):** Creates candidate values via $\tanh$:
       $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
     - **Cell State Update ($C_t$):** Linear combination of preserved memory and new candidate:
       $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
     - **Output Gate ($o_t$):** Determines what proportion of cell memory to output as hidden state:
       $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
     - **Hidden State ($h_t$):** Filtered cell state passed to next step:
       $$h_t = o_t \odot \tanh(C_t)$$
2. **Stacked LSTM Architecture:**
   - **LSTM Layer 1:** 64 units, `return_sequences=True` (maintains sequence dimension $(N, 24, 64)$ to feed sequential representations into the second recurrent layer).
   - **Dropout 1:** 0.2 recurrent dropout to prevent overfitting on specific sequence trajectories.
   - **LSTM Layer 2:** 32 units, `return_sequences=False` (aggregates the 24-step sequence into a single 32-dimensional context vector $(N, 32)$).
   - **Dropout 2:** 0.3 dropout.
   - **Dense Classification Head:** Dense(64, ReLU) $\rightarrow$ Dense(6, Softmax).
3. **Key Findings for LSTM:**
   - **Test Accuracy:** 75.01% | **Macro Precision:** 0.7410 | **Macro Recall:** 0.7225 | **Macro F1-Score:** 0.7297.
   - **Temporal Memory:** LSTM demonstrated strong capability in tracking gradual day-to-night pollutant build-ups (e.g. nocturnal inversion layer causing PM2.5 spikes). However, pure LSTM was susceptible to lagging behind sudden abrupt environmental shocks (e.g. unexpected episodic burning).

### 3.3 Exact Files & Folders Shivam Must Study

| Path | Purpose & What to Look For |
|:---|:---|
| [`src/models/lstm.py`](../src/models/lstm.py) | Definition of `build_lstm_model()`: Stacked LSTM layers, `return_sequences` configuration, Dropout, Dense softmax output. |
| [`src/training/train_lstm.py`](../src/training/train_lstm.py) | Standalone LSTM training script, Adam optimizer, model checkpointing, loss history logging. |
| [`notebooks/08_lstm_model.ipynb`](../notebooks/08_lstm_model.ipynb) | Interactive notebook for LSTM training, loss curves, confusion matrix, per-class classification report. |
| [`models/lstm/best_model.keras`](../models/lstm/best_model.keras) | Pre-trained Stacked LSTM model checkpoint (2.0 MB). |
| [`models/lstm_pollutant/`](../models/lstm_pollutant/) & [`models/lstm_weather/`](../models/lstm_weather/) | Checkpoints used to analyze pure-pollutant vs pollutant+weather recurrent stability. |
| [`results/plots/cm_lstm.png`](../results/plots/cm_lstm.png) | Normalized confusion matrix for Stacked LSTM on 8,941 test samples. |
| [`results/metrics/lstm_history.json`](../results/metrics/lstm_history.json) | Loss, accuracy, val_loss, and val_accuracy across all training epochs. |
| [`tests/test_models.py`](../tests/test_models.py) | `TestLSTMModel` test suite testing output dimensions and probability distribution sums ($\sum p = 1.0$). |
| [`docs/SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md`](SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md) | Section 7: Full mathematical equations for LSTM gates and recurrent memory transfer. |

### 3.4 How to Run and Demonstrate Shivam's Work
```bash
# 1. Train the standalone Stacked LSTM model:
python src/training/train_lstm.py

# 2. View training curves and loss convergence:
python run_visualizations.py

# 3. Run unit tests verifying LSTM architecture:
python -m unittest tests.test_models.TestLSTMModel -v
```

### 3.5 Likely Viva Questions for Shivam
1. *Why did you choose LSTM over standard vanilla Recurrent Neural Networks (RNN)?*
   - **Answer:** Standard RNNs suffer from vanishing and exploding gradients when backpropagating through time (BPTT) over 24 time steps. LSTM introduces an additive constant error carousel through its Cell State ($C_t$), enabling stable gradient flow across long temporal horizons.
2. *Why is `return_sequences=True` required on the first LSTM layer but `False` on the second?*
   - **Answer:** The first layer must pass a sequence of vectors for every time step $(N, 24, 64)$ to the next recurrent layer. The second layer compresses the entire sequence into a single summary state $(N, 32)$ suitable for feeding into the dense classification layer.
3. *How does LSTM model the diurnal pattern of air pollutants?*
   - **Answer:** Through cyclic temporal feature inputs (sin/cos of hour and month) and internal cell states, the model remembers the typical morning rush hour and nocturnal drop in boundary layer height.

---

## 4. 👑 Sushant Prasad — Hybrid CNN-LSTM Model, Pipeline & Project Lead

### 4.1 Scope of Responsibility
Sushant serves as **Project Lead and System Architect**. He owns the **Hybrid CNN–LSTM Deep Learning Architecture**, the end-to-end data engineering pipeline, the 82-feature extraction suite, the 6-step ablation study, the on-demand datetime prediction system, the early-warning module, and formal statistical significance verification (McNemar's test).

### 4.2 Core Theoretical & Engineering Concepts to Master
1. **The Hybrid CNN–LSTM Architecture (State of the Art):**
   - Cascading mechanism: **Spatial feature extraction followed by temporal progression modeling**.
   - Input: $(N, 24, 82)$
   - `Conv1D(64, kernel_size=3, padding='same', activation='relu')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling1D(pool_size=2)` (Shape: $(N, 12, 64)$)
   - `Conv1D(128, kernel_size=3, padding='same', activation='relu')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling1D(pool_size=2)` (Shape: $(N, 6, 128)$)
   - `LSTM(64, return_sequences=True)` $\rightarrow$ `Dropout(0.2)`
   - `LSTM(32, return_sequences=False)` $\rightarrow$ `Dropout(0.3)`
   - `Dense(64, activation='relu')` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(6, activation='softmax')`.
   - **Benchmark Performance:** **80.64% Accuracy | 0.8059 Macro F1-Score** (Statistically superior to CNN $p < 0.001$ and LSTM $p < 0.001$ via McNemar's test).
2. **CPCB AQI Calculation Engine:**
   - Non-linear piecewise linear interpolation:
     $$I_p = \frac{I_{HI} - I_{LO}}{B_{HI} - B_{LO}} \cdot (C_p - B_{LO}) + I_{LO}$$
   - Overall AQI via maximum operator: $\text{AQI} = \max(I_{\text{PM2.5}}, I_{\text{PM10}}, I_{\text{NO2}}, I_{\text{SO2}}, I_{\text{CO}}, I_{\text{O3}})$.
3. **82-Feature Engineering Architecture:**
   - 6 Raw Pollutants + 36 Lag features (1h, 2h, 3h, 6h, 12h, 24h across 6 pollutants)
   - 24 Rolling Window statistics (3h, 6h, 12h, 24h rolling means for PM2.5, PM10, NO2, O3)
   - 8 Cyclic Temporal Encodings ($\sin/\cos$ of hour, day-of-week, day-of-year, month)
   - 3 Pollutant Interaction Ratios ($\text{PM}_{2.5}/\text{PM}_{10}$, $\text{NO}_2/\text{SO}_2$, $\text{CO}/\text{O}_3$)
   - 5 Raw Weather Variables (Temperature, Humidity, Wind Speed, Pressure, Dew Point).
4. **Extension 2: 6-Step Feature Ablation Study:**
   - Step 1 (Base 6 Pollutants): 76.54% Accuracy, 0.7601 F1
   - Step 2 (+ Lags): 78.12% Accuracy, 0.7818 F1 (+2.17% F1 gain)
   - Step 3 (+ Rolling Averages): **79.88% Accuracy, 0.7966 F1** (Largest single performance leap)
   - Steps 4–6: Refinement with cyclic encodings, ratios, and meteorological variables.
5. **Extension 3: Threshold-Based Early Warning Alert Module:**
   - Triggers automated alerts when predicted AQI reaches **Very Poor** or **Severe** ($\text{AQI} \ge 301$).
   - Performance: **93.67% Alert Precision | 87.51% Alert Recall | 90.48% Alert F1-Score**.
6. **On-Demand Live Prediction System:**
   - Single datetime inference using Hybrid CNN-LSTM with confidence score and CPCB health advisory.

### 4.3 Exact Files & Folders Sushant Must Study

| Path | Purpose & What to Look For |
|:---|:---|
| [`src/models/cnn_lstm.py`](../src/models/cnn_lstm.py) | Hybrid architecture: Cascaded Conv1D, BatchNorm, MaxPooling, LSTM, and Dense layers. |
| [`src/training/train_cnn_lstm.py`](../src/training/train_cnn_lstm.py) | Training loop, learning rate scheduler, checkpointing, early stopping. |
| [`src/preprocessing/`](../src/preprocessing/) | `aqi_calculator.py` (CPCB breakpoints), `feature_engineering.py` (82 features), `scaler.py` (Z-score), `sequence_generator.py` (sliding windows). |
| [`src/prediction/`](../src/prediction/) | `predict_any_datetime.py` (datetime prediction runner), `early_warning.py` (alert module), `health_risk.py` (guidance). |
| [`predict_any_datetime.py`](../predict_any_datetime.py) | Main user-facing inference CLI script. |
| [`run_pipeline.py`](../run_pipeline.py) & [`run_training.py`](../run_training.py) | Full data preparation and training orchestration scripts. |
| [`run_ablation_study.py`](../run_ablation_study.py) & [`run_full_pipeline.py`](../run_full_pipeline.py) | 6-step ablation runner and end-to-end master pipeline runner. |
| [`src/evaluation/statistical_tests.py`](../src/evaluation/statistical_tests.py) | McNemar's non-parametric paired tests and contingency matrices. |
| [`docs/DETAILED_PROJECT_REPORT.md`](DETAILED_PROJECT_REPORT.md) | Master 26-section project report covering the complete theoretical, mathematical, and experimental foundation. |
| [`docs/SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md`](SYSTEM_ARCHITECTURE_AND_METHODOLOGY.md) | Complete system architecture, equations, and tensor flow specifications. |
| [`docs/RESULTS_AND_EVALUATION.md`](RESULTS_AND_EVALUATION.md) | Full empirical benchmarks, McNemar test results, and ablation analysis. |

### 4.4 How to Run and Demonstrate Sushant's Work
```bash
# 1. Run the full data-to-model pipeline:
python run_pipeline.py

# 2. Train the state-of-the-art Hybrid CNN-LSTM model:
python src/training/train_cnn_lstm.py

# 3. Run the 6-step feature ablation study:
python run_ablation_study.py

# 4. Predict AQI for any arbitrary datetime:
python predict_any_datetime.py "2026-11-26 12:00"

# 5. Run all 34 unit tests:
python -m unittest discover -s tests -v
```

### 4.5 Likely Viva Questions for Sushant
1. *Why does the Hybrid CNN–LSTM outperform standalone CNN and LSTM models?*
   - **Answer:** Conv1D filters first transform high-dimensional, noisy multi-pollutant inputs (82 features) into high-level spatial feature representations. The LSTM then processes the temporal progression of these distilled feature representations, effectively decoupling spatial pattern recognition from temporal trend modeling.
2. *How is statistical significance established between the models?*
   - **Answer:** Using McNemar's paired contingency test on the 8,941 test samples. The test evaluates discordant predictions between models:
     $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
     The resulting $p$-values are $< 0.001$, mathematically confirming the Hybrid model's superiority is not due to random sampling chance.
3. *What was the most important finding of the 6-step ablation study?*
   - **Answer:** Rolling window averages (Step 3) contributed the single highest performance jump (+1.5% F1-score), proving that multi-scale smoothed temporal history is more informative than instantaneous pollutant spikes alone.
4. *How does the early warning module ensure public health safety?*
   - **Answer:** It flags predictions falling into Very Poor (301–400) and Severe (401–500+) with a 93.67% precision and 87.51% recall, preventing false sense of security during toxic smog episodes.

---

## 5. 📅 Step-by-Step 4-Phase Team Study Roadmap

```
Phase 1 (Days 1–2)         Phase 2 (Days 3–5)              Phase 3 (Days 6–7)            Phase 4 (Days 8–9)
┌──────────────────────┐   ┌───────────────────────────┐   ┌─────────────────────────┐   ┌────────────────────────┐
│  FOUNDATIONAL STUDY  │   │  CORE MODULE DEEP DIVE    │   │ CROSS-MODULE INTEGRATION│   │ MOCK VIVA & REHEARSAL  │
├──────────────────────┤   ├───────────────────────────┤   ├─────────────────────────┤   ├────────────────────────┤
│ • Read README.md     │   │ • Harshit: index.html, JS │   │ • Test live UI polling  │   │ • Cross-questioning    │
│ • Read User Guide    │   │ • Adarsh: cnn.py, Conv1D  │   │ • Validate ensemble     │   │ • Verify formulas      │
│ • Run app/app.py     │   │ • Shivam: lstm.py, gates  │   │ • Verify McNemar tests  │   │ • Slide preparation    │
│ • Pass 34 unit tests │   │ • Sushant: cnn_lstm.py    │   │ • Check ablation output │   │ • Code walkthroughs    │
└──────────────────────┘   └───────────────────────────┘   └─────────────────────────┘   └────────────────────────┘
```

### Phase 1: Foundational Orientation (All 4 Members)
1. Read the root [`README.md`](../README.md) and [`docs/PROJECT_USER_GUIDE.md`](PROJECT_USER_GUIDE.md).
2. Clone and run the project locally; run the test suite (`python -m unittest discover -s tests -v`) to confirm 34/34 tests pass.
3. Launch the dashboard (`python app/app.py`) and inspect each UI component.

### Phase 2: Domain Specialization (Individual Work)
1. **Harshit:** Master the polling mechanism in `script.js` and trace how `latest_prediction.json` updates the HTML DOM.
2. **Adarsh:** Master 1D convolution equations, study `notebooks/07_cnn_model.ipynb`, and analyze `results/plots/cm_cnn.png`.
3. **Shivam:** Master LSTM gate equations, study `notebooks/08_lstm_model.ipynb`, and analyze `results/plots/cm_lstm.png`.
4. **Sushant:** Trace `run_full_pipeline.py`, study the 82 features in `feature_engineering.py`, and analyze McNemar's test script.

### Phase 3: Integration & Cross-Verification (Team Collaboration)
1. Harshit and Sushant verify that `predict_any_datetime.py` writes properly to `results/latest_prediction.json` and reflects on the dashboard.
2. Adarsh and Shivam compare CNN vs LSTM confusion matrices (`results/plots/cm_cnn.png` vs `results/plots/cm_lstm.png`) with Sushant's Hybrid matrix (`cm_hybrid.png`).
3. Sushant conducts a full dry run of the 6-step ablation study and meteorological comparison scripts.

### Phase 4: Mock Viva Voce & Presentation Defense
1. The team conducts internal peer mock interviews using the questions outlined in Sections 1.5, 2.5, 3.5, and 4.5.
2. Each member prepares their specific slide section for the final academic presentation.

---

## 6. 📂 Quick File-to-Member Navigation Index

```
AQI-Prediction-CNN-LSTM/
│
├── app/                                ──► [HARSHIT GIRI]
│   ├── app.py                          ──► [HARSHIT GIRI & SUSHANT PRASAD]
│   ├── templates/index.html            ──► [HARSHIT GIRI]
│   └── static/css/style.css, js/       ──► [HARSHIT GIRI]
│
├── src/
│   ├── models/
│   │   ├── cnn.py                      ──► [ADARSH RAJ]
│   │   ├── lstm.py                     ──► [SHIVAM VERMA]
│   │   └── cnn_lstm.py                 ──► [SUSHANT PRASAD]
│   ├── training/
│   │   ├── train_cnn.py                ──► [ADARSH RAJ]
│   │   ├── train_lstm.py               ──► [SHIVAM VERMA]
│   │   └── train_cnn_lstm.py           ──► [SUSHANT PRASAD]
│   ├── preprocessing/                  ──► [SUSHANT PRASAD]
│   ├── prediction/                     ──► [SUSHANT PRASAD]
│   ├── data/                           ──► [SUSHANT PRASAD]
│   └── evaluation/                     ──► [SUSHANT PRASAD, ADARSH RAJ, SHIVAM VERMA]
│
├── notebooks/
│   ├── 07_cnn_model.ipynb              ──► [ADARSH RAJ]
│   ├── 08_lstm_model.ipynb             ──► [SHIVAM VERMA]
│   ├── 09_cnn_lstm_model.ipynb         ──► [SUSHANT PRASAD]
│   ├── 10_ablation_study.ipynb         ──► [SUSHANT PRASAD & TEAM]
│   ├── 11_meteorological_ext.ipynb     ──► [ADARSH RAJ & SHIVAM VERMA]
│   ├── 12_model_evaluation.ipynb       ──► [SUSHANT PRASAD & TEAM]
│   └── 13_early_warning.ipynb          ──► [SUSHANT PRASAD]
│
├── results/
│   ├── plots/cm_cnn.png                ──► [ADARSH RAJ]
│   ├── plots/cm_lstm.png               ──► [SHIVAM VERMA]
│   ├── plots/cm_hybrid.png             ──► [SUSHANT PRASAD]
│   ├── ablation/                       ──► [SUSHANT PRASAD]
│   ├── meteorological/                 ──► [ADARSH RAJ & SHIVAM VERMA]
│   └── latest_prediction.json          ──► [HARSHIT GIRI & SUSHANT PRASAD]
│
├── docs/                               ──► [ALL MEMBERS]
│   ├── PROJECT_USER_GUIDE.md           ──► [ALL MEMBERS]
│   ├── DETAILED_PROJECT_REPORT.md      ──► [ALL MEMBERS]
│   ├── SYSTEM_ARCHITECTURE_...md       ──► [SUSHANT, ADARSH, SHIVAM]
│   ├── RESULTS_AND_EVALUATION.md       ──► [ALL MEMBERS]
│   └── api_documentation.md            ──► [HARSHIT GIRI & SUSHANT PRASAD]
│
└── tests/                              ──► [ALL MEMBERS (34 Tests)]
```

---
*Created for academic project evaluation and viva-voce preparation — Department of Computer Science & Information Technology, G.L. Bajaj Institute of Technology and Management.*
