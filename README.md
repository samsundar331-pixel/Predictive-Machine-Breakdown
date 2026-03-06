<div align="center">

# 🏭 SMARTPREDICT 2.0
https://predictive-machine-breakdown.streamlit.app
### Zero-Surprise Predictive Maintenance System for CNC Machines

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Predict failures before they happen. Reduce downtime. Save cost.**

</div>

---

## 🎯 Problem Statement

> *"Unexpected machine breakdowns increase downtime and maintenance costs."*

Industrial CNC machines fail unexpectedly, causing:
- **Unplanned downtime** costing ₹15,000–₹2,00,000/hour
- **Reactive maintenance** — wait for failure, then repair
- **No visibility** into which machine will fail next

**SMARTPREDICT 2.0** solves this with real-time AI-powered predictions, giving your team **48–72 hours advance warning** before a failure occurs.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate IoT sensor dataset (19,020 rows, 15 machines)
python dataset/generate_data.py

# 3. Train hybrid AI model (~2.5 minutes)
python model/train_pipeline.py

# 4. Launch dashboard
streamlit run app.py
```

**Open: https://predictive-machine-breakdown.streamlit.app** → Dashboard is live!

---

## 📦 Pretrained Model

The trained Random Forest model is hosted on Hugging Face.

Download it here:

https://huggingface.co/monish-73/predictive-machine-breakdown-rf

Place the downloaded file in:

```
model/rf_classifier.pkl
```


## 🏗️ System Architecture

<div align="center">
<img src="docs/images/architecture.png" alt="System Architecture" width="92%"/>
</div>

### Layer Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **IoT Simulation** | Python NumPy | Generate realistic sensor streams for 15 CNC machines |
| **Feature Engineering** | Pandas | 7 raw + 3 derived = 10 features per reading |
| **Hybrid AI Engine** | scikit-learn Ensemble | Classify condition + detect anomalies + predict RUL |
| **Alert Engine** | Rule + ML Fusion | Physics-based override + model prediction |
| **Dashboard** | Streamlit + Plotly | Real-time fleet monitoring UI |

---

## 🤖 AI Model Pipeline

<div align="center">
<img src="docs/images/ml_pipeline.png" alt="ML Training Pipeline" width="95%"/>
</div>

### Hybrid Voting Ensemble (The Core Innovation)

```
         Random Forest (200 trees, w=2)   ─┐
    +    Gradient Boosting (150 trees, w=3) ├──► Soft-Voting Ensemble ──► Final Prediction
    +    Extra-Trees (200 trees, w=2)      ─┘        (94.1% accuracy)

    +    Isolation Forest (unsupervised)             ──► Anomaly Flag
    +    GBM Regressor                               ──► RUL (hours)
    +    Physics-based multi-sensor override         ──► Safety layer
```

Why three classifiers?
- **Random Forest**: Robust baseline, handles outlier spikes
- **Gradient Boosting**: Learns from boundary mistakes iteratively (highest weight)
- **Extra-Trees**: Maximally randomised splits = low prediction variance

> **Final vote** = weighted average of probability outputs from all 3 models

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ✅ Hybrid Ensemble Accuracy | **94.1%** |
| ✅ 5-Fold Cross-Validation | **94.1% ± 0.29%** |
| 🔍 Anomaly Detection (Isolation Forest) | 31.3% flagged |
| ⏱️ RUL Prediction MAE | 217.5 hours |
| ⏱️ RUL R² Score | 0.808 |
| 📦 Dataset Size | 19,020 rows × 15 machines |
| ⏳ Training Time | ~2.5 minutes |

### Confusion Matrix

<img src="docs/images/confusion_matrix.png" alt="Confusion Matrix" width="55%"/>

The confusion matrix shows:
- **Diagonal values >> off-diagonal** (correct predictions dominate)
- Misclassifications only appear between **adjacent classes** (Normal↔EarlyFault, EarlyFault↔Critical)
- **No direct Normal→Critical misclassifications** (model understands severity ordering)

### Feature Importance

<img src="docs/images/feature_importance.png" alt="Feature Importance" width="70%"/>

### RUL Prediction Scatter

<img src="docs/images/rul_prediction.png" alt="RUL Prediction" width="55%"/>

---

## 📡 Sensors Simulated

| Sensor | Unit | Normal Range | Critical Threshold |
|--------|------|-------------|-------------------|
| **Vibration** | mm/s | 0.8 – 2.0 | > 5.0 |
| **Temperature** | °C | 50 – 65 | > 85 |
| **Acoustic / Sound** | dB | 62 – 72 | > 88 |
| **Load / Current** | % | 40 – 55 | > 78 |
| **Oil Pressure** | bar | 4.0 – 4.8 | < 2.6 |
| **RPM Drift** | % | -1 – 2 | > 8 |
| **Runtime Hours** | hrs | 0 – 500 | > 1000 |

> **Engineered features**: `vib/temp ratio`, `thermal stress (T × runtime/1000)`, `load × RPM product`

---

## 🖥️ Dashboard Pages

### 🏭 Fleet Overview — Real-Time Machine Health
<img src="docs/images/fleet_overview.png" alt="Fleet Overview" width="100%"/>

- **15 machine cards** in 3-column grid with risk badges and failure probability bars
- Fleet health donut chart (Normal / Warning / High / Critical)
- KPI row: Total → Healthy → Warning → High Risk → Critical

### 📡 Live Sensor Gauges + AI Explainability
<img src="docs/images/fleet_gauges.png" alt="Sensor Gauges" width="100%"/>

- Real-time gauges for the most critical machine
- AI root-cause analysis panel explaining *why* the alert triggered

---

### 📊 Sensor Deep Dive Analysis
<img src="docs/images/sensor_analysis.png" alt="Sensor Analysis" width="100%"/>

- Select any machine + time window
- Multi-sensor time-series overlay chart
- Sensor correlation heatmap
- Condition distribution box plots
- Vibration vs. Temperature scatter with condition colouring

---

### 🧠 Model Insights & Performance
<img src="docs/images/model_insights.png" alt="Model Insights Top" width="100%"/>
<img src="docs/images/model_charts.png" alt="Model Insights Charts" width="100%"/>

- Accuracy + CV metrics summary cards
- **Hybrid Ensemble Architecture** table with component breakdown
- Training pipeline step-by-step
- Confusion matrix, feature importance bar chart
- Interactive RUL estimator (adjust sensor sliders → instant prediction)

---

### 🎮 Demo Mode — Simulate Machine Failure
<img src="docs/images/demo_mode.png" alt="Demo Mode" width="100%"/>

- Click **"🚨 Simulate Machine Failure"** button
- Watch 6 sensor charts escalate: **Normal → Early Fault → Critical** in real-time
- AI generates root-cause explanation at each stage
- Emergency maintenance recommendation appears at Critical phase

---

## 📁 Project Structure

```
Machine_breakdown/
│
├── 📂 dataset/
│   ├── generate_data.py           ← IoT sensor simulator (19,020 rows, 15 machines)
│   └── sensor_data.csv            ← Generated dataset (auto-created)
│
├── 📂 model/
│   ├── train_pipeline.py          ← Hybrid Ensemble training pipeline
│   ├── rf_classifier.pkl          ← Trained VotingClassifier (RF+GBM+ExtraTrees)
│   ├── scaler.pkl                 ← StandardScaler
│   ├── isolation_forest.pkl       ← Anomaly detector
│   ├── rul_regressor.pkl          ← GBM RUL regressor
│   ├── feature_cols.pkl           ← Feature column list
│   ├── metrics.json               ← Performance metrics
│   ├── confusion_matrix.png       ← Training confusion matrix
│   ├── feature_importance.png     ← RF feature importance
│   └── rul_prediction.png         ← RUL actual vs predicted scatter
│
├── 📂 alerts/
│   └── alert_engine.py            ← Alert logic: ML + physics-based override
│
├── 📂 dashboard/
│   ├── overview.py                ← Fleet Overview page
│   ├── sensor_detail.py           ← Sensor Analysis page
│   ├── model_insights.py          ← Model Insights page
│   └── demo_mode.py               ← Demo Mode with failure simulation
│
├── 📂 docs/
│   └── images/                    ← All README images
│
├── app.py                         ← Streamlit main entry point
├── requirements.txt
└── README.md
```

---

## 🔮 Alert Decision Logic

```python
# Dual-layer decision: Physics safety net + ML model
if (vibration > 5.0) + (temperature > 85°C) + (sound > 88 dB)
   + (load > 78%) + (oil < 2.6 bar) + (rpm_drift > 8%) >= 3:
    risk = "CRITICAL"   # Physics override — no model needed
elif model.predict(X) == "Critical":
    risk = "CRITICAL"   # Model confirms
elif model.predict(X) == "EarlyFault":
    risk = "HIGH" or "WARNING"   # Severity from fail probability
else:
    risk = "NORMAL" (unless RUL < 24hrs or anomaly detected)
```

---
## 🛠️ Requirements

```
streamlit>=1.36.0
pandas>=2.2.0
numpy==2.3.2
scikit-learn==1.5.2
scipy>=1.11.4
joblib>=1.3.2
plotly>=5.20.0
seaborn>=0.13.2
matplotlib>=3.8.4
huggingface_hub
altair>=4.2.2
```

Install: `pip install -r requirements.txt`

---

<div align="center">

**Built for the Hackathon 2026 · SMARTPREDICT 2.0**

*"Predict before it breaks. Fix before it costs."*

</div>
