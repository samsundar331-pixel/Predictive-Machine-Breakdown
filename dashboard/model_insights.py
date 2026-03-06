"""
Model Insights Page — Feature Importance, RUL Prediction, Metrics
"""
import streamlit as st
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib

MODEL_DIR = "model"

METRIC_CARDS = {
    "Classifier Accuracy": ("accuracy", "classifier", "%", 100),
    "CV Mean Accuracy":    ("cv_mean",  "classifier", "%", 100),
    "RUL MAE":             ("mae",      "rul_regressor", " hrs", 1),
    "RUL R² Score":        ("r2",       "rul_regressor", "", 1),
}


def render():
    st.markdown("""
    <div class='hero-title'>🧠 Model Insights & Performance</div>
    <p class='hero-sub'>AI model evaluation, feature importance, and RUL analysis</p>
    <hr style='border-color:#2a2a4a; margin:10px 0 20px;'>
    """, unsafe_allow_html=True)

    # ── Load metrics ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        st.warning("⚠️ Models not trained. Run `python model/train_pipeline.py` first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Model Performance Summary</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    for col, (name, (key, group, suffix, mult)) in zip(cols, METRIC_CARDS.items()):
        val = metrics.get(group, {}).get(key, 0)
        display = f"{val * mult:.1f}{suffix}" if suffix not in ["", " hrs"] else (
            f"{val:.4f}" if suffix == "" else f"{val:.1f}{suffix}"
        )
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
              <div style='font-size:1.7em; font-weight:800; color:#4f8ef7;'>{display}</div>
              <div style='font-size:0.8em; color:#8899cc; margin-top:4px;'>{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model architecture explanation ────────────────────────────────────────
    col_arch, col_pipeline = st.columns(2)

    with col_arch:
        st.markdown("<div class='section-title'>Model Architecture</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
        <div style='font-size:0.82em; font-weight:700; color:#a78bfa; margin-bottom:8px; border-bottom:1px solid #2a3a5a; padding-bottom:5px;'>
          🔀 HYBRID VOTING ENSEMBLE CLASSIFIER (weights: 2:3:2)
        </div>
        <table width='100%' style='font-size:0.82em; color:#c0d0ff; border-collapse:collapse;'>
          <tr style='border-bottom:1px solid #2a3a5a;'>
            <th style='padding:6px; text-align:left; color:#9ab3ff;'>Component</th>
            <th style='padding:6px; color:#9ab3ff;'>Role</th>
            <th style='padding:6px; color:#9ab3ff;'>Config</th>
          </tr>
          <tr><td style='padding:5px;'>🌲 Random Forest</td>
              <td style='padding:5px; text-align:center;'>Base: robust, low variance</td>
              <td style='padding:5px; text-align:center;'>200 trees, w=2</td></tr>
          <tr><td style='padding:5px;'>⚡ Gradient Boosting</td>
              <td style='padding:5px; text-align:center;'>Boost: error-corrective</td>
              <td style='padding:5px; text-align:center;'>150 trees, lr=0.08, w=3</td></tr>
          <tr><td style='padding:5px;'>🌀 Extra-Trees</td>
              <td style='padding:5px; text-align:center;'>Randomised splits, low bias</td>
              <td style='padding:5px; text-align:center;'>200 trees, w=2</td></tr>
          <tr style='border-top:1px solid #2a3a5a;'>
              <td style='padding:5px;'>🔍 Isolation Forest</td>
              <td style='padding:5px; text-align:center;'>Anomaly detection</td>
              <td style='padding:5px; text-align:center;'>200 trees, unsupervised</td></tr>
          <tr><td style='padding:5px;'>📈 GBM Regressor</td>
              <td style='padding:5px; text-align:center;'>RUL prediction (hours)</td>
              <td style='padding:5px; text-align:center;'>200 trees, lr=0.08</td></tr>
        </table>
        <div style='font-size:0.74em; color:#6677aa; margin-top:10px;'>
          Soft-voting: P(class) = weighted avg of component probabilities.<br>
          SMOTE oversampling for class balance. 10 engineered features.
        </div>
        </div>
        """, unsafe_allow_html=True)


    with col_pipeline:
        st.markdown("<div class='section-title'>Training Pipeline Steps</div>", unsafe_allow_html=True)
        steps = [
            ("1", "Data Ingestion", "Load 5100-row IoT sensor CSV"),
            ("2", "Feature Engineering", "Ratio, stress, product features"),
            ("3", "SMOTE Resampling",  "Balance class distribution"),
            ("4", "RF Classifier",     "Train 3-class condition model"),
            ("5", "Isolation Forest",  "Unsupervised anomaly detector"),
            ("6", "RUL Regressor",     "Predict remaining life in hours"),
            ("7", "Model Persistence", "Save .pkl files with joblib"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:14px; padding:8px 0;
                        border-bottom:1px solid #1a2040;'>
              <div style='width:28px; height:28px; border-radius:50%; background:#4f8ef7;
                          display:flex; align-items:center; justify-content:center;
                          font-weight:700; font-size:0.82em; flex-shrink:0;'>{num}</div>
              <div>
                <div style='color:#c0d0ff; font-weight:600; font-size:0.88em;'>{title}</div>
                <div style='color:#6677aa; font-size:0.76em;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Confusion matrix & Feature importance images ───────────────────────
    img_col1, img_col2 = st.columns(2)

    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
    rul_path= os.path.join(MODEL_DIR, "rul_prediction.png")

    with img_col1:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Train models to generate confusion matrix.")

    with img_col2:
        st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Train models to generate feature importance plot.")

    st.markdown("---")
    st.markdown("<div class='section-title'>RUL Prediction: Actual vs Predicted</div>", unsafe_allow_html=True)
    if os.path.exists(rul_path):
        st.image(rul_path, use_container_width=False, width=600)
    else:
        st.info("Train models to generate RUL prediction plot.")

    # ── Live RUL prediction slider ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🔧 Interactive RUL Estimator</div>", unsafe_allow_html=True)
    st.markdown("Adjust sensor sliders to see predicted Remaining Useful Life:")

    s1, s2, s3, s4 = st.columns(4)
    with s1: vib  = st.slider("Vibration (mm/s)", 0.5, 10.0, 2.0, 0.1)
    with s2: temp = st.slider("Temperature (°C)",  30.0, 120.0, 65.0, 0.5)
    with s3: load = st.slider("Load (%)",          10.0, 100.0, 50.0, 1.0)
    with s4: rph  = st.slider("Runtime (hrs)",     0.0, 2000.0, 400.0, 10.0)

    reg_path = os.path.join(MODEL_DIR, "rul_regressor.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    fcols_path  = os.path.join(MODEL_DIR, "feature_cols.pkl")

    if os.path.exists(reg_path):
        reg    = joblib.load(reg_path)
        scaler = joblib.load(scaler_path)
        fcols  = joblib.load(fcols_path)

        reading = {
            "vibration": vib, "temperature": temp,
            "sound_db": 70.0, "load_pct": load,
            "runtime_hrs": rph, "oil_pressure": 4.0, "rpm_drift_pct": 2.0,
            "vib_temp_ratio": vib / (temp + 1e-3),
            "load_rpm_product": load * 2.0,
            "thermal_stress": temp * rph / 1000,
        }
        X = scaler.transform(np.array([[reading.get(f, 0) for f in fcols]]))
        rul_pred = float(reg.predict(X)[0])
        rul_days = rul_pred / 24

        col_rul_val, col_rul_msg = st.columns([1, 3])
        with col_rul_val:
            rul_color = "#00dc82" if rul_pred > 200 else "#ffdc00" if rul_pred > 48 else "#ff3c3c"
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
              <div style='font-size:2.2em; font-weight:800; color:{rul_color};'>{rul_pred:.0f}</div>
              <div style='color:#8899cc;'>Predicted RUL (hrs)</div>
              <div style='color:{rul_color}; font-size:0.85em;'>{rul_days:.1f} days</div>
            </div>
            """, unsafe_allow_html=True)
        with col_rul_msg:
            if rul_pred > 200:
                msg = "✅ Machine is healthy. No immediate maintenance required."
            elif rul_pred > 48:
                msg = "⚠️ Early signs of wear. Schedule maintenance within the next few days."
            else:
                msg = "🚨 CRITICAL: Machine is near failure. Schedule emergency maintenance NOW."
            st.info(msg)
