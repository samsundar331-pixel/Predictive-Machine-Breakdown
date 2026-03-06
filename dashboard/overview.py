"""
Fleet Overview Page — Machine Health Status for All Machines
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alerts.alert_engine import predict_and_alert, THRESHOLDS

# ── Live fleet snapshot (15 machines, conditions randomly scattered) ──────────
# Defective machines are NOT grouped at the end — they appear at random positions
# Fleet truth: 11 Normal, 2 EarlyFault, 2 Critical  (randomised positions)
DEFAULT_READINGS = [
    # M01 — Normal
    {"machine_id":"M01","vibration":1.2,"temperature":57,"sound_db":65,"load_pct":44,"runtime_hrs":115,"oil_pressure":4.5,"rpm_drift_pct":0.4},
    # M02 — 🚨 CRITICAL (random position)
    {"machine_id":"M02","vibration":6.5,"temperature":92,"sound_db":94,"load_pct":84,"runtime_hrs":1280,"oil_pressure":2.2,"rpm_drift_pct":10.2},
    # M03 — Normal
    {"machine_id":"M03","vibration":1.0,"temperature":60,"sound_db":64,"load_pct":43,"runtime_hrs":310,"oil_pressure":4.6,"rpm_drift_pct":0.3},
    # M04 — ⚠️ EARLY FAULT (random position)
    {"machine_id":"M04","vibration":3.0,"temperature":74,"sound_db":80,"load_pct":64,"runtime_hrs":600,"oil_pressure":3.5,"rpm_drift_pct":4.0},
    # M05 — Normal
    {"machine_id":"M05","vibration":1.1,"temperature":56,"sound_db":66,"load_pct":45,"runtime_hrs":250,"oil_pressure":4.5,"rpm_drift_pct":0.5},
    # M06 — Normal
    {"machine_id":"M06","vibration":1.5,"temperature":59,"sound_db":65,"load_pct":47,"runtime_hrs":400,"oil_pressure":4.2,"rpm_drift_pct":1.0},
    # M07 — ⚠️ EARLY FAULT (random position)
    {"machine_id":"M07","vibration":2.8,"temperature":72,"sound_db":79,"load_pct":62,"runtime_hrs":550,"oil_pressure":3.6,"rpm_drift_pct":3.8},
    # M08 — Normal
    {"machine_id":"M08","vibration":1.2,"temperature":61,"sound_db":66,"load_pct":50,"runtime_hrs":290,"oil_pressure":4.3,"rpm_drift_pct":0.7},
    # M09 — Normal
    {"machine_id":"M09","vibration":1.6,"temperature":58,"sound_db":68,"load_pct":46,"runtime_hrs":340,"oil_pressure":4.4,"rpm_drift_pct":0.9},
    # M10 — 🚨 CRITICAL (random position)
    {"machine_id":"M10","vibration":5.8,"temperature":89,"sound_db":92,"load_pct":82,"runtime_hrs":1180,"oil_pressure":2.4,"rpm_drift_pct": 9.6},
    # M11 — Normal
    {"machine_id":"M11","vibration":1.4,"temperature":60,"sound_db":67,"load_pct":48,"runtime_hrs":160,"oil_pressure":4.3,"rpm_drift_pct":0.6},
    # M12 — Normal
    {"machine_id":"M12","vibration":1.3,"temperature":57,"sound_db":66,"load_pct":45,"runtime_hrs":210,"oil_pressure":4.5,"rpm_drift_pct":0.5},
    # M13 — Normal
    {"machine_id":"M13","vibration":1.1,"temperature":58,"sound_db":65,"load_pct":44,"runtime_hrs":175,"oil_pressure":4.4,"rpm_drift_pct":0.4},
    # M14 — Normal
    {"machine_id":"M14","vibration":1.5,"temperature":59,"sound_db":67,"load_pct":46,"runtime_hrs":320,"oil_pressure":4.3,"rpm_drift_pct":0.8},
    # M15 — Normal
    {"machine_id":"M15","vibration":1.3,"temperature":56,"sound_db":65,"load_pct":43,"runtime_hrs":140,"oil_pressure":4.5,"rpm_drift_pct":0.6},
]

RISK_COLOR = {
    "Normal":   "#00dc82",
    "Warning":  "#ffdc00",
    "High":     "#ff9600",
    "Critical": "#ff3c3c",
}

def _gauge(value, title, min_val=0, max_val=100, unit="", color="#4f8ef7"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#9ab3ff"}},
        number={"suffix": unit, "font": {"size": 22, "color": "#e0eaff"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#445566"},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(20,20,40,0.8)",
            "bordercolor": "#2a2a4a",
            "steps": [
                {"range": [min_val, max_val*0.4], "color":"rgba(0,220,130,0.08)"},
                {"range": [max_val*0.4, max_val*0.7], "color":"rgba(255,220,0,0.08)"},
                {"range": [max_val*0.7, max_val], "color":"rgba(255,60,60,0.12)"},
            ],
            "threshold": {"line": {"color": "#ff3c3c", "width": 3}, "value": max_val*0.75},
        }
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
    )
    return fig


def render():
    readings = st.session_state.get("live_readings", DEFAULT_READINGS)

    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-title'>🏭 SMARTPREDICT Fleet Overview</div>
    <p class='hero-sub'>Real-time predictive health monitoring for all CNC machines</p>
    <hr style='border-color:#2a2a4a; margin:10px 0 20px;'>
    """, unsafe_allow_html=True)

    # ── Compute alerts ───────────────────────────────────────────────────────
    try:
        alerts = [predict_and_alert(r) for r in readings]
    except Exception as e:
        st.error(f"⚠️ Models not trained yet. Run `python model/train_pipeline.py` first.\n\n{e}")
        return

    counts = {"Normal": 0, "Warning": 0, "High": 0, "Critical": 0}
    for a in alerts:
        counts[a.risk_level] = counts.get(a.risk_level, 0) + 1

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("🏭 Total Machines", len(alerts))
    with k2:
        st.metric("✅ Healthy",  counts["Normal"],   delta=None)
    with k3:
        st.metric("⚠️ Warning",  counts["Warning"],  delta=None)
    with k4:
        st.metric("🔴 High Risk", counts["High"],    delta=None)
    with k5:
        st.metric("🚨 Critical",  counts["Critical"], delta=None)

    st.markdown("---")

    # ── Fleet health donut ────────────────────────────────────────────────────
    col_donut, col_cards = st.columns([1, 2])

    with col_donut:
        st.markdown("<div class='section-title'>Fleet Health Distribution</div>", unsafe_allow_html=True)
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [RISK_COLOR[l] for l in labels]

        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#0f0f1a", width=3)),
            textfont=dict(size=13),
        ))
        fig_donut.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#aabbdd")),
            annotations=[dict(
                text=f"<b>{len(alerts)}</b><br>Machines",
                x=0.5, y=0.5,
                font_size=16,
                font_color="#c0d0ff",
                showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_cards:
        st.markdown("<div class='section-title'>Machine Status Cards <span style='color:#4455aa;font-size:0.8em;'>(15 machines)</span></div>", unsafe_allow_html=True)

        # Render 3-column grid of machine cards
        n_cols = 3
        rows_of_alerts = [alerts[i:i+n_cols] for i in range(0, len(alerts), n_cols)]
        for row in rows_of_alerts:
            cols = st.columns(n_cols)
            for col, alert in zip(cols, row):
                risk_cls  = f"alert-{alert.risk_level.lower()}"
                badge_cls = f"risk-badge-{alert.risk_level}"
                fp_pct    = alert.failure_prob * 100
                bar_color = RISK_COLOR[alert.risk_level]
                with col:
                    st.markdown(f"""
                    <div class="{risk_cls}" style="padding:12px 14px; margin:4px 0;">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:0.95em; color:#e0eaff;">🔩 {alert.machine_id}</span>
                        <span class="{badge_cls}">{alert.risk_level}</span>
                      </div>
                      <div style="color:#99aacc; font-size:0.76em; margin:6px 0 2px;">
                        <b style='color:#c0d0ff'>{alert.condition}</b>
                        &nbsp;|&nbsp; <b style='color:{bar_color}'>{fp_pct:.0f}%</b> fail prob
                      </div>
                      <div style="color:#7788aa; font-size:0.74em;">
                        RUL: <b style='color:#9ab3ff'>{alert.rul_hours:.0f} hrs</b>
                        &nbsp;{'⚠️ Anomaly' if alert.anomaly else ''}
                      </div>
                      <div style="background:rgba(0,0,0,0.3); border-radius:4px; height:5px; margin:7px 0 0;">
                        <div style="width:{min(fp_pct,100):.0f}%; height:5px; background:{bar_color};
                                    border-radius:4px; box-shadow:0 0 6px {bar_color}55;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


    # ── Sensor gauge row ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Live Sensor Gauges — Most Critical Machine</div>", unsafe_allow_html=True)
    critical_alert = max(alerts, key=lambda a: a.failure_prob)
    s = critical_alert.raw_sensors

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(_gauge(s["vibration"],   "Vibration",   0, 10,  " mm/s", "#4f8ef7"), use_container_width=True)
    with g2: st.plotly_chart(_gauge(s["temperature"],  "Temperature", 0, 130, "°C",    "#ff9600"), use_container_width=True)
    with g3: st.plotly_chart(_gauge(s["sound_db"],    "Acoustic",    40, 120, " dB",  "#a78bfa"), use_container_width=True)
    with g4: st.plotly_chart(_gauge(s["load_pct"],    "Load",        0, 100, "%",     "#34d399"), use_container_width=True)

    # ── Alert explainability ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🧠 AI Explainability — Why this alert triggered</div>", unsafe_allow_html=True)
    worst = max(alerts, key=lambda a: a.failure_prob)

    st.markdown(f"""
    <div class="alert-{worst.risk_level.lower()}">
      <b style='color:#e0eaff; font-size:1.05em;'>Machine {worst.machine_id} — Root Cause Analysis</b><br><br>
      {''.join(f'<div style="color:#c0d0ff; font-size:0.88em; padding:2px 0;">🔸 {r}</div>' for r in worst.reasons)}
      <div style='margin-top:10px; font-size:0.82em; color:#9ab3ff;'>{worst.recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Dashboard refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
