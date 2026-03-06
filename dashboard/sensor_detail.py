"""
Sensor Detail Analysis Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("dataset", "sensor_data.csv")

SENSOR_META = {
    "vibration":     {"label": "Vibration",    "unit": "mm/s", "color": "#4f8ef7", "warn": 3.0,  "crit": 6.0},
    "temperature":   {"label": "Temperature",  "unit": "°C",   "color": "#ff9600", "warn": 72.0, "crit": 90.0},
    "sound_db":      {"label": "Acoustic dB",  "unit": "dB",   "color": "#a78bfa", "warn": 78.0, "crit": 90.0},
    "load_pct":      {"label": "Load Current", "unit": "%",    "color": "#34d399", "warn": 65.0, "crit": 80.0},
    "oil_pressure":  {"label": "Oil Pressure", "unit": "bar",  "color": "#f59e0b", "warn": 3.5,  "crit": 2.5},
    "rpm_drift_pct": {"label": "RPM Drift",    "unit": "%",    "color": "#ec4899", "warn": 5.0,  "crit": 8.0},
}

COND_COLORS = {0: "#00dc82", 1: "#ffdc00", 2: "#ff3c3c"}
COND_LABELS = {0: "Normal", 1: "Early Fault", 2: "Critical"}


@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def render():
    st.markdown("""
    <div class='hero-title'>📊 Sensor Deep Dive Analysis</div>
    <p class='hero-sub'>Explore sensor trends, distributions, and cross-correlations</p>
    <hr style='border-color:#2a2a4a; margin:10px 0 20px;'>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.warning("Dataset not found. Run `python dataset/generate_data.py` first.")
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        machine = st.selectbox("Select Machine", sorted(df["machine_id"].unique()))
    with col_ctrl2:
        n_points = st.slider("Data Points", 100, 1000, 300, step=50)
    with col_ctrl3:
        sensors_selected = st.multiselect(
            "Sensors to Plot",
            options=list(SENSOR_META.keys()),
            default=["vibration", "temperature", "sound_db"],
            format_func=lambda k: SENSOR_META[k]["label"]
        )

    mdf = df[df["machine_id"] == machine].tail(n_points).copy()
    mdf = mdf.reset_index(drop=True)
    mdf["x"] = range(len(mdf))

    if not sensors_selected:
        st.info("Select at least one sensor above.")
        return

    # ── Multi-sensor time series ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>Sensor Time Series</div>", unsafe_allow_html=True)

    fig = make_subplots(
        rows=len(sensors_selected), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[SENSOR_META[s]["label"] + f" ({SENSOR_META[s]['unit']})" for s in sensors_selected]
    )

    for i, sensor in enumerate(sensors_selected, start=1):
        meta = SENSOR_META[sensor]
        fig.add_trace(go.Scatter(
            x=mdf["x"], y=mdf[sensor],
            mode="lines",
            name=meta["label"],
            line=dict(color=meta["color"], width=1.8),
            fill="tozeroy",
            fillcolor=meta["color"],
            opacity=0.15,
        ), row=i, col=1)

        # Warning / Critical lines
        fig.add_hline(y=meta["warn"], line_dash="dot",  line_color="#ffdc00", opacity=0.6, row=i, col=1)
        fig.add_hline(y=meta["crit"], line_dash="dash", line_color="#ff3c3c", opacity=0.6, row=i, col=1)

        # Color bad zones
        critical_idx = mdf[mdf[sensor] > meta["crit"]].index
        for idx in critical_idx:
            fig.add_vrect(x0=idx-0.5, x1=idx+0.5,
                          fillcolor="rgba(255,60,60,0.12)", line_width=0, row=i, col=1)

    fig.update_layout(
        height=180 * len(sensors_selected),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,30,0.6)",
        font=dict(color="#aabbdd"),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(gridcolor="#1a2040", zeroline=False)
    fig.update_yaxes(gridcolor="#1a2040", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    col_heat, col_dist = st.columns(2)

    with col_heat:
        st.markdown("<div class='section-title'>Sensor Correlation Matrix</div>", unsafe_allow_html=True)
        corr_cols = list(SENSOR_META.keys())
        corr = mdf[corr_cols].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Corr"),
        )
        fig_heat.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aabbdd"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_dist:
        st.markdown("<div class='section-title'>Condition Distribution</div>", unsafe_allow_html=True)
        dist_sensor = st.selectbox("Sensor for Distribution", sensors_selected, key="dist_sensor",
                                   format_func=lambda k: SENSOR_META[k]["label"])
        fig_box = go.Figure()
        for cond_id, cond_label in COND_LABELS.items():
            sub = df[df["condition"] == cond_id][dist_sensor]
            fig_box.add_trace(go.Box(
                y=sub, name=cond_label,
                marker_color=COND_COLORS[cond_id],
                boxmean=True,
            ))
        fig_box.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,30,0.6)",
            font=dict(color="#aabbdd"),
            xaxis=dict(gridcolor="#1a2040"),
            yaxis=dict(gridcolor="#1a2040"),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color="#aabbdd")),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Scatter: Vibration vs Temperature ────────────────────────────────────
    st.markdown("<div class='section-title'>Vibration vs Temperature (All Machines, by Condition)</div>",
                unsafe_allow_html=True)
    fig_sc = px.scatter(
        df.sample(1500, random_state=42),
        x="vibration", y="temperature",
        color="label",
        color_discrete_map={"Normal":"#00dc82", "EarlyFault":"#ffdc00", "Critical":"#ff3c3c"},
        opacity=0.65,
        size_max=6,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig_sc.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,30,0.7)",
        font=dict(color="#aabbdd"),
        legend=dict(title="Condition", font=dict(color="#aabbdd")),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
