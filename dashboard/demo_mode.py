"""
Demo Mode Page — Simulate Machine Failure in Real-Time
Judges can click a button and watch graphs change + alerts trigger.
"""
import streamlit as st
import time, random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NORMAL_STATE = {
    "vibration":    1.1,
    "temperature":  54.0,
    "sound_db":     63.0,
    "load_pct":     42.0,
    "rpm_drift_pct":0.3,
    "oil_pressure": 4.6,
}
FAULT_STATE = {
    "vibration":    2.9,
    "temperature":  74.0,
    "sound_db":     80.0,
    "load_pct":     65.0,
    "rpm_drift_pct":4.2,
    "oil_pressure": 3.6,
}
CRITICAL_STATE = {
    "vibration":    7.8,
    "temperature":  98.0,
    "sound_db":     99.0,
    "load_pct":     90.0,
    "rpm_drift_pct":11.2,
    "oil_pressure": 2.0,
}

SENSOR_COLORS = {
    "vibration":    "#4f8ef7",
    "temperature":  "#ff9600",
    "sound_db":     "#a78bfa",
    "load_pct":     "#34d399",
    "rpm_drift_pct":"#ec4899",
    "oil_pressure": "#f59e0b",
}
SENSOR_LABELS = {
    "vibration":    "Vibration (mm/s)",
    "temperature":  "Temperature (°C)",
    "sound_db":     "Acoustic (dB)",
    "load_pct":     "Load Current (%)",
    "rpm_drift_pct":"RPM Drift (%)",
    "oil_pressure": "Oil Pressure (bar)",
}

def _add_noise(val, scale=0.05):
    return val + random.gauss(0, abs(val) * scale + 0.01)

def _lerp(a, b, t):
    """Linear interpolation between dicts a and b."""
    return {k: a[k] + (b[k] - a[k]) * t for k in a}

def _make_live_chart(history: list, state_label: str):
    """Multi-panel chart showing 6 sensors over time."""
    sensors = list(SENSOR_COLORS.keys())
    labels  = [SENSOR_LABELS[s] for s in sensors]

    fig = make_subplots(rows=3, cols=2, shared_xaxes=False,
                        vertical_spacing=0.12, horizontal_spacing=0.1,
                        subplot_titles=labels)

    for idx, sensor in enumerate(sensors):
        row = idx // 2 + 1
        col = idx % 2  + 1
        vals = [h[sensor] for h in history]
        xs   = list(range(len(vals)))
        color= SENSOR_COLORS[sensor]
        fig.add_trace(go.Scatter(
            x=xs, y=vals,
            mode="lines",
            name=SENSOR_LABELS[sensor],
            line=dict(color=color, width=2.2),
            fill="tozeroy",
            fillcolor=color,
            opacity=0.18,
            showlegend=False,
        ), row=row, col=col)

    # State-based title color
    title_colors = {"Normal":"#00dc82","Early Fault":"#ffdc00","CRITICAL FAILURE":"#ff3c3c"}
    title_color  = title_colors.get(state_label, "#4f8ef7")

    fig.update_layout(
        height=520,
        title=dict(
            text=f"🔴 LIVE SENSOR FEED — <b style='color:{title_color};'>{state_label}</b>",
            font=dict(size=15, color=title_color)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(15,15,30,0.7)",
        font=dict(color="#aabbdd"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor="#1a2040", zeroline=False, showticklabels=False)
    fig.update_yaxes(gridcolor="#1a2040", zeroline=False)
    return fig


def render():
    st.markdown("""
    <div class='hero-title'>🎮 Demo Mode — Simulate Machine Failure</div>
    <p class='hero-sub'>Watch sensors escalate in real-time as a CNC machine approaches failure</p>
    <hr style='border-color:#2a2a4a; margin:10px 0 20px;'>
    """, unsafe_allow_html=True)

    # ── Control Panel ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        speed = st.selectbox("Simulation Speed", ["Slow (Real-world)", "Normal", "Fast (Demo)"], index=1)
    with ctrl2:
        machine_id = st.selectbox("Target Machine", ["M01","M02","M03","M04","M05"], index=2)
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_sim = st.button("🚨 Simulate Machine Failure", type="primary", use_container_width=True)
    
    reset_btn = st.button("🔄 Reset to Normal State", use_container_width=False)

    delay_map = {"Slow (Real-world)": 0.4, "Normal": 0.15, "Fast (Demo)": 0.05}
    sleep_s   = delay_map[speed]

    # ── Initialise session history ────────────────────────────────────────────
    if "demo_history" not in st.session_state or reset_btn:
        st.session_state.demo_history = [
            {k: _add_noise(v, 0.03) for k, v in NORMAL_STATE.items()}
            for _ in range(30)
        ]
        st.session_state.demo_state   = "Normal"
        st.session_state.demo_running = False

    # ── Status banner ─────────────────────────────────────────────────────────
    state = st.session_state.demo_state
    banner_cls = {"Normal":"alert-normal","Early Fault":"alert-warning","CRITICAL FAILURE":"alert-critical"}
    banner_icon = {"Normal":"✅","Early Fault":"⚠️","CRITICAL FAILURE":"🚨"}
    st.markdown(f"""
    <div class='{banner_cls.get(state, "alert-normal")}' style='margin:10px 0;'>
      <b style='font-size:1.2em;'>{banner_icon.get(state,'ℹ️')} Machine {machine_id} — {state}</b>
      <br><span style='font-size:0.85em; color:#aabbdd;'>
        {'Everything nominal. Sensors within safe range.' if state=='Normal'
         else 'Elevated sensor readings detected. Monitoring closely…'
         if state=='Early Fault'
         else '🚨 CRITICAL FAILURE IMMINENT — Emergency shutdown recommended!'}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Live chart placeholder ─────────────────────────────────────────────────
    chart_ph    = st.empty()
    alert_ph    = st.empty()
    schedule_ph = st.empty()

    def _render_chart():
        fig = _make_live_chart(st.session_state.demo_history, st.session_state.demo_state)
        chart_ph.plotly_chart(fig, use_container_width=True)

    _render_chart()

    # ── Alert details (always visible) ────────────────────────────────────────
    def _render_alerts():
        readings_now = st.session_state.demo_history[-1]
        reasons = []
        if readings_now["vibration"]    > 3.0:  reasons.append(f"🔸 High vibration ({readings_now['vibration']:.2f} mm/s) → bearing wear")
        if readings_now["temperature"]  > 72.0: reasons.append(f"🔸 Elevated temp ({readings_now['temperature']:.1f}°C) → cooling stress")
        if readings_now["sound_db"]     > 78.0: reasons.append(f"🔸 Acoustic anomaly ({readings_now['sound_db']:.1f} dB)")
        if readings_now["load_pct"]     > 65.0: reasons.append(f"🔸 Overload ({readings_now['load_pct']:.1f}%)")
        if readings_now["rpm_drift_pct"]> 5.0:  reasons.append(f"🔸 RPM drift ({readings_now['rpm_drift_pct']:.1f}%) → misalignment")
        if readings_now["oil_pressure"] < 3.0:  reasons.append(f"🔸 Low oil pressure ({readings_now['oil_pressure']:.2f} bar)")

        if reasons:
            alert_ph.markdown(f"""
            <div class='alert-{("critical" if state=="CRITICAL FAILURE" else "high" if state=="Early Fault" else "normal")}'>
              <b>🧠 AI Explainability — Root Causes</b><br>
              {''.join(f'<div style="font-size:0.85em; padding:2px 0; color:#c0d0ff;">{r}</div>' for r in reasons)}
            </div>
            """, unsafe_allow_html=True)
        else:
            alert_ph.markdown("""
            <div class='alert-normal'>
              <b>✅ AI Explanation:</b> All sensors within normal operating range. Machine is healthy.
            </div>
            """, unsafe_allow_html=True)

    _render_alerts()

    # ── SIMULATION ────────────────────────────────────────────────────────────
    if run_sim and not st.session_state.demo_running:
        st.session_state.demo_running = True

        PHASES = [
            ("Normal",          NORMAL_STATE,   10),
            ("Early Fault",     FAULT_STATE,    20),
            ("CRITICAL FAILURE",CRITICAL_STATE, 15),
        ]

        for phase_name, phase_state, n_steps in PHASES:
            st.session_state.demo_state = phase_name

            # Get previous phase state
            prev_state = st.session_state.demo_history[-1] if st.session_state.demo_history else NORMAL_STATE

            for step in range(n_steps):
                t = step / max(n_steps - 1, 1)
                interp = _lerp(prev_state, phase_state, t)
                noisy  = {k: _add_noise(v, 0.04) for k, v in interp.items()}
                st.session_state.demo_history.append(noisy)

                # Keep last 60 points
                if len(st.session_state.demo_history) > 60:
                    st.session_state.demo_history = st.session_state.demo_history[-60:]

                _render_chart()
                _render_alerts()
                time.sleep(sleep_s)

            prev_state = phase_state

        st.session_state.demo_running = False

        # Final schedule recommendation
        schedule_ph.markdown(f"""
        <div class='alert-critical'>
          <b style='font-size:1.05em;'>📅 SMARTPREDICT Maintenance Recommendation</b><br><br>
          <div style='color:#c0d0ff; font-size:0.9em;'>
            Machine <b>{machine_id}</b> bearing is likely to fail within <b>48 hours</b>.<br><br>
            ⛔ <b>Immediate Action Required:</b> Schedule emergency maintenance NOW.<br>
            🔧 Estimated maintenance window: <b>Tonight 10:00 PM – 2:00 AM</b><br>
            💡 Recommended checks: Bearing replacement, lubrication system, spindle alignment.<br>
            ⏱️ Predicted downtime if unaddressed: <b>72–120 hours</b> (vs 4 hrs for planned maintenance).
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Current sensor values table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>Current Sensor Readings</div>", unsafe_allow_html=True)

    latest = st.session_state.demo_history[-1] if st.session_state.demo_history else NORMAL_STATE
    rows   = []
    warn_levels = {"vibration":3.0,"temperature":72.0,"sound_db":78.0,
                   "load_pct":65.0,"rpm_drift_pct":5.0,"oil_pressure":3.5}
    for sensor, label in SENSOR_LABELS.items():
        val  = latest.get(sensor, 0)
        warn = warn_levels.get(sensor, 0)
        status = "🔴 ALERT" if val > warn * 1.3 else "⚠️ Warning" if val > warn else "✅ OK"
        if sensor == "oil_pressure":
            status = "🔴 ALERT" if val < 2.5 else "⚠️ Warning" if val < 3.5 else "✅ OK"
        rows.append({"Sensor": label, "Value": f"{val:.2f}", "Status": status})

    tbl_df = pd.DataFrame(rows)
    st.dataframe(tbl_df, hide_index=True, use_container_width=True)
