"""
SMARTPREDICT 2.0 — Predictive Maintenance Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main Streamlit application entry-point.

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="SMARTPREDICT 2.0",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #151528 50%, #0d1117 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #0a0a1a 100%);
    border-right: 1px solid #2a2a4a;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.12) 0%, rgba(100,60,200,0.08) 100%);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 6px 0;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(79,142,247,0.2); }

.alert-critical {
    background: linear-gradient(135deg, rgba(255,60,60,0.18) 0%, rgba(200,20,20,0.10) 100%);
    border: 1px solid rgba(255,60,60,0.5);
    border-radius: 12px; padding: 16px; margin: 6px 0;
    animation: pulse-red 2s infinite;
}
.alert-high {
    background: linear-gradient(135deg, rgba(255,150,0,0.18) 0%, rgba(200,100,0,0.10) 100%);
    border: 1px solid rgba(255,150,0,0.5);
    border-radius: 12px; padding: 16px; margin: 6px 0;
}
.alert-warning {
    background: linear-gradient(135deg, rgba(255,220,0,0.15) 0%, rgba(200,170,0,0.08) 100%);
    border: 1px solid rgba(255,220,0,0.4);
    border-radius: 12px; padding: 16px; margin: 6px 0;
}
.alert-normal {
    background: linear-gradient(135deg, rgba(0,220,130,0.12) 0%, rgba(0,160,100,0.08) 100%);
    border: 1px solid rgba(0,220,130,0.35);
    border-radius: 12px; padding: 16px; margin: 6px 0;
}

@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,60,60,0.3); }
    50%      { box-shadow: 0 0 0 8px rgba(255,60,60,0.0); }
}

.risk-badge-Critical { background:#ff3c3c; color:#fff; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; }
.risk-badge-High     { background:#ff9600; color:#fff; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; }
.risk-badge-Warning  { background:#ffdc00; color:#111; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; }
.risk-badge-Normal   { background:#00dc82; color:#111; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; }

.section-title {
    color: #9ab3ff;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(154,179,255,0.2);
    margin-bottom: 16px;
}
.hero-title {
    font-size: 2.2em;
    font-weight: 800;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}
.hero-sub { color:#8899cc; font-size:1.0em; margin-top:0; }

/* Streamlit widget overrides */
div[data-testid="stMetricValue"] { font-size: 1.8em !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #8899cc !important; font-size: 0.82em !important; }
</style>
""", unsafe_allow_html=True)

# ─── Page router ─────────────────────────────────────────────────────────────
from dashboard import overview, sensor_detail, model_insights, demo_mode

PAGES = {
    "🏭  Fleet Overview"    : overview,
    "📊  Sensor Analysis"   : sensor_detail,
    "🧠  Model Insights"    : model_insights,
    "🎮  Demo Mode"         : demo_mode,
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:2.5em;'>🏭</div>
        <div style='font-size:1.2em; font-weight:800; color:#4f8ef7;'>SMARTPREDICT</div>
        <div style='font-size:0.72em; color:#6677aa; letter-spacing:0.15em;'>VERSION 2.0</div>
    </div>
    <hr style='border-color:#2a2a4a; margin:10px 0;'>
    """, unsafe_allow_html=True)

    page_name = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2a2a4a; margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>System Status</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82em; color:#aabbdd; line-height:1.9;'>
      🟢 &nbsp;AI Models Loaded<br>
      🟢 &nbsp;IoT Sensor Feed Active<br>
      🟢 &nbsp;Alert Engine Online<br>
      🟢 &nbsp;Dashboard Live
    </div>
    """, unsafe_allow_html=True)

# ─── Render selected page ─────────────────────────────────────────────────────
PAGES[page_name].render()
