"""
Failure Alert Engine
─────────────────────
Evaluates live/batch sensor readings and triggers:
  • Risk-level alerts (Low / Medium / High / Critical)
  • Human-readable explanations for WHY the alert fired
  • Maintenance scheduling recommendations
"""

import os, json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR   = "model"
THRESHOLDS  = {
    "failure_prob_medium"  : 0.35,
    "failure_prob_high"    : 0.65,
    "failure_prob_critical": 0.80,
    "rul_warning_hrs"      : 72,    # Warn if RUL < 72 hrs
    "rul_critical_hrs"     : 24,
    "vibration_high"       : 3.5,
    "temperature_high"     : 80.0,
    "sound_high"           : 85.0,
    "load_high"            : 75.0,
    "rpm_drift_high"       : 5.0,
}

_LABEL_MAP = {0: "Normal", 1: "EarlyFault", 2: "Critical"}


def _load_models():
    """Load saved models from disk (cached import)."""
    rf_path = hf_hub_download(
        repo_id="monish-73/predictive-machine-breakdown-rf",
        filename="rf_classifier.pkl",
        force_download=True
    )

    clf = joblib.load(rf_path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    iso    = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    reg    = joblib.load(os.path.join(MODEL_DIR, "rul_regressor.pkl"))
    fcols  = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return clf, scaler, iso, reg, fcols


# ─── Alert Data Class ─────────────────────────────────────────────────────────
@dataclass
class MachineAlert:
    machine_id    : str
    timestamp     : str
    risk_level    : str          # Normal / Warning / High / Critical
    condition     : str          # Normal / EarlyFault / Critical
    failure_prob  : float
    rul_hours     : float
    anomaly       : bool
    reasons       : list = field(default_factory=list)
    recommendation: str  = ""
    raw_sensors   : dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def to_message(self) -> str:
        lines = [
            f"[{self.risk_level}]  Machine {self.machine_id}",
            f"  Condition    : {self.condition}",
            f"  Failure Prob : {self.failure_prob*100:.1f}%",
            f"  RUL          : {self.rul_hours:.0f} hrs",
            f"  Anomaly      : {'YES ⚠️' if self.anomaly else 'No'}",
        ]
        if self.reasons:
            lines.append("  Root Causes  :")
            for r in self.reasons:
                lines.append(f"    • {r}")
        lines.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(lines)


# ─── Explainability ──────────────────────────────────────────────────────────
def _build_reasons(sensors: dict) -> list:
    """Generate human-readable explanation of alert causes."""
    reasons = []
    v   = sensors.get("vibration",    0)
    t   = sensors.get("temperature",  0)
    s   = sensors.get("sound_db",     0)
    l   = sensors.get("load_pct",     0)
    rpm = sensors.get("rpm_drift_pct",0)
    oil = sensors.get("oil_pressure", 99)

    if v > THRESHOLDS["vibration_high"]:
        reasons.append(f"High vibration ({v:.2f} mm/s) → bearing wear suspected")
    if t > THRESHOLDS["temperature_high"]:
        reasons.append(f"Elevated temperature ({t:.1f}°C) → cooling system stress")
    if s > THRESHOLDS["sound_high"]:
        reasons.append(f"Acoustic anomaly ({s:.1f} dB) → structural resonance or gear fault")
    if l > THRESHOLDS["load_high"]:
        reasons.append(f"High load current ({l:.1f}%) → motor overload condition")
    if abs(rpm) > THRESHOLDS["rpm_drift_high"]:
        reasons.append(f"RPM drift ({rpm:.1f}%) → shaft imbalance or misalignment")
    if oil < 3.0:
        reasons.append(f"Low oil pressure ({oil:.2f} bar) → lubrication failure risk")

    if not reasons:
        reasons.append("Cumulative sensor degradation trend detected by ML model")
    return reasons


# ─── Maintenance Scheduling ──────────────────────────────────────────────────
def _schedule_maintenance(rul_hours: float, risk_level: str) -> str:
    now = datetime.now()
    if risk_level == "Critical":
        return "⛔ EMERGENCY: Stop machine immediately. Maintenance required NOW."
    elif risk_level == "High":
        target = now + timedelta(hours=4)
        return f"⚠️  Schedule maintenance within 4 hours → by {target.strftime('%I:%M %p, %b %d')}"
    elif risk_level == "Warning":
        # Find next low-production window (10 PM – 2 AM)
        tonight_10pm = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now >= tonight_10pm:
            tonight_10pm += timedelta(days=1)
        window_end = tonight_10pm + timedelta(hours=4)
        return (
            f"🔧  Schedule maintenance tonight between "
            f"{tonight_10pm.strftime('%I:%M %p')} – {window_end.strftime('%I:%M %p, %b %d')} "
            f"(low-production window). RUL remaining: {rul_hours:.0f} hrs."
        )
    else:
        return f"✅  No immediate action needed. Next scheduled check in {min(int(rul_hours/24), 30)} days."


# ─── Core Prediction ─────────────────────────────────────────────────────────
_MODELS = None   # lazy load

def predict_and_alert(sensor_reading: dict) -> MachineAlert:
    """
    Given a dict of sensor values, return a structured MachineAlert.

    sensor_reading keys:
      machine_id, vibration, temperature, sound_db,
      load_pct, runtime_hrs, oil_pressure, rpm_drift_pct
    """
    global _MODELS
    if _MODELS is None:
        _MODELS = _load_models()
    clf, scaler, iso, reg, fcols = _MODELS

    # Feature engineering (mirror train_pipeline.py)
    v   = sensor_reading.get("vibration",     1.2)
    t   = sensor_reading.get("temperature",   55.0)
    l   = sensor_reading.get("load_pct",      45.0)
    rpm = sensor_reading.get("rpm_drift_pct", 0.0)
    rth = sensor_reading.get("runtime_hrs",   100.0)

    sr = dict(sensor_reading)
    sr["vib_temp_ratio"]    = v / (t + 1e-3)
    sr["load_rpm_product"]  = l * abs(rpm)
    sr["thermal_stress"]    = t * rth / 1000

    X_raw = np.array([[sr.get(f, 0.0) for f in fcols]])
    X     = scaler.transform(X_raw)

    # Predictions
    cond_id    = int(clf.predict(X)[0])          # 0=Normal, 1=EarlyFault, 2=Critical
    proba      = clf.predict_proba(X)[0]          # [p_normal, p_early, p_critical]
    fail_prob  = float(proba[2] + 0.5 * proba[1]) # weighted failure probability
    rul        = float(reg.predict(X)[0])
    is_anomaly = iso.predict(X)[0] == -1

    # ── Physics-based multi-sensor hard override ────────────────────────────
    # If multiple sensors simultaneously exceed critical physical limits,
    # escalate regardless of model class prediction (handles boundary cases)
    vib  = sensor_reading.get("vibration",     0)
    temp = sensor_reading.get("temperature",   0)
    snd  = sensor_reading.get("sound_db",      0)
    load = sensor_reading.get("load_pct",      0)
    oil  = sensor_reading.get("oil_pressure",  9)
    rpm  = sensor_reading.get("rpm_drift_pct", 0)

    critical_sensors = sum([
        vib  > 5.0,
        temp > 85.0,
        snd  > 88.0,
        load > 78.0,
        oil  < 2.6,
        abs(rpm) > 8.0,
    ])
    early_sensors = sum([
        vib  > 2.5,
        temp > 68.0,
        snd  > 76.0,
        load > 58.0,
        oil  < 3.8,
        abs(rpm) > 3.0,
    ])

    if critical_sensors >= 3:          # 3+ sensors in critical zone → Critical
        hard_risk = "Critical"
    elif critical_sensors >= 1 or early_sensors >= 3:
        hard_risk = "High" if early_sensors >= 4 else "Warning"
    else:
        hard_risk = None               # let model decide

    # ── Risk level: model condition is primary, physics override takes precedence
    if hard_risk == "Critical":
        risk = "Critical"
    elif cond_id == 2:
        risk = "Critical"
    elif cond_id == 1:
        if hard_risk == "High" or fail_prob >= THRESHOLDS["failure_prob_high"] or rul <= THRESHOLDS["rul_warning_hrs"]:
            risk = "High"
        else:
            risk = "Warning"
    else:                                                          # Model says Normal
        if hard_risk in ("Critical", "High"):
            risk = hard_risk
        elif rul <= THRESHOLDS["rul_critical_hrs"]:
            risk = "Critical"
        elif rul <= THRESHOLDS["rul_warning_hrs"] or is_anomaly:
            risk = "Warning"
        else:
            risk = "Normal"

    reasons = _build_reasons(sensor_reading)
    recommendation = _schedule_maintenance(rul, risk)

    alert = MachineAlert(
        machine_id    = str(sensor_reading.get("machine_id", "M01")),
        timestamp     = datetime.now().isoformat(),
        risk_level    = risk,
        condition     = _LABEL_MAP[cond_id],
        failure_prob  = round(fail_prob, 4),
        rul_hours     = round(rul, 1),
        anomaly       = bool(is_anomaly),
        reasons       = reasons,
        recommendation= recommendation,
        raw_sensors   = {k: v for k, v in sensor_reading.items() if k != "machine_id"},
    )
    return alert


def evaluate_fleet(sensor_readings: list) -> list:
    """Evaluate a list of sensor readings and return all alerts."""
    alerts = [predict_and_alert(s) for s in sensor_readings]
    critical = [a for a in alerts if a.risk_level in ("Critical", "High")]
    if critical:
        print(f"\n🚨  {len(critical)} machine(s) require attention!\n")
        for a in critical:
            print(a.to_message())
            print("-" * 55)
    return alerts


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate 5 machines with varied states
    # 15 machines: M01-M11 Normal, M12-M13 EarlyFault, M14-M15 Critical
    demo_readings = [
        {"machine_id":"M01","vibration":1.2,"temperature":57,"sound_db":65,"load_pct":44,"runtime_hrs":115,"oil_pressure":4.5,"rpm_drift_pct":0.4},
        {"machine_id":"M02","vibration":6.5,"temperature":92,"sound_db":94,"load_pct":84,"runtime_hrs":1280,"oil_pressure":2.2,"rpm_drift_pct":10.2},
        {"machine_id":"M03","vibration":1.0,"temperature":60,"sound_db":64,"load_pct":43,"runtime_hrs":310,"oil_pressure":4.6,"rpm_drift_pct":0.3},
        {"machine_id":"M04","vibration":3.0,"temperature":74,"sound_db":80,"load_pct":64,"runtime_hrs":600,"oil_pressure":3.5,"rpm_drift_pct":4.0},
        {"machine_id":"M05","vibration":1.1,"temperature":56,"sound_db":66,"load_pct":45,"runtime_hrs":250,"oil_pressure":4.5,"rpm_drift_pct":0.5},
        {"machine_id":"M06","vibration":1.5,"temperature":59,"sound_db":65,"load_pct":47,"runtime_hrs":400,"oil_pressure":4.2,"rpm_drift_pct":1.0},
        {"machine_id":"M07","vibration":2.8,"temperature":72,"sound_db":79,"load_pct":62,"runtime_hrs":550,"oil_pressure":3.6,"rpm_drift_pct":3.8},
        {"machine_id":"M08","vibration":1.2,"temperature":61,"sound_db":66,"load_pct":50,"runtime_hrs":290,"oil_pressure":4.3,"rpm_drift_pct":0.7},
        {"machine_id":"M09","vibration":1.6,"temperature":58,"sound_db":68,"load_pct":46,"runtime_hrs":340,"oil_pressure":4.4,"rpm_drift_pct":0.9},
        {"machine_id":"M10","vibration":5.8,"temperature":89,"sound_db":92,"load_pct":82,"runtime_hrs":1180,"oil_pressure":2.4,"rpm_drift_pct":9.6},
        {"machine_id":"M11","vibration":1.4,"temperature":60,"sound_db":67,"load_pct":48,"runtime_hrs":160,"oil_pressure":4.3,"rpm_drift_pct":0.6},
        {"machine_id":"M12","vibration":1.3,"temperature":57,"sound_db":66,"load_pct":45,"runtime_hrs":210,"oil_pressure":4.5,"rpm_drift_pct":0.5},
        {"machine_id":"M13","vibration":1.1,"temperature":58,"sound_db":65,"load_pct":44,"runtime_hrs":175,"oil_pressure":4.4,"rpm_drift_pct":0.4},
        {"machine_id":"M14","vibration":1.5,"temperature":59,"sound_db":67,"load_pct":46,"runtime_hrs":320,"oil_pressure":4.3,"rpm_drift_pct":0.8},
        {"machine_id":"M15","vibration":1.3,"temperature":56,"sound_db":65,"load_pct":43,"runtime_hrs":140,"oil_pressure":4.5,"rpm_drift_pct":0.6},
    ]

    print("=" * 60)
    print("  SMARTPREDICT – Fleet Alert Evaluation")
    print("=" * 60)
    alerts = evaluate_fleet(demo_readings)

    for a in alerts:
        print(a.to_message())
        print("=" * 60)
