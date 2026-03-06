"""
IoT Sensor Data Generator — 15 Machines, Fully Random Condition Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design:
  • Every machine goes through ALL 3 conditions (Normal → EarlyFault → Critical)
    following a realistic temporal degradation curve.
  • Which phase each machine is currently IN is random — some start healthy,
    some are mid-life, some are near-end — so NO machine is permanently
    labelled healthy or defective by position.
  • Live-reading snapshot at the current timestamp is RANDOM per machine.

Target accuracy: 88–94%  (class overlap baked in via Gaussian tails + 3% label noise)
Total rows: ~24 000
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

SEED       = 99
N_MACHINES = 15
# Per-phase samples per machine  (baseline; amplified for current phase)
BASE_N = {0: 360, 1: 210, 2: 90}

PARAMS = {
    #            vibration          temperature       sound_db
    #            load_pct           oil_pressure      rpm_drift_pct
    0: dict(vibration=(1.5, 0.58),  temperature=(59.0,  6.0),
            sound_db=(67.0,  5.0),  load_pct=(46.0,    9.0),
            oil_pressure=(4.35, 0.40), rpm_drift_pct=(0.6, 1.4)),

    1: dict(vibration=(2.9, 0.88),  temperature=(73.0,  8.0),
            sound_db=(79.0,  6.5),  load_pct=(63.0,   10.0),
            oil_pressure=(3.5, 0.55), rpm_drift_pct=(4.0, 2.1)),

    2: dict(vibration=(5.3, 1.45),  temperature=(90.0,  9.5),
            sound_db=(91.0,  8.0),  load_pct=(81.0,    9.5),
            oil_pressure=(2.3, 0.65), rpm_drift_pct=(9.0, 2.9)),
}
SENSORS = list(PARAMS[0].keys())
CLIP = {
    "vibration":    (0.3, 18.0), "temperature":  (28.0, 125.0),
    "sound_db":     (45.0, 115.0), "load_pct":   (10.0, 99.5),
    "oil_pressure": (0.5, 7.0),  "rpm_drift_pct":(-3.0, 22.0),
}
COND_LABEL = {0: "Normal", 1: "EarlyFault", 2: "Critical"}

# Random "current phase" for each machine — truly randomised
# Roughly: 11 machines currently Normal, 2 EarlyFault, 2 Critical
# but DISTRIBUTED RANDOMLY across the 15 positions
_RNG_PHASE = np.random.default_rng(SEED)
_POOL      = [0]*11 + [1]*2 + [2]*2
_RNG_PHASE.shuffle(_POOL)
MACHINE_CURRENT_PHASE = {i+1: _POOL[i] for i in range(N_MACHINES)}


def _clip(arr, s):
    return np.clip(arr, CLIP[s][0], CLIP[s][1])


def _block(n, cond, rng):
    p = PARAMS[cond]
    rows = {}
    for s in SENSORS:
        mu, sigma = p[s]
        v = rng.normal(mu, sigma, n)
        # Temporal drift within block
        drift = np.linspace(0, sigma * 0.30 * max(cond, 0.3), n)
        v += drift if s != "oil_pressure" else -drift
        # Outlier spikes ~1.5%
        mask = rng.random(n) < 0.015
        v    = np.where(mask, v * rng.uniform(1.2, 1.7, n), v)
        rows[s] = _clip(v, s)
    return rows


def _adjacent_bleed(df, cond, rng, frac=0.035):
    """Contaminate boundary rows with adjacent-class readings."""
    n_bl = max(1, int(len(df) * frac))
    nb   = {0: 1, 1: rng.integers(0, 2)*2, 2: 1}[cond]   # 0→1, 1→0or2, 2→1
    idx  = rng.choice(df.index, n_bl, replace=False)
    p    = PARAMS[nb]
    for s in SENSORS:
        mu, sigma = p[s]
        df.loc[idx, s] = _clip(rng.normal(mu, sigma * 0.9, n_bl), s)
    return df


def generate_machine(mid, rng):
    cur_phase = MACHINE_CURRENT_PHASE[mid]
    frames = []
    for cond in range(3):
        # Amplify current phase 3× to represent where the machine IS NOW
        n = BASE_N[cond] * (3 if cond == cur_phase else 1)
        raw   = _block(n, cond, rng)
        block = pd.DataFrame(raw)
        block = _adjacent_bleed(block, cond, rng)

        rt  = np.cumsum(rng.uniform(0.05, 0.18, n))
        rul_base = {0: 1800, 1: 260, 2: 35}[cond]
        rul = np.clip(rul_base - rt * (cond + 1) * 0.35
                      + rng.normal(0, rul_base * 0.15, n), 0, 4500)

        fp_base = {0: 0.07, 1: 0.47, 2: 0.86}[cond]
        block["runtime_hrs"]  = rt.round(2)
        block["rul_hours"]    = rul.round(1)
        block["failure_prob"] = np.clip(fp_base + rng.normal(0, 0.07, n), 0, 0.99).round(4)
        block["condition"]    = cond
        block["label"]        = COND_LABEL[cond]
        block["machine_id"]   = f"M{mid:02d}"
        frames.append(block)

    mdf = pd.concat(frames, ignore_index=True)

    # Label noise: 3% flips to adjacent class only
    flip = rng.random(len(mdf)) < 0.030
    for i in mdf[flip].index:
        c  = int(mdf.at[i, "condition"])
        nb = list({max(0, c-1), min(2, c+1)} - {c})
        if nb:
            nc = int(rng.choice(nb))
            mdf.at[i, "condition"] = nc
            mdf.at[i, "label"]     = COND_LABEL[nc]
    return mdf


def build_dataset():
    frames = []
    for mid in range(1, N_MACHINES + 1):
        rng = np.random.default_rng(SEED + mid * 61)
        frames.append(generate_machine(mid, rng))

    df = pd.concat(frames, ignore_index=True)

    # Feature engineering
    df["vib_temp_ratio"]   = (df["vibration"] / (df["temperature"] + 1e-3)).round(5)
    df["load_rpm_product"] = (df["load_pct"] * df["rpm_drift_pct"].abs()).round(3)
    df["thermal_stress"]   = (df["temperature"] * df["runtime_hrs"] / 1000).round(4)

    # Timestamps
    base = datetime(2024, 1, 1)
    ts   = [base + timedelta(minutes=i * 5) for i in range(len(df))]
    df.insert(0, "timestamp", ts)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    print("Generating 15-machine IoT dataset (random condition distribution)…")
    df = build_dataset()
    out = os.path.join("dataset", "sensor_data.csv")
    df.to_csv(out, index=False)

    print(f"✅  Saved → {out}  ({len(df):,} rows, {df['machine_id'].nunique()} machines)")
    print(f"\nGlobal class distribution:\n{df['label'].value_counts()}")
    print(f"\nCurrent operational phase per machine:")
    for mid, ph in MACHINE_CURRENT_PHASE.items():
        print(f"  M{mid:02d} → {COND_LABEL[ph]}")
    print(f"\nSensor overlap (vibration):")
    print(df.groupby("label")["vibration"].agg(["mean","std","min","max"]).round(2))
