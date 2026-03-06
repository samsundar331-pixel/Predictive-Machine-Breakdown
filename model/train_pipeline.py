"""
Hybrid Ensemble Training Pipeline — SMARTPREDICT 2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model Stack (why each component):
┌────────────────────────────────────────────────────────────────┐
│ 1. Random Forest (200 trees)      → robust, handles outliers   │
│ 2. Gradient Boosting (150 trees)  → error-corrective boosting  │
│ 3. Extra-Trees (200 trees)        → randomised splits, low var │
│    ↓ Soft-Voting Ensemble (weights 2:3:2)                      │
│ 4. Isolation Forest               → unsupervised anomaly det.  │
│ 5. Gradient Boosting Regressor    → RUL prediction             │
└────────────────────────────────────────────────────────────────┘

Target test accuracy: 88–94%
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               ExtraTreesClassifier,
                               VotingClassifier,
                               IsolationForest,
                               GradientBoostingRegressor)
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      accuracy_score, mean_absolute_error,
                                      mean_squared_error, r2_score)
from imblearn.over_sampling   import SMOTE

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "sensor_data.csv")
MODEL_DIR   = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "vibration", "temperature", "sound_db",
    "load_pct",  "runtime_hrs", "oil_pressure", "rpm_drift_pct",
    "vib_temp_ratio", "load_rpm_product", "thermal_stress",
]
TARGET_CLS = "condition"
TARGET_RUL = "rul_hours"


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data(path):
    print(f"\n📂  Loading dataset from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Build engineered features if missing (older CSV)
    if "vib_temp_ratio" not in df.columns:
        df["vib_temp_ratio"]   = df["vibration"] / (df["temperature"] + 1e-3)
        df["load_rpm_product"] = df["load_pct"] * df["rpm_drift_pct"].abs()
        df["thermal_stress"]   = df["temperature"] * df["runtime_hrs"] / 1000

    print(f"   Rows: {len(df):,}  | Classes:\n{df['label'].value_counts()}")
    return df


# ─── Hybrid Voting Classifier ─────────────────────────────────────────────────
def train_hybrid_classifier(df, feature_cols):
    print("\n🔀  Building Hybrid Ensemble Classifier…")
    print("    Components: RandomForest + GradientBoosting + ExtraTrees  (soft-vote)")

    X = df[feature_cols].values
    y = df[TARGET_CLS].values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.20, stratify=y, random_state=42
    )

    # Oversample minority to equalise training
    sm = SMOTE(random_state=42)
    X_tr_rs, y_tr_rs = sm.fit_resample(X_train, y_train)

    # ── Component models ──────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=18,
        min_samples_split=5, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08,
        max_depth=6, subsample=0.85,
        random_state=42
    )
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=18,
        min_samples_split=5, class_weight="balanced",
        random_state=42, n_jobs=-1
    )

    # ── Soft-Voting Hybrid Ensemble ────────────────────────────────────────────
    hybrid = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        voting="soft",
        weights=[2, 3, 2],    # GBM weighted highest (best at boundary cases)
        n_jobs=-1,
    )

    print("    Training ensemble (this takes ~30 s)…")
    hybrid.fit(X_tr_rs, y_tr_rs)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = hybrid.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    labels = ["Normal", "EarlyFault", "Critical"]

    print(f"\n   ✅ Hybrid Ensemble Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=labels))

    # 5-fold CV on the full scaled set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(hybrid, X_sc, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"   5-Fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix – Hybrid Ensemble", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance (from RF component)
    fi = rf  # need to fit rf separately for importance
    rf.fit(X_tr_rs, y_tr_rs)  # already fitted inside VotingClassifier too
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
    plt.title("Feature Importances (RF component)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    metrics = {
        "accuracy": round(acc, 4),
        "cv_mean" : round(float(cv_scores.mean()), 4),
        "cv_std"  : round(float(cv_scores.std()),  4),
    }
    return hybrid, scaler, metrics


# ─── Isolation Forest ─────────────────────────────────────────────────────────
def train_anomaly_detector(df, feature_cols, scaler):
    print("\n🔍  Training Isolation Forest (Anomaly Detector)…")
    normal_df = df[df[TARGET_CLS] == 0]
    X_normal  = scaler.transform(normal_df[feature_cols].values)

    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1)
    iso.fit(X_normal)

    X_all  = scaler.transform(df[feature_cols].values)
    preds  = iso.predict(X_all)
    rate   = (preds == -1).mean() * 100
    print(f"   ✅ Anomaly flag rate on full dataset: {rate:.1f}%")
    return iso


# ─── RUL: Gradient Boosting Regressor ─────────────────────────────────────────
def train_rul_regressor(df, feature_cols, scaler):
    print("\n⏱️   Training Gradient Boosting Regressor for RUL…")
    X = scaler.transform(df[feature_cols].values)
    y = df[TARGET_RUL].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    reg = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.08,
        max_depth=6, subsample=0.85,
        random_state=42
    )
    reg.fit(X_tr, y_tr)

    y_pred = reg.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    print(f"   ✅ RUL  MAE: {mae:.1f} hrs | RMSE: {rmse:.1f} hrs | R²: {r2:.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_te[:600], y_pred[:600], alpha=0.35, color="#4f8ef7",
                edgecolors="white", s=18)
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
    plt.xlabel("Actual RUL (hrs)")
    plt.ylabel("Predicted RUL (hrs)")
    plt.title("RUL Prediction – Actual vs Predicted (GBM)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "rul_prediction.png"), dpi=150)
    plt.close()

    metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}
    return reg, metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    banner = "SMARTPREDICT – Hybrid Ensemble Training Pipeline"
    print("=" * 60)
    print(f"  {banner}")
    print("=" * 60)

    df = load_data(DATA_PATH)
    feature_cols = FEATURE_COLS

    clf, scaler, clf_m = train_hybrid_classifier(df, feature_cols)
    iso               = train_anomaly_detector(df, feature_cols, scaler)
    reg, rul_m        = train_rul_regressor(df, feature_cols, scaler)

    # Save
    joblib.dump(clf,         os.path.join(MODEL_DIR, "rf_classifier.pkl"))   # keeps same filename for compatibility
    joblib.dump(scaler,      os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(iso,         os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(reg,         os.path.join(MODEL_DIR, "rul_regressor.pkl"))
    joblib.dump(feature_cols,os.path.join(MODEL_DIR, "feature_cols.pkl"))

    all_metrics = {"classifier": clf_m, "rul_regressor": rul_m,
                   "model_type": "Hybrid Voting Ensemble (RF + GBM + ExtraTrees)"}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✅  All models saved to '{MODEL_DIR}/'")
    print(f"⏱️   Total training time: {time.time()-t0:.1f}s")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
