"""
Advanced Enhancement Modules
1. Hybrid Physics+ML model using Darcy's Law concepts
2. Climate change scenario simulation
3. Anomaly detection (Isolation Forest + z-score)
4. Explainable AI (SHAP feature importance summary)
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "chennai_groundwater.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. HYBRID PHYSICS + ML ────────────────────────────────────────────────────

def darcy_recharge_estimate(rainfall_mm: np.ndarray,
                             soil_permeability: float,
                             urban_fraction: float) -> np.ndarray:
    """
    Physics-based recharge estimate inspired by Darcy's Law.
    R = (1 - urban_fraction) * K * rainfall_mm * 0.01
    where K = hydraulic conductivity proxy (soil_permeability).
    """
    effective_recharge = (1 - urban_fraction) * soil_permeability * rainfall_mm * 0.01
    return np.clip(effective_recharge, 0, None)


def build_hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment feature set with physics-derived recharge estimate."""
    df = df.copy()
    df["darcy_recharge"] = darcy_recharge_estimate(
        df["rainfall_mm"].values,
        df["soil_permeability"].values,
        df["urban_fraction"].values,
    )
    # Water balance residual: net stress
    df["water_balance"] = df["darcy_recharge"] - (
        df["evapotranspiration"] * 0.05 + df["urban_fraction"] * 0.3
    )
    return df


class HybridPhysicsML:
    """
    Hybrid model: physics layer computes Darcy recharge residual,
    ML layer (RF) corrects the physics with data-driven residuals.
    """

    def __init__(self):
        self.physics_coeff  = None
        self.ml_model       = RandomForestRegressor(n_estimators=200,
                                                    random_state=42, n_jobs=-1)
        self.scaler         = StandardScaler()
        self._feature_cols  = [
            "darcy_recharge", "water_balance", "rainfall_mm",
            "temperature_c", "urban_fraction", "soil_permeability",
            "population_density", "ndvi"
        ]

    def fit(self, df_train: pd.DataFrame):
        df_train = build_hybrid_features(df_train)
        X = df_train[self._feature_cols].fillna(0)
        y = df_train["groundwater_level"]

        # Physics baseline: linear fit of darcy_recharge → GW level
        self.physics_coeff = np.polyfit(df_train["darcy_recharge"], y, deg=1)
        physics_pred = np.polyval(self.physics_coeff, df_train["darcy_recharge"])
        residuals = y.values - physics_pred

        # ML corrects residuals
        X_sc = self.scaler.fit_transform(X)
        self.ml_model.fit(X_sc, residuals)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = build_hybrid_features(df)
        X  = df[self._feature_cols].fillna(0)
        physics_pred = np.polyval(self.physics_coeff, df["darcy_recharge"])
        X_sc = self.scaler.transform(X)
        residual_corr = self.ml_model.predict(X_sc)
        return physics_pred + residual_corr

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ── 2. CLIMATE CHANGE SCENARIO SIMULATION ─────────────────────────────────────

CLIMATE_SCENARIOS = {
    "RCP_2.6_Optimistic": {"rainfall_delta": +0.05, "temp_delta": +1.0,
                            "urban_delta": +0.02},
    "RCP_4.5_Moderate":   {"rainfall_delta": -0.10, "temp_delta": +2.0,
                            "urban_delta": +0.04},
    "RCP_8.5_Pessimistic":{"rainfall_delta": -0.20, "temp_delta": +3.5,
                            "urban_delta": +0.08},
}


def simulate_climate_scenario(df: pd.DataFrame, scenario_name: str,
                               n_years: int = 30) -> pd.DataFrame:
    """
    Project future groundwater levels under a climate scenario.
    Returns a DataFrame with projected monthly values.
    """
    params = CLIMATE_SCENARIOS[scenario_name]
    last_row = df.sort_values("date").groupby("zone").last().reset_index()
    projections = []

    for _, row in last_row.iterrows():
        for y in range(1, n_years + 1):
            for m in range(1, 13):
                rain_proj = row["rainfall_mm"] * (1 + params["rainfall_delta"])
                temp_proj = row["temperature_c"] + params["temp_delta"]
                urb_proj  = min(row["urban_fraction"] + params["urban_delta"] * y / n_years, 1.0)

                # Simplified GW projection
                recharge = (1 - urb_proj) * row["soil_permeability"] * rain_proj * 0.01
                extraction = 0.08 + urb_proj * 0.3
                gw_delta = -recharge + extraction + (temp_proj - 25) * 0.003

                gw_proj = row["groundwater_level"] + gw_delta * y * 0.05
                gw_proj = np.clip(gw_proj, 0.5, 40.0)

                projections.append({
                    "zone":             row["zone"],
                    "year":             df["date"].dt.year.max() + y,
                    "month":            m,
                    "scenario":         scenario_name,
                    "rainfall_proj":    round(rain_proj, 2),
                    "temp_proj":        round(temp_proj, 2),
                    "urban_proj":       round(urb_proj, 3),
                    "gw_level_proj":    round(float(gw_proj), 3),
                })
    return pd.DataFrame(projections)


def plot_climate_scenarios(df: pd.DataFrame, zone: str, out_dir: str):
    """Plot projected GW levels for all scenarios for a zone."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"RCP_2.6_Optimistic": "#2ecc71",
              "RCP_4.5_Moderate":   "#f39c12",
              "RCP_8.5_Pessimistic":"#e74c3c"}

    for scenario in CLIMATE_SCENARIOS:
        proj = simulate_climate_scenario(df, scenario, n_years=30)
        z    = proj[proj["zone"] == zone].groupby("year")["gw_level_proj"].mean()
        ax.plot(z.index, z.values, label=scenario, color=colors[scenario],
                linewidth=2, marker="o", markersize=3)

    historical = df[df["zone"] == zone].groupby(df["date"].dt.year)["groundwater_level"].mean()
    ax.plot(historical.index, historical.values, label="Historical",
            color="steelblue", linewidth=1.5, linestyle="--")

    ax.set_title(f"Climate Change Scenario Projections — {zone}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Groundwater Level (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"climate_scenario_{zone.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ── 3. ANOMALY DETECTION ──────────────────────────────────────────────────────

class GroundwaterAnomalyDetector:
    """
    Two-layer anomaly detection:
    1. Isolation Forest (model-based)
    2. Rolling z-score (statistical)
    """

    def __init__(self, contamination: float = 0.05, z_threshold: float = 3.0):
        self.iso_forest    = IsolationForest(contamination=contamination,
                                             random_state=42, n_jobs=-1)
        self.z_threshold   = z_threshold
        self._feature_cols = ["groundwater_level", "rainfall_mm", "temperature_c"]

    def fit(self, df: pd.DataFrame):
        X = df[self._feature_cols].fillna(0)
        self.iso_forest.fit(X)
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        X  = df[self._feature_cols].fillna(0)

        # Isolation Forest
        df["iso_anomaly"] = (self.iso_forest.predict(X) == -1).astype(int)
        df["iso_score"]   = -self.iso_forest.score_samples(X)

        # Rolling z-score on groundwater level
        df["gw_rolling_mean"] = df["groundwater_level"].rolling(6, min_periods=1).mean()
        df["gw_rolling_std"]  = df["groundwater_level"].rolling(6, min_periods=1).std().fillna(1)
        df["z_score"]         = (
            (df["groundwater_level"] - df["gw_rolling_mean"])
            / df["gw_rolling_std"]
        )
        df["z_anomaly"] = (df["z_score"].abs() > self.z_threshold).astype(int)

        # Combined flag
        df["anomaly"] = ((df["iso_anomaly"] == 1) | (df["z_anomaly"] == 1)).astype(int)
        return df


def plot_anomalies(df_zone: pd.DataFrame, zone: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_zone["date"], df_zone["groundwater_level"],
            label="GW Level", color="steelblue", linewidth=1)
    anomalies = df_zone[df_zone["anomaly"] == 1]
    ax.scatter(anomalies["date"], anomalies["groundwater_level"],
               color="red", zorder=5, s=40, label="Anomaly", marker="x")
    ax.set_title(f"Anomaly Detection — {zone}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Groundwater Level (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"anomaly_{zone.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 1. Hybrid Physics+ML
    print("=== Hybrid Physics+ML ===")
    train_df = df[df["date"] <= "2020-12-31"]
    test_df  = df[df["date"] >  "2020-12-31"]
    hybrid   = HybridPhysicsML()
    hybrid.fit(train_df)
    preds    = hybrid.predict(test_df)
    from utils.evaluation import regression_metrics
    regression_metrics(test_df["groundwater_level"].values, preds,
                       label="HybridPhysicsML")
    hybrid.save(os.path.join(MODEL_DIR, "hybrid_physics_ml.pkl"))

    # 2. Climate Scenarios (first zone)
    print("\n=== Climate Change Scenarios ===")
    first_zone = df["zone"].unique()[0]
    plot_climate_scenarios(df, first_zone, OUTPUT_DIR)
    print(f"  Scenario plot for {first_zone} saved.")

    # 3. Anomaly Detection
    print("\n=== Anomaly Detection ===")
    detector = GroundwaterAnomalyDetector()
    detector.fit(df)
    for zone in df["zone"].unique()[:3]:
        zdf = df[df["zone"] == zone].copy().reset_index(drop=True)
        zdf = detector.detect(zdf)
        n_anomalies = zdf["anomaly"].sum()
        print(f"  {zone}: {n_anomalies} anomalies detected")
        plot_anomalies(zdf, zone, OUTPUT_DIR)

    joblib.dump(detector, os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
    print("\nAdvanced modules pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
