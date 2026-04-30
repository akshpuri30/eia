"""
Synthetic Dataset Generator for Chennai Groundwater Prediction System
Generates realistic multi-source environmental data for 10 Chennai zones (2000-2023)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Reproducibility
np.random.seed(42)

# ── Zone definitions (name, lat, lon, coastal, urban_growth_rate) ─────────────
ZONES = {
    "Tambaram":       (12.924, 80.127, False, 0.035),
    "Adyar":          (13.001, 80.257, True,  0.020),
    "Anna Nagar":     (13.085, 80.209, False, 0.015),
    "Velachery":      (12.978, 80.220, False, 0.040),
    "Sholinganallur": (12.901, 80.228, True,  0.050),
    "Chromepet":      (12.952, 80.143, False, 0.030),
    "Perambur":       (13.117, 80.233, False, 0.018),
    "Manali":         (13.165, 80.263, True,  0.025),
    "Avadi":          (13.115, 80.098, False, 0.045),
    "Porur":          (13.035, 80.157, False, 0.038),
}

SOIL_TYPES = ["Sandy", "Clayey", "Loamy", "Rocky", "Alluvial"]
SOIL_PERMEABILITY = {"Sandy": 0.9, "Clayey": 0.2, "Loamy": 0.6, "Rocky": 0.1, "Alluvial": 0.75}

START_DATE = datetime(2000, 1, 1)
END_DATE   = datetime(2023, 12, 31)


def _monthly_dates():
    dates = []
    d = START_DATE
    while d <= END_DATE:
        dates.append(d)
        # advance one month
        month = d.month % 12 + 1
        year  = d.year + (1 if d.month == 12 else 0)
        d = d.replace(year=year, month=month, day=1)
    return dates


def _rainfall(dates, is_coastal: bool) -> np.ndarray:
    """Northeast monsoon (Oct-Dec) dominant for Chennai; coastal gets more."""
    rain = []
    for d in dates:
        m = d.month
        # Northeast monsoon peak
        if m in [10, 11, 12]:
            base = 200 if is_coastal else 160
        # Southwest monsoon secondary
        elif m in [6, 7, 8, 9]:
            base = 80
        else:
            base = 20
        noise = np.random.gamma(shape=2, scale=base / 2)
        rain.append(noise)
    return np.array(rain)


def _temperature(dates) -> np.ndarray:
    """Chennai temperature range 22-42°C with seasonal variation."""
    temps = []
    for d in dates:
        m = d.month
        if m in [4, 5, 6]:
            base = 38
        elif m in [12, 1, 2]:
            base = 25
        else:
            base = 32
        temps.append(base + np.random.normal(0, 1.5))
    return np.array(temps)


def _groundwater_level(dates, rainfall, temperature, urban_rate,
                        soil_perm, is_coastal) -> np.ndarray:
    """
    Simulate groundwater table depth (meters below ground level).
    Higher value = deeper = more depleted.
    """
    n = len(dates)
    gw = np.zeros(n)

    # Initial level — realistic Chennai range (5–20 m below surface)
    gw[0] = np.random.uniform(8, 20) if not is_coastal else np.random.uniform(5, 15)

    for i in range(1, n):
        year_offset = (dates[i].year - 2000)
        # Recharge from rainfall (monsoon replenishment, net downward = lower depth)
        recharge = rainfall[i] * soil_perm * 0.002   # much smaller coefficient
        # Extraction grows sharply with urbanization
        extraction = 0.12 + urban_rate * year_offset * 0.02
        # Temperature evapotranspiration effect
        evap_loss = (temperature[i] - 25) * 0.005
        # Coastal saline intrusion adds stress
        coastal_stress = 0.03 if is_coastal else 0
        # Long-term depletion trend (dominant)
        trend = urban_rate * 0.05

        delta = -recharge + extraction + evap_loss + coastal_stress + trend
        gw[i] = gw[i - 1] + delta + np.random.normal(0, 0.25)
        # Physical bounds: 1–35 m below surface
        gw[i] = np.clip(gw[i], 1.0, 35.0)

    return gw


def _population_density(dates, base_density, urban_rate) -> np.ndarray:
    years = np.array([(d.year - 2000) for d in dates])
    return base_density * np.exp(urban_rate * years / 5)


def _land_use_urban_fraction(dates, base_urban, urban_rate) -> np.ndarray:
    years = np.array([(d.year - 2000) for d in dates])
    return np.clip(base_urban + urban_rate * years / 4, 0, 1)


def generate_zone_data(zone_name: str) -> pd.DataFrame:
    lat, lon, is_coastal, urban_rate = ZONES[zone_name]
    soil = np.random.choice(SOIL_TYPES)
    perm = SOIL_PERMEABILITY[soil]

    dates      = _monthly_dates()
    rainfall   = _rainfall(dates, is_coastal)
    temp       = _temperature(dates)
    gw         = _groundwater_level(dates, rainfall, temp, urban_rate, perm, is_coastal)
    pop        = _population_density(dates, np.random.uniform(8000, 25000), urban_rate)
    urban_frac = _land_use_urban_fraction(dates, np.random.uniform(0.3, 0.7), urban_rate)

    # Evapotranspiration proxy
    et = temp * 0.6 + np.random.normal(0, 1, len(dates))
    # NDVI proxy (vegetation index) decreases with urbanization
    ndvi = 0.55 - urban_frac * 0.3 + np.random.normal(0, 0.03, len(dates))

    df = pd.DataFrame({
        "date":               dates,
        "zone":               zone_name,
        "latitude":           lat,
        "longitude":          lon,
        "is_coastal":         int(is_coastal),
        "soil_type":          soil,
        "soil_permeability":  perm,
        "rainfall_mm":        np.round(rainfall, 2),
        "temperature_c":      np.round(temp, 2),
        "evapotranspiration": np.round(et, 2),
        "ndvi":               np.round(np.clip(ndvi, 0, 1), 3),
        "population_density": np.round(pop, 0),
        "urban_fraction":     np.round(urban_frac, 3),
        "groundwater_level":  np.round(gw, 3),   # target: meters below surface
    })

    # Risk label — bins calibrated to the simulated depth range
    df["risk_level"] = pd.cut(
        df["groundwater_level"],
        bins=[0, 12, 20, 36],
        labels=["Low", "Medium", "High"]
    )
    df["risk_code"] = df["risk_level"].map({"Low": 0, "Medium": 1, "High": 2})
    return df


def generate_full_dataset(output_dir: str = ".") -> pd.DataFrame:
    all_dfs = []
    for zone in ZONES:
        print(f"  Generating data for zone: {zone}")
        all_dfs.append(generate_zone_data(zone))

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["zone", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = os.path.join(output_dir, "chennai_groundwater.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Dataset saved → {out_path}")
    print(f"  Shape: {df.shape}  |  Zones: {df['zone'].nunique()}  |  Date range: {df['date'].min().date()} – {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    print("=== Generating Chennai Groundwater Dataset ===")
    df = generate_full_dataset(output_dir=os.path.dirname(__file__) or ".")
    print(df.describe())
