"""
Data Preprocessing & Feature Engineering Pipeline
Handles loading, cleaning, lag features, rolling stats, drought index,
seasonal indicators, and train/test splitting with time-series awareness.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

# ── Constants ────────────────────────────────────────────────────────────────
LAG_STEPS      = [1, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
FEATURE_COLS   = None   # populated after engineering


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.sort_values(["zone", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode soil_type; one-hot-encode zone."""
    df = df.copy()
    le = LabelEncoder()
    df["soil_type_enc"] = le.fit_transform(df["soil_type"])
    df = pd.get_dummies(df, columns=["zone"], prefix="zone", drop_first=False)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    # NE monsoon: Oct-Dec = 1; SW monsoon: Jun-Sep = 2; dry: 0
    df["season"] = df["month"].map(lambda m:
        2 if m in [10, 11, 12] else (1 if m in [6, 7, 8, 9] else 0))
    df["is_monsoon"] = (df["season"] > 0).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone lag features for groundwater_level and rainfall."""
    df = df.copy()
    for zone_col in [c for c in df.columns if c.startswith("zone_")]:
        mask = df[zone_col] == 1
        for lag in LAG_STEPS:
            df.loc[mask, f"gw_lag_{lag}"]   = df.loc[mask, "groundwater_level"].shift(lag)
            df.loc[mask, f"rain_lag_{lag}"]  = df.loc[mask, "rainfall_mm"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone rolling mean/std for rainfall and temperature."""
    df = df.copy()
    for zone_col in [c for c in df.columns if c.startswith("zone_")]:
        mask = df[zone_col] == 1
        for w in ROLLING_WINDOWS:
            df.loc[mask, f"rain_roll_mean_{w}"] = (
                df.loc[mask, "rainfall_mm"].rolling(w, min_periods=1).mean())
            df.loc[mask, f"temp_roll_mean_{w}"] = (
                df.loc[mask, "temperature_c"].rolling(w, min_periods=1).mean())
            df.loc[mask, f"gw_roll_std_{w}"]    = (
                df.loc[mask, "groundwater_level"].rolling(w, min_periods=1).std().fillna(0))
    return df


def add_drought_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified SPI-like drought index based on rainfall z-score
    per calendar month across all years.
    """
    df = df.copy()
    rain_stats = df.groupby("month")["rainfall_mm"].agg(["mean", "std"]).reset_index()
    rain_stats.columns = ["month", "rain_monthly_mean", "rain_monthly_std"]
    df = df.merge(rain_stats, on="month", how="left")
    df["spi"] = (
        (df["rainfall_mm"] - df["rain_monthly_mean"])
        / df["rain_monthly_std"].replace(0, 1)
    )
    df.drop(columns=["rain_monthly_mean", "rain_monthly_std"], inplace=True)
    return df


def add_urbanization_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite urbanization stress index:
    combines urban_fraction, population_density (normalized), and ndvi inverse.
    """
    df = df.copy()
    pop_norm = (df["population_density"] - df["population_density"].min()) / (
        df["population_density"].max() - df["population_density"].min() + 1e-9)
    df["urbanization_index"] = (
        0.5 * df["urban_fraction"] + 0.3 * pop_norm + 0.2 * (1 - df["ndvi"])
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline in correct order."""
    df = encode_categoricals(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_drought_index(df)
    df = add_urbanization_index(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ML feature columns (exclude targets and metadata)."""
    exclude = {
        "date", "soil_type", "risk_level", "groundwater_level",
        "risk_code", "month"
    }
    return [c for c in df.columns if c not in exclude]


def split_train_test(df: pd.DataFrame, test_years: int = 3):
    """
    Time-series aware split: last `test_years` years as test set.
    Returns X_train, X_test, y_train, y_test (regression targets).
    """
    cutoff = df["date"].max() - pd.DateOffset(years=test_years)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()

    feature_cols = get_feature_columns(df)
    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train["groundwater_level"]
    y_test  = test["groundwater_level"]
    y_train_cls = train["risk_code"]
    y_test_cls  = test["risk_code"]

    return X_train, X_test, y_train, y_test, y_train_cls, y_test_cls, feature_cols


def scale_features(X_train, X_test, save_path: str = None):
    """Fit StandardScaler on train, transform both splits."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    if save_path:
        joblib.dump(scaler, save_path)
    return X_train_sc, X_test_sc, scaler


def get_ts_splits(X, y, n_splits: int = 5):
    """Return TimeSeriesSplit cross-validation splits."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X, y))


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/chennai_groundwater.csv"
    raw = load_data(csv)
    print("Raw shape:", raw.shape)
    eng = engineer_features(raw)
    print("Engineered shape:", eng.shape)
    print("Features:", get_feature_columns(eng)[:10], "...")
