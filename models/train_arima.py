"""
ARIMA Time-Series Baseline
Auto-selects (p,d,q) order per zone using AIC minimization,
generates multi-step forecasts, and saves diagnostic plots.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from itertools import product

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluation import regression_metrics

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "chennai_groundwater.csv")


def check_stationarity(series: pd.Series) -> bool:
    result = adfuller(series.dropna())
    return result[1] < 0.05   # p-value < 0.05 → stationary


def auto_select_order(series: pd.Series,
                      p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
    """Grid-search ARIMA(p,d,q) by AIC."""
    best_aic, best_order = np.inf, (1, 1, 1)
    for p, d, q in product(range(*p_range), range(*d_range), range(*q_range)):
        if p + d + q == 0:
            continue
        try:
            model = SARIMAX(series, order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


def fit_arima(series: pd.Series, order: tuple):
    model = SARIMAX(series, order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    return model.fit(disp=False)


def forecast_arima(fitted_model, steps: int) -> np.ndarray:
    forecast = fitted_model.forecast(steps=steps)
    return np.array(forecast)


def plot_arima_diagnostics(fitted_model, zone_name: str, out_dir: str):
    fig = fitted_model.plot_diagnostics(figsize=(12, 8))
    fig.suptitle(f"ARIMA Diagnostics — {zone_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"ARIMA_{zone_name.replace(' ', '_')}_diagnostics.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_arima_forecast(actual, forecast, zone_name, forecast_start_idx, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(actual)), actual, label="Actual", color="steelblue")
    fc_idx = range(forecast_start_idx, forecast_start_idx + len(forecast))
    ax.plot(fc_idx, forecast, label="ARIMA Forecast", color="tomato", linestyle="--")
    ax.axvline(forecast_start_idx, color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"ARIMA Forecast — {zone_name}")
    ax.set_xlabel("Month Index")
    ax.set_ylabel("Groundwater Level (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"ARIMA_{zone_name.replace(' ', '_')}_forecast.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def run_pipeline(data_path: str = DATA_PATH, zones: list = None, future_steps: int = 24):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["date"])
    if zones is None:
        zones = df["zone"].unique().tolist()

    all_metrics = []
    for zone in zones:
        print(f"\n  ARIMA for zone: {zone}")
        series = (df[df["zone"] == zone]
                  .set_index("date")["groundwater_level"]
                  .asfreq("MS")
                  .interpolate())

        split = int(len(series) * 0.8)
        train, test = series.iloc[:split], series.iloc[split:]

        # ADF stationarity check
        stationary = check_stationarity(train)
        print(f"    Stationary: {stationary}")

        order, aic = auto_select_order(train)
        print(f"    Best order: {order}  AIC: {aic:.2f}")

        fitted = fit_arima(train, order)
        forecast = forecast_arima(fitted, steps=len(test))

        metrics = regression_metrics(test.values, forecast, label=f"ARIMA_{zone}")
        all_metrics.append(metrics)

        plot_arima_forecast(series.values, forecast, zone, split, OUTPUT_DIR)
        plot_arima_diagnostics(fitted, zone, OUTPUT_DIR)

        # Future forecast
        future = forecast_arima(fitted, steps=future_steps)
        future_dates = pd.date_range(series.index[-1], periods=future_steps + 1,
                                     freq="MS")[1:]

        future_df = pd.DataFrame({"date": future_dates,
                                  "zone": zone,
                                  "predicted_gw": future})
        future_path = os.path.join(OUTPUT_DIR,
                                   f"ARIMA_{zone.replace(' ', '_')}_future.csv")
        future_df.to_csv(future_path, index=False)

        # Persist fitted model
        fitted.save(os.path.join(MODEL_DIR,
                                 f"ARIMA_{zone.replace(' ', '_')}.pkl"))

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(os.path.join(OUTPUT_DIR, "ARIMA_metrics.csv"), index=False)
    print("\n=== ARIMA Summary ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_pipeline()
