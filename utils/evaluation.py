"""
Evaluation utilities: regression & classification metrics, residual analysis,
cross-validation helpers, and model comparison reporting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def regression_metrics(y_true, y_pred, label: str = "") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"[{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"model": label, "RMSE": rmse, "MAE": mae, "R2": r2}


def classification_metrics(y_true, y_pred, label: str = "") -> str:
    report = classification_report(y_true, y_pred,
                                   target_names=["Low", "Medium", "High"])
    if label:
        print(f"\n=== {label} Classification Report ===\n{report}")
    return report


def plot_predicted_vs_actual(y_true, y_pred, label: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time-series overlay
    axes[0].plot(np.array(y_true), label="Actual", alpha=0.7, linewidth=1)
    axes[0].plot(np.array(y_pred), label="Predicted", alpha=0.7, linewidth=1)
    axes[0].set_title(f"{label} — Predicted vs Actual")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Groundwater level (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    mn, mx = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    axes[1].plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"{label} — Scatter")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{label.replace(' ', '_')}_pred_vs_actual.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_residuals(y_true, y_pred, label: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, alpha=0.6, linewidth=0.8)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_title(f"{label} — Residuals")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_title(f"{label} — Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{label.replace(' ', '_')}_residuals.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, label: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{label} — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label.replace(' ', '_')}_confusion.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def compare_models(results: list, out_dir: str = "outputs") -> pd.DataFrame:
    """
    results: list of dicts with keys model, RMSE, MAE, R2
    Returns sorted DataFrame and saves bar chart.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results).sort_values("RMSE")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        colors = ["#2ecc71" if v == df[metric].min() else "#3498db"
                  for v in df[metric]] if metric != "R2" else \
                 ["#2ecc71" if v == df[metric].max() else "#3498db"
                  for v in df[metric]]
        ax.barh(df["model"], df[metric], color=colors)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\nModel comparison saved → {path}")
    return df
