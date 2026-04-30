"""
ML Model Training Pipeline
Trains Linear Regression, Random Forest, and XGBoost regressors.
Includes GridSearch/RandomSearch hyperparameter tuning and cross-validation.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import (
    load_data, engineer_features, split_train_test, scale_features,
    get_feature_columns
)
from utils.evaluation import (
    regression_metrics, plot_predicted_vs_actual,
    plot_residuals, compare_models
)

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "chennai_groundwater.csv")


def build_models():
    """Return dict of model_name → (model, param_grid)."""
    return {
        "Ridge_Regression": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]}
        ),
        "Random_Forest": (
            RandomForestRegressor(n_jobs=-1, random_state=42),
            {
                "n_estimators":      [100, 200, 300],
                "max_depth":         [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "max_features":      ["sqrt", "log2"],
            }
        ),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=42,
                         n_jobs=-1, verbosity=0),
            {
                "n_estimators":  [100, 200, 300],
                "max_depth":     [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample":     [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        ),
        "Gradient_Boosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators":  [100, 200],
                "max_depth":     [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample":     [0.8, 1.0],
            }
        ),
    }


def tune_and_train(name, model, param_grid, X_train, y_train, cv_splits):
    print(f"\n  Tuning {name}...")
    search = RandomizedSearchCV(
        model, param_grid,
        n_iter=20, cv=cv_splits, scoring="neg_root_mean_squared_error",
        n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"    Best params: {search.best_params_}")
    print(f"    CV RMSE:     {-search.best_score_:.4f}")
    return best


def plot_feature_importance(model, feature_names, model_name, out_dir):
    """Plot top-20 feature importances (RF / XGBoost)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="steelblue")
    ax.set_title(f"{model_name} — Feature Importances (Top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    Saved → {path}")


def run_pipeline(data_path: str = DATA_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Loading & engineering features ===")
    raw = load_data(data_path)
    eng = engineer_features(raw)

    X_train, X_test, y_train, y_test, _, _, feature_cols = split_train_test(eng)
    X_train_sc, X_test_sc, scaler = scale_features(
        X_train, X_test,
        save_path=os.path.join(MODEL_DIR, "scaler.pkl")
    )

    # TimeSeriesSplit for CV
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    trained = {}
    for name, (model, pgrid) in build_models().items():
        best = tune_and_train(name, model, pgrid, X_train_sc, y_train, tscv)
        y_pred = best.predict(X_test_sc)

        metrics = regression_metrics(y_test, y_pred, label=name)
        results.append(metrics)
        trained[name] = best

        plot_predicted_vs_actual(y_test.values, y_pred, name, OUTPUT_DIR)
        plot_residuals(y_test.values, y_pred, name, OUTPUT_DIR)
        plot_feature_importance(best, feature_cols, name, OUTPUT_DIR)

        joblib.dump(best, os.path.join(MODEL_DIR, f"{name}.pkl"))

    comparison = compare_models(results, OUTPUT_DIR)
    comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

    # Save best model alias
    best_name = comparison.iloc[0]["model"]
    joblib.dump(trained[best_name], os.path.join(MODEL_DIR, "best_regressor.pkl"))
    joblib.dump(feature_cols,       os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print(f"\nBest model: {best_name}")
    return trained, comparison


if __name__ == "__main__":
    run_pipeline()
