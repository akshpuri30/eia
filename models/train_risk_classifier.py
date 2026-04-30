"""
Risk Classification Module
Trains Logistic Regression and Random Forest classifiers to predict
groundwater depletion risk (Low / Medium / High).
Includes SHAP explainability, calibration curves, and probability outputs.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
from xgboost import XGBClassifier

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("shap not installed — SHAP explanations disabled")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import (
    load_data, engineer_features, split_train_test, scale_features
)
from utils.evaluation import plot_confusion_matrix

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "chennai_groundwater.csv")

RISK_LABELS = ["Low", "Medium", "High"]


def build_classifiers():
    return {
        "Logistic_Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "saga"]}
        ),
        "RF_Classifier": (
            RandomForestClassifier(n_jobs=-1, random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth":    [None, 10, 20],
                "min_samples_split": [2, 5],
            }
        ),
        "XGB_Classifier": (
            XGBClassifier(objective="multi:softprob", num_class=3,
                          random_state=42, n_jobs=-1, verbosity=0,
                          eval_metric="mlogloss"),
            {
                "n_estimators":  [100, 200],
                "max_depth":     [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
            }
        ),
    }


def tune_classifier(name, model, pgrid, X_train, y_train, tscv):
    print(f"  Tuning {name}...")
    search = RandomizedSearchCV(
        model, pgrid, n_iter=15, cv=tscv,
        scoring="f1_macro", n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"    Best params: {search.best_params_}")
    print(f"    CV F1 macro: {search.best_score_:.4f}")
    return best


def compute_shap(model, X_train: np.ndarray, X_test: np.ndarray,
                 feature_names: list, model_name: str, out_dir: str):
    """Compute SHAP values and save summary plot."""
    if not SHAP_OK:
        return

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train)

        shap_vals = explainer.shap_values(X_test[:200])

        # For multi-class, shap_vals is a list; use class 2 (High risk)
        if isinstance(shap_vals, list):
            sv = shap_vals[2]
        else:
            sv = shap_vals

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_test[:200], feature_names=feature_names,
                          show=False, max_display=20)
        plt.title(f"SHAP Summary — {model_name} (High Risk class)")
        plt.tight_layout()
        path = os.path.join(out_dir, f"SHAP_{model_name}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  SHAP plot saved → {path}")
    except Exception as e:
        print(f"  SHAP failed for {model_name}: {e}")


def plot_feature_importance_cls(model, feature_names, name, out_dir):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], imp[idx], color="coral")
    ax.set_title(f"{name} — Feature Importances")
    plt.tight_layout()
    path = os.path.join(out_dir, f"CLS_{name}_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()


def run_pipeline(data_path: str = DATA_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Loading & engineering features ===")
    raw = load_data(data_path)
    eng = engineer_features(raw)

    (X_train, X_test, y_train_reg, y_test_reg,
     y_train_cls, y_test_cls, feature_cols) = split_train_test(eng)

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Use StratifiedKFold for classifiers — TimeSeriesSplit produces single-class folds
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, (model, pgrid) in build_classifiers().items():
        print(f"\n--- {name} ---")
        best = tune_classifier(name, model, pgrid,
                               X_train_sc, y_train_cls, skf)
        y_pred = best.predict(X_test_sc)

        report = classification_report(
            y_test_cls, y_pred,
            labels=[0, 1, 2], target_names=RISK_LABELS, output_dict=True
        )
        print(classification_report(y_test_cls, y_pred,
                                    labels=[0, 1, 2], target_names=RISK_LABELS))

        plot_confusion_matrix(y_test_cls, y_pred, name, OUTPUT_DIR)
        plot_feature_importance_cls(best, feature_cols, name, OUTPUT_DIR)
        compute_shap(best, X_train_sc, X_test_sc, feature_cols, name, OUTPUT_DIR)

        joblib.dump(best, os.path.join(MODEL_DIR, f"CLS_{name}.pkl"))
        results[name] = report["macro avg"]["f1-score"]

    # Best classifier
    best_name = max(results, key=results.get)
    best_model = joblib.load(os.path.join(MODEL_DIR, f"CLS_{best_name}.pkl"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_classifier.pkl"))
    print(f"\nBest classifier: {best_name}  (F1={results[best_name]:.4f})")
    return results


def predict_risk(features_scaled: np.ndarray,
                 model_path: str = None) -> tuple:
    """
    Predict risk for a single sample.
    Returns (label, probabilities_dict).
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    model = joblib.load(model_path)
    pred   = model.predict(features_scaled)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
    else:
        proba = np.zeros(3)
        proba[pred] = 1.0
    label = RISK_LABELS[pred]
    proba_dict = dict(zip(RISK_LABELS, proba.round(3)))
    return label, proba_dict


if __name__ == "__main__":
    run_pipeline()
