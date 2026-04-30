"""
Master Training Script
Runs the full pipeline end-to-end:
  1. Generate dataset
  2. Train ML models (Ridge, RF, XGBoost, GBM)
  3. Train LSTM/GRU deep learning models
  4. Train ARIMA baselines
  5. Train risk classifiers (LR, RF, XGB)
  6. Run advanced modules (Hybrid, Climate, Anomaly)
  7. Generate GIS visualizations

Usage:
    python train_all.py [--skip-lstm] [--skip-arima] [--zones ZONE1 ZONE2]
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)


def banner(text):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main(args):
    t0 = time.time()

    # ── 1. Generate Dataset ────────────────────────────────────────────────
    banner("Step 1 / 7 — Generate Synthetic Dataset")
    from data.generate_dataset import generate_full_dataset
    data_dir = os.path.join(ROOT, "data")
    df = generate_full_dataset(output_dir=data_dir)

    # ── 2. ML Models ──────────────────────────────────────────────────────
    banner("Step 2 / 7 — Train ML Models (Ridge / RF / XGBoost)")
    from models.train_ml import run_pipeline as ml_pipeline
    ml_pipeline()

    # ── 3. LSTM / GRU ─────────────────────────────────────────────────────
    if not args.skip_lstm:
        banner("Step 3 / 7 — Train LSTM Models")
        from models.train_lstm import run_pipeline as lstm_pipeline
        zones = args.zones if args.zones else None
        lstm_pipeline(model_type="LSTM", zones=zones)
        lstm_pipeline(model_type="GRU",  zones=zones)
    else:
        print("  [skipped] LSTM/GRU training")

    # ── 4. ARIMA ──────────────────────────────────────────────────────────
    if not args.skip_arima:
        banner("Step 4 / 7 — Train ARIMA Baselines")
        from models.train_arima import run_pipeline as arima_pipeline
        arima_pipeline(zones=args.zones)
    else:
        print("  [skipped] ARIMA training")

    # ── 5. Risk Classifiers ───────────────────────────────────────────────
    banner("Step 5 / 7 — Train Risk Classifiers + SHAP")
    from models.train_risk_classifier import run_pipeline as cls_pipeline
    cls_pipeline()

    # ── 6. Advanced Modules ───────────────────────────────────────────────
    banner("Step 6 / 7 — Advanced: Hybrid, Climate, Anomaly")
    from models.advanced_modules import run_pipeline as adv_pipeline
    adv_pipeline()

    # ── 7. GIS Visualizations ─────────────────────────────────────────────
    banner("Step 7 / 7 — GIS & Spatial Analysis")
    from gis.spatial_analysis import run_pipeline as gis_pipeline
    gis_pipeline()

    elapsed = time.time() - t0
    banner(f"✅  ALL DONE  ({elapsed/60:.1f} min)")
    print(f"\n  Outputs saved to: {os.path.join(ROOT, 'outputs')}")
    print(f"  Models saved to:  {os.path.join(ROOT, 'models', 'saved')}")
    print(f"\n  Launch web app:\n    streamlit run webapp/app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chennai Groundwater Full Training Pipeline")
    parser.add_argument("--skip-lstm",  action="store_true",
                        help="Skip LSTM/GRU training (slow on CPU)")
    parser.add_argument("--skip-arima", action="store_true",
                        help="Skip ARIMA training")
    parser.add_argument("--zones", nargs="+", default=None,
                        help="Limit deep learning to specific zones")
    args = parser.parse_args()
    main(args)
