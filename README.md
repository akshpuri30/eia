# Chennai Groundwater Prediction & Risk Assessment System
### AI-Powered · Machine Learning · Deep Learning · GIS · Time-Series Forecasting

---

## Project Overview

A production-ready, modular system that predicts groundwater levels across
10 Chennai zones and classifies depletion risk (Low / Medium / High) using
multi-source environmental data, advanced ML/DL models, and GIS spatial analysis.

---

## Architecture

```
final/
│
├── data/
│   └── generate_dataset.py        # Synthetic multi-source dataset generator
│
├── utils/
│   ├── preprocessing.py           # Feature engineering pipeline
│   └── evaluation.py              # Metrics & visualization helpers
│
├── models/
│   ├── train_ml.py                # Ridge, RF, XGBoost, GBM regressors
│   ├── train_lstm.py              # Bidirectional LSTM & GRU
│   ├── train_arima.py             # ARIMA time-series baseline
│   ├── train_risk_classifier.py   # Risk classification + SHAP
│   ├── advanced_modules.py        # Hybrid Physics+ML, Climate Scenarios, Anomaly Detection
│   └── saved/                     # Persisted .pkl / .keras model files
│
├── gis/
│   └── spatial_analysis.py        # Folium maps, IDW interpolation, heatmaps
│
├── webapp/
│   └── app.py                     # Streamlit interactive web application
│
├── outputs/                        # All generated plots and CSVs
├── train_all.py                    # Master end-to-end training script
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Training Pipeline
```bash
# Full pipeline (dataset → all models → GIS outputs)
python train_all.py

# Skip slow LSTM training (recommended for first run)
python train_all.py --skip-lstm --skip-arima
```

### 3. Launch Web App
```bash
streamlit run webapp/app.py
```

---

## Pipeline Modules

### Data Generation (`data/generate_dataset.py`)
- 10 Chennai zones, 2000–2023, monthly granularity (~2,880 rows)
- Features: rainfall, temperature, evapotranspiration, NDVI, population density,
  urban fraction, soil type, permeability, lat/lon, coastal flag
- Targets: `groundwater_level` (regression), `risk_level` (classification)

### Feature Engineering (`utils/preprocessing.py`)
- Lag features: t-1, t-3, t-6, t-12
- Rolling statistics: 3, 6, 12 month windows
- SPI drought index, urbanization composite index
- Cyclical month encoding (sin/cos), season indicators

### ML Models (`models/train_ml.py`)
| Model             | Tuning       |
|-------------------|--------------|
| Ridge Regression  | Alpha search |
| Random Forest     | RandomSearch |
| XGBoost           | RandomSearch |
| Gradient Boosting | RandomSearch |

### Deep Learning (`models/train_lstm.py`)
- Bidirectional LSTM & GRU
- Sliding window sequences (12 months lookback)
- Multi-step forecasting (3-month ahead)
- Early stopping, ReduceLR, ModelCheckpoint

### Time-Series Baseline (`models/train_arima.py`)
- Auto ARIMA order selection via AIC grid search
- ADF stationarity test
- 24-month future forecasting with diagnostic plots

### Risk Classification (`models/train_risk_classifier.py`)
- Logistic Regression, Random Forest, XGBoost classifiers
- SHAP explainability plots
- Confusion matrices, F1/precision/recall

### Advanced Modules (`models/advanced_modules.py`)
1. **Hybrid Physics+ML**: Darcy's Law recharge estimate + RF residual correction
2. **Climate Scenarios**: RCP 2.6 / 4.5 / 8.5 projections (30-year horizon)
3. **Anomaly Detection**: Isolation Forest + rolling z-score

### GIS Analysis (`gis/spatial_analysis.py`)
- Folium interactive maps with risk-colored markers and heatmap layer
- IDW spatial interpolation contour maps
- Zone risk scatter maps and annual trend plots

---

## Web Application Pages

| Page | Description |
|------|-------------|
| Dashboard | KPIs, trends, risk pie chart |
| Prediction | Input parameters → GW level gauge + risk probability bars |
| Analytics | Time series, rainfall correlation, seasonal patterns |
| GIS Maps | Interactive folium map + IDW static map |
| Risk Assessment | Zone-by-zone risk cards + bar chart |
| Climate Scenarios | Multi-scenario 30-year projections |
| Anomaly Detection | Isolation Forest + z-score with interactive threshold |

---

## Evaluation Metrics

- **Regression**: RMSE, MAE, R²
- **Classification**: Accuracy, Precision, Recall, F1-macro
- **Validation**: TimeSeriesSplit 5-fold cross-validation
- **Explainability**: SHAP summary plots

---

## Chennai-Specific Considerations

- **Northeast Monsoon dominance** (Oct–Dec): peak recharge season
- **Coastal zones** (Adyar, Sholinganallur, Manali): saline intrusion modeled
- **Rapid urbanization**: Velachery, Avadi, Sholinganallur have highest urban growth rates
- **Long-term depletion trend**: embedded in simulation via extraction + urbanization stress

---

## Technologies

Python · TensorFlow/Keras · scikit-learn · XGBoost · statsmodels ·
Folium · GeoPandas · Matplotlib · SHAP · Streamlit
