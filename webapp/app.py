"""
Chennai Groundwater Prediction & Risk Assessment System
Interactive Streamlit Web Application
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# Path setup — must happen before any local imports
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from models.advanced_modules import (
    GroundwaterAnomalyDetector, simulate_climate_scenario, CLIMATE_SCENARIOS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chennai Groundwater AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(ROOT, "data",   "chennai_groundwater.csv")
MODEL_DIR   = os.path.join(ROOT, "models", "saved")
OUTPUT_DIR  = os.path.join(ROOT, "outputs")

ZONE_COORDS = {
    "Tambaram":       (12.924, 80.127),
    "Adyar":          (13.001, 80.257),
    "Anna Nagar":     (13.085, 80.209),
    "Velachery":      (12.978, 80.220),
    "Sholinganallur": (12.901, 80.228),
    "Chromepet":      (12.952, 80.143),
    "Perambur":       (13.117, 80.233),
    "Manali":         (13.165, 80.263),
    "Avadi":          (13.115, 80.098),
    "Porur":          (13.035, 80.157),
}

RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
SOIL_TYPES  = ["Sandy", "Clayey", "Loamy", "Rocky", "Alluvial"]
SOIL_PERM   = {"Sandy": 0.9, "Clayey": 0.2, "Loamy": 0.6, "Rocky": 0.1, "Alluvial": 0.75}


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Run: python data/generate_dataset.py")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_resource
def load_models():
    models = {}
    for name in ["best_regressor", "best_classifier", "scaler",
                 "anomaly_detector", "hybrid_physics_ml"]:
        p = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(p):
            models[name] = joblib.load(p)
    feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    if os.path.exists(feature_path):
        models["feature_cols"] = joblib.load(feature_path)
    return models


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_badge(label: str) -> str:
    color = RISK_COLORS.get(label, "gray")
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:bold">{label}</span>'


def make_gauge(value: float, min_v=0, max_v=35, title="GW Level") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"aspect": "equal"})
    norm = (value - min_v) / (max_v - min_v)
    norm = np.clip(norm, 0, 1)
    theta = np.pi * (1 - norm)
    color = "#2ecc71" if norm < 0.33 else ("#f39c12" if norm < 0.66 else "#e74c3c")
    # Background arc
    t = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color="#ddd", linewidth=18, solid_capstyle="round")
    # Value arc
    t2 = np.linspace(0, np.pi * norm, 200)
    ax.plot(np.cos(t2), np.sin(t2), color=color, linewidth=18, solid_capstyle="round")
    ax.text(0, -0.15, f"{value:.2f} m", ha="center", va="center",
            fontsize=14, fontweight="bold", color=color)
    ax.text(0, -0.42, title, ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")
    plt.tight_layout()
    return fig


def build_input_vector(zone, soil, rainfall, temp, et, ndvi,
                       pop_density, urban_frac, year, month,
                       feature_cols):
    """Build a single-row feature DataFrame for prediction."""
    lat, lon = ZONE_COORDS[zone]
    is_coastal = int(zone in ["Adyar", "Sholinganallur", "Manali"])

    row = {
        "latitude":           lat,
        "longitude":          lon,
        "is_coastal":         is_coastal,
        "soil_type_enc":      SOIL_TYPES.index(soil),
        "soil_permeability":  SOIL_PERM[soil],
        "rainfall_mm":        rainfall,
        "temperature_c":      temp,
        "evapotranspiration": et,
        "ndvi":               ndvi,
        "population_density": pop_density,
        "urban_fraction":     urban_frac,
        "year":               year,
        "month":              month,
        "month_sin":          np.sin(2 * np.pi * month / 12),
        "month_cos":          np.cos(2 * np.pi * month / 12),
        "season":             2 if month in [10,11,12] else (1 if month in [6,7,8,9] else 0),
        "is_monsoon":         int(month in range(6, 13)),
        "spi":                0.0,
        "urbanization_index": 0.5 * urban_frac + 0.3 * (pop_density / 30000) + 0.2 * (1 - ndvi),
    }

    # Zone one-hot
    for z in ZONE_COORDS:
        row[f"zone_{z}"] = int(z == zone)

    # Lags & rolling (fill with sensible defaults when unknown)
    for lag in [1, 3, 6, 12]:
        row[f"gw_lag_{lag}"]   = 10.0
        row[f"rain_lag_{lag}"] = rainfall * 0.9

    for w in [3, 6, 12]:
        row[f"rain_roll_mean_{w}"] = rainfall
        row[f"temp_roll_mean_{w}"] = temp
        row[f"gw_roll_std_{w}"]    = 0.5

    df_row = pd.DataFrame([row])

    if feature_cols:
        # Align to trained feature columns
        for c in feature_cols:
            if c not in df_row.columns:
                df_row[c] = 0
        df_row = df_row[feature_cols]

    return df_row


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
_sidebar_img = os.path.join(os.path.dirname(__file__), "sidebar_map.png")
if os.path.exists(_sidebar_img):
    st.sidebar.image(_sidebar_img, use_container_width=True)
st.sidebar.title("💧 Chennai Groundwater AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "🗺️ GIS Maps",
     "⚠️ Risk Assessment", "🌡️ Climate Scenarios", "🔍 Anomaly Detection"]
)

df = load_dataset()
models = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("Chennai Groundwater Prediction & Risk Assessment System")
    st.markdown(
        """
        > An AI-powered platform integrating **Machine Learning**, **Deep Learning**,
        > **GIS spatial analysis**, and **Time-Series forecasting** to monitor and predict
        > groundwater depletion across Chennai's 10 administrative zones.
        """
    )

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        latest = df[df["date"] == df["date"].max()]
        avg_gw = latest["groundwater_level"].mean()
        high_risk = (latest["risk_code"] == 2).sum()
        col1.metric("Avg GW Level (latest)", f"{avg_gw:.2f} m")
        col2.metric("High Risk Zones", int(high_risk))
        col3.metric("Dataset Records", f"{len(df):,}")
        col4.metric("Zones Monitored", df["zone"].nunique())

        st.markdown("---")

        # Recent trend
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Annual Mean Groundwater Level")
            annual = df.groupby([df["date"].dt.year, "zone"])["groundwater_level"].mean().reset_index()
            annual.columns = ["year", "zone", "gw"]
            fig, ax = plt.subplots(figsize=(8, 4))
            for z in annual["zone"].unique():
                sub = annual[annual["zone"] == z]
                ax.plot(sub["year"], sub["gw"], linewidth=1.2, alpha=0.8, label=z)
            ax.set_xlabel("Year"); ax.set_ylabel("GW Level (m)")
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Risk Distribution (Latest Year)")
            risk_counts = latest["risk_level"].value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(risk_counts.values, labels=risk_counts.index,
                    colors=["#2ecc71", "#f39c12", "#e74c3c"],
                    autopct="%1.0f%%", startangle=140)
            ax2.set_title("Zone Risk Distribution")
            st.pyplot(fig2)
            plt.close()

        st.subheader("Zone Summary Table (Latest)")
        tbl = latest[["zone","groundwater_level","risk_level","rainfall_mm",
                       "temperature_c","urban_fraction"]].copy()
        tbl.columns = ["Zone","GW Level (m)","Risk","Rainfall (mm)","Temp (°C)","Urban Frac"]
        st.dataframe(tbl.set_index("Zone").style.format(
            {"GW Level (m)": "{:.2f}", "Rainfall (mm)": "{:.1f}",
             "Temp (°C)": "{:.1f}", "Urban Frac": "{:.2f}"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("Groundwater Level Prediction")
    st.info("Enter environmental parameters to predict groundwater level and depletion risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Location & Soil")
        zone       = st.selectbox("Chennai Zone", list(ZONE_COORDS.keys()))
        soil       = st.selectbox("Soil Type", SOIL_TYPES)
        month      = st.slider("Month", 1, 12, 6)
        year       = st.slider("Year", 2000, 2035, 2024)

    with col2:
        st.subheader("Climate Inputs")
        rainfall   = st.slider("Rainfall (mm)", 0.0, 600.0, 120.0, 5.0)
        temp       = st.slider("Temperature (°C)", 15.0, 48.0, 32.0, 0.5)
        et         = st.slider("Evapotranspiration (mm)", 0.0, 30.0, 15.0, 0.5)
        ndvi       = st.slider("NDVI (vegetation index)", 0.0, 1.0, 0.45, 0.01)

    with col3:
        st.subheader("Socio-Economic")
        pop_density = st.slider("Population Density (per km²)", 1000, 60000, 15000, 500)
        urban_frac  = st.slider("Urban Fraction", 0.0, 1.0, 0.55, 0.01)

    if st.button("🔮 Predict Groundwater Level & Risk", type="primary"):
        feature_cols = models.get("feature_cols")
        input_df     = build_input_vector(zone, soil, rainfall, temp, et,
                                          ndvi, pop_density, urban_frac,
                                          year, month, feature_cols)

        scaler = models.get("scaler")
        if scaler is not None:
            input_sc = scaler.transform(input_df)
        else:
            input_sc = input_df.values

        r_col, c_col = st.columns(2)

        with r_col:
            st.subheader("Groundwater Level Prediction")
            regressor = models.get("best_regressor")
            if regressor is not None:
                gw_pred = float(regressor.predict(input_sc)[0])
                fig_gauge = make_gauge(gw_pred, title=f"GW Level ({zone})")
                st.pyplot(fig_gauge)
                plt.close()
                st.metric("Predicted Groundwater Depth", f"{gw_pred:.2f} m",
                          help="Depth below ground surface")
            else:
                st.warning("Regressor not found. Run train_ml.py first.")

        with c_col:
            st.subheader("Depletion Risk Classification")
            classifier = models.get("best_classifier")
            if classifier is not None:
                risk_pred = classifier.predict(input_sc)[0]
                label     = ["Low", "Medium", "High"][risk_pred]
                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(input_sc)[0]
                else:
                    proba = np.zeros(3); proba[risk_pred] = 1.0

                st.markdown(f"### Risk Level: {risk_badge(label)}",
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                fig_prob, ax_p = plt.subplots(figsize=(5, 2.5))
                bars = ax_p.barh(["Low", "Medium", "High"], proba,
                                 color=["#2ecc71", "#f39c12", "#e74c3c"])
                ax_p.set_xlim(0, 1)
                ax_p.set_xlabel("Probability")
                ax_p.set_title("Risk Probability Distribution")
                for bar, v in zip(bars, proba):
                    ax_p.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                              f"{v:.1%}", va="center", fontsize=9)
                st.pyplot(fig_prob)
                plt.close()
            else:
                st.warning("Classifier not found. Run train_risk_classifier.py first.")

        st.markdown("---")
        st.subheader("Input Summary")
        st.json({
            "Zone": zone, "Soil": soil,
            "Rainfall_mm": rainfall, "Temperature_C": temp,
            "NDVI": ndvi, "Urban_Fraction": urban_frac,
            "Month": month, "Year": year,
        })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("Historical Analytics & Correlations")
    if df.empty:
        st.warning("Dataset not loaded.")
        st.stop()

    zone_sel = st.selectbox("Select Zone", df["zone"].unique())
    zdf = df[df["zone"] == zone_sel].copy()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Time Series", "Rainfall Correlation", "Seasonal Analysis", "Raw Data"]
    )

    with tab1:
        st.subheader(f"Groundwater Level Time Series — {zone_sel}")
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(zdf["date"], zdf["groundwater_level"], color="steelblue", linewidth=1)
        axes[0].set_ylabel("GW Level (m)")
        axes[0].set_title("Groundwater Level")
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(zdf["date"], zdf["rainfall_mm"], color="cornflowerblue", alpha=0.7, width=20)
        axes[1].set_ylabel("Rainfall (mm)")
        axes[1].set_title("Monthly Rainfall")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(zdf["date"], zdf["temperature_c"], color="tomato", linewidth=1)
        axes[2].set_ylabel("Temperature (°C)")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Temperature")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Rainfall vs Groundwater Level Correlation")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sc = ax2.scatter(zdf["rainfall_mm"], zdf["groundwater_level"],
                         c=zdf["date"].dt.year, cmap="viridis", alpha=0.5, s=15)
        plt.colorbar(sc, ax=ax2, label="Year")
        corr = zdf["rainfall_mm"].corr(zdf["groundwater_level"])
        ax2.set_xlabel("Rainfall (mm)")
        ax2.set_ylabel("Groundwater Level (m)")
        ax2.set_title(f"Pearson r = {corr:.3f}")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()
        st.info(
            "Negative correlation expected: higher rainfall → lower GW depth (recharge raises table)."
        )

    with tab3:
        st.subheader("Seasonal GW Patterns")
        monthly_avg = zdf.groupby(zdf["date"].dt.month)["groundwater_level"].mean()
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        months_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
        ax3.bar(months_lbl, monthly_avg.values, color="steelblue", edgecolor="white")
        ax3.set_title("Average Monthly Groundwater Level")
        ax3.set_ylabel("GW Level (m)")
        ax3.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig3)
        plt.close()
        st.caption("NE monsoon (Oct–Dec) recharge typically lowers GW depth.")

    with tab4:
        st.subheader("Raw Data Explorer")
        st.dataframe(zdf[["date","groundwater_level","rainfall_mm","temperature_c",
                           "urban_fraction","risk_level"]].reset_index(drop=True),
                     use_container_width=True)
        st.download_button(
            "Download Zone CSV", zdf.to_csv(index=False),
            file_name=f"{zone_sel}_data.csv", mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GIS MAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ GIS Maps":
    st.title("GIS & Spatial Analysis")

    # Interactive folium map
    map_path = os.path.join(OUTPUT_DIR, "chennai_gw_map.html")
    if os.path.exists(map_path):
        st.subheader("Interactive Groundwater Map")
        with open(map_path, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=False)
    else:
        st.info("Run gis/spatial_analysis.py to generate the interactive map.")

    col1, col2 = st.columns(2)
    with col1:
        idw_path = os.path.join(OUTPUT_DIR, "idw_groundwater_map.png")
        if os.path.exists(idw_path):
            st.subheader("IDW Interpolation Map")
            st.image(idw_path, use_container_width=True)

    with col2:
        risk_path = os.path.join(OUTPUT_DIR, "risk_heatmap.png")
        if os.path.exists(risk_path):
            st.subheader("Risk Heatmap")
            st.image(risk_path, use_container_width=True)

    trend_path = os.path.join(OUTPUT_DIR, "gw_trend_map.png")
    if os.path.exists(trend_path):
        st.subheader("Zone Trend Map")
        st.image(trend_path, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.title("Groundwater Depletion Risk Assessment")
    if df.empty:
        st.stop()

    year_sel  = st.slider("Select Year", int(df["date"].dt.year.min()),
                           int(df["date"].dt.year.max()), 2023)
    year_df   = df[df["date"].dt.year == year_sel].copy()
    zone_risk = (year_df.groupby("zone")
                 .agg(gw_mean=("groundwater_level","mean"),
                      risk_code=("risk_code", lambda x: x.mode()[0]))
                 .reset_index())
    zone_risk["risk_label"] = zone_risk["risk_code"].map(
        {0:"Low",1:"Medium",2:"High"})

    st.subheader(f"Zone Risk Overview — {year_sel}")
    cols = st.columns(5)
    for i, (_, row) in enumerate(zone_risk.iterrows()):
        with cols[i % 5]:
            color = RISK_COLORS.get(row["risk_label"], "#888")
            st.markdown(
                f"""<div style='border:2px solid {color};border-radius:10px;
                padding:10px;text-align:center;margin:4px'>
                <b>{row['zone']}</b><br>
                <span style='color:{color};font-size:1.1em'>{row['risk_label']}</span><br>
                <small>{row['gw_mean']:.2f} m</small></div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [RISK_COLORS[r] for r in zone_risk["risk_label"]]
    ax.bar(zone_risk["zone"], zone_risk["gw_mean"], color=colors, edgecolor="white")
    ax.set_title(f"Mean Groundwater Level by Zone ({year_sel})")
    ax.set_ylabel("GW Level (m)")
    ax.set_xticklabels(zone_risk["zone"], rotation=30, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for patch, label in zip(ax.patches, zone_risk["risk_label"]):
        ax.text(patch.get_x() + patch.get_width()/2, patch.get_height() + 0.15,
                label, ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    conf_path = os.path.join(OUTPUT_DIR, "RF_Classifier_confusion.png")
    if os.path.exists(conf_path):
        st.subheader("Classifier Confusion Matrix")
        st.image(conf_path, width=450)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLIMATE SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌡️ Climate Scenarios":
    st.title("Climate Change Scenario Simulation")
    st.markdown(
        """
        Project groundwater levels under IPCC RCP emission scenarios.
        - **RCP 2.6** — Aggressive mitigation (optimistic)
        - **RCP 4.5** — Moderate mitigation
        - **RCP 8.5** — Business-as-usual (pessimistic)
        """
    )

    if df.empty:
        st.stop()

    zone_sel = st.selectbox("Select Zone", list(ZONE_COORDS.keys()), key="cli_zone")
    n_years  = st.slider("Projection Horizon (years)", 10, 50, 30)

    if st.button("Run Scenario Simulation"):

        fig, ax = plt.subplots(figsize=(12, 5))
        palette = {"RCP_2.6_Optimistic": "#2ecc71",
                   "RCP_4.5_Moderate":   "#f39c12",
                   "RCP_8.5_Pessimistic":"#e74c3c"}

        for sc_name in CLIMATE_SCENARIOS:
            proj = simulate_climate_scenario(df, sc_name, n_years=n_years)
            zp   = proj[proj["zone"] == zone_sel].groupby("year")["gw_level_proj"].mean()
            ax.plot(zp.index, zp.values, label=sc_name.replace("_", " "),
                    color=palette[sc_name], linewidth=2, marker="o", markersize=3)

        hist = df[df["zone"] == zone_sel].groupby(
            df["date"].dt.year)["groundwater_level"].mean()
        ax.plot(hist.index, hist.values, label="Historical", color="steelblue",
                linewidth=1.5, linestyle="--")

        ax.set_title(f"Groundwater Projections — {zone_sel}")
        ax.set_xlabel("Year"); ax.set_ylabel("GW Level (m)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "Higher GW level (m) = deeper water table = greater depletion. "
            "RCP 8.5 scenario shows severe depletion by mid-century."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.title("Anomaly Detection — Sudden Groundwater Drops")
    st.markdown(
        "Detects anomalous groundwater behaviour using **Isolation Forest** "
        "and **rolling z-score** analysis."
    )

    if df.empty:
        st.stop()

    zone_sel = st.selectbox("Select Zone", df["zone"].unique(), key="anom_zone")
    threshold = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)

    if st.button("Run Anomaly Detection"):
        zdf = df[df["zone"] == zone_sel].copy().reset_index(drop=True)
        detector = GroundwaterAnomalyDetector(z_threshold=threshold)
        detector.fit(df)
        result = detector.detect(zdf)

        n_anom = result["anomaly"].sum()
        st.metric("Anomalies Detected", int(n_anom))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(result["date"], result["groundwater_level"],
                label="GW Level", color="steelblue", linewidth=1)
        anoms = result[result["anomaly"] == 1]
        ax.scatter(anoms["date"], anoms["groundwater_level"],
                   color="red", zorder=5, s=50, label=f"Anomaly (n={n_anom})", marker="x")
        ax.set_title(f"Anomaly Detection — {zone_sel}")
        ax.set_xlabel("Date"); ax.set_ylabel("GW Level (m)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if n_anom > 0:
            st.subheader("Anomalous Records")
            st.dataframe(anoms[["date","groundwater_level","z_score","iso_score"]]
                         .reset_index(drop=True), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Chennai Groundwater AI System · Built with Streamlit · "
    "Machine Learning + GIS + Time-Series Forecasting"
)
