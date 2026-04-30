"""
GIS & Spatial Analysis Module
Creates interactive Folium maps, risk heatmaps, and spatial interpolation
(IDW) for groundwater levels across Chennai zones.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional GIS libraries — gracefully degrade if not installed
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False
    print("folium not installed — interactive maps disabled")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_OK = True
except ImportError:
    GEOPANDAS_OK = False
    print("geopandas not installed — shapefile export disabled")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "chennai_groundwater.csv")

# Chennai bounding box
CHENNAI_LAT, CHENNAI_LON = 13.082, 80.270


def idw_interpolation(points: np.ndarray, values: np.ndarray,
                      grid_x: np.ndarray, grid_y: np.ndarray,
                      power: float = 2.0) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation.
    points: (N, 2) array of [lon, lat]
    values: (N,) array
    grid_x, grid_y: meshgrid arrays
    """
    grid_shape = grid_x.shape
    grid_pts   = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    result     = np.zeros(len(grid_pts))

    for i, gp in enumerate(grid_pts):
        dists = np.sqrt(((points - gp) ** 2).sum(axis=1))
        if np.any(dists == 0):
            result[i] = values[dists == 0][0]
        else:
            weights    = 1.0 / (dists ** power)
            result[i]  = np.sum(weights * values) / np.sum(weights)

    return result.reshape(grid_shape)


def get_zone_summary(df: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """Aggregate per-zone statistics (latest year or given year)."""
    if year is not None:
        df = df[df["date"].dt.year == year]
    else:
        df = df[df["date"] == df["date"].max()]

    agg = df.groupby("zone").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        gw_level=("groundwater_level", "mean"),
        risk_code=("risk_code", lambda x: x.mode()[0] if len(x) else 1),
        rainfall=("rainfall_mm", "mean"),
        urban_frac=("urban_fraction", "mean"),
    ).reset_index()

    agg["risk_label"] = agg["risk_code"].map({0: "Low", 1: "Medium", 2: "High"})
    return agg


def create_folium_map(zone_agg: pd.DataFrame, out_path: str) -> str:
    """Interactive folium map with zone markers and heatmap layer."""
    if not FOLIUM_OK:
        print("  Skipping folium map (not installed)")
        return ""

    m = folium.Map(location=[CHENNAI_LAT, CHENNAI_LON], zoom_start=11,
                   tiles="CartoDB positron")

    # Risk colour map
    risk_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

    for _, row in zone_agg.iterrows():
        color  = risk_colors.get(row["risk_label"], "blue")
        popup_html = f"""
        <b>{row['zone']}</b><br>
        GW Level: {row['gw_level']:.2f} m<br>
        Risk: <span style='color:{color}'><b>{row['risk_label']}</b></span><br>
        Rainfall: {row['rainfall']:.1f} mm<br>
        Urban Frac: {row['urban_frac']:.2f}
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=12,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['zone']} ({row['risk_label']})"
        ).add_to(m)

    # Heatmap layer (intensity = normalised GW level)
    max_gw = zone_agg["gw_level"].max()
    heat_data = [
        [r["latitude"], r["longitude"], r["gw_level"] / max_gw]
        for _, r in zone_agg.iterrows()
    ]
    HeatMap(heat_data, name="GW Depletion Heatmap",
            min_opacity=0.3, radius=40, blur=25).add_to(m)

    folium.LayerControl().add_to(m)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    print(f"  Interactive map saved → {out_path}")
    return out_path


def create_idw_map(zone_agg: pd.DataFrame, out_path: str):
    """Static matplotlib IDW interpolation map of GW levels."""
    lons = zone_agg["longitude"].values
    lats = zone_agg["latitude"].values
    vals = zone_agg["gw_level"].values

    # Grid covering Chennai bounding box
    lon_grid = np.linspace(lons.min() - 0.05, lons.max() + 0.05, 200)
    lat_grid = np.linspace(lats.min() - 0.05, lats.max() + 0.05, 200)
    xx, yy   = np.meshgrid(lon_grid, lat_grid)

    points   = np.column_stack([lons, lats])
    grid_val = idw_interpolation(points, vals, xx, yy)

    fig, ax = plt.subplots(figsize=(9, 8))
    cf = ax.contourf(xx, yy, grid_val, levels=20, cmap="RdYlGn_r", alpha=0.85)
    plt.colorbar(cf, ax=ax, label="Groundwater Level (m below surface)")

    scatter = ax.scatter(lons, lats, c=vals, cmap="RdYlGn_r",
                         edgecolors="black", s=120, zorder=5, linewidths=0.8)
    for _, row in zone_agg.iterrows():
        ax.annotate(row["zone"], (row["longitude"], row["latitude"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_title("Chennai Groundwater Level — IDW Interpolation", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  IDW map saved → {out_path}")
    return out_path


def create_risk_heatmap(zone_agg: pd.DataFrame, out_path: str):
    """Risk category bar chart + spatial scatter by risk."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    risk_counts = zone_agg["risk_label"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    axes[0].bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Risk Distribution Across Zones")
    axes[0].set_ylabel("Number of Zones")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Spatial scatter colored by risk
    risk_color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    for _, row in zone_agg.iterrows():
        axes[1].scatter(row["longitude"], row["latitude"],
                        c=risk_color_map[row["risk_label"]], s=200,
                        edgecolors="black", zorder=5, linewidths=0.8)
        axes[1].annotate(row["zone"], (row["longitude"], row["latitude"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8)

    for label, color in risk_color_map.items():
        axes[1].scatter([], [], c=color, label=label, s=100)
    axes[1].legend(title="Risk Level")
    axes[1].set_title("Spatial Risk Map — Chennai Zones")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Risk heatmap saved → {out_path}")
    return out_path


def create_trend_map(df: pd.DataFrame, out_path: str):
    """Annual mean GW level trend per zone."""
    annual = (df.groupby(["zone", df["date"].dt.year])["groundwater_level"]
              .mean().reset_index())
    annual.columns = ["zone", "year", "gw_level"]

    pivot = annual.pivot(index="year", columns="zone", values="gw_level")

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", markersize=3,
                linewidth=1.5, label=col)
    ax.set_title("Annual Mean Groundwater Level Trends — Chennai Zones")
    ax.set_xlabel("Year")
    ax.set_ylabel("Groundwater Level (m below surface)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Trend map saved → {out_path}")
    return out_path


def run_pipeline(data_path: str = DATA_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(data_path, parse_dates=["date"])
    zone_agg = get_zone_summary(df, year=2023)

    create_folium_map(zone_agg,
                      os.path.join(OUTPUT_DIR, "chennai_gw_map.html"))
    create_idw_map(zone_agg,
                   os.path.join(OUTPUT_DIR, "idw_groundwater_map.png"))
    create_risk_heatmap(zone_agg,
                        os.path.join(OUTPUT_DIR, "risk_heatmap.png"))
    create_trend_map(df,
                     os.path.join(OUTPUT_DIR, "gw_trend_map.png"))
    print("\nGIS pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
