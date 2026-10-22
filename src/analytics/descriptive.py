from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.analytics import descriptive_summary
from src.utils.graphing import save_figure
from src.utils.paths import project_path


def run_descriptive(data: pd.DataFrame) -> dict:
    result = descriptive_summary(data)
    project_path("reports/tables/descriptive_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    data.select_dtypes(include=np.number).describe().T.to_csv(project_path("reports/tables/numeric_summary.csv"))
    if "vessel_type" in data:
        data["vessel_type"].value_counts(dropna=False).rename_axis("vessel_type").to_csv(project_path("reports/tables/vessel_type_counts.csv"))
        if "sog" in data:
            grouped = data.groupby("vessel_type", dropna=False, observed=True)["sog"].agg(["count", "mean", "median", "std"]).reset_index()
            grouped.to_csv(project_path("reports/tables/speed_by_vessel_type.csv"), index=False)
            fig, ax = plt.subplots(figsize=(11, 6))
            grouped.sort_values("median").tail(25).plot.barh(x="vessel_type", y="median", legend=False, ax=ax)
            ax.set(title="Median observed speed by vessel type (top 25 by median)", xlabel="Median speed over ground (knots)", ylabel="Vessel type code")
            save_figure(fig, "reports/figures/descriptive/speed_by_vessel_type.png")
    hourly = data.groupby(data["timestamp"].dt.floor("h")).size().rename("observations").rename_axis("hour_utc").reset_index()
    hourly.to_csv(project_path("reports/tables/observation_counts_over_time.csv"), index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly.plot(x="hour_utc", y="observations", ax=ax, legend=False)
    ax.set(title="AIS observations over time", xlabel="Time (UTC)", ylabel="Observations per hour")
    save_figure(fig, "reports/figures/descriptive/observation_counts_over_time.png")
    for field, filename, title, label in (("hour_utc", "hourly_traffic_pattern", "Hourly traffic pattern (UTC)", "Hour of day"), ("day_of_week_utc", "daily_traffic_pattern", "Daily traffic pattern (UTC; one date)", "Day of week (0=Monday)")):
        if field in data:
            counts = data.groupby(field).size().rename("observations").reset_index()
            counts.to_csv(project_path(f"reports/tables/{filename}.csv"), index=False)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(counts[field], counts["observations"])
            ax.set(title=title, xlabel=label, ylabel="Observations")
            save_figure(fig, f"reports/figures/descriptive/{filename}.png")
    completeness = data.isna().sum().rename("missing_count").to_frame()
    completeness["completeness_fraction"] = 1 - completeness["missing_count"] / max(1, len(data))
    completeness.to_csv(project_path("reports/tables/data_completeness.csv"))
    fig, ax = plt.subplots(figsize=(11, 6))
    (completeness["completeness_fraction"] * 100).sort_values().plot.barh(ax=ax)
    ax.set(title="Data completeness by field", xlabel="Present observations (%)", ylabel="Field")
    save_figure(fig, "reports/figures/descriptive/data_completeness.png")
    movement = data.groupby("vessel_id", observed=True).agg(observations=("timestamp", "size"), first_seen=("timestamp", "min"), last_seen=("timestamp", "max"))
    movement["duration_hours"] = (movement["last_seen"] - movement["first_seen"]).dt.total_seconds() / 3600
    movement.to_csv(project_path("reports/tables/vessel_movement_summaries.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    movement["observations"].clip(upper=movement["observations"].quantile(.99)).plot.hist(bins=40, ax=ax)
    ax.set(title="Vessel observation-count distribution (display clipped at 99th percentile)", xlabel="AIS observations per vessel", ylabel="Vessels")
    save_figure(fig, "reports/figures/descriptive/vessel_movement_summaries.png")
    spatial = data[["latitude", "longitude"]].sample(min(len(data), 10000), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.hist2d(spatial["longitude"], spatial["latitude"], bins=100, cmap="magma", cmin=1)
    fig.colorbar(image[3], ax=ax, label="Sampled position count")
    ax.set(title="Geographic traffic density (deterministic sample, n<=10,000)", xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
    save_figure(fig, "reports/figures/descriptive/geographic_traffic_density.png")
    if {"vessel_id", "timestamp", "longitude", "latitude", "elapsed_hours"}.issubset(data.columns):
        ordered = data.sort_values(["vessel_id", "timestamp"], kind="stable")
        previous_lon = ordered.groupby("vessel_id", observed=True)["longitude"].shift().to_numpy()
        previous_lat = ordered.groupby("vessel_id", observed=True)["latitude"].shift().to_numpy()
        valid = ordered["elapsed_hours"].le(6).fillna(False).to_numpy().copy()
        valid &= np.isfinite(previous_lon) & np.isfinite(previous_lat)
        valid &= ordered["longitude"].notna().to_numpy() & ordered["latitude"].notna().to_numpy()
        segment_indices = np.flatnonzero(valid)
        if len(segment_indices):
            segment_indices = segment_indices[:50000]
            current_lon = ordered["longitude"].to_numpy()
            current_lat = ordered["latitude"].to_numpy()
            segments = [[(previous_lon[i], previous_lat[i]), (current_lon[i], current_lat[i])] for i in segment_indices]
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.add_collection(LineCollection(segments, linewidths=.35, alpha=.12, rasterized=True))
            ax.autoscale()
            ax.set(title=f"Observed vessel route segments (gap-safe, n={len(segments):,})", xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
            save_figure(fig, "reports/figures/descriptive/route_density.png")
        else:
            result["route_density_not_generated"] = "No adjacent observations within the six-hour segment limit."
    else:
        result["route_density_not_generated"] = "Route segments require vessel, timestamp, WGS84 coordinates, and elapsed time."
    result["tables"] = sorted(str(path.relative_to(project_path("."))) for path in project_path("reports/tables").glob("*.csv"))
    project_path("artifacts/metrics/descriptive_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result