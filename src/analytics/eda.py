from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.graphing import save_figure
from src.utils.paths import project_path


FIGURES = "reports/figures/eda"


def run_eda(data: pd.DataFrame, sample_size: int = 5000, seed: int = 42) -> dict:
    sns.set_theme(style="whitegrid", context="notebook")
    figures, reasons = [], {}
    sample = data.sample(min(len(data), sample_size), random_state=seed) if len(data) else data.copy()

    def save(name, fig):
        figures.append(str(save_figure(fig, f"{FIGURES}/{name}")))

    missing = data.isna().mean().sort_values(ascending=False)
    if len(missing):
        fig, ax = plt.subplots(figsize=(12, 6))
        (missing * 100).plot.bar(ax=ax)
        ax.set(title="Missing values by field", xlabel="Field", ylabel="Missing observations (%)")
        ax.tick_params(axis="x", rotation=60)
        save("01_missing_values.png", fig)
    audit_fields = [name for name in data if name.startswith("audit_")]
    if audit_fields:
        fig, ax = plt.subplots(figsize=(10, 5))
        data[audit_fields].sum().sort_values(ascending=False).plot.bar(ax=ax)
        ax.set(title="Data quality flags in processed sample", xlabel="Audit flag", ylabel="Flagged observations")
        ax.tick_params(axis="x", rotation=35)
        save("02_data_quality_summary.png", fig)
    else:
        reasons["02_data_quality_summary.png"] = "No audit flags were present in the supplied data."

    if "sog" in sample and sample["sog"].notna().any():
        for number, suffix, kind in ((3, "histogram", "hist"), (4, "boxplot", "box"), (5, "violinplot", "violin")):
            fig, ax = plt.subplots(figsize=(9, 5))
            values = sample["sog"].dropna()
            if kind == "hist":
                sns.histplot(values, bins=50, ax=ax)
                ax.set(xlabel="Speed over ground (knots)", ylabel="Observations")
            elif kind == "box":
                sns.boxplot(x=values, ax=ax)
                ax.set(xlabel="Speed over ground (knots)", ylabel="")
            else:
                sns.violinplot(x=values, cut=0, ax=ax)
                ax.set(xlabel="Speed over ground (knots)", ylabel="")
            ax.set_title(f"Speed over ground {suffix}")
            save(f"{number:02d}_speed_{suffix}.png", fig)
    else:
        for name in ("03_speed_histogram.png", "04_speed_boxplot.png", "05_speed_violinplot.png"):
            reasons[name] = "Speed-over-ground field is unavailable or entirely missing."

    if "cog" in sample and sample["cog"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(sample["cog"].dropna().astype("float64") % 360, bins=np.arange(0, 361, 10), ax=ax)
        ax.set(title="Course over ground distribution", xlabel="Course (degrees)", ylabel="Observations")
        save("06_course_distribution.png", fig)
    else:
        reasons["06_course_distribution.png"] = "Course-over-ground field is unavailable or entirely missing."
    if "vessel_type" in sample and sample["vessel_type"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sample["vessel_type"].astype(str).value_counts().head(25).sort_values().plot.barh(ax=ax)
        ax.set(title="Most frequent vessel type codes", xlabel="Observations", ylabel="Vessel type code")
        save("07_vessel_type_frequency.png", fig)
    else:
        reasons["07_vessel_type_frequency.png"] = "Vessel type field is unavailable or entirely missing."

    numeric = sample.select_dtypes(include=np.number).drop(columns=[c for c in sample if c.startswith("audit_")], errors="ignore")
    if numeric.shape[1] >= 2:
        for matrix, name, title in ((numeric.corr(), "08_correlation_heatmap.png", "Numeric feature correlation"), (numeric.cov(), "09_covariance_heatmap.png", "Numeric feature covariance")):
            fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * .6), max(6, matrix.shape[1] * .5)))
            sns.heatmap(matrix, cmap="vlag", center=0 if name.startswith("08") else None, ax=ax)
            ax.set_title(title)
            save(name, fig)
        selected = [c for c in ("sog", "distance_km", "elapsed_hours", "course_change_deg", "turn_rate_deg_per_hour") if c in numeric]
        if len(selected) >= 2:
            pair = numeric[selected].sample(min(len(numeric), 1000), random_state=seed)
            grid = sns.pairplot(pair, corner=True, plot_kws={"s": 8, "alpha": .35})
            grid.figure.suptitle("Pairwise movement features (reproducible sample n<=1000)", y=1.02)
            figures.append(str(save_figure(grid.figure, f"{FIGURES}/10_pairplot.png")))
        else:
            reasons["10_pairplot.png"] = "Fewer than two suitable movement features are available."
    else:
        for name in ("08_correlation_heatmap.png", "09_covariance_heatmap.png", "10_pairplot.png"):
            reasons[name] = "At least two numeric fields are required."

    if {"sog", "turn_rate_deg_per_hour"}.issubset(sample.columns):
        plot = sample[["sog", "turn_rate_deg_per_hour"]].dropna()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(plot["sog"], plot["turn_rate_deg_per_hour"], s=8, alpha=.25)
        ax.set(title="Speed versus turning rate", xlabel="Speed over ground (knots)", ylabel="Turn rate (degrees/hour)")
        save("11_speed_vs_turning_rate.png", fig)
    else:
        reasons["11_speed_vs_turning_rate.png"] = "Requires both SOG and derived turn-rate fields."
    for field, name, title, label in (("elapsed_hours", "12_sampling_interval_distribution.png", "Sampling intervals (clipped at 24 h for display)", "Elapsed time (hours)"), ("distance_km", "13_trajectory_distance_distribution.png", "Trajectory segment distances (clipped at 100 km for display)", "Distance (km)")):
        if field in sample and sample[field].notna().any():
            fig, ax = plt.subplots(figsize=(9, 5))
            values = sample[field].dropna().clip(upper=24 if field == "elapsed_hours" else 100)
            sns.histplot(values, bins=48, ax=ax)
            ax.set(title=title, xlabel=label, ylabel="Observations")
            save(name, fig)
        else:
            reasons[name] = f"No usable {field} feature is available."

    if {"latitude", "longitude"}.issubset(sample.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        hist = ax.hist2d(sample["longitude"], sample["latitude"], bins=100, cmap="magma", cmin=1)
        fig.colorbar(hist[3], ax=ax, label="Sampled position count")
        ax.set(title="Geographic AIS traffic density (sampled positions)", xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
        save("14_geographic_traffic_density.png", fig)
    else:
        reasons["14_geographic_traffic_density.png"] = "Latitude and longitude are required."
    for field, name, title, label in (("hour_utc", "15_hourly_traffic_pattern.png", "Hourly observation counts (UTC)", "Hour of day (UTC)"), ("day_of_week_utc", "16_daily_traffic_pattern.png", "Daily observation counts (UTC)", "Day of week (UTC; 0=Monday)")):
        if field in data:
            fig, ax = plt.subplots(figsize=(9, 5))
            data[field].value_counts().sort_index().plot.bar(ax=ax)
            ax.set(title=title, xlabel=label, ylabel="Observations")
            save(name, fig)
        else:
            reasons[name] = f"Derived field {field} is unavailable."
    hourly = data.groupby(data["timestamp"].dt.floor("h")).size() if "timestamp" in data else pd.Series(dtype=float)
    if len(hourly) >= 2:
        fig, ax = plt.subplots(figsize=(11, 5))
        hourly.plot(ax=ax)
        ax.set(title="Observation counts over time", xlabel="Timestamp (UTC)", ylabel="Observations per hour")
        save("17_temporal_distribution_comparison.png", fig)
    else:
        reasons["17_temporal_distribution_comparison.png"] = "At least two timestamp bins are required."
    if "sog" in sample:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(x=sample["sog"].dropna(), ax=ax, showfliers=True)
        ax.set(title="Speed outlier screening (not labeled anomalies)", xlabel="Speed over ground (knots)", ylabel="")
        save("18_outlier_analysis.png", fig)
    else:
        reasons["18_outlier_analysis.png"] = "Speed field is unavailable."

    fields = [c for c in ("sog", "derived_speed_knots", "distance_km", "elapsed_hours", "course_change_deg", "turn_rate_deg_per_hour") if c in sample and sample[c].notna().any()]
    pca_data = sample[fields].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pca_data) >= 3 and len(fields) >= 2:
        projected = PCA(n_components=2, random_state=seed).fit_transform(StandardScaler().fit_transform(pca_data))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(projected[:, 0], projected[:, 1], s=8, alpha=.25)
        ax.set(title=f"PCA exploration (n={len(projected):,}, movement features)", xlabel="Principal component 1", ylabel="Principal component 2")
        save("19_pca_exploration.png", fig)
        correlations = pca_data.corr().abs()
        table = pd.DataFrame([{"feature": field, "max_abs_correlation": float(correlations[field].drop(field).max())} for field in fields])
        table.to_csv(project_path("reports/tables/feature_multicollinearity.csv"), index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        table.set_index("feature")["max_abs_correlation"].plot.bar(ax=ax)
        ax.set(title="Feature multicollinearity screen", xlabel="Feature", ylabel="Maximum absolute pairwise correlation")
        save("20_feature_multicollinearity.png", fig)
    else:
        reasons["19_pca_exploration.png"] = "Requires at least three complete rows and two movement features."
        reasons["20_feature_multicollinearity.png"] = "Insufficient complete numeric movement features."
    result = {"status": "completed", "rows": len(data), "seed": seed, "sample_size": len(sample), "figures": figures, "not_generated": reasons}
    project_path("artifacts/metrics/eda_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result