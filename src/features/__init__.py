"""Trajectory feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
	lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
	dlat, dlon = lat2 - lat1, lon2 - lon1
	value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
	return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


def engineer_trajectory_features(data: pd.DataFrame, maximum_gap_hours: float = 6) -> pd.DataFrame:
	frame = data.sort_values(["vessel_id", "timestamp"], kind="stable").copy()
	grouped = frame.groupby("vessel_id", sort=False)
	frame["elapsed_hours"] = grouped["timestamp"].diff().dt.total_seconds() / 3600
	frame["distance_km"] = haversine_km(
		grouped["latitude"].shift(), grouped["longitude"].shift(), frame["latitude"], frame["longitude"]
	)
	invalid_gap = frame["elapsed_hours"].le(0) | frame["elapsed_hours"].gt(maximum_gap_hours)
	frame.loc[invalid_gap, "distance_km"] = np.nan
	frame["derived_speed_knots"] = frame["distance_km"] / frame["elapsed_hours"] * 0.539957
	if "sog" in frame:
		frame["speed_change"] = grouped["sog"].diff()
		frame["acceleration_knots_per_hour"] = frame["speed_change"] / frame["elapsed_hours"]
	if "cog" in frame:
		course_delta = grouped["cog"].diff().astype("float64")
		frame["course_change_deg"] = ((course_delta + 180) % 360) - 180
		frame["turn_rate_deg_per_hour"] = frame["course_change_deg"] / frame["elapsed_hours"]
	frame["hour_utc"] = frame["timestamp"].dt.hour
	frame["day_of_week_utc"] = frame["timestamp"].dt.dayofweek
	return frame


def chronological_split(data: pd.DataFrame, train_fraction: float = 0.7, validation_fraction: float = 0.15):
	if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
		raise ValueError("Split fractions must be positive and sum to less than one.")
	ordered = data.sort_values("timestamp", kind="stable")
	if len(ordered) < 3:
		raise ValueError("At least three observations are required for chronological splitting.")
	first = max(1, min(len(ordered) - 2, int(len(ordered) * train_fraction)))
	second = max(first + 1, min(len(ordered) - 1, int(len(ordered) * (train_fraction + validation_fraction))))
	return ordered.iloc[:first].copy(), ordered.iloc[first:second].copy(), ordered.iloc[second:].copy()