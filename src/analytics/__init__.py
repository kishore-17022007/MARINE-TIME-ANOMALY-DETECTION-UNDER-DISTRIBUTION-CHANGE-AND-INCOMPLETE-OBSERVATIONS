"""Exploratory, descriptive, and diagnostic analytics."""

import pandas as pd
from scipy.stats import kruskal


def descriptive_summary(data: pd.DataFrame) -> dict:
	summary = {
		"observations": int(len(data)),
		"vessels": int(data["vessel_id"].nunique()),
		"time_start_utc": data["timestamp"].min().isoformat() if len(data) else None,
		"time_end_utc": data["timestamp"].max().isoformat() if len(data) else None,
		"missing_by_column": data.isna().sum().astype(int).to_dict(),
	}
	numeric = [name for name in ("sog", "cog", "heading", "distance_km", "elapsed_hours") if name in data]
	if numeric:
		summary["numeric_describe"] = data[numeric].describe().to_dict()
	if "vessel_type" in data:
		vessel_types = data["vessel_type"].astype("string").fillna("<missing>")
		summary["vessel_type_counts"] = vessel_types.value_counts().head(30).to_dict()
	return summary


def vessel_speed_group_test(data: pd.DataFrame, minimum_vessels: int = 5) -> dict:
	"""Kruskal-Wallis on per-vessel median speed; association is not causation."""
	if not {"vessel_type", "vessel_id", "sog"}.issubset(data.columns):
		return {"status": "not_run", "reason": "vessel_type, vessel_id, and sog are required"}
	vessel = data.groupby(["vessel_type", "vessel_id"], observed=True)["sog"].median().dropna().reset_index()
	groups = [group["sog"].to_numpy() for _, group in vessel.groupby("vessel_type", observed=True) if len(group) >= minimum_vessels]
	if len(groups) < 2:
		return {"status": "not_run", "reason": "fewer than two vessel types have enough vessel-level observations"}
	statistic, p_value = kruskal(*groups)
	return {
		"status": "computed", "test": "Kruskal-Wallis on per-vessel median speed",
		"groups": len(groups), "vessel_sample_sizes": [int(len(group)) for group in groups],
		"statistic": float(statistic), "p_value": float(p_value),
		"interpretation": "An association between vessel-type groups and median speed; not a causal result.",
	}