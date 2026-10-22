from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scipy.stats import chi2

from src.analytics import vessel_speed_group_test
from src.utils.graphing import save_figure
from src.utils.paths import project_path


def run_diagnostics(data: pd.DataFrame) -> dict:
    result = vessel_speed_group_test(data)
    result.update({"assumptions": ["Independent vessel-level medians are the analysis units.", "Kruskal-Wallis compares distributions without a normality assumption.", "Each eligible group must contain at least five vessels."], "effect_size": None, "confidence_interval": None})
    if result.get("status") == "computed":
        vessel = data.groupby(["vessel_type", "vessel_id"], observed=True)["sog"].median().dropna().reset_index()
        groups = [group["sog"].to_numpy() for _, group in vessel.groupby("vessel_type", observed=True) if len(group) >= 5]
        total = sum(len(group) for group in groups)
        epsilon_squared = max(0.0, (result["statistic"] - len(groups) + 1) / max(total - len(groups), 1))
        log_p_value = float(chi2.logsf(result["statistic"], len(groups) - 1))
        if result["p_value"] == 0.0:
            result["p_value_numeric_limit"] = f"< {np.finfo(float).tiny:.3g}"
            result["log_p_value"] = log_p_value
        else:
            result["log_p_value"] = float(np.log(result["p_value"]))
        result["epsilon_squared"] = epsilon_squared
        result["effect_size"] = {"name": "Kruskal-Wallis epsilon-squared", "value": epsilon_squared, "interpretation": "rank-based association magnitude, not causation"}
        generator = np.random.default_rng(42)
        bootstrap_effects = []
        for _ in range(300):
            resampled = [generator.choice(group, size=len(group), replace=True) for group in groups]
            bootstrap_h = float(kruskal(*resampled).statistic)
            bootstrap_effects.append(max(0.0, (bootstrap_h - len(groups) + 1) / max(total - len(groups), 1)))
        low, high = np.quantile(bootstrap_effects, [0.025, 0.975])
        result["confidence_interval"] = {"name": "95% percentile bootstrap interval for epsilon-squared", "lower": float(low), "upper": float(high), "resamples": len(bootstrap_effects), "seed": 42}
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(epsilon_squared, 0, xerr=[[epsilon_squared - low], [high - epsilon_squared]], fmt="o", capsize=5)
        ax.set(title="Bootstrap interval for Kruskal-Wallis epsilon-squared", xlabel="Epsilon-squared (association effect size)", yticks=[])
        save_figure(fig, "reports/figures/diagnostic/effect_size_confidence_interval.png")
        medians = [float(np.median(group)) for group in groups]
        fig, ax = plt.subplots(figsize=(11, 7))
        categories = sorted(vessel["vessel_type"].dropna().unique(), key=str)
        eligible_categories = [category for category in categories if int((vessel["vessel_type"] == category).sum()) >= 5]
        distributions = [vessel.loc[vessel["vessel_type"] == category, "sog"].to_numpy() for category in eligible_categories]
        ax.boxplot(distributions, vert=False, showfliers=False)
        ax.set_yticks(range(1, len(eligible_categories) + 1), labels=[str(category) for category in eligible_categories])
        ax.set(title="Vessel-level median speed by vessel-type code", xlabel="Median speed over ground (knots)", ylabel="Vessel type code (groups with >=5 vessels)")
        save_figure(fig, "reports/figures/diagnostic/group_distribution_comparison.png")
        result["group_medians_knots"] = medians
        result["group_sizes"] = [int(len(group)) for group in groups]
    else:
        result["status"] = "not_run"
    project_path("reports/tables/diagnostic_tests.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{"test": result.get("test"), "status": result.get("status"), "groups": result.get("groups"), "statistic": result.get("statistic"), "p_value": result.get("p_value"), "effect_size": result.get("epsilon_squared"), "reason": result.get("reason")}]).to_csv(project_path("reports/tables/diagnostic_tests.csv"), index=False)
    project_path("artifacts/metrics/diagnostics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result