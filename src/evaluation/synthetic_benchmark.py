"""Reproducible controlled AIS anomaly benchmark over held-out observations."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from src.features import chronological_split, engineer_trajectory_features, haversine_km


SCENARIOS = {
    "abrupt_speed_change": {"values": (3.0, 8.0, 15.0), "unit": "knots", "description": "Add a constant speed step to a held-out vessel segment; clamp SOG to 0-50 knots."},
    "sudden_deceleration": {"values": (3.0, 8.0, 15.0), "unit": "knots", "description": "Subtract speed from a held-out vessel segment, floored at 0 knots."},
    "position_jump": {"values": (2.0, 10.0, 30.0), "unit": "kilometres", "description": "Apply a spherical-geodesic displacement to a contiguous segment."},
    "route_deviation": {"values": (0.5, 2.0, 5.0), "unit": "kilometres", "description": "Apply a varying cross-track displacement using a spherical destination calculation."},
    "prolonged_stop": {"values": (8, 16, 24), "unit": "observations", "description": "Set SOG to zero and hold position constant for a segment selected from a vessel track."},
    "missing_observations": {"values": (0.1, 0.25, 0.5), "unit": "fraction", "description": "Mask movement fields on a reproducible fraction of rows to represent missing AIS reports; rows and timestamps are retained."},
    "movement_distribution_shift": {"values": (1.0, 4.0, 8.0), "unit": "knots", "description": "From a chronological change point onward, shift SOG and rotate course for all held-out vessels."},
}

MODELS = {
    "fixed": "baseline_ensemble.joblib",
    "shadow_adaptive": "adaptive_shadow_ensemble.joblib",
}


def _destination(lat: float, lon: float, distance_km: float, bearing_rad: float) -> tuple[float, float]:
    radius_km = 6371.0088
    lat1, lon1 = np.radians([lat, lon])
    angular = distance_km / radius_km
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular) + np.cos(lat1) * np.sin(angular) * np.cos(bearing_rad))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float((np.degrees(lon2) + 180) % 360 - 180)


def _event_indices(test: pd.DataFrame, scenario: str, severity: float, rng: np.random.Generator) -> tuple[str, pd.Index]:
    sizes = test.groupby("vessel_id", dropna=True, observed=True).size()
    eligible = sizes[sizes >= 12].sort_values(ascending=False)
    if eligible.empty:
        raise ValueError("Held-out data has no vessel track with at least 12 observations.")
    candidates = eligible.head(min(20, len(eligible)))
    vessel = str(rng.choice(candidates.index.to_numpy()))
    track = test[test["vessel_id"].astype(str) == vessel].sort_values("timestamp", kind="stable")
    width = min(len(track), max(4, int(severity) if scenario == "prolonged_stop" else min(24, max(8, len(track) // 4))))
    if scenario == "movement_distribution_shift":
        return vessel, test.index[int(len(test) * 0.5):]
    start = int(rng.integers(0, max(1, len(track) - width + 1)))
    return vessel, track.index[start:start + width]


def inject_scenario(
    original: pd.DataFrame,
    test_indices: pd.Index,
    scenario: str,
    severity: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return a modified full-context copy, labels indexed to held-out rows, and metadata."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    test = original.loc[test_indices]
    vessel, event_indices = _event_indices(test, scenario, severity, rng)
    modified = original.copy(deep=True)
    event_labels = pd.Series(False, index=test_indices, dtype=bool)
    event_labels.loc[event_indices] = True

    if scenario == "movement_distribution_shift":
        rows = modified.loc[event_indices]
        modified.loc[event_indices, "sog"] = (rows["sog"].fillna(0) + severity).clip(0, 50)
        if "cog" in modified:
            course = pd.to_numeric(rows["cog"], errors="coerce").fillna(0).to_numpy(dtype=float)
            modified.loc[event_indices, "cog"] = (course + min(60.0, severity * 6.0)) % 360
    elif scenario == "abrupt_speed_change":
        modified.loc[event_indices, "sog"] = (modified.loc[event_indices, "sog"].fillna(0) + severity).clip(0, 50)
    elif scenario == "sudden_deceleration":
        modified.loc[event_indices, "sog"] = (modified.loc[event_indices, "sog"].fillna(0) - severity).clip(lower=0)
    elif scenario in ("position_jump", "route_deviation"):
        track_rows = modified.loc[event_indices].sort_values("timestamp", kind="stable")
        for offset, (row_index, row) in enumerate(track_rows.iterrows()):
            if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
                continue
            bearing = 0.8 if scenario == "position_jump" else (0.8 + 0.35 * np.sin(offset / 3))
            distance = severity if scenario == "position_jump" else severity * (0.5 + 0.5 * np.sin(offset / 4) ** 2)
            modified.loc[row_index, ["latitude", "longitude"]] = _destination(
                float(row["latitude"]), float(row["longitude"]), distance, bearing
            )
    elif scenario == "prolonged_stop":
        track_rows = modified.loc[event_indices].sort_values("timestamp", kind="stable")
        first = track_rows.iloc[0]
        modified.loc[event_indices, "sog"] = 0.0
        modified.loc[event_indices, "latitude"] = first["latitude"]
        modified.loc[event_indices, "longitude"] = first["longitude"]
    elif scenario == "missing_observations":
        track_rows = modified.loc[event_indices].sort_values("timestamp", kind="stable")
        count = max(1, int(np.ceil(len(track_rows) * severity)))
        masked = track_rows.index[np.linspace(0, len(track_rows) - 1, count, dtype=int)]
        fields = [field for field in ("sog", "cog", "heading", "latitude", "longitude") if field in modified]
        modified.loc[masked, fields] = np.nan
        event_labels.loc[~event_labels.index.isin(masked)] = False
        event_labels.loc[masked] = True
        event_indices = masked

    metadata = {
        "scenario": scenario,
        "severity": severity,
        "severity_unit": SCENARIOS[scenario]["unit"],
        "seed": seed,
        "vessel_id": vessel,
        "event_start_utc": test.loc[event_indices, "timestamp"].min().isoformat(),
        "event_end_utc": test.loc[event_indices, "timestamp"].max().isoformat(),
        "event_rows": int(len(event_indices)),
        "description": SCENARIOS[scenario]["description"],
    }
    return modified, event_labels, metadata


def _model_input(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    features = bundle["features"]
    medians = pd.Series(bundle["medians"])
    missing = [name for name in features if name not in frame]
    if missing:
        raise ValueError(f"Model features are missing from benchmark frame: {missing}")
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(dtype=float)


def _score_model(frame: pd.DataFrame, bundle: dict) -> dict:
    """Transform with the persisted feature contract and pass the matrix to inference."""
    return bundle["model"].predict_scores(_model_input(frame, bundle))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def _evaluate(y_true: np.ndarray, scores: np.ndarray, flags: np.ndarray, timestamps: pd.Series) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, flags, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    precision_ci = _wilson(int(tp), int(tp + fp))
    recall_ci = _wilson(int(tp), int(tp + fn))
    fpr_ci = _wilson(int(fp), int(fp + tn))
    positives = np.flatnonzero(y_true)
    detected = np.flatnonzero((y_true > 0) & flags)
    delay = None
    if len(detected) and len(positives):
        event_start = pd.to_datetime(timestamps.iloc[positives].min(), utc=True)
        first_detection = pd.to_datetime(timestamps.iloc[detected].min(), utc=True)
        delay = max(0.0, (first_detection - event_start).total_seconds())
    return {
        "precision": float(precision), "precision_ci_low": precision_ci[0], "precision_ci_high": precision_ci[1],
        "recall": float(recall), "recall_ci_low": recall_ci[0], "recall_ci_high": recall_ci[1],
        "f1": float(f1), "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_positive_rate_ci_low": fpr_ci[0], "false_positive_rate_ci_high": fpr_ci[1],
        "detection_delay_seconds": delay,
    }


def _save_historical_scores(test: pd.DataFrame, scored: dict[str, dict], output: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    records = []
    observations = test[[column for column in ("vessel_id", "timestamp", "sog", "latitude", "longitude") if column in test]].copy()
    observations["classification"] = "real historical AIS analysis without ground-truth labels"
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, result in scored.items():
        scores = result["score"]
        threshold = result["threshold"]
        records.append({"model": name, "rows": len(scores), "flagged_rows": int(result["flag"].sum()),
                        "flagged_percent": float(result["flag"].mean() * 100), "score_mean": float(np.mean(scores)),
                        "score_std": float(np.std(scores)), "score_median": float(np.median(scores)),
                        "score_p95": float(np.quantile(scores, .95)), "score_p99": float(np.quantile(scores, .99)),
                        "training_calibrated_threshold": float(threshold),
                        "classification": "unlabeled historical AIS; flagged percentage is not a false-positive rate"})
        observations[f"{name}_anomaly_score"] = scores
        observations[f"{name}_anomaly_flag"] = result["flag"]
        ax.hist(scores, bins=60, alpha=.45, label=f"{name} (threshold={threshold:.3f})", density=True)
    ax.set(title="Untouched held-out historical AIS anomaly scores", xlabel="Fused anomaly score", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "historical_score_distributions.png", dpi=180)
    plt.close(fig)
    table = pd.DataFrame(records)
    table.to_csv(output / "historical_score_distributions.csv", index=False)
    observations.to_parquet(output / "held_out_scores.parquet", index=False)
    return table


def _write_plots(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summary = metrics.groupby(["model", "scenario"], as_index=False)[["precision", "recall", "f1"]].mean()
    for metric in ("precision", "recall", "f1"):
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = summary.pivot(index="scenario", columns="model", values=metric)
        pivot.plot.bar(ax=ax)
        ax.set(title=f"Synthetic-label {metric.title()} by model and anomaly type", ylabel=metric.title(), ylim=(0, 1))
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output / f"{metric}_by_anomaly_type.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for scenario, group in metrics.groupby("scenario"):
        severe = group[group["severity_rank"] == group["severity_rank"].max()]
        for model, model_group in severe.groupby("model"):
            axes[0].scatter(model_group["false_positive_rate"], model_group["recall"], label=f"{scenario}: {model}", s=36)
    axes[0].set(xlabel="Synthetic-label false-positive rate", ylabel="Recall", title="FPR and recall (highest severity)")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    detected = metrics[metrics["detection_delay_seconds"].notna()]
    axes[1].scatter(detected["severity_rank"], detected["detection_delay_seconds"], alpha=.45)
    undetected = int(metrics["detection_delay_seconds"].isna().sum())
    axes[1].set(xlabel="Severity rank", ylabel="Detection delay (seconds)", title=f"Event detection delay ({undetected} cases undetected)")
    fig.tight_layout()
    fig.savefig(output / "false_positive_rate_and_detection_delay.png", dpi=180)
    plt.close(fig)

    model_names = list(metrics["model"].unique())
    confusion = metrics.groupby(["model", "scenario"], as_index=False)[["true_negative", "false_positive", "false_negative", "true_positive"]].mean()
    fig, axes = plt.subplots(len(model_names), len(SCENARIOS), figsize=(18, max(5, 4 * len(model_names))), squeeze=False)
    for row, model in enumerate(model_names):
        for column, scenario in enumerate(SCENARIOS):
            cell = confusion[(confusion.model == model) & (confusion.scenario == scenario)]
            if cell.empty:
                axes[row, column].axis("off"); continue
            values = cell.iloc[0][["true_negative", "false_positive", "false_negative", "true_positive"]].to_numpy(dtype=float).reshape(2, 2)
            axes[row, column].imshow(values, cmap="Blues")
            axes[row, column].set(title=f"{model}\n{scenario}", xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Injected label")
            for (y, x), value in np.ndenumerate(values):
                axes[row, column].text(x, y, f"{value:,.0f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def _write_roc_pr_curves(cases: list[dict], seeds: tuple[int, ...], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred_seed = seeds[0]
    for model in MODELS:
        selected = [case for case in cases if case["model"] == model and case["seed"] == preferred_seed
                    and case["severity_rank"] == len(SCENARIOS[case["scenario"]]["values"])]
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        for case in selected:
            labels = np.r_[np.zeros(len(case["control_scores"]), dtype=int), np.ones(len(case["scores"]), dtype=int)]
            scores = np.r_[case["control_scores"], case["scores"]]
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            ax_roc.plot(fpr, tpr, label=f"{case['scenario']} AUC={auc:.2f}")
            ax_pr.plot(recall, precision, label=f"{case['scenario']} AP={ap:.2f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="#888888", label="chance")
        ax_roc.set(xlabel="Counterfactual-control false-positive rate", ylabel="Injected-event detection rate", title=f"ROC, severe scenarios, seed {preferred_seed}", xlim=(0, 1), ylim=(0, 1))
        ax_pr.set(xlabel="Injected-event recall", ylabel="Precision", title=f"Precision-recall, severe scenarios, seed {preferred_seed}", xlim=(0, 1), ylim=(0, 1))
        ax_roc.legend(fontsize=7); ax_pr.legend(fontsize=7)
        fig_roc.tight_layout(); fig_pr.tight_layout()
        fig_roc.savefig(output / f"roc_curves_{model}.png", dpi=180)
        fig_pr.savefig(output / f"precision_recall_curves_{model}.png", dpi=180)
        plt.close(fig_roc); plt.close(fig_pr)


def _evaluate_counterfactual(control_scores: np.ndarray, event_scores: np.ndarray, threshold: float, timestamps: pd.Series, seed: int = 42) -> dict:
    labels = np.r_[np.zeros(len(control_scores), dtype=int), np.ones(len(event_scores), dtype=int)]
    scores = np.r_[control_scores, event_scores]
    flags = scores >= threshold
    result = _evaluate(labels, scores, flags, pd.concat([timestamps.reset_index(drop=True), timestamps.reset_index(drop=True)], ignore_index=True))
    result.update(_bootstrap_auc_intervals(labels, scores, seed=seed))
    event_flags = event_scores >= threshold
    detected = np.flatnonzero(event_flags)
    delay = None
    if len(detected) and len(timestamps):
        event_start = pd.to_datetime(timestamps.iloc[0], utc=True)
        detection_time = pd.to_datetime(timestamps.iloc[int(detected[0])], utc=True)
        delay = max(0.0, (detection_time - event_start).total_seconds())
    result["detection_delay_seconds"] = delay
    result["event_rows_detected"] = int(event_flags.sum())
    result["event_rows_total"] = int(len(event_flags))
    result["counterfactual_control_rows"] = int(len(control_scores))
    result["classification_basis"] = "paired original counterfactual rows (control) versus modified copies (injected event); controls are not verified normal AIS"
    return result


def _bootstrap_auc_intervals(labels: np.ndarray, scores: np.ndarray, seed: int, resamples: int = 200) -> dict:
    if len(np.unique(labels)) < 2:
        return {"roc_auc_ci_low": None, "roc_auc_ci_high": None, "pr_auc_ci_low": None, "pr_auc_ci_high": None}
    rng = np.random.default_rng(seed)
    indices_by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    roc_values = []
    pr_values = []
    for _ in range(resamples):
        indices = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in indices_by_class])
        sample_labels = labels[indices]
        sample_scores = scores[indices]
        roc_values.append(roc_auc_score(sample_labels, sample_scores))
        pr_values.append(average_precision_score(sample_labels, sample_scores))
    return {
        "roc_auc_ci_low": float(np.quantile(roc_values, .025)),
        "roc_auc_ci_high": float(np.quantile(roc_values, .975)),
        "pr_auc_ci_low": float(np.quantile(pr_values, .025)),
        "pr_auc_ci_high": float(np.quantile(pr_values, .975)),
    }


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "scenario", "severity", "severity_unit", "severity_rank"]
    aggregate = metrics.groupby(keys, as_index=False).agg(
        precision_mean=("precision", "mean"), precision_seed_sd=("precision", "std"),
        recall_mean=("recall", "mean"), recall_seed_sd=("recall", "std"),
        f1_mean=("f1", "mean"), f1_seed_sd=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"),
        roc_auc_ci_low_mean=("roc_auc_ci_low", "mean"), roc_auc_ci_high_mean=("roc_auc_ci_high", "mean"),
        pr_auc_ci_low_mean=("pr_auc_ci_low", "mean"), pr_auc_ci_high_mean=("pr_auc_ci_high", "mean"),
        false_positive_rate_mean=("false_positive_rate", "mean"),
        detection_delay_seconds_mean=("detection_delay_seconds", "mean"),
        event_prevalence=("event_prevalence", "mean"),
    )
    aggregate["uncertainty_note"] = "Seed SD is descriptive across configured seeds; per-run metric Wilson intervals are in scenario_metrics_by_seed.csv."
    return aggregate


def run_experiment(experiment_dir: str | Path, seeds: tuple[int, ...] = (42, 43, 44)) -> dict:
    """Benchmark both persisted model bundles without modifying experiment inputs."""
    import joblib

    experiment = Path(experiment_dir)
    data_path = experiment / "data" / "processed" / "ais_sample.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed experiment data: {data_path}")
    data = pd.read_parquet(data_path).sort_values(["timestamp", "vessel_id"], kind="stable")
    train, validation, test = chronological_split(data)
    test_indices = test.index
    model_bundles = {}
    for model_name, filename in MODELS.items():
        model_path = experiment / "artifacts" / "models" / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing existing model bundle: {model_path}")
        model_bundles[model_name] = joblib.load(model_path)

    output = experiment / "reports" / "benchmark"
    labels_dir = output / "injected_test_cases"
    trajectories_dir = output / "synthetic_trajectories"
    historical_dir = output / "real_historical_unlabeled"
    figures_dir = output / "figures"
    labels_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    historical_scored = {}
    original_features = engineer_trajectory_features(data)
    original_test = original_features.loc[test_indices].sort_values("timestamp", kind="stable")
    for model_name, bundle in model_bundles.items():
        result = _score_model(original_test, bundle)
        historical_scored[model_name] = result
    historical_table = _save_historical_scores(original_test, historical_scored, historical_dir)

    records = []
    event_positions = []
    for scenario, definition in SCENARIOS.items():
        for severity_rank, severity in enumerate(definition["values"], start=1):
            for seed in seeds:
                modified, labels, metadata = inject_scenario(data, test_indices, scenario, severity, seed)
                featured = engineer_trajectory_features(modified)
                scenario_test = featured.loc[test_indices].sort_values("timestamp", kind="stable")
                scenario_labels = labels.loc[scenario_test.index].astype(int).to_numpy()
                event_mask = scenario_labels.astype(bool)
                event_indices = scenario_test.index[event_mask]
                event_frame = scenario_test.loc[event_indices]
                control_frame = original_test.loc[event_indices]
                source_test = data.loc[scenario_test.index]
                injection_rows = scenario_test.loc[scenario_labels.astype(bool)].copy()
                injection_rows.insert(0, "label_injected_anomaly", 1)
                injection_rows["scenario"] = scenario
                injection_rows["severity"] = severity
                injection_rows["severity_unit"] = definition["unit"]
                injection_rows["seed"] = seed
                injection_rows["experiment"] = experiment.name
                for column in ("sog", "latitude", "longitude", "cog"):
                    if column in source_test:
                        injection_rows[f"original_{column}"] = source_test.loc[injection_rows.index, column]
                case_name = f"{scenario}_severity-{severity_rank}_seed-{seed}"
                case_path = trajectories_dir / f"{case_name}.parquet"
                injection_rows.to_parquet(case_path, index=False)
                (labels_dir / f"{case_name}.json").write_text(json.dumps({
                    **metadata,
                    "label_semantics": "1 marks rows directly modified by the controlled injection",
                    "background_semantics": "not verified normal",
                }, indent=2), encoding="utf-8")

                for model_name, bundle in model_bundles.items():
                    result = _score_model(event_frame, bundle)
                    control_result = _score_model(control_frame, bundle)
                    event_times = event_frame["timestamp"].reset_index(drop=True)
                    metric = _evaluate_counterfactual(
                        control_result["score"], result["score"],
                        result["threshold"], event_times, seed=seed,
                    )
                    record = {**metadata, "experiment": experiment.name, "model": model_name,
                              "severity_rank": severity_rank, "event_prevalence": 0.5,
                              "threshold": float(result["threshold"]), "case_file": str(case_path.relative_to(experiment)),
                              "label_scope": "paired original event-row controls versus modified counterfactual copies; controls are not verified normal"}
                    record.update(metric)
                    records.append(record)
                    event_positions.append({"scenario": scenario, "severity": severity, "seed": seed,
                                            "model": model_name, "scores": result["score"],
                                            "control_scores": control_result["score"],
                                            "timestamps": event_times.to_numpy(),
                                            "severity_rank": severity_rank})

    metrics = pd.DataFrame(records)
    metrics_path = output / "scenario_metrics_by_seed.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = _summary(metrics)
    summary_path = output / "scenario_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    adaptation = _run_adaptation_comparison(data, train, validation, test, test_indices, model_bundles, seeds)
    adaptation_path = output / "fixed_vs_adaptive_drift.csv"
    adaptation.to_csv(adaptation_path, index=False)
    _write_plots(metrics, figures_dir)
    _write_roc_pr_curves(event_positions, seeds, figures_dir)
    _plot_adaptation(adaptation, figures_dir)

    result = {
        "status": "completed",
        "experiment": experiment.name,
        "evaluation_classification": "Controlled synthetic evaluation with known injected-event labels",
        "historical_classification": "Real historical AIS analysis without ground-truth labels",
        "verified_real_world_evaluation": "not available; no independently established event labels were provided",
        "models": list(model_bundles), "scenarios": list(SCENARIOS), "seeds": list(seeds),
        "held_out_rows": int(len(test)), "threshold_method": "existing model bundles; thresholds fitted from training data (adaptive shadow was fitted on train plus validation only)",
        "metrics_csv": str(metrics_path.relative_to(experiment)), "summary_csv": str(summary_path.relative_to(experiment)),
        "historical_scores_csv": str((historical_dir / "historical_score_distributions.csv").relative_to(experiment)),
        "historical_observation_scores": str((historical_dir / "held_out_scores.parquet").relative_to(experiment)),
        "injected_labels_dir": str(labels_dir.relative_to(experiment)),
        "synthetic_trajectories_dir": str(trajectories_dir.relative_to(experiment)),
        "adaptation_csv": str(adaptation_path.relative_to(experiment)),
        "figures_dir": str(figures_dir.relative_to(experiment)),
        "interpretation": "Metrics assess controlled injected scenarios only. Background rows are real held-out AIS without labels, so false-positive rates are relative to the injected-event target and are not verified real-world false-alarm rates.",
    }
    (output / "benchmark_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _run_adaptation_comparison(data, train, validation, test, test_indices, bundles, seeds):
    """Replay a shift and fit only on an earlier, unlabeled post-shift adaptation chunk."""
    from src.drift import detect_median_shift
    from src.models import AnomalyEnsemble

    rows = []
    control_test = engineer_trajectory_features(data).loc[test_indices]
    for seed in seeds:
        modified, labels, _ = inject_scenario(
            data, test_indices, "movement_distribution_shift",
            SCENARIOS["movement_distribution_shift"]["values"][1], seed,
        )
        scenario_test = engineer_trajectory_features(modified).loc[test_indices].sort_values("timestamp", kind="stable")
        control_test_ordered = control_test.loc[scenario_test.index]
        event_labels = labels.loc[scenario_test.index].to_numpy(dtype=bool)
        change_index = int(np.flatnonzero(event_labels)[0])
        change_time = pd.to_datetime(scenario_test["timestamp"].iloc[change_index], utc=True)

        fixed_scores = _score_model(scenario_test, bundles["fixed"])
        drift_event = next((event for event in detect_median_shift(fixed_scores["score"]) if event.index >= change_index), None)
        detected_index = min(len(scenario_test) - 1, drift_event.index + 49) if drift_event else None
        detected_time = pd.to_datetime(scenario_test["timestamp"].iloc[detected_index], utc=True) if detected_index is not None else None

        adaptation_start = change_index
        adaptation_end = min(len(scenario_test) - 1, change_index + max(100, int(len(scenario_test) * .1)))
        adaptation_chunk = scenario_test.iloc[adaptation_start:adaptation_end]
        evaluation_chunk = scenario_test.iloc[adaptation_end:]
        evaluation_indices = evaluation_chunk.index
        evaluation_labels = labels.loc[evaluation_indices].to_numpy(dtype=bool)
        control_chunk = control_test_ordered.loc[evaluation_indices]
        if not evaluation_labels.any() or adaptation_chunk.empty:
            continue

        evaluation_bundles = dict(bundles)
        fixed_bundle = bundles["fixed"]
        adaptation_values = _model_input(adaptation_chunk, fixed_bundle)
        refit = AnomalyEnsemble(
            contamination=fixed_bundle["model"].contamination,
            alert_quantile=fixed_bundle["model"].alert_quantile,
            seed=seed,
        ).fit(adaptation_values)
        evaluation_bundles["online_refit_after_drift"] = {
            "model": refit,
            "features": fixed_bundle["features"],
            "medians": fixed_bundle["medians"],
        }

        for model_name, bundle in evaluation_bundles.items():
            event_scores = _score_model(evaluation_chunk, bundle)
            control_scores = _score_model(control_chunk, bundle)
            paired = _evaluate_counterfactual(
                control_scores["score"][evaluation_labels], event_scores["score"][evaluation_labels],
                event_scores["threshold"],
                evaluation_chunk.loc[evaluation_labels, "timestamp"].reset_index(drop=True), seed=seed,
            )
            if model_name == "online_refit_after_drift":
                pre_change_alert_percent = None
                model_update_source = "refit on initial post-change unlabeled chunk only"
            else:
                pre_change = _score_model(scenario_test.iloc[:change_index], bundle)
                pre_change_alert_percent = float(pre_change["flag"].mean() * 100) if len(pre_change["flag"]) else None
                model_update_source = "saved before test; no post-drift update"
            rows.append({
                "seed": seed, "model": model_name, "model_update_source": model_update_source,
                "drift_introduced_at_utc": change_time.isoformat(),
                "drift_detected_at_utc": detected_time.isoformat() if detected_time is not None else None,
                "drift_detection_delay_seconds": max(0.0, (detected_time - change_time).total_seconds()) if detected_time is not None else None,
                "drift_detection_method": "existing median-shift screen (100-score baseline, 50-score windows, 4 MAD); detection time is end of first qualifying window",
                "adaptation_started_at_utc": pd.to_datetime(adaptation_chunk["timestamp"].min(), utc=True).isoformat(),
                "adaptation_ended_at_utc": pd.to_datetime(adaptation_chunk["timestamp"].max(), utc=True).isoformat(),
                "adaptation_rows": int(len(adaptation_chunk)),
                "adaptation_event": model_update_source,
                "evaluation_started_at_utc": pd.to_datetime(evaluation_chunk["timestamp"].min(), utc=True).isoformat(),
                "evaluation_rows": int(len(evaluation_chunk)),
                "pre_change_unlabeled_alert_percent": pre_change_alert_percent,
                "post_change_alert_percent": float(event_scores["flag"].mean() * 100),
                "post_change_precision": paired["precision"], "post_change_recall": paired["recall"],
                "post_change_f1": paired["f1"], "post_change_roc_auc": paired["roc_auc"],
                "post_change_pr_auc": paired["pr_auc"], "post_change_counterfactual_fpr": paired["false_positive_rate"],
                "ground_truth_note": "Injected rows are labeled; paired original rows are counterfactual controls, not certified normal AIS.",
            })
    return pd.DataFrame(rows)


def _plot_adaptation(adaptation: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    long = adaptation.melt(id_vars=["model", "seed"], value_vars=["pre_change_unlabeled_alert_percent", "post_change_alert_percent"], var_name="period", value_name="alert_percent")
    long["period"] = long["period"].str.replace("_alert_percent", "", regex=False).str.replace("_", " ")
    for model, group in long.groupby("model"):
        means = group.groupby("period")["alert_percent"].mean()
        axes[0].plot(means.index, means.values, marker="o", label=model)
    axes[0].set(title="Alert rate around controlled change", ylabel="Flagged rows (%)", xlabel="Period")
    for model, group in adaptation.groupby("model"):
        axes[1].plot(group["seed"], group["post_change_f1"], marker="o", label=f"{model} post-change F1")
    axes[1].set(title="Paired post-change synthetic F1", xlabel="Random seed", ylabel="F1", ylim=(0, 1))
    axes[0].legend()
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fixed_vs_adaptive_before_after_drift.png", dpi=180)
    plt.close(fig)

