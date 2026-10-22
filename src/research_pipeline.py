from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.descriptive import run_descriptive
from src.analytics.diagnostics import run_diagnostics
from src.analytics.eda import run_eda
from src.pipeline import ROOT, download_source, load_config, process_file, train_baseline
from src.utils.graphing import save_figure
from src.utils.logging_config import configure_logging
from src.utils.paths import ensure_output_dirs, project_path


STAGES = [
    ("01_download_data.py", "Acquire and inspect official data"),
    ("02_clean_data.py", "Clean and feature the bounded sample"),
    ("03_run_eda.py", "Generate exploratory figures"),
    ("04_run_descriptive_analytics.py", "Generate descriptive tables and figures"),
    ("05_run_diagnostic_analytics.py", "Run vessel-level diagnostics"),
    ("06_engineer_features.py", "Validate engineered movement features"),
    ("07_train_models.py", "Fit and persist baseline model"),
    ("08_detect_drift.py", "Monitor chronological score and feature shifts"),
    ("09_run_simulation.py", "Run controlled speed-change scenario"),
    ("10_evaluate_models.py", "Summarize held-out and simulation evaluation"),
    ("11_generate_report.py", "Generate HTML research report"),
]
REUSABLE_OUTPUTS = {
    "01_download_data.py": "data/manifests/source_manifest.json",
    "02_clean_data.py": "data/processed/ais_sample.parquet",
    "03_run_eda.py": "artifacts/metrics/eda_summary.json",
    "04_run_descriptive_analytics.py": "artifacts/metrics/descriptive_summary.json",
    "05_run_diagnostic_analytics.py": "artifacts/metrics/diagnostics.json",
    "06_engineer_features.py": "artifacts/metrics/feature_summary.json",
    "07_train_models.py": "artifacts/models/baseline_ensemble.joblib",
    "08_detect_drift.py": "artifacts/metrics/drift_report.json",
    "09_run_simulation.py": "artifacts/metrics/simulation_results.json",
    "10_evaluate_models.py": "artifacts/metrics/evaluation_summary.json",
    "11_generate_report.py": "reports/generated/maritime_anomaly_research_report.html",
}


def _base_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Maritime research stage: {stage}")
    parser.add_argument("--input", help="Input CSV or Parquet path")
    parser.add_argument("--start-date", help="Inclusive UTC start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Inclusive UTC end date (YYYY-MM-DD)")
    parser.add_argument("--bbox", help="Optional west,south,east,north geographic selection")
    parser.add_argument("--sample-size", type=int, help="Maximum reproducible analysis sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ROOT), help="Project/output root")
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse valid outputs when available")
    return parser


def _input_path(path: str | None):
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _get_data(args) -> pd.DataFrame:
    path = _input_path(args.input) if args.input else project_path("data/processed/ais_sample.parquet")
    if not path.exists():
        raw = project_path("data/raw/ais-2024-01-01.parquet")
        if raw.exists():
            config = load_config()
            config["data"]["sample_rows"] = int(args.sample_size or config["data"]["sample_rows"])
            _, details = process_file(raw, config)
            project_path("artifacts/metrics/cleaning_report.json").write_text(json.dumps(details["quality"], indent=2, default=str), encoding="utf-8")
            path = project_path("data/processed/ais_sample.parquet")
        else:
            raise FileNotFoundError("Processed AIS sample missing; run stage 01 then stage 02.")
    if path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError("Input must be a Parquet or CSV dataset.")
    if args.start_date:
        data = data[data["timestamp"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        data = data[data["timestamp"] < pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)]
    if args.bbox:
        west, south, east, north = map(float, args.bbox.split(","))
        data = data[data["longitude"].between(west, east) & data["latitude"].between(south, north)]
    if args.sample_size and len(data) > args.sample_size:
        data = data.sample(args.sample_size, random_state=args.seed).sort_values("timestamp", kind="stable")
    return data


def run_stage(stage: str, args) -> dict:
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    os.environ["MARITIME_OUTPUT_DIR"] = str(output_root.resolve())
    ensure_output_dirs()
    logger = configure_logging(f"maritime.{stage}")
    config = load_config(_input_path(args.config) if args.config else ROOT / "config" / "config.yaml")
    config["seed"] = args.seed
    if args.sample_size:
        config["data"]["sample_rows"] = args.sample_size

    if args.skip_existing and stage in REUSABLE_OUTPUTS:
        reusable = project_path(REUSABLE_OUTPUTS[stage])
        has_selection = any((args.input, args.start_date, args.end_date, args.bbox, args.sample_size, args.config)) or args.seed != 42
        if reusable.exists() and not has_selection:
            logger.info("Stage %s reused existing output: %s", stage, reusable)
            return {"status": "reused", "output": str(reusable)}
        logger.info("Stage %s not skipped: expected output missing or explicit input requires fresh work", stage)

    if stage == "01_download_data.py":
        raw_path = download_source(config)
        from src.pipeline import inspect_file
        manifest = inspect_file(raw_path)
        manifest.update({"source_url": config["data"]["source_url"], "accessed_at_utc": datetime.now(timezone.utc).isoformat(), "license": "CC0 1.0 Universal (source documentation)"})
        project_path("data/manifests/source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = {"status": "completed", "path": str(raw_path), "rows": manifest["rows"], "bytes": manifest["bytes"]}
    elif stage == "02_clean_data.py":
        input_path = _input_path(args.input) if args.input else project_path(config["data"]["raw_dir"]) / config["data"]["source_filename"]
        frame, details = process_file(input_path, config)
        project_path("artifacts/metrics/cleaning_report.json").write_text(json.dumps(details["quality"], indent=2, default=str), encoding="utf-8")
        result = {"status": "completed", "rows": len(frame), "quality": details["quality"]}
    elif stage == "03_run_eda.py":
        result = run_eda(_get_data(args), args.sample_size or 5000, args.seed)
    elif stage == "04_run_descriptive_analytics.py":
        result = run_descriptive(_get_data(args))
    elif stage == "05_run_diagnostic_analytics.py":
        result = run_diagnostics(_get_data(args))
    elif stage == "06_engineer_features.py":
        data = _get_data(args)
        features = [name for name in ("elapsed_hours", "distance_km", "derived_speed_knots", "speed_change", "acceleration_knots_per_hour", "course_change_deg", "turn_rate_deg_per_hour", "hour_utc", "day_of_week_utc") if name in data]
        from src.analytics.feature_selection import correlation_screen
        result = {"status": "completed", "rows": len(data), "features": features, "missing_by_feature": data[features].isna().sum().astype(int).to_dict(), "multicollinearity_screen": correlation_screen(data, features)}
        project_path("artifacts/metrics/feature_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif stage == "07_train_models.py":
        result = train_baseline(config)
        result["status"] = "completed"
    elif stage == "08_detect_drift.py":
        scores_path = project_path("artifacts/reports/baseline_test_scores.csv")
        if not scores_path.exists():
            raise FileNotFoundError("Baseline scores missing; run stage 07 first.")
        scores = pd.read_csv(scores_path)
        from src.drift.monitoring import detect_median_shift
        events = [event.__dict__ for event in detect_median_shift(scores["anomaly_score"].to_numpy())]
        result = {"status": "completed", "method": "median shift against first 100 score observations; not concept-drift proof", "scores": len(scores), "events": events, "interpretation": "Distribution-shift screen only; no adaptive update was performed."}
        project_path("artifacts/metrics/drift_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(12, 5))
        scores["anomaly_score"].rolling(500, min_periods=1).median().plot(ax=ax)
        ax.set(title="Held-out anomaly score monitoring (rolling median, window=500)", xlabel="Chronological test observation", ylabel="Fused anomaly score")
        save_figure(fig, "reports/figures/drift/anomaly_score_monitoring.png")
    elif stage == "09_run_simulation.py":
        model_path = project_path("artifacts/models/baseline_ensemble.joblib")
        if not model_path.exists():
            raise FileNotFoundError("Persisted detector missing; run stage 07 first.")
        data = _get_data(args)
        sizes = data.groupby("vessel_id", observed=True).size()
        eligible = sizes[sizes >= 30]
        if eligible.empty:
            result = {"status": "not_run", "reason": "No vessel in selected observations has at least 30 rows."}
        else:
            vessel_id = eligible.sort_values(ascending=False).index[0]
            track = data[data["vessel_id"] == vessel_id].sort_values("timestamp").head(2000).copy()
            from src.simulation.scenarios import inject_speed_scenario
            from src.features import engineer_trajectory_features
            modified = inject_speed_scenario(track, severity=1.8, start_fraction=.5)
            modified = engineer_trajectory_features(modified)
            simulation_copy = project_path("data/interim/simulation_speed_change.parquet")
            modified.to_parquet(simulation_copy, index=False)
            bundle = joblib.load(model_path)
            features = bundle["features"]
            medians = pd.Series(bundle["medians"])
            original_x = track[features].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy()
            altered_x = modified[features].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy()
            original_scores = bundle["model"].predict_scores(original_x)
            altered_scores = bundle["model"].predict_scores(altered_x)
            start = int(modified["injection_start_index"].iloc[0])
            injected = np.arange(len(modified)) >= start
            result = {"status": "completed", "is_simulated": True, "scenario": "controlled_speed_change", "vessel_id": str(vessel_id), "rows": len(track), "injection_start_index": start, "severity_multiplier": 1.8, "original_alerts": int(original_scores["flag"].sum()), "scenario_alerts": int(altered_scores["flag"].sum()), "injected_interval_detection_fraction": float(altered_scores["flag"][injected].mean()) if injected.any() else None, "warning": "Controlled synthetic intervention on a copy; not a real-world anomaly or operational detector validation."}
            table = modified[["timestamp", "sog"]].copy()
            table["is_injected"] = injected
            table["original_score"] = original_scores["score"]
            table["scenario_score"] = altered_scores["score"]
            table.to_csv(project_path("reports/tables/simulation_timeline.csv"), index=False)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(table["timestamp"], table["sog"], label="SOG with simulated intervention")
            ax.axvline(table["timestamp"].iloc[start], linestyle="--", label="simulated injection begins")
            ax.set(title="Controlled speed-change scenario (simulated copy)", xlabel="Timestamp (UTC)", ylabel="Speed over ground (knots)")
            ax.legend()
            save_figure(fig, "reports/figures/simulation/simulated_speed_change_timeline.png")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(table["timestamp"], table["original_score"], label="Original")
            ax.plot(table["timestamp"], table["scenario_score"], label="Simulated speed change")
            ax.axvline(table["timestamp"].iloc[start], linestyle="--", label="simulated injection begins")
            ax.set(title="Persisted baseline response to a controlled scenario", xlabel="Timestamp (UTC)", ylabel="Fused anomaly score")
            ax.legend()
            save_figure(fig, "reports/figures/simulation/simulation_score_response.png")
        project_path("artifacts/metrics/simulation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif stage == "10_evaluate_models.py":
        metric_path = project_path("artifacts/metrics/baseline_evaluation.json")
        legacy_path = project_path("artifacts/reports/baseline_evaluation.json")
        simulation_path = project_path("artifacts/metrics/simulation_results.json")
        baseline = metric_path if metric_path.exists() else legacy_path
        result = {"status": "completed", "real_label_metrics": "not computable: no ground-truth labels", "baseline": json.loads(baseline.read_text()), "simulation": json.loads(simulation_path.read_text()) if simulation_path.exists() else {"status": "incomplete"}, "roc_pr_curves": "not generated without valid labels"}
        project_path("artifacts/metrics/evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif stage == "11_generate_report.py":
        from src.reports.generate_report import generate_report
        result = generate_report()
    else:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    logger.info("Stage %s completed", stage)
    return result


def stage_main(stage: str) -> None:
    run_stage(stage, _base_parser(stage).parse_args())


def pipeline_main() -> None:
    parser = _base_parser("run_pipeline")
    parser.add_argument("--from-stage", choices=[stage for stage, _ in STAGES], default=STAGES[0][0])
    parser.add_argument("--to-stage", choices=[stage for stage, _ in STAGES], default=STAGES[-1][0])
    parser.add_argument("--stages", nargs="*", choices=[stage for stage, _ in STAGES])
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    os.environ["MARITIME_OUTPUT_DIR"] = str(output_root.resolve())
    names = [stage for stage, _ in STAGES]
    stages = args.stages or names[names.index(args.from_stage):names.index(args.to_stage) + 1]
    summary = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "stages": []}
    child_args = []
    for option, value in (("--start-date", args.start_date), ("--end-date", args.end_date), ("--bbox", args.bbox), ("--sample-size", args.sample_size), ("--seed", args.seed), ("--output-dir", args.output_dir), ("--config", args.config)):
        if value is not None:
            child_args.extend([option, str(value)])
    try:
        for script, title in STAGES:
            if script not in stages:
                continue
            reusable = project_path(REUSABLE_OUTPUTS[script])
            has_selection = any((args.input, args.start_date, args.end_date, args.bbox, args.sample_size, args.config)) or args.seed != 42
            if args.skip_existing and reusable.exists() and not has_selection:
                reason = f"expected output exists: {reusable}"
                print(f"REUSED {script}: {reason}", flush=True)
                summary["stages"].append({"script": script, "title": title, "status": "reused", "reason": reason})
                continue
            stage_args = list(child_args)
            if args.input and script in ("01_download_data.py", "02_clean_data.py", "03_run_eda.py", "04_run_descriptive_analytics.py", "05_run_diagnostic_analytics.py", "06_engineer_features.py"):
                stage_args.extend(["--input", args.input])
            command = [sys.executable, str(ROOT / "scripts" / script), *stage_args]
            if args.skip_existing:
                command.append("--skip-existing")
            print(f"EXECUTING {script}: {title}", flush=True)
            started = time.time()
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            summary["stages"].append({"script": script, "title": title, "status": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "duration_seconds": round(time.time() - started, 2)})
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
    except subprocess.CalledProcessError as error:
        summary.update({"status": "failed", "failed_stage_exit_code": error.returncode})
        project_path("reports/generated/execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Pipeline stopped at failed stage; later stages were not run.", file=sys.stderr)
        raise SystemExit(error.returncode)
    summary.update({"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    project_path("reports/generated/execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("All selected stages completed.")