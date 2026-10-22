from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta, timezone, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.analytics.eda import run_eda
from src.features import engineer_trajectory_features
from src.pipeline import ROOT, download_source, file_md5_base64, inspect_file, load_config, process_file
from src.research_pipeline import run_stage, STAGES
from src.utils.paths import project_path


START_DATE = date(2024, 1, 1)
DURATIONS = {"1-day": 1, "7-day": 7, "20-day": 20, "30-day": 30}
EXPERIMENT_ROOT = ROOT / "experiments"
DAILY_CACHE = ROOT / "data" / "interim" / "duration_daily_cache"
TOTAL_SAMPLE_SIZE = 200_000
DOWNLOAD_WORKERS = 4


def _source_config(day: date, config: dict) -> dict:
    day_config = copy.deepcopy(config)
    filename = f"ais-{day.isoformat()}.parquet"
    day_config["data"]["source_filename"] = filename
    day_config["data"]["source_url"] = (
        "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/"
        f"{filename}"
    )
    day_config["data"]["source_date"] = day.isoformat()
    day_config["data"]["raw_dir"] = str(ROOT / "data" / "raw")
    return day_config


def _days_for_duration(days: int) -> list[date]:
    return [START_DATE + timedelta(days=offset) for offset in range(days)]


def _maximum_cached_sample(day: date) -> int:
    offset = (day - START_DATE).days
    if offset == 0:
        return TOTAL_SAMPLE_SIZE
    if offset < 7:
        return math.ceil(TOTAL_SAMPLE_SIZE / 7)
    if offset < 20:
        return math.ceil(TOTAL_SAMPLE_SIZE / 20)
    return math.ceil(TOTAL_SAMPLE_SIZE / 30)


def _acquire_day(day: date, config: dict) -> tuple[date, Path, dict]:
    day_config = _source_config(day, config)
    path = download_source(day_config)
    inspection = inspect_file(path)
    inspection.update({
        "source_date_requested": day.isoformat(),
        "source_url": day_config["data"]["source_url"],
        "content_md5_base64": file_md5_base64(path),
    })
    return day, path, inspection


def _read_source_dates(path: Path, inspection: dict, aliases: dict) -> dict:
    candidates = [column["name"] for column in inspection["columns"]]
    mapping = {}
    from src.schema import discover_columns
    mapping = discover_columns(candidates, aliases)
    timestamp_col = mapping.get("timestamp")
    if timestamp_col is None:
        return {"status": "unverified", "reason": "No configured timestamp column in Parquet schema."}
    timestamps = pq.read_table(path, columns=[timestamp_col])[timestamp_col].to_pandas()
    timestamps = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna()
    actual_days = sorted(timestamps.dt.strftime("%Y-%m-%d").unique().tolist())
    return {
        "status": "verified" if actual_days == [inspection["source_date_requested"]] else "date_mismatch",
        "rows_with_valid_timestamp": int(len(timestamps)),
        "timestamp_min_utc": timestamps.min().isoformat() if len(timestamps) else None,
        "timestamp_max_utc": timestamps.max().isoformat() if len(timestamps) else None,
        "actual_utc_dates": actual_days,
        "requested_date": inspection["source_date_requested"],
    }


def _daily_sample(day: date, path: Path, inspection: dict, config: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = DAILY_CACHE / f"{day.isoformat()}.parquet"
    cache_meta = DAILY_CACHE / f"{day.isoformat()}.json"
    max_rows = _maximum_cached_sample(day)
    source_md5 = inspection["content_md5_base64"]
    sampling_version = "evenly_spaced_parquet_rows_v1"
    schema_config_hash = hashlib.sha256(
        json.dumps(config["data"]["columns"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache_file.exists() and cache_meta.exists():
        cached_metadata = json.loads(cache_meta.read_text(encoding="utf-8"))
        if (
            cached_metadata.get("source_md5_base64") == source_md5
            and cached_metadata.get("requested_sample_cap") == max_rows
            and cached_metadata.get("schema_config_sha256") == schema_config_hash
            and cached_metadata.get("sampling_version") == sampling_version
            and cached_metadata.get("seed") == seed
        ):
            return pd.read_parquet(cache_file), cached_metadata["quality"]

    cache_root = ROOT / "data" / "interim" / "duration_cache_work"
    cache_root.mkdir(parents=True, exist_ok=True)
    previous_output = os.environ.get("MARITIME_OUTPUT_DIR")
    os.environ["MARITIME_OUTPUT_DIR"] = str(cache_root)
    try:
        day_config = copy.deepcopy(config)
        day_config["data"]["raw_dir"] = str(ROOT / "data" / "raw")
        day_config["data"]["processed_dir"] = "processed"
        day_config["data"]["sample_rows"] = max_rows
        day_config["data"]["source_filename"] = path.name
        sampled, details = process_file(path, day_config)
    finally:
        if previous_output is None:
            os.environ.pop("MARITIME_OUTPUT_DIR", None)
        else:
            os.environ["MARITIME_OUTPUT_DIR"] = previous_output
    sampled.to_parquet(cache_file, index=False)
    cache_meta.write_text(json.dumps({
        "source_md5_base64": source_md5,
        "source_bytes": inspection["bytes"],
        "source_rows": inspection["rows"],
        "sample_rows": len(sampled),
        "requested_sample_cap": max_rows,
        "schema_config_sha256": schema_config_hash,
        "sampling_version": sampling_version,
        "seed": seed,
        "quality": details["quality"],
    }, indent=2, default=str), encoding="utf-8")
    return sampled, details["quality"]


def _prepare_experiment(duration_name: str, days: int, config: dict, seed: int) -> dict:
    dates = _days_for_duration(days)
    coverage = {}
    acquired = {}
    failures = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_acquire_day, day, config): day for day in dates}
        for future in as_completed(futures):
            day = futures[future]
            try:
                _, path, inspection = future.result()
                acquired[day] = (path, inspection)
                coverage[day.isoformat()] = _read_source_dates(path, inspection, config["data"]["columns"])
            except Exception as error:
                failures.append({"date": day.isoformat(), "error": f"{type(error).__name__}: {error}"})
                coverage[day.isoformat()] = {"status": "missing", "reason": str(error)}

    experiment_dir = EXPERIMENT_ROOT / duration_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "coverage.json").write_text(json.dumps({
        "requested_start": dates[0].isoformat(),
        "requested_end": dates[-1].isoformat(),
        "expected_dates": [day.isoformat() for day in dates],
        "coverage": coverage,
        "acquisition_failures": failures,
    }, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{duration_name}: {len(failures)} source days could not be acquired; see coverage.json")
    mismatches = [day for day, details in coverage.items() if details["status"] != "verified"]
    if mismatches:
        raise RuntimeError(f"{duration_name}: actual timestamps did not match requested dates: {mismatches}")

    previous_output = os.environ.get("MARITIME_OUTPUT_DIR")
    os.environ["MARITIME_OUTPUT_DIR"] = str(experiment_dir)
    frames = []
    quality_by_day = {}
    try:
        base_quota, remainder = divmod(TOTAL_SAMPLE_SIZE, len(acquired))
        for day_index, day in enumerate(dates):
            path, inspection = acquired[day]
            sampled, quality = _daily_sample(day, path, inspection, config, seed)
            quota = base_quota + int(day_index < remainder)
            if len(sampled) > quota:
                indices = np.linspace(0, len(sampled) - 1, quota, dtype=int)
                sampled = sampled.iloc[indices].copy()
            frames.append(sampled)
            quality_by_day[day.isoformat()] = {
                "source_rows": inspection["rows"],
                "sampled_rows": len(sampled),
                "quality": quality,
                "timestamp_coverage": coverage[day.isoformat()],
            }
    finally:
        if previous_output is None:
            os.environ.pop("MARITIME_OUTPUT_DIR", None)
        else:
            os.environ["MARITIME_OUTPUT_DIR"] = previous_output

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.sort_values(["vessel_id", "timestamp"], kind="stable")
    derived = ["elapsed_hours", "distance_km", "derived_speed_knots", "speed_change", "acceleration_knots_per_hour", "course_change_deg", "turn_rate_deg_per_hour", "hour_utc", "day_of_week_utc"]
    audit_columns = [column for column in data if column.startswith("audit_")]
    base_columns = [column for column in data if column not in derived]
    base_columns = [column for column in base_columns if column != "geometry"]
    data = data[base_columns]
    data = engineer_trajectory_features(data)
    output_dir = experiment_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_dir / "ais_sample.parquet", index=False)
    if audit_columns:
        data[audit_columns].to_parquet(output_dir / "cleaning_audit.parquet", index=False)
    (experiment_dir / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": duration_name,
        "requested_start": dates[0].isoformat(),
        "requested_end": dates[-1].isoformat(),
        "verified_dates": list(coverage),
        "missing_dates": [],
        "source_rows_total": int(sum(item["source_rows"] for item in quality_by_day.values())),
        "sample_rows": len(data),
        "sample_seed": seed,
        "sample_policy": "Equal total cap of 200,000; deterministic evenly spaced sample within each verified daily Parquet object.",
        "source_url_template": "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-YYYY-MM-DD.parquet",
        "daily_coverage": coverage,
        "daily_quality": quality_by_day,
    }
    (experiment_dir / "data" / "manifests" / "source_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    quality_summary = {
        "input_rows": int(sum(item["source_rows"] for item in quality_by_day.values())),
        "sampled_rows": int(sum(item["sampled_rows"] for item in quality_by_day.values())),
        "retained_rows": len(data),
        "excluded_unusable_rows": int(sum(item["quality"]["excluded_unusable_rows"] for item in quality_by_day.values())),
        "duplicate_rows_flagged": int(sum(item["quality"]["duplicate_rows_flagged"] for item in quality_by_day.values())),
        "daily_quality": quality_by_day,
    }
    (experiment_dir / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "artifacts" / "metrics" / "cleaning_report.json").write_text(json.dumps(quality_summary, indent=2, default=str), encoding="utf-8")
    experiment_config = copy.deepcopy(config)
    experiment_config["seed"] = seed
    experiment_config["data"]["raw_dir"] = str(ROOT / "data" / "raw")
    experiment_config["data"]["processed_dir"] = "data/processed"
    experiment_config["data"]["manifest_dir"] = "data/manifests"
    experiment_config["data"]["sample_rows"] = TOTAL_SAMPLE_SIZE
    config_path = experiment_dir / "experiment_config.yaml"
    config_path.write_text(yaml.safe_dump(experiment_config, sort_keys=False), encoding="utf-8")
    return {
        "duration": duration_name,
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "sample_rows": len(data),
        "coverage": coverage,
        "quality": quality_summary,
    }


def run_duration(duration_name: str, days: int, seed: int = 42) -> dict:
    config = load_config()
    prepared = _prepare_experiment(duration_name, days, config, seed)
    output = Path(prepared["experiment_dir"])
    config_path = prepared["config_path"]
    args = SimpleNamespace(
        input=None,
        start_date=START_DATE.isoformat(),
        end_date=(START_DATE + timedelta(days=days - 1)).isoformat(),
        bbox=None,
        sample_size=TOTAL_SAMPLE_SIZE,
        seed=seed,
        output_dir=str(output),
        config=config_path,
        skip_existing=False,
    )
    os.environ["MARITIME_OUTPUT_DIR"] = str(output)
    stages = [name for name, _ in STAGES if name not in ("01_download_data.py", "02_clean_data.py")]
    run_records = []
    try:
        for stage in stages:
            result = run_stage(stage, args)
            run_records.append({"stage": stage, "status": result.get("status", "completed"), "result": result if stage != "03_run_eda.py" else {"figure_count": len(result.get("figures", [])), "not_generated": result.get("not_generated", {})}})
    except Exception as error:
        prepared["status"] = "failed"
        prepared["failed_stage"] = stages[len(run_records)] if len(run_records) < len(stages) else "unknown"
        prepared["error"] = f"{type(error).__name__}: {error}"
        prepared["stages"] = run_records
        output.joinpath("experiment_summary.json").write_text(json.dumps(prepared, indent=2, default=str), encoding="utf-8")
        raise
    prepared["status"] = "completed"
    prepared["stages"] = run_records
    prepared["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.joinpath("experiment_summary.json").write_text(json.dumps(prepared, indent=2, default=str), encoding="utf-8")
    return prepared


def generate_master_report(results: list[dict]) -> Path:
    rows = []
    for result in results:
        result = dict(result)
        root = Path(result["experiment_dir"])
        summary_path = root / "experiment_summary.json"
        if summary_path.exists():
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved_summary.get("status") == "completed":
                result.update(saved_summary)
        baseline_path = root / "artifacts" / "metrics" / "baseline_evaluation.json"
        drift_path = root / "artifacts" / "metrics" / "drift_report.json"
        simulation_path = root / "artifacts" / "metrics" / "simulation_results.json"
        adaptive_path = root / "artifacts" / "metrics" / "fixed_vs_adaptive.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        drift = json.loads(drift_path.read_text()) if drift_path.exists() else {}
        simulation = json.loads(simulation_path.read_text()) if simulation_path.exists() else {}
        adaptive = json.loads(adaptive_path.read_text()) if adaptive_path.exists() else {}
        rows.append({
            "duration": result["duration"],
            "status": result.get("status"),
            "verified_dates": len(result.get("coverage", {})),
            "sample_rows": result.get("sample_rows"),
            "train_rows": baseline.get("train_rows"),
            "validation_rows": baseline.get("validation_rows"),
            "test_rows": baseline.get("test_rows"),
            "training_seconds": baseline.get("training_seconds"),
            "inference_seconds": baseline.get("inference_seconds"),
            "test_alerts": baseline.get("test_alerts"),
            "test_alert_percent": baseline.get("test_alert_percent"),
            "adaptive_status": adaptive.get("status", "not_run"),
            "adaptive_test_alerts": adaptive.get("adaptive_test_alerts"),
            "adaptive_test_alert_percent": adaptive.get("adaptive_test_alert_percent"),
            "adaptive_training_seconds": adaptive.get("adaptive_training_seconds"),
            "labeled_metrics_status": baseline.get("label_status", "not recorded"),
            "drift_events": len(drift.get("events", [])),
            "simulation_alerts": simulation.get("scenario_alerts"),
        })
    comparison = pd.DataFrame(rows)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(EXPERIMENT_ROOT / "master_comparison.csv", index=False)
    table = comparison.to_html(index=False, escape=True, border=1)
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Four-duration AIS experiment comparison</title><style>body{{font:16px/1.5 Georgia,serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%;font:14px system-ui}}th,td{{border:1px solid #aaa;padding:.45rem;text-align:left}}th{{background:#eef2f4}}.warning{{padding:1rem;border-left:4px solid #9a6630}}</style></head><body><h1>Four-duration AIS experiment comparison</h1><p>Starting date: 2024-01-01. Each duration uses the same configured detector and 200,000-row maximum deterministic sample. Samples are date-stratified from independently checksum-verified daily source objects. Missing or mismatched dates are recorded in each experiment's coverage manifest.</p><div class="warning">No ground-truth anomaly labels are available. Alert-count differences are not accuracy comparisons; no duration or adaptive model is declared superior.</div>{table}<h2>Per-experiment reports</h2><ul>{''.join(f'<li><a href="{html.escape(result["duration"] + "/reports/generated/maritime_anomaly_research_report.html")}">{html.escape(result["duration"])} report</a></li>' for result in results if result.get("status") == "completed")}</ul></body></html>"""
    master_path = EXPERIMENT_ROOT / "master_comparison_report.html"
    master_path.write_text(report, encoding="utf-8")
    return master_path


def main() -> None:
    global DOWNLOAD_WORKERS, TOTAL_SAMPLE_SIZE
    parser = argparse.ArgumentParser(description="Run isolated 1/7/20/30-day real AIS duration experiments.")
    parser.add_argument("--duration", choices=[*DURATIONS, "all"], default="all")
    parser.add_argument("--sample-size", type=int, default=TOTAL_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-workers", type=int, default=1)
    args = parser.parse_args()
    TOTAL_SAMPLE_SIZE = args.sample_size
    DOWNLOAD_WORKERS = args.download_workers
    selected = DURATIONS if args.duration == "all" else {args.duration: DURATIONS[args.duration]}
    results = []
    for name, days in selected.items():
        print(f"Running {name} from {START_DATE.isoformat()} for {days} day(s)", flush=True)
        try:
            results.append(run_duration(name, days, seed=args.seed))
        except Exception as error:
            result = {"duration": name, "status": "failed", "experiment_dir": str(EXPERIMENT_ROOT / name), "error": f"{type(error).__name__}: {error}"}
            results.append(result)
            print(f"FAILED {name}: {result['error']}", file=sys.stderr, flush=True)
    if args.duration == "all":
        print(f"Master report: {generate_master_report(results)}", flush=True)
    if any(result.get("status") != "completed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()