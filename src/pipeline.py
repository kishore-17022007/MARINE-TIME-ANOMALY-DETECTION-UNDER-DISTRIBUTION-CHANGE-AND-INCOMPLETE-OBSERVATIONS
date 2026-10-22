from __future__ import annotations
import argparse
import base64
import hashlib
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import certifi
import pandas as pd
import pyarrow.parquet as pq
import yaml
import joblib
import matplotlib.pyplot as plt
from src.utils.graphing import save_figure
from src.utils.paths import project_path

from src.cleaning import canonicalize, clean_positions
from src.features import engineer_trajectory_features
from src.schema import discover_columns

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("maritime.pipeline")

def open_url(
    url: str,
    *,
    method: str = "GET",
    headers: dict | None = None,
    timeout: int = 60,
    retries: int = 5,
    backoff_seconds: float = 1.0,
):
    request = urllib.request.Request(url, headers=headers or {}, method=method)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(
                request,
                context=ssl.create_default_context(cafile=certifi.where()),
                timeout=timeout,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                raise
            sleep_for = backoff_seconds * attempt
            LOGGER.warning(
                "Request to %s failed on attempt %s/%s (%s); retrying in %.1f seconds",
                url,
                attempt,
                retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request to {url} failed without an exception.")

def file_md5_base64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def load_config(path: Path = ROOT / "config" / "config.yaml") -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def inspect_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"AIS source file not found: {path}. Run `python -m src.pipeline download` or follow the manual download instructions in README.md.")
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    try:
        geo_metadata = json.loads(parquet.schema_arrow.metadata.get(b"geo", b"{}").decode())
    except (AttributeError, json.JSONDecodeError):
        geo_metadata = {}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ],
        "geo_metadata": geo_metadata,
    }

def download_source(config: dict) -> Path:
    data = config["data"]
    raw_dir = project_path(data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / data["source_filename"]
    with open_url(data["source_url"], method="HEAD", timeout=30, retries=5, backoff_seconds=1.5) as response:
        expected_size = int(response.headers.get("Content-Length", "0"))
        expected_md5 = response.headers.get("Content-MD5")
    limit = int(data["maximum_download_mb"] * 1024 * 1024)
    if expected_size <= 0 or expected_size > limit:
        raise ValueError(f"Refusing download of {expected_size:,} bytes; configured cap is {limit:,} bytes.")
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if destination.stat().st_size == expected_size:
            actual_md5 = file_md5_base64(destination)
            if not expected_md5 or actual_md5 == expected_md5:
                LOGGER.info("Using verified existing raw file: %s", destination)
                return destination
        if not temporary.exists():
            destination.replace(temporary)
        else:
            raise FileExistsError(f"Both incomplete source files exist: {destination} and {temporary}; inspect them before proceeding.")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Partial file is larger than expected ({offset:,} > {expected_size:,} bytes).")
    if offset == expected_size:
        if expected_md5 and file_md5_base64(temporary) != expected_md5:
            raise IOError("Completed partial file failed the source Content-MD5 integrity check.")
        temporary.replace(destination)
        return destination
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    if offset:
        LOGGER.info("Resuming at %.1f / %.1f MB", offset / 2**20, expected_size / 2**20)
    digest = hashlib.md5()
    received = offset
    try:
        with open_url(data["source_url"], headers=headers, timeout=60, retries=5, backoff_seconds=1.5) as response:
            if offset and response.status != 206:
                offset = 0
                received = 0
            with temporary.open("ab" if offset else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    received += len(chunk)
                    if received > limit:
                        raise ValueError("Download exceeded configured size limit.")
                    if received % (25 * 1024 * 1024) < 1024 * 1024:
                        LOGGER.info("Downloaded %.1f / %.1f MB", received / 2**20, expected_size / 2**20)
        if received != expected_size:
            raise IOError(f"Incomplete download: expected {expected_size:,} bytes, received {received:,}.")
        actual_md5 = file_md5_base64(temporary)
        if expected_md5 and actual_md5 != expected_md5:
            raise IOError("Downloaded file failed the source Content-MD5 integrity check.")
        temporary.replace(destination)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise
    LOGGER.info("Verified download: %s (%s bytes)", destination, received)
    return destination

def process_file(path: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    details = inspect_file(path)
    aliases = config["data"]["columns"]
    source_columns = [field["name"] for field in details["columns"]]
    mapping = discover_columns(source_columns, aliases)
    geometry_name = next(
        (field["name"] for field in details["columns"] if field["name"].casefold() in {"geometry", "geom", "shape"}),
        None,
    )
    wanted = list(dict.fromkeys(mapping.values()))
    if geometry_name and not {"latitude", "longitude"}.issubset(mapping):
        wanted.append(geometry_name)
    if not wanted:
        raise ValueError(f"No configured columns were found. Source schema: {source_columns}")

    parquet = pq.ParquetFile(path)
    step = max(1, details["rows"] // int(config["data"]["sample_rows"]))
    batches = []
    rows_seen = 0
    sampled_rows = 0
    for batch in parquet.iter_batches(batch_size=65_536, columns=wanted):
        batch_frame = batch.to_pandas()
        offset = (-rows_seen) % step
        sampled = batch_frame.iloc[offset::step]
        batches.append(sampled)
        rows_seen += len(batch_frame)
        sampled_rows += len(sampled)
        if sampled_rows >= int(config["data"]["sample_rows"]):
            break
    raw = pd.concat(batches, ignore_index=True).head(int(config["data"]["sample_rows"]))
    if geometry_name and not {"latitude", "longitude"}.issubset(mapping):
        try:
            from shapely import from_wkb
        except ImportError as exc:
            raise RuntimeError("This GeoParquet source uses WKB geometry; install shapely to decode it.") from exc
        geometries = from_wkb(raw[geometry_name].to_numpy())
        raw["longitude"] = [geometry.x if geometry is not None else None for geometry in geometries]
        raw["latitude"] = [geometry.y if geometry is not None else None for geometry in geometries]
        mapping["latitude"] = "latitude"
        mapping["longitude"] = "longitude"
    canonical_aliases = {key: [value] for key, value in mapping.items()}
    canonical = canonicalize(raw, canonical_aliases)
    clean, audit, quality = clean_positions(canonical)
    featured = engineer_trajectory_features(clean)
    output_dir = project_path(config["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_dir / "ais_sample.parquet", index=False)
    audit.to_parquet(output_dir / "cleaning_audit.parquet", index=False)
    quality.update({"source_rows": details["rows"], "sampled_rows": len(raw), "schema_mapping": mapping})
    return featured, {"inspection": details, "quality": quality}


def train_baseline(config: dict) -> dict:
    from src.drift import detect_median_shift
    from src.features import chronological_split
    from src.models import AnomalyEnsemble

    source = project_path(config["data"]["processed_dir"]) / "ais_sample.parquet"
    if not source.exists():
        raise FileNotFoundError("Processed AIS sample missing; run `python -m src.pipeline process` first.")
    data = pd.read_parquet(source)
    candidates = ["sog", "derived_speed_knots", "distance_km", "elapsed_hours", "course_change_deg", "turn_rate_deg_per_hour"]
    features = [name for name in candidates if name in data and data[name].notna().any()]
    if len(features) < 2:
        raise ValueError(f"At least two usable movement features are needed; found {features}.")
    train, validation, test = chronological_split(data)
    medians = train[features].median()
    train_x = train[features].replace([float("inf"), -float("inf")], pd.NA).fillna(medians).to_numpy()
    validation_x = validation[features].replace([float("inf"), -float("inf")], pd.NA).fillna(medians).to_numpy()
    test_x = test[features].replace([float("inf"), -float("inf")], pd.NA).fillna(medians).to_numpy()
    model_config = config["models"]
    model = AnomalyEnsemble(
        contamination=model_config["contamination"],
        alert_quantile=model_config["alert_quantile"],
        seed=config["seed"],
    )
    training_started = time.perf_counter()
    model.fit(train_x)
    training_seconds = time.perf_counter() - training_started
    validation_scores = model.predict_scores(validation_x)
    inference_started = time.perf_counter()
    test_scores = model.predict_scores(test_x)
    inference_seconds = time.perf_counter() - inference_started
    scored = test[["vessel_id", "timestamp"]].copy()
    scored["anomaly_score"] = test_scores["score"]
    scored["anomaly_flag"] = test_scores["flag"]
    for name, values in test_scores["components"].items():
        scored[f"score_{name}"] = values
    report = {
        "model": "Robust-scaled Isolation Forest + novelty Local Outlier Factor",
        "features": features,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold": test_scores["threshold"],
        "test_alerts": int(test_scores["flag"].sum()),
        "test_alert_percent": float(test_scores["flag"].mean() * 100),
        "validation_alerts": int(validation_scores["flag"].sum()),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "seed": int(config["seed"]),
        "model_configuration": {
            "contamination": model_config["contamination"],
            "alert_quantile": model_config["alert_quantile"],
            "isolation_forest_estimators": 200,
            "lof_neighbors": min(max(2, int(np.sqrt(len(train_x)))), len(train_x) - 1),
            "features": features,
        },
        "drift_events_on_test_score": [event.__dict__ for event in detect_median_shift(test_scores["score"])],
        "label_status": "No ground-truth anomaly labels; supervised accuracy metrics are not reported.",
    }
    reports = project_path("artifacts/reports")
    reports.mkdir(parents=True, exist_ok=True)
    scored.to_csv(reports / "baseline_test_scores.csv", index=False)
    (reports / "baseline_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_dir = project_path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = project_path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "medians": medians.to_dict()}, model_dir / "baseline_ensemble.joblib")
    adaptive_model = AnomalyEnsemble(
        contamination=model_config["contamination"],
        alert_quantile=model_config["alert_quantile"],
        seed=config["seed"],
    )
    adaptive_started = time.perf_counter()
    adaptive_model.fit(np.concatenate([train_x, validation_x], axis=0))
    adaptive_training_seconds = time.perf_counter() - adaptive_started
    adaptive_inference_started = time.perf_counter()
    adaptive_test = adaptive_model.predict_scores(test_x)
    adaptive_inference_seconds = time.perf_counter() - adaptive_inference_started
    adaptive_result = {
        "status": "computed_shadow_update",
        "fixed_training_rows": len(train),
        "adaptive_training_rows": len(train) + len(validation),
        "validation_used_for_shadow_update": True,
        "fixed_test_alerts": int(test_scores["flag"].sum()),
        "fixed_test_alert_percent": float(test_scores["flag"].mean() * 100),
        "adaptive_test_alerts": int(adaptive_test["flag"].sum()),
        "adaptive_test_alert_percent": float(adaptive_test["flag"].mean() * 100),
        "fixed_training_seconds": training_seconds,
        "adaptive_training_seconds": adaptive_training_seconds,
        "fixed_inference_seconds": inference_seconds,
        "adaptive_inference_seconds": adaptive_inference_seconds,
        "ground_truth_status": "unavailable; alert rates are not performance metrics",
        "interpretation": "Unsupervised shadow refit comparison only; no superiority or operational adaptation claim.",
    }
    (metrics_dir / "fixed_vs_adaptive.json").write_text(json.dumps(adaptive_result, indent=2), encoding="utf-8")
    adaptive_path = model_dir / "adaptive_shadow_ensemble.joblib"
    joblib.dump({"model": adaptive_model, "features": features, "medians": medians.to_dict()}, adaptive_path)
    report["fixed_vs_adaptive"] = adaptive_result

    explanation_rows = []
    explanation_size = min(len(test_x), 2000)
    explanation_x = test_x[:explanation_size].copy()
    reference_scores = model.predict_scores(explanation_x)["score"]
    rng = np.random.default_rng(config["seed"])
    for index, feature in enumerate(features):
        permuted_x = explanation_x.copy()
        permuted_x[:, index] = rng.permutation(permuted_x[:, index])
        permuted_scores = model.predict_scores(permuted_x)["score"]
        explanation_rows.append({"feature": feature, "mean_absolute_score_change": float(np.mean(np.abs(permuted_scores - reference_scores))), "rows_permuted": explanation_size, "seed": int(config["seed"]), "interpretation": "Model-score sensitivity; not feature causality or validated importance."})
    explanation_frame = pd.DataFrame(explanation_rows).sort_values("mean_absolute_score_change", ascending=False)
    explanation_frame.to_csv(metrics_dir / "feature_sensitivity.csv", index=False)
    explanation_fig, explanation_ax = plt.subplots(figsize=(9, 5))
    explanation_frame.sort_values("mean_absolute_score_change").plot.barh(x="feature", y="mean_absolute_score_change", legend=False, ax=explanation_ax)
    explanation_ax.set(title="Permutation sensitivity of held-out anomaly scores", xlabel="Mean absolute score change", ylabel="Feature")
    save_figure(explanation_fig, "reports/figures/explainability/permutation_score_sensitivity.png")
    (metrics_dir / "baseline_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_scores["score"], bins=60)
    ax.axvline(test_scores["threshold"], linestyle="--", label="training-calibrated threshold")
    ax.set(title="Held-out anomaly score distribution", xlabel="Fused anomaly score", ylabel="Test observations")
    ax.legend()
    save_figure(fig, "reports/figures/models/anomaly_score_distribution.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    scored.assign(hour_utc=scored["timestamp"].dt.floor("h")).groupby("hour_utc")["anomaly_score"].median().plot(ax=ax)
    ax.set(title="Chronological test anomaly scores", xlabel="Time (UTC)", ylabel="Median fused anomaly score")
    save_figure(fig, "reports/figures/models/chronological_evaluation.png")
    return report

def run(command: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    path = project_path(config["data"]["raw_dir"]) / config["data"]["source_filename"]
    if command == "download":
        path = download_source(config)
    if command == "download" or command == "inspect":
        report = inspect_file(path)
        output = project_path(config["data"]["manifest_dir"])
        output.mkdir(parents=True, exist_ok=True)
        report["source_url"] = config["data"]["source_url"]
        report["accessed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["license"] = "CC0 1.0 Universal (source documentation)"
        (output / "source_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Inspected %s rows; schema written to data/manifests/source_manifest.json", report["rows"])
    if command == "process":
        frame, result = process_file(path, config)
        output = project_path("artifacts/reports")
        output.mkdir(parents=True, exist_ok=True)
        (output / "cleaning_report.json").write_text(json.dumps(result["quality"], indent=2), encoding="utf-8")
        LOGGER.info("Processed %s sampled rows; retained %s", result["quality"]["sampled_rows"], len(frame))
    if command == "train":
        LOGGER.info("Baseline results: %s", json.dumps(train_baseline(config)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire and process the configured official AIS source.")
    parser.add_argument("command", choices=("download", "inspect", "process", "train"))
    run(parser.parse_args().command)