from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import project_path


def _read_json(path: str) -> dict:
    target = project_path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


def _html_table(headers: list[str], rows: list[list[object]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body_html = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _figure_cards() -> tuple[list[str], list[dict]]:
    root = project_path("reports/figures")
    image_paths = sorted(path for path in root.rglob("*.png") if path.is_file())
    cards = []
    for path in image_paths:
        relative = Path(os.path.relpath(path, project_path("reports/generated")))
        caption = path.stem.replace("_", " ").replace("-", " ").capitalize()
        cards.append({"path": str(relative), "caption": caption, "category": path.parent.name})
    return [f'<figure><a href="{html.escape(card["path"])}"><img src="{html.escape(card["path"])}" alt="{html.escape(card["caption"])}"></a><figcaption><strong>{html.escape(card["caption"])}</strong><br>{html.escape(card["category"])} figure</figcaption></figure>' for card in cards], cards


def generate_report() -> dict:
    manifest = _read_json("data/manifests/source_manifest.json")
    quality = _read_json("artifacts/metrics/cleaning_report.json") or _read_json("artifacts/reports/cleaning_report.json")
    eda = _read_json("artifacts/metrics/eda_summary.json")
    descriptive = _read_json("artifacts/metrics/descriptive_summary.json")
    diagnostic = _read_json("artifacts/metrics/diagnostics.json")
    baseline = _read_json("artifacts/metrics/baseline_evaluation.json") or _read_json("artifacts/reports/baseline_evaluation.json")
    drift = _read_json("artifacts/metrics/drift_report.json")
    simulation = _read_json("artifacts/metrics/simulation_results.json")
    evaluation = _read_json("artifacts/metrics/evaluation_summary.json")
    experiment_name = manifest.get("experiment")
    duration_days = len(manifest.get("daily_coverage", {})) if experiment_name else 1
    requested_start = manifest.get("requested_start", "2024-01-01")
    requested_end = manifest.get("requested_end", requested_start)
    interval = diagnostic.get("confidence_interval") or {}
    interval_text = (
        f"{interval['lower']:.4f} to {interval['upper']:.4f} "
        f"(percentile bootstrap; n={interval.get('resamples', 0)}, seed={interval.get('seed', 'n/a')})"
        if {"lower", "upper"}.issubset(interval)
        else "not computed"
    )
    cleaning_table = _html_table(
        ["Input rows", "Retained rows", "Excluded unusable", "Duplicates flagged"],
        [[quality.get("input_rows", "incomplete"), quality.get("retained_rows", "incomplete"), quality.get("excluded_unusable_rows", "incomplete"), quality.get("duplicate_rows_flagged", "incomplete")]],
    )
    diagnostic_table = _html_table(
        ["Test", "Status", "Eligible groups", "Statistic", "p-value", "Epsilon-squared", "95% CI"],
        [[diagnostic.get("test", "not run"), diagnostic.get("status", "incomplete"), diagnostic.get("groups", "n/a"), diagnostic.get("statistic", "n/a"), diagnostic.get("p_value_numeric_limit", diagnostic.get("p_value", "n/a")), diagnostic.get("epsilon_squared", "n/a"), interval_text]],
    )
    model_table = _html_table(
        ["Model", "Train", "Validation", "Test", "Threshold", "Test alerts"],
        [[baseline.get("model", "incomplete"), baseline.get("train_rows", "n/a"), baseline.get("validation_rows", "n/a"), baseline.get("test_rows", "n/a"), baseline.get("threshold", "n/a"), baseline.get("test_alerts", "n/a")]],
    )
    simulation_table = _html_table(
        ["Scenario", "Status", "Rows", "Original alerts", "Scenario alerts", "Injected interval detection"],
        [[simulation.get("scenario", "not run"), simulation.get("status", "incomplete"), simulation.get("rows", "n/a"), simulation.get("original_alerts", "n/a"), simulation.get("scenario_alerts", "n/a"), simulation.get("injected_interval_detection_fraction", "n/a")]],
    )
    figure_html, figure_index = _figure_cards()
    generated = project_path("reports/generated")
    generated.mkdir(parents=True, exist_ok=True)
    index_rows = "\n".join(f'<li><a href="{html.escape(item["path"])}">{html.escape(item["caption"])}</a> ({html.escape(item["category"])})</li>' for item in figure_index)
    index = f"<!doctype html><html><head><meta charset='utf-8'><title>Figure Index</title></head><body><h1>Figure Index</h1><ul>{index_rows}</ul></body></html>"
    (generated / "figure_index.html").write_text(index, encoding="utf-8")
    figure_section = "\n".join(figure_html) if figure_html else "<p>Incomplete: no report figure PNGs have been generated.</p>"
    benchmark_index = project_path("reports/generated/controlled_synthetic_benchmark_index.html")
    benchmark_note = (
        '<h2>Controlled Synthetic Benchmark</h2><p>See the <a href="controlled_synthetic_benchmark_index.html">controlled synthetic benchmark index</a> for generated 1-day and 7-day results. These are synthetic-label evaluations, not verified real-world incident metrics.</p>'
        if benchmark_index.exists() else ""
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>An Adaptive Multimodel Framework for Reliable Maritime Anomaly Detection Under Concept Drift</title>
<style>body{{font:16px/1.6 Georgia,serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202a32}}h1,h2{{font-family:system-ui,sans-serif;line-height:1.2}}h1{{font-size:2.2rem}}h2{{margin-top:2.5rem;border-bottom:1px solid #ccd4da;padding-bottom:.35rem}}.notice{{background:#f0f3f5;padding:1rem;border-left:4px solid #557482}}.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.25rem}}figure{{margin:0;padding:0 0 1rem;border-bottom:1px solid #ccd4da}}figure img{{width:100%;height:auto}}figcaption{{font-size:.9rem}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4da;padding:.45rem;text-align:left}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>An Adaptive Multimodel Framework for Reliable Maritime Anomaly Detection Under Concept Drift</h1>
<p><strong>Generated:</strong> {datetime.now(timezone.utc).isoformat()}</p>
<h2>Abstract</h2><p>This report summarizes the {html.escape(str(experiment_name or 'original one-day'))} experiment for {duration_days} requested day(s), from {html.escape(requested_start)} through {html.escape(requested_end)}. It documents source provenance, date-specific processing, analysis, chronological unsupervised baseline scoring, and controlled simulation only where outputs exist. The current experiments do not establish real-world anomaly accuracy or adaptive concept-drift performance.</p>
<h2>Research Objectives and Questions</h2><ol><li>Can a reproducible pipeline acquire and quality-audit official AIS data?</li><li>What movement and observation patterns appear in the configured one-day sample?</li><li>Can an unsupervised baseline rank observations for review under chronological evaluation?</li><li>How does that baseline respond to a clearly labeled controlled speed-change scenario?</li></ol>
<h2>Data Sources and Provenance</h2><p>Source URL template: <code>{html.escape(str(manifest.get('source_url_template', manifest.get('source_url', 'Incomplete: source manifest unavailable'))))}</code>. Requested days: {duration_days}. Actual coverage is verified from each Parquet timestamp column and recorded day-by-day in the source manifest. No observations are generated to fill absent dates.</p><p>Manifest license field: {html.escape(str(manifest.get('license', 'CC0 1.0 Universal (source documentation)')))}. Each daily object is accepted only after its advertised Content-MD5 has been verified.</p>
<h2>Dataset Description</h2><p>Processed observations: {descriptive.get('observations', quality.get('sampled_rows', 'incomplete'))}; vessels: {descriptive.get('vessels', 'incomplete')}; requested time range (UTC): {html.escape(requested_start)} through {html.escape(requested_end)}; observed time range: {html.escape(str(descriptive.get('time_start_utc', 'incomplete')))} to {html.escape(str(descriptive.get('time_end_utc', 'incomplete')))}. The deterministic sample seed and date-stratified sampling policy are recorded in the manifest.</p>
<h2>Cleaning and Preprocessing</h2><p>Coordinates are decoded from WKB and checked for WGS84 bounds; gaps over six hours do not yield movement distance or speed features.</p>{cleaning_table}
<h2>EDA Findings</h2><p>EDA status: {html.escape(str(eda.get('status', 'incomplete: EDA stage not run')))}. Generated figures: {len(eda.get('figures', []))}. Pairwise and geographic displays use documented deterministic samples. See the figure index for actual images.</p>
<h2>Descriptive Analytics</h2><p>Descriptive status: {'completed' if descriptive else 'incomplete: stage not run'}. Reusable CSV tables are in <a href="../tables/">reports/tables</a>. The 1-day experiment is a one-date baseline; longer duration experiments have separate outputs and coverage manifests.</p>
<h2>Diagnostic Analytics and Statistical Tests</h2><p>{html.escape(str(diagnostic.get('interpretation', diagnostic.get('reason', 'No diagnostic result available.'))).rstrip('.'))}. Grouped tests use vessel-level medians; association is not causation. A very small p-value is bounded by floating-point resolution as shown.</p>{diagnostic_table}
<h2>Feature Engineering</h2><p>Features include geodesic segment distance, elapsed time, derived speed, speed/course changes, turn rate, and UTC calendar fields where the source supports them. Initial observations and invalid temporal gaps naturally lack segment features.</p>
<h2>Model Methodology</h2><p>Model: {html.escape(str(baseline.get('model', 'incomplete: model stage not run')))}. Robust scaling and score calibration are fitted on the chronological training partition. Training/inference timing, seed, and configuration are recorded. A deep autoencoder and temporal model are not implemented and are not claimed as experiment results.</p>
<h2>Evaluation Protocol and Results</h2>{model_table}<p>Ground-truth anomaly labels are unavailable. ROC-AUC, precision, recall, F1, and precision-recall curves are therefore not computed. Alerts are relative unusualness indicators only.</p>
<h2>Concept-Drift Experiments</h2><p>{html.escape(str(drift.get('interpretation', 'Incomplete: drift monitoring stage not run.')))} Events in the executed score screen: {len(drift.get('events', []))}. The persisted fixed-versus-adaptive comparison is a shadow refit; without labels, differences in alert counts are not evidence that either model is better.</p>
<h2>Simulation Experiments</h2><p>{html.escape(str(simulation.get('warning', simulation.get('reason', 'No simulation output is available.'))))}</p>{simulation_table}
<h2>Model Explanations, Limitations and Ethics</h2><p>No SHAP package or labeled outcome source is configured. Model-derived score components are not real-world causal explanations. AIS gaps may reflect reception and reporting conditions. Scores should not be used to infer intent or make operational decisions without independent evidence, domain review, and validated labels.</p>
<h2>Conclusions</h2><p>The implemented slice demonstrates checksum-verified NOAA data acquisition, a reproducible one-day sample, static report figures, descriptive/diagnostic outputs, and an unsupervised chronological baseline. Reliable anomaly detection under concept drift remains an open research objective: multi-day representativeness, labeled or independently adjudicated evaluation, drift replay, adaptive rollback tests, temporal models, and explainability are incomplete.</p>
<h2>Figure Index</h2><p><a href="figure_index.html">Open standalone figure index</a>.</p><div class="figures">{figure_section}</div>
{benchmark_note}
<h2>References</h2><ul><li><a href="https://github.com/ocm-marinecadastre/ais-vessel-traffic">NOAA Marine Cadastre AIS data and access instructions</a></li><li><a href="https://github.com/ocm-marinecadastre/ais-vessel-traffic/blob/main/data/ais-broadcast-points-2024-readme.md">AIS broadcast product metadata</a></li></ul>
</body></html>"""
    report_path = generated / "maritime_anomaly_research_report.html"
    report_path.write_text(report, encoding="utf-8")
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "html_report": str(report_path.relative_to(project_path("."))), "figure_index": str((generated / "figure_index.html").relative_to(project_path("."))), "figure_count": len(figure_index), "pdf": {"status": "not generated", "reason": "No PDF library configured; HTML report is the supported output."}, "report_scope": "real acquired one-day NOAA sample; unexecuted experiments marked incomplete"}
    (generated / "execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary