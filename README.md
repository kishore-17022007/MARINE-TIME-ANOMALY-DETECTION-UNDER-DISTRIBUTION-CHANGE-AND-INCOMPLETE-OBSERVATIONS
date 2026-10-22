# Maritime Anomaly Research

Python-only research project for official AIS source acquisition, preprocessing, analytics, unsupervised baseline scoring, distribution-shift screening, controlled simulation, and an interactive Streamlit dashboard. Anomaly scores are review signals only and do not establish intent, unsafe conduct, or real-world anomaly accuracy.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the recommended reading order: data and preprocessing, model code, model results, overall results, then the dashboard.

## Data Source

The configured source is NOAA Office for Coastal Management / Marine Cadastre daily broadcast GeoParquet for 2024-01-01:

`https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-2024-01-01.parquet`

The checked object is 227,181,435 bytes and its advertised Content-MD5 is checked before it is accepted. The manifest records the observed schema, GeoParquet metadata, source URL, byte size, and row count. The source is documented as WKB Point geometry, WGS84, UTC, minute-downsampled, and CC0 1.0. Source references are in the report and initial source notes below.

## Setup

Use Python 3.11 or later, install the requirements, and select that interpreter in VS Code:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Launch the maritime digital-twin simulator for interactive replay, AIS transmission views, synthetic vessel animation, and the scenario-driven model comparison:

```bash
streamlit run simulator_app.py
```

The simulator is intentionally split between:

- synthetic educational AIS data for vessel animation and scenario demonstration; and
- historical NOAA research data for the project’s verified data-processing and anomaly-detection flow.

Launch the supporting research dashboard:

```bash
streamlit run app.py
```

Run every stage in dependency order, stopping on the first failure:

```bash
python scripts/run_pipeline.py --from-stage 01_download_data.py
```

The official object is about 227 MB; it can take several minutes on a slow connection. To rerun analysis without re-downloading, the verified file and processed output already exist locally. The numbered scripts are also independently executable:

```bash
python scripts/02_clean_data.py --sample-size 250000 --seed 42
python scripts/03_run_eda.py --sample-size 3000 --seed 42
python scripts/04_run_descriptive_analytics.py
python scripts/05_run_diagnostic_analytics.py
python scripts/06_engineer_features.py
python scripts/07_train_models.py
python scripts/08_detect_drift.py
python scripts/09_run_simulation.py
python scripts/10_evaluate_models.py
python scripts/11_generate_report.py
streamlit run app.py
```

## Duration Comparisons

For the final verified analysis, only the 1-day and 7-day runs remain in scope. These are the only completed experiment folders retained in `experiments/`.

```bash
python scripts/run_duration_experiments.py --duration 1-day
python scripts/run_duration_experiments.py --duration 7-day
```

Each experiment starts on 2024-01-01 and acquires/reuses the official daily NOAA Parquet object for the requested dates. The downloader verifies content length and advertised MD5; each Parquet timestamp column is scanned to verify actual UTC date coverage. Missing or mismatched dates fail that experiment and are recorded in `experiments/<duration>/coverage.json`. Original raw objects remain in `data/raw/`; date-specific samples, model artifacts, metrics, figures, and tables are isolated under `experiments/<duration>/`.

The same 200,000-row maximum is stratified across requested dates with deterministic seed 42 by default. Override consistently with `--sample-size` and `--seed`. Current models are the existing Isolation Forest + novelty LOF ensemble and an unsupervised shadow refit; there are no adjudicated anomaly labels, so alert counts/rates are not accuracy metrics. Existing autoencoder, temporal, adaptive rollback, and SHAP support are not implemented. The explanation output is permutation sensitivity of model scores, not causality or validated importance.

The final comparison table is `experiments/master_comparison.csv` for the verified 1-day and 7-day runs. Select the completed duration in the Streamlit Dataset control to browse its saved results.

### Controlled Synthetic Evaluation

Run the reproducible benchmark against both preserved experiment samples and existing fixed/shadow model bundles:

```bash
python scripts/run_synthetic_benchmark.py
```

The benchmark uses seeds 42, 43, and 44 by default. It writes per-experiment CSV, JSON, Parquet, and PNG outputs under `experiments/<duration>/reports/benchmark/`. Metrics compare each injected event case with its unmodified counterfactual copy; those controls are not independently verified normal AIS. Separate untouched held-out historical score summaries are unlabeled analyses. The benchmark does not create HTML reports, overwrite original data, or retrain saved models. Metrics describe controlled synthetic cases only, not verified real-world maritime incidents.

Common options: `--input` for a CSV or Parquet path, `--start-date`, `--end-date`, `--bbox west,south,east,north`, `--sample-size`, `--seed`, `--config`, `--output-dir`, and `--skip-existing`. The runner prints each executed stage and stops safely if any subprocess fails. Existing outputs are reused only when a valid expected output file exists; skipped stages are not treated as newly executed results.

## Outputs

- `data/manifests/source_manifest.json`: acquired source metadata and observed schema.
- `data/processed/ais_sample.parquet` and `cleaning_audit.parquet`: deterministic sampled data and row-level audit.
- `artifacts/models/baseline_ensemble.joblib`: persisted robust-scaled Isolation Forest plus novelty LOF ensemble.
- `artifacts/metrics/`: cleaning, EDA, descriptive, diagnostic, feature, drift, simulation, and evaluation JSON results.
- `reports/figures/`: static analytical report figures saved as 300-DPI PNGs by Python scripts.
- `reports/tables/`: reusable CSV and JSON result tables.
- `reports/generated/`: overall CSV and JSON comparison outputs. HTML reports have been removed from this workspace.
- `artifacts/logs/pipeline.log`: stage execution log.

Matplotlib/Seaborn figures are saved as 300-DPI PNGs. Plotly utility output may also write HTML in workflows that export Plotly figures; the project-wide HTML cleanup removed current `.html` output files. PDF is not generated because no PDF library is configured.

## Findings and Limits

The current run processes 250,000 rows sampled deterministically from one official day. Cleaning retains rows with usable time and coordinates and preserves audit flags. The unsupervised baseline uses chronological train/validation/test partitions. There are no ground-truth anomaly labels, so precision, recall, F1, ROC-AUC, and PR-AUC are not computed. A vessel-level Kruskal-Wallis diagnostic reports group sizes, statistic, p-value bound, effect size, and bootstrap interval; it is an association test, not a causal finding.

The controlled speed-change simulation operates on a copy and uses the persisted detector. This run detected 0% of the injected interval for the selected vessel; it is a limited synthetic scenario, not a validation of real-world detector performance. The score monitor is a median-shift screen, not proof of concept drift. Adaptation/rollback, multi-day replay, deep autoencoder and temporal models, SHAP, multi-scenario evaluation, and adjudicated labels remain incomplete. The one-day sample cannot support seasonal or broad route-generalization claims.

## Python-Only Policy

All workflows are implemented as `.py` modules and scripts. No Jupyter notebooks are used or retained. Tests run with:

```bash
python -m pytest -q
```

## Source References

- [Marine Cadastre AIS data and access instructions](https://github.com/ocm-marinecadastre/ais-vessel-traffic)
- [Official daily broadcast filename list](https://github.com/ocm-marinecadastre/ais-vessel-traffic/blob/main/data/daily-2024-parquet-filenames.md)
- [Daily broadcast product metadata](https://github.com/ocm-marinecadastre/ais-vessel-traffic/blob/main/data/ais-broadcast-points-2024-readme.md)
- [NOAA InPort AIS Vessel Tracks metadata](https://www.fisheries.noaa.gov/inport/item/80415)
- [NOAA Digital Coast vessel traffic overview](https://www.coast.noaa.gov/digitalcoast/data/vesseltraffic.html)
