# Project Structure

Use this order to follow the research workflow. Existing canonical paths are kept intact because the pipeline, saved experiment configs, and dashboard refer to them directly.

## 1. Data and Preprocessing

Source acquisition, validation, cleaning, exploration, and feature engineering happen first.

- Acquisition and preprocessing scripts: `scripts/01_download_data.py`, `scripts/02_clean_data.py`
- Exploration and feature stages: `scripts/03_run_eda.py` through `scripts/06_engineer_features.py`
- Shared source data: `data/raw/`, `data/interim/`, `data/manifests/`
- Processed baseline sample: `data/processed/`
- Preprocessing and analytics implementation: `src/pipeline.py`, `src/cleaning.py`, `src/schema.py`, `src/features/`, `src/analytics/`
- Per-experiment processed data and coverage: `experiments/1-day/data/`, `experiments/1-day/coverage.json`, `experiments/7-day/data/`, `experiments/7-day/coverage.json`
- Per-experiment preprocessing and EDA results: `experiments/<duration>/artifacts/metrics/cleaning_report.json`, `eda_summary.json`, `feature_summary.json`; supporting figures and tables are in `experiments/<duration>/reports/`

## 2. Model Code

- Model implementations: `src/models/`
- Training and scoring orchestration: `src/pipeline.py`, `src/research_pipeline.py`
- Model training stage: `scripts/07_train_models.py`
- Shared feature contract and training configuration: `config/config.yaml`

## 3. Models and Model Results

- Preserved experiment bundles: `experiments/1-day/artifacts/models/` and `experiments/7-day/artifacts/models/`
- Each experiment contains `baseline_ensemble.joblib` and `adaptive_shadow_ensemble.joblib`.
- Training/evaluation outputs: `experiments/<duration>/artifacts/metrics/baseline_evaluation.json`, `fixed_vs_adaptive.json`, and `experiments/<duration>/artifacts/reports/baseline_test_scores.csv`
- Global score sensitivity results: `experiments/<duration>/artifacts/metrics/feature_sensitivity.csv` and `experiments/<duration>/reports/figures/explainability/`
- No ground-truth labels exist for historical AIS, so its alert percentages are not accuracy metrics.

## 4. Overall and Comparative Results

- Verified experiment summaries: `experiments/1-day/experiment_summary.json` and `experiments/7-day/experiment_summary.json`
- Duration comparison: `experiments/master_comparison.csv`
- Controlled benchmark metrics: `experiments/<duration>/reports/benchmark/scenario_metrics_by_seed.csv` and `scenario_metrics_summary.csv`
- Unlabeled historical score distributions: `experiments/<duration>/reports/benchmark/real_historical_unlabeled/`
- Controlled event labels and modified test rows: `experiments/<duration>/reports/benchmark/injected_test_cases/` and `synthetic_trajectories/`
- Drift replay: `experiments/<duration>/reports/benchmark/fixed_vs_adaptive_drift.csv`
- Charts and analytical tables: `experiments/<duration>/reports/benchmark/figures/`, `experiments/<duration>/reports/figures/`, and `experiments/<duration>/reports/tables/`
- Root-level overall comparison tables/figures: `reports/generated/` (CSV/JSON only), `reports/tables/`, and `reports/figures/`
- Controlled metrics describe injected scenarios, not verified real-world incidents. Counterfactual controls are not certified normal AIS.

## 5. Dashboard and Application

Keep the user-facing application last in the reading order.

- Streamlit entry point: `app.py`
- Research analytics, historical replay, XAI evidence, and future-mode UI: `src/dashboard_app.py`
- Scenario transformations and saved-model inference: `src/simulation/dashboard.py`
- Synthetic evaluation implementation and runner: `src/evaluation/synthetic_benchmark.py`, `scripts/run_synthetic_benchmark.py`
- Launch: `python -m streamlit run app.py`
- Future operational mode is a demonstration; no live AIS feed is connected.

## Tests and Dependencies

- Tests: `tests/`
- Dependencies: `requirements.txt`
- Run tests with `python -m pytest -q`.