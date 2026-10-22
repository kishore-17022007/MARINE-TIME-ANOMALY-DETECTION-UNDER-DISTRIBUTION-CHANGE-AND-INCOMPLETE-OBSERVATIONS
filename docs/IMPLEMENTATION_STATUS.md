# Implementation Status

Updated: 2026-10-17

## Completed in the Python-only implementation

- Verified an official, public Marine Cadastre NOAA AIS daily GeoParquet object and schema documentation. Configured one 2024-01-01 object, 227,181,435 bytes, below the 350 MiB cap; HTTP advertises an MD5 checksum. Source says daily GeoParquet 1.0.0 / WKB Point / WGS84 / UTC / minute-downsampled / CC0.
- Implemented config-driven source download, bounded size check, retry-compatible URL handling, byte-range resume, checksum verification, Parquet inspection, batch sampling, schema mapping, cleaning audit, and separate processed outputs.
- Implemented geodesic/gap-safe features, chronological split, robust-scaled Isolation Forest plus novelty LOF score fusion, a simple feature distribution-shift utility, descriptive summary and group test, and a labeled speed-injection utility.
- Added a modular Python package layout, numbered CLI stages, ordered pipeline runner, output directories, shared 300-DPI graph writer, and saved-report integration in Streamlit.
- Downloaded and verified the official 2024-01-01 GeoParquet object (227,181,435 bytes) against the source MD5; manifest records 7,293,408 source rows and the observed GeoParquet schema.
- Processed a deterministic 250,000-row sample: all rows had usable timestamps and WGS84 coordinates; generated the cleaning audit and movement features.
- Trained the unsupervised baseline with chronological partitions (175,000 train / 37,500 validation / 37,500 test); 430 test observations were flagged at the calibrated threshold. These are exploratory alerts only; there are no ground-truth labels, so predictive accuracy is not established.
- Generated 35 static PNG figures across EDA, descriptive, diagnostic, model, drift, and simulation categories. EDA used a deterministic sample; pairplots are capped at 1,000 rows. Every report image link resolves.
- Generated reusable CSV/JSON result tables, a persisted 581 MiB baseline model, and an HTML report with all 35 figure paths and actual cleaning, diagnostic, model, and simulation tables.
- Ran the controlled speed-change simulation with the persisted detector; for the selected 66-row track it produced 0 original alerts and 0 scenario alerts, with 0% injected-interval detection. This is explicitly a narrow synthetic test, not operational validation.
- Added a vessel-level Kruskal-Wallis diagnostic with group sizes, p-value resolution bound, epsilon-squared effect size, and seeded 300-resample confidence interval.
- Removed all notebook files. The repository contains no `.ipynb` files; all workflows use Python scripts.
- Updated Streamlit dashboard to display saved figure/table/report/model/drift/simulation outputs and clear not-yet-generated states.
- Updated the source alias for observed `base_date_time`; cast Arrow-backed course changes to NumPy-backed float for pandas 3 modulo compatibility, with regression coverage.
- `python -m pytest -q`: 7 passed.
- `python -m compileall -q src scripts app.py`: passed.
- Pipeline stages 01-11 completed successfully on the verified source with a 250,000-row sample and seed 42; custom relative and absolute output directories were smoke-tested.
- Plotly/Kaleido PNG export was not exercised for saved report figures; Kaleido is unavailable. Static report PNGs are generated using Matplotlib/Seaborn at 300 DPI. PDF export was not generated because no PDF library is configured.

## Blocked or unfinished

- Deep temporal/autoencoder models, validation-gated adaptive update/rollback, multi-period replay, route/AIS-corruption scenarios, and model explainability/ablation are not implemented.
- No ground-truth anomaly labels are available. ROC-AUC, precision, recall, F1, and precision-recall plots are correctly omitted.
- One day and one controlled scenario cannot establish anomaly accuracy, adaptation under concept drift, causal explanations, or generalization to other days or regions.

## Resume Steps

1. Acquire independent additional days/regions and validate sampling, feature behavior, and geographic coverage.
2. Define and adjudicate ground-truth or domain-reviewed labels before reporting supervised performance.
3. Implement validated adaptive promotion/rollback and multi-period drift replay before claiming drift resilience.
4. Add explainability and ablation with explicit limits on model-derived explanations.