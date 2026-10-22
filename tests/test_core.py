import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.cleaning import clean_positions
from src.drift import detect_median_shift
from src.features import chronological_split, haversine_km
from src.models import AnomalyEnsemble
from src.schema import discover_columns, validate_schema
from src.simulation import inject_speed_scenario


def test_python_only_package_and_graphing_utilities():
    from src.analytics.feature_selection import correlation_screen
    from src.features.trajectory_features import haversine_km as package_haversine
    from src.utils.graphing import save_figure
    from src.utils.paths import ROOT

    assert package_haversine(0, 0, 0, 1) > 100
    result = correlation_screen(pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]}), ["a", "b"])
    assert result["status"] == "computed"
    assert ROOT.exists()
    assert not list(ROOT.rglob("*.ipynb"))
    assert callable(save_figure)


def test_schema_mapping_and_required_fields():
    mapped = discover_columns(["MMSI", "BaseDateTime", "LAT", "LON"], {
        "vessel_id": ["mmsi"], "timestamp": ["basedatetime"], "latitude": ["lat"], "longitude": ["lon"]
    })
    validate_schema(mapped)
    assert mapped["vessel_id"] == "MMSI"
    with pytest.raises(ValueError):
        validate_schema({})


def test_cleaning_flags_without_imputing_or_mutating():
    sample = pd.DataFrame({
        "vessel_id": ["v1", "v1", "v1"],
        "timestamp": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:00Z", None]),
        "latitude": [10.0, 10.0, 95.0], "longitude": [20.0, 20.0, 20.0], "sog": [4.0, 4.0, 90.0]
    })
    clean, audit, report = clean_positions(sample)
    assert len(clean) == 1
    assert report["duplicate_rows_flagged"] == 1
    assert report["excluded_unusable_rows"] == 1
    assert audit["audit_suspicious_speed"].sum() == 1
    assert sample.loc[2, "latitude"] == 95.0


def test_geodesic_and_gap_safe_features():
    assert 110 < haversine_km(0, 0, 0, 1) < 112
    track = pd.DataFrame({
        "vessel_id": ["v1", "v1", "v2"],
        "timestamp": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T02:00Z"]),
        "latitude": [0.0, 0.0, 0.0], "longitude": [0.0, 1.0, 5.0]
    })
    train, validation, test = chronological_split(track)
    assert len(train) + len(validation) + len(test) == len(track)
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()


def test_course_change_supports_arrow_backed_numeric_columns():
    from src.features import engineer_trajectory_features

    track = pd.DataFrame({
        "vessel_id": ["v1", "v1", "v1"],
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "latitude": [0.0, 0.0, 0.0],
        "longitude": [0.0, 0.1, 0.2],
        "cog": pd.Series([359.0, 1.0, 5.0], dtype="float64[pyarrow]"),
    })
    featured = engineer_trajectory_features(track)
    assert featured["course_change_deg"].iloc[1] == 2.0
    assert featured["course_change_deg"].iloc[2] == 4.0


def test_ensemble_scores_new_points_and_calibrates_threshold():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(200, 3))
    model = AnomalyEnsemble(seed=42).fit(values)
    result = model.predict_scores([[0.0, 0.0, 0.0], [12.0, 12.0, 12.0]])
    assert result["score"].shape == (2,)
    assert result["flag"].dtype == bool
    assert result["threshold"] == model.threshold


def test_open_url_retries_after_transient_errors(monkeypatch):
    import urllib.error
    import src.pipeline as pipeline

    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, context=None, timeout=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.URLError("temporary timeout")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    response = pipeline.open_url("https://example.com", retries=3, backoff_seconds=0)
    assert response is not None
    assert calls["count"] == 3


def test_drift_and_simulation_are_explicit():
    assert detect_median_shift(np.r_[np.zeros(100), np.full(50, 10.0)])
    track = pd.DataFrame({"sog": [5.0] * 20})
    scenario = inject_speed_scenario(track, severity=2.0)
    assert (track.sog == 5.0).all()
    assert scenario.is_simulated.all()
    assert scenario.injected_scenario.iloc[0] == "controlled_speed_change"


def test_speed_injection_changes_engineered_model_input_and_reaches_inference():
    from src.evaluation.synthetic_benchmark import _model_input, _score_model
    from src.features import engineer_trajectory_features

    track = pd.DataFrame({
        "vessel_id": ["v1"] * 8,
        "timestamp": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
        "latitude": [0.0] * 8,
        "longitude": np.linspace(0, 0.07, 8),
        "sog": np.full(8, 5.0),
        "cog": np.full(8, 90.0),
    }, index=np.arange(20, 28))
    modified = inject_speed_scenario(track, severity=2.0, start_fraction=0.5)
    original_features = engineer_trajectory_features(track)
    modified_features = engineer_trajectory_features(modified)
    bundle = {"features": ["sog", "derived_speed_knots", "speed_change"],
              "medians": {"sog": 5.0, "derived_speed_knots": 5.0, "speed_change": 0.0}}

    class SpyModel:
        received = None

        def predict_scores(self, values):
            self.received = values.copy()
            return {"score": values.sum(axis=1), "flag": values.sum(axis=1) > 0, "threshold": 0.0}

    model = SpyModel()
    bundle["model"] = model
    before = _model_input(original_features, bundle)
    after = _model_input(modified_features, bundle)
    assert not np.array_equal(before, after)
    _score_model(modified_features, bundle)
    np.testing.assert_array_equal(model.received, after)
    assert (modified_features.iloc[int(modified.injection_start_index.iloc[0]):]["sog"] == 10.0).all()


def test_synthetic_scenarios_label_only_injected_rows_and_use_valid_geography():
    from src.evaluation.synthetic_benchmark import inject_scenario

    count = 80
    data = pd.DataFrame({
        "vessel_id": ["v1"] * count,
        "timestamp": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
        "latitude": np.zeros(count), "longitude": np.linspace(0, 0.79, count),
        "sog": np.full(count, 8.0), "cog": np.full(count, 90.0), "heading": np.full(count, 90.0),
    })
    heldout = data.index[40:]
    changed, labels, metadata = inject_scenario(data, heldout, "position_jump", 10.0, 42)
    assert labels.index.equals(heldout)
    assert labels.sum() > 0
    assert metadata["scenario"] == "position_jump"
    assert not changed.loc[data.index.difference(labels.index), "latitude"].ne(data.loc[data.index.difference(labels.index), "latitude"]).any()
    event_rows = labels[labels].index
    distance = haversine_km(data.loc[event_rows, "latitude"], data.loc[event_rows, "longitude"], changed.loc[event_rows, "latitude"], changed.loc[event_rows, "longitude"])
    assert np.isfinite(distance).all()
    assert ((distance > 0) & (distance <= 10.01)).all()
    assert data.latitude.eq(0).all()


def test_dashboard_scenarios_are_reversible_and_inference_scores_the_modified_track():
    from src.models import AnomalyEnsemble
    from src.simulation.dashboard import apply_scenarios, score_simulated_track

    rng = np.random.default_rng(17)
    observations = 80
    baseline = pd.DataFrame({
        "vessel_id": ["vessel-a"] * observations,
        "timestamp": pd.date_range("2024-05-01", periods=observations, freq="min", tz="UTC"),
        "latitude": np.linspace(36.0, 36.1, observations),
        "longitude": np.linspace(-122.0, -121.9, observations),
        "sog": np.full(observations, 8.0),
        "cog": np.full(observations, 45.0),
    })
    normal, no_events = apply_scenarios(baseline, [], seed=42)
    changed, event_log = apply_scenarios(baseline, ["Unusual speed increase"], severity=2, start_fraction=.5, duration_fraction=.2, seed=42)
    assert no_events.empty
    assert not normal.is_simulated.any()
    assert baseline.sog.eq(8.0).all()
    assert changed.sog.max() == 16.0
    assert changed.is_simulated.sum() == 16
    assert not np.array_equal(normal.sog.to_numpy(), changed.sog.to_numpy())
    assert set(event_log.scenario) == {"Unusual speed increase"}

    training = rng.normal(size=(200, 3))
    model = AnomalyEnsemble(seed=42).fit(training)
    bundle = {"model": model, "features": ["sog", "derived_speed_knots", "speed_change"], "medians": {"sog": 8.0, "derived_speed_knots": 0.0, "speed_change": 0.0}}
    scored, metadata = score_simulated_track(changed, bundle)
    assert len(scored) == observations
    assert scored.anomaly_score.notna().all()
    assert scored.threshold.nunique() == 1
    assert metadata["features"] == bundle["features"]
    assert scored.loc[changed.scenario_state != "normal movement", "anomaly_score"].notna().all()


def test_dashboard_map_handles_arrow_flags_and_separates_simulated_points():
    from src.dashboard_app import _make_track_figure

    track = pd.DataFrame({
        "vessel_id": ["v1", "v1", "v1"],
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "latitude": [36.0, 36.1, 36.2],
        "longitude": [-122.0, -121.9, -121.8],
        "sog": pd.Series([5.0, 7.0, 6.0], dtype="float32[pyarrow]"),
        "fixed_score": [0.1, 1.2, 0.2],
        "fixed_flag": pd.Series([False, True, False], dtype="bool[pyarrow]"),
        "is_simulated": [False, True, False],
    })
    figure = _make_track_figure(track, "fixed", "test")
    trace_names = {trace.name for trace in figure.data}
    assert "Track start" in trace_names
    assert "Track end / direction" in trace_names
    assert "Simulated + flagged" in trace_names
    assert "Historical, not flagged" in trace_names


def test_synthetic_ais_generator_creates_valid_maritime_tracks():
    from src.simulation.maritime_twin import generate_synthetic_ais

    data = generate_synthetic_ais(vessel_count=8, seed=7, step_minutes=10, duration_minutes=180)
    assert len(data) > 0
    assert set(["vessel_id", "vessel_type", "timestamp", "latitude", "longitude", "sog", "cog", "heading"]).issubset(data.columns)
    assert data["latitude"].between(-90, 90).all()
    assert data["longitude"].between(-180, 180).all()
    assert data["timestamp"].is_monotonic_increasing
    assert data["vessel_id"].nunique() == 8
    assert data["sog"].between(0, 30).all()