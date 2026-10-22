"""Streamlit research analytics, historical replay, and simulator interface."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import descriptive_summary, vessel_speed_group_test
from src.evaluation.synthetic_benchmark import _model_input, _score_model
from src.features import chronological_split, engineer_trajectory_features
from src.simulation.dashboard import SCENARIOS, apply_scenarios, score_simulated_track


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {"Fixed ensemble": "baseline_ensemble.joblib", "Shadow adaptive": "adaptive_shadow_ensemble.joblib"}
MODEL_KEYS = {"Fixed ensemble": "fixed", "Shadow adaptive": "shadow_adaptive"}


@st.cache_data(show_spinner="Loading verified experiment data")
def load_data(experiment: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "experiments" / experiment / "data" / "processed" / "ais_sample.parquet")


@st.cache_data
def load_json(path: str) -> dict:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


@st.cache_resource
def load_bundle(experiment: str, model_name: str) -> dict | None:
    import joblib

    path = ROOT / "experiments" / experiment / "artifacts" / "models" / MODEL_FILES[model_name]
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner="Scoring chronological held-out AIS observations")
def load_test_scores(experiment: str) -> pd.DataFrame:
    data = load_data(experiment).sort_values("timestamp", kind="stable")
    _, _, test = chronological_split(data)
    featured = engineer_trajectory_features(data).loc[test.index].sort_values("timestamp", kind="stable")
    result = featured[["vessel_id", "timestamp"]].copy()
    result["vessel_id"] = result["vessel_id"].astype(str)
    for model_name, output_key in (("Fixed ensemble", "fixed"), ("Shadow adaptive", "shadow_adaptive")):
        bundle = load_bundle(experiment, model_name)
        if bundle is None:
            continue
        scores = _score_model(featured, bundle)
        result[f"{output_key}_score"] = scores["score"]
        result[f"{output_key}_flag"] = scores["flag"]
    return result


def _read_artifact(experiment: str, relative: str) -> dict:
    return load_json(str(ROOT / "experiments" / experiment / relative))


def _fmt_count(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _coverage(experiment: str) -> tuple[dict, list[str]]:
    manifest = _read_artifact(experiment, "data/manifests/source_manifest.json")
    expected = pd.date_range(manifest.get("requested_start", "2024-01-01"), manifest.get("requested_end", "2024-01-01"), freq="D").strftime("%Y-%m-%d").tolist()
    verified = set(manifest.get("verified_dates", []))
    return manifest, [day for day in expected if day not in verified]


def _decorate() -> None:
    st.markdown("""<style>
    .stApp { background: #081923; color: #e7f0f2; }
    [data-testid="stSidebar"] { background: #102833; border-right: 1px solid #24414b; }
    h1, h2, h3 { color: #e9f1ed; }
    [data-testid="stMetric"] { background: #112d38; padding: 14px 16px; border: 1px solid #24434d; border-radius: 5px; }
    [data-testid="stMetricValue"] { color: #72d0c2; }
    .result-label { color: #f2c66d; font-size: .82rem; letter-spacing: .04em; text-transform: uppercase; }
    </style>""", unsafe_allow_html=True)


def _show_overview(experiments: list[str], all_frames: dict[str, pd.DataFrame]) -> None:
    st.subheader("Verified research runs")
    overview_rows = []
    for experiment in experiments:
        frame = all_frames[experiment]
        manifest, missing_dates = _coverage(experiment)
        metrics = _read_artifact(experiment, "artifacts/metrics/baseline_evaluation.json")
        quality = _read_artifact(experiment, "artifacts/metrics/cleaning_report.json")
        summary = _read_artifact(experiment, "experiment_summary.json")
        fixed_adaptive = _read_artifact(experiment, "artifacts/metrics/fixed_vs_adaptive.json")
        overview_rows.append({
            "experiment": experiment,
            "status": summary.get("status", "summary missing"),
            "verified dates": len(manifest.get("verified_dates", [])),
            "missing dates": ", ".join(missing_dates) or "none",
            "observations": len(frame),
            "vessels": frame["vessel_id"].nunique(),
            "test alerts (fixed)": metrics.get("test_alerts"),
            "test alert rate (fixed, %)": metrics.get("test_alert_percent"),
            "test alerts (shadow)": fixed_adaptive.get("adaptive_test_alerts"),
            "test alert rate (shadow, %)": fixed_adaptive.get("adaptive_test_alert_percent"),
            "training seconds": metrics.get("training_seconds"),
            "inference seconds": metrics.get("inference_seconds"),
            "retained rows": quality.get("retained_rows"),
        })
    st.dataframe(pd.DataFrame(overview_rows), width="stretch", hide_index=True)
    st.caption("Historical AIS evaluation is unlabeled. Alert percentages describe model output, not real-world accuracy.")

    compare = []
    for experiment in experiments:
        metrics = _read_artifact(experiment, "artifacts/metrics/baseline_evaluation.json")
        adaptive = _read_artifact(experiment, "artifacts/metrics/fixed_vs_adaptive.json")
        for label, training, inference, alerts, rate in (
            ("Fixed ensemble", metrics.get("training_seconds"), metrics.get("inference_seconds"), metrics.get("test_alerts"), metrics.get("test_alert_percent")),
            ("Shadow adaptive", adaptive.get("adaptive_training_seconds"), adaptive.get("adaptive_inference_seconds"), adaptive.get("adaptive_test_alerts"), adaptive.get("adaptive_test_alert_percent")),
        ):
            compare.append({"experiment": experiment, "model": label, "training seconds": training, "inference seconds": inference, "test alerts": alerts, "test alert percent": rate, "test rows": metrics.get("test_rows"), "memory usage": "not recorded"})
    comparison = pd.DataFrame(compare)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(comparison, x="experiment", y="training seconds", color="model", barmode="group", title="Training time"), width="stretch")
    with right:
        st.plotly_chart(px.bar(comparison, x="experiment", y="inference seconds", color="model", barmode="group", title="Held-out inference time"), width="stretch")
    st.dataframe(comparison, width="stretch", hide_index=True)

    benchmark_rows = []
    for experiment in experiments:
        path = ROOT / "experiments" / experiment / "reports" / "benchmark" / "scenario_metrics_summary.csv"
        if path.exists():
            table = pd.read_csv(path)
            table.insert(0, "experiment", experiment)
            benchmark_rows.append(table)
    if benchmark_rows:
        st.subheader("Controlled synthetic evaluation with known injected labels")
        st.dataframe(pd.concat(benchmark_rows, ignore_index=True), width="stretch", hide_index=True)
        st.caption("These metrics compare injected cases with unmodified counterfactual copies. They do not establish verified real-world anomaly accuracy.")
    else:
        st.info("Synthetic benchmark artifacts are not available yet. Historical analyses remain unlabeled.")


def _make_track_figure(frame: pd.DataFrame, selected_model: str, title: str):
    import plotly.graph_objects as go

    if frame.empty:
        return None
    display = frame.sort_values(["vessel_id", "timestamp"], kind="stable").copy()
    display["vessel_display"] = display["vessel_id"].astype(str)
    gaps = display.groupby("vessel_id", sort=False)["timestamp"].diff().dt.total_seconds().div(3600)
    display["track_segment"] = gaps.gt(6).groupby(display["vessel_id"]).cumsum().astype(str)
    display["route_group"] = display["vessel_display"] + "-" + display["track_segment"]
    score_column = f"{selected_model}_score" if f"{selected_model}_score" in display else "anomaly_score"
    flag_column = f"{selected_model}_flag" if f"{selected_model}_flag" in display else "anomaly_flag"
    has_score = score_column in display and display[score_column].notna().any()
    flags = (
        display[flag_column].fillna(False).to_numpy(dtype=bool)
        if flag_column in display else np.zeros(len(display), dtype=bool)
    )
    simulated_flags = (
        display["is_simulated"].fillna(False).to_numpy(dtype=bool)
        if "is_simulated" in display else np.zeros(len(display), dtype=bool)
    )
    display["status"] = [
        "Simulated + flagged" if is_flagged and is_simulated else
        "Flagged" if is_flagged else
        "Simulated, not flagged" if is_simulated else
        "Historical, not flagged"
        for is_flagged, is_simulated in zip(flags, simulated_flags)
    ]
    figure = go.Figure()
    palette = px.colors.qualitative.Safe
    for index, (route_name, segment) in enumerate(display.groupby("route_group", sort=False)):
        vessel = str(segment["vessel_display"].iloc[0])
        color = palette[index % len(palette)]
        figure.add_trace(go.Scattermap(
            lat=pd.to_numeric(segment["latitude"], errors="coerce").tolist(),
            lon=pd.to_numeric(segment["longitude"], errors="coerce").tolist(),
            mode="lines", line={"width": 2, "color": color}, name=f"{vessel} track",
            legendgroup=vessel, showlegend=index < 12,
            hovertemplate=f"Vessel {vessel}<br>Observed route segment<extra></extra>",
        ))
        if len(segment):
            for endpoint, row in (("Track start", segment.iloc[0]), ("Track end / direction", segment.iloc[-1])):
                figure.add_trace(go.Scattermap(
                    lat=[float(row["latitude"])], lon=[float(row["longitude"])], mode="markers",
                    marker={"size": 11, "color": "#f0b65a" if endpoint == "Track start" else "#e66b58",
                            "symbol": "circle" if endpoint == "Track start" else "triangle"},
                    name=endpoint, legendgroup=endpoint, showlegend=index == 0,
                    customdata=[[vessel, str(row["timestamp"]), row.get("cog")]],
                    hovertemplate=("Vessel %{customdata[0]} · " + endpoint +
                                   "<br>%{customdata[1]}<br>Course %{customdata[2]}°<extra></extra>"),
                ))
    for status, segment in display.groupby("status", sort=False):
        status_label = str(status)
        custom_rows = []
        for _, row in segment.iterrows():
            custom_rows.append([
                str(row["vessel_display"]), str(row["timestamp"]),
                row.get("sog"), row.get("cog"), row.get("vessel_type"),
                row.get(score_column) if has_score else None,
                bool(row.get("is_simulated", False)), str(row.get("scenario_state", "historical")),
            ])
        figure.add_trace(go.Scattermap(
            lat=pd.to_numeric(segment["latitude"], errors="coerce").tolist(),
            lon=pd.to_numeric(segment["longitude"], errors="coerce").tolist(),
            mode="markers", marker={"size": 9, "color": {
                "Simulated + flagged": "#ed6a5a", "Flagged": "#f2b84b",
                "Simulated, not flagged": "#5d9cec", "Historical, not flagged": "#56c7b6",
            }.get(status_label, "#56c7b6")},
            name=status_label, customdata=custom_rows,
            hovertemplate=("Vessel %{customdata[0]}<br>Timestamp %{customdata[1]}<br>SOG %{customdata[2]} kn<br>"
                           "COG %{customdata[3]}<br>Type %{customdata[4]}<br>Score %{customdata[5]}<br>"
                           "Simulated %{customdata[6]}<br>Scenario %{customdata[7]}<extra></extra>"),
        ))
    figure.update_layout(
        title=title, map={"style": "open-street-map", "zoom": 2},
        legend_title_text="Track / observation status", margin={"l": 0, "r": 0, "t": 48, "b": 0}, height=560,
    )
    return figure


@st.fragment(run_every="1s")
def _historical_replay(frame: pd.DataFrame, experiment: str, selected_model: str) -> None:
    if frame.empty:
        st.info("No selected observations are available for replay.")
        return
    key = f"history_frame_{experiment}"
    playing_key = f"history_playing_{experiment}"
    steps = pd.Series(frame["timestamp"].dropna().drop_duplicates().sort_values().to_numpy())
    if len(steps) > 50:
        steps = steps.iloc[np.linspace(0, len(steps) - 1, 50, dtype=int)].reset_index(drop=True)
    if key not in st.session_state:
        st.session_state[key] = 0
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False
    controls = st.columns(4)
    if controls[0].button("Play" if not st.session_state[playing_key] else "Pause", key=f"play_{experiment}"):
        st.session_state[playing_key] = not st.session_state[playing_key]
    if controls[1].button("Step forward", key=f"step_{experiment}"):
        st.session_state[key] = (st.session_state[key] + 1) % len(steps)
    if controls[2].button("Reset", key=f"reset_{experiment}"):
        st.session_state[key] = 0
        st.session_state[playing_key] = False
    speed = controls[3].selectbox("Playback", ["1x", "2x", "4x"], key=f"speed_{experiment}")
    if st.session_state[playing_key]:
        increment = {"1x": 1, "2x": 2, "4x": 4}[speed]
        st.session_state[key] = (st.session_state[key] + increment) % len(steps)
    step = st.slider("Historical timeline", 0, max(0, len(steps) - 1), key=key)
    current_time = pd.Timestamp(steps.iloc[step])
    st.caption(f"Historical playback timestamp: {current_time.isoformat()} | Positions are observed AIS points; lines break at gaps over six hours.")
    visible = frame[frame["timestamp"] <= current_time]
    if len(visible) > 2500:
        visible = visible.sample(2500, random_state=42).sort_values("timestamp")
    figure = _make_track_figure(visible, selected_model, "Historical AIS vessel tracks")
    if figure is not None:
        st.plotly_chart(figure, width="stretch", key=f"history_map_{experiment}")


def _render_analytics(experiments: list[str], frames: dict[str, pd.DataFrame], experiment: str) -> None:
    tabs = st.tabs(["Research overview", "Historical map", "Explainable alerts", "Drift & adaptation", "Research outputs"])
    with tabs[0]:
        frame = frames[experiment]
        summary = descriptive_summary(frame)
        manifest, missing_dates = _coverage(experiment)
        quality = _read_artifact(experiment, "artifacts/metrics/cleaning_report.json")
        first, second, third, fourth = st.columns(4)
        first.metric("AIS observations", f"{len(frame):,}")
        second.metric("Distinct vessels", f"{frame['vessel_id'].nunique():,}")
        third.metric("Verified UTC dates", f"{len(manifest.get('verified_dates', [])):,}")
        fourth.metric("Retained / sampled", f"{quality.get('retained_rows', 0):,} / {quality.get('sampled_rows', 0):,}")
        st.caption(f"Observed span: {summary['time_start_utc']} to {summary['time_end_utc']} | Missing expected dates: {', '.join(missing_dates) or 'none'} | Pipeline: {_read_artifact(experiment, 'experiment_summary.json').get('status', 'unknown')}")
        missing = pd.Series(summary["missing_by_column"], name="missing observations").sort_values(ascending=False)
        st.plotly_chart(px.bar(missing, title="Observed missingness by field"), width="stretch")
        model_rows = []
        for model_name, prefix in (("Fixed ensemble", "fixed"), ("Shadow adaptive", "adaptive")):
            evaluation = _read_artifact(experiment, "artifacts/metrics/baseline_evaluation.json")
            adaptive = _read_artifact(experiment, "artifacts/metrics/fixed_vs_adaptive.json")
            model_rows.append({
                "model": model_name,
                "test observations": evaluation.get("test_rows"),
                "flagged": evaluation.get("test_alerts") if prefix == "fixed" else adaptive.get("adaptive_test_alerts"),
                "flagged percent": evaluation.get("test_alert_percent") if prefix == "fixed" else adaptive.get("adaptive_test_alert_percent"),
                "training seconds": evaluation.get("training_seconds") if prefix == "fixed" else adaptive.get("adaptive_training_seconds"),
                "inference seconds": evaluation.get("inference_seconds") if prefix == "fixed" else adaptive.get("adaptive_inference_seconds"),
                "AUC / accuracy": "not available for unlabeled real AIS",
            })
        st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)
        st.info("No independently labeled real AIS is available. Do not interpret alert rates as accuracy, precision, or recall.")

    with tabs[1]:
        frame = frames[experiment].copy()
        scores = load_test_scores(experiment)
        frame["vessel_id"] = frame["vessel_id"].astype(str)
        frame = frame.merge(scores, on=["vessel_id", "timestamp"], how="left", validate="many_to_one")
        min_date = frame["timestamp"].min().date()
        max_date = frame["timestamp"].max().date()
        date_range = st.date_input("UTC date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key=f"map_dates_{experiment}")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            frame = frame[frame["timestamp"].dt.date.between(date_range[0], date_range[1])]
        model_name = st.selectbox("Model score", ["fixed", "shadow_adaptive"], format_func=lambda item: "Fixed ensemble" if item == "fixed" else "Shadow adaptive", key=f"map_model_{experiment}")
        anomaly_status = st.selectbox("Anomaly status", ["All observations", "Flagged", "Unflagged", "Scored observations only"], key=f"map_status_{experiment}")
        counts = frame.groupby("vessel_id", observed=True).size().sort_values(ascending=False)
        visible_default = counts.head(4).index.astype(str).tolist()
        vessel_options = counts.head(100).index.astype(str).tolist()
        visible_vessels = st.multiselect("Visible vessels", vessel_options, default=visible_default, key=f"visible_vessels_{experiment}")
        frame = frame[frame["vessel_id"].isin(visible_vessels)]
        flag_column = f"{model_name}_flag"
        if anomaly_status == "Flagged":
            frame = frame[frame[flag_column].fillna(False)]
        elif anomaly_status == "Unflagged":
            frame = frame[frame[flag_column].eq(False)]
        elif anomaly_status == "Scored observations only":
            frame = frame[frame[f"{model_name}_score"].notna()]
        st.caption("The model score is available on the chronological held-out test partition; other historical points remain explicitly unscored.")
        if anomaly_status in ("All observations", "Scored observations only"):
            _historical_replay(frame, experiment, model_name)
        elif not frame.empty:
            figure = _make_track_figure(frame, model_name, "Filtered historical AIS positions")
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
        if visible_vessels:
            selected_vessel = st.selectbox("Inspect vessel", visible_vessels, key=f"inspect_vessel_{experiment}")
            inspect = frames[experiment][frames[experiment]["vessel_id"].astype(str) == selected_vessel].sort_values("timestamp")
            st.dataframe(inspect[[column for column in ("timestamp", "latitude", "longitude", "sog", "cog", "vessel_type") if column in inspect]], width="stretch", hide_index=True)

    with tabs[2]:
        score_frame = load_test_scores(experiment)
        model_prefix = st.selectbox("Alerting model", ["fixed", "shadow_adaptive"], format_func=lambda item: "Fixed ensemble" if item == "fixed" else "Shadow adaptive", key=f"explain_model_{experiment}")
        flagged = score_frame[score_frame[f"{model_prefix}_flag"]].sort_values(f"{model_prefix}_score", ascending=False)
        if flagged.empty:
            st.info("No flagged test observations are available in this saved model output.")
        else:
            choices = flagged.head(500).copy()
            choices["label"] = choices["timestamp"].astype(str) + " | vessel " + choices["vessel_id"].astype(str)
            selected_label = st.selectbox("Flagged test observation", choices["label"].tolist(), key=f"alert_choice_{experiment}_{model_prefix}")
            selected = choices[choices["label"] == selected_label].iloc[0]
            source = frames[experiment]
            observation = source[(source["vessel_id"].astype(str) == str(selected["vessel_id"])) & (source["timestamp"] == selected["timestamp"])].iloc[0]
            bundle = load_bundle(experiment, "Fixed ensemble" if model_prefix == "fixed" else "Shadow adaptive")
            if bundle is None:
                st.error("The selected saved model bundle is unavailable; local model evidence cannot be displayed.")
                return
            explanation = _score_model(pd.DataFrame([observation]), bundle)
            cols = st.columns(4)
            cols[0].metric("Anomaly score", f"{float(selected[f'{model_prefix}_score']):.3f}")
            cols[1].metric("Decision threshold", f"{float(explanation['threshold']):.3f}")
            cols[2].metric("Latitude / longitude", f"{observation['latitude']:.4f}, {observation['longitude']:.4f}")
            cols[3].metric("Vessel", str(selected["vessel_id"]))
            st.write(f"Timestamp: {selected['timestamp']} | SOG: {observation.get('sog', 'n/a')} knots | COG: {observation.get('cog', 'n/a')} degrees | Vessel type: {observation.get('vessel_type', 'n/a')}")
            other_prefix = "shadow_adaptive" if model_prefix == "fixed" else "fixed"
            matching = score_frame[(score_frame["vessel_id"] == str(selected["vessel_id"])) & (score_frame["timestamp"] == selected["timestamp"])]
            supported = bool(matching.iloc[0][f"{other_prefix}_flag"]) if not matching.empty else False
            st.write(f"Multiple-model agreement: {'both saved models flagged this observation' if supported else 'only the selected model flagged it'}.")
            components = pd.DataFrame({"model score component": list(explanation["components"]), "component anomaly score": [float(v[0]) for v in explanation["components"].values()]})
            st.dataframe(components, width="stretch", hide_index=True)
            vessel_history = source[source["vessel_id"].astype(str) == str(selected["vessel_id"])]
            historical_comparison = {column: {"observation": observation.get(column), "vessel median": vessel_history[column].median()} for column in ("sog", "derived_speed_knots", "distance_km", "speed_change", "turn_rate_deg_per_hour") if column in source}
            st.write("Comparison with this sampled vessel's historical behavior")
            st.dataframe(pd.DataFrame(historical_comparison).T, width="stretch")
            quality_columns = [column for column in observation.index if column.startswith("audit_")]
            warnings = [column for column in quality_columns if bool(observation[column])]
            st.write("Data-quality warnings: " + (", ".join(warnings) if warnings else "none recorded for this row"))
            st.warning("The ensemble exposes Isolation Forest and LOF score components and global permutation sensitivity, but no validated per-observation feature attribution. These are model signals, not causal explanations or proof of unsafe or unlawful behavior.")
            sensitivity_path = ROOT / "experiments" / experiment / "artifacts" / "metrics" / "feature_sensitivity.csv"
            if sensitivity_path.exists():
                st.caption("Global score sensitivity, not this observation's local attribution")
                st.dataframe(pd.read_csv(sensitivity_path).head(10), width="stretch", hide_index=True)

    with tabs[3]:
        drift = _read_artifact(experiment, "artifacts/metrics/drift_report.json")
        baseline_scores_path = ROOT / "experiments" / experiment / "artifacts" / "reports" / "baseline_test_scores.csv"
        if baseline_scores_path.exists():
            baseline_scores = pd.read_csv(baseline_scores_path, parse_dates=["timestamp"])
            st.plotly_chart(px.line(baseline_scores, x="timestamp", y="anomaly_score", title="Fixed model chronological anomaly scores"), width="stretch")
        events = drift.get("events", [])
        st.write(f"Historical score-shift screen: {len(events)} event(s). {drift.get('interpretation', 'No saved interpretation.')}")
        if events:
            st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
        adaptive_path = ROOT / "experiments" / experiment / "reports" / "benchmark" / "fixed_vs_adaptive_drift.csv"
        if adaptive_path.exists():
            st.subheader("Controlled drift replay; synthetic labels")
            adaptation = pd.read_csv(adaptive_path)
            st.dataframe(adaptation, width="stretch", hide_index=True)
            figure = ROOT / "experiments" / experiment / "reports" / "benchmark" / "figures" / "fixed_vs_adaptive_before_after_drift.png"
            if figure.exists():
                st.image(str(figure), width="stretch")
        st.caption("The saved shadow model is a validation refit. Any synthetic post-change refit is a controlled demonstration; neither establishes natural historical concept drift.")

    with tabs[4]:
        st.caption("Result labels: real historical AIS without ground-truth labels; controlled synthetic evaluation with known injected labels; independently verified real-world evaluation is unavailable.")
        paths = sorted((ROOT / "experiments" / experiment / "reports").rglob("*.csv")) + sorted((ROOT / "experiments" / experiment / "reports").rglob("*.json"))
        paths += sorted((ROOT / "experiments" / experiment / "reports").rglob("*.png"))
        if paths:
            selected_path = st.selectbox("Generated research artifact", paths, format_func=lambda item: str(item.relative_to(ROOT)))
            if selected_path.suffix == ".csv":
                st.dataframe(pd.read_csv(selected_path), width="stretch", hide_index=True)
                st.download_button("Download selected CSV", selected_path.read_bytes(), selected_path.name, "text/csv")
            elif selected_path.suffix == ".json":
                st.json(json.loads(selected_path.read_text(encoding="utf-8")))
                st.download_button("Download selected JSON", selected_path.read_bytes(), selected_path.name, "application/json")
            else:
                st.image(str(selected_path), width="stretch")
                st.download_button("Download selected figure", selected_path.read_bytes(), selected_path.name, "image/png")
        else:
            st.info("No optional report artifacts are available for this experiment.")


def _animation_frame_data(frame: pd.DataFrame, speed: str) -> pd.DataFrame:
    usable = frame.dropna(subset=["latitude", "longitude", "timestamp"]).sort_values("timestamp")
    if usable.empty:
        return usable
    unique_times = usable["timestamp"].drop_duplicates().sort_values()
    frame_count = min(36, len(unique_times))
    moments = unique_times.iloc[np.linspace(0, len(unique_times) - 1, frame_count, dtype=int)].tolist()
    rows = []
    for frame_number, moment in enumerate(moments):
        past = usable[usable["timestamp"] <= moment]
        if past.empty:
            continue
        latest = past.groupby("vessel_id", sort=False).tail(1).copy()
        age_hours = (pd.Timestamp(moment) - latest["timestamp"]).dt.total_seconds() / 3600
        latest = latest[age_hours <= 6].copy()
        latest["frame"] = frame_number
        latest["frame_time"] = moment
        latest["position_source"] = "last observed AIS point; not interpolated"
        latest["alert_status"] = np.where(latest["anomaly_flag"], "Flagged", "Not flagged")
        rows.append(latest)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.fragment(run_every="1s")
def _simulator_playback(animated: pd.DataFrame, experiment: str) -> None:
    if animated.empty:
        st.info("No valid timestamped positions are available to replay.")
        return
    total_frames = int(animated["frame"].max()) + 1
    index_key = f"sim_frame_{experiment}"
    playing_key = f"sim_playing_{experiment}"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False
    controls = st.columns(4)
    if controls[0].button("Pause" if st.session_state[playing_key] else "Play", key=f"sim_play_pause_{experiment}"):
        st.session_state[playing_key] = not st.session_state[playing_key]
    if controls[1].button("Step forward", key=f"sim_step_{experiment}"):
        st.session_state[index_key] = (st.session_state[index_key] + 1) % total_frames
    if controls[2].button("Reset playback", key=f"sim_reset_playback_{experiment}"):
        st.session_state[index_key] = 0
        st.session_state[playing_key] = False
    speed = controls[3].selectbox("Playback speed", ["0.5x", "1x", "2x", "4x"], index=1, key=f"sim_play_speed_{experiment}")
    if st.session_state[playing_key]:
        if speed == "0.5x":
            half_tick_key = f"sim_half_tick_{experiment}"
            st.session_state[half_tick_key] = not st.session_state.get(half_tick_key, False)
            increment = int(st.session_state[half_tick_key])
        else:
            increment = {"1x": 1, "2x": 2, "4x": 4}[speed]
        st.session_state[index_key] = (st.session_state[index_key] + increment) % total_frames
    selected_frame = st.slider("Simulation timeline", 0, max(0, total_frames - 1), key=index_key)
    current = animated[animated["frame"] == selected_frame].copy()
    if current.empty:
        return
    frame_time = current["frame_time"].iloc[0]
    st.caption(f"Playback state: {pd.Timestamp(frame_time).isoformat()} | Displayed positions are last observed points no older than six hours; no gap interpolation is used.")
    figure = px.scatter_map(
        current, lat="latitude", lon="longitude", color="alert_status", hover_name="vessel_id",
        hover_data={"timestamp": True, "sog": True, "cog": True, "anomaly_score": ":.3f", "position_source": True},
        map_style="open-street-map", zoom=2,
        color_discrete_map={"Flagged": "#e66b58", "Not flagged": "#56c7b6"},
        title="Current observed vessel positions",
    )
    figure.update_layout(height=500, margin={"l": 0, "r": 0, "t": 48, "b": 0})
    st.plotly_chart(figure, width="stretch", key=f"sim_state_map_{experiment}")


def _render_simulator(experiment: str, data: pd.DataFrame) -> None:
    st.subheader("Interactive animated maritime simulation")
    st.markdown('<span class="result-label">Baseline: real historical AIS | Active scenarios: controlled simulation</span>', unsafe_allow_html=True)
    sizes = data.groupby(data["vessel_id"].astype(str), observed=True).size().sort_values(ascending=False)
    candidates = sizes[sizes >= 20].head(60).index.tolist()
    if not candidates:
        st.warning("No vessel has enough timestamped observations for a replay in this sample.")
        return

    reset = st.button("Reset Scenario", key=f"reset_scenario_{experiment}")
    scenario_keys = {scenario: f"sim_scenario_{experiment}_{index}" for index, scenario in enumerate(SCENARIOS)}
    if reset:
        for scenario, key in scenario_keys.items():
            st.session_state[key] = scenario == "Normal vessel movement"
    toggle_columns = st.columns(4)
    active_scenarios = []
    for index, scenario in enumerate(SCENARIOS):
        checked = scenario == "Normal vessel movement" and not any(st.session_state.get(key, False) for key in scenario_keys.values())
        value = st.checkbox(scenario, value=checked, key=scenario_keys[scenario])
        if value:
            active_scenarios.append(scenario)

    left, right = st.columns([2, 1])
    with left:
        selected_vessels = st.multiselect("Simulator vessels", candidates[:30], default=candidates[: min(3, len(candidates))], key=f"sim_vessels_{experiment}")
        selected_model = st.selectbox("Connected model", list(MODEL_FILES), key=f"sim_model_{experiment}")
    with right:
        severity = st.slider("Scenario severity", 0.5, 5.0, 1.8, 0.1, key=f"sim_severity_{experiment}")
        start_fraction = st.slider("Event start (% of track)", 0, 90, 50, 5, key=f"sim_start_{experiment}") / 100
        duration_fraction = st.slider("Event duration (% of track)", 5, 100, 20, 5, key=f"sim_duration_{experiment}") / 100
        seed = st.number_input("Reproducible random seed", min_value=0, max_value=999999, value=42, step=1, key=f"sim_seed_{experiment}")

    if not selected_vessels:
        st.info("Select at least one vessel to play or simulate.")
        return
    bundle = load_bundle(experiment, selected_model)
    if bundle is None:
        st.error("The selected saved model bundle is unavailable; simulation inference is disabled.")
        return

    simulation_frames = []
    event_rows = []
    normal_rows = []
    for vessel_id in selected_vessels[:5]:
        baseline_track = data[data["vessel_id"].astype(str) == vessel_id].sort_values("timestamp").tail(500).copy()
        normal_rows.append(baseline_track)
        enabled = [scenario for scenario in active_scenarios if scenario != "Normal vessel movement"]
        if enabled:
            modified, event_log = apply_scenarios(baseline_track, enabled, severity, start_fraction, duration_fraction, int(seed))
        else:
            modified, event_log = apply_scenarios(baseline_track, [], severity, start_fraction, duration_fraction, int(seed))
        scored, model_info = score_simulated_track(modified, bundle)
        scored["vessel_id"] = scored["vessel_id"].astype(str)
        scored["source_class"] = "controlled synthetic simulation" if enabled else "real historical AIS analysis without ground-truth labels"
        simulation_frames.append(scored)
        if not event_log.empty:
            event_log["experiment"] = experiment
            event_log["model"] = selected_model
            event_log["seed"] = int(seed)
            event_log["source_class"] = "controlled synthetic simulation"
            lookup = scored[["timestamp", "vessel_id", "anomaly_score", "anomaly_flag", "threshold", "score_isolation_forest", "score_local_outlier_factor"]]
            event_log["vessel_id"] = event_log["vessel_id"].astype(str)
            event_log = event_log.merge(lookup, on=["timestamp", "vessel_id"], how="left")
            event_log["model_explanation"] = "Component anomaly scores available; no validated local feature attribution."
            event_log["drift_status"] = np.where(
                event_log["scenario"].eq("Changed traffic / speed distribution"),
                "controlled shift active; score monitor is descriptive",
                "not a drift scenario",
            )
            event_rows.extend(event_log.to_dict("records"))

    simulated = pd.concat(simulation_frames, ignore_index=True)
    baseline = pd.concat(normal_rows, ignore_index=True)
    current_anomaly_count = int(simulated["anomaly_flag"].sum())
    kpis = st.columns(4)
    kpis[0].metric("Selected vessels", len(simulation_frames))
    kpis[1].metric("Scored observations", f"{len(simulated):,}")
    kpis[2].metric("Model flags", f"{current_anomaly_count:,}")
    kpis[3].metric("Active scenario types", len([s for s in active_scenarios if s != "Normal vessel movement"]))
    if event_rows:
        event_frame = pd.DataFrame(event_rows)
        event_keys = st.session_state.setdefault(f"sim_seen_{experiment}", set())
        log = st.session_state.setdefault(f"sim_log_{experiment}", [])
        for event in event_rows:
            event_key = (str(event["timestamp"]), str(event["vessel_id"]), event["scenario"], event["model"], int(event["seed"]))
            if event_key not in event_keys:
                event_keys.add(event_key)
                log.append(event)
        st.session_state[f"sim_log_{experiment}"] = log[-5000:]

    st.subheader("Vessel movement timeline")
    animated = _animation_frame_data(simulated, "1x")
    if not animated.empty:
        _simulator_playback(animated, experiment)
        route = _make_track_figure(simulated, "fixed", "Observed route history (gaps over six hours are split)")
        if route is not None:
            st.plotly_chart(route, width="stretch")
    else:
        st.warning("No valid coordinates are available for animated display; model inference still uses imputed features where supported.")

    baseline_featured = engineer_trajectory_features(baseline)
    simulation_featured = engineer_trajectory_features(simulated)
    comparison_features = [feature for feature in bundle["features"] if feature in baseline_featured and feature in simulation_featured]
    if comparison_features:
        before_values = baseline_featured[comparison_features].select_dtypes(include=["number"]).mean()
        after_values = simulation_featured[comparison_features].select_dtypes(include=["number"]).mean()
        comparison = pd.DataFrame({"baseline mean": before_values, "simulated mean": after_values})
        comparison["change"] = comparison["simulated mean"] - comparison["baseline mean"]
        st.subheader("Model input feature changes")
        st.dataframe(comparison, width="stretch")
    st.caption(f"Actual inference: {model_info['features']} | threshold: {model_info['threshold']:.3f} | score direction: higher means more unusual. Local causal attribution is unavailable.")

    log = pd.DataFrame(st.session_state.get(f"sim_log_{experiment}", []))
    clear_col, export_col = st.columns(2)
    if clear_col.button("Clear Simulation Events", key=f"clear_sim_log_{experiment}"):
        st.session_state[f"sim_log_{experiment}"] = []
        log = pd.DataFrame()
    if not log.empty:
        export_col.download_button("Export event log CSV", log.to_csv(index=False), f"{experiment}_simulation_events.csv", "text/csv", key=f"export_sim_{experiment}")
        st.dataframe(log.tail(500), width="stretch", hide_index=True)
    else:
        st.caption("No simulation events are logged. Historical source data and research outputs are not affected by Clear Simulation Events.")


def _render_future_mode(experiment: str, data: pd.DataFrame) -> None:
    st.subheader("Future operational mode")
    st.markdown('<span class="result-label">DEMONSTRATION ONLY · archival replay / simulated streaming · no live AIS feed connected</span>', unsafe_allow_html=True)
    manifest, missing_dates = _coverage(experiment)
    summary = _read_artifact(experiment, "experiment_summary.json")
    baseline = _read_artifact(experiment, "artifacts/metrics/baseline_evaluation.json")
    drift = _read_artifact(experiment, "artifacts/metrics/drift_report.json")
    latest = data["timestamp"].max()
    cells = st.columns(4)
    cells[0].metric("Ingestion", "Archival source verified" if summary.get("status") == "completed" else "Unavailable")
    cells[1].metric("Latest AIS timestamp", str(latest))
    cells[2].metric("Vessel count", f"{data['vessel_id'].nunique():,}")
    cells[3].metric("Model version", "Saved ensemble bundle")
    cells = st.columns(4)
    cells[0].metric("Processing latency", f"{baseline.get('inference_seconds', float('nan')):.2f} s / test split" if baseline.get("inference_seconds") is not None else "not measured")
    cells[1].metric("Alert queue", f"{baseline.get('test_alerts', 0):,} historical flags")
    cells[2].metric("Drift screen", f"{len(drift.get('events', []))} saved events")
    cells[3].metric("Missing dates", ", ".join(missing_dates) or "none")
    st.warning("This page is not connected to a real-time feed. Freshness, latency, queue, and drift fields describe saved historical artifacts or demo status only.")
    st.markdown("**Replaceable deployment path:** AIS ingestion adapter → timestamp/schema validation → event processing and feature engineering → versioned model inference → drift monitoring → reviewed alert queue. Live ingestion, durable state, authentication, and operational alert delivery are not implemented in this research demonstration.")


def run_app() -> None:
    st.set_page_config(page_title="Maritime Research Monitor", page_icon="⚓", layout="wide")
    _decorate()
    completed = []
    for experiment in ("1-day", "7-day"):
        summary_path = ROOT / "experiments" / experiment / "experiment_summary.json"
        if summary_path.exists() and load_json(str(summary_path)).get("status") == "completed":
            completed.append(experiment)
    if not completed:
        st.error("No completed verified 1-day or 7-day experiment is available.")
        return
    experiment = st.sidebar.selectbox("Verified experiment", completed, key="active_experiment") or completed[0]
    st.sidebar.caption("NOAA AIS historical research · results are unlabeled unless explicitly marked synthetic")
    frames = {name: load_data(name) for name in completed}
    st.title("Maritime Research Monitor")
    st.caption("Historical vessel behavior, model evidence, and controlled simulation. Anomaly flags warrant review; they do not establish intent.")
    _show_overview(completed, frames)
    analytics_tab, simulator_tab, future_tab = st.tabs(["Research analytics & XAI", "Animated maritime simulator", "Future operational mode"])
    with analytics_tab:
        _render_analytics(completed, frames, experiment)
    with simulator_tab:
        _render_simulator(experiment, frames[experiment])
    with future_tab:
        _render_future_mode(experiment, frames[experiment])