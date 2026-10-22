from __future__ import annotations

import pandas as pd

from src.schema import discover_columns, validate_schema


def canonicalize(raw: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    mapping = discover_columns(raw.columns, aliases)
    validate_schema(mapping)
    result = raw.rename(columns={source: target for target, source in mapping.items()}).copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    for column in ("latitude", "longitude", "sog", "cog", "heading"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def clean_positions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Keep questionable rows with audit flags; exclude only unusable coordinates/time."""
    frame = data.copy()
    frame["audit_invalid_timestamp"] = frame["timestamp"].isna()
    frame["audit_invalid_coordinate"] = (
        frame["latitude"].isna()
        | frame["longitude"].isna()
        | ~frame["latitude"].between(-90, 90)
        | ~frame["longitude"].between(-180, 180)
    )
    frame["audit_duplicate"] = frame.duplicated(
        ["vessel_id", "timestamp", "latitude", "longitude"], keep="first"
    )
    frame["audit_suspicious_speed"] = False
    if "sog" in frame:
        frame["audit_suspicious_speed"] = frame["sog"].notna() & (
            (frame["sog"] < 0) | (frame["sog"] > 70)
        )
    frame["audit_missing_vessel_id"] = frame["vessel_id"].isna()
    excluded = frame["audit_invalid_timestamp"] | frame["audit_invalid_coordinate"]
    clean = frame.loc[~excluded & ~frame["audit_duplicate"]].copy()
    clean = clean.sort_values(["vessel_id", "timestamp"], kind="stable")
    report = {
        "input_rows": int(len(frame)),
        "retained_rows": int(len(clean)),
        "excluded_unusable_rows": int(excluded.sum()),
        "duplicate_rows_flagged": int(frame["audit_duplicate"].sum()),
        "suspicious_speed_rows_flagged": int(frame["audit_suspicious_speed"].sum()),
        "missing_vessel_id_rows_flagged": int(frame["audit_missing_vessel_id"].sum()),
        "missingness_before": frame.isna().sum().astype(int).to_dict(),
        "missingness_after": clean.isna().sum().astype(int).to_dict(),
    }
    return clean, frame, report