from __future__ import annotations

from collections.abc import Mapping, Sequence


CANONICAL_FIELDS = ("vessel_id", "timestamp", "latitude", "longitude")


def discover_columns(
    available: Sequence[str], aliases: Mapping[str, Sequence[str]]
) -> dict[str, str]:
    """Map configured aliases to actual source columns without guessing silently."""
    normalized = {str(name).casefold(): str(name) for name in available}
    found: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            actual = normalized.get(str(candidate).casefold())
            if actual is not None:
                found[canonical] = actual
                break
    return found


def validate_schema(columns: Mapping[str, str]) -> None:
    missing = [field for field in CANONICAL_FIELDS if field not in columns]
    if missing:
        raise ValueError(
            "Required AIS fields were not mapped: " + ", ".join(missing)
        )