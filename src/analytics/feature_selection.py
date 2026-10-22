from __future__ import annotations

import pandas as pd


def correlation_screen(data: pd.DataFrame, columns: list[str], threshold: float = 0.95) -> dict:
    """Return highly correlated pairs as a redundancy screen, not causal evidence."""
    available = [name for name in columns if name in data]
    if len(available) < 2:
        return {"status": "not_run", "reason": "At least two requested numeric columns are required."}
    correlation = data[available].corr(numeric_only=True).abs()
    pairs = []
    for index, first in enumerate(correlation.columns):
        for second in correlation.columns[index + 1:]:
            value = float(correlation.loc[first, second])
            if value >= threshold:
                pairs.append({"feature_a": first, "feature_b": second, "absolute_correlation": value})
    return {"status": "computed", "threshold": threshold, "pairs": pairs, "interpretation": "Redundancy screen only; correlation does not establish causation."}