"""Drift monitoring and adaptation interfaces."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DriftEvent:
	index: int
	baseline_median: float
	window_median: float
	threshold: float
	evidence: str = "feature-distribution shift; not proof of concept drift"


def detect_median_shift(values, baseline_size: int = 100, window_size: int = 50, threshold_mad: float = 4.0):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	if len(values) < baseline_size + window_size:
		return []
	baseline = values[:baseline_size]
	center = float(np.median(baseline))
	scale = max(float(np.median(np.abs(baseline - center))), 1e-9)
	events = []
	for end in range(baseline_size + window_size, len(values) + 1, window_size):
		current = float(np.median(values[end - window_size:end]))
		if abs(current - center) > threshold_mad * scale:
			events.append(DriftEvent(end - window_size, center, current, threshold_mad * scale))
	return events