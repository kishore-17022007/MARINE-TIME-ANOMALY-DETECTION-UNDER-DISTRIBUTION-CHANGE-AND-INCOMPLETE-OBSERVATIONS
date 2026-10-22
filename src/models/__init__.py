"""Baseline, temporal, and ensemble model interfaces."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


class AnomalyEnsemble:
	"""Unsupervised baseline combining robust-scaled IF and novelty LOF."""

	def __init__(self, contamination: float = 0.01, alert_quantile: float = 0.99, seed: int = 42):
		self.contamination = contamination
		self.alert_quantile = alert_quantile
		self.seed = seed

	def fit(self, values):
		values = np.asarray(values, dtype=float)
		self.scaler = RobustScaler().fit(values)
		scaled = self.scaler.transform(values)
		self.isolation = IsolationForest(contamination=self.contamination, random_state=self.seed, n_estimators=200).fit(scaled)
		neighbors = min(max(2, int(np.sqrt(len(scaled)))), len(scaled) - 1)
		self.lof = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, contamination=self.contamination).fit(scaled)
		scores = self._component_scores(scaled)
		self.centers = {name: float(np.median(score)) for name, score in scores.items()}
		self.scales = {name: max(float(np.median(np.abs(score - self.centers[name]))), 1e-9) for name, score in scores.items()}
		self.threshold = float(np.quantile(self._fuse(scores), self.alert_quantile))
		return self

	def _component_scores(self, values):
		return {"isolation_forest": -self.isolation.score_samples(values), "local_outlier_factor": -self.lof.score_samples(values)}

	def _fuse(self, scores):
		return np.mean([(scores[name] - self.centers[name]) / self.scales[name] for name in scores], axis=0)

	def predict_scores(self, values):
		components = self._component_scores(self.scaler.transform(np.asarray(values, dtype=float)))
		score = self._fuse(components)
		return {"score": score, "flag": score >= self.threshold, "components": components, "threshold": self.threshold}