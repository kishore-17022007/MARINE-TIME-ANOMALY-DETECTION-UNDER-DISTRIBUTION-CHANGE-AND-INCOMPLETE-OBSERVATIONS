from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def labelled_metrics(labels, scores) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"status": "not_computable", "reason": "both positive and negative labels are required"}
    return {
        "status": "computed",
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_curve": [array.tolist() for array in roc_curve(labels, scores)[:3]],
        "precision_recall_curve": [array.tolist() for array in precision_recall_curve(labels, scores)],
    }