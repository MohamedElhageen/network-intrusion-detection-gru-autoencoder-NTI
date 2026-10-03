"""Classical baseline: Isolation Forest on individual flows, aggregated per window."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .config import SEED


def fit_isolation_forest(flows: np.ndarray, n_estimators: int = 100,
                         max_flows: int = 300_000, seed: int = SEED) -> IsolationForest:
    """Fit on benign flows only (n, n_features)."""
    if len(flows) > max_flows:
        rng = np.random.default_rng(seed)
        flows = flows[rng.choice(len(flows), max_flows, replace=False)]
    model = IsolationForest(n_estimators=n_estimators, max_samples=256,
                            random_state=seed, n_jobs=-1)
    return model.fit(flows)


def window_scores(model: IsolationForest, W: np.ndarray, agg: str = "mean") -> np.ndarray:
    """Anomaly score per window = mean (or max) of the per-flow scores. Higher = more anomalous."""
    n, w, f = W.shape
    flow_scores = -model.score_samples(W.reshape(-1, f)).reshape(n, w)
    return flow_scores.max(axis=1) if agg == "max" else flow_scores.mean(axis=1)
