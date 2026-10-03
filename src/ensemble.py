"""Equal-weight score ensemble of the GRU autoencoder and the Isolation Forest."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def fit_ensemble(ref_scores: dict) -> dict:
    """ref_scores = {model: scores of NORMAL validation windows}.
    Returns {model: [mean, std]} of the log-scores, used to put both detectors on one scale."""
    stats = {}
    for name, s in ref_scores.items():
        logs = np.log(np.asarray(s) + EPS)
        stats[name] = [float(logs.mean()), float(logs.std())]
    return stats


def ensemble_score(scores: dict, stats: dict) -> np.ndarray:
    """Average of the standardised log-scores (higher = more anomalous)."""
    z = [(np.log(np.asarray(scores[n]) + EPS) - m) / sd for n, (m, sd) in stats.items()]
    return np.mean(z, axis=0)
