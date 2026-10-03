"""Flow-level preprocessing: signed log1p -> z-score (fit on benign train only) -> clip."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CLIP, CORR_THRESHOLD, MIN_STD, NON_FEATURE_COLS, SEED


def signed_log1p(x):
    """log1p that keeps the sign (some CICIDS features are legitimately negative)."""
    return np.sign(x) * np.log1p(np.abs(x))


class FlowPreprocessor:
    """Selects features and scales them. Fit ONLY on benign training flows."""

    def __init__(self, corr_threshold=CORR_THRESHOLD, clip=CLIP, min_std=MIN_STD,
                 sample_size=200_000, seed=SEED):
        self.corr_threshold = corr_threshold
        self.clip = clip
        self.min_std = min_std
        self.sample_size = sample_size
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "FlowPreprocessor":
        cand = [c for c in df.select_dtypes("number").columns if c not in NON_FEATURE_COLS]
        X = df[cand]
        if len(X) > self.sample_size:
            X = X.sample(self.sample_size, random_state=self.seed)
        Xt = pd.DataFrame(signed_log1p(X.to_numpy(dtype=np.float64)), columns=cand)

        Xt = Xt.loc[:, Xt.std() > 1e-9]                       # constant in benign train
        corr = Xt.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        self.dropped_corr_ = [c for c in upper.columns if (upper[c] > self.corr_threshold).any()]
        self.features_ = [c for c in Xt.columns if c not in self.dropped_corr_]

        kept = Xt[self.features_]
        self.mean_ = kept.mean().to_numpy()
        self.std_ = np.maximum(kept.std().to_numpy(), self.min_std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features_ if c not in df.columns]
        if missing:
            raise KeyError(f"Input is missing {len(missing)} expected CICIDS2017 columns, e.g. {missing[:3]}")
        X = df[self.features_].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = (signed_log1p(X) - self.mean_) / self.std_
        return np.clip(X, -self.clip, self.clip).astype(np.float32)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FlowPreprocessor":
        return joblib.load(path)
