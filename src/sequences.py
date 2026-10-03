"""Turn a time-ordered stream of flows into sliding windows (the time-series step)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_index(n: int, window: int, stride: int, keys=None) -> np.ndarray:
    """Row indices (n_windows, window) of each window.

    keys=None : windows slide over the whole time-ordered stream.
    keys=array: a window only contains consecutive flows that share the same key
                (e.g. Destination IP); windows are returned in time order of their first flow.
    Rows of the input must already be sorted by time.
    """
    if keys is None:
        starts = np.arange(0, n - window + 1, stride)
        return starts[:, None] + np.arange(window)[None, :]
    codes = pd.factorize(np.asarray(keys))[0]
    order = np.argsort(codes, kind="stable")              # keeps time order inside each key
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    g_start, g_end = np.r_[0, bounds], np.r_[bounds, n]
    parts = [s + np.arange(0, e - s - window + 1, stride)
             for s, e in zip(g_start, g_end) if e - s >= window]
    if not parts:
        return np.empty((0, window), dtype=np.int64)
    starts = np.concatenate(parts)
    idx = order[starts[:, None] + np.arange(window)[None, :]]
    return idx[np.argsort(idx[:, 0], kind="stable")]


def make_windows(X: np.ndarray, y: np.ndarray, type_codes: np.ndarray, window: int, stride: int, keys=None):
    """Build windows of `window` consecutive flows.

    Returns
    -------
    W      (n_windows, window, n_features) float32
    y_win  1 if ANY flow in the window is an attack (an alert is useful even if
           only part of the window is malicious), else 0
    t_win  attack-type code = most frequent non-benign type in the window, 0 if benign
    """
    idx = window_index(len(X), window, stride, keys)
    if len(idx) == 0:
        return (np.empty((0, window, X.shape[1]), np.float32), np.empty(0, np.int8), np.empty(0, np.int16))

    W = X[idx]
    y_win = y[idx].max(axis=1).astype(np.int8)

    codes = type_codes[idx]
    k_max = int(type_codes.max()) if len(type_codes) else 0
    if k_max == 0:
        t_win = np.zeros(len(idx), dtype=np.int16)
    else:
        counts = np.stack([(codes == k).sum(axis=1) for k in range(1, k_max + 1)], axis=1)
        t_win = np.where(y_win == 1, counts.argmax(axis=1) + 1, 0).astype(np.int16)
    return W, y_win, t_win
