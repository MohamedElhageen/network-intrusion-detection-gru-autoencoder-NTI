"""Build train / val / test window datasets from the cleaned flow table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (BENIGN, LABEL_COL, PROC_DIR, STRIDE_EVAL, STRIDE_TRAIN,
                     TEST_DAYS, TRAIN_DAYS, VAL_DAYS, WINDOW, WINDOW_GROUP_BY)
from .preprocessing import FlowPreprocessor
from .sequences import make_windows


def fit_preprocessor(df: pd.DataFrame) -> FlowPreprocessor:
    """Fit on benign Monday flows only (no attack information, no test information)."""
    train = df[df["day"].isin(TRAIN_DAYS) & (df["is_attack"] == 0)]
    if train.empty:
        raise ValueError(f"No benign training flows found for days {TRAIN_DAYS}")
    return FlowPreprocessor().fit(train)


def build_splits(df: pd.DataFrame, pre: FlowPreprocessor, window: int = WINDOW) -> dict:
    """Return a dict with X_/y_/t_ arrays for 'train', 'val', 'test' + class names.

    Windows never cross a day boundary. Each window is labelled attack if it
    contains at least one attack flow; t_ holds the attack-type code.
    """
    classes = [BENIGN] + sorted(c for c in df[LABEL_COL].unique() if c != BENIGN)
    code = {c: i for i, c in enumerate(classes)}
    plan = {"train": (TRAIN_DAYS, STRIDE_TRAIN), "val": (VAL_DAYS, STRIDE_EVAL),
            "test": (TEST_DAYS, STRIDE_EVAL)}

    out = {"classes": np.array(classes), "window": np.array(window),
           "stride_train": np.array(STRIDE_TRAIN), "group_by": np.array(WINDOW_GROUP_BY or "")}
    for split, (days, stride) in plan.items():
        parts = []
        for day in days:
            d = df[df["day"] == day]
            if d.empty:
                continue
            keys = d[WINDOW_GROUP_BY].to_numpy() if WINDOW_GROUP_BY else None
            parts.append(make_windows(pre.transform(d), d["is_attack"].to_numpy(),
                                      d[LABEL_COL].map(code).to_numpy(), window, stride, keys))
        if not parts:
            raise ValueError(f"No data for {split} days {days}")
        out[f"X_{split}"] = np.concatenate([p[0] for p in parts])
        out[f"y_{split}"] = np.concatenate([p[1] for p in parts])
        out[f"t_{split}"] = np.concatenate([p[2] for p in parts])
    return out


def save_splits(splits: dict, path=None) -> None:
    path = path or PROC_DIR / "sequences.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **splits)


def load_splits(path=None) -> dict:
    with np.load(path or PROC_DIR / "sequences.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
