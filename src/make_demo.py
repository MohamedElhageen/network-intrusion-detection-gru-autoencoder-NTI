"""Create replayable demo streams: benign traffic -> an attack starts -> traffic continues."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import BENIGN, DEMO_DIR, LABEL_COL, TEST_DAYS


def build_demo_files(df: pd.DataFrame, out_dir=DEMO_DIR, pre_rows: int = 300,
                     post_rows: int = 500, min_attack: int = 10) -> list:
    """One CSV per attack family from the unseen test days (raw CICIDS columns + Label)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["day"].isin(TEST_DAYS)].sort_values("Timestamp", kind="stable").reset_index(drop=True)
    written = []
    for label in sorted(sub.loc[sub[LABEL_COL] != BENIGN, LABEL_COL].unique()):
        pos = np.flatnonzero((sub[LABEL_COL] == label).to_numpy())
        if len(pos) < min_attack:
            continue
        chunk = sub.iloc[max(0, pos[0] - pre_rows): min(len(sub), pos[0] + post_rows)]
        name = "demo_" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") + ".csv"
        chunk.to_csv(out_dir / name, index=False)
        written.append(name)
    return written
