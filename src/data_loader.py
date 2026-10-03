"""Loading and cleaning the raw CICIDS2017 'GeneratedLabelledFlows' CSV files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BENIGN, LABEL_COL, PROC_DIR, RAW_DIR


def normalize_labels(s: pd.Series) -> pd.Series:
    """'Web Attack \\x96 Brute Force' -> 'Web Attack - Brute Force' (latin1 artefact)."""
    return (
        s.astype(str)
        .str.replace("\x96", "-", regex=False)
        .str.replace("\ufffd", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_timestamps(s: pd.Series) -> pd.Series:
    """Parse CICIDS2017 timestamps correctly.

    Three quirks of the raw files:
      1. dates are day-first (3/7/2017 = 3 July, not 7 March);
      2. some rows have seconds ("8:42:15") and some do not ("8:42"), so a single
         inferred format turns the others into NaT -> use format="mixed";
      3. times are 12-hour clock WITHOUT AM/PM; the capture runs ~08:00-17:00,
         so hours 1..7 are really 13..19 (afternoon).
    Already-ISO strings (e.g. our own demo CSVs) are parsed as they are.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    s = s.astype(str)
    if s.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.to_datetime(s, format="ISO8601", errors="coerce")
    ts = pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")
    hour = ts.dt.hour
    pm = ((hour >= 1) & (hour <= 7)).astype(int) * 12
    return ts + pd.to_timedelta(pm, unit="h")


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as notebook 01, in one reusable function."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df[LABEL_COL] = normalize_labels(df[LABEL_COL])

    num = df.select_dtypes("number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().drop_duplicates()

    const = [c for c in df.select_dtypes("number").columns if df[c].nunique() <= 1]
    df = df.drop(columns=const)

    df["Timestamp"] = parse_timestamps(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    df["is_attack"] = (df[LABEL_COL] != BENIGN).astype(np.int8)
    df["day"] = df["Timestamp"].dt.day_name()
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read every CSV in raw_dir, concatenate and clean."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    frames = [pd.read_csv(f, encoding="latin1", low_memory=False) for f in files]
    for f in frames:
        f.columns = f.columns.str.strip()
    return clean_raw(pd.concat(frames, ignore_index=True))


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Load data/processed/clean.parquet (written by notebook 01), else the raw CSVs."""
    path = Path(path) if path else PROC_DIR / "clean.parquet"
    if not path.exists():
        return load_raw()
    df = pd.read_parquet(path)
    df[LABEL_COL] = normalize_labels(df[LABEL_COL])
    df["Timestamp"] = parse_timestamps(df["Timestamp"])
    df["is_attack"] = (df[LABEL_COL] != BENIGN).astype(np.int8)
    df["day"] = df["Timestamp"].dt.day_name()
    return df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
