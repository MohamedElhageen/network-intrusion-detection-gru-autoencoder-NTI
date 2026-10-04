"""Shared pieces for the Streamlit pages: model artifacts, scoring helpers, styling and UI components."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.baseline import window_scores
from src.config import BENIGN, DEMO_DIR, LABEL_COL, MODEL_DIR, RESULTS_DIR
from src.data_loader import normalize_labels, parse_timestamps
from src.ensemble import ensemble_score
from src.models import load_model, reconstruction_errors
from src.preprocessing import FlowPreprocessor
from src.sequences import window_index

C = dict(bg="#0B1E33", panel="#12294A", panel2="#16325A", border="#1E3A5C", text="#E8EEF5",
         muted="#9AA9B8", teal="#19C3B1", red="#E5484D", amber="#F2A541", steel="#5B8FD0")


# ----------------------------------------------------------------- artifacts
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "threshold.json").read_text())
    iso_path = MODEL_DIR / "isolation_forest.joblib"
    iso = joblib.load(iso_path) if ("ensemble" in meta and iso_path.exists()) else None
    return (FlowPreprocessor.load(MODEL_DIR / "preprocessor.joblib"),
            load_model(MODEL_DIR / "gru_ae.pt"), iso, meta)


@st.cache_data
def iso_threshold(rule: str = "pct95"):
    """Isolation Forest's own alarm line (not stored in threshold.json; read from the results table)."""
    path = RESULTS_DIR / "metrics_test.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path)
    row = t[(t["model"] == "isolation_forest") & (t["threshold_rule"] == rule)]
    return float(row["threshold"].iloc[0]) if len(row) else None


def demo_files() -> dict:
    return {p.stem.replace("demo_", "").replace("_", " "): p for p in sorted(DEMO_DIR.glob("demo_*.csv"))}


def read_flows(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    if "Timestamp" in df.columns:
        df["Timestamp"] = parse_timestamps(df["Timestamp"])
        df = df.sort_values("Timestamp", kind="stable")
    if LABEL_COL in df.columns:
        df[LABEL_COL] = normalize_labels(df[LABEL_COL])
    return df.reset_index(drop=True)


def windows_of(df, meta):
    """Row indices (n_windows, window) of non-overlapping windows, as used in evaluation and the app."""
    w = meta["window"]
    key = meta.get("group_by")
    keys = df[key].to_numpy() if key and key in df.columns else None
    return window_index(len(df), w, w, keys)


def score_flows(df, pre, model, iso, meta, use_ensemble):
    """Non-overlapping windows of consecutive flows -> anomaly score per window.

    Returns (scores, ground-truth label per window or None, first flow row of each window).
    """
    X = pre.transform(df)
    idx = windows_of(df, meta)
    if len(idx) == 0:
        return None
    Xw = X[idx]
    score = reconstruction_errors(model, Xw)
    if use_ensemble:
        score = ensemble_score({"gru_ae": score, "isolation_forest": window_scores(iso, Xw)},
                               meta["ensemble"]["stats"])
    truth = None
    if LABEL_COL in df.columns:
        labels = df[LABEL_COL].to_numpy()[idx]
        truth = np.array([next((x for x in row if x != BENIGN), BENIGN) for row in labels])
    return score, truth, idx[:, 0]


def confidence(score, det_meta):
    """Share of normal validation windows that look LESS anomalous than this one."""
    q = np.asarray(det_meta["benign_quantiles"])
    return np.searchsorted(q, score) / (len(q) - 1)


# ----------------------------------------------------------------- styling
CSS = f"""
<style>
:root {{ --teal:{C['teal']}; --red:{C['red']}; --amber:{C['amber']}; --panel:{C['panel']};
         --border:{C['border']}; --muted:{C['muted']}; --text:{C['text']}; }}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
    font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif !important; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"], footer {{ display: none; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ font-size: 0.95rem; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted); }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* hero banner */
.hero {{ background: linear-gradient(135deg, #123156 0%, #0B1E33 70%); border: 1px solid var(--border);
         border-radius: 18px; padding: 26px 30px 22px; margin-bottom: 22px; }}
.hero-kicker {{ color: var(--teal); font-size: .78rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }}
.hero-title {{ font-size: 2.1rem; font-weight: 700; color: var(--text); margin: 6px 0 4px; }}
.hero-sub {{ color: var(--muted); font-size: 1rem; max-width: 900px; }}
.chips {{ margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{ background: rgba(25,195,177,.12); color: var(--teal); border: 1px solid rgba(25,195,177,.35);
         border-radius: 999px; padding: 3px 12px; font-size: .78rem; font-weight: 600; }}

/* section titles */
.sec {{ margin: 30px 0 10px; }}
.sec-kicker {{ color: var(--teal); font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }}
.sec-title {{ font-size: 1.45rem; font-weight: 700; color: var(--text); margin-top: 2px; }}
.sec-sub {{ color: var(--muted); font-size: .95rem; margin-top: 4px; max-width: 1000px; }}

/* stat cards */
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 8px 0 6px; }}
.stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px 18px; }}
.stat-label {{ color: var(--muted); font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}
.stat-value {{ font-size: 1.85rem; font-weight: 700; margin-top: 2px; line-height: 1.15; }}
.stat-note {{ color: var(--muted); font-size: .8rem; margin-top: 2px; }}

/* alert banners */
.banner {{ border-radius: 14px; padding: 14px 18px; font-size: 1.02rem; display: flex; gap: 12px; align-items: center;
           border: 1px solid; margin: 4px 0 10px; }}
.banner b {{ font-weight: 700; }}
.banner-alert {{ background: rgba(229,72,77,.14); border-color: rgba(229,72,77,.55); color: #FFD9DA; }}
.banner-ok {{ background: rgba(25,195,177,.10); border-color: rgba(25,195,177,.45); color: #C9F4EE; }}
.banner-info {{ background: rgba(91,143,208,.12); border-color: rgba(91,143,208,.45); color: #D5E4F7; }}
.banner-icon {{ font-size: 1.35rem; }}

/* explanation callout */
.note {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px 18px;
         color: var(--text); font-size: .95rem; line-height: 1.5; }}
.note b {{ color: var(--teal); }}

/* equation table */
.eq {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--panel); border: 1px solid var(--border);
       border-radius: 14px; overflow: hidden; font-size: .95rem; }}
.eq th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: .78rem; letter-spacing: .06em;
          text-transform: uppercase; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
.eq td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); color: var(--text); }}
.eq tr:last-child td {{ border-bottom: none; font-weight: 700; background: rgba(25,195,177,.08); }}
.eq .num {{ font-variant-numeric: tabular-nums; text-align: right; }}

/* verdict pills */
.verdicts {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.verdict {{ border-radius: 14px; padding: 12px 16px; border: 1px solid var(--border); background: var(--panel); }}
.verdict .v-name {{ color: var(--muted); font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }}
.verdict .v-res {{ font-size: 1.15rem; font-weight: 700; margin-top: 2px; }}
.verdict .v-detail {{ color: var(--muted); font-size: .82rem; }}

/* widgets */
div[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 12px 16px; }}
.stButton > button {{ border-radius: 10px; font-weight: 600; }}
div[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; }}
</style>
"""


def apply_style():
    st.html(CSS)


def hero(kicker, title, sub, chips=()):
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.html(f'<div class="hero"><div class="hero-kicker">{kicker}</div><div class="hero-title">{title}</div>'
            f'<div class="hero-sub">{sub}</div>{f"<div class=chips>{chip_html}</div>" if chips else ""}</div>')


def section(kicker, title, sub=""):
    st.html(f'<div class="sec"><div class="sec-kicker">{kicker}</div><div class="sec-title">{title}</div>'
            f'{f"<div class=sec-sub>{sub}</div>" if sub else ""}</div>')


def stats(items):
    """items: list of (label, value, colour, note)."""
    cards = "".join(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        f'{f"<div class=stat-note>{note}</div>" if note else ""}</div>'
        for label, value, color, note in items)
    return f'<div class="stats">{cards}</div>'


def banner(kind, icon, text):
    return f'<div class="banner banner-{kind}"><span class="banner-icon">{icon}</span><span>{text}</span></div>'


def chart_style(chart, height=None):
    """Dark, theme-matched Altair styling."""
    if height:
        chart = chart.properties(height=height)
    return (chart.configure(background="transparent", font="Segoe UI")
                 .configure_view(strokeWidth=0)
                 .configure_axis(labelColor=C["muted"], titleColor=C["muted"], gridColor=C["border"],
                                 domainColor=C["border"], tickColor=C["border"], labelFontSize=11, titleFontSize=12,
                                 titleFontWeight=600)
                 .configure_legend(labelColor=C["text"], titleColor=C["muted"], labelFontSize=11, orient="bottom")
                 .configure_title(color=C["text"], fontSize=14, anchor="start", fontWeight=600))


def score_chart(score, thr, total, title, truth=None, selected=None):
    """Anomaly score per window with the alarm line, alerts in red, optional ground truth and selected window."""
    d = pd.DataFrame({"window": np.arange(len(score)), "score": score})
    d["state"] = np.where(d["score"] > thr, "alert", "normal")
    lo = min(float(d["score"].min()), thr)
    hi = max(float(d["score"].max()), thr)
    pad = (hi - lo) * 0.12 or 1.0
    d["base"] = lo - pad
    x = alt.X("window:Q", title="window (10 consecutive flows each)", scale=alt.Scale(domain=[0, max(total - 1, 1)]))
    y = alt.Y("score:Q", title="anomaly score", scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False))
    layers = []
    if truth is not None:
        att = pd.DataFrame({"window": np.flatnonzero(np.asarray(truth) != BENIGN)})
        if len(att):
            layers.append(alt.Chart(att).mark_rule(color=C["red"], opacity=0.16, strokeWidth=6).encode(x="window:Q"))
    layers += [
        alt.Chart(d).mark_area(color=C["teal"], opacity=0.10).encode(x=x, y=y, y2="base:Q"),
        alt.Chart(d).mark_line(color=C["teal"], strokeWidth=2.2).encode(x=x, y=y),
        alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color=C["amber"], strokeDash=[6, 4], strokeWidth=2).encode(y="t:Q"),
        alt.Chart(d[d["state"] == "alert"]).mark_circle(color=C["red"], size=70, opacity=1).encode(
            x=x, y=y, tooltip=[alt.Tooltip("window:Q"), alt.Tooltip("score:Q", format=".4f")]),
    ]
    if selected is not None:
        layers.append(alt.Chart(pd.DataFrame({"window": [selected]})).mark_rule(color=C["text"], strokeWidth=2).encode(x="window:Q"))
    return chart_style(alt.layer(*layers).properties(title=title), height=300)
