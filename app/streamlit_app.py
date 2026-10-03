"""Network intrusion detector - Streamlit demo.

Run from the repo root:   streamlit run app/streamlit_app.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.baseline import window_scores
from src.config import BENIGN, DEMO_DIR, LABEL_COL, MODEL_DIR
from src.data_loader import normalize_labels, parse_timestamps
from src.ensemble import ensemble_score
from src.models import load_model, reconstruction_errors
from src.preprocessing import FlowPreprocessor
from src.sequences import window_index

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------- artifacts
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "threshold.json").read_text())
    iso_path = MODEL_DIR / "isolation_forest.joblib"
    iso = joblib.load(iso_path) if ("ensemble" in meta and iso_path.exists()) else None
    return (FlowPreprocessor.load(MODEL_DIR / "preprocessor.joblib"),
            load_model(MODEL_DIR / "gru_ae.pt"), iso, meta)


def read_flows(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    if "Timestamp" in df.columns:
        df["Timestamp"] = parse_timestamps(df["Timestamp"])
        df = df.sort_values("Timestamp", kind="stable")
    if LABEL_COL in df.columns:
        df[LABEL_COL] = normalize_labels(df[LABEL_COL])
    return df.reset_index(drop=True)


def score_flows(df, pre, model, iso, meta, use_ensemble):
    """Non-overlapping windows of consecutive flows -> anomaly score per window.

    Returns (scores, ground-truth label per window or None, first flow row of each window).
    """
    w = meta["window"]
    X = pre.transform(df)
    key = meta.get("group_by")
    keys = df[key].to_numpy() if key and key in df.columns else None
    idx = window_index(len(X), w, w, keys)
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


def window_table(score, truth, first_rows, thr, det_meta):
    t = pd.DataFrame({"window": np.arange(len(score)), "first_flow_row": first_rows,
                      "score": score, "margin": score - thr, "confidence": confidence(score, det_meta)})
    if truth is not None:
        t["ground_truth"] = truth
    return t


# ----------------------------------------------------------------------- UI
st.title("🛡️ Network Intrusion Detector")
st.caption("Trained on normal traffic only. A window of consecutive flows that looks too unlike normal "
           "traffic (high anomaly score) is flagged. Attacks are detected without ever having been seen in training.")

try:
    pre, model, iso, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in `models/`. Run `python -m src.train` first (see README).")
    st.stop()

W = meta["window"]
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Simulate an attack (live replay)", "Upload traffic CSV"])
    options = ["GRU autoencoder"] + (["Ensemble: GRU + Isolation Forest"] if iso is not None else [])
    det_name = st.radio("Detector", options, index=len(options) - 1)
    use_ensemble = det_name.startswith("Ensemble")
    det_meta = meta["ensemble"] if use_ensemble else meta
    thr = st.number_input("Anomaly threshold", min_value=-10.0, max_value=100.0,
                          value=float(det_meta["threshold"]), step=0.01, format="%.4f",
                          key=f"thr_{use_ensemble}",
                          help="Saved default was chosen on the validation days. Lower = more sensitive "
                               "(more attacks caught, more false alarms).")
    grp = f" to the same {meta['group_by']}" if meta.get("group_by") else ""
    st.caption(f"Window = {W} consecutive flows{grp} · model input = {meta['n_features']} features")

# ------------------------------------------------------------ simulate mode
if mode.startswith("Simulate"):
    demos = sorted(DEMO_DIR.glob("demo_*.csv"))
    if not demos:
        st.warning("No demo streams in `data/demo/`. Run `python -m src.train` to create them.")
        st.stop()
    names = {p.stem.replace("demo_", "").replace("_", " "): p for p in demos}
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Attack scenario (replayed from real CICIDS2017 traffic the model never trained on)",
                          list(names))
    delay = c2.slider("Seconds per window", 0.0, 1.0, 0.15, 0.05)

    if st.button("▶ Start live replay", type="primary"):
        df = read_flows(names[choice])
        res = score_flows(df, pre, model, iso, meta, use_ensemble)
        if res is None:
            st.error("Stream too short for one window.")
            st.stop()
        score, truth, first_rows = res
        banner, chart = st.empty(), st.empty()
        alerts = []
        for i in range(len(score)):
            chart.line_chart(pd.DataFrame({"anomaly score": score[: i + 1],
                                           "threshold": np.full(i + 1, thr)}))
            truth_txt = f" · ground truth: **{truth[i]}**" if truth is not None else ""
            if score[i] > thr:
                conf = float(confidence(score[i], det_meta))
                banner.error(f"🚨 ALERT — window {i}: score {score[i]:.3f} (threshold {thr:.3f}) · "
                             f"more anomalous than {conf:.1%} of normal traffic{truth_txt}")
                alerts.append(i)
            else:
                banner.success(f"✅ Normal — window {i}{truth_txt}")
            time.sleep(delay)
        tbl = window_table(score, truth, first_rows, thr, det_meta)
        st.subheader(f"Summary: {len(alerts)} of {len(score)} windows flagged")
        if truth is not None:
            is_att = truth != BENIGN
            flagged = score > thr
            m1, m2, m3 = st.columns(3)
            m1.metric("Attack windows caught", f"{flagged[is_att].sum()} / {is_att.sum()}")
            m2.metric("False alarms on normal windows", f"{flagged[~is_att].sum()} / {(~is_att).sum()}")
            m3.metric("First alert at window", alerts[0] if alerts else "—",
                      help=f"attack begins at window ≈ {int(np.argmax(is_att))}" if is_att.any() else None)
        st.dataframe(tbl[tbl["score"] > thr].round(4), use_container_width=True, hide_index=True)

# -------------------------------------------------------------- upload mode
else:
    up = st.file_uploader("CICIDS2017-format CSV (flow features, one row per flow, time-ordered)", type="csv")
    if up is not None:
        df = read_flows(up)
        try:
            res = score_flows(df, pre, model, iso, meta, use_ensemble)
        except KeyError as e:
            st.error(str(e).strip("'"))
            st.stop()
        if res is None:
            st.error(f"Need at least {W} flows.")
            st.stop()
        score, truth, first_rows = res
        flagged = score > thr
        a, b, c = st.columns(3)
        a.metric("Flows analysed", f"{len(df):,}")
        b.metric("Windows flagged", f"{flagged.sum()} / {len(score)}")
        c.metric("Share flagged", f"{flagged.mean():.1%}")
        st.line_chart(pd.DataFrame({"anomaly score": score, "threshold": np.full(len(score), thr)}))
        tbl = window_table(score, truth, first_rows, thr, det_meta)
        st.subheader("Flagged windows")
        st.dataframe(tbl[flagged].sort_values("score", ascending=False).round(4),
                     use_container_width=True, hide_index=True)
        if truth is not None:
            st.caption("Ground truth found in the `Label` column: "
                       f"{int((truth != BENIGN).sum())} attack windows in this file.")
    else:
        st.info("Upload a file, or switch to *Simulate an attack* in the sidebar for the demo.")
