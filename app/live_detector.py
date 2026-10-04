"""Page 1 - Live Detector: replay a recorded attack or upload a traffic file."""
import time

import numpy as np
import pandas as pd
import streamlit as st

from common import (BENIGN, C, banner, confidence, demo_files, hero, load_artifacts, read_flows,
                    score_chart, score_flows, section, stats)


def window_table(score, truth, first_rows, thr, det_meta):
    t = pd.DataFrame({"window": np.arange(len(score)), "first_flow_row": first_rows,
                      "score": score, "margin": score - thr, "confidence": confidence(score, det_meta)})
    if truth is not None:
        t["ground_truth"] = truth
    return t


TABLE_CFG = {
    "score": st.column_config.NumberColumn("score", format="%.4f"),
    "margin": st.column_config.NumberColumn("margin over line", format="%.4f"),
    "confidence": st.column_config.ProgressColumn("more unusual than", format="percent", min_value=0, max_value=1),
}

hero("NTI · Machine Learning Capstone", "🛡️ Network Intrusion Detector",
     "Trained on normal traffic only. A window of 10 consecutive flows that looks too unlike normal traffic "
     "(high anomaly score) raises an alert, even for attack types never seen in training.",
     chips=("GRU autoencoder", "Isolation Forest", "Ensemble", "CICIDS2017"))

try:
    pre, model, iso, meta = load_artifacts()
except FileNotFoundError:
    st.html(banner("alert", "⚠️", "Model files not found in <b>models/</b>. Run <b>python -m src.train</b> first (see README)."))
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
    names = demo_files()
    if not names:
        st.html(banner("alert", "⚠️", "No demo streams in <b>data/demo/</b>. Run <b>python -m src.train</b> to create them."))
        st.stop()
    section("Live replay", "Replay a real attack",
            "Recorded CICIDS2017 traffic from days the model never trained on, played back window by window.")
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Attack scenario", list(names))
    delay = c2.slider("Seconds per window", 0.0, 1.0, 0.15, 0.05)

    if st.button("Start live replay", type="primary", icon=":material/play_arrow:"):
        df = read_flows(names[choice])
        res = score_flows(df, pre, model, iso, meta, use_ensemble)
        if res is None:
            st.html(banner("alert", "⚠️", "Stream too short for one window."))
            st.stop()
        score, truth, first_rows = res
        status, chart = st.empty(), st.empty()
        alerts = []
        for i in range(len(score)):
            chart.altair_chart(score_chart(score[: i + 1], thr, len(score), f"{det_name} · anomaly score per window"),
                               width="stretch")
            truth_txt = f" · ground truth: <b>{truth[i]}</b>" if truth is not None else ""
            if score[i] > thr:
                conf = float(confidence(score[i], det_meta))
                status.html(banner("alert", "🚨", f"<b>ALERT</b> · window {i} · score {score[i]:.3f} (line {thr:.3f}) · "
                                                  f"more unusual than {conf:.1%} of normal traffic{truth_txt}"))
                alerts.append(i)
            else:
                status.html(banner("ok", "✅", f"<b>Normal</b> · window {i}{truth_txt}"))
            time.sleep(delay)

        tbl = window_table(score, truth, first_rows, thr, det_meta)
        section("Summary", f"{len(alerts)} of {len(score)} windows flagged")
        if truth is not None:
            is_att = truth != BENIGN
            flagged = score > thr
            start = f"attack begins ≈ window {int(np.argmax(is_att))}" if is_att.any() else "no attack in this stream"
            st.html(stats([
                ("Attack windows caught", f"{flagged[is_att].sum()} / {is_att.sum()}", C["teal"], "windows containing attack flows"),
                ("False alarms", f"{flagged[~is_att].sum()} / {(~is_att).sum()}", C["red"], "normal windows flagged"),
                ("First alert at window", alerts[0] if alerts else "—", C["amber"], start),
            ]))
        st.dataframe(tbl[tbl["score"] > thr].round(4), hide_index=True, column_config=TABLE_CFG)

# -------------------------------------------------------------- upload mode
else:
    section("Analyse a file", "Upload traffic",
            "Any CICIDS2017-format CSV: flow features, one row per flow, time-ordered. A Label column is optional.")
    up = st.file_uploader("CICIDS2017-format CSV", type="csv")
    if up is not None:
        df = read_flows(up)
        try:
            res = score_flows(df, pre, model, iso, meta, use_ensemble)
        except KeyError as e:
            st.html(banner("alert", "⚠️", str(e).strip("'")))
            st.stop()
        if res is None:
            st.html(banner("alert", "⚠️", f"Need at least {W} flows."))
            st.stop()
        score, truth, first_rows = res
        flagged = score > thr
        st.html(stats([
            ("Flows analysed", f"{len(df):,}", C["text"], None),
            ("Windows flagged", f"{flagged.sum()} / {len(score)}", C["red"], None),
            ("Share flagged", f"{flagged.mean():.1%}", C["amber"], None),
        ]))
        st.altair_chart(score_chart(score, thr, len(score), f"{det_name} · anomaly score per window", truth=truth),
                        width="stretch")
        tbl = window_table(score, truth, first_rows, thr, det_meta)
        section("Flagged windows", "Most suspicious first")
        st.dataframe(tbl[flagged].sort_values("score", ascending=False).round(4), hide_index=True, column_config=TABLE_CFG)
        if truth is not None:
            st.caption("Ground truth found in the `Label` column: "
                       f"{int((truth != BENIGN).sum())} attack windows in this file.")
    else:
        st.html(banner("info", "ℹ️", "Upload a file, or switch to <b>Simulate an attack</b> in the sidebar for the demo."))
