"""Page 2 - Inside the Model: how the GRU rebuilds a window and how the ensemble reaches its decision."""
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from common import (BENIGN, C, LABEL_COL, banner, chart_style, demo_files, hero, iso_threshold, load_artifacts,
                    read_flows, score_chart, section, stats, windows_of)
from src.ensemble import EPS

hero("Explainability", "🔬 Inside the Model",
     "Pick any window of 10 flows and look inside: how the GRU autoencoder rebuilds it, where the rebuild fails, "
     "and how the ensemble turns two opinions into one decision.",
     chips=("Reconstruction", "Feature errors", "Ensemble breakdown"))

try:
    pre, model, iso, meta = load_artifacts()
except FileNotFoundError:
    st.html(banner("alert", "⚠️", "Model files not found in <b>models/</b>. Run <b>python -m src.train</b> first (see README)."))
    st.stop()


@st.cache_data(show_spinner="Running the models on this traffic…")
def analyse(content: bytes):
    """Score every non-overlapping window and keep everything needed to explain it."""
    pre, model, iso, meta = load_artifacts()
    df = read_flows(io.BytesIO(content))
    idx = windows_of(df, meta)
    if len(idx) == 0:
        return None
    X = pre.transform(df)[idx]                                   # (n, 10, 37) scaled flows
    with torch.no_grad():
        rec = model(torch.from_numpy(X)).numpy()                 # rebuilt windows
    sq = (rec - X) ** 2
    out = {"X": X, "rec": rec, "sq": sq, "gru": sq.mean(axis=(1, 2)), "first_rows": idx[:, 0]}
    if iso is not None:
        n, w, f = X.shape
        out["iso_flow"] = -iso.score_samples(X.reshape(-1, f)).reshape(n, w)
        out["iso"] = out["iso_flow"].mean(axis=1)
    out["labels"] = df[LABEL_COL].to_numpy()[idx] if LABEL_COL in df.columns else None
    return out


# ------------------------------------------------------------ data source
with st.sidebar:
    st.header("Traffic to explain")
    source = st.radio("Source", ["Demo scenario", "Upload CSV"])
    content, name = None, None
    if source == "Demo scenario":
        names = demo_files()
        if names:
            options = list(names)
            name = st.selectbox("Scenario", options, index=options.index("DDoS") if "DDoS" in options else 0)
            content = names[name].read_bytes()
    else:
        up = st.file_uploader("CICIDS2017-format CSV", type="csv")
        if up is not None:
            name, content = up.name, up.getvalue()
    st.caption("Scores use the saved models and their default alarm lines (95th percentile of normal practice-day traffic).")

if content is None:
    st.html(banner("info", "ℹ️", "Choose a demo scenario or upload a file in the sidebar."))
    st.stop()
try:
    A = analyse(content)
except KeyError as e:
    st.html(banner("alert", "⚠️", str(e).strip("'")))
    st.stop()
if A is None:
    st.html(banner("alert", "⚠️", f"Need at least {meta['window']} flows."))
    st.stop()

features = list(pre.features_)
n = len(A["gru"])
gru_thr = float(meta["threshold"])
normal_med = float(meta["benign_quantiles"][len(meta["benign_quantiles"]) // 2])
labels = A["labels"]
truth = (np.array([next((x for x in row if x != BENIGN), BENIGN) for row in labels]) if labels is not None else None)

# ------------------------------------------------------------ step 1: pick a window
section("Step 1", "Choose a window",
        "Each point is one window of 10 consecutive flows. Red bands mark windows that really contain attack flows; "
        "the dashed line is the GRU's alarm line.")

key = f"win::{name}"
attack_ws = np.flatnonzero(truth != BENIGN) if truth is not None else np.array([], dtype=int)
normal_ws = np.flatnonzero(truth == BENIGN) if truth is not None else np.arange(n)
picks = {
    "First attack window": int(attack_ws[0]) if len(attack_ws) else None,
    "Highest GRU error": int(np.argmax(A["gru"])),
    "Typical normal window": int(normal_ws[np.argmin(np.abs(A["gru"][normal_ws] - normal_med))]) if len(normal_ws) else None,
}
if key not in st.session_state:
    st.session_state[key] = picks["First attack window"] if picks["First attack window"] is not None else picks["Highest GRU error"]


def _set(v):
    st.session_state[key] = v


cols = st.columns(len(picks) + 1)
for col, (label, v) in zip(cols, picks.items()):
    col.button(label, on_click=_set, args=(v,), disabled=v is None, width="stretch")
win = st.slider("Window", 0, n - 1, key=key)

st.altair_chart(score_chart(A["gru"], gru_thr, n, "GRU reconstruction error per window", truth=truth, selected=win),
                width="stretch")

err = float(A["gru"][win])
row_labels = labels[win] if labels is not None else None
n_att = int(sum(x != BENIGN for x in row_labels)) if row_labels is not None else None
gt = truth[win] if truth is not None else "unknown"
st.html(stats([
    ("Window", f"{win}", C["text"], f"flows {A['first_rows'][win]}–{A['first_rows'][win] + meta['window'] - 1} of the file"),
    ("GRU error", f"{err:.3f}", C["red"] if err > gru_thr else C["teal"], f"alarm line {gru_thr:.3f}"),
    ("vs typical normal", f"{err / normal_med:.1f}×", C["amber"], f"normal median {normal_med:.3f}"),
    ("Ground truth", gt, C["red"] if gt not in (BENIGN, "unknown") else C["teal"],
     f"{n_att} of {meta['window']} flows are attacks" if n_att is not None else "no Label column"),
]))

# ------------------------------------------------------------ step 2: inside the GRU
section("Step 2 · Inside the GRU", "Real window vs rebuilt window",
        "Every cell is one feature of one flow, in standard deviations from normal Monday traffic. The autoencoder "
        "squeezes the real window through 64 numbers and rebuilds it; the right-hand map shows where the rebuild failed.")

real, rec, sq = A["X"][win], A["rec"][win], A["sq"][win]                 # each (10, 37)
flow_ids = [str(i + 1) for i in range(real.shape[0])]


def heatmap(mat, title, scale, legend_title, show_labels):
    d = pd.DataFrame(mat.T, index=features, columns=flow_ids).reset_index().melt(
        id_vars="index", var_name="flow", value_name="value").rename(columns={"index": "feature"})
    ch = alt.Chart(d).mark_rect(stroke=C["bg"], strokeWidth=0.6).encode(
        x=alt.X("flow:O", sort=flow_ids, title="flow in window", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("feature:N", sort=features, title=None,
                axis=alt.Axis(labelLimit=210, labelOverlap=False, labelFontSize=10) if show_labels else None),
        color=alt.Color("value:Q", scale=scale, legend=alt.Legend(title=legend_title, gradientLength=150)),
        tooltip=["feature", "flow", alt.Tooltip("value:Q", format=".3f")])
    return chart_style(ch.properties(title=title), height=len(features) * 17)


div_scale = alt.Scale(domain=[-4, 0, 4], range=[C["steel"], C["panel"], C["amber"]], clamp=True, interpolate="rgb")
err_scale = alt.Scale(domain=[0, 1, 2], range=[C["panel"], C["amber"], C["red"]], clamp=True, interpolate="rgb")
h1, h2, h3 = st.columns([1.55, 1, 1])
h1.altair_chart(heatmap(real, "Real window", div_scale, "scaled value", True), width="stretch")
h2.altair_chart(heatmap(rec, "Rebuilt by the GRU", div_scale, "scaled value", False), width="stretch")
h3.altair_chart(heatmap(sq, "Squared error", err_scale, "error (capped at 2)", False), width="stretch")

share = sq.sum(axis=0) / sq.sum()
top = pd.DataFrame({"feature": features, "share": share}).nlargest(8, "share")
flow_err = pd.DataFrame({"flow": flow_ids, "error": sq.mean(axis=1),
                         "label": list(row_labels) if row_labels is not None else ["unknown"] * len(flow_ids)})
flow_err["kind"] = np.where(flow_err["label"].isin([BENIGN, "unknown"]), "normal flow", "attack flow")

b1, b2 = st.columns(2)
feat_chart = (alt.Chart(top).mark_bar(color=C["amber"], cornerRadiusEnd=4).encode(
                  y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                  x=alt.X("share:Q", title="share of this window's error", axis=alt.Axis(format="%")),
                  tooltip=["feature", alt.Tooltip("share:Q", format=".1%")])
              + alt.Chart(top).mark_text(align="left", dx=5, color=C["text"], fontSize=11).encode(
                  y=alt.Y("feature:N", sort="-x"), x="share:Q", text=alt.Text("share:Q", format=".0%")))
b1.altair_chart(chart_style(feat_chart.properties(title="Which features failed to rebuild"), height=290), width="stretch")
flow_chart = alt.Chart(flow_err).mark_bar(cornerRadiusEnd=4).encode(
    x=alt.X("flow:O", sort=flow_ids, title="flow in window", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("error:Q", title="mean squared error"),
    color=alt.Color("kind:N", scale=alt.Scale(domain=["normal flow", "attack flow"], range=[C["steel"], C["red"]]),
                    legend=alt.Legend(title=None)),
    tooltip=["flow", "label", alt.Tooltip("error:Q", format=".3f")])
b2.altair_chart(chart_style(flow_chart.properties(title="Which flows failed to rebuild"), height=290), width="stretch")

f1, f2, f3 = top["feature"].iloc[0], top["feature"].iloc[1], top["feature"].iloc[2]
verdict = ("is <b>above</b> the GRU's alarm line" if err > gru_thr else "stays <b>below</b> the GRU's alarm line")
mix = (f" {n_att} of the 10 flows are attack flows ({gt})." if n_att else
       " All 10 flows are normal." if n_att == 0 else "")
st.html(f'<div class="note">This window\'s error is <b>{err:.3f}</b>, {err / normal_med:.1f}× a typical normal window, '
        f'and {verdict} ({gru_thr:.3f}).{mix} The rebuild failed most on <b>{f1}</b> '
        f'({top["share"].iloc[0]:.0%} of the error), <b>{f2}</b> and <b>{f3}</b>.</div>')

# ------------------------------------------------------------ step 3: the ensemble
section("Step 3 · The ensemble", "Two opinions, one decision",
        "Both scores are logged, standardised against normal practice-day traffic (so 0 = typical normal), "
        "and averaged. The alert fires when the average crosses the ensemble's alarm line.")

if iso is None or "ensemble" not in meta:
    st.html(banner("info", "ℹ️", "No ensemble was trained with these models."))
    st.stop()

(mg, sg), (mi, si) = meta["ensemble"]["stats"]["gru_ae"], meta["ensemble"]["stats"]["isolation_forest"]
g, i_ = err, float(A["iso"][win])
lg, li = np.log(g + EPS), np.log(i_ + EPS)
zg, zi = (lg - mg) / sg, (li - mi) / si
ens = (zg + zi) / 2
ens_thr = float(meta["ensemble"]["threshold"])
if_thr = iso_threshold(meta.get("default_rule", "pct95"))

st.html(f"""
<table class="eq">
  <tr><th>Detector</th><th class="num">Raw score</th><th class="num">log(score)</th><th class="num">Normal mean</th>
      <th class="num">Normal std</th><th class="num">z = (log − mean) / std</th></tr>
  <tr><td>GRU autoencoder</td><td class="num">{g:.4f}</td><td class="num">{lg:.3f}</td><td class="num">{mg:.3f}</td>
      <td class="num">{sg:.3f}</td><td class="num">{zg:+.2f}</td></tr>
  <tr><td>Isolation Forest</td><td class="num">{i_:.4f}</td><td class="num">{li:.3f}</td><td class="num">{mi:.3f}</td>
      <td class="num">{si:.3f}</td><td class="num">{zi:+.2f}</td></tr>
  <tr><td>Ensemble = average of the two z</td><td></td><td></td><td></td><td></td><td class="num">{ens:+.2f}</td></tr>
</table>""")


def verdict_card(title, score, thr, fmt):
    if thr is None:
        return (f'<div class="verdict"><div class="v-name">{title}</div><div class="v-res" style="color:{C["muted"]}">n/a</div>'
                f'<div class="v-detail">no saved alarm line</div></div>')
    alert = score > thr
    col, res = (C["red"], "🚨 Alert") if alert else (C["teal"], "✅ Normal")
    return (f'<div class="verdict"><div class="v-name">{title}</div><div class="v-res" style="color:{col}">{res}</div>'
            f'<div class="v-detail">{score:{fmt}} vs alarm line {thr:{fmt}}</div></div>')


g_alert, e_alert = g > gru_thr, ens > ens_thr
i_alert = if_thr is not None and i_ > if_thr
if e_alert and g_alert and i_alert:
    why = "Both detectors find this window unusual, so the ensemble alerts with confidence."
elif e_alert and not g_alert:
    why = ("The GRU alone would stay quiet, but the Isolation Forest's second opinion pushes the combined score "
           "over the line. This is where the ensemble's extra recall comes from.")
elif e_alert:
    why = "The GRU's signal is strong enough to carry the combined score over the line."
elif i_alert and not g_alert:
    why = ("The Isolation Forest reacts to an individually odd flow, but the GRU says the sequence looks normal, "
           "so the ensemble stays quiet. This is how it avoids many of the forest's false alarms.")
elif g_alert:
    why = ("The GRU alone would alert, but the Isolation Forest finds these flows ordinary, pulling the combined "
           "score below the line.")
else:
    why = "Both detectors see ordinary traffic, so the ensemble stays quiet."

st.html(f'<div class="verdicts" style="margin-top:14px">{verdict_card("GRU alone", g, gru_thr, ".3f")}'
        f'{verdict_card("Isolation Forest alone", i_, if_thr, ".3f")}'
        f'{verdict_card("Ensemble (final decision)", ens, ens_thr, "+.2f")}</div>'
        f'<div class="note" style="margin-top:12px">{why}</div>')

zc1, zc2 = st.columns(2, gap="large")
zd = pd.DataFrame({"part": ["GRU (z)", "Isolation Forest (z)", "Ensemble"], "z": [zg, zi, ens],
                   "kind": ["gru", "iso", "ens"]})
lo, hi = min(-1.0, zd["z"].min()) - 0.5, max(ens_thr, zd["z"].max()) + 1.0
xz = alt.X("z:Q", title="standard deviations above typical normal traffic", scale=alt.Scale(domain=[lo, hi]))
yz = alt.Y("part:N", sort=list(zd["part"]), title=None, scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2),
           axis=alt.Axis(labelFontSize=12))
zchart = alt.layer(
    alt.Chart(zd).mark_bar(cornerRadiusEnd=4).encode(
        y=yz, x=xz,
        color=alt.Color("kind:N", scale=alt.Scale(domain=["gru", "iso", "ens"], range=[C["teal"], C["steel"], C["amber"]]),
                        legend=None),
        tooltip=["part", alt.Tooltip("z:Q", format="+.2f")]),
    alt.Chart(zd).mark_text(align="left", dx=6, color=C["text"], fontSize=12, fontWeight=600).encode(
        y=yz, x=xz, text=alt.Text("z:Q", format="+.2f")),
    alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=C["muted"], opacity=0.6).encode(x="x:Q"),
    alt.Chart(pd.DataFrame({"x": [ens_thr]})).mark_rule(color=C["red"], strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q"),
    alt.Chart(pd.DataFrame({"x": [ens_thr], "t": [f"alarm {ens_thr:.2f}"]})).mark_text(
        align="left", dx=6, color=C["red"], fontSize=11).encode(x="x:Q", y=alt.value(8), text="t:N"),
)
zc1.html('<div style="font-weight:600;font-size:14px;color:#E8EEF5;margin:6px 0 2px">Each opinion on one scale (0 = typical normal)</div>')
zc1.altair_chart(chart_style(zchart, height=260),
                 width="stretch")

iso_flow = pd.DataFrame({"flow": flow_ids, "score": A["iso_flow"][win], "label": flow_err["label"], "kind": flow_err["kind"]})
iso_lo = float(min(iso_flow["score"].min(), if_thr or 1, i_)) - 0.02
iso_hi = float(max(iso_flow["score"].max(), if_thr or 0, i_)) + 0.02
iso_flow["base"] = iso_lo
yi = alt.Y("score:Q", title="Isolation Forest score", scale=alt.Scale(domain=[iso_lo, iso_hi], nice=False))
iso_chart = alt.layer(
    alt.Chart(iso_flow).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("flow:O", sort=flow_ids, title="flow in window", axis=alt.Axis(labelAngle=0)),
        y=yi, y2="base:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["normal flow", "attack flow"], range=[C["steel"], C["red"]]),
                        legend=alt.Legend(title=None)),
        tooltip=["flow", "label", alt.Tooltip("score:Q", format=".3f")]),
    alt.Chart(pd.DataFrame({"y": [i_]})).mark_rule(color=C["teal"], strokeWidth=2.5).encode(y="y:Q"),
    *([alt.Chart(pd.DataFrame({"y": [if_thr]})).mark_rule(color=C["amber"], strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")]
      if if_thr is not None else []),
)
zc2.html('<div style="font-weight:600;font-size:14px;color:#E8EEF5;margin:6px 0 2px">Isolation Forest: each flow judged on its own</div>')
zc2.altair_chart(chart_style(iso_chart, height=260),
                 width="stretch")
zc2.caption("Teal line = the window's average (its Isolation Forest score). Dashed line = the forest's own alarm line.")
