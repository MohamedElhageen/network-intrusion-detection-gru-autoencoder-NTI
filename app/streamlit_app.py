"""Network intrusion detector - Streamlit app.

Run from the repo root:   streamlit run app/streamlit_app.py

Two pages:
  live_detector.py  - replay a recorded attack or upload a traffic CSV
  inside_model.py   - look inside the GRU autoencoder and the ensemble for any window
"""
import streamlit as st

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="wide")

from common import apply_style  # noqa: E402  (after set_page_config)

apply_style()
nav = st.navigation([
    st.Page("live_detector.py", title="Live Detector", icon=":material/shield:", default=True),
    st.Page("inside_model.py", title="Inside the Model", icon=":material/biotech:"),
])
nav.run()
