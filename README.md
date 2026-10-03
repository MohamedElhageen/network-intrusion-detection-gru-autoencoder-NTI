# Network Intrusion Detection with a GRU Autoencoder (CICIDS2017)

Time-series anomaly detection for network traffic. A GRU autoencoder learns what **normal** traffic looks like; a window of consecutive flows it cannot reconstruct is flagged as an attack. Compared against an Isolation Forest baseline, combined with it in a score ensemble, and served in a Streamlit app with a live attack replay.

> Machine Learning Capstone · team of 4 · final discussion 04-10-2026

## Why sequences instead of single flows
One flow (a single SYN packet, one short connection) often looks harmless. Attacks show up as **patterns over consecutive flows**: hundreds of identical short connections (port scan), floods of requests (DoS / DDoS), repeated logins (brute force). The model therefore scores windows of 10 time-ordered flows.

## Pipeline
```
raw CSVs ──► clean (inf/NaN, duplicates, constants, timestamps)
         ──► drop identifiers (Source IP, Flow ID ... would leak the label)
         ──► signed log1p ─► z-score (fit on benign Monday only) ─► clip ±10
         ──► sliding windows of 10 consecutive flows
         ──► GRU autoencoder (trained on benign windows) ─► reconstruction error
         ──► + Isolation Forest score  ─► standardised, averaged  (ensemble)
         ──► score > threshold  ⇒  ALERT
```

**Evaluation design (no leakage):** Monday (benign) = train · Tuesday + Wednesday = validation (hyper-parameters, alert threshold) · Thursday + Friday = test, containing web attacks, infiltration, botnet, port scan and DDoS that the model never saw.

## Results
*Filled in after running `python -m src.train` (see `results/metrics_test.csv` and `results/per_attack_*.csv`).*


|Model	            |FPR (all benign)	|FPR (excl. attacker-linked)	|false alarms linked to attacker |F1 (all)	|F1 (excl.)	|ROC-AUC (excl.)	|PR-AUC (excl.)|
|isolation_forest   |       0.162       |               0.130           |              0.240	         |   0.648	|   0.680	|      0.891	    |     0.705    |
|gru_ae	            |       0.149       |               0.115           |              0.271	         |   0.771	|   0.809	|      0.889	    |     0.611    |
|ensemble	        |       0.162       |               0.128           |              0.248	         |   0.771	|   0.807	|      0.902	    |     0.666    |


## Repository layout
```
app/streamlit_app.py     Streamlit demo (upload CSV  |  live attack replay)
notebooks/               01_eda_preprocessing.ipynb, 02_modeling.ipynb
src/                     config, data_loader, preprocessing, sequences, pipeline,
                         baseline, models, ensemble, evaluate, experiment, make_demo, train, calibrate
tests/                   pytest suite (runs on small synthetic CICIDS-style data)
data/raw/                put the 8 CICIDS2017 CSVs here (not committed)
data/demo/               replayable attack streams used by the app
models/                  trained artifacts used by the app
results/                 metric tables
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Download **CICIDS2017 "GeneratedLabelledFlows"** (`TrafficLabelling/*.csv`, 8 files) from the Canadian Institute for Cybersecurity and place the CSVs in `data/raw/`.

## Usage
**1. Train everything (≈ one command)**
```bash
python -m src.train              # add --skip-gru for the baseline only, --epochs N to shorten
```
or step through `notebooks/01_eda_preprocessing.ipynb` then `02_modeling.ipynb` (same code, with plots). Both write `models/`, `results/` and `data/demo/`.

**Change the alert threshold without retraining**
```bash
python -m src.calibrate --fpr 0.05   # ~5 % false alarms on normal validation traffic; omit --fpr to just print the trade-off table
```

**2. Run the app**
```bash
streamlit run app/streamlit_app.py
```
* **Simulate an attack (live replay):** pick an attack family; real traffic is replayed window by window, the error curve crosses the threshold and an alert banner appears.
* **Upload traffic CSV:** any CICIDS2017-format file (time-ordered flows). Shows flagged windows with an anomaly score and a confidence value (share of normal validation windows that scored lower).

**3. Tests**
```bash
pytest -q
```

## Data-quality issues found and fixed
* Timestamps are **day-first**, mix `H:MM` and `H:MM:SS`, and use a **12-hour clock without AM/PM** (3:30 means 15:30). Naive parsing silently dropped ~529 k rows and spread one week over four months. See `parse_timestamps`.
* `Source IP` `172.16.0.1` sent ~558 k flows and 99 % of them are attacks, so identifier columns (Source/Destination IP, Flow ID) are excluded from the features.
* Labels contain a stray `\x96` character (`Web Attack – Brute Force`), normalised on load.
* Infinite values, NaNs, ~9 % duplicate rows and constant columns removed.

## Limitations and future work
* Zero-day coverage is partial: attacks that look statistically like normal traffic (slow, low-volume, e.g. Infiltration, Heartbleed) are the hardest; they also have very few samples.
* The threshold is tuned on one day of traffic; in production it would need recalibration as normal behaviour drifts.
* Windows are built over the global flow stream; per-host or per-connection windows and an attention/Transformer encoder are natural next steps.
* CICIDS2017 is a lab capture; real networks will need retraining on their own benign traffic.

## Team
| Role | Member |
|---|---|
| Data & EDA | |
| Modeling | |
| Deployment & code quality | |
| Presentation & coordination | |
