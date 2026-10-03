"""Central configuration: paths, split plan and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
DEMO_DIR = ROOT / "data" / "demo"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

LABEL_COL = "Label"
BENIGN = "BENIGN"

# Columns that identify a flow but must never be used as model inputs
# (Source IP / Flow ID would leak the label: 172.16.0.1 is the attacker).
NON_FEATURE_COLS = {
    "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Timestamp", LABEL_COL, "is_attack", "day",
}

# Split by capture day -> no temporal leakage, and test days contain attack
# families the model has never seen (a zero-day style evaluation).
TRAIN_DAYS = ["Monday"]                          # benign only
VAL_DAYS = ["Tuesday", "Wednesday"]              # Patator brute force + DoS family -> threshold / tuning
TEST_DAYS = ["Thursday", "Friday"]               # web attacks, infiltration, bot, port scan, DDoS (never seen)

# Sliding windows of consecutive flows (time-ordered)
WINDOW = 10
# None  -> windows over the global time-ordered flow stream
# "Destination IP" -> windows of 10 consecutive flows going to the SAME host (per-host behaviour).
# The column is only used to group flows; it is never a model feature.
WINDOW_GROUP_BY = None
STRIDE_TRAIN = 5     # overlapping windows for training
STRIDE_EVAL = 10     # non-overlapping windows for validation / test / app

# Preprocessing
CORR_THRESHOLD = 0.95
CLIP = 10.0          # clip z-scores to [-CLIP, CLIP]
MIN_STD = 0.1        # floor on the scaler std (keeps rare-flag features bounded)

# GRU autoencoder
HIDDEN = 256   # best row of the tuning table (val PR-AUC 0.75 vs 0.71 for 128/32)
LATENT = 64
EPOCHS = 40
BATCH = 256
LR = 1e-3
PATIENCE = 4
SEED = 42
