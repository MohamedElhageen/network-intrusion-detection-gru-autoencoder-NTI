"""Tiny fake CICIDS2017-style data (same column names and timestamp quirks) for tests."""
import numpy as np
import pandas as pd

DAYS = {"Monday": 3, "Tuesday": 4, "Wednesday": 5, "Thursday": 6, "Friday": 7}
ATTACKS = {"Tuesday": "FTP-Patator", "Wednesday": "DoS Hulk",
           "Thursday": "Web Attack \x96 Brute Force", "Friday": "PortScan"}


def _stamp(day_num, sec_of_day, with_seconds):
    h, rem = divmod(sec_of_day, 3600)
    m, s = divmod(rem, 60)
    h12 = h - 12 if h > 12 else h                       # 12-hour clock, no AM/PM (as in the real files)
    return f"{day_num}/7/2017 {h12}:{m:02d}" + (f":{s:02d}" if with_seconds else "")


def make_day(day, n=4000, seed=0):
    rng = np.random.default_rng(seed + DAYS[day])
    f = rng.lognormal(mean=[8, 1.5, 1.2, 6, 5, 4, 3], sigma=1.0, size=(n, 7))
    df = pd.DataFrame(f, columns=["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                                  "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean",
                                  "Bwd Packet Length Std"])
    df["Subflow Fwd Packets"] = df["Total Fwd Packets"]              # exact duplicate -> must be pruned
    df["Avg Fwd Segment Size"] = df["Fwd Packet Length Mean"]        # exact duplicate -> must be pruned
    df["Destination Port"] = rng.choice([80, 443, 53, 22], n).astype(float)
    df["Protocol"] = 6.0
    df["SYN Flag Count"] = rng.integers(0, 2, n).astype(float)
    df["Constant"] = 1.0                                             # constant column -> dropped
    label = np.array(["BENIGN"] * n, dtype=object)
    if day in ATTACKS:
        a, b = int(n * 0.4), int(n * 0.6)                            # one attack burst in the middle
        df.loc[a:b, "Flow Duration"] *= 0.001
        df.loc[a:b, "Flow Packets/s"] *= 50
        df.loc[a:b, "Total Fwd Packets"] = 1.0
        label[a:b + 1] = ATTACKS[day]
    times = np.sort(rng.integers(8 * 3600, 17 * 3600, n))
    df.insert(0, "Flow ID", [f"10.0.0.{i % 250}-1.1.1.1-{i}-80-6" for i in range(n)])
    df.insert(1, "Source IP", "172.16.0.1")
    df.insert(2, "Source Port", 1234.0)
    df.insert(3, "Destination IP", "192.168.10.50")
    df["Timestamp"] = [_stamp(DAYS[day], int(t), i % 3 == 0) for i, t in enumerate(times)]
    df["Label"] = label
    df.columns = [" " + c if c in ("Flow Duration", "Label") else c for c in df.columns]   # leading spaces like the real files
    return df


def make_raw_frame(n=4000):
    return pd.concat([make_day(d, n) for d in DAYS], ignore_index=True)
