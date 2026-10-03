import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baseline import fit_isolation_forest, window_scores          # noqa: E402
from src.data_loader import clean_raw, normalize_labels, parse_timestamps  # noqa: E402
from src.ensemble import ensemble_score, fit_ensemble              # noqa: E402
from src.evaluate import best_f1_threshold, per_attack_table, results_table  # noqa: E402
from src.pipeline import build_splits, fit_preprocessor               # noqa: E402
from src.sequences import make_windows, window_index                   # noqa: E402
from synthetic import make_raw_frame                                  # noqa: E402


def test_timestamp_parsing_handles_mixed_formats_and_pm():
    s = pd.Series(["7/7/2017 3:30", "7/7/2017 3:30:15", "3/7/2017 9:05", "3/7/2017 12:10"])
    ts = parse_timestamps(s)
    assert ts.notna().all()                                   # nothing becomes NaT
    assert (ts.dt.month == 7).all()                           # day-first, July not March/April
    assert ts.iloc[0].hour == 15 and ts.iloc[1].hour == 15    # 3:30 -> 15:30
    assert ts.iloc[2].hour == 9 and ts.iloc[3].hour == 12     # morning / noon untouched
    assert ts.iloc[0].day_name() == "Friday"


def test_iso_timestamps_are_left_alone():
    ts = parse_timestamps(pd.Series(["2017-07-07 15:30:00"]))
    assert ts.iloc[0].hour == 15


def test_label_normalisation():
    out = normalize_labels(pd.Series(["Web Attack \x96 Brute Force", "BENIGN", "FTP-Patator"]))
    assert list(out) == ["Web Attack - Brute Force", "BENIGN", "FTP-Patator"]


def test_make_windows_labels_and_shapes():
    X = np.arange(30, dtype=np.float32).reshape(15, 2)
    y = np.zeros(15, dtype=int); y[7] = 1
    t = np.zeros(15, dtype=int); t[7] = 2
    W, yw, tw = make_windows(X, y, t, window=5, stride=5)
    assert W.shape == (3, 5, 2) and list(yw) == [0, 1, 0] and list(tw) == [0, 2, 0]
    assert np.array_equal(W[1, 0], X[5])                      # windows are consecutive flows


def test_grouped_windows_never_mix_keys_and_are_time_ordered():
    keys = np.array(list("ABABABABABAAAB"))                    # rows are in time order
    idx = window_index(len(keys), window=3, stride=3, keys=keys)
    assert idx.shape[1] == 3
    for row in idx:
        assert len(set(keys[row])) == 1                        # one host per window
        assert (np.diff(row) > 0).all()                        # flows stay in time order
    assert (np.diff(idx[:, 0]) >= 0).all()                     # windows sorted by start time
    assert idx.shape[0] == 4                                   # 8 A-flows -> 2 windows, 6 B-flows -> 2 windows
    assert np.array_equal(window_index(7, 3, 3), [[0, 1, 2], [3, 4, 5]])   # keys=None = global stream


def test_ensemble_puts_detectors_on_one_scale_and_ranks_anomalies_higher():
    rng = np.random.default_rng(0)
    normal = {"gru": rng.lognormal(-2.5, 0.5, 5000), "iso": rng.lognormal(-0.7, 0.1, 5000)}   # very different scales
    stats = fit_ensemble(normal)
    test_norm = ensemble_score({"gru": normal["gru"][:500], "iso": normal["iso"][:500]}, stats)
    test_att = ensemble_score({"gru": np.full(500, 0.6), "iso": np.full(500, 0.75)}, stats)
    assert abs(test_norm.mean()) < 0.2                          # normal traffic is centred near 0
    assert test_att.min() > np.percentile(test_norm, 99)        # clearly separated


def test_end_to_end_baseline_beats_chance_and_no_leakage():
    df = clean_raw(make_raw_frame())
    assert set(df["day"]) == {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    assert "Constant" not in df.columns
    assert df["Label"].str.contains("\x96").sum() == 0

    pre = fit_preprocessor(df)
    assert "Subflow Fwd Packets" not in pre.features_          # redundant features pruned
    assert not {"Source IP", "Flow ID", "Timestamp"} & set(pre.features_)   # no label leakage

    sp = build_splits(df, pre)
    assert sp["y_train"].sum() == 0                            # training day is purely benign
    assert sp["y_test"].sum() > 0 and sp["X_test"].shape[1:] == (10, len(pre.features_))
    assert np.abs(sp["X_train"]).max() <= 10.0 + 1e-6          # clipped

    flows = sp["X_train"][::2].reshape(-1, len(pre.features_))
    iso = fit_isolation_forest(flows, n_estimators=50)
    s_val, s_test = window_scores(iso, sp["X_val"]), window_scores(iso, sp["X_test"])
    thr = best_f1_threshold(sp["y_val"], s_val)
    table = results_table(sp["y_test"], {"iso": s_test}, {"iso": {"val_f1": thr}})
    assert table.loc[0, "roc_auc"] > 0.9
    pa = per_attack_table(sp["t_test"], sp["classes"], s_test, thr)
    assert {"BENIGN", "PortScan"} <= set(pa["class"])          # Friday is a test day
    assert "DoS Hulk" not in set(pa["class"])                  # Wednesday is validation


def test_gru_autoencoder_shapes_and_training():
    import pytest
    pytest.importorskip("torch")
    from src.models import reconstruction_errors, train_autoencoder
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(size=(512, 10, 8)).astype("f4"), rng.normal(size=(64, 10, 8)).astype("f4")
    model, hist = train_autoencoder(Xtr, Xva, epochs=2, verbose=False)
    err = reconstruction_errors(model, Xva)
    assert err.shape == (64,) and len(hist) >= 1
    assert reconstruction_errors(model, Xva * 8).mean() > err.mean()   # anomalies reconstruct worse
