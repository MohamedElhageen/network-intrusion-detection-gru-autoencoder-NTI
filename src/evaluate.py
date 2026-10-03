"""Metrics suited to imbalanced intrusion detection (accuracy alone is misleading)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def percentile_threshold(benign_scores: np.ndarray, q: float = 99.0) -> float:
    """Unsupervised rule: flag anything above the q-th percentile of benign scores."""
    return float(np.percentile(benign_scores, q))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold that maximises F1 on a labelled validation set."""
    if y.sum() == 0 or y.sum() == len(y):
        return percentile_threshold(s, 99.0)
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def metrics_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def results_table(y: np.ndarray, scores: dict, thresholds: dict) -> pd.DataFrame:
    """One row per (model, threshold rule). scores[model] -> array, thresholds[model] -> {rule: value}."""
    rows = []
    for model, s in scores.items():
        roc, pr = roc_auc_score(y, s), average_precision_score(y, s)
        for rule, thr in thresholds[model].items():
            rows.append({"model": model, "threshold_rule": rule, "threshold": thr,
                         "roc_auc": roc, "pr_auc": pr, **metrics_at_threshold(y, s, thr)})
    return pd.DataFrame(rows)


def per_attack_table(types: np.ndarray, classes, s: np.ndarray, thr: float) -> pd.DataFrame:
    """Detection rate per attack family (for BENIGN the same number is the false-positive rate)."""
    rows = []
    for k, name in enumerate(classes):
        m = types == k
        if m.sum() == 0:
            continue
        rows.append({"class": name, "windows": int(m.sum()),
                     "flagged_%": round(100 * float((s[m] > thr).mean()), 2)})
    return pd.DataFrame(rows).sort_values("flagged_%", ascending=False).reset_index(drop=True)


def operating_points(y_val, s_val, y_test, s_test, fprs=(0.01, 0.02, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Precision/recall trade-off. Threshold = benign-validation quantile for each false-alarm budget,
    then measured on the test set. This is the table to pick the alert threshold from."""
    benign = s_val[y_val == 0]
    rows = []
    for f in fprs:
        thr = float(np.quantile(benign, 1 - f))
        rows.append({"target_fpr": f, "threshold": thr, **metrics_at_threshold(y_test, s_test, thr)})
    return pd.DataFrame(rows)
