"""Train both models, pick thresholds, evaluate, and export deployment artifacts."""
from __future__ import annotations

import json

import joblib
import numpy as np

from .baseline import fit_isolation_forest, window_scores
from .config import EPOCHS, MODEL_DIR
from .ensemble import ensemble_score, fit_ensemble
from .evaluate import best_f1_threshold, percentile_threshold, results_table


def run_experiment(splits: dict, skip_gru: bool = False, epochs: int = EPOCHS, verbose: bool = True) -> dict:
    """Fit Isolation Forest (+ GRU autoencoder) on benign train windows and score val/test.

    Thresholds are chosen on the VALIDATION day only (never on test):
      * 'val_f1'  - maximises F1 on Tuesday (uses attack labels)
      * 'pct99' / 'pct95' - percentile of benign validation scores (label-free; ~1 % / ~5 % false alarms)
    """
    X_all = splits["X_train"][splits["y_train"] == 0]        # benign windows, chronological
    n_es, gap = max(1, int(0.1 * len(X_all))), 2
    X_fit, X_es = X_all[:-(n_es + gap)], X_all[-n_es:]        # last 10 % -> early stopping

    models, scores = {}, {"val": {}, "test": {}}
    history = None

    # ---- classical baseline (non-overlapping windows -> independent flows)
    step = max(1, int(splits["window"]) // int(splits["stride_train"]))
    flows = X_fit[::step].reshape(-1, X_fit.shape[2])
    if verbose:
        print(f"Isolation Forest on {len(flows):,} benign flows ...")
    models["isolation_forest"] = fit_isolation_forest(flows)
    for part in ("val", "test"):
        scores[part]["isolation_forest"] = window_scores(models["isolation_forest"], splits[f"X_{part}"])

    # ---- sequence model
    if not skip_gru:
        from .models import reconstruction_errors, train_autoencoder   # torch only when needed
        if verbose:
            print(f"GRU autoencoder on {len(X_fit):,} benign windows ...")
        models["gru_ae"], history = train_autoencoder(X_fit, X_es, epochs=epochs, verbose=verbose)
        for part in ("val", "test"):
            scores[part]["gru_ae"] = reconstruction_errors(models["gru_ae"], splits[f"X_{part}"])

    # ---- ensemble of both detectors (log-scores standardised on NORMAL validation windows)
    y_val, y_test = splits["y_val"], splits["y_test"]
    ensemble_stats = None
    if "gru_ae" in models:
        ensemble_stats = fit_ensemble({n: scores["val"][n][y_val == 0] for n in ("gru_ae", "isolation_forest")})
        for part in ("val", "test"):
            scores[part]["ensemble"] = ensemble_score({n: scores[part][n] for n in ensemble_stats}, ensemble_stats)

    # ---- thresholds from validation only, then evaluate on test
    thresholds = {}
    for name, s_val in scores["val"].items():
        thresholds[name] = {"val_f1": best_f1_threshold(y_val, s_val),
                            "pct99": percentile_threshold(s_val[y_val == 0], 99.0),
                            "pct95": percentile_threshold(s_val[y_val == 0], 95.0)}
    table = results_table(y_test, scores["test"], thresholds)
    return {"models": models, "scores": scores, "thresholds": thresholds,
            "history": history, "table": table, "ensemble_stats": ensemble_stats}


def export_artifacts(pre, result: dict, splits: dict, out_dir=MODEL_DIR, default_rule: str = "val_f1") -> None:
    """Write everything the Streamlit app needs into models/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pre.save(out_dir / "preprocessor.joblib")
    joblib.dump(result["models"]["isolation_forest"], out_dir / "isolation_forest.joblib")
    if "gru_ae" in result["models"]:
        from .models import save_model
        save_model(result["models"]["gru_ae"], out_dir / "gru_ae.pt")

        y_val = splits["y_val"]
        benign_val = result["scores"]["val"]["gru_ae"][y_val == 0]
        meta = {
            "model": "gru_ae",
            "window": int(splits["window"]),
            "group_by": str(splits.get("group_by", "")) or None,
            "n_features": int(splits["X_train"].shape[2]),
            "default_rule": default_rule,
            "thresholds": result["thresholds"]["gru_ae"],
            "threshold": result["thresholds"]["gru_ae"][default_rule],
            # empirical CDF of benign validation errors -> 'confidence' in the app
            "benign_quantiles": np.quantile(benign_val, np.linspace(0, 1, 1001)).tolist(),
        }
        if result.get("ensemble_stats"):
            ens_benign = result["scores"]["val"]["ensemble"][y_val == 0]
            meta["ensemble"] = {
                "stats": result["ensemble_stats"],
                "thresholds": result["thresholds"]["ensemble"],
                "threshold": result["thresholds"]["ensemble"][default_rule],
                "benign_quantiles": np.quantile(ens_benign, np.linspace(0, 1, 1001)).tolist(),
            }
        (out_dir / "threshold.json").write_text(json.dumps(meta, indent=2))


def quick_gru_val_score(splits: dict, hidden: int, latent: int, epochs: int = 8) -> float:
    """Hyper-parameter search helper: short GRU run -> PR-AUC on the VALIDATION day (never test)."""
    from sklearn.metrics import average_precision_score
    from .models import reconstruction_errors, train_autoencoder
    X = splits["X_train"][splits["y_train"] == 0]
    n_es = max(1, int(0.1 * len(X)))
    model, _ = train_autoencoder(X[:-(n_es + 2)], X[-n_es:], hidden=hidden, latent=latent,
                                 epochs=epochs, verbose=False)
    return float(average_precision_score(splits["y_val"], reconstruction_errors(model, splits["X_val"])))
