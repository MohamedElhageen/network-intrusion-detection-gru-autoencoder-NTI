"""Re-pick the alert threshold WITHOUT retraining.

    python -m src.calibrate                       # print the precision/recall trade-off
    python -m src.calibrate --fpr 0.05            # ... and make ~5 % false alarms the app default
    python -m src.calibrate --detector gru        # calibrate the GRU alone instead of the ensemble
"""
import argparse
import json

import joblib
import numpy as np

from .baseline import window_scores
from .config import MODEL_DIR, RESULTS_DIR
from .ensemble import ensemble_score
from .evaluate import operating_points
from .models import load_model, reconstruction_errors
from .pipeline import load_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, help="false-alarm budget on normal validation windows, e.g. 0.05")
    ap.add_argument("--detector", choices=["gru", "ensemble"], default=None)
    args = ap.parse_args()

    meta_path = MODEL_DIR / "threshold.json"
    meta = json.loads(meta_path.read_text())
    det = args.detector or ("ensemble" if "ensemble" in meta else "gru")

    splits = load_splits()
    gru = load_model(MODEL_DIR / "gru_ae.pt")
    s = {part: reconstruction_errors(gru, splits[f"X_{part}"]) for part in ("val", "test")}
    if det == "ensemble":
        iso = joblib.load(MODEL_DIR / "isolation_forest.joblib")
        for part in ("val", "test"):
            s[part] = ensemble_score({"gru_ae": s[part], "isolation_forest": window_scores(iso, splits[f"X_{part}"])},
                                     meta["ensemble"]["stats"])

    table = operating_points(splits["y_val"], s["val"], splits["y_test"], s["test"])
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / f"operating_points_{det}.csv", index=False)
    print(f"detector: {det}")
    print(table.round(4).to_string(index=False))

    if args.fpr is not None:
        thr = float(np.quantile(s["val"][splits["y_val"] == 0], 1 - args.fpr))
        target = meta["ensemble"] if det == "ensemble" else meta
        target.update({"threshold": thr})
        meta["default_rule"] = f"fpr_{args.fpr}"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\n{det} threshold set to {thr:.4f} (target false-alarm rate {args.fpr:.0%}).")


if __name__ == "__main__":
    main()
