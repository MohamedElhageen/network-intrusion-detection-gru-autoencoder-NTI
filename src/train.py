"""End-to-end training:  python -m src.train [--skip-gru] [--epochs 30]

Reads data/processed/clean.parquet (or data/raw/*.csv), builds windows, trains the
Isolation Forest baseline and the GRU autoencoder, evaluates on the unseen test days,
and writes models/ + results/.
"""
import argparse

from .config import EPOCHS, RESULTS_DIR
from .data_loader import load_clean
from .evaluate import per_attack_table
from .experiment import export_artifacts, run_experiment
from .make_demo import build_demo_files
from .pipeline import build_splits, fit_preprocessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gru", action="store_true", help="only run the Isolation Forest baseline")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    df = load_clean()
    print(f"{len(df):,} clean flows loaded")
    pre = fit_preprocessor(df)
    print(f"{len(pre.features_)} features kept ({len(pre.dropped_corr_)} dropped as redundant)")
    splits = build_splits(df, pre)
    for s in ("train", "val", "test"):
        print(f"{s:5s}: {len(splits['X_' + s]):>8,} windows, {splits['y_' + s].mean():.1%} attack")

    res = run_experiment(splits, skip_gru=args.skip_gru, epochs=args.epochs)
    RESULTS_DIR.mkdir(exist_ok=True)
    res["table"].to_csv(RESULTS_DIR / "metrics_test.csv", index=False)
    print("\nTEST RESULTS\n", res["table"].round(4).to_string(index=False))

    for name, s in res["scores"]["test"].items():
        thr = res["thresholds"][name]["val_f1"]
        pa = per_attack_table(splits["t_test"], splits["classes"], s, thr)
        pa.to_csv(RESULTS_DIR / f"per_attack_{name}.csv", index=False)
        print(f"\nPer-class flagged % - {name}\n", pa.to_string(index=False))

    export_artifacts(pre, res, splits)
    build_demo_files(df)
    print("\nArtifacts written to models/, results/ and data/demo/")


if __name__ == "__main__":
    main()
