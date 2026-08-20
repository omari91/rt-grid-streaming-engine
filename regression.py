"""
Standalone regression entry point, matching README's documented usage
(`python regression.py`) and requirements.txt's scikit-learn dependency.

This reproduces the exact E2 state-aware regression already computed by
run_e2_recall_audit.py (same data pipeline, same seed, same 70/30 split),
cross-checked here with scikit-learn's LinearRegression instead of raw
numpy.linalg.lstsq, as an independent verification that the two
implementations agree. It additionally persists the fitted coefficients
themselves to final_output/regression_coefficients.csv -- previously only
printed to stdout by run_e2_recall_audit.py, never saved as a file, which
is the concrete artifact this script was missing before now.

Usage
-----
    python regression.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from engine import (
    GridSimulator,
    LocalCsvIngestionLayer,
    SyntheticLoadProvider,
    OUTPUT_DIR,
    RUN_SEED,
    VOLTAGE_MIN_PU,
)


def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    provider = SyntheticLoadProvider(simulator.load_multipliers)

    print("E2 (default synthetic operating-state) regression, scikit-learn cross-check")

    cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)
    critical = cycle_df[cycle_df["critical_event"] == True]
    solved = critical.dropna(subset=["raw_vm_ref_pu"]).copy()
    solved["multiplier"] = solved["event_idx"].apply(lambda i: provider.get_multiplier(int(i)))

    # Same 70/30 split, same seed as run_e2_recall_audit.py -- this is a
    # cross-check of the fitting method, not a different experiment.
    rng = np.random.default_rng(RUN_SEED)
    idx = rng.permutation(len(solved))
    n_train = int(0.7 * len(solved))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train, test = solved.iloc[train_idx], solved.iloc[test_idx]

    X_train = train[["multiplier", "load_mw"]].to_numpy()
    y_train = train["raw_vm_ref_pu"].to_numpy()
    X_test = test[["multiplier", "load_mw"]].to_numpy()
    y_test = test["raw_vm_ref_pu"].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)
    b_mult, b_load = model.coef_
    intercept = model.intercept_

    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)

    print(f"Regression: V ~ {intercept:.4f} + {b_mult:.4f}*multiplier + {b_load:.4f}*P_load")
    print(f"Train R^2 = {r2_train:.4f} (n_train={len(train)})")
    print(f"Held-out R^2 = {r2_test:.4f} (n_test={len(test)})")

    r_mult = np.corrcoef(solved["multiplier"], solved["raw_vm_ref_pu"])[0, 1]
    r_load = np.corrcoef(solved["load_mw"], solved["raw_vm_ref_pu"])[0, 1]
    print(f"corr(voltage, multiplier) = {r_mult:.4f}")
    print(f"corr(voltage, load_mw)    = {r_load:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = pd.DataFrame([{
        "experiment": "E2",
        "intercept": intercept,
        "coef_multiplier": b_mult,
        "coef_load_mw": b_load,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "n_train": len(train),
        "n_test": len(test),
        "corr_voltage_multiplier": r_mult,
        "corr_voltage_load_mw": r_load,
        "fitting_library": "scikit-learn==1.9.0",
        "cross_check_of": "run_e2_recall_audit.py (numpy.linalg.lstsq)",
    }])
    out_path = os.path.join(OUTPUT_DIR, "regression_coefficients.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
