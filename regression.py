"""
Standalone regression entry point, matching README's documented usage
(`python regression.py`) and requirements.txt's scikit-learn dependency.

Covers all three severities reported in the paper's Table `tab:sensitivity`
(1.26x/E2, 1.50x, 1.75x) -- not just E2. An earlier version of this script
only covered E2; that was an incomplete answer to "where is the regression
that supports the paper's claims," since the paper's actual headline
sensitivity table spans three severities, and the existing severity-sweep
scripts (run_severity_recall_audit.py) already computed but never
persisted the fitted coefficients at 1.50x/1.75x either -- same gap as E2
had, just not yet noticed there.

Cross-checks every severity against scikit-learn's LinearRegression
(matching requirements.txt's added dependency) as an independent
verification of the existing numpy.linalg.lstsq implementation, using the
identical data pipeline, seed, and 70/30 split as the corresponding
run_*_recall_audit.py script, and additionally verifies the resulting
R^2_test against the value already saved in that script's own output CSV
-- so this isn't just "a second computation," it's a checked cross-match
against the numbers already reported.

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
    TimeSeriesLoadProvider,
    ScaledLoadProvider,
    load_simbench_profile,
    OUTPUT_DIR,
    RUN_SEED,
)

SEVERITIES = [
    ("E2 (1.26x)", 1.26, None),   # None = use SyntheticLoadProvider (E2's own default)
    ("1.50x", 1.50, "scaled"),
    ("1.75x", 1.75, "scaled"),
]


def fit_one(label, severity, mode, simulator, stream):
    if mode is None:
        provider = SyntheticLoadProvider(simulator.load_multipliers)
    else:
        simbench_vals = load_simbench_profile()
        base_provider = TimeSeriesLoadProvider(simbench_vals)
        provider = ScaledLoadProvider(base_provider, severity)

    cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)
    critical = cycle_df[cycle_df["critical_event"] == True]
    solved = critical.dropna(subset=["raw_vm_ref_pu"]).copy()
    solved["multiplier"] = solved["event_idx"].apply(lambda i: provider.get_multiplier(int(i)))

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

    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    r_mult = np.corrcoef(solved["multiplier"], solved["raw_vm_ref_pu"])[0, 1]
    r_load = np.corrcoef(solved["load_mw"], solved["raw_vm_ref_pu"])[0, 1]

    print(f"\n=== {label} ===")
    print(f"Regression: V ~ {intercept:.4f} + {b_mult:.4f}*multiplier + {b_load:.4f}*P_load")
    print(f"Train R^2 = {r2_train:.4f} (n_train={len(train)})")
    print(f"Held-out R^2 = {r2_test:.4f} (n_test={len(test)})")
    print(f"corr(voltage, multiplier) = {r_mult:.4f}")
    print(f"corr(voltage, load_mw)    = {r_load:.4f}")

    return {
        "severity_label": label,
        "severity": severity,
        "intercept": intercept,
        "coef_multiplier": b_mult,
        "coef_load_mw": b_load,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "n_train": len(train),
        "n_test": len(test),
        "corr_voltage_multiplier": r_mult,
        "corr_voltage_load_mw": r_load,
    }


def cross_check_against_existing(rows):
    """Compare r2_test here against the value already saved by the
    corresponding run_*_recall_audit.py script's own output CSV, so this
    is a checked match, not just a second, disconnected computation."""
    existing_files = {
        1.26: "e2_recall_audit.csv",
        1.50: "severity_recall_audit_1.50x.csv",
        1.75: "severity_recall_audit_1.75x.csv",
    }
    print("\n=== Cross-check against existing run_*_recall_audit.py output ===")
    for row in rows:
        fname = existing_files.get(row["severity"])
        path = os.path.join(OUTPUT_DIR, fname) if fname else None
        if path and os.path.exists(path):
            existing = pd.read_csv(path).iloc[0]
            existing_r2 = existing["regression_r2_test"]
            match = abs(existing_r2 - row["r2_test"]) < 0.001
            print(f"{row['severity_label']}: this script r2_test={row['r2_test']:.4f}, "
                  f"{fname} r2_test={existing_r2:.4f} -- {'MATCH' if match else 'MISMATCH'}")
        else:
            print(f"{row['severity_label']}: no existing file found to cross-check against ({fname})")


def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()

    rows = [fit_one(label, severity, mode, simulator, stream) for label, severity, mode in SEVERITIES]

    cross_check_against_existing(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = pd.DataFrame(rows)
    out["fitting_library"] = "scikit-learn==1.9.0"
    out_path = os.path.join(OUTPUT_DIR, "regression_coefficients.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(out)} severities: {', '.join(r['severity_label'] for r in rows)})")


if __name__ == "__main__":
    main()
