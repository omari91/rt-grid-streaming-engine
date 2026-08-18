"""
Re-runs the E2 (default synthetic operating-state) recall audit and
out-of-sample regression test at a larger sample size, for direct,
equal-footing comparison with the severity-sweep points (1.5x, 1.75x),
which needed n=4500 for adequate statistical power. Mirrors
run_severity_recall_audit.py's methodology exactly, using
SyntheticLoadProvider instead of ScaledLoadProvider.

Usage
-----
    python run_e2_recall_audit.py [sample_size]
"""
import sys
import numpy as np
import pandas as pd

from engine import (
    GridSimulator,
    LocalCsvIngestionLayer,
    SyntheticLoadProvider,
    OUTPUT_DIR,
    RUN_SEED,
    VOLTAGE_MIN_PU,
)
import os

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 16676  # full non-critical population (census)


def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    provider = SyntheticLoadProvider(simulator.load_multipliers)

    print(f"E2 (default synthetic operating-state), sample_size={SAMPLE_SIZE}")

    cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)
    critical = cycle_df[cycle_df["critical_event"] == True]
    solved = critical.dropna(subset=["raw_vm_ref_pu"])
    n_critical = len(critical)
    n_violations_critical = int((solved["raw_vm_ref_pu"] < VOLTAGE_MIN_PU).sum())
    print(f"Critical-path: {n_critical} events, {n_violations_critical} confirmed violations")

    sample_df, recall_summary_df = simulator.run_recall_audit(cycle_df, sample_size=SAMPLE_SIZE, load_provider=provider)
    recall_row = recall_summary_df.iloc[0]
    print(f"Recall audit: {int(recall_row['sample_violations_found'])} violations in "
          f"{int(recall_row['sample_size'])}-event sample "
          f"(FN rate {recall_row['estimated_fn_rate']*100:.1f}%, "
          f"95% CI [{recall_row['estimated_fn_rate_ci95_low']*100:.1f}%, "
          f"{recall_row['estimated_fn_rate_ci95_high']*100:.1f}%])")

    est_missed = recall_row["estimated_missed_violations_in_population"]
    est_total_violations = n_violations_critical + est_missed
    baseline_recall = n_violations_critical / est_total_violations if est_total_violations else float("nan")
    print(f"Estimated total violation population: {est_total_violations:.0f}; "
          f"magnitude-only recall: {baseline_recall*100:.1f}%")

    solved = solved.copy()
    solved["multiplier"] = solved["event_idx"].apply(lambda i: provider.get_multiplier(int(i)))
    rng = np.random.default_rng(RUN_SEED)
    idx = rng.permutation(len(solved))
    n_train = int(0.7 * len(solved))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train, test = solved.iloc[train_idx], solved.iloc[test_idx]

    X_train = np.column_stack([train["multiplier"], train["load_mw"], np.ones(len(train))])
    y_train = train["raw_vm_ref_pu"].to_numpy()
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    b_mult, b_load, intercept = coef

    def predict(mult, load):
        return intercept + b_mult * mult + b_load * load

    y_test_pred = predict(test["multiplier"], test["load_mw"])
    y_test_true = test["raw_vm_ref_pu"].to_numpy()
    ss_res = np.sum((y_test_true - y_test_pred) ** 2)
    ss_tot = np.sum((y_test_true - y_test_true.mean()) ** 2)
    r2_test = 1 - ss_res / ss_tot
    print(f"Regression: V ~ {intercept:.4f} + {b_mult:.4f}*mult + {b_load:.4f}*P_load, "
          f"held-out R^2 = {r2_test:.4f} (n_train={len(train)}, n_test={len(test)})")

    sample_df = sample_df.copy()
    sample_df["multiplier"] = sample_df["event_idx"].apply(lambda i: provider.get_multiplier(int(i)))
    sample_df["v_pred"] = predict(sample_df["multiplier"], sample_df["load_mw"])
    sample_df["regression_flag"] = sample_df["v_pred"] < VOLTAGE_MIN_PU

    n_sample_violations = int(sample_df["sampled_violation"].sum())
    n_caught = int((sample_df["sampled_violation"] & sample_df["regression_flag"]).sum())
    n_flagged = int(sample_df["regression_flag"].sum())

    z = 1.959963984540054
    if n_sample_violations:
        p_hat = n_caught / n_sample_violations
        n = n_sample_violations
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        ci_low, ci_high = max(0.0, center - margin), min(1.0, center + margin)
    else:
        p_hat = ci_low = ci_high = float("nan")

    print(f"\nOut-of-sample regression rule on recall sample:")
    print(f"  Known violations in sample: {n_sample_violations}")
    if n_sample_violations:
        print(f"  Caught: {n_caught}/{n_sample_violations} "
              f"({p_hat*100:.1f}%, Wilson 95% CI [{ci_low*100:.1f}%, {ci_high*100:.1f}%])")
    print(f"  Total flagged by regression rule: {n_flagged}/{len(sample_df)}")

    if n_sample_violations:
        catch_rate = n_caught / n_sample_violations
        additional_caught = catch_rate * est_missed
        new_total_caught = n_violations_critical + additional_caught
        new_recall = new_total_caught / est_total_violations * 100
        print(f"  Extrapolated overall recall with state-aware rule: {new_recall:.1f}%")

    out = {
        "experiment": "E2",
        "sample_size": SAMPLE_SIZE,
        "n_critical": n_critical,
        "n_violations_critical": n_violations_critical,
        "sample_violations_found": n_sample_violations,
        "estimated_fn_rate": recall_row["estimated_fn_rate"],
        "estimated_total_violations": est_total_violations,
        "baseline_recall_pct": baseline_recall * 100,
        "regression_r2_test": r2_test,
        "regression_n_caught": n_caught,
        "regression_catch_rate_pct": p_hat * 100 if n_sample_violations else float("nan"),
        "regression_catch_rate_ci95_low": ci_low * 100 if n_sample_violations else float("nan"),
        "regression_catch_rate_ci95_high": ci_high * 100 if n_sample_violations else float("nan"),
    }
    suffix = "" if SAMPLE_SIZE == 16676 else f"_n{SAMPLE_SIZE}"
    out_path = os.path.join(OUTPUT_DIR, f"e2_recall_audit{suffix}.csv")
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
