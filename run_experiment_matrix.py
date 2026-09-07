"""
Runs the E1-E6 operating-state experiment matrix requested in review and
writes final_output/experiment_matrix.csv.

Each row applies the same real redispatch event stream and the same
selector/validation pipeline; only the operating-state (load) model differs:

  E1  FixedLoadProvider           nominal CIGRE loads, no variation (control)
  E2  SyntheticLoadProvider       deterministic hashed table (main-study default;
                                  confirmed winter-only, curated for overloads --
                                  see engine.py's load_multipliers comment)
  E3  EmpiricalSampledLoadProvider  real SimBench values, drawn i.i.d. per event
  E4  TimeSeriesLoadProvider      real SimBench values, in original time order
  E5a TimeSeriesLoadProvider      real, non-curated winter CIGRE data (seasonal check)
  E5b TimeSeriesLoadProvider      real, non-curated transition CIGRE data (seasonal check)
  E6  TimeSeriesLoadProvider      real, full-year, calendar-following CIGRE seasonal data
                                  (every event assigned its own real season/hour-of-day)

E5/E6 close the "did you consider weather season" gap: E2's table is confirmed winter-only
(CarlosGS20 repo's own README), so E5a/E5b test real non-curated winter/transition data in
isolation, and E6 tests the full real year, calendar-correct, in one pass. All three find
zero violations at native relative severity with the loading-dominates-magnitude mechanism
intact, confirming E2's violations are attributable to its curated 26%-above-peak severity,
not to season (paper.tex Sec. "Does This Survive Independent Load Data?").

Usage
-----
    python run_experiment_matrix.py
"""
import numpy as np
import pandas as pd

from engine import (
    GridSimulator,
    LocalCsvIngestionLayer,
    FixedLoadProvider,
    SyntheticLoadProvider,
    EmpiricalSampledLoadProvider,
    TimeSeriesLoadProvider,
    load_simbench_profile,
    load_cigre_seasonal_profile,
    build_full_year_seasonal_multiplier,
    OUTPUT_DIR,
)
import os


def summarize_run(label, cycle_df, recall_summary_df, load_provider, simulator):
    critical = cycle_df[cycle_df["critical_event"] == True]
    solved = critical.dropna(subset=["raw_vm_ref_pu"])
    n_events = len(cycle_df)
    n_critical = len(critical)
    n_violations = int((solved["raw_vm_ref_pu"] < 0.90).sum())
    n_solver_failures = int((~solved["converged"]).sum())

    recall_row = recall_summary_df.iloc[0]
    fn_rate = recall_row["estimated_fn_rate"]
    est_missed = recall_row["estimated_missed_violations_in_population"]
    est_total_violations = n_violations + est_missed
    est_recall = n_violations / est_total_violations if est_total_violations else float("nan")

    critical_latency = critical["cycle_time_ms"]
    deadline_miss_rate = (critical["deadline_miss"].sum() / n_critical) if n_critical else float("nan")

    # r(load multiplier, voltage) and r(redispatch magnitude, voltage) on solved critical events
    if len(solved) > 2:
        multipliers = np.array([load_provider.get_multiplier(int(idx)) for idx in solved["event_idx"]])
        v = solved["raw_vm_ref_pu"].to_numpy()
        redispatch = solved["load_mw"].to_numpy()
        r_load_v = float(np.corrcoef(multipliers, v)[0, 1]) if np.std(multipliers) > 0 else float("nan")
        r_redispatch_v = float(np.corrcoef(redispatch, v)[0, 1]) if np.std(redispatch) > 0 else float("nan")
    else:
        r_load_v = float("nan")
        r_redispatch_v = float("nan")

    return {
        "experiment": label,
        "n_events": n_events,
        "n_critical": n_critical,
        "n_violations": n_violations,
        "n_solver_failures": n_solver_failures,
        "sampled_fn_rate": fn_rate,
        "estimated_recall": est_recall,
        "nr_calls_critical_path": len(solved),
        "median_latency_ms": float(critical_latency.median()) if n_critical else float("nan"),
        "p95_latency_ms": float(critical_latency.quantile(0.95)) if n_critical else float("nan"),
        "p99_latency_ms": float(critical_latency.quantile(0.99)) if n_critical else float("nan"),
        "deadline_miss_rate": deadline_miss_rate,
        "r_load_vs_voltage": r_load_v,
        "r_redispatch_vs_voltage": r_redispatch_v,
    }


def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    print(f"Loaded real event stream: n={len(stream)}")

    simbench_vals = load_simbench_profile()
    print(f"Loaded SimBench profile (G0-A_pload): n={len(simbench_vals)}, "
          f"min={simbench_vals.min():.4f}, max={simbench_vals.max():.4f}")

    winter_vals = load_cigre_seasonal_profile("winter")
    transition_vals = load_cigre_seasonal_profile("transition")
    print(f"Loaded real, non-curated CIGRE winter profile: n={len(winter_vals)}, "
          f"mean={winter_vals.mean():.4f}x, peak={winter_vals.max():.4f}x")
    print(f"Loaded real, non-curated CIGRE transition profile: n={len(transition_vals)}, "
          f"mean={transition_vals.mean():.4f}x, peak={transition_vals.max():.4f}x")

    redispatch_csv_path = os.path.join("data", "redispatch_1yr.csv")
    full_year_vals = build_full_year_seasonal_multiplier(redispatch_csv_path)
    print(f"Built full-year, calendar-following seasonal multiplier: n={len(full_year_vals)}, "
          f"mean={full_year_vals.mean():.4f}x, peak={full_year_vals.max():.4f}x")
    assert len(full_year_vals) == len(stream), (
        f"full-year multiplier length {len(full_year_vals)} != stream length {len(stream)} -- "
        "build_full_year_seasonal_multiplier() must stay aligned 1:1 with fetch_stream()"
    )

    experiments = {
        "E1_fixed": FixedLoadProvider(),
        "E2_synthetic": SyntheticLoadProvider(simulator.load_multipliers),
        "E3_simbench_iid": EmpiricalSampledLoadProvider(simbench_vals),
        "E4_simbench_timeseries": TimeSeriesLoadProvider(simbench_vals),
        "E5a_real_winter_noncurated": TimeSeriesLoadProvider(winter_vals),
        "E5b_real_transition_noncurated": TimeSeriesLoadProvider(transition_vals),
        "E6_full_year_real_season": TimeSeriesLoadProvider(full_year_vals),
    }

    rows = []
    for label, provider in experiments.items():
        print(f"\n=== Running {label} ===")
        cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)
        _, recall_summary_df = simulator.run_recall_audit(cycle_df, load_provider=provider)
        row = summarize_run(label, cycle_df, recall_summary_df, provider, simulator)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

        cycle_df.to_csv(os.path.join(OUTPUT_DIR, f"cycle_times_{label}.csv"), index=False)

    matrix_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "experiment_matrix.csv")
    matrix_df.to_csv(out_path, index=False)
    print(f"\nSaved experiment matrix to {out_path}")
    print(matrix_df.to_string(index=False))


if __name__ == "__main__":
    main()
