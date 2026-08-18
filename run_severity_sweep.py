"""
Severity-sweep experiment: find the onset-of-violation loading severity for
this network, under a real load *shape* (SimBench, time-ordered), rather than
adopting a third-party-curated severity level (E2's 1.26x) without knowing
where the actual threshold lies.

ScaledLoadProvider wraps TimeSeriesLoadProvider (real SimBench shape) and
applies a single, explicit multiplier on top of it, uniformly across the
whole event stream. The multiplier is swept; nothing here is chosen by
inspecting which value produces violations before running. The initial
range (1.00x-1.30x, bracketing SimBench's own unscaled peak and E2's
known-to-violate 1.26x table) found no violations even past E2's own
peak, so the range was extended upward to locate the actual onset.

Usage
-----
    python run_severity_sweep.py
"""
import pandas as pd

from engine import (
    GridSimulator,
    LocalCsvIngestionLayer,
    TimeSeriesLoadProvider,
    ScaledLoadProvider,
    load_simbench_profile,
    OUTPUT_DIR,
)
import os

SEVERITY_LEVELS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.26, 1.30, 1.35, 1.40, 1.50, 1.75, 2.00]


def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    simbench_vals = load_simbench_profile()
    base_provider = TimeSeriesLoadProvider(simbench_vals)

    print(f"Real event stream: n={len(stream)}")
    print(f"SimBench base shape: min={simbench_vals.min():.4f}, max={simbench_vals.max():.4f}")
    print(f"Severity levels to sweep: {SEVERITY_LEVELS}\n")

    rows = []
    for severity in SEVERITY_LEVELS:
        provider = ScaledLoadProvider(base_provider, severity)
        cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)

        critical = cycle_df[cycle_df["critical_event"] == True]
        solved = critical.dropna(subset=["raw_vm_ref_pu"])
        n_critical = len(critical)
        n_violations = int((solved["raw_vm_ref_pu"] < 0.90).sum())
        min_v = float(solved["raw_vm_ref_pu"].min()) if len(solved) else float("nan")
        n_solver_failures = int((~solved["converged"]).sum())

        row = {
            "severity": severity,
            "peak_multiplier": float(simbench_vals.max()) * severity,
            "n_critical": n_critical,
            "n_violations": n_violations,
            "n_solver_failures": n_solver_failures,
            "min_voltage_pu": min_v,
        }
        rows.append(row)
        print(f"severity={severity:.2f} (peak={row['peak_multiplier']:.3f}x nominal): "
              f"{n_violations} violations, min V={min_v:.4f} p.u., "
              f"{n_solver_failures} solver failures")

    sweep_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "severity_sweep.csv")
    sweep_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(sweep_df.to_string(index=False))

    onset = sweep_df[sweep_df["n_violations"] > 0]
    if len(onset):
        first = onset.iloc[0]
        print(f"\nOnset of violations: severity={first['severity']:.2f} "
              f"(peak={first['peak_multiplier']:.3f}x documented network peak)")
    else:
        print("\nNo violations found in swept range.")


if __name__ == "__main__":
    main()
