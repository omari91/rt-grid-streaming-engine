import os
import sys
import pandas as pd
sys.path.append('supplementary_pypsa_meshed_grid/precision_improvements')

from engine_dynamic import (
    GridSimulator,
    LocalCsvIngestionLayer,
    TimeSeriesLoadProvider,
    ScaledLoadProvider,
    load_simbench_profile
)

SEVERITY_LEVELS = [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00]

def main():
    print("Running dynamic selector sweep...")
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    
    # We use the built-in simbench profile for the background load
    simbench_vals = load_simbench_profile()
    base_provider = TimeSeriesLoadProvider(simbench_vals)

    rows = []
    for severity in SEVERITY_LEVELS:
        print(f"Running severity={severity}...")
        provider = ScaledLoadProvider(base_provider, severity)
        cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)
        
        n_total = len(cycle_df)
        n_critical_flagged = cycle_df["critical_event"].sum()
        
        n_violations = int((cycle_df["raw_vm_ref_pu"] < 0.90).sum())
        caught_mask = (cycle_df["critical_event"] == True) & (cycle_df["raw_vm_ref_pu"] < 0.90)
        n_violations_caught = int(caught_mask.sum())
        
        precision = (n_violations_caught / n_critical_flagged * 100.0) if n_critical_flagged > 0 else 0.0
        recall = (n_violations_caught / n_violations * 100.0) if n_violations > 0 else 0.0
        
        row = {
            "severity": severity,
            "total_events": n_total,
            "flagged": n_critical_flagged,
            "actual_violations": n_violations,
            "violations_caught": n_violations_caught,
            "recall_pct": recall,
            "precision_pct": precision
        }
        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    out_dir = "supplementary_pypsa_meshed_grid/precision_improvements"
    out_path = os.path.join(out_dir, "dynamic_severity_sweep_results.csv")
    sweep_df.to_csv(out_path, index=False)
    
    print(f"\nSaved {out_path}")
    print(sweep_df.to_string(index=False))

if __name__ == "__main__":
    main()
