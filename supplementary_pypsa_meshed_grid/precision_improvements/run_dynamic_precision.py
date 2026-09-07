import os
import sys
sys.path.append('supplementary_pypsa_meshed_grid/precision_improvements')
import pandas as pd
from engine_dynamic import GridSimulator, SyntheticLoadProvider

def main():
    # We run this from the root folder, so data is just "data"
    data_dir = "data"
    
    # Check if data exists
    if not os.path.exists(os.path.join(data_dir, "redispatch_1yr.csv")):
        print("Run from the root directory: python supplementary_pypsa_meshed_grid/precision_improvements/run_dynamic_precision.py")
        return
        
    from engine_dynamic import LocalCsvIngestionLayer
    ingestor = LocalCsvIngestionLayer()
    stream = ingestor.fetch_stream()

    print("Running dynamic precision improvements sandbox on PyPSA-DE grid...")
    simulator = GridSimulator()
    
    # Run the streaming pipeline
    cycle_df = simulator.run_streaming_pipeline(stream)
    
    print("\n--- RESULTS ---")
    n_total = len(cycle_df)
    n_critical_flagged = cycle_df["critical_event"].sum()
    
    # A true violation is when vm_pu < VOLTAGE_MIN_PU (0.90) and the AC solver converged.
    # In the cycle_df, the "raw" grid's voltage is stored in "raw_vm_ref_pu"
    n_violations = int((cycle_df["raw_vm_ref_pu"] < 0.90).sum())
    
    # To calculate how many true violations were actually caught, we look at where BOTH
    # critical_event is True AND raw_vm_ref_pu < 0.90
    caught_mask = (cycle_df["critical_event"] == True) & (cycle_df["raw_vm_ref_pu"] < 0.90)
    n_violations_caught = int(caught_mask.sum())
    
    precision = (n_violations_caught / n_critical_flagged * 100.0) if n_critical_flagged > 0 else 0.0
    recall = (n_violations_caught / n_violations * 100.0) if n_violations > 0 else 0.0
    
    print(f"Total Events Evaluated: {n_total}")
    print(f"Physical Violations: {n_violations}")
    print(f"Flagged by Dynamic Selector: {n_critical_flagged}")
    print(f"Violations Caught: {n_violations_caught}")
    print(f"Recall:    {recall:.2f}%")
    print(f"Precision: {precision:.2f}%")

if __name__ == "__main__":
    main()
