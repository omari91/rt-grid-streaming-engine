"""
Standalone latency measurement, Network 1 (CIGRE MV), with the AC solver's
JIT compilation explicitly warmed up before timing begins (Prof. Lu review,
item 8). Must be run as an independent process invocation -- not chained
after other experiments in the same process -- since numba JIT state
persists within a process and would make cycle times incomparable to a
genuinely cold-start measurement.

Run this script 5 times as 5 separate process invocations to reproduce the
"five independent runs" deadline-miss-rate range.

Usage
-----
    python run_latency_standalone.py <run_label>
"""
import sys
import os
import pandas as pd

from engine import GridSimulator, LocalCsvIngestionLayer, OUTPUT_DIR

RUN_LABEL = sys.argv[1] if len(sys.argv) > 1 else "run1"


def main():
    simulator = GridSimulator()
    stream = LocalCsvIngestionLayer().fetch_stream()

    cycle_df = simulator.run_streaming_pipeline(stream)
    latency_df = simulator.build_latency_tables(cycle_df)

    out_path = os.path.join(OUTPUT_DIR, f"latency_network1_{RUN_LABEL}.csv")
    latency_df.to_csv(out_path, index=False)
    cycle_df.to_csv(os.path.join(OUTPUT_DIR, f"cycle_times_network1_{RUN_LABEL}.csv"), index=False)

    print(f"=== Network 1 latency, {RUN_LABEL} ===")
    print(f"JIT warm-up (separate, not counted in steady-state latency): {simulator.jit_warmup_ms:.1f} ms")
    print(latency_df.to_string(index=False))

    critical = cycle_df[cycle_df["critical_event"] == True]
    misses = int(critical["deadline_miss"].sum())
    print(f"\nCritical-path deadline misses: {misses}/{len(critical)} ({misses/len(critical)*100:.3f}%)")
    if misses:
        miss_rows = critical[critical["deadline_miss"] == True]
        print(miss_rows[["event_idx", "cycle_time_ms"]].to_string(index=False))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
