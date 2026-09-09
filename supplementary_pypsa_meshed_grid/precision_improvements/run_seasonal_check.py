"""E5: does E2's headline result hold under a genuinely different season?

WHY THIS SCRIPT EXISTS
-----------------------
E2's default operating-state table (self.load_multipliers in GridSimulator) is
sourced from CarlosGS20/Typical-load-profile-MV-CIGRE-benchmark's "5-days test
case" -- confirmed directly from that repository's own README: "This case is
based on winter consumption profiles." E2 is therefore a winter-only operating
state applied uniformly across a full year of real redispatch events, and
paper.tex's current text (Sec. "Does This Survive Independent Load Data?")
explains E2's severity via the 26%-above-documented-peak curation only, not the
season -- see writing/etecom-revision/email-prof-lu-e2-seasonality.md.

This script asks the direct next question the same repository lets us answer
with real data, not assumption: the same source (CarlosGS20 repo, "Two seasonal
scenarios of 3-days test case") also ships a genuinely different season
(Transition, i.e. spring/autumn) built the same way as the Winter data, from
real per-node CIGRE consumption profiles (Porsinger et al. 2017) -- NOT the
"curated to produce overloads" Scenario A/B construction. Building a real,
non-curated Transition-season E2 variant and running it through the exact same
audit machinery as run_experiment_matrix.py answers: is the winter table
uniquely severe, or would a different real season produce a broadly similar
result?

METHOD
------
1. The two per-node CSVs (Active_Node_Consumption_{winter,transition}.csv, 15
   CIGRE nodes x 72 hourly timesteps = 3 days) are summed across nodes to get
   each season's real, real-shaped total network active-power demand.
2. Each is normalized into a per-unit-of-documented-peak multiplier by
   dividing by 44.742 MW -- pandapower's create_cigre_network_mv() total
   nominal load (net.load.p_mw.sum()), the same Rudion et al. 2006 documented
   peak already used as this paper's E1 baseline (project_e2_provenance_chain
   memory). STATED ASSUMPTION, not verified against the source paper's own
   units: the CSV's raw values are treated as already being in MW. This is
   consistent with the order of magnitude (CSV Node 1 winter peak =
   13.50 MW vs. pandapower's real Node 1 residential rating, Load R1 =
   14.994 MW) but not independently confirmed against Porsinger et al. 2017
   directly (paywalled, not accessible during this check). The comparison
   between seasons is robust to this assumption either way, since the same
   conversion is applied to both seasons identically -- what would NOT be
   robust is comparing either season's absolute violation count against E2's
   own (different-dataset, different-curation) severity in isolation.
3. Each per-unit series is cycled (idx % len(values), matching E4's own
   TimeSeriesLoadProvider convention) across the real, full redispatch event
   stream and run through the unmodified audit pipeline
   (GridSimulator.run_streaming_pipeline / run_recall_audit), identical to
   E1-E4 in run_experiment_matrix.py.

Usage:
    python run_seasonal_check.py
Requires: data/cigre_seasonal/Active_Node_Consumption_{winter,transition}.csv
(pulled from CarlosGS20/Typical-load-profile-MV-CIGRE-benchmark, "Two seasonal
scenarios of 3-days test case" directory).
"""
import csv
import os

import numpy as np
import pandas as pd
import pandapower.networks as pn

import sys
sys.path.append('supplementary_pypsa_meshed_grid/precision_improvements')
from engine_dynamic import GridSimulator, LocalCsvIngestionLayer, TimeSeriesLoadProvider, OUTPUT_DIR
from run_experiment_matrix import summarize_run

DATA_DIR = "data/cigre_seasonal"
DOCUMENTED_PEAK_MW = None  # computed below from pandapower, not hard-coded


def load_seasonal_aggregate(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_t = len(header) - 1
        agg = np.zeros(n_t)
        for row in reader:
            vals = np.array([float(x) for x in row[1:]])
            agg += vals
    return agg


def main():
    global DOCUMENTED_PEAK_MW
    net = pn.create_cigre_network_mv(with_der=False)
    DOCUMENTED_PEAK_MW = float(net.load.p_mw.sum())
    print(f"Documented network peak (Rudion et al. 2006, via pandapower): {DOCUMENTED_PEAK_MW:.3f} MW")

    winter_agg = load_seasonal_aggregate(os.path.join(DATA_DIR, "Active_Node_Consumption_winter.csv"))
    transition_agg = load_seasonal_aggregate(os.path.join(DATA_DIR, "Active_Node_Consumption_transition.csv"))

    winter_mult = winter_agg / DOCUMENTED_PEAK_MW
    transition_mult = transition_agg / DOCUMENTED_PEAK_MW

    print(f"\nWinter (real, non-curated):     n={len(winter_mult)}, "
          f"mean={winter_mult.mean():.4f}x, peak={winter_mult.max():.4f}x")
    print(f"Transition (real, non-curated): n={len(transition_mult)}, "
          f"mean={transition_mult.mean():.4f}x, peak={transition_mult.max():.4f}x")
    print(f"Transition/Winter mean ratio: {transition_mult.mean()/winter_mult.mean():.4f} "
          f"({(transition_mult.mean()/winter_mult.mean() - 1)*100:+.2f}%)")
    print(f"Existing E2 table (curated Scenario A) peak, for reference: "
          f"{GridSimulator().load_multipliers.max():.4f}x -- NOT directly comparable, "
          f"different dataset/curation, shown only as context")

    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    print(f"\nLoaded real event stream: n={len(stream)}")

    experiments = {
        "E5a_real_winter_noncurated": TimeSeriesLoadProvider(winter_mult),
        "E5b_real_transition_noncurated": TimeSeriesLoadProvider(transition_mult),
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
    out_path = os.path.join(OUTPUT_DIR, "seasonal_check_matrix.csv")
    matrix_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(matrix_df.to_string(index=False))


if __name__ == "__main__":
    main()
