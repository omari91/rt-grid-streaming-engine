"""E6: replace E2's winter-only table with a real, full-year, season-following
operating-state model, and check whether the results change.

WHY THIS SCRIPT EXISTS
-----------------------
run_seasonal_check.py (E5a/E5b) tested two isolated 3-day season snapshots in
comparable isolation and found no violations either way, at native relative
severity. That's informative but not the full answer: our redispatch event
stream is a genuine full real year (18 Jul 2025 - 18 Jul 2026,
data/redispatch_1yr.csv, real BEGINN_DATUM/BEGINN_UHRZEIT per event), so the
most rigorous close of the "did you consider season" gap is to build an
operating-state model that actually follows the real calendar across that
whole year -- not two snapshots -- and re-run the exact same audit.

DATA
----
CarlosGS20/Typical-load-profile-MV-CIGRE-benchmark's "15-days typical year
test case" ships exactly this: 360 real hourly per-node consumption values
(15 CIGRE nodes) split into three real 5-day (120-hour) blocks, in the order
stated by that directory's own readme.md: "the transition season between
winter and summer, five days of winter, and finally, five days of summer."
    t0-119:   Transition
    t120-239: Winter
    t240-359: Summer
This is a different (richer, full-3-season) CarlosGS20 dataset from the one
run_seasonal_check.py used (the 3-day "Two seasonal scenarios" directory,
winter+transition only, no summer). The two datasets' Winter/Transition
aggregates do NOT agree on which is higher (3-day set: Transition +4.75% over
Winter; 15-day set: Winter is highest, Transition second, Summer lowest) --
stated here plainly rather than silently reconciled, since we cannot verify
against Porsinger et al. 2017 directly (paywalled).

METHOD
------
1. Each 120-hour season block is summed across all 15 node rows (real
   aggregate network demand, same convention as run_seasonal_check.py) and
   normalized by the documented CIGRE peak (44.742 MW, pandapower
   create_cigre_network_mv() net.load.p_mw.sum()) into a per-unit multiplier
   array of length 120.
2. Every one of the 20,586 REAL events in data/redispatch_1yr.csv is assigned
   a season from its real BEGINN_DATUM month (Widely-used meteorological
   convention: Winter=Dec/Jan/Feb, Summer=Jun/Jul/Aug, else Transition -- the
   same 3-way split Porsinger et al. and this data source use).
3. Within a season, the event's real hour-of-day (from BEGINN_UHRZEIT) is
   preserved exactly. Which of that season's 5 template days applies is
   assigned by a running per-season calendar-day counter mod 5 (the Nth
   distinct real calendar date seen in that season, across the whole year,
   modulo 5) -- NOT a claimed real weekday/Saturday/Sunday alignment, which
   the source data does not unambiguously support (checked directly: within
   the Winter block, days 0 and 2 are byte-identical, which is consistent
   with a repeated "weekday" template but does not by itself establish which
   of the remaining days are Saturday/Sunday). Stated plainly as a modelling
   choice, not verified calendar fact.
4. The resulting 20,586-length multiplier array, aligned 1:1 with
   LocalCsvIngestionLayer.fetch_stream()'s row order, is passed to
   TimeSeriesLoadProvider (idx % len(values) is then a no-op since lengths
   match exactly) and run through the unmodified audit pipeline, identical to
   E1-E5.

Usage:
    python run_full_year_seasonal_check.py
Requires: data/cigre_seasonal/Active_Node_Consumption_15day_full.csv (pulled
from CarlosGS20/Typical-load-profile-MV-CIGRE-benchmark, "15-days typical year
test case/Scenario" directory) and data/redispatch_1yr.csv (already in repo).
"""
import csv
import os

import numpy as np
import pandas as pd
import pandapower.networks as pn

from engine import GridSimulator, LocalCsvIngestionLayer, TimeSeriesLoadProvider, OUTPUT_DIR
from run_experiment_matrix import summarize_run

DATA_DIR = "data/cigre_seasonal"
REDISPATCH_CSV = os.path.join("data", "redispatch_1yr.csv")

SEASON_BOUNDS = {  # t-index ranges within the 360-hour block, per its own readme order
    "Transition": (0, 120),
    "Winter": (120, 240),
    "Summer": (240, 360),
}


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (6, 7, 8):
        return "Summer"
    return "Transition"


def load_node_matrix(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[float(x) for x in row[1:]] for row in reader]
    return np.array(rows)


def main():
    net = pn.create_cigre_network_mv(with_der=False)
    documented_peak_mw = float(net.load.p_mw.sum())
    print(f"Documented network peak: {documented_peak_mw:.3f} MW")

    matrix = load_node_matrix(os.path.join(DATA_DIR, "Active_Node_Consumption_15day_full.csv"))
    agg = matrix.sum(axis=0)  # real total network demand, 360 real hourly values
    season_tables = {}
    for season, (lo, hi) in SEASON_BOUNDS.items():
        table = agg[lo:hi] / documented_peak_mw
        season_tables[season] = table
        print(f"{season:10s}: n={len(table)}, mean={table.mean():.4f}x, peak={table.max():.4f}x")

    # Real per-event calendar assignment.
    df = pd.read_csv(REDISPATCH_CSV, sep=";")
    dt = pd.to_datetime(df["BEGINN_DATUM"] + " " + df["BEGINN_UHRZEIT"], format="%d.%m.%Y %H:%M")
    seasons = dt.dt.month.map(month_to_season)

    season_day_counter = {"Winter": {}, "Summer": {}, "Transition": {}}
    day_in_cycle = np.zeros(len(df), dtype=int)
    for season in season_tables:
        mask = (seasons == season).to_numpy()
        dates = dt[mask].dt.date
        seen = {}
        cycle_vals = []
        for d in dates:
            if d not in seen:
                seen[d] = len(seen)
            cycle_vals.append(seen[d] % 5)
        day_in_cycle[mask] = cycle_vals

    hours = dt.dt.hour.to_numpy()
    multiplier = np.empty(len(df))
    for season, table in season_tables.items():
        mask = (seasons == season).to_numpy()
        idx_in_table = day_in_cycle[mask] * 24 + hours[mask]
        multiplier[mask] = table[idx_in_table]

    print(f"\nEvent count by season: {seasons.value_counts().to_dict()}")
    print(f"Full-year multiplier: n={len(multiplier)}, mean={multiplier.mean():.4f}x, "
          f"min={multiplier.min():.4f}x, max={multiplier.max():.4f}x")

    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    assert len(stream) == len(multiplier), (
        f"stream/multiplier length mismatch: {len(stream)} vs {len(multiplier)} -- "
        "fetch_stream() row order must match redispatch_1yr.csv exactly"
    )
    print(f"\nLoaded real event stream: n={len(stream)}")

    provider = TimeSeriesLoadProvider(multiplier)
    print("\n=== Running E6_full_year_real_season ===")
    cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)
    _, recall_summary_df = simulator.run_recall_audit(cycle_df, load_provider=provider)
    row = summarize_run("E6_full_year_real_season", cycle_df, recall_summary_df, provider, simulator)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "cycle_times_E6_full_year_real_season.csv"), index=False)
    matrix_df = pd.DataFrame([row])
    out_path = os.path.join(OUTPUT_DIR, "full_year_seasonal_check.csv")
    matrix_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(matrix_df.to_string(index=False))


if __name__ == "__main__":
    main()
