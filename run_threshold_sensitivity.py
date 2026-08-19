"""
Threshold sensitivity analysis (Prof. Lu review, item 7, optional "if space
permits"): sweeps the selector's magnitude percentile threshold and reports
the resulting recall/precision trade-off under E2, holding everything else
fixed (same calibration window, same RoCoF check, same exhaustive audit
methodology). Answers "what does the selector's screening trade-off look
like at other operating points on this rule," not just the one point
reported as the headline result.

Usage
-----
    python run_threshold_sensitivity.py
"""
import pandas as pd
import os

from engine import GridSimulator, LocalCsvIngestionLayer, SyntheticLoadProvider, OUTPUT_DIR

PERCENTILES = [90, 93, 95, 97, 99]


def main():
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()

    rows = []
    for pct in PERCENTILES:
        simulator = GridSimulator()
        provider = SyntheticLoadProvider(simulator.load_multipliers)
        cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider, selector_percentile=pct)

        critical = cycle_df[cycle_df["critical_event"] == True]
        solved = critical.dropna(subset=["raw_vm_ref_pu"])
        n_critical = len(critical)
        tp = int((solved["raw_vm_ref_pu"] < 0.90).sum())

        _, recall_summary_df = simulator.run_recall_audit(cycle_df, sample_size=16676, load_provider=provider)
        recall_row = recall_summary_df.iloc[0]
        fn = int(recall_row["sample_violations_found"])

        fp = n_critical - tp
        precision = tp / n_critical * 100 if n_critical else float("nan")
        recall = tp / (tp + fn) * 100 if (tp + fn) else float("nan")

        row = {
            "percentile": pct,
            "n_critical": n_critical,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_pct": precision,
            "recall_pct": recall,
        }
        rows.append(row)
        print(f"p{pct}: n_critical={n_critical}, TP={tp}, FP={fp}, FN={fn}, "
              f"precision={precision:.2f}%, recall={recall:.2f}%")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
