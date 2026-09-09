"""Camera-ready revision analysis for ETECOM 2026 Review 1's points 3 and 4:

3. A combined, overall system recall metric (already answered by
   run_e2_recall_audit.py / run_severity_recall_audit.py's "Extrapolated
   overall recall" line -- this script does not repeat that).
4. A random, budget-matched baseline, and a magnitude-only vs.
   step-change-only vs. combined ablation, both evaluated against
   EXHAUSTIVE ground truth (every one of the 20,586 events, not a sample):
   critical-path events are already AC-solved by the main pipeline;
   non-critical events are audited exhaustively here via the same
   run_recall_audit() call the paper's own reported numbers use
   (full census, sample_size=16,676), not the 500-event convenience
   default. This produces ONE full-population ground-truth table used for
   every comparison below, so the random baseline, the ablation, and the
   reported E2 selector are all measured on identical footing.

RANDOM, BUDGET-MATCHED BASELINE
--------------------------------
The real selector flags 4,055/20,586 events (19.7%). A uniformly random
subset of the same size, drawn with the same budget, is simulated
MONTE-CARLO style (10,000 draws, RUN_SEED-derived) against the same
exhaustive ground truth, reporting the empirical recall/precision
distribution -- not just the analytic expectation (budget/population =
19.7%), so the comparison carries its own uncertainty bounds rather than
a single point estimate.

ABLATION
--------
Re-derives which events would be flagged under three rules, all using the
SAME frozen calibration-window thresholds already fit by the real
OnlineSmartSelector (no new fitting, no leakage):
  - magnitude-only:    cur > p95
  - step-change-only:  abs(cur - prev) > step_thresh
  - combined (as shipped): magnitude OR step-change
Each rule's flagged set is checked against the same exhaustive ground
truth for recall and precision.

Usage:
    python run_camera_ready_analysis.py
"""
import numpy as np
import pandas as pd

import sys
sys.path.append('supplementary_pypsa_meshed_grid/precision_improvements')
from engine_dynamic import (
    GridSimulator,
    LocalCsvIngestionLayer,
    SyntheticLoadProvider,
    OnlineSmartSelector,
    CALIBRATION_FRACTION,
    VOLTAGE_MIN_PU,
    RUN_SEED,
    OUTPUT_DIR,
)
import os

N_MONTE_CARLO = 10_000


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    provider = SyntheticLoadProvider(simulator.load_multipliers)

    print(f"Loaded real event stream: n={len(stream)}")
    cycle_df = simulator.run_streaming_pipeline(stream, load_provider=provider)

    # Exhaustive ground truth: critical-path events already AC-solved by the
    # main pipeline; audit every non-critical event too (full census).
    n_noncritical = int((cycle_df["critical_event"] == False).sum())
    sample_df, recall_summary_df = simulator.run_recall_audit(
        cycle_df, sample_size=n_noncritical, load_provider=provider
    )
    print(f"Exhaustive non-critical audit: {len(sample_df)}/{n_noncritical} events "
          f"({int(sample_df['sampled_violation'].sum())} violations found)")

    ground_truth = cycle_df.set_index("event_idx")["raw_vm_ref_pu"].copy()
    for idx, row in sample_df.set_index("event_idx").iterrows():
        ground_truth.loc[idx] = row["sampled_vm_pu"]

    is_violation = (ground_truth < VOLTAGE_MIN_PU).fillna(False).to_numpy()
    n_total = len(cycle_df)
    n_total_violations = int(is_violation.sum())
    print(f"\nFull-population ground truth assembled: n={n_total}, "
          f"total confirmed violations={n_total_violations} "
          f"(cross-check vs. paper's reported 1,544)")

    flagged_real = (cycle_df["critical_event"] == True).to_numpy()
    n_flagged_real = int(flagged_real.sum())

    # --- Random, budget-matched baseline (Monte Carlo) ---
    rng = np.random.default_rng(RUN_SEED)
    all_idx = np.arange(n_total)
    recalls, precisions = [], []
    for _ in range(N_MONTE_CARLO):
        draw = rng.choice(all_idx, size=n_flagged_real, replace=False)
        flagged_mask = np.zeros(n_total, dtype=bool)
        flagged_mask[draw] = True
        tp = int((flagged_mask & is_violation).sum())
        recalls.append(tp / n_total_violations)
        precisions.append(tp / n_flagged_real)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    print(f"\n=== Random, budget-matched baseline (n_flagged={n_flagged_real}, "
          f"{N_MONTE_CARLO} draws) ===")
    print(f"Recall: mean={recalls.mean()*100:.2f}%, "
          f"95% range=[{np.percentile(recalls,2.5)*100:.2f}%, {np.percentile(recalls,97.5)*100:.2f}%]")
    print(f"Precision: mean={precisions.mean()*100:.2f}%, "
          f"95% range=[{np.percentile(precisions,2.5)*100:.2f}%, {np.percentile(precisions,97.5)*100:.2f}%]")

    # --- Real selector, for direct comparison on identical ground truth ---
    real_tp = int((flagged_real & is_violation).sum())
    real_recall_p, real_recall_lo, real_recall_hi = wilson_ci(real_tp, n_total_violations)
    real_prec_p, real_prec_lo, real_prec_hi = wilson_ci(real_tp, n_flagged_real)
    print(f"\n=== Real magnitude/step-change selector (for comparison) ===")
    print(f"n_flagged={n_flagged_real}, tp={real_tp}")
    print(f"Recall: {real_recall_p*100:.2f}% (Wilson 95% CI [{real_recall_lo*100:.2f}%, {real_recall_hi*100:.2f}%])")
    print(f"Precision: {real_prec_p*100:.2f}% (Wilson 95% CI [{real_prec_lo*100:.2f}%, {real_prec_hi*100:.2f}%])")

    # --- Ablation: magnitude-only vs. step-change-only vs. combined ---
    calibration_n = max(1, int(len(stream) * CALIBRATION_FRACTION))
    selector = OnlineSmartSelector(stream[:calibration_n])
    p95 = selector.p95
    step_thresh = selector.rocof_thresh
    print(f"\nFrozen thresholds (from calibration window, unchanged): "
          f"p95={p95:.4f}, step_thresh={step_thresh:.4f}")

    mag_only = np.zeros(n_total, dtype=bool)
    step_only = np.zeros(n_total, dtype=bool)
    combined = np.zeros(n_total, dtype=bool)
    prev = stream[0]
    for i, cur in enumerate(stream):
        is_mag = cur > p95
        is_step = abs(cur - prev) > step_thresh
        mag_only[i] = is_mag
        step_only[i] = is_step
        combined[i] = is_mag or is_step
        prev = cur

    print(f"\n=== Ablation (against identical exhaustive ground truth) ===")
    rows = []
    for name, mask in [("magnitude-only", mag_only), ("step-change-only", step_only),
                        ("combined (as shipped)", combined)]:
        n_flag = int(mask.sum())
        tp = int((mask & is_violation).sum())
        rec_p, rec_lo, rec_hi = wilson_ci(tp, n_total_violations)
        prec_p, prec_lo, prec_hi = wilson_ci(tp, n_flag)
        print(f"{name:25s}: n_flagged={n_flag:6d}, tp={tp:5d}, "
              f"recall={rec_p*100:5.1f}% [{rec_lo*100:.1f},{rec_hi*100:.1f}], "
              f"precision={prec_p*100:5.1f}% [{prec_lo*100:.1f},{prec_hi*100:.1f}]")
        rows.append({
            "rule": name, "n_flagged": n_flag, "tp": tp,
            "recall_pct": rec_p * 100, "recall_ci95_low": rec_lo * 100, "recall_ci95_high": rec_hi * 100,
            "precision_pct": prec_p * 100, "precision_ci95_low": prec_lo * 100, "precision_ci95_high": prec_hi * 100,
        })

    rows.append({
        "rule": "random_budget_matched_baseline", "n_flagged": n_flagged_real, "tp": int(recalls.mean() * n_total_violations),
        "recall_pct": recalls.mean() * 100, "recall_ci95_low": np.percentile(recalls, 2.5) * 100,
        "recall_ci95_high": np.percentile(recalls, 97.5) * 100,
        "precision_pct": precisions.mean() * 100, "precision_ci95_low": np.percentile(precisions, 2.5) * 100,
        "precision_ci95_high": np.percentile(precisions, 97.5) * 100,
    })

    out_path = os.path.join(OUTPUT_DIR, "camera_ready_ablation_and_baseline.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
