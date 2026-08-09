# rt-grid-streaming-engine

Reference implementation for the paper: *"Streaming AC Verification for Real-Time Redispatch Monitoring: A Recall and Latency Study."*

This repository implements a streaming AC power-flow verification pipeline: a fast, physics-free statistical event selector, combined with selective Newton-Raphson (NR) AC validation, evaluated on one year of real German TSO redispatch telemetry against the CIGRE medium-voltage (MV) benchmark network.

## Key Features

- **Real data:** Driven by real redispatch telemetry (`data/redispatch_1yr.csv`, N=20,586 events), not synthetic-only.
- **Physics-based validation:** Full AC Newton-Raphson power flow (with backward/forward-sweep fallback) on flagged events, not a surrogate.
- **Measured, not assumed, detection recall:** A sampled audit with a Wilson-score confidence interval quantifies what the fast selector actually misses.
- **Measured latency:** Per-path cycle-time distributions against a 20 ms control-cycle deadline.
- **Reproducible:** Fixed seed, pinned dependency versions, and a SHA-256 hash of the input data recorded at run time.

## Methodology

The pipeline screens each incoming redispatch event with an $O(1)$ statistical filter (percentile and rate-of-change thresholds on event magnitude, no power-flow solve). Only flagged ("critical") events receive a full AC Newton-Raphson solve against the CIGRE MV benchmark network (15 buses, PV/wind DER), mapped onto the setpoint of the network's largest controllable generator. Three dispatch policies (baseline, droop, voltage-aware) are compared against this ground truth.

The minimum steady-state voltage threshold (0.90 p.u.) is set to the legally compliant EN 50160:2022 §4.2.2.1 worst-case bound for low-voltage supply (230 V −15%), consistent with the ±10%/−15% envelope confirmed in VDE FNN Leitfaden Anhang C, Table 2 (*Langsame Spannungsänderung*).

See `paper.tex` for the full methodology, results, and honestly-reported limitations (the selector's real detection recall is far from complete — see the paper's Results and Discussion sections).

## Prerequisites

- Python 3.8+ (developed and tested on 3.13)
- Operating System: Windows, macOS, or Linux
- The real dataset at `data/redispatch_1yr.csv` (a synthetic Beta-distributed fallback is used automatically if this file is absent, but results will not match the paper)

## Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/omari91/rt-grid-streaming-engine.git
   cd rt-grid-streaming-engine
   ```

2. **Install pinned dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**

   ```bash
   python main.py
   ```

   *Outputs (audit summary, recall audit, latency tables, figures, and a reproducibility header with pinned versions + input data hash) are saved to `final_output/`.*

## Citation

If you use this code in your research, please consider citing our work. The full citation for the corresponding IEEE paper will be added upon publication.
