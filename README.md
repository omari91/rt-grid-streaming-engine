# rt-grid-streaming-engine

Reference implementation for the paper: *"Streaming AC Voltage-Risk Screening for Distribution Grids: Recall and Latency under Real Redispatch Events."*

This repository implements a streaming AC power-flow screening pipeline: a fast, physics-free statistical event selector, combined with selective Newton-Raphson (NR) AC validation, driven by one year of real German TSO redispatch telemetry applied to the CIGRE medium-voltage (MV) benchmark network. The redispatch events are real operational records; the resulting voltage outcomes are obtained from AC power-flow *simulation* on a benchmark network, not measured feeder telemetry — see "Experimental Scope" below.

## Key Features

- **Real event stream:** Driven by real redispatch telemetry (`data/redispatch_1yr.csv`, N=20,586 events), not synthetic-only.
- **Physics-based validation:** Full AC Newton-Raphson power flow (with backward/forward-sweep fallback) on flagged events, not a surrogate.
- **Pluggable operating-state models:** the same pipeline runs under a fixed baseline, a deterministic synthetic table, or real independently-sourced SimBench load profiles (see below), to test whether conclusions depend on how the network's background loading is generated.
- **Measured, not assumed, detection recall:** A sampled audit with a Wilson-score confidence interval quantifies what the fast selector actually misses.
- **Measured latency:** Per-path cycle-time distributions against a 20 ms control-cycle deadline.
- **Reproducible:** Fixed seed, pinned dependency versions, and a SHA-256 hash of the input data recorded at run time.

## Experimental Scope

Each simulated event combines three distinct components: (1) a real redispatch record from German TSO operational data; (2) the CIGRE MV benchmark network, a standardised reproducible model, not a specific measured feeder; and (3) a network operating state (background loading) applied at the moment the event is evaluated. The voltage outcome is a **simulated voltage-limit violation**, not a claim of a real, measured grid violation. See `paper.tex`, Sec. "Experimental Scope and Terminology," for the full framing.

## Methodology

The pipeline screens each incoming redispatch event with an $O(1)$ statistical filter (percentile and rate-of-change thresholds on event magnitude, no power-flow solve). Only flagged ("critical") events receive a full AC Newton-Raphson solve against the CIGRE MV benchmark network (15 buses, PV/wind DER), mapped onto the setpoint of the network's largest controllable generator. Three dispatch policies (baseline, droop, voltage-aware) are evaluated against this ground truth, though in practice they are dispatch-equivalent on this dataset (the curtailment gate never activates within this generator's capacity — see the paper's Discussion).

The minimum steady-state voltage threshold (0.90 p.u.) follows EN 50160:2022 §4.2.2.1's worst-case bound, defined there for low-voltage supply; we apply it at MV as a conservative proxy, consistent with the ±10% steady-state band the German MV connection code VDE-AR-N 4110 specifies directly at this voltage level.

### Operating-state models (`engine.py`)

The load-multiplier source is a pluggable `load_provider`, passed to `GridSimulator.run_streaming_pipeline()` and `run_recall_audit()`:

- `FixedLoadProvider` — no operating-state variation; nominal CIGRE loads for every event (control condition, E1).
- `SyntheticLoadProvider` — deterministic table hashed from the run seed and event index (main-study default, E2).
- `EmpiricalSampledLoadProvider` — real SimBench load values, drawn i.i.d. per event, seeded (E3).
- `TimeSeriesLoadProvider` — real SimBench load values, in their original time-ordered sequence (E4).

`load_simbench_profile()` loads real, independently published load-profile data via the [`simbench`](https://pypi.org/project/simbench/) package (Meinecke et al. 2020, ODbL-1.0-licensed); no separate download is required, the profile CSVs ship inside the package.

See `paper.tex` for the full methodology, results, and honestly-reported limitations (the selector's estimated detection recall is far from complete — see the paper's Results and Discussion sections).

## Prerequisites

- Python 3.8+ (developed and tested on 3.13 and 3.14)
- Operating System: Windows, macOS, or Linux
- The real dataset at `data/redispatch_1yr.csv` (a synthetic Beta-distributed fallback is used automatically if this file is absent, but results will not match the paper)

## Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/omari91/rt-grid-streaming-engine.git
   cd rt-grid-streaming-engine
   ```

2. **Create and activate a local Python virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3. **Install pinned dependencies (including scikit-learn) inside the venv:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the pipeline:**

   ```bash
   python main.py
   ```

   *Outputs (audit summary, recall audit, latency tables, regression results, figures, and a reproducibility header with pinned versions + input data hash) are saved to `final_output/`.*

5. **(Optional) Run the operating-state experiment matrix (E1–E4):**

   ```bash
   python run_experiment_matrix.py
   ```

   *Runs the same event stream and pipeline under all four operating-state models (fixed, synthetic, SimBench i.i.d., SimBench time-series) and saves a per-experiment comparison (`final_output/experiment_matrix.csv`), including violation counts, sampled recall, latency percentiles, and the loading-vs-voltage / redispatch-vs-voltage correlations for each.*


6. **Run the regression experiment:**

   ```bash
   python regression.py
   ```

   *This will run a linear regression on voltage data using scikit-learn. Results (metrics) will be printed to the console. Ensure all dependencies are installed as per requirements.txt.*
## Output Manifest

All experiment outputs are tracked in [`final_output/manifest.json`](final_output/manifest.json), which maps each output file to its generating script, scenario parameters, and provenance. This includes regression, recall audit, severity sweep, and threshold sensitivity results. See the manifest for detailed mapping of outputs to scenarios and scripts.
# Reproducible Experiments with Docker

## Quick Start (Docker)

1. **Build the Docker image:**
   ```sh
   docker build -t grid-audit-repro .
   ```

2. **Run the main experiment:**
   ```sh
   docker run --rm grid-audit-repro
   ```
   This will execute `python run_experiment_matrix.py` inside the container and generate all experiment outputs, saving them to `final_output/experiment_matrix.csv`.

3. **Persist output files to your host machine:**
   ```sh
   docker run --rm -v "$PWD/final_output:/app/final_output" grid-audit-repro
   ```
   This mounts your local `final_output` directory to the container, so outputs are written directly to your host.

4. **Override the command to run any other script:**
   ```sh
   docker run --rm grid-audit-repro python regression.py
   docker run --rm grid-audit-repro python run_severity_sweep.py
   ```
   Replace the script name as needed to run other analyses.

---

This setup ensures full reproducibility and portability of your experiments. All dependencies and the Python version are frozen in the Docker image, minimizing environmental differences.


## Citation

If you use this code in your research, please consider citing our work. The full citation for the corresponding IEEE paper will be added upon publication.
