# Supplementary Benchmarking: PyPSA-DE Meshed Topology

This folder contains the scripts, data, and analytical findings for the supplementary benchmarking conducted on the PyPSA-DE grid. 

This supplementary work was performed to answer a critical structural question regarding the paper's findings: **Does the magnitude-only event selector fail solely because the CIGRE MV benchmark is a radial (tree-like) distribution grid, or is the limitation universal across grid architectures?**

To prove this, we adapted a meshed, heavily-looped transmission grid (PyPSA-DE) and evaluated it at two different physical scales:
1. **High Voltage (HV):** Native 380kV transmission physics.
2. **Medium Voltage (MV):** Scaled down to 20kV distribution constraints.

## Key Findings

The results definitively confirm the core thesis of the paper: **magnitude-only screening fails to outperform random chance on meshed grids, regardless of the voltage level.**

Because power flows through multiple paths in a meshed topology, the *location* and *background state* of an event dictates voltage drop far more than the raw *magnitude* of the injection. 

### 1. PyPSA-DE HV Transmission Grid Results
In the transmission grid, the voltage is extremely stiff. Even massive MW injections rarely cause voltage violations unless the entire grid is under intense, unrealistic stress (5x background load).

| Selector | Flagged Events | True Violations Caught | Recall | Precision |
|----------|----------------|------------------------|--------|-----------|
| **Random Baseline** | 4,055 | 953 | 19.7% | 23.5% |
| **Combined (Proposed)** | 4,055 | 725 | 14.9% | 17.8% |

On the HV grid, the proposed selector actually performs worse than random guessing.

### 2. PyPSA-DE MV Distribution Grid Results
By scaling PyPSA down to 20kV, reducing line lengths to 2-15km, and restoring the 1/1000 scaling on the telemetry data, we accurately simulated a distribution grid. The grid exhibited 405 physical violations under an automatic 4.0x stress multiplier.

| Selector | Flagged Events | True Violations Caught | Recall | Precision |
|----------|----------------|------------------------|--------|-----------|
| **Random Baseline** | 4,055 | 400 | 19.68% | 9.8% |
| **Combined (Proposed)** | 4,055 | 405 | 19.92% | 9.9% |

*Conclusion:* The magnitude-only selector performs identically to tossing a coin. 90% of the AC solves it triggers are wasted on false alarms.

## Repository Organization

- `pypsa_to_pandapower.py`: Script to parse the original PyPSA-DE `.nc` file and convert the Northern German sub-grid into a `pandapower` JSON grid scaled to 20kV MV physics.
- `engine_pypsa.py`: A modified version of the main `engine.py` that dynamically loads the PyPSA-DE subgrid (instead of the radial CIGRE MV grid), applies adaptive stress multipliers to force convergence/vulnerability, and correctly maps TSO telemetry to PyPSA network generation.
- `final_output_pypsa_hv/`: The full suite of analysis results when running the PyPSA-DE grid at its native 380kV High Voltage scale.
- `final_output_pypsa_mv/`: The full suite of analysis results when running the PyPSA-DE grid scaled to 20kV Medium Voltage physics.

## Reproducibility
To reproduce these supplementary results:
1. Run `python pypsa_to_pandapower.py` to generate the grid JSON.
2. Temporarily replace the root `engine.py` with `engine_pypsa.py`.
3. Run `bash run_all_scripts.sh`.

**Note:** The root `engine.py` evaluates the primary CIGRE MV benchmark by default, which is the core evaluation reported in the main body of the paper.
