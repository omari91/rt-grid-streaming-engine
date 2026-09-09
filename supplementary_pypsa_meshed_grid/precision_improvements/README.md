# Precision Improvements Sandbox

This directory contains the isolated execution of the core `rt-grid-streaming-engine` test suite against the new **Background-Aware Dynamic Selector** (`engine_dynamic.py`).

All scripts inside this directory have been copied from the root and modified to import `engine_dynamic.py` instead of the baseline `engine.py`. Results are written to `final_output_cigre/` (CIGRE MV grid) and `final_output_pypsa/` (PyPSA-DE meshed grid) — the root `final_output/` is **never touched**.

---

## What Changed: Dynamic Selector

The `OnlineSmartSelector.is_critical()` method was updated to accept the current grid `background_multiplier` and dynamically scale its threshold:

```python
def is_critical(self, cur: float, prev: float, background_multiplier: float = 1.0) -> bool:
    # High background load → small threshold (more sensitive).
    # Low background load  → large threshold (less sensitive).
    dynamic_p95 = self.p95 * (1.0 / max(background_multiplier, 0.1))
    if cur > dynamic_p95:
        return True
    if abs(cur - prev) > self.rocof_thresh:
        return True
    return False
```

The `PhysicsEngine` was also parameterised via `GRID_TYPE` environment variable to switch between:
- `GRID_TYPE=cigre` — original CIGRE MV benchmark (no stress multiplier)
- `GRID_TYPE=pypsa` — PyPSA-DE meshed subgrid (auto-converging stress multiplier, settled at ×4.0)

---

## Results

### CIGRE MV Grid (`final_output_cigre/`)

Same testbed as `paper.tex` — direct apples-to-apples comparison.

| Selector | Flagged | True Violations (TP) | Recall | Precision |
|---|---|---|---|---|
| Magnitude-only | 1,627 | 189 | 10.0% | 11.6% |
| Step-change-only | 3,595 | 356 | 18.9% | 9.9% |
| **Combined — Dynamic** | **4,055** | **404** | **21.4%** | **10.0%** |
| Random baseline | 3,849 | 352 | 18.7% | 9.1% |

**E2 Recall Audit:** regression catch rate = **82.0%** (CI95: 79.9–83.9%)

### PyPSA-DE Meshed Grid (`final_output_pypsa/`)

New stressed meshed grid — stress multiplier converged at ×4.0.

| Selector | Flagged | True Violations (TP) | Recall | Precision |
|---|---|---|---|---|
| Magnitude-only | 1,627 | 162 | 8.0% | 10.0% |
| Step-change-only | 3,595 | 361 | 17.8% | 10.0% |
| **Combined — Dynamic** | **4,055** | **405** | **19.9%** | **10.0%** |
| Random baseline | 3,849 | 379 | 18.7% | 9.9% |

**E2 Recall Audit:** regression catch rate = **100.0%** (CI95: 99.8–100.0%)

---

## Comparison vs. paper.tex Baseline

The following numbers are hardcoded in `paper.tex` (static selector, CIGRE MV grid):

| Metric | paper.tex (static) | Dynamic — CIGRE | Δ |
|---|---|---|---|
| Combined Recall | 15.5% | **21.4%** | +5.9 pp ↑ |
| Combined Precision | 5.9% | **10.0%** | +4.1 pp ↑ |
| True Violations Found | 239 | **404** | +165 (+69%) |
| E2 Regression Catch Rate | 20.5% | **82.0%** | +61.5 pp ↑ |

> **Note:** The violation population differs between paper.tex (1,544) and the sandbox (≈1,885 for CIGRE, ≈2,068 for PyPSA-DE) because the dynamic background signal changes how many violations the grid actually generates. Before incorporating these numbers into the paper, a fixed-oracle comparison (same 1,544 violations, dynamic selector applied) is recommended for a strict apples-to-apples test.

---

## How to Reproduce

From the repository root:

```bash
# Run both grids (takes ~30–40 min)
bash supplementary_pypsa_meshed_grid/precision_improvements/run_both_grids.sh

# Or run a single grid
export GRID_TYPE=cigre  # or 'pypsa'
bash supplementary_pypsa_meshed_grid/precision_improvements/run_all_scripts.sh
```

Output CSVs will appear in:
- `supplementary_pypsa_meshed_grid/precision_improvements/final_output_cigre/final_output/`
- `supplementary_pypsa_meshed_grid/precision_improvements/final_output_pypsa/final_output/`
