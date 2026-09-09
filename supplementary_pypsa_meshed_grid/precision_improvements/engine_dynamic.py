import os
import time
import logging
import warnings
import json
import hashlib
import platform
import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower as pp
import pandapower.networks as pn
from pandapower.auxiliary import LoadflowNotConverged


# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR = "supplementary_pypsa_meshed_grid/precision_improvements/final_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 10, "font.family": "serif", "figure.dpi": 220})

RUN_SEED = 42
RNG = np.random.default_rng(RUN_SEED)

GLOBAL_TRAFO_LIMIT_MW = 45.0
DER_CAPACITY_MW = 1.5
VOLTAGE_MIN_PU = 0.90
VOLTAGE_TARGET_PU = 0.91
PF_Q_RATIO = 0.33
DEFAULT_STREAM_N = 500
OP_DEADLINE_MS = 20.0
CALIBRATION_FRACTION = 0.10  # initial stream window used to fit selector thresholds, then frozen


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

def _file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "not-found"
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

def build_repro_header(stream_n: int = DEFAULT_STREAM_N, data_path: str = os.path.join("data", "redispatch_1yr.csv")) -> dict:
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scipy": _pkg_version("scipy"),
            "pandapower": _pkg_version("pandapower"),
            "matplotlib": _pkg_version("matplotlib"),
            "seaborn": _pkg_version("seaborn"),
        },
        "experiment": {
            "seed": RUN_SEED,
            "stream_n": stream_n,
            "global_trafo_limit_mw": GLOBAL_TRAFO_LIMIT_MW,
            "voltage_min_pu": VOLTAGE_MIN_PU,
            "voltage_target_pu": VOLTAGE_TARGET_PU,
            "pf_q_ratio": PF_Q_RATIO,
            "deadline_ms": OP_DEADLINE_MS,
        },
        "data": {
            "path": data_path,
            "sha256": _file_sha256(data_path),
        },
    }

def write_repro_header(output_dir: str = OUTPUT_DIR, stream_n: int = DEFAULT_STREAM_N, data_path: str = os.path.join("data", "redispatch_1yr.csv")) -> dict:
    header = build_repro_header(stream_n, data_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "run_metadata.json"
    out_path.write_text(json.dumps(header, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("REPRODUCIBILITY HEADER")
    print("=" * 64)
    print(json.dumps(header, indent=2))
    print("=" * 64)
    return header


# ==============================================================================
# DATA INGESTION
# ==============================================================================
class LocalCsvIngestionLayer:
    """Reads real TSO telemetry from redispatch_1yr.csv, scales to MV constraints."""

    def fetch_stream(self, n: int | None = None) -> np.ndarray:
        """Returns the full real event stream by default. Pass n to cap it
        (e.g. for quick local testing); omit it to use every row in the CSV."""
        csv_path = os.path.join("data", "redispatch_1yr.csv")
        if not os.path.exists(csv_path):
            return RNG.beta(2, 4, n or DEFAULT_STREAM_N) * DER_CAPACITY_MW

        df = pd.read_csv(csv_path, sep=";")

        # Parse European decimals
        df["MITTLERE_LEISTUNG_MW"] = df["MITTLERE_LEISTUNG_MW"].astype(str).str.replace(",", ".").astype(float)

        df["delta_mw"] = df["MITTLERE_LEISTUNG_MW"] / 1000.0
        # Ensure no negative redispatch events (as direction was discarded by the model's design)
        df["delta_mw"] = df["delta_mw"].clip(lower=0.0)

        stream = df["delta_mw"].values
        if n is not None and len(stream) > n:
            stream = stream[:n]
        return stream


class OnlineSmartSelector:
    """Flags high-magnitude or sharp-ramp events for high-fidelity handling.

    Thresholds are fit once from an initial calibration window (the first
    CALIBRATION_FRACTION of the stream) and then frozen -- not recomputed
    from the full stream, which would let the selector implicitly condition
    on future events relative to any point in its own operational history.
    Magnitude-only (95th percentile), not a two-sided band: with direction
    discarded during the redispatch-to-generator mapping (Sec. Dataset),
    the event stream is a non-negative magnitude series, and a lower-tail
    threshold on it has no clear physical meaning -- dropped in favour of
    a single high-magnitude threshold plus the rate-of-change check.
    """

    def __init__(self, calibration_data: np.ndarray, percentile: float = 95):
        self.p95 = np.percentile(calibration_data, percentile)
        diffs = np.diff(calibration_data)
        self.rocof_thresh = float(np.std(diffs) * 1.2) if len(diffs) else 0.0

    def is_critical(self, cur: float, prev: float, background_multiplier: float = 1.0) -> bool:
        # Dynamic precision logic: scale threshold inversely with background load.
        # High background load -> small threshold (more sensitive).
        # Low background load -> large threshold (less sensitive).
        dynamic_p95 = self.p95 * (1.0 / max(background_multiplier, 0.1))
        if cur > dynamic_p95:
            return True
        if abs(cur - prev) > self.rocof_thresh:
            return True
        return False


# ==============================================================================
# CONTROLLERS
# ==============================================================================
class VoltageAwareController:
    def __init__(self, limit_mw: float, v_target_pu: float = VOLTAGE_TARGET_PU):
        self.limit = limit_mw
        self.v_target = v_target_pu

    def compute(self, load_mw: float, current_v: float) -> float:
        p_safe = min(load_mw, self.limit)
        if current_v < self.v_target:
            sensitivity = (1.0 - self.v_target) / max(1.0 - current_v, 1e-9)
            p_safe = min(p_safe, load_mw * sensitivity)
        return max(0.0, p_safe)


class BaselineController:
    def __init__(self, limit_mw: float):
        self.limit = limit_mw

    def compute(self, load_mw: float) -> float:
        return min(load_mw, self.limit)


class DroopController:
    def __init__(self, limit_mw: float, k_mw_per_pu: float = 20.0):
        self.limit = limit_mw
        self.k = k_mw_per_pu

    def compute(self, load_mw: float, current_v: float) -> float:
        if current_v < VOLTAGE_TARGET_PU:
            curtailment = self.k * (VOLTAGE_TARGET_PU - current_v)
            return max(0.0, min(load_mw - curtailment, self.limit))
        return min(load_mw, self.limit)


# ==============================================================================
# PHYSICS
# ==============================================================================
def linear_surrogate_estimate(load_mw: float) -> float:
    """Fast linear surrogate voltage estimate for streaming control."""
    v = 1.0 - 0.0000042 * load_mw
    return float(np.clip(v, 0.55, 1.0))


@dataclass
class EvalResult:
    load_mw: float
    min_vm_pu: float
    line_loading_pct: float
    converged: bool = True


class PhysicsEngine:
    """Pandapower NR/BFSW reference model for reproducible AC validation."""

    def __init__(self):
        import os
        subgrid_path = os.path.join("data", "pypsa_de_subgrid.json")
        grid_type = os.environ.get("GRID_TYPE", "pypsa") # Default to pypsa for existing scripts
        
        if grid_type == "pypsa" and os.path.exists(subgrid_path):
            self.net = pp.from_json(subgrid_path)
            if len(self.net.sgen) > 0:
                # Map redispatch events to the largest regional generator in the PyPSA subgrid
                self.gen_idx = self.net.sgen.p_mw.idxmax()
            else:
                self.gen_idx = 0
                
            # Artificially stress the MV grid to force voltage vulnerability.
            # Without this, the grid may be too robust to violate limits.
            stress_multiplier = 10.0
            if len(self.net.load) > 0:
                self.net.load.p_mw *= stress_multiplier
                self.net.load.q_mvar *= stress_multiplier

            # Ensure initial AC load flow converges
            converged = False
            while stress_multiplier >= 1.0 and not converged:
                try:
                    pp.runpp(self.net)
                    # Also check if it's too close to voltage collapse (e.g., v < 0.8)
                    if self.net.res_bus.vm_pu.min() < 0.8:
                        raise pp.LoadflowNotConverged("Voltage too low, consider it unconverged")
                    converged = True
                except pp.LoadflowNotConverged:
                    print(f"Warning: Stressed network (x{stress_multiplier}) did not converge.")
                    # Revert current multiplier and try a lower one
                    self.net.load.p_mw /= stress_multiplier
                    self.net.load.q_mvar /= stress_multiplier
                    stress_multiplier -= 2.0
                    if stress_multiplier >= 1.0:
                        self.net.load.p_mw *= stress_multiplier
                        self.net.load.q_mvar *= stress_multiplier
                        
            if not converged:
                raise RuntimeError("Initial network did not converge even at 1x load.")
                
            print(f"Successfully initialized grid with stress multiplier x{stress_multiplier}")
        else:
            self.net = pn.create_cigre_network_mv(with_der="pv_wind")
            self.gen_idx = self.net.sgen[self.net.sgen.name == 'WKA 7'].index[0]

        self.nominal_p_mw = self.net.load.p_mw.copy()
        self.nominal_q_mvar = self.net.load.q_mvar.copy()

    def set_dynamic_load(self, multiplier: float):
        """Applies a dynamic load multiplier to all loads in the network to simulate peak conditions."""
        self.net.load.p_mw = self.nominal_p_mw * multiplier
        self.net.load.q_mvar = self.nominal_q_mvar * multiplier

    def solve_reference(self, load_mw: float) -> EvalResult:
        # load_mw here represents the redispatch setpoint for the mapped generator.
        # The engine models these events as new demand that threatens undervoltage, 
        # so we assign it as negative generation (consumption).
        clipped_setpoint = float(max(0.0, load_mw))
        self.net.sgen.at[self.gen_idx, "p_mw"] = -clipped_setpoint

        attempts = [
            dict(algorithm="nr", init="flat", max_iteration=30, tolerance_mva=1e-6),
            dict(algorithm="nr", init="results", max_iteration=30, tolerance_mva=1e-6),
            dict(algorithm="bfsw", max_iteration=80, tolerance_mva=1e-6),
        ]

        for kwargs in attempts:
            try:
                pp.runpp(self.net, calculate_voltage_angles=False, **kwargs)
                min_vm = float(self.net.res_bus.vm_pu.min())
                line_loading = float(self.net.res_line.loading_percent.max()) if len(self.net.res_line) else 0.0
                return EvalResult(load_mw=clipped_setpoint, min_vm_pu=min_vm, line_loading_pct=line_loading, converged=True)
            except LoadflowNotConverged:
                continue
            except Exception:
                continue

        return EvalResult(load_mw=clipped_setpoint, min_vm_pu=np.nan, line_loading_pct=np.nan, converged=False)


# ==============================================================================
# SIMULATION CORE
# ==============================================================================
class SyntheticLoadProvider:
    """Deterministic, reproducible hourly load-multiplier table, hashed from
    the run seed and event index. This is the default operating-state model
    used throughout the main study (E2)."""

    def __init__(self, multipliers: np.ndarray, seed: int = RUN_SEED):
        self.multipliers = multipliers
        self.seed = seed

    def get_multiplier(self, idx: int) -> float:
        mi = int(hashlib.md5(f"{self.seed}_{idx}".encode()).hexdigest(), 16) % len(self.multipliers)
        return float(self.multipliers[mi])


class FixedLoadProvider:
    """No operating-state variation: the network's nominal CIGRE loads are
    used unchanged for every event. Control condition (E1) isolating how
    much voltage-risk information the redispatch event carries on its own,
    with background loading held fixed."""

    def get_multiplier(self, idx: int) -> float:
        return 1.0


def load_simbench_profile(column: str = "G0-A_pload", scenario: int = 0) -> np.ndarray:
    """Real, independently-published load-profile values from SimBench
    (Meinecke et al. 2020, ODbL-1.0), used as the operating-state source for
    E3/E4. Decoupled from SimBench's own grid topology: this returns only the
    relative load-scaling time series, applied here to the unrelated CIGRE MV
    network. `column` defaults to a general commercial/business profile
    (BDEW G0-A), the closest standard category to an MV feeder mix."""
    import simbench as sb

    profiles = sb.get_all_simbench_profiles(scenario)["load"]
    return profiles[column].to_numpy()


# Documented network peak used to normalize the real CIGRE seasonal data below into
# per-unit-of-peak multipliers, on the same basis as this paper's E1 baseline and E2's
# curated 1.26x severity: pandapower create_cigre_network_mv().net.load.p_mw.sum(),
# which reproduces Rudion et al. 2006's (ref:rudion2006) documented Table I peak-load
# specification (verified node-by-node against pandapower's defaults).
CIGRE_DOCUMENTED_PEAK_MW = 44.742


def _load_cigre_node_matrix(path: str) -> np.ndarray:
    """Real per-node hourly active-power consumption (CIGRE nodes x hours) from
    CarlosGS20/Typical-load-profile-MV-CIGRE-benchmark (reference.bib: ref:carlosgs2020),
    built from Porsinger et al. 2017's (ref:porsinger2017) seasonal profiles. Returns the
    raw per-node matrix; callers sum across nodes for the network-wide aggregate."""
    df = pd.read_csv(path)
    return df.drop(columns=df.columns[0]).to_numpy()


def load_cigre_seasonal_profile(season: str) -> np.ndarray:
    """Real, non-curated seasonal operating-state multiplier (per unit of
    CIGRE_DOCUMENTED_PEAK_MW) for the E5 seasonal-robustness check (paper.tex Sec. "Does
    This Survive Independent Load Data?"). `season` is 'winter' or 'transition'. Unlike E2's
    table (CarlosGS20 "5-days test case", curated to produce overloads, and confirmed by
    that repository's own README to be winter-only), this data comes from CarlosGS20's "Two
    seasonal scenarios of 3-days test case" -- ordinary representative data, not curated for
    violations. Data: data/cigre_seasonal/Active_Node_Consumption_{season}.csv."""
    path = os.path.join("data", "cigre_seasonal", f"Active_Node_Consumption_{season.lower()}.csv")
    matrix = _load_cigre_node_matrix(path)
    return matrix.sum(axis=0) / CIGRE_DOCUMENTED_PEAK_MW


def build_full_year_seasonal_multiplier(redispatch_csv_path: str) -> np.ndarray:
    """Real, full-year, calendar-following operating-state multiplier for the E6 check
    (paper.tex Sec. "Does This Survive Independent Load Data?"): every real event in
    redispatch_csv_path is assigned its own real season from its real calendar month
    (Winter=Dec/Jan/Feb, Summer=Jun/Jul/Aug, else Transition -- the same 3-way split
    CarlosGS20/Porsinger et al. use) and its own real hour-of-day, then stepped through that
    season's real 5-day CIGRE profile from CarlosGS20's "15-days typical year test case"
    (data/cigre_seasonal/Active_Node_Consumption_15day_full.csv: 360 real hourly values,
    ordered Transition/Winter/Summer per that directory's own README).

    Which of the 5 template days within a season applies is assigned by a running
    per-season calendar-day counter modulo 5 (the Nth distinct real date seen in that
    season, across the whole year) -- NOT a claimed real weekday/Saturday/Sunday alignment,
    which this data does not unambiguously support (checked directly: within the Winter
    block, days 0 and 2 are byte-identical, consistent with a repeated "weekday" template
    but not sufficient on its own to label the remaining days). Stated plainly as a
    modelling choice, not a verified calendar fact.

    Returned array is aligned 1:1 with LocalCsvIngestionLayer.fetch_stream()'s row order
    (same source CSV, same row order, no filtering in either)."""
    full_year_path = os.path.join("data", "cigre_seasonal", "Active_Node_Consumption_15day_full.csv")
    matrix = _load_cigre_node_matrix(full_year_path)
    agg = matrix.sum(axis=0)
    season_bounds = {"Transition": (0, 120), "Winter": (120, 240), "Summer": (240, 360)}
    season_tables = {s: agg[lo:hi] / CIGRE_DOCUMENTED_PEAK_MW for s, (lo, hi) in season_bounds.items()}

    df = pd.read_csv(redispatch_csv_path, sep=";")
    dt = pd.to_datetime(df["BEGINN_DATUM"] + " " + df["BEGINN_UHRZEIT"], format="%d.%m.%Y %H:%M")

    def month_to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (6, 7, 8):
            return "Summer"
        return "Transition"

    seasons = dt.dt.month.map(month_to_season)
    day_in_cycle = np.zeros(len(df), dtype=int)
    for season in season_tables:
        mask = (seasons == season).to_numpy()
        seen = {}
        cycle_vals = []
        for d in dt[mask].dt.date:
            if d not in seen:
                seen[d] = len(seen)
            cycle_vals.append(seen[d] % 5)
        day_in_cycle[mask] = cycle_vals

    hours = dt.dt.hour.to_numpy()
    multiplier = np.empty(len(df))
    for season, table in season_tables.items():
        mask = (seasons == season).to_numpy()
        multiplier[mask] = table[day_in_cycle[mask] * 24 + hours[mask]]
    return multiplier


class EmpiricalSampledLoadProvider:
    """Real SimBench load values drawn i.i.d. (with replacement), seeded by
    run seed and event index, breaking any correlation with event order.
    Isolates whether the loading-voltage relationship reported under
    SyntheticLoadProvider (E2) depends on that provider's own deterministic,
    hashed construction, or holds under an operating-state source sampled
    independently of the event stream (E3; see Sec. Limitations)."""

    def __init__(self, values: np.ndarray, seed: int = RUN_SEED):
        self.values = values
        self.seed = seed

    def get_multiplier(self, idx: int) -> float:
        rng = np.random.default_rng([self.seed, idx])
        return float(rng.choice(self.values))


class TimeSeriesLoadProvider:
    """Real SimBench load values used in their original time-ordered
    sequence (event index modulo profile length), preserving the profile's
    real temporal structure -- unlike EmpiricalSampledLoadProvider's i.i.d.
    draws. Tests whether the loading-voltage relationship holds under a
    realistic, time-varying operating state (E4)."""

    def __init__(self, values: np.ndarray):
        self.values = values

    def get_multiplier(self, idx: int) -> float:
        return float(self.values[idx % len(self.values)])


class ScaledLoadProvider:
    """Wraps a real, representative load-shape source (e.g. SimBench) and
    applies a single, explicit severity multiplier on top of it, uniformly.
    Used to find the onset-of-violation severity for this network under a
    real load shape, swept transparently rather than borrowed from a
    third-party curation (Sec. Limitations)."""

    def __init__(self, base_provider, severity: float):
        self.base_provider = base_provider
        self.severity = severity

    def get_multiplier(self, idx: int) -> float:
        return self.base_provider.get_multiplier(idx) * self.severity


class GridSimulator:
    def __init__(self):
        self.physics = PhysicsEngine()
        
        # Dynamically set controller limit based on the transmission generator capacity
        if len(self.physics.net.sgen) > 0:
            transmission_limit = float(self.physics.net.sgen.p_mw.at[self.physics.gen_idx])
        else:
            transmission_limit = 5000.0
            
        # If capacity was imported as 0 (or very small), use a huge fallback
        if transmission_limit < 1.0:
            transmission_limit = 5000.0
            
        self.prop_ctrl = VoltageAwareController(transmission_limit)
        self.base_ctrl = BaselineController(transmission_limit)
        self.droop_ctrl = DroopController(transmission_limit, k_mw_per_pu=20000.0)
        
        # Empirical daily load multipliers derived from the Typical Load Profile benchmark
        # (Winter Scenario A) -- CarlosGS20/Typical-load-profile-MV-CIGRE-benchmark
        # (ref:carlosgs2020) "5-days test case", curated by that repository to produce
        # network overloads and voltage-limit problems; confirmed winter-only directly from
        # that repository's own README ("This case is based on winter consumption
        # profiles"). This is E2's main-study default table (paper.tex Sec. "Does This
        # Survive Independent Load Data?"). Its seasonal robustness is checked in
        # load_cigre_seasonal_profile() / build_full_year_seasonal_multiplier() below (E5/E6:
        # real, non-curated winter/summer/transition data, including a full real-calendar-year
        # check against every event in redispatch_1yr.csv) -- both find zero violations at
        # native relative severity and an intact loading-dominates-magnitude mechanism,
        # confirming E2's violations are attributable to its curated 26%-above-peak severity,
        # not to season.
        self.load_multipliers = np.array([
            0.13, 0.11, 0.08, 0.06, 0.06, 0.08, 0.18, 0.38, 0.58, 0.77, 
            0.88, 0.94, 0.95, 0.91, 0.86, 0.82, 0.85, 1.05, 1.25, 1.26, 
            1.20, 1.05, 0.81, 0.44, 0.23, 0.16, 0.11, 0.08, 0.07, 0.08,
            0.18, 0.38, 0.58, 0.77, 0.88, 0.94, 0.95, 0.91, 0.86, 0.82, 
            0.85, 1.05, 1.25, 1.26, 1.20, 1.05, 0.81, 0.44, 0.25, 0.18, 
            0.13, 0.09, 0.07, 0.07, 0.09, 0.15, 0.28, 0.45, 0.65, 0.81, 
            0.89, 0.91, 0.87, 0.81, 0.77, 0.75, 0.79, 0.95, 1.15, 1.22, 
            1.12, 0.95, 0.71, 0.41, 0.24, 0.18, 0.12, 0.09, 0.07, 0.06, 
            0.08, 0.14, 0.26, 0.43, 0.63, 0.80, 0.88, 0.91, 0.87, 0.81, 
            0.77, 0.75, 0.79, 0.95, 1.15, 1.22, 1.12, 0.95, 0.71, 0.41, 
            0.23, 0.16, 0.11, 0.08, 0.07, 0.08, 0.18, 0.38, 0.58, 0.77, 
            0.88, 0.94, 0.95, 0.91, 0.86, 0.82, 0.85, 1.05, 1.25, 1.26, 
            1.20, 1.05, 0.81, 0.44
        ])

    def run_streaming_pipeline(self, stream: np.ndarray, load_provider=None, selector_percentile: float = 95):
        if load_provider is None:
            load_provider = SyntheticLoadProvider(self.load_multipliers)
        calibration_n = max(1, int(len(stream) * CALIBRATION_FRACTION))
        selector = OnlineSmartSelector(stream[:calibration_n], percentile=selector_percentile)

        # Warm up the AC solver's JIT-compiled kernels (internal to pandapower,
        # not this codebase) before timing begins, so the one-time compile cost
        # is reported separately (self.jit_warmup_ms) rather than misattributed
        # to whichever real critical event happens to run first.
        t_warmup = time.perf_counter()
        self.physics.solve_reference(0.0)
        self.jit_warmup_ms = (time.perf_counter() - t_warmup) * 1e3

        records = []
        prev = float(stream[0]) if len(stream) else 0.0

        for idx, load in enumerate(stream):
            load = float(load)

            # Operating-state model is pluggable (Sec. Experimental Scope);
            # default reproduces the deterministic synthetic multiplier table.
            current_multiplier = load_provider.get_multiplier(idx)
            self.physics.set_dynamic_load(current_multiplier)

            t0 = time.perf_counter()

            t_sel = time.perf_counter()
            critical_event = selector.is_critical(load, prev, current_multiplier)
            selector_ms = (time.perf_counter() - t_sel) * 1e3

            t_surr = time.perf_counter()
            v_est = linear_surrogate_estimate(load)
            surr_ms = (time.perf_counter() - t_surr) * 1e3

            t_base = time.perf_counter()
            p_base = self.base_ctrl.compute(load)
            base_ctrl_ms = (time.perf_counter() - t_base) * 1e3

            t_droop = time.perf_counter()
            p_droop = self.droop_ctrl.compute(load, v_est)
            droop_ctrl_ms = (time.perf_counter() - t_droop) * 1e3

            t_prop = time.perf_counter()
            p_prop = self.prop_ctrl.compute(load, v_est) if critical_event or v_est < VOLTAGE_MIN_PU else min(load, DER_CAPACITY_MW)
            prop_ctrl_ms = (time.perf_counter() - t_prop) * 1e3

            nr_ms = 0.0
            ref_raw = ref_base = ref_droop = ref_prop = None
            if critical_event:
                t_nr = time.perf_counter()
                # Dispatch values are frequently identical across raw/base/droop/prop
                # (e.g. whenever a controller's curtailment gate doesn't activate);
                # solve NR once per unique value instead of once per name.
                dispatch = {"raw": load, "base": p_base, "droop": p_droop, "prop": p_prop}
                solved_by_value = {}
                for name, val in dispatch.items():
                    key = round(val, 6)
                    if key not in solved_by_value:
                        solved_by_value[key] = self.physics.solve_reference(val)
                ref_raw = solved_by_value[round(dispatch["raw"], 6)]
                ref_base = solved_by_value[round(dispatch["base"], 6)]
                ref_droop = solved_by_value[round(dispatch["droop"], 6)]
                ref_prop = solved_by_value[round(dispatch["prop"], 6)]
                nr_ms = (time.perf_counter() - t_nr) * 1e3

            total_ms = (time.perf_counter() - t0) * 1e3

            records.append(
                {
                    "event_idx": idx,
                    "load_mw": load,
                    "critical_event": critical_event,
                    "selector_time_ms": selector_ms,
                    "surrogate_time_ms": surr_ms,
                    "baseline_ctrl_ms": base_ctrl_ms,
                    "droop_ctrl_ms": droop_ctrl_ms,
                    "proposed_ctrl_ms": prop_ctrl_ms,
                    "nr_time_ms": nr_ms,
                    "cycle_time_ms": total_ms,
                    "deadline_miss": total_ms > OP_DEADLINE_MS,
                    "surrogate_vm_pu": v_est,
                    "raw_vm_ref_pu": np.nan if ref_raw is None else ref_raw.min_vm_pu,
                    "converged": True if ref_raw is None else ref_raw.converged,
                    "base_vm_ref_pu": np.nan if ref_base is None else ref_base.min_vm_pu,
                    "droop_vm_ref_pu": np.nan if ref_droop is None else ref_droop.min_vm_pu,
                    "prop_vm_ref_pu": np.nan if ref_prop is None else ref_prop.min_vm_pu,
                    "p_base_mw": p_base,
                    "p_droop_mw": p_droop,
                    "p_prop_mw": p_prop,
                }
            )
            prev = load

        df = pd.DataFrame(records)
        return df

    def run_benchmark_audit(self, cycle_df: pd.DataFrame):
        audited = cycle_df.dropna(subset=["raw_vm_ref_pu"]).copy()

        # Only evaluate physical violations if the solver converged
        audited["actual_violation"] = (audited["raw_vm_ref_pu"] < VOLTAGE_MIN_PU) & (audited["converged"] == True)
        # Symmetric with droop/prop: does this controller's own dispatched setpoint,
        # once solved with real AC power flow, still leave the network in violation?
        audited["base_violation"] = audited["base_vm_ref_pu"] < VOLTAGE_MIN_PU
        audited["droop_violation"] = audited["droop_vm_ref_pu"] < VOLTAGE_MIN_PU
        audited["prop_violation"] = audited["prop_vm_ref_pu"] < VOLTAGE_MIN_PU

        summary = {
            "total_true_physical_violations": int(audited["actual_violation"].sum()),
            "baseline_controller_violations": int(audited["base_violation"].sum()),
            "droop_controller_violations": int(audited["droop_violation"].sum()),
            "proposed_controller_violations": int(audited["prop_violation"].sum()),
            "critical_solver_failures": int((~audited["converged"]).sum()),
        }
        # NOTE: this audit only covers events already flagged critical, since NR
        # is never run on non-critical events in the streaming loop. The
        # selector's false-negative rate is NOT measurable from this table alone
        # (it would be tautologically 0) -- see run_recall_audit() for the real,
        # sampled estimate.

        summary_df = pd.DataFrame([summary])
        return audited, summary_df

    def run_recall_audit(self, cycle_df: pd.DataFrame, sample_size: int = 500, seed: int = RUN_SEED, load_provider=None):
        """Estimates the event-selector's false-negative rate on real data.

        Non-critical events never receive a physics solve in the streaming
        loop, so we can't know from cycle_df alone whether the selector missed
        any real violations. This draws a random sample of events the selector
        called non-critical and runs real AC power flow on them (replaying the
        same per-event operating state used in the original streaming pass),
        to get an honest, sample-based estimate with a confidence interval.

        sample_size=500 here is only a convenience default for quick/exploratory
        calls; it was never statistically justified and is NOT what the paper's
        reported numbers use. All reported results call this with the full
        non-critical population (16,676) for a zero-sampling-uncertainty census
        -- see run_severity_recall_audit.py / run_e2_recall_audit.py.
        """
        if load_provider is None:
            load_provider = SyntheticLoadProvider(self.load_multipliers)
        noncritical = cycle_df[cycle_df["critical_event"] == False]
        rng = np.random.default_rng(seed)
        n = min(sample_size, len(noncritical))
        sample_idx = rng.choice(noncritical.index.values, size=n, replace=False)
        sample = cycle_df.loc[sample_idx].copy()

        converged_flags, violation_flags, vm_results = [], [], []
        for _, row in sample.iterrows():
            self.physics.set_dynamic_load(load_provider.get_multiplier(int(row["event_idx"])))
            result = self.physics.solve_reference(row["load_mw"])
            converged_flags.append(result.converged)
            vm_results.append(result.min_vm_pu)
            violation_flags.append(bool(result.converged and result.min_vm_pu < VOLTAGE_MIN_PU))

        sample["sampled_converged"] = converged_flags
        sample["sampled_vm_pu"] = vm_results
        sample["sampled_violation"] = violation_flags

        solver_failures = int((~sample["sampled_converged"]).sum())
        n_valid = n - solver_failures
        violations_found = int(sample["sampled_violation"].sum())

        # Wilson score 95% CI on the false-negative rate within the sampled population
        z = 1.959963984540054
        p_hat = violations_found / n_valid if n_valid else 0.0
        denom = 1 + z**2 / n_valid if n_valid else 1.0
        center = (p_hat + z**2 / (2 * n_valid)) / denom if n_valid else 0.0
        margin = (z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_valid)) / n_valid)) / denom if n_valid else 0.0

        summary = {
            "noncritical_population": int(len(noncritical)),
            "sample_size": int(n),
            "sample_solver_failures": solver_failures,
            "sample_violations_found": violations_found,
            "estimated_fn_rate": p_hat,
            "estimated_fn_rate_ci95_low": max(0.0, center - margin),
            "estimated_fn_rate_ci95_high": min(1.0, center + margin),
            "estimated_missed_violations_in_population": p_hat * len(noncritical),
        }
        return sample, pd.DataFrame([summary])

    def build_latency_tables(self, cycle_df: pd.DataFrame):
        rows = []
        groups = {
            "all_events": cycle_df,
            "critical_events": cycle_df[cycle_df["critical_event"] == True],
            "noncritical_events": cycle_df[cycle_df["critical_event"] == False],
        }

        for name, df in groups.items():
            if len(df) == 0:
                continue
            rows.append(
                {
                    "group": name,
                    "n_events": int(len(df)),
                    "avg_cycle_ms": float(df["cycle_time_ms"].mean()),
                    "median_cycle_ms": float(df["cycle_time_ms"].median()),
                    "p95_cycle_ms": float(df["cycle_time_ms"].quantile(0.95)),
                    "p99_cycle_ms": float(df["cycle_time_ms"].quantile(0.99)),
                    "avg_nr_ms": float(df["nr_time_ms"].mean()),
                    "deadline_misses": int(df["deadline_miss"].sum()),
                }
            )

        return pd.DataFrame(rows)

    def benchmark_control_kernel(self):
        n_vals = [10_000, 100_000, 1_000_000]
        results = []

        for N in n_vals:
            trials = []
            for _ in range(5):
                x = RNG.random(N) * 10.0
                t0 = time.perf_counter()
                _ = 1.0 / (1.0 + np.exp(-0.5 * x))
                elapsed = time.perf_counter() - t0
                trials.append((N / elapsed) / 1e6)
            results.append({"N": N, "mean_mops": float(np.mean(trials)), "std_mops": float(np.std(trials))})

        return pd.DataFrame(results)

    def generate_figures(self, audited_df: pd.DataFrame, cycle_df: pd.DataFrame, throughput_df: pd.DataFrame):
        vis = audited_df.head(20).copy()
        if len(vis) == 0:
            vis = cycle_df.head(20).copy()
            vis["base_vm_ref_pu"] = vis["surrogate_vm_pu"]
            vis["droop_vm_ref_pu"] = vis["surrogate_vm_pu"]
            vis["prop_vm_ref_pu"] = vis["surrogate_vm_pu"]

        fig1, ax1 = plt.subplots(figsize=(6.8, 3.8))
        ax1.plot(vis.index, vis["base_vm_ref_pu"], label="Baseline", color="#c0392b", ls="--", lw=1.5, marker="x")
        ax1.plot(vis.index, vis["droop_vm_ref_pu"], label="Droop", color="#f39c12", ls="-.", lw=1.4, marker="^")
        ax1.plot(vis.index, vis["prop_vm_ref_pu"], label="Proposed", color="#27ae60", lw=2.0, marker="o")
        ax1.axhline(VOLTAGE_MIN_PU, color="black", ls=":", label=f"Limit ({VOLTAGE_MIN_PU} p.u.)")
        ax1.set_title("Voltage Stability Comparison", fontweight="bold")
        ax1.set_xlabel("Simulation Steps")
        ax1.set_ylabel("Voltage (p.u.)")
        ax1.legend(fontsize=8, loc="best")
        fig1.tight_layout()
        fig1.savefig(os.path.join(OUTPUT_DIR, "Voltage_Stability.png"))
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6.2, 3.4))
        ax2.errorbar(throughput_df["N"], throughput_df["mean_mops"], yerr=throughput_df["std_mops"], fmt="o-", capsize=4, color="#2980b9")
        ax2.set_xscale("log")
        ax2.set_title("Vectorized Control Throughput", fontweight="bold")
        ax2.set_xlabel("Operation Count")
        ax2.set_ylabel("Million Ops/Sec")
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUTPUT_DIR, "Scalability.png"))
        plt.close(fig2)

        # Empirical CDF of cycle time, critical vs. non-critical path -- replaces an earlier
        # log-scale scatter trace (Sec. Computational Cost) that obscured how often the
        # system approaches or misses the 20 ms deadline; a CDF makes the miss rate a single
        # readable crossing point instead of requiring the reader to count sparse dots.
        fig4, ax4 = plt.subplots(figsize=(6.6, 3.4))
        for label, mask, color in [
            ("Critical path", cycle_df["critical_event"] == True, "#c0392b"),
            ("Non-critical path", cycle_df["critical_event"] == False, "#2980b9"),
        ]:
            times = np.sort(cycle_df.loc[mask, "cycle_time_ms"].to_numpy())
            if len(times) == 0:
                continue
            cdf = np.arange(1, len(times) + 1) / len(times)
            ax4.plot(times, cdf, color=color, lw=1.6, label=f"{label} (n={len(times)})")
        ax4.axvline(OP_DEADLINE_MS, color="black", ls=":", label="20 ms deadline")
        ax4.set_xscale("log")
        ax4.set_title("Cycle-Time CDF by Path", fontweight="bold")
        ax4.set_xlabel("Cycle Time (ms, log scale)")
        ax4.set_ylabel("Cumulative Fraction of Events")
        ax4.set_ylim(0, 1.02)
        ax4.legend(fontsize=8, loc="lower right")
        fig4.tight_layout()
        fig4.savefig(os.path.join(OUTPUT_DIR, "Cycle_Time_CDF.png"))
        plt.close(fig4)


def print_reports(summary_df: pd.DataFrame, latency_df: pd.DataFrame, recall_df: pd.DataFrame):
    summary = summary_df.iloc[0].to_dict()
    print("\n" + "=" * 64)
    print("AUDIT REPORT: STREAMING SCREENING VS. NR GROUND TRUTH")
    print("=" * 64)
    for k, v in summary.items():
        print(f"{k:35s}: {int(v)}")
    print("=" * 64)

    print("\nLATENCY REPORT")
    print("=" * 64)
    print(latency_df.to_string(index=False))
    print("=" * 64)

    print("\nRECALL AUDIT (sampled, non-critical population)")
    print("=" * 64)
    print(recall_df.to_string(index=False))
    print("=" * 64)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    simulator = GridSimulator()
    ingest = LocalCsvIngestionLayer()
    stream = ingest.fetch_stream()
    write_repro_header(stream_n=len(stream))

    print(
        f"STREAM FINGERPRINT | n={len(stream)} min={stream.min():.6f} "
        f"max={stream.max():.6f} mean={stream.mean():.6f} "
        f"std={stream.std():.6f}"
    )

    cycle_df = simulator.run_streaming_pipeline(stream)
    audited_df, summary_df = simulator.run_benchmark_audit(cycle_df)
    recall_sample_df, recall_summary_df = simulator.run_recall_audit(cycle_df)
    latency_df = simulator.build_latency_tables(cycle_df)
    throughput_df = simulator.benchmark_control_kernel()

    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "cycle_times.csv"), index=False)
    audited_df.to_csv(os.path.join(OUTPUT_DIR, "audit_events.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "audit_summary.csv"), index=False)
    recall_sample_df.to_csv(os.path.join(OUTPUT_DIR, "recall_audit_sample.csv"), index=False)
    recall_summary_df.to_csv(os.path.join(OUTPUT_DIR, "recall_audit_summary.csv"), index=False)
    latency_df.to_csv(os.path.join(OUTPUT_DIR, "latency_summary.csv"), index=False)
    throughput_df.to_csv(os.path.join(OUTPUT_DIR, "throughput_scaling.csv"), index=False)

    simulator.generate_figures(audited_df, cycle_df, throughput_df)
    print_reports(summary_df, latency_df, recall_summary_df)
    print(f"\nPipeline complete. Outputs saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
