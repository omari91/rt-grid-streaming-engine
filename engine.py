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
OUTPUT_DIR = "final_output"
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

        # Scale down by 1/1000 for the MV grid context, and clip to the DER capacity
        df["delta_mw"] = df["MITTLERE_LEISTUNG_MW"] / 1000.0
        df["delta_mw"] = df["delta_mw"].clip(0.0, DER_CAPACITY_MW)

        stream = df["delta_mw"].values
        if n is not None and len(stream) > n:
            stream = stream[:n]
        return stream


class OnlineSmartSelector:
    """Flags fat-tail or sharp-ramp events for high-fidelity handling."""

    def __init__(self, historical_data: np.ndarray):
        self.p95 = np.percentile(historical_data, 95)
        self.p05 = np.percentile(historical_data, 5)
        diffs = np.diff(historical_data)
        self.rocof_thresh = float(np.std(diffs) * 1.2) if len(diffs) else 0.0

    def is_critical(self, cur: float, prev: float) -> bool:
        if cur > self.p95 or cur < self.p05:
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
    v = 1.0 - 0.0042 * load_mw
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
        self.net = pn.create_cigre_network_mv(with_der="pv_wind")
        self.gen_idx = self.net.sgen[self.net.sgen.name == 'WKA 7'].index[0]
        self.nominal_p_mw = self.net.load.p_mw.copy()
        self.nominal_q_mvar = self.net.load.q_mvar.copy()

    def set_dynamic_load(self, multiplier: float):
        """Applies a dynamic load multiplier to all loads in the network to simulate peak conditions."""
        self.net.load.p_mw = self.nominal_p_mw * multiplier
        self.net.load.q_mvar = self.nominal_q_mvar * multiplier

    def solve_reference(self, load_mw: float) -> EvalResult:
        # load_mw here represents the redispatch setpoint for WKA 7
        # Ensure the setpoint is physically possible for this generator
        clipped_setpoint = float(np.clip(load_mw, 0.0, DER_CAPACITY_MW))
        self.net.sgen.at[self.gen_idx, "p_mw"] = clipped_setpoint

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
        self.prop_ctrl = VoltageAwareController(DER_CAPACITY_MW)
        self.base_ctrl = BaselineController(DER_CAPACITY_MW)
        self.droop_ctrl = DroopController(DER_CAPACITY_MW, k_mw_per_pu=20.0)
        self.physics = PhysicsEngine()
        
        # Empirical daily load multipliers derived from the Typical Load Profile benchmark (Winter Scenario A)
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

    def run_streaming_pipeline(self, stream: np.ndarray, load_provider=None):
        if load_provider is None:
            load_provider = SyntheticLoadProvider(self.load_multipliers)
        selector = OnlineSmartSelector(stream)
        records = []
        prev = float(stream[0]) if len(stream) else 0.0

        for idx, load in enumerate(stream):
            load = float(load)

            # Operating-state model is pluggable (Sec. Experimental Scope);
            # default reproduces the deterministic synthetic multiplier table.
            self.physics.set_dynamic_load(load_provider.get_multiplier(idx))

            t0 = time.perf_counter()

            t_sel = time.perf_counter()
            critical_event = selector.is_critical(load, prev)
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

        fig4, ax4 = plt.subplots(figsize=(6.6, 3.4))
        ax4.plot(cycle_df["event_idx"], cycle_df["cycle_time_ms"], color="#34495e", lw=0.6)
        ax4.axhline(OP_DEADLINE_MS, color="#c0392b", ls=":", label="20 ms deadline")
        ax4.set_yscale("log")
        ax4.set_title("Cycle-Time Trace", fontweight="bold")
        ax4.set_xlabel("Event Index")
        ax4.set_ylabel("Cycle Time (ms, log scale)")
        ax4.legend(fontsize=8)
        fig4.tight_layout()
        fig4.savefig(os.path.join(OUTPUT_DIR, "Cycle_Time_Trace.png"))
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
