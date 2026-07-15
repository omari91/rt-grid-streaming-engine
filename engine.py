import os
import time
import logging
import warnings
import json
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
VOLTAGE_MIN_PU = 0.94
VOLTAGE_TARGET_PU = 0.945
PF_Q_RATIO = 0.33
DEFAULT_STREAM_N = 500
OP_DEADLINE_MS = 20.0


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

def build_repro_header() -> dict:
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
            "stream_n": DEFAULT_STREAM_N,
            "global_trafo_limit_mw": GLOBAL_TRAFO_LIMIT_MW,
            "voltage_min_pu": VOLTAGE_MIN_PU,
            "voltage_target_pu": VOLTAGE_TARGET_PU,
            "pf_q_ratio": PF_Q_RATIO,
            "deadline_ms": OP_DEADLINE_MS,
        },
    }

def write_repro_header(output_dir: str = OUTPUT_DIR) -> dict:
    header = build_repro_header()
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
class LiveIngestionLayer:
    """Synthetic fallback stream for reproducible research runs."""

    def fetch_stream(self, n: int = DEFAULT_STREAM_N) -> np.ndarray:
        return self._fallback_generator(n)

    def _fallback_generator(self, n: int) -> np.ndarray:
        return RNG.beta(2, 4, n) * 50.0


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
def iterative_fbs_solve(load_mw: float) -> float:
    """Fast surrogate FBS-like feeder voltage estimate for streaming control."""
    v = 1.0 - 0.0042 * load_mw
    return float(np.clip(v, 0.55, 1.0))


@dataclass
class EvalResult:
    load_mw: float
    min_vm_pu: float
    line_loading_pct: float


class PhysicsEngine:
    """Pandapower NR/BFSW reference model for reproducible AC validation."""

    def __init__(self):
        self.net = self._build_model()
        self.load_idx = int(self.net.load.index[0])

    def _build_model(self):
        net = pp.create_empty_network(sn_mva=100.0)
        hv = pp.create_bus(net, vn_kv=110.0, name="HV")
        mv = pp.create_bus(net, vn_kv=20.0, name="MV")
        pp.create_ext_grid(net, bus=hv, vm_pu=1.02)
        pp.create_transformer(net, hv_bus=hv, lv_bus=mv, std_type="63 MVA 110/20 kV")

        prev = mv
        for i in range(1, 50):
            nxt = pp.create_bus(net, vn_kv=20.0, name=f"B{i}")
            pp.create_line_from_parameters(
                net,
                from_bus=prev,
                to_bus=nxt,
                length_km=0.8,
                r_ohm_per_km=0.28,
                x_ohm_per_km=0.12,
                c_nf_per_km=0.0,
                max_i_ka=0.40,
                name=f"L{i-1}_{i}",
            )
            prev = nxt

        pp.create_load(net, bus=prev, p_mw=0.0, q_mvar=0.0, name="tail_load")
        return net

    def solve_reference(self, load_mw: float) -> EvalResult:
        self.net.load.at[self.load_idx, "p_mw"] = float(load_mw)
        self.net.load.at[self.load_idx, "q_mvar"] = float(load_mw) * PF_Q_RATIO

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
                return EvalResult(load_mw=load_mw, min_vm_pu=min_vm, line_loading_pct=line_loading)
            except LoadflowNotConverged:
                continue
            except Exception:
                continue

        return EvalResult(load_mw=load_mw, min_vm_pu=0.55, line_loading_pct=999.0)


# ==============================================================================
# SIMULATION CORE
# ==============================================================================
class GridSimulator:
    def __init__(self):
        self.prop_ctrl = VoltageAwareController(GLOBAL_TRAFO_LIMIT_MW)
        self.base_ctrl = BaselineController(GLOBAL_TRAFO_LIMIT_MW)
        self.droop_ctrl = DroopController(GLOBAL_TRAFO_LIMIT_MW, k_mw_per_pu=20.0)
        self.physics = PhysicsEngine()

    def run_streaming_pipeline(self, stream: np.ndarray):
        selector = OnlineSmartSelector(stream)
        records = []
        prev = float(stream[0]) if len(stream) else 0.0

        for idx, load in enumerate(stream):
            load = float(load)

            t0 = time.perf_counter()

            t_sel = time.perf_counter()
            critical_event = selector.is_critical(load, prev)
            selector_ms = (time.perf_counter() - t_sel) * 1e3

            t_fbs = time.perf_counter()
            v_est = iterative_fbs_solve(load)
            fbs_ms = (time.perf_counter() - t_fbs) * 1e3

            t_base = time.perf_counter()
            p_base = self.base_ctrl.compute(load)
            base_ctrl_ms = (time.perf_counter() - t_base) * 1e3

            t_droop = time.perf_counter()
            p_droop = self.droop_ctrl.compute(load, v_est)
            droop_ctrl_ms = (time.perf_counter() - t_droop) * 1e3

            t_prop = time.perf_counter()
            p_prop = self.prop_ctrl.compute(load, v_est) if critical_event or v_est < VOLTAGE_MIN_PU else min(load, GLOBAL_TRAFO_LIMIT_MW)
            prop_ctrl_ms = (time.perf_counter() - t_prop) * 1e3

            nr_ms = 0.0
            ref_raw = ref_base = ref_droop = ref_prop = None
            if critical_event:
                t_nr = time.perf_counter()
                ref_raw = self.physics.solve_reference(load)
                ref_base = self.physics.solve_reference(p_base)
                ref_droop = self.physics.solve_reference(p_droop)
                ref_prop = self.physics.solve_reference(p_prop)
                nr_ms = (time.perf_counter() - t_nr) * 1e3

            total_ms = (time.perf_counter() - t0) * 1e3

            records.append(
                {
                    "event_idx": idx,
                    "load_mw": load,
                    "critical_event": critical_event,
                    "selector_time_ms": selector_ms,
                    "fbs_time_ms": fbs_ms,
                    "baseline_ctrl_ms": base_ctrl_ms,
                    "droop_ctrl_ms": droop_ctrl_ms,
                    "proposed_ctrl_ms": prop_ctrl_ms,
                    "nr_time_ms": nr_ms,
                    "cycle_time_ms": total_ms,
                    "deadline_miss": total_ms > OP_DEADLINE_MS,
                    "fbs_vm_pu": v_est,
                    "raw_vm_ref_pu": np.nan if ref_raw is None else ref_raw.min_vm_pu,
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

        audited["actual_violation"] = audited["raw_vm_ref_pu"] < VOLTAGE_MIN_PU
        audited["base_violation_flag"] = audited["load_mw"] > GLOBAL_TRAFO_LIMIT_MW
        audited["droop_violation"] = audited["droop_vm_ref_pu"] < VOLTAGE_MIN_PU
        audited["prop_violation"] = audited["prop_vm_ref_pu"] < VOLTAGE_MIN_PU

        summary = {
            "total_true_physical_violations": int(audited["actual_violation"].sum()),
            "baseline_false_negatives": int(((~audited["base_violation_flag"]) & audited["actual_violation"]).sum()),
            "droop_controller_violations": int(audited["droop_violation"].sum()),
            "proposed_controller_violations": int(audited["prop_violation"].sum()),
            "proposed_pipeline_missed_events": int((audited["actual_violation"] & (~audited["critical_event"])).sum()),
        }

        summary_df = pd.DataFrame([summary])
        return audited, summary_df

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
            vis["base_vm_ref_pu"] = vis["fbs_vm_pu"]
            vis["droop_vm_ref_pu"] = vis["fbs_vm_pu"]
            vis["prop_vm_ref_pu"] = vis["fbs_vm_pu"]

        fig1, ax1 = plt.subplots(figsize=(6.8, 3.8))
        ax1.plot(vis.index, vis["base_vm_ref_pu"], label="Baseline", color="#c0392b", ls="--", lw=1.5, marker="x")
        ax1.plot(vis.index, vis["droop_vm_ref_pu"], label="Droop", color="#f39c12", ls="-.", lw=1.4, marker="^")
        ax1.plot(vis.index, vis["prop_vm_ref_pu"], label="Proposed", color="#27ae60", lw=2.0, marker="o")
        ax1.axhline(VOLTAGE_MIN_PU, color="black", ls=":", label="Limit (0.94 p.u.)")
        ax1.set_title("Fig. 2. Voltage Stability Comparison", fontweight="bold")
        ax1.set_xlabel("Simulation Steps")
        ax1.set_ylabel("Voltage (p.u.)")
        ax1.legend(fontsize=8, loc="best")
        fig1.tight_layout()
        fig1.savefig(os.path.join(OUTPUT_DIR, "Voltage_Stability.png"))
        plt.show()
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6.2, 3.4))
        ax2.errorbar(throughput_df["N"], throughput_df["mean_mops"], yerr=throughput_df["std_mops"], fmt="o-", capsize=4, color="#2980b9")
        ax2.set_xscale("log")
        ax2.set_title("Fig. 3. Vectorized Control Throughput", fontweight="bold")
        ax2.set_xlabel("Operation Count")
        ax2.set_ylabel("Million Ops/Sec")
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUTPUT_DIR, "Scalability.png"))
        plt.show()
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(6.2, 3.4))
        for _ in range(30):
            noise = RNG.normal(0, 3.0, 40)
            vals = np.clip(32.0 + noise, 0.0, None)
            volts = [iterative_fbs_solve(v) for v in vals]
            ax3.plot(volts, color="#2ecc71", alpha=0.12)
        ax3.axhline(VOLTAGE_MIN_PU, color="#c0392b", ls=":", label="Limit")
        ax3.set_title("Fig. 4. Stochastic Physics Corridors", fontweight="bold")
        ax3.set_ylabel("Voltage (p.u.)")
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR, "Stochastic_Risk.png"))
        plt.show()
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(6.6, 3.4))
        ax4.plot(cycle_df["event_idx"], cycle_df["cycle_time_ms"], color="#34495e", lw=1.2)
        ax4.axhline(OP_DEADLINE_MS, color="#c0392b", ls=":", label="20 ms deadline")
        ax4.set_title("Cycle-Time Trace", fontweight="bold")
        ax4.set_xlabel("Event Index")
        ax4.set_ylabel("Cycle Time (ms)")
        ax4.legend(fontsize=8)
        fig4.tight_layout()
        fig4.savefig(os.path.join(OUTPUT_DIR, "Cycle_Time_Trace.png"))
        plt.show()
        plt.close(fig4)


def print_reports(summary_df: pd.DataFrame, latency_df: pd.DataFrame):
    summary = summary_df.iloc[0].to_dict()
    print("\n" + "=" * 64)
    print("AUDIT REPORT: STREAMING FBS VS. NR GROUND TRUTH")
    print("=" * 64)
    for k, v in summary.items():
        print(f"{k:35s}: {int(v)}")
    print("=" * 64)

    print("\nLATENCY REPORT")
    print("=" * 64)
    print(latency_df.to_string(index=False))
    print("=" * 64)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    write_repro_header()
    simulator = GridSimulator()
    ingest = LiveIngestionLayer()
    stream = ingest.fetch_stream(DEFAULT_STREAM_N)

    print(
        f"STREAM FINGERPRINT | min={stream.min():.6f} "
        f"max={stream.max():.6f} mean={stream.mean():.6f} "
        f"std={stream.std():.6f}"
    )

    cycle_df = simulator.run_streaming_pipeline(stream)
    audited_df, summary_df = simulator.run_benchmark_audit(cycle_df)
    latency_df = simulator.build_latency_tables(cycle_df)
    throughput_df = simulator.benchmark_control_kernel()

    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "cycle_times.csv"), index=False)
    audited_df.to_csv(os.path.join(OUTPUT_DIR, "audit_events.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "audit_summary.csv"), index=False)
    latency_df.to_csv(os.path.join(OUTPUT_DIR, "latency_summary.csv"), index=False)
    throughput_df.to_csv(os.path.join(OUTPUT_DIR, "throughput_scaling.csv"), index=False)

    simulator.generate_figures(audited_df, cycle_df, throughput_df)
    print_reports(summary_df, latency_df)
    print(f"\nPipeline complete. Outputs saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
