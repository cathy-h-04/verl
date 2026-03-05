"""Baseline mid-run time vs total GPU power, one line per phase.

Single figure with 2-3 stacked subplots (one per selected mid-run iteration):
- x: time within iteration (s), reset per run-window
- y: total GPU power (W), summed across GPUs/ranks per timestamp
- lines: phase_name (no line connecting across different phases)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier0/midrun_time_vs_gpu_power_by_phase_baselines.png")

MODEL_BASELINE_SLURM_NAMES = {"llama_new_baseline", "qwen_new_baseline"}
GRID_DT_S = 0.1  # 100ms grid for resampling/alignment
N_ITERS = 3


def _mode_string(series: pd.Series) -> str:
    if series.empty:
        return "unknown"
    x = series.astype(str).value_counts(dropna=False)
    return str(x.index[0]) if len(x) else "unknown"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    required_summary = ["run_id", "is_checkpoint_continuation", "join_coverage_rate", "phase_boundary_integrity_rate"]
    miss = [c for c in required_summary if c not in run_summary.columns]
    if miss:
        raise ValueError(f"run_summary_view missing required columns: {miss}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    df = run_summary.merge(runs[["run_id", "slurm_job_name"]], on="run_id", how="left", validate="one_to_one").copy()
    df["is_checkpoint_continuation"] = df["is_checkpoint_continuation"].fillna(False).astype(bool)
    df["join_coverage_rate"] = pd.to_numeric(df["join_coverage_rate"], errors="coerce")
    df["phase_boundary_integrity_rate"] = pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce")
    df["slurm_job_name"] = df["slurm_job_name"].astype(str).str.lower()
    mask = (
        (~df["is_checkpoint_continuation"])
        & (df["join_coverage_rate"] == 1.0)
        & (df["phase_boundary_integrity_rate"] == 1.0)
        & (df["slurm_job_name"].isin(MODEL_BASELINE_SLURM_NAMES))
    )
    selected = df.loc[mask, ["run_id", "slurm_job_name"]].drop_duplicates().copy()
    if selected.empty:
        raise ValueError("No baseline runs found for slurm_job_name in {llama_new_baseline, qwen_new_baseline}.")
    return selected


def _choose_mid_iterations(df: pd.DataFrame, selected_run_ids: set[str]) -> list[int]:
    iter_bounds = (
        df[df["run_id"].isin(selected_run_ids)]
        .groupby("run_id", as_index=False)["iteration"]
        .agg(iter_min="min", iter_max="max")
    )
    if iter_bounds.empty:
        raise ValueError("No iteration bounds available in periodic samples for selected baseline runs.")
    iter_bounds["mid"] = np.floor((iter_bounds["iter_min"] + iter_bounds["iter_max"]) / 2.0).astype(int)
    global_mid = int(np.floor(iter_bounds["mid"].median()))
    candidates = [global_mid - 1, global_mid, global_mid + 1]

    available = set(df["iteration"].dropna().astype(int).unique().tolist())
    selected = [k for k in candidates if k in available]
    if not selected:
        # Fallback: nearest available iterations around global_mid.
        all_iters = sorted(available)
        if not all_iters:
            raise ValueError("No valid iterations in periodic samples.")
        selected = sorted(all_iters, key=lambda x: abs(x - global_mid))[:N_ITERS]
        selected.sort()
    return selected[:N_ITERS]


def _resample_run_iteration_window(run_iter_df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    run_id = str(run_iter_df["run_id"].iloc[0])
    iteration = int(run_iter_df["iteration"].iloc[0])
    run_iter_df = run_iter_df.sort_values("ts_monotonic_ns").copy()

    t0_ns = int(run_iter_df["ts_monotonic_ns"].min())
    t1_ns = int(run_iter_df["ts_monotonic_ns"].max())
    if t1_ns <= t0_ns:
        return pd.DataFrame(columns=["run_id", "iteration", "t_s", "power_sum_w", "phase_name"])

    dt_ns = int(round(dt_s * 1e9))
    grid_ns = np.arange(t0_ns, t1_ns + dt_ns, dt_ns, dtype=np.int64)
    t_rel_s = (grid_ns - t0_ns) / 1e9

    sum_power = np.zeros_like(t_rel_s, dtype=float)
    n_series_used = 0

    # Interpolate each GPU/rank trace to common grid, then sum.
    for _, g in run_iter_df.groupby(["node", "rank", "gpu_index"], dropna=False):
        g = g.sort_values("ts_monotonic_ns")
        x = g["ts_monotonic_ns"].to_numpy(dtype=np.int64)
        y = g["gpu_power_w"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        interp = np.interp(grid_ns, x_unique, y_unique, left=np.nan, right=np.nan)
        valid = np.isfinite(interp)
        if valid.any():
            sum_power[valid] += interp[valid]
            n_series_used += 1

    if n_series_used == 0:
        return pd.DataFrame(columns=["run_id", "iteration", "t_s", "power_sum_w", "phase_name"])

    # Phase at grid timestamps from nearest raw timestamp mode label.
    phase_timeline = (
        run_iter_df.groupby("ts_monotonic_ns", as_index=False)["phase_name"]
        .agg(_mode_string)
        .sort_values("ts_monotonic_ns")
    )
    raw_ts = phase_timeline["ts_monotonic_ns"].to_numpy(dtype=np.int64)
    raw_phase = phase_timeline["phase_name"].astype(str).to_numpy()
    # nearest-neighbor map grid->phase
    idx_right = np.searchsorted(raw_ts, grid_ns, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(raw_ts) - 1)
    idx_right = np.clip(idx_right, 0, len(raw_ts) - 1)
    choose_right = np.abs(raw_ts[idx_right] - grid_ns) < np.abs(grid_ns - raw_ts[idx_left])
    nearest_idx = np.where(choose_right, idx_right, idx_left)
    phase_grid = raw_phase[nearest_idx]

    out = pd.DataFrame(
        {
            "run_id": run_id,
            "iteration": iteration,
            "t_s": t_rel_s,
            "power_sum_w": sum_power,
            "phase_name": phase_grid,
        }
    )
    out = out[np.isfinite(out["power_sum_w"])].copy()
    return out


def main() -> None:
    selected_runs = _select_baseline_runs()
    selected_run_ids = set(selected_runs["run_id"].astype(str).tolist())

    periodic, _ = load_view("hardware_periodic")
    required = ["run_id", "node", "rank", "gpu_index", "iteration", "ts_monotonic_ns", "phase_name", "gpu_power_mW", "record_type"]
    missing = [c for c in required if c not in periodic.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")

    df = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)][required].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df["gpu_power_mW"] = pd.to_numeric(df["gpu_power_mW"], errors="coerce")
    df = df.dropna(subset=["iteration", "ts_monotonic_ns", "gpu_power_mW", "phase_name"]).copy()
    df["iteration"] = df["iteration"].astype(int)
    df["ts_monotonic_ns"] = df["ts_monotonic_ns"].astype(np.int64)
    df["gpu_power_w"] = df["gpu_power_mW"] / 1000.0
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"] != "validation"].copy()

    selected_iters = _choose_mid_iterations(df, selected_run_ids)
    print(f"selected baseline run_count={len(selected_run_ids)}")
    print(f"selected iterations (mid-run)={selected_iters}")
    print("selected runs:")
    print(selected_runs.sort_values("run_id").to_string(index=False))

    # Build per run+iteration resampled windows.
    window_frames = []
    for run_id in sorted(selected_run_ids):
        run_df = df[df["run_id"].astype(str) == run_id]
        for k in selected_iters:
            rk = run_df[run_df["iteration"] == k]
            if rk.empty:
                continue
            win = _resample_run_iteration_window(rk, dt_s=GRID_DT_S)
            if not win.empty:
                window_frames.append(win)

    if not window_frames:
        raise ValueError("No resampled windows found for selected baseline runs and mid-run iterations.")

    all_windows = pd.concat(window_frames, ignore_index=True)
    all_windows["phase_name"] = all_windows["phase_name"].astype(str).str.lower()

    # Sanity logs requested.
    for k in selected_iters:
        kdf = all_windows[all_windows["iteration"] == k].copy()
        if kdf.empty:
            print(f"iteration={k}: no samples")
            continue
        duration_by_run = (
            kdf.groupby("run_id", as_index=False)["t_s"].max().rename(columns={"t_s": "duration_s"})
        )
        phases = sorted(kdf["phase_name"].dropna().unique().tolist())
        mean_power_phase = (
            kdf.groupby("phase_name", as_index=False)["power_sum_w"].mean().sort_values("phase_name")
        )
        print(
            f"iteration={k}: runs={duration_by_run['run_id'].nunique()}, "
            f"duration_s_mean={duration_by_run['duration_s'].mean():.3f}, "
            f"duration_s_min={duration_by_run['duration_s'].min():.3f}, "
            f"duration_s_max={duration_by_run['duration_s'].max():.3f}"
        )
        print(f"  phases={phases}")
        print("  mean_power_by_phase_W:")
        print(mean_power_phase.to_string(index=False))

    phase_order = sorted(all_windows["phase_name"].dropna().unique().tolist())
    phase_colors = {p: plt.cm.tab10(i % 10) for i, p in enumerate(phase_order)}

    fig, axes = plt.subplots(len(selected_iters), 1, figsize=(12, 3.6 * len(selected_iters)), sharex=False, sharey=True)
    if len(selected_iters) == 1:
        axes = [axes]

    global_ymax = float(all_windows["power_sum_w"].max()) if len(all_windows) else 1.0
    global_ymin = 0.0

    for ax, k in zip(axes, selected_iters):
        kdf = all_windows[all_windows["iteration"] == k].copy()
        if kdf.empty:
            ax.set_title(f"iteration {k} (no data)")
            continue

        # For each phase, aggregate across runs at matching resampled t_s.
        for phase in phase_order:
            p = kdf[kdf["phase_name"] == phase]
            if p.empty:
                continue
            line = p.groupby("t_s", as_index=False)["power_sum_w"].mean().sort_values("t_s")
            ax.plot(
                line["t_s"],
                line["power_sum_w"],
                linewidth=2.0,
                color=phase_colors[phase],
                label=phase,
            )

        ax.set_title(f"iteration {k}")
        ax.set_ylabel("Total GPU Power (W)")
        ax.grid(alpha=0.2)
        ax.set_ylim(global_ymin, global_ymax * 1.05)

    axes[-1].set_xlabel("Time within iteration (s)")
    handles = [plt.Line2D([0], [0], color=phase_colors[p], lw=2.0, label=p) for p in phase_order]
    fig.legend(handles=handles, title="phase_name", loc="upper center", ncol=min(5, max(1, len(handles))), frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Baseline: Time vs Total GPU Power (2-3 Mid-Run Iterations), Lines by Phase", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
