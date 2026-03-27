"""PPO GPU power over time by phase, faceted by scaling configuration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/phase_power_over_time.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")
GRID_DT_S = 0.1
TARGET_ITERATION = 29
PHASE_ORDER = ("rollout", "rl_policy", "training")
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
PHASE_COLORS = {
    "rollout": "#4C78A8",
    "rl_policy": "#54A24B",
    "training": "#F58518",
}
CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
CONFIG_DISPLAY = {
    "2xA100": "2x A100",
    "2xH200": "2x H200",
    "4xA100": "4x A100",
    "4xH200": "4x H200",
}


def _config_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_a100" in rid:
        return "2xA100"
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    if "4gpu_h200" in rid:
        return "4xH200"
    return "Unknown"


def _mode_string(series: pd.Series) -> str:
    if series.empty:
        return "unknown"
    counts = series.astype(str).value_counts(dropna=False)
    return str(counts.index[0]) if len(counts) else "unknown"


def _select_scaling_runs() -> pd.DataFrame:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")
    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(summary_df[["run_id", "policy"]], on="run_id", how="inner", validate="one_to_one")
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected = selected[selected["policy_norm"] == "ppo"].copy()
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])
    selected = selected[selected["config"].isin(CONFIG_ORDER)].copy()
    if selected.empty:
        raise ValueError("No scaling runs selected.")
    return selected[["run_id", "config"]].drop_duplicates()


def _resample_run_iteration_window(run_iter_df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    run_id = str(run_iter_df["run_id"].iloc[0])
    iteration = int(run_iter_df["global_step_canonical"].iloc[0])
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

    phase_timeline = (
        run_iter_df.groupby("ts_monotonic_ns", as_index=False)["phase_name"]
        .agg(_mode_string)
        .sort_values("ts_monotonic_ns")
    )
    raw_ts = phase_timeline["ts_monotonic_ns"].to_numpy(dtype=np.int64)
    raw_phase = phase_timeline["phase_name"].astype(str).to_numpy()
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
    return out[np.isfinite(out["power_sum_w"])].copy()


def main() -> None:
    selected_runs = _select_scaling_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    eligible_steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    eligible_steps = apply_analysis_ok(eligible_steps)
    if "is_validation_step" in eligible_steps.columns:
        eligible_steps = eligible_steps[~eligible_steps["is_validation_step"].fillna(False)].copy()
    eligible_steps["global_step_canonical"] = pd.to_numeric(eligible_steps["global_step_canonical"], errors="coerce")
    eligible_steps = eligible_steps[
        eligible_steps["global_step_canonical"] == TARGET_ITERATION
    ][["run_id", "global_step_canonical"]].drop_duplicates()
    if eligible_steps.empty:
        raise ValueError(f"No analysis-valid scaling steps found for iteration {TARGET_ITERATION}.")

    periodic, _ = load_view("hardware_periodic")
    required = [
        "run_id",
        "node",
        "rank",
        "gpu_index",
        "global_step_canonical",
        "ts_monotonic_ns",
        "phase_name",
        "gpu_power_mW",
        "record_type",
        "source",
    ]
    missing = [c for c in required if c not in periodic.columns]
    if missing:
        raise ValueError(f"hardware_periodic missing required columns: {missing}")

    df = periodic[periodic["run_id"].astype(str).isin(selected_run_ids)][required].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df = df.merge(eligible_steps, on=["run_id", "global_step_canonical"], how="inner")
    df["ts_monotonic_ns"] = pd.to_numeric(df["ts_monotonic_ns"], errors="coerce")
    df["gpu_power_mW"] = pd.to_numeric(df["gpu_power_mW"], errors="coerce")
    df = df.dropna(subset=["ts_monotonic_ns", "gpu_power_mW", "phase_name"]).copy()
    df["ts_monotonic_ns"] = df["ts_monotonic_ns"].astype(np.int64)
    df["gpu_power_w"] = df["gpu_power_mW"] / 1000.0
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(PHASE_ORDER)].copy()

    window_frames = []
    for (_, _), g in df.groupby(["run_id", "global_step_canonical"], dropna=False):
        win = _resample_run_iteration_window(g, dt_s=GRID_DT_S)
        if not win.empty:
            window_frames.append(win)
    if not window_frames:
        raise ValueError("No resampled iteration windows found for scaling runs.")

    all_windows = pd.concat(window_frames, ignore_index=True)
    all_windows = all_windows.merge(selected_runs, on="run_id", how="left", validate="many_to_one")
    stitched = (
        all_windows.groupby(["config", "phase_name", "t_s"], as_index=False)["power_sum_w"]
        .mean()
        .sort_values(["config", "phase_name", "t_s"])
    )

    print("selected run_ids by config:")
    print(selected_runs.sort_values(["config", "run_id"]).to_string(index=False))
    print(f"selected iteration={TARGET_ITERATION}")

    fig, axes = plt.subplots(len(CONFIG_ORDER), 1, figsize=(12.0, 13.2), sharex=True, sharey=True)
    ymax = float(stitched["power_sum_w"].max()) if len(stitched) else 1.0

    for ax, config in zip(axes, CONFIG_ORDER):
        config_df = stitched[stitched["config"] == config]
        for phase in PHASE_ORDER:
            phase_df = config_df[config_df["phase_name"] == phase]
            if phase_df.empty:
                continue
            ax.scatter(
                phase_df["t_s"],
                phase_df["power_sum_w"],
                s=10,
                alpha=0.85,
                color=PHASE_COLORS[phase],
                label=PHASE_DISPLAY[phase],
                edgecolors="none",
            )
        ax.set_title(CONFIG_DISPLAY[config], fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, ymax * 1.08)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Total GPU Power (W)")

    axes[-1].set_xlabel("Time (s)")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=PHASE_COLORS[phase], markersize=6, label=PHASE_DISPLAY[phase])
        for phase in PHASE_ORDER
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.96), fontsize=11)
    fig.suptitle(
        f"PPO GPU Power by Phase and Configuration (Iteration {TARGET_ITERATION})",
        fontweight="bold",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="phase_power_over_time",
        run_ids=selected_run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["hardware_periodic", "step_fact_view", "runs", "run_summary_view"],
            "target_iteration": TARGET_ITERATION,
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
