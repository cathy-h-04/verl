"""Baseline mid-run time vs total GPU power across iterations 28-29.

Single figure:
- x: time within iteration (s), reset per run-window
- y: total GPU power (W), summed across GPUs/ranks per timestamp
- lines: phase_name, averaged across selected runs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/baselines/phase_power_over_time.png")

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
GRID_DT_S = 0.1
TARGET_ITERATIONS = (28, 29)
PHASE_ORDER = ["rollout", "rl_policy", "training"]
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
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _mode_string(series: pd.Series) -> str:
    if series.empty:
        return "unknown"
    counts = series.astype(str).value_counts(dropna=False)
    return str(counts.index[0]) if len(counts) else "unknown"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    required = ["run_id", "policy", "model", "logical_run_group"]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    expected_slurm = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & slurm_job_mask & checkpoint_mask
    ][["run_id", "model_facet", "policy_norm", "slurm_job_name"]].drop_duplicates()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


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
    selected_runs = _select_baseline_runs()
    selected_run_ids = set(selected_runs["run_id"].astype(str))

    step_fact, _ = load_view("step_fact_view")
    eligible_steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    eligible_steps = apply_analysis_ok(eligible_steps)
    if "is_validation_step" in eligible_steps.columns:
        eligible_steps = eligible_steps[~eligible_steps["is_validation_step"].fillna(False)].copy()
    eligible_steps["global_step_canonical"] = pd.to_numeric(
        eligible_steps["global_step_canonical"], errors="coerce"
    )
    eligible_steps = eligible_steps[
        eligible_steps["global_step_canonical"].isin(TARGET_ITERATIONS)
    ][["run_id", "global_step_canonical"]].drop_duplicates()
    if eligible_steps.empty:
        raise ValueError("No analysis-valid baseline steps found for iterations 28-29.")

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
    for (run_id, step), g in df.groupby(["run_id", "global_step_canonical"], dropna=False):
        win = _resample_run_iteration_window(g, dt_s=GRID_DT_S)
        if not win.empty:
            window_frames.append(win)
    if not window_frames:
        raise ValueError("No resampled windows found for baseline steps 28-29.")

    all_windows = pd.concat(window_frames, ignore_index=True)
    iter_durations = (
        all_windows.groupby("iteration", as_index=False)["t_s"].max().rename(columns={"t_s": "duration_s"})
        .sort_values("iteration")
    )
    offsets: dict[int, float] = {}
    running_offset = 0.0
    gap_s = 0.6
    for row in iter_durations.itertuples(index=False):
        iteration = int(row.iteration)
        offsets[iteration] = running_offset
        running_offset += float(row.duration_s) + gap_s

    stitched_frames = []
    per_run_iter = (
        all_windows.merge(selected_runs[["run_id", "model_facet"]], on="run_id", how="left")
        .groupby(["model_facet", "iteration", "phase_name", "t_s"], as_index=False)["power_sum_w"]
        .mean()
        .sort_values(["model_facet", "iteration", "phase_name", "t_s"])
    )
    for row in per_run_iter.itertuples(index=False):
        stitched_frames.append(
            {
                "model_facet": row.model_facet,
                "iteration": int(row.iteration),
                "phase_name": row.phase_name,
                "x_s": float(row.t_s) + offsets[int(row.iteration)],
                "power_sum_w": float(row.power_sum_w),
            }
        )
    stitched = pd.DataFrame(stitched_frames).sort_values(["model_facet", "phase_name", "x_s"])

    print("selected run_ids:")
    print(selected_runs.sort_values(["model_facet", "policy_norm"]).to_string(index=False))
    print(f"selected iterations={list(TARGET_ITERATIONS)}")
    print(
        all_windows.groupby(["iteration", "run_id"], as_index=False)["t_s"]
        .max()
        .rename(columns={"t_s": "duration_s"})
        .sort_values(["iteration", "run_id"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, sharey=True)
    ymax = float(stitched["power_sum_w"].max()) if len(stitched) else 1.0

    boundaries = []
    for row in iter_durations.itertuples(index=False):
        iteration = int(row.iteration)
        start = offsets[iteration]
        end = start + float(row.duration_s)
        boundaries.append((iteration, start, end))

    for ax, model_facet in zip(axes, ["Llama", "Qwen"]):
        model_df = stitched[stitched["model_facet"] == model_facet]
        for phase in PHASE_ORDER:
            phase_df = model_df[model_df["phase_name"] == phase]
            if phase_df.empty:
                continue
            ax.plot(
                phase_df["x_s"],
                phase_df["power_sum_w"],
                linewidth=2.4,
                color=PHASE_COLORS[phase],
                label=PHASE_DISPLAY[phase],
            )
        if boundaries:
            ax.axvline(boundaries[0][1], color="0.75", linewidth=0.8, alpha=0.7)
        if len(boundaries) > 1:
            ax.axvline(boundaries[1][1], color="#D62728", linewidth=1.6, alpha=0.95)
        if boundaries:
            ax.axvline(boundaries[-1][2], color="0.75", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Total GPU Power (W)")
        ax.set_ylim(0.0, ymax * 1.08)
        ax.grid(alpha=0.2)
        ax.set_title(MODEL_DISPLAY[model_facet], fontweight="bold")

    axes[-1].set_xlabel("Time across iterations 28-29 (s)")
    handles = [
        plt.Line2D([0], [0], color=PHASE_COLORS[phase], lw=2.4, label=PHASE_DISPLAY[phase])
        for phase in PHASE_ORDER
    ]
    fig.legend(handles=handles, title="Phase", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("GPU Power by Phase and Model, Iterations 28-29", fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.86), h_pad=2.0)
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
