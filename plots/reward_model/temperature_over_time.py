"""GPU temperature over time (iterations 28-29) for reward-mechanism runs.

Replaces transition-window view with an over-time stitched-iteration view.
- x: time across iterations 28-29 (s)
- y: mean GPU temperature across GPUs (°C)
- panels: policy
- lines: reward mechanism (reward function vs reward model)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/reward_model/temperature_over_time.png")
GRID_DT_S = 0.1
TARGET_ITERATIONS = (28, 29)

TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
POLICY_COLOR = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
EXPERIMENT_LINESTYLE = {
    "Llama Reward Function": "-",
    "Llama Reward Model": "--",
}
EXPERIMENT_ALPHA = {
    "Llama Reward Function": 0.5,
    "Llama Reward Model": 1.0,
}


def _experiment_facet(slurm_job_name: str, logical_run_group: str) -> str:
    slurm_text = str(slurm_job_name).strip().lower()
    logical_text = str(logical_run_group).strip().lower()
    for facet in TARGET_EXPERIMENT_FACETS:
        expected_slurm = TARGET_SLURM_JOB_NAME_BY_FACET[facet]
        logical_prefixes = LOGICAL_GROUP_PREFIXES_BY_FACET[facet]
        if slurm_text == expected_slurm and logical_text.startswith(logical_prefixes):
            return facet
    return "Other"


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["experiment_facet"] = [
        _experiment_facet(slurm_job_name=slurm_job_name, logical_run_group=logical_run_group_value)
        for slurm_job_name, logical_run_group_value in zip(runs_df["slurm_job_name"], runs_df["logical_run_group"])
    ]

    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["experiment_facet"].isin(TARGET_EXPERIMENT_FACETS)
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )
    integrity_mask = (
        (pd.to_numeric(runs_df["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(runs_df["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        if {"join_coverage_rate", "phase_boundary_integrity_rate"}.issubset(runs_df.columns)
        else True
    )

    selected = runs_df[non_rollout_knob_mask & target_mask & checkpoint_mask & integrity_mask][
        ["run_id", "experiment_facet", "policy_norm"]
    ].drop_duplicates()
    if selected.empty:
        raise ValueError("No reward-mechanism runs selected.")
    return selected


def _resample_run_iteration_window(run_iter_df: pd.DataFrame, metric_col: str, dt_s: float) -> pd.DataFrame:
    run_id = str(run_iter_df["run_id"].iloc[0])
    iteration = int(run_iter_df["global_step_canonical"].iloc[0])
    run_iter_df = run_iter_df.sort_values("ts_monotonic_ns").copy()

    t0_ns = int(run_iter_df["ts_monotonic_ns"].min())
    t1_ns = int(run_iter_df["ts_monotonic_ns"].max())
    if t1_ns <= t0_ns:
        return pd.DataFrame(columns=["run_id", "iteration", "t_s", "metric_mean"])

    dt_ns = int(round(dt_s * 1e9))
    grid_ns = np.arange(t0_ns, t1_ns + dt_ns, dt_ns, dtype=np.int64)
    t_rel_s = (grid_ns - t0_ns) / 1e9

    stacked = []
    for _, g in run_iter_df.groupby(["node", "rank", "gpu_index"], dropna=False):
        g = g.sort_values("ts_monotonic_ns")
        x = g["ts_monotonic_ns"].to_numpy(dtype=np.int64)
        y = g[metric_col].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        interp = np.interp(grid_ns, x_unique, y_unique, left=np.nan, right=np.nan)
        stacked.append(interp)

    if not stacked:
        return pd.DataFrame(columns=["run_id", "iteration", "t_s", "metric_mean"])

    mat = np.vstack(stacked)
    valid_cols = np.isfinite(mat).any(axis=0)
    metric_mean = np.full(mat.shape[1], np.nan, dtype=float)
    if valid_cols.any():
        metric_mean[valid_cols] = np.nanmean(mat[:, valid_cols], axis=0)
    out = pd.DataFrame(
        {
            "run_id": run_id,
            "iteration": iteration,
            "t_s": t_rel_s,
            "metric_mean": metric_mean,
        }
    )
    return out[np.isfinite(out["metric_mean"])].copy()


def main() -> None:
    selected_runs = _select_runs()
    selected_run_ids = set(selected_runs["run_id"].astype(str))

    step_fact, _ = load_view("step_fact_view")
    eligible_steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    eligible_steps = apply_analysis_ok(eligible_steps)
    if "is_validation_step" in eligible_steps.columns:
        eligible_steps = eligible_steps[~eligible_steps["is_validation_step"].fillna(False)].copy()
    eligible_steps["global_step_canonical"] = pd.to_numeric(eligible_steps["global_step_canonical"], errors="coerce")
    eligible_steps = eligible_steps[
        eligible_steps["global_step_canonical"].isin(TARGET_ITERATIONS)
    ][["run_id", "global_step_canonical"]].drop_duplicates()
    if eligible_steps.empty:
        raise ValueError("No analysis-valid reward-model steps found for iterations 28-29.")

    periodic, _ = load_view("hardware_periodic")
    required = [
        "run_id",
        "node",
        "rank",
        "gpu_index",
        "global_step_canonical",
        "ts_monotonic_ns",
        "temp_gpu_C",
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
    df["temp_gpu_C"] = pd.to_numeric(df["temp_gpu_C"], errors="coerce")
    df = df.dropna(subset=["ts_monotonic_ns", "temp_gpu_C"]).copy()
    df["ts_monotonic_ns"] = df["ts_monotonic_ns"].astype(np.int64)

    window_frames = []
    for (_run_id, _step), g in df.groupby(["run_id", "global_step_canonical"], dropna=False):
        win = _resample_run_iteration_window(g, metric_col="temp_gpu_C", dt_s=GRID_DT_S)
        if not win.empty:
            window_frames.append(win)
    if not window_frames:
        raise ValueError("No resampled windows found for reward-model steps 28-29.")

    all_windows = pd.concat(window_frames, ignore_index=True)
    iter_durations = (
        all_windows.groupby("iteration", as_index=False)["t_s"].max().rename(columns={"t_s": "duration_s"}).sort_values("iteration")
    )

    offsets: dict[int, float] = {}
    running_offset = 0.0
    gap_s = 0.6
    for row in iter_durations.itertuples(index=False):
        iteration = int(row.iteration)
        offsets[iteration] = running_offset
        running_offset += float(row.duration_s) + gap_s

    stitched = (
        all_windows.merge(selected_runs[["run_id", "experiment_facet", "policy_norm"]], on="run_id", how="left")
        .groupby(["experiment_facet", "policy_norm", "iteration", "t_s"], as_index=False)["metric_mean"]
        .mean()
    )
    stitched["x_s"] = stitched.apply(lambda r: float(r["t_s"]) + offsets[int(r["iteration"])], axis=1)

    print("selected reward-model run_ids:")
    print(selected_runs.sort_values(["experiment_facet", "policy_norm"]).to_string(index=False))
    print(f"selected iterations={list(TARGET_ITERATIONS)}")

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.1), sharex=True, sharey=True)
    ymax = float(stitched["metric_mean"].max()) if len(stitched) else 1.0
    ymin = float(stitched["metric_mean"].min()) if len(stitched) else 0.0
    pad = max(1.0, (ymax - ymin) * 0.08)

    boundaries = []
    for row in iter_durations.itertuples(index=False):
        iteration = int(row.iteration)
        start = offsets[iteration]
        end = start + float(row.duration_s)
        boundaries.append((iteration, start, end))

    for ax, policy in zip(axes, TARGET_POLICIES):
        sub = stitched[stitched["policy_norm"] == policy].copy()
        for facet in TARGET_EXPERIMENT_FACETS:
            fsub = sub[sub["experiment_facet"] == facet].sort_values("x_s")
            if fsub.empty:
                continue
            ax.plot(
                fsub["x_s"],
                fsub["metric_mean"],
                color=POLICY_COLOR[policy],
                linestyle=EXPERIMENT_LINESTYLE[facet],
                linewidth=2.3,
                alpha=EXPERIMENT_ALPHA[facet],
                label=EXPERIMENT_DISPLAY[facet] if policy == TARGET_POLICIES[0] else None,
            )

        if boundaries:
            ax.axvline(boundaries[0][1], color="0.75", linewidth=0.8, alpha=0.7)
        if len(boundaries) > 1:
            ax.axvline(boundaries[1][1], color="#D62728", linewidth=1.3, alpha=0.95)
        if boundaries:
            ax.axvline(boundaries[-1][2], color="0.75", linewidth=0.8, alpha=0.7)

        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.set_ylabel("GPU Temp (°C)")
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time across iterations 28-29 (s)")
    for ax in axes:
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color="#333333", linestyle=EXPERIMENT_LINESTYLE[facet], lw=2.3, label=EXPERIMENT_DISPLAY[facet])
        for facet in TARGET_EXPERIMENT_FACETS
    ]
    fig.legend(handles=handles, title="Reward Mechanism", frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("GPU Temperature Over Time by Policy and Reward Mechanism (Iterations 28-29)", fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=1.5)

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
