"""Task-imbalance decomposition for task runs (association view).

If rollout CPU/DRAM power is roughly stable, then:

    cpu_dram_j_per_token = cpu_dram_power_w / throughput_tokens_s

This visual tests whether higher task imbalance (straggler ratio) is associated
with lower throughput and higher CPU/DRAM energy intensity.

This figure shows, per policy and dataset:
- straggler ratio vs throughput (tokens/s)
- straggler ratio vs rollout CPU/DRAM power (W)
- straggler ratio vs rollout CPU/DRAM energy intensity (J/token)

Token correction mirrors existing task analyses:
- PPO/GRPO denominator = logged rollout total tokens
- ReMax denominator = rollout total + prompt + output*(gen_max/gen)
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
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper
from plots.task.line_style import DATASET_ALPHAS, DATASET_COLORS, DATASET_LINESTYLES


OUTPATH = Path("plots/out/task/orchestration_tax_straggler.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
TARGET_SLURM_TO_DATASET = {
    "llama_rm_gsm8k": "gsm8k",
    "llama_rm_rlhf": "full-hh-rlhf",
}
DATASET_ORDER = ("gsm8k", "full-hh-rlhf")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    required = ["run_id", "policy", "logical_run_group"]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    df = run_summary.merge(runs[["run_id", "slurm_job_name"]], on="run_id", how="left", validate="one_to_one").copy()
    df["policy_norm"] = df["policy"].astype(str).str.lower()
    df["slurm_job_name_lc"] = df["slurm_job_name"].astype(str).str.lower()
    df["dataset_group"] = df["slurm_job_name_lc"].map(TARGET_SLURM_TO_DATASET)
    logical_group = df["logical_run_group"].astype(str).str.lower()

    target_mask = df["policy_norm"].isin(TARGET_POLICIES) & df["dataset_group"].isin(DATASET_ORDER)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    checkpoint_mask = (
        ~df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in df.columns
        else True
    )
    integrity_mask = (
        (pd.to_numeric(df["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        if {"join_coverage_rate", "phase_boundary_integrity_rate"}.issubset(df.columns)
        else True
    )

    selected = df[target_mask & non_rollout_knob_mask & checkpoint_mask & integrity_mask].copy()
    if selected.empty:
        raise ValueError("No target runs selected for orchestration tax analysis.")

    expected = {(d, p) for d in DATASET_ORDER for p in TARGET_POLICIES}
    observed = set(zip(selected["dataset_group"], selected["policy_norm"]))
    missing_pairs = expected - observed
    if missing_pairs:
        raise ValueError(f"Missing dataset-policy combinations: {sorted(missing_pairs)}")

    return selected[["run_id", "policy_norm", "dataset_group"]].drop_duplicates()


def _load_remax_gen_ratio(run_ids: list[str]) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame(columns=["run_id", "global_step_canonical", "genmax_over_gen"])

    pt, _ = load_view("phase_timings_long")
    df = pt[
        pt["run_id"].astype(str).isin(run_ids)
        & pt["subphase_name"].isin(["gen", "gen_max"])
        & pt["metric_unit"].astype(str).eq("s")
    ][["run_id", "global_step_canonical", "subphase_name", "value"]].copy()

    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["global_step_canonical", "value"]).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)

    piv = (
        df.pivot_table(index=["run_id", "global_step_canonical"], columns="subphase_name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    piv["genmax_over_gen"] = piv["gen_max"] / piv["gen"]
    piv.loc[~np.isfinite(piv["genmax_over_gen"]), "genmax_over_gen"] = np.nan
    return piv[["run_id", "global_step_canonical", "genmax_over_gen"]]


def _load_step_side(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step, _ = load_view("step_fact_view")
    needed = [
        "run_id",
        "global_step_canonical",
        "step_rollout_total_tokens",
        "step_rollout_output_tokens",
        "straggler_ratio",
    ]
    missing = [c for c in needed if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    out = step[step["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    before = out.copy()
    out = apply_analysis_ok(out)
    print(f"step filtering={explain_filtering(before, out)}")

    for col in ["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "straggler_ratio"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "straggler_ratio"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)

    out = out[(out["step_rollout_total_tokens"] > 0) & np.isfinite(out["straggler_ratio"]) & (out["straggler_ratio"] >= 0)].copy()
    out["step_rollout_prompt_tokens"] = out["step_rollout_total_tokens"] - out["step_rollout_output_tokens"]

    out = out.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    remax_ids = selected_runs[selected_runs["policy_norm"] == "remax"]["run_id"].astype(str).tolist()
    remax_ratio = _load_remax_gen_ratio(remax_ids)
    out = out.merge(remax_ratio, on=["run_id", "global_step_canonical"], how="left")
    out["genmax_over_gen"] = pd.to_numeric(out["genmax_over_gen"], errors="coerce")

    run_medians = out.groupby("run_id")["genmax_over_gen"].median().to_dict()
    out["genmax_over_gen"] = out.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    out["corrected_rollout_total_tokens"] = out["step_rollout_total_tokens"]
    remax_mask = out["policy_norm"] == "remax"
    out.loc[remax_mask, "corrected_rollout_total_tokens"] = (
        out.loc[remax_mask, "step_rollout_total_tokens"]
        + out.loc[remax_mask, "step_rollout_prompt_tokens"]
        + out.loc[remax_mask, "step_rollout_output_tokens"] * out.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )

    out = out[out["corrected_rollout_total_tokens"] > 0].copy()
    return out[["run_id", "global_step_canonical", "policy_norm", "dataset_group", "straggler_ratio", "corrected_rollout_total_tokens"]]


def _load_rollout_cpu_dram(selected_runs: pd.DataFrame) -> pd.DataFrame:
    phase, _ = load_view("phase_fact_view")
    needed = ["run_id", "global_step_canonical", "phase_name", "cpu_dram_energy_j", "phase_time_s"]
    missing = [c for c in needed if c not in phase.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    out = phase[phase["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    before = out.copy()
    out = apply_analysis_ok(out)
    print(f"phase filtering={explain_filtering(before, out)}")

    out["phase_name"] = out["phase_name"].astype(str).str.lower()
    out = out[out["phase_name"] == "rollout"].copy()
    out["global_step_canonical"] = pd.to_numeric(out["global_step_canonical"], errors="coerce")
    out["cpu_dram_energy_j"] = pd.to_numeric(out["cpu_dram_energy_j"], errors="coerce")
    out["phase_time_s"] = pd.to_numeric(out["phase_time_s"], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "cpu_dram_energy_j", "phase_time_s"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out = out[np.isfinite(out["cpu_dram_energy_j"]) & np.isfinite(out["phase_time_s"])].copy()
    out = out[(out["cpu_dram_energy_j"] >= 0) & (out["phase_time_s"] > 0)].copy()

    return out[["run_id", "global_step_canonical", "cpu_dram_energy_j", "phase_time_s"]]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.unique(x).size < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def main() -> None:
    selected_runs = _select_runs()
    print("selected runs:")
    print(selected_runs.sort_values(["dataset_group", "policy_norm", "run_id"]).to_string(index=False))

    step_df = _load_step_side(selected_runs)
    phase_df = _load_rollout_cpu_dram(selected_runs)

    plot_df = step_df.merge(phase_df, on=["run_id", "global_step_canonical"], how="inner", validate="one_to_one")
    plot_df["throughput_corrected_tokens_s"] = plot_df["corrected_rollout_total_tokens"] / plot_df["phase_time_s"]
    plot_df["cpu_dram_power_w"] = plot_df["cpu_dram_energy_j"] / plot_df["phase_time_s"]
    plot_df["cpu_dram_j_per_corrected_token"] = plot_df["cpu_dram_energy_j"] / plot_df["corrected_rollout_total_tokens"]
    for c in ["throughput_corrected_tokens_s", "cpu_dram_power_w", "cpu_dram_j_per_corrected_token"]:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.dropna(subset=["throughput_corrected_tokens_s", "cpu_dram_power_w", "cpu_dram_j_per_corrected_token"]).copy()
    plot_df = plot_df[
        (plot_df["throughput_corrected_tokens_s"] > 0)
        & (plot_df["cpu_dram_power_w"] >= 0)
        & (plot_df["cpu_dram_j_per_corrected_token"] >= 0)
    ].copy()

    if plot_df.empty:
        raise ValueError("No rows left after joining and cleaning.")

    rows_used = plot_df.groupby(["policy_norm", "dataset_group"], dropna=False).size().rename("n_rows").reset_index()
    print("\nrows used by policy,dataset:")
    print(rows_used.sort_values(["policy_norm", "dataset_group"]).to_string(index=False))

    run_level = (
        plot_df.groupby(["run_id", "policy_norm", "dataset_group"], dropna=False)
        .agg(
            straggler_ratio_mean=("straggler_ratio", "mean"),
            throughput_mean=("throughput_corrected_tokens_s", "mean"),
            cpu_dram_power_mean=("cpu_dram_power_w", "mean"),
            cpu_dram_jpt_mean=("cpu_dram_j_per_corrected_token", "mean"),
        )
        .reset_index()
    )
    print("\nrun-level means:")
    print(run_level.sort_values(["dataset_group", "policy_norm", "run_id"]).to_string(index=False))

    per_policy_dataset = []
    for policy in TARGET_POLICIES:
        for dataset in DATASET_ORDER:
            sub = plot_df[(plot_df["policy_norm"] == policy) & (plot_df["dataset_group"] == dataset)].copy()
            x = sub["straggler_ratio"].to_numpy(dtype=float)
            y = sub["cpu_dram_j_per_corrected_token"].to_numpy(dtype=float)
            fit = _fit_line(x, y)
            corr = _safe_corr(x, y)
            per_policy_dataset.append(
                {
                    "policy_norm": policy,
                    "dataset_group": dataset,
                    "n_rows": int(sub.shape[0]),
                    "straggler_ratio_mean": float(np.nanmean(x)) if x.size else float("nan"),
                    "cpu_dram_jpt_mean": float(np.nanmean(y)) if y.size else float("nan"),
                    "slope_jpt_per_straggler": float(fit[0]) if fit is not None else float("nan"),
                    "pearson_r": corr,
                }
            )
    stats_df = pd.DataFrame(per_policy_dataset)
    print("\npolicy,dataset trend stats (smoking-gun table):")
    print(stats_df.sort_values(["policy_norm", "dataset_group"]).to_string(index=False))

    decomposition_stats = []
    for policy in TARGET_POLICIES:
        for dataset in DATASET_ORDER:
            sub = plot_df[(plot_df["policy_norm"] == policy) & (plot_df["dataset_group"] == dataset)].copy()
            if sub.empty:
                continue
            x = sub["straggler_ratio"].to_numpy(dtype=float)
            y_thr = sub["throughput_corrected_tokens_s"].to_numpy(dtype=float)
            y_pow = sub["cpu_dram_power_w"].to_numpy(dtype=float)
            y_jpt = sub["cpu_dram_j_per_corrected_token"].to_numpy(dtype=float)

            fit_thr = _fit_line(x, y_thr)
            fit_pow = _fit_line(x, y_pow)
            fit_jpt = _fit_line(x, y_jpt)
            decomposition_stats.append(
                {
                    "policy_norm": policy,
                    "dataset_group": dataset,
                    "n_rows": int(sub.shape[0]),
                    "throughput_slope": float(fit_thr[0]) if fit_thr is not None else float("nan"),
                    "throughput_r": _safe_corr(x, y_thr),
                    "power_slope_w_per_straggler": float(fit_pow[0]) if fit_pow is not None else float("nan"),
                    "power_r": _safe_corr(x, y_pow),
                    "jpt_slope": float(fit_jpt[0]) if fit_jpt is not None else float("nan"),
                    "jpt_r": _safe_corr(x, y_jpt),
                    "power_cv": float(np.nanstd(y_pow) / np.nanmean(y_pow)) if np.nanmean(y_pow) > 0 else float("nan"),
                }
            )

    decomp_df = pd.DataFrame(decomposition_stats)
    print("\ndecomposition stats by policy,dataset:")
    print(decomp_df.sort_values(["policy_norm", "dataset_group"]).to_string(index=False))

    fig, axes = plt.subplots(2, len(TARGET_POLICIES), figsize=(18.2, 8.7), sharex=True)
    if len(TARGET_POLICIES) == 1:
        axes = np.array(axes).reshape(2, 1)

    metric_rows = [
        ("throughput_corrected_tokens_s", "Throughput (tokens/s)"),
        ("cpu_dram_j_per_corrected_token", "CPU/DRAM Energy Intensity (J/token)"),
    ]

    for col_i, policy in enumerate(TARGET_POLICIES):
        psub = plot_df[plot_df["policy_norm"] == policy].copy()
        for row_i, (metric_col, metric_label) in enumerate(metric_rows):
            ax = axes[row_i, col_i]
            for dataset in DATASET_ORDER:
                sub = psub[psub["dataset_group"] == dataset].copy()
                if sub.empty:
                    continue

                x = sub["straggler_ratio"].to_numpy(dtype=float)
                y = sub[metric_col].to_numpy(dtype=float)
                ax.scatter(
                    x,
                    y,
                    s=17,
                    alpha=0.30,
                    color=DATASET_COLORS[dataset],
                    edgecolors="none",
                )

                fit = _fit_line(x, y)
                if fit is not None:
                    slope, intercept = fit
                    xline = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 120)
                    yline = slope * xline + intercept
                    ax.plot(
                        xline,
                        yline,
                        color=DATASET_COLORS[dataset],
                        linestyle=DATASET_LINESTYLES[dataset],
                        linewidth=2.3,
                        alpha=DATASET_ALPHAS[dataset],
                    )

            if row_i == 0:
                ax.set_title(POLICY_DISPLAY.get(policy, policy.upper()), fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(metric_label)
            if row_i == 1:
                ax.set_xlabel("Rollout Straggler Ratio")
            ax.grid(alpha=0.23)
            ax.set_axisbelow(True)
            x_max = float(plot_df["straggler_ratio"].quantile(0.995)) if not plot_df.empty else 1.0
            ax.set_xlim(0.0, max(1e-6, x_max * 1.05))

    handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            linestyle=DATASET_LINESTYLES[dataset],
            linewidth=2.8,
            alpha=DATASET_ALPHAS[dataset],
            label=("gsm8k" if dataset == "gsm8k" else "full-hh-rlhf"),
        )
        for dataset in DATASET_ORDER
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "Straggler Ratio vs Throughput and CPU/DRAM Energy Intensity",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
