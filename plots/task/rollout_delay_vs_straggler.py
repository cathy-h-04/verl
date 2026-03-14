"""Straggler ratio vs rollout delay index (policy-normalized).

Replaces the prior redundant sync-vs-straggler view.

Design:
- X: rollout straggler ratio
- Y: rollout delay index
    where delay = rollout_phase_time_s / normalized workload tokens
    and index = delay / policy-median(delay)
- Facet: policy
- Hue: dataset (gsm8k vs full-hh-rlhf)
- Statistic: binned median + IQR band
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.task.line_style import DATASET_ALPHAS, DATASET_COLORS, DATASET_LINESTYLES

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/task/rollout_delay_vs_straggler.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
TARGET_SLURM_TO_DATASET = {
    "llama_rm_gsm8k": "gsm8k",
    "llama_rm_rlhf": "full-hh-rlhf",
}
DATASET_ORDER = ("gsm8k", "full-hh-rlhf")
DATASET_COLOR = {"gsm8k": "#1f77b4", "full-hh-rlhf": "#d62728"}
N_BINS = 18


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
        raise ValueError("No target runs selected.")

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


def _load_corrected_tokens_straggler(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step, _ = load_view("step_fact_view")
    needed = ["run_id", "global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "straggler_ratio"]
    missing = [c for c in needed if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    out = step[step["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    before = out.copy()
    out = apply_analysis_ok(out)
    print(f"step filtering={explain_filtering(before, out)}")

    for c in ["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "straggler_ratio"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "step_rollout_total_tokens", "step_rollout_output_tokens", "straggler_ratio"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out = out[(out["step_rollout_total_tokens"] > 0) & np.isfinite(out["straggler_ratio"]) & (out["straggler_ratio"] > 0)].copy()
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


def _load_rollout_time(selected_runs: pd.DataFrame) -> pd.DataFrame:
    phase, _ = load_view("phase_fact_view")
    needed = ["run_id", "global_step_canonical", "phase_name", "phase_time_s"]
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
    out["phase_time_s"] = pd.to_numeric(out["phase_time_s"], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "phase_time_s"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out = out[out["phase_time_s"] > 0].copy()

    return out[["run_id", "global_step_canonical", "phase_time_s"]]


def _binned_stats(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["x", "y_med", "y_q25", "y_q75", "n"])

    x = df["straggler_ratio"].to_numpy(dtype=float)
    y = df["latency_index"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5:
        return pd.DataFrame(columns=["x", "y_med", "y_q25", "y_q75", "n"])

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return pd.DataFrame(columns=["x", "y_med", "y_q25", "y_q75", "n"])

    rows: list[dict[str, float]] = []
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == edges.size - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if mask.sum() < 3:
            continue
        xb = x[mask]
        yb = y[mask]
        rows.append(
            {
                "x": float(np.median(xb)),
                "y_med": float(np.median(yb)),
                "y_q25": float(np.quantile(yb, 0.25)),
                "y_q75": float(np.quantile(yb, 0.75)),
                "n": float(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    selected_runs = _select_runs()
    print("selected runs:")
    print(selected_runs.sort_values(["dataset_group", "policy_norm", "run_id"]).to_string(index=False))

    step = _load_corrected_tokens_straggler(selected_runs)
    phase = _load_rollout_time(selected_runs)
    df = step.merge(phase, on=["run_id", "global_step_canonical"], how="inner", validate="one_to_one")

    df["latency_s_per_corrected_token"] = df["phase_time_s"] / df["corrected_rollout_total_tokens"]
    df = df[np.isfinite(df["latency_s_per_corrected_token"])].copy()
    df = df[df["latency_s_per_corrected_token"] > 0].copy()
    if df.empty:
        raise ValueError("No rows after joining step/phase data.")

    policy_medians = df.groupby("policy_norm")["latency_s_per_corrected_token"].median().to_dict()
    df["latency_index"] = df.apply(
        lambda row: row["latency_s_per_corrected_token"] / policy_medians.get(row["policy_norm"], np.nan),
        axis=1,
    )
    df = df[np.isfinite(df["latency_index"]) & (df["latency_index"] > 0)].copy()

    summary = (
        df.groupby(["policy_norm", "dataset_group"], dropna=False)
        .agg(
            n=("straggler_ratio", "size"),
            straggler_median=("straggler_ratio", "median"),
            latency_index_median=("latency_index", "median"),
            latency_index_p90=("latency_index", lambda s: float(s.quantile(0.90))),
        )
        .reset_index()
        .sort_values(["policy_norm", "dataset_group"])
    )
    print("\nsummary by policy,dataset:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(TARGET_POLICIES), figsize=(17.8, 5.5), sharey=True)
    if len(TARGET_POLICIES) == 1:
        axes = [axes]

    x_max = float(df["straggler_ratio"].quantile(0.995))
    x_max = max(x_max, 1e-6)

    for ax, policy in zip(axes, TARGET_POLICIES):
        psub = df[df["policy_norm"] == policy].copy()

        for dataset in DATASET_ORDER:
            sub = psub[psub["dataset_group"] == dataset].copy()
            if sub.empty:
                continue
            b = _binned_stats(sub, n_bins=N_BINS)
            if b.empty:
                continue

            ax.fill_between(
                b["x"].to_numpy(dtype=float),
                b["y_q25"].to_numpy(dtype=float),
                b["y_q75"].to_numpy(dtype=float),
                color=DATASET_COLORS[dataset],
                alpha=0.14,
                linewidth=0,
            )
            ax.plot(
                b["x"].to_numpy(dtype=float),
                b["y_med"].to_numpy(dtype=float),
                color=DATASET_COLORS[dataset],
                linestyle=DATASET_LINESTYLES[dataset],
                linewidth=2.8,
                alpha=DATASET_ALPHAS[dataset],
            )

        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="#666666", alpha=0.8)
        ax.set_title(POLICY_DISPLAY.get(policy, policy.upper()), fontweight="bold")
        ax.set_xlabel("Rollout Straggler Ratio")
        ax.set_xlim(0, x_max)
        ax.grid(alpha=0.24)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Rollout Delay Index (policy median = 1.0)")
    handles = [
        Line2D([0], [0], color=DATASET_COLORS["gsm8k"], linestyle=DATASET_LINESTYLES["gsm8k"], linewidth=2.8, alpha=DATASET_ALPHAS["gsm8k"], label="gsm8k"),
        Line2D([0], [0], color=DATASET_COLORS["full-hh-rlhf"], linestyle=DATASET_LINESTYLES["full-hh-rlhf"], linewidth=2.8, alpha=DATASET_ALPHAS["full-hh-rlhf"], label="full-hh-rlhf"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Straggler Severity vs Rollout Latency Cost by Policy and Dataset", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
