"""Baseline rollout response-length distribution + tail stats.

Plot 1 (left): stacked bar distribution by policy using bucket fractions:
  rollout/response_len_bucket_* / rollout_num_sequences

Plot 2 (right): dot + CI by policy for:
  rollout_mean_output_len, rollout/response_length_p50, rollout/response_length_p95
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/response_length_distribution_and_tail_stats_baselines.png")
INCLUDE_VALIDATION = False

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = {"ppo", "remax", "grpo"}
POLICY_ORDER = ["ppo", "remax", "grpo"]
TARGET_MODEL_FACETS = ("Llama", "Qwen")
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")

BUCKET_KEY_TO_LABEL = {
    "rollout/response_len_bucket_lt_256": "<256",
    "rollout/response_len_bucket_256_768": "256-768",
    "rollout/response_len_bucket_gt_768": ">768",
}
BUCKET_ORDER = [
    "rollout/response_len_bucket_lt_256",
    "rollout/response_len_bucket_256_768",
    "rollout/response_len_bucket_gt_768",
]
BUCKET_COLORS = {
    "rollout/response_len_bucket_lt_256": "#4C78A8",
    "rollout/response_len_bucket_256_768": "#F58518",
    "rollout/response_len_bucket_gt_768": "#54A24B",
}
MODEL_COLORS = {
    "Llama": "#1f77b4",
    "Qwen": "#d62728",
}

TAIL_METRICS = [
    "rollout_mean_output_len",
    "rollout/response_length_p50",
    "rollout/response_length_p95",
]
TAIL_LABELS = {
    "rollout_mean_output_len": "mean",
    "rollout/response_length_p50": "p50",
    "rollout/response_length_p95": "p95",
}
TAIL_COLORS = {
    "rollout_mean_output_len": "#4C78A8",
    "rollout/response_length_p50": "#F58518",
    "rollout/response_length_p95": "#54A24B",
}
MODEL_MARKERS = {
    "Llama": "o",
    "Qwen": "s",
}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _load_baseline_runs() -> pd.DataFrame:
    runs_summary, _ = load_view("run_summary_view")
    runs_meta, _ = load_view("runs")
    required_summary = ["run_id", "policy", "model", "logical_run_group"]
    missing = [c for c in required_summary if c not in runs_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs_meta.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    runs_df = runs_summary.merge(
        runs_meta[["run_id", "slurm_job_name"]],
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
    ].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected under requested slurm_job_name constraints.")
    return selected


def _load_eligible_steps(selected_run_ids: list[str]) -> pd.DataFrame:
    step_fact, _ = load_view("step_fact_view")
    required = ["run_id", "global_step_canonical"]
    missing = [c for c in required if c not in step_fact.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    step_df = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)].copy()
    step_before = step_df.copy()
    step_df = apply_analysis_ok(step_df)
    filtering = explain_filtering(step_before, step_df)
    print(f"step filtering={filtering}")
    if not INCLUDE_VALIDATION and "is_validation_step" in step_df.columns:
        step_df = step_df[~step_df["is_validation_step"].fillna(False)].copy()
    return step_df[["run_id", "global_step_canonical"]].drop_duplicates()


def _load_bucket_counts(selected_run_ids: list[str], eligible_steps: pd.DataFrame) -> pd.DataFrame:
    step_long, _ = load_view("step_metrics_long")
    needed = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing = [c for c in needed if c not in step_long.columns]
    if missing:
        raise ValueError(f"step_metrics_long missing required columns: {missing}")

    buckets = step_long[
        step_long["run_id"].astype(str).isin(selected_run_ids) & step_long["metric_key"].isin(BUCKET_ORDER)
    ][needed].copy()
    buckets = buckets.merge(eligible_steps, on=["run_id", "global_step_canonical"], how="inner")
    buckets["metric_value_float"] = pd.to_numeric(buckets["metric_value_float"], errors="coerce")
    buckets = buckets.dropna(subset=["metric_value_float"]).copy()
    pivoted = (
        buckets.pivot_table(
            index=["run_id", "global_step_canonical"],
            columns="metric_key",
            values="metric_value_float",
            aggfunc="mean",
        )
        .reindex(columns=BUCKET_ORDER)
        .reset_index()
    )
    return pivoted


def _load_rollout_sequences_and_mean(selected_run_ids: list[str], eligible_steps: pd.DataFrame) -> pd.DataFrame:
    tok_df, _ = load_view("tokens_and_steps")
    needed = ["run_id", "global_step_canonical", "rollout_num_sequences", "rollout_mean_output_len"]
    missing = [c for c in needed if c not in tok_df.columns]
    if missing:
        raise ValueError(f"tokens_and_steps missing required columns: {missing}")

    tok = tok_df[tok_df["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    tok = tok.merge(eligible_steps, on=["run_id", "global_step_canonical"], how="inner")
    for c in ["rollout_num_sequences", "rollout_mean_output_len"]:
        tok[c] = pd.to_numeric(tok[c], errors="coerce")
    tok = (
        tok.groupby(["run_id", "global_step_canonical"], as_index=False)[["rollout_num_sequences", "rollout_mean_output_len"]]
        .max()
    )
    return tok


def _load_tail_stats_from_step_metrics(selected_run_ids: list[str], eligible_steps: pd.DataFrame) -> pd.DataFrame:
    step_long, _ = load_view("step_metrics_long")
    needed = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    tails = step_long[
        step_long["run_id"].astype(str).isin(selected_run_ids)
        & step_long["metric_key"].isin(["rollout/response_length_p50", "rollout/response_length_p95"])
    ][needed].copy()
    tails = tails.merge(eligible_steps, on=["run_id", "global_step_canonical"], how="inner")
    tails["metric_value_float"] = pd.to_numeric(tails["metric_value_float"], errors="coerce")
    tails = tails.dropna(subset=["metric_value_float"]).copy()
    return tails


def main() -> None:
    selected_runs = _load_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    eligible_steps = _load_eligible_steps(selected_run_ids)
    buckets = _load_bucket_counts(selected_run_ids, eligible_steps)
    tok = _load_rollout_sequences_and_mean(selected_run_ids, eligible_steps)
    tails = _load_tail_stats_from_step_metrics(selected_run_ids, eligible_steps)

    meta = selected_runs[["run_id", "policy_norm", "model_facet"]].drop_duplicates()
    bucket_df = buckets.merge(tok, on=["run_id", "global_step_canonical"], how="inner").merge(meta, on="run_id", how="left")
    bucket_df = bucket_df.dropna(subset=["rollout_num_sequences"]).copy()
    bucket_df = bucket_df[bucket_df["rollout_num_sequences"] > 0].copy()

    for key in BUCKET_ORDER:
        bucket_df[f"{key}_frac"] = bucket_df[key] / bucket_df["rollout_num_sequences"]

    dist_rows: list[dict[str, float | str]] = []
    for (model, policy), g in bucket_df.groupby(["model_facet", "policy_norm"], dropna=False):
        denom = float(g["rollout_num_sequences"].sum())
        row: dict[str, float | str] = {"model_facet": model, "policy_norm": policy}
        for key in BUCKET_ORDER:
            numer = float(g[key].sum())
            row[key] = numer / denom if denom > 0 else np.nan
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows)

    mean_len = tok.merge(meta, on="run_id", how="left")[["run_id", "global_step_canonical", "policy_norm", "model_facet", "rollout_mean_output_len"]]
    mean_len = mean_len.rename(columns={"rollout_mean_output_len": "metric_value_float"})
    mean_len["metric_key"] = "rollout_mean_output_len"
    tail_df = tails.merge(meta, on="run_id", how="left")
    tail_plot_df = pd.concat(
        [
            mean_len[["run_id", "global_step_canonical", "policy_norm", "model_facet", "metric_key", "metric_value_float"]],
            tail_df[["run_id", "global_step_canonical", "policy_norm", "model_facet", "metric_key", "metric_value_float"]],
        ],
        ignore_index=True,
    )
    tail_plot_df = tail_plot_df.dropna(subset=["metric_value_float"]).copy()

    tail_summary = (
        tail_plot_df.groupby(["model_facet", "policy_norm", "metric_key"], dropna=False)["metric_value_float"]
        .agg(mean="mean", q025=lambda s: s.quantile(0.025), q975=lambda s: s.quantile(0.975), n="count")
        .reset_index()
    )

    run_counts = (
        selected_runs.groupby(["model_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["model_facet", "policy_norm"])
    )
    print("runs included by (model, policy):")
    print(run_counts.to_string(index=False))
    print("distribution summary (fraction by bucket):")
    print(dist_df.sort_values(["model_facet", "policy_norm"]).to_string(index=False))
    print("tail summary (mean and 95% interval):")
    print(tail_summary.sort_values(["model_facet", "policy_norm", "metric_key"]).to_string(index=False))

    fig, (ax_dist, ax_tail) = plt.subplots(1, 2, figsize=(15, 6))

    x_base = np.arange(len(POLICY_ORDER), dtype=float)
    bar_w = 0.34
    model_offsets = {"Llama": -bar_w / 2.0, "Qwen": bar_w / 2.0}
    for model in TARGET_MODEL_FACETS:
        model_df = dist_df[dist_df["model_facet"] == model].set_index("policy_norm")
        x = np.array([x_base[i] + model_offsets[model] for i in range(len(POLICY_ORDER))], dtype=float)
        bottoms = np.zeros(len(POLICY_ORDER), dtype=float)
        for bucket in BUCKET_ORDER:
            vals = np.array([float(model_df.loc[p, bucket]) if p in model_df.index else np.nan for p in POLICY_ORDER], dtype=float)
            vals = np.nan_to_num(vals, nan=0.0)
            ax_dist.bar(
                x,
                vals,
                width=bar_w,
                bottom=bottoms,
                color=BUCKET_COLORS[bucket],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9 if model == "Llama" else 0.65,
            )
            bottoms += vals
        for xi, yi in zip(x, bottoms):
            ax_dist.text(xi, min(yi + 0.02, 1.04), model, ha="center", va="bottom", fontsize=8, rotation=90)

    ax_dist.set_xticks(x_base)
    ax_dist.set_xticklabels(POLICY_ORDER)
    ax_dist.set_xlabel("policy")
    ax_dist.set_ylabel("fraction of sequences")
    ax_dist.set_ylim(0, 1.08)
    ax_dist.set_title("Distribution: response length buckets\n(fractions from bucket_count / rollout_num_sequences)")
    ax_dist.grid(axis="y", alpha=0.2)

    metric_offsets = {
        "rollout_mean_output_len": -0.18,
        "rollout/response_length_p50": 0.0,
        "rollout/response_length_p95": 0.18,
    }
    model_small_offsets = {"Llama": -0.04, "Qwen": 0.04}
    for metric in TAIL_METRICS:
        for model in TARGET_MODEL_FACETS:
            sub = tail_summary[(tail_summary["metric_key"] == metric) & (tail_summary["model_facet"] == model)].set_index("policy_norm")
            xs = []
            ys = []
            err_lo = []
            err_hi = []
            for i, pol in enumerate(POLICY_ORDER):
                if pol not in sub.index:
                    continue
                row = sub.loc[pol]
                x = x_base[i] + metric_offsets[metric] + model_small_offsets[model]
                xs.append(x)
                ys.append(float(row["mean"]))
                err_lo.append(float(row["mean"] - row["q025"]))
                err_hi.append(float(row["q975"] - row["mean"]))
            if xs:
                ax_tail.errorbar(
                    xs,
                    ys,
                    yerr=[err_lo, err_hi],
                    fmt=MODEL_MARKERS[model],
                    color=TAIL_COLORS[metric],
                    ecolor=TAIL_COLORS[metric],
                    elinewidth=1.0,
                    capsize=2.5,
                    markersize=5.5,
                    markerfacecolor=MODEL_COLORS[model],
                    markeredgecolor="black",
                    linestyle="none",
                    alpha=0.9,
                )
                # Label each point so metric identity and magnitude are readable without legend lookup.
                for x, y in zip(xs, ys):
                    ax_tail.text(
                        x,
                        y + 4.0,
                        f"{y:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=TAIL_COLORS[metric],
                    )

    ax_tail.set_xticks(x_base)
    ax_tail.set_xticklabels(POLICY_ORDER)
    ax_tail.set_xlabel("policy")
    ax_tail.set_ylabel("tokens")
    ax_tail.set_title("Tail stats by policy (dot = mean, CI = 2.5-97.5%)")
    ax_tail.grid(axis="y", alpha=0.2)

    bucket_handles = [Patch(facecolor=BUCKET_COLORS[b], edgecolor="white", label=BUCKET_KEY_TO_LABEL[b]) for b in BUCKET_ORDER]
    metric_handles = [Line2D([0], [0], marker="o", color=TAIL_COLORS[m], linestyle="None", markerfacecolor=TAIL_COLORS[m], label=TAIL_LABELS[m]) for m in TAIL_METRICS]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[m],
            color="black",
            linestyle="None",
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="black",
            label=m,
        )
        for m in TARGET_MODEL_FACETS
    ]
    fig.legend(handles=bucket_handles, title="length bucket", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.23, 0.98))
    fig.legend(handles=metric_handles, title="tail metric", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.57, 0.98))
    fig.legend(handles=model_handles, title="model", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.86, 0.98))

    fig.suptitle("Baseline rollout response-length distribution and tail statistics", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
