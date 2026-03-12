"""Communication Time Absolute vs Relative (Tax Diagnostic).

Motivation:
- If comm_fraction is higher, determine whether comm time is truly larger
  or just larger relative to faster step time.

Plot:
- X: comm_s/step_all_ops (absolute)
- Y: comm_fraction/step = comm_s/step_all_ops / step_s (relative)
- Facet: policy
- Color: platform
- Marks: step-level points + run-level means
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier1/communication_time_absolute_vs_relative_tax_diagnostic_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

POLICY_ORDER = ["ppo", "remax", "grpo"]
PLATFORM_ORDER = ["2xH200", "4xA100"]
PLATFORM_COLORS = {
    "2xH200": "#4c78a8",
    "4xA100": "#f58518",
}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _norm_policy(x: str) -> str:
    return str(x).strip().lower().replace("remx", "remax")


def main() -> None:
    sf, _ = load_view("step_fact_view")
    wide, _ = load_view("step_metrics_wide_curated")
    step_long, _ = load_view("step_metrics_long")

    sf = sf[sf["run_id"].astype(str).isin(RUN_IDS)].copy()
    before = sf.copy()
    sf = apply_analysis_ok(sf)
    print(f"filtering={explain_filtering(before, sf)}")
    if sf.empty:
        raise ValueError("No step rows left after filtering for selected runs.")

    key_cols = ["run_id", "global_step_canonical", "policy", "step_time_s"]
    missing_sf = [c for c in key_cols if c not in sf.columns]
    if missing_sf:
        raise ValueError(f"step_fact_view missing required columns: {missing_sf}")
    keys = sf[key_cols].drop_duplicates().copy()
    keys["policy_norm"] = keys["policy"].map(_norm_policy)

    needed = ["run_id", "global_step_canonical", "metric_timing_s_step"]
    missing_w = [c for c in needed if c not in wide.columns]
    if missing_w:
        raise ValueError(f"step_metrics_wide_curated missing required columns: {missing_w}")

    needed_long = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing_l = [c for c in needed_long if c not in step_long.columns]
    if missing_l:
        raise ValueError(f"step_metrics_long missing required columns: {missing_l}")

    w = wide[wide["run_id"].astype(str).isin(RUN_IDS)][needed].copy()
    w["metric_timing_s_step"] = pd.to_numeric(w["metric_timing_s_step"], errors="coerce")

    m = step_long[step_long["run_id"].astype(str).isin(RUN_IDS)][needed_long].copy()
    m = m[m["metric_key"].astype(str).str.startswith("comm_s/")].copy()
    m = m[~m["metric_key"].isin(["comm_s/step", "comm_s/total"])].copy()
    m["metric_value_float"] = pd.to_numeric(m["metric_value_float"], errors="coerce")
    m = m.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    comm_by_step = (
        m.groupby(["run_id", "global_step_canonical"], dropna=False)["metric_value_float"]
        .sum(min_count=1)
        .reset_index(name="comm_s_step_all_ops")
    )

    df = keys.merge(w, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(comm_by_step, on=["run_id", "global_step_canonical"], how="inner")
    df["step_time_s"] = pd.to_numeric(df["step_time_s"], errors="coerce")
    df["step_s_for_fraction"] = df["metric_timing_s_step"].where(df["metric_timing_s_step"] > 0, df["step_time_s"])
    df = df[df["step_s_for_fraction"] > 0].copy()

    df["comm_s_step_abs"] = df["comm_s_step_all_ops"]
    df["comm_fraction_step"] = df["comm_s_step_abs"] / df["step_s_for_fraction"]
    df["platform"] = df["run_id"].map(_platform_from_run_id)
    df = df[
        df["policy_norm"].isin(POLICY_ORDER)
        & df["platform"].isin(PLATFORM_ORDER)
    ].copy()
    df = df.dropna(subset=["comm_s_step_abs", "comm_fraction_step"]).copy()

    if df.empty:
        raise ValueError("No valid points to plot after metric computation.")

    summary = (
        df.groupby(["policy_norm", "platform"], dropna=False)
        .agg(
            n_steps=("comm_s_step_abs", "size"),
            mean_comm_s_step=("comm_s_step_abs", "mean"),
            mean_comm_fraction=("comm_fraction_step", "mean"),
            median_comm_s_step=("comm_s_step_abs", "median"),
            median_comm_fraction=("comm_fraction_step", "median"),
        )
        .reset_index()
        .sort_values(["policy_norm", "platform"])
    )
    print("policy x platform summary:")
    print(summary.to_string(index=False))

    run_means = (
        df.groupby(["run_id", "policy_norm", "platform"], dropna=False)
        .agg(
            run_mean_comm_s_step=("comm_s_step_abs", "mean"),
            run_mean_comm_fraction=("comm_fraction_step", "mean"),
            n_steps=("comm_s_step_abs", "size"),
        )
        .reset_index()
        .sort_values(["policy_norm", "platform", "run_id"])
    )
    print("run-level means:")
    print(run_means.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharex=True, sharey=True)
    for ax, policy in zip(axes, POLICY_ORDER):
        sub = df[df["policy_norm"] == policy].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        for platform in PLATFORM_ORDER:
            g = sub[sub["platform"] == platform]
            if g.empty:
                continue
            ax.scatter(
                g["comm_s_step_abs"],
                g["comm_fraction_step"],
                s=18,
                alpha=0.23,
                color=PLATFORM_COLORS[platform],
                edgecolors="none",
                zorder=1,
            )

            rm = run_means[(run_means["policy_norm"] == policy) & (run_means["platform"] == platform)]
            if not rm.empty:
                row = rm.iloc[0]
                ax.scatter(
                    [row["run_mean_comm_s_step"]],
                    [row["run_mean_comm_fraction"]],
                    s=150,
                    marker="D",
                    color=PLATFORM_COLORS[platform],
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=4,
                )
                ax.annotate(
                    f"{platform} mean",
                    (row["run_mean_comm_s_step"], row["run_mean_comm_fraction"]),
                    textcoords="offset points",
                    xytext=(5, 6),
                    fontsize=8,
                )

        ax.set_title(policy.upper(), pad=8)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        ax.set_xlabel("comm_s/step (absolute, all comm_s/*)")

    axes[0].set_ylabel("comm_fraction/step (all comm_s/* / step_s)")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor=PLATFORM_COLORS[p], markeredgecolor="none", alpha=0.6, label=p)
        for p in PLATFORM_ORDER
    ]
    marker_handle = plt.Line2D([0], [0], marker="D", linestyle="None", markersize=7, markerfacecolor="#aaaaaa", markeredgecolor="black", label="run mean")
    fig.legend(handles + [marker_handle], [*PLATFORM_ORDER, "run mean"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.96), title="Platform")

    fig.suptitle("Communication Time Absolute vs Relative (All Communication Ops)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
