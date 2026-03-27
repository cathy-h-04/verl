"""Efficiency frontier for all ingested llama scaling runs.

Data grain: run-level aggregates from step_fact_view after shared filtering.
X-axis: mean throughput_tokens_s across included steps.
Y-axis: overall_j_per_output_token = sum(step_total_energy_j) / sum(step_rollout_output_tokens).
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
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper, scatter_paper


OUTPATH = Path("plots/out/scale/energy_throughput_pareto.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
POLICY_COLORS = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}

PLATFORM_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")
PLATFORM_MARKERS = {
    "2xA100": "*",
    "2xH200": "o",
    "4xA100": "s",
    "4xH200": "^",
}

FIGURE_TITLE_SIZE = 18
AXIS_TITLE_SIZE = 15
TICK_LABEL_SIZE = 13
LEGEND_FONT_SIZE = 12


def _platform_from_run_id(run_id: str) -> str:
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


def _load_llama_scaling_run_ids() -> list[str]:
    runs_df, _ = load_view("runs")
    required = ["run_id", "run_dir"]
    missing = [col for col in required if col not in runs_df.columns]
    if missing:
        raise ValueError(f"runs is missing required selection columns {missing}")

    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)].copy()
    if selected.empty:
        raise ValueError("No llama_scaling runs found in runs.parquet.")
    return selected["run_id"].astype(str).drop_duplicates().tolist()


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


def _pareto_mask(df: pd.DataFrame) -> pd.Series:
    x = pd.to_numeric(df["throughput_tokens_s"], errors="coerce")
    y = pd.to_numeric(df["overall_j_per_output_token"], errors="coerce")
    out = pd.Series(True, index=df.index, dtype=bool)
    for idx in df.index:
        xi = x.loc[idx]
        yi = y.loc[idx]
        dominates = ((x >= xi) & (y <= yi) & ((x > xi) | (y < yi))).fillna(False)
        if bool(dominates.any()):
            out.loc[idx] = False
    return out


def main() -> None:
    run_ids = _load_llama_scaling_run_ids()
    step_df, _ = load_view("step_fact_view")
    available = set(step_df["run_id"].astype(str).unique().tolist())
    missing = sorted(run_id for run_id in run_ids if run_id not in available)
    if missing:
        raise ValueError(f"Missing run IDs in step_fact_view: {missing}")

    df = step_df[step_df["run_id"].astype(str).isin(run_ids)].copy()
    if df.empty:
        raise ValueError("No rows found for selected run IDs before filtering.")

    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")
    if df.empty:
        raise ValueError("No rows remain after apply_analysis_ok.")

    required_numeric = ["step_total_energy_j", "step_rollout_total_tokens", "step_rollout_output_tokens", "step_time_s"]
    for col in required_numeric:
        if col not in df.columns:
            raise ValueError(f"step_fact_view is missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df = df.dropna(
        subset=["global_step_canonical", "step_total_energy_j", "step_rollout_total_tokens", "step_rollout_output_tokens", "step_time_s"]
    ).copy()
    df["global_step_canonical"] = df["global_step_canonical"].astype(int)
    df = df[df["step_time_s"] > 0].copy()
    df["step_rollout_prompt_tokens"] = df["step_rollout_total_tokens"] - df["step_rollout_output_tokens"]

    remax_ids = sorted(df.loc[df["policy"].astype(str).str.lower() == "remax", "run_id"].astype(str).unique().tolist())
    df = df.merge(_load_remax_gen_ratio(remax_ids), on=["run_id", "global_step_canonical"], how="left")
    df["genmax_over_gen"] = pd.to_numeric(df["genmax_over_gen"], errors="coerce")
    run_medians = df.groupby("run_id")["genmax_over_gen"].median().to_dict()
    df["genmax_over_gen"] = df.apply(
        lambda row: run_medians.get(row["run_id"], np.nan) if pd.isna(row["genmax_over_gen"]) else row["genmax_over_gen"],
        axis=1,
    )

    policy_norm = df["policy"].astype(str).str.lower().replace({"remx": "remax"})
    remax_mask = policy_norm == "remax"
    df["corrected_step_tokens"] = df["step_rollout_total_tokens"]
    df.loc[remax_mask, "corrected_step_tokens"] = (
        df.loc[remax_mask, "step_rollout_total_tokens"]
        + df.loc[remax_mask, "step_rollout_prompt_tokens"]
        + df.loc[remax_mask, "step_rollout_output_tokens"] * df.loc[remax_mask, "genmax_over_gen"].fillna(1.0)
    )
    df = df[df["corrected_step_tokens"] > 0].copy()
    if df.empty:
        raise ValueError("No rows with positive corrected_step_tokens after filtering.")
    df["throughput_corrected_tokens_s"] = df["corrected_step_tokens"] / df["step_time_s"]

    grouped = (
        df.groupby("run_id", dropna=False)
        .agg(
            throughput_tokens_s=("throughput_corrected_tokens_s", "mean"),
            total_energy_j=("step_total_energy_j", "sum"),
            corrected_tokens=("corrected_step_tokens", "sum"),
            policy=("policy", "first"),
        )
        .reset_index()
    )
    grouped["policy"] = grouped["policy"].astype(str).str.lower().replace({"remx": "remax"})
    grouped["platform"] = grouped["run_id"].map(_platform_from_run_id)
    grouped["overall_j_per_output_token"] = grouped["total_energy_j"] / grouped["corrected_tokens"]
    grouped = grouped.dropna(subset=["throughput_tokens_s", "overall_j_per_output_token"]).copy()
    grouped = grouped[grouped["policy"].isin(POLICY_ORDER) & grouped["platform"].isin(PLATFORM_ORDER)].copy()

    if grouped.empty:
        raise ValueError("No run-level points available for plotting.")

    grouped["is_pareto"] = _pareto_mask(grouped)
    print("run-level aggregates:")
    print(
        grouped[
            ["run_id", "policy", "platform", "throughput_tokens_s", "overall_j_per_output_token", "is_pareto"]
        ]
        .sort_values(["policy", "platform"])
        .to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(9.8, 7.1))

    for platform in PLATFORM_ORDER:
        subset = grouped[grouped["platform"] == platform].sort_values("throughput_tokens_s")
        if len(subset) < 2:
            continue
        ax.plot(
            subset["throughput_tokens_s"],
            subset["overall_j_per_output_token"],
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.25,
            zorder=1,
        )

    for policy in POLICY_ORDER:
        for platform in PLATFORM_ORDER:
            subset = grouped[(grouped["policy"] == policy) & (grouped["platform"] == platform)]
            if subset.empty:
                continue
            ax.scatter(
                subset["throughput_tokens_s"],
                subset["overall_j_per_output_token"],
                s=175,
                marker=PLATFORM_MARKERS[platform],
                c=POLICY_COLORS[policy],
                edgecolors="black",
                linewidths=1.0,
                alpha=0.98,
                zorder=3,
            )

    scatter_paper(ax)
    ax.set_xlabel("Throughput (tokens/s)", fontsize=AXIS_TITLE_SIZE)
    ax.set_ylabel("Energy per output token (J/token)", fontsize=AXIS_TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.75, alpha=0.28)

    policy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=POLICY_DISPLAY[policy],
            markerfacecolor=POLICY_COLORS[policy],
            markeredgecolor="black",
            markersize=10,
        )
        for policy in POLICY_ORDER
    ]
    platform_handles = [
        Line2D(
            [0],
            [0],
            marker=PLATFORM_MARKERS[platform],
            color="black",
            linestyle="None",
            label=platform,
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
        )
        for platform in PLATFORM_ORDER
    ]
    leg1 = ax.legend(handles=policy_handles, loc="upper right", frameon=False, fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=platform_handles, loc="lower left", frameon=False, fontsize=LEGEND_FONT_SIZE)
    ax.add_artist(leg2)

    fig.suptitle("Scaling Tradeoff: Throughput vs Energy Efficiency", y=0.985, fontweight="bold", fontsize=FIGURE_TITLE_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="energy_throughput_pareto",
        run_ids=run_ids,
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["step_fact_view"],
            "filter": "apply_analysis_ok",
            "proxy_formula": "ppo/grpo use rollout_total_tokens; remax uses rollout_total + rollout_prompt + (gen_max/gen)*rollout_output",
            "aggregation": "run-level mean corrected-token throughput and summed J/corrected-token denominator",
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
