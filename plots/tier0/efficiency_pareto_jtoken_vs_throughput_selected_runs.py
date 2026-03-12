"""Plot 0.1: Efficiency Pareto (J/Token vs Throughput) for selected runs.

Data grain: run-level aggregates from step_fact_view after shared filtering.
X-axis: perf/throughput proxy (mean throughput_tokens_s across included steps).
Y-axis: overall_j_per_output_token = sum(step_total_energy_j) / sum(step_rollout_output_tokens).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/efficiency_pareto_jtoken_vs_throughput_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

POLICY_ORDER = ["ppo", "remax", "grpo"]
POLICY_COLORS = {
    "ppo": "#4c78a8",
    "remax": "#f58518",
    "grpo": "#54a24b",
}

PLATFORM_ORDER = ["2xH200", "4xA100"]
PLATFORM_MARKERS = {
    "2xH200": "o",
    "4xA100": "s",
}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "Unknown"


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
    step, _ = load_view("step_fact_view")
    available = set(step["run_id"].astype(str).unique().tolist())
    missing = sorted(run_id for run_id in RUN_IDS if run_id not in available)
    if missing:
        raise ValueError(f"Missing run IDs in step_fact_view: {missing}")

    df = step[step["run_id"].astype(str).isin(RUN_IDS)].copy()
    if df.empty:
        raise ValueError("No rows found for selected RUN_IDS before filtering.")

    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"filtering={explain_filtering(before, df)}")
    if df.empty:
        raise ValueError("No rows remain after apply_analysis_ok.")

    for col in ["throughput_tokens_s", "step_total_energy_j", "step_rollout_output_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["step_rollout_output_tokens"] > 0].copy()
    if df.empty:
        raise ValueError("No rows with positive step_rollout_output_tokens after filtering.")

    grouped = (
        df.groupby("run_id", dropna=False)
        .agg(
            throughput_tokens_s=("throughput_tokens_s", "mean"),
            total_energy_j=("step_total_energy_j", "sum"),
            rollout_output_tokens=("step_rollout_output_tokens", "sum"),
            policy=("policy", "first"),
        )
        .reset_index()
    )
    grouped["policy"] = grouped["policy"].astype(str).str.lower().replace({"remx": "remax"})
    grouped["platform"] = grouped["run_id"].map(_platform_from_run_id)
    grouped["overall_j_per_output_token"] = grouped["total_energy_j"] / grouped["rollout_output_tokens"]
    grouped = grouped.dropna(subset=["throughput_tokens_s", "overall_j_per_output_token"]).copy()

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

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for policy in POLICY_ORDER:
        for platform in PLATFORM_ORDER:
            subset = grouped[(grouped["policy"] == policy) & (grouped["platform"] == platform)]
            if subset.empty:
                continue
            ax.scatter(
                subset["throughput_tokens_s"],
                subset["overall_j_per_output_token"],
                s=120,
                marker=PLATFORM_MARKERS[platform],
                c=POLICY_COLORS[policy],
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                zorder=3,
            )

    pareto = grouped[grouped["is_pareto"]].sort_values("throughput_tokens_s")
    if len(pareto) >= 2:
        ax.plot(
            pareto["throughput_tokens_s"],
            pareto["overall_j_per_output_token"],
            color="black",
            linewidth=1.5,
            linestyle="-",
            alpha=0.9,
            label="Pareto frontier",
            zorder=2,
        )
    if not pareto.empty:
        ax.scatter(
            pareto["throughput_tokens_s"],
            pareto["overall_j_per_output_token"],
            s=220,
            marker="*",
            c="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=4,
        )

    for _, row in grouped.iterrows():
        label = f"{row['policy'].upper()} / {row['platform']}"
        ax.annotate(
            label,
            xy=(row["throughput_tokens_s"], row["overall_j_per_output_token"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_ylabel("Overall J / output token")
    ax.set_title("Plot 0.1: Efficiency Pareto (J/Token vs Throughput)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    policy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=policy.upper(),
            markerfacecolor=POLICY_COLORS[policy],
            markeredgecolor="black",
            markersize=9,
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
            markersize=9,
        )
        for platform in PLATFORM_ORDER
    ]
    leg1 = ax.legend(handles=policy_handles, title="Policy", loc="upper right", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=platform_handles, title="Platform", loc="lower left", frameon=False)

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
