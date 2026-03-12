"""Phase dominance donut map for selected runs, grouped by policy x platform.

Style mirrors plots/tier0/phase_dominance_map_baselines.py:
- Outer ring: energy_share
- Inner ring: time_share
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


INCLUDE_VALIDATION = False
OUTPATH = Path("plots/out/figures/tier0/phase_dominance_map_by_policy_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PLATFORM_ORDER = ("2xH200", "4xA100")
POLICY_ORDER = ("ppo", "remax", "grpo")
PHASE_ORDER = ("rollout", "training", "rl_policy")

PHASE_COLORS = {
    "rollout": "#4C78A8",
    "training": "#F58518",
    "rl_policy": "#54A24B",
}


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _annotate_ring_percentages(
    ax: plt.Axes,
    wedges: list,
    values: list[float],
    *,
    radius_mid: float,
    min_pct_to_label: float = 5.0,
) -> None:
    total = float(sum(values))
    if total <= 0:
        return

    for wedge, value in zip(wedges, values):
        pct = 100.0 * float(value) / total
        if pct < min_pct_to_label:
            continue
        theta_deg = 0.5 * (float(wedge.theta1) + float(wedge.theta2))
        theta = math.radians(theta_deg)
        x = radius_mid * math.cos(theta)
        y = radius_mid * math.sin(theta)
        ax.text(
            x,
            y,
            f"{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="black")],
            zorder=6,
        )


def main() -> None:
    phase_df, _ = load_view("phase_fact_view")
    required_cols = ["run_id", "phase_name", "energy_share", "time_share", "policy"]
    missing_required = [col for col in required_cols if col not in phase_df.columns]
    if missing_required:
        raise ValueError(
            f"phase_fact_view is missing required columns {missing_required}. "
            f"Available columns: {list(phase_df.columns)}"
        )

    selected = phase_df[phase_df["run_id"].astype(str).isin(RUN_IDS)].copy()
    if selected.empty:
        raise ValueError(f"No rows found for selected RUN_IDS: {RUN_IDS}")

    before = selected.copy()
    selected = apply_analysis_ok(selected)
    print(f"filtering={explain_filtering(before, selected)}")
    if selected.empty:
        raise ValueError("No rows remain after apply_analysis_ok.")

    selected["policy_norm"] = selected["policy"].astype(str).str.lower().replace({"remx": "remax"})
    selected["platform"] = selected["run_id"].map(_platform_from_run_id)
    selected["phase_bucket"] = selected["phase_name"].map(_phase_bucket)

    if not INCLUDE_VALIDATION:
        selected = selected[selected["phase_bucket"] != "validation"].copy()

    selected = selected[selected["phase_bucket"].isin(PHASE_ORDER)].copy()
    selected = selected[selected["platform"].isin(PLATFORM_ORDER)].copy()
    selected = selected[selected["policy_norm"].isin(POLICY_ORDER)].copy()

    for col in ["time_share", "energy_share"]:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=["time_share", "energy_share"]).copy()

    if selected.empty:
        raise ValueError("No valid rows to plot after phase/platform/policy filtering.")

    print("runs included by (platform, policy):")
    print(
        selected.groupby(["platform", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["platform", "policy_norm"])
        .to_string(index=False)
    )

    print("points included by (platform, policy, phase):")
    print(
        selected.groupby(["platform", "policy_norm", "phase_bucket"], dropna=False)
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(["platform", "policy_norm", "phase_bucket"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(
        len(PLATFORM_ORDER),
        len(POLICY_ORDER),
        figsize=(14, 9),
        subplot_kw={"aspect": "equal"},
    )

    for row_idx, platform in enumerate(PLATFORM_ORDER):
        platform_df = selected[selected["platform"] == platform]
        for col_idx, policy in enumerate(POLICY_ORDER):
            ax = axes[row_idx][col_idx]
            panel_df = platform_df[platform_df["policy_norm"] == policy]

            phase_means = (
                panel_df.groupby("phase_bucket", dropna=False)[["time_share", "energy_share"]]
                .mean()
                .reindex(PHASE_ORDER)
                .fillna(0.0)
            )

            time_vals = phase_means["time_share"].clip(lower=0.0).to_numpy()
            energy_vals = phase_means["energy_share"].clip(lower=0.0).to_numpy()
            colors = [PHASE_COLORS[p] for p in PHASE_ORDER]

            if float(time_vals.sum() + energy_vals.sum()) <= 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.set_axis_off()
                continue

            outer_wedges, _ = ax.pie(
                energy_vals if energy_vals.sum() > 0 else [1.0],
                labels=None,
                colors=colors if energy_vals.sum() > 0 else ["#D9D9D9"],
                radius=1.0,
                wedgeprops={"width": 0.32, "edgecolor": "white", "linewidth": 0.8},
            )
            inner_wedges, _ = ax.pie(
                time_vals if time_vals.sum() > 0 else [1.0],
                labels=None,
                colors=colors if time_vals.sum() > 0 else ["#BDBDBD"],
                radius=0.66,
                wedgeprops={"width": 0.32, "edgecolor": "white", "linewidth": 0.8},
            )
            _annotate_ring_percentages(
                ax,
                outer_wedges,
                energy_vals.tolist() if hasattr(energy_vals, "tolist") else list(energy_vals),
                radius_mid=0.84,
            )
            _annotate_ring_percentages(
                ax,
                inner_wedges,
                time_vals.tolist() if hasattr(time_vals, "tolist") else list(time_vals),
                radius_mid=0.50,
            )

            n_runs = panel_df["run_id"].nunique()
            ax.set_title(f"{platform} | {policy}\nouter=energy_share, inner=time_share, n_runs={n_runs}", fontsize=10)

    phase_handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=p) for p in PHASE_ORDER]
    fig.suptitle("Phase Dominance Map by Policy (Selected Runs)", y=0.99)
    fig.legend(handles=phase_handles, title="phase colors", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
