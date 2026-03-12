"""Component-level phase energy accounting for selected baseline runs.

Shows energy-weighted GPU vs CPU/DRAM shares by phase, with policies grouped
inside each platform panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier1/component_energy_shares_phase_selected_runs.png")

RUN_IDS = [
    "stage1_llama8b_grpo_2gpu_h200_20260306_033327",
    "stage1_llama8b_grpo_4gpu_a100_20260306_185149",
    "stage1_llama8b_ppo_2gpu_h200_20260306_015225",
    "stage1_llama8b_ppo_4gpu_a100_20260306_171626",
    "stage1_llama8b_remax_2gpu_h200_20260306_024810",
    "stage1_llama8b_remax_4gpu_a100_20260306_182154",
]

PHASE_ORDER = ("rollout", "training", "rl_policy")
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_OFFSETS = {"ppo": -0.22, "remax": 0.0, "grpo": 0.22}
PLATFORM_ORDER = ("2xH200", "4xA100")

POLICY_COLORS = {"ppo": "#1f77b4", "remax": "#ff7f0e", "grpo": "#2ca02c"}


def _platform_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    return "other"


def _norm_policy(policy: str) -> str:
    return str(policy).strip().lower().replace("remx", "remax")


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def main() -> None:
    phase_df, _ = load_view("phase_fact_view")

    required = ["run_id", "policy", "phase_name", "gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]
    missing = [c for c in required if c not in phase_df.columns]
    if missing:
        raise ValueError(f"phase_fact_view missing required columns: {missing}")

    phase_df = phase_df[phase_df["run_id"].astype(str).isin(RUN_IDS)].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"filtering={explain_filtering(before, phase_df)}")
    if phase_df.empty:
        raise ValueError("No phase rows remain after filtering for selected RUN_IDS.")

    phase_df["policy_norm"] = phase_df["policy"].map(_norm_policy)
    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df["platform"] = phase_df["run_id"].map(_platform_from_run_id)
    phase_df = phase_df[
        phase_df["policy_norm"].isin(POLICY_ORDER)
        & phase_df["phase_bucket"].isin(PHASE_ORDER)
        & phase_df["platform"].isin(PLATFORM_ORDER)
    ].copy()

    for c in ["gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]:
        phase_df[c] = pd.to_numeric(phase_df[c], errors="coerce")
    phase_df = phase_df.dropna(subset=["gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]).copy()
    phase_df = phase_df[phase_df["total_energy_j"] > 0].copy()
    if phase_df.empty:
        raise ValueError("No valid numeric energy rows remain.")

    component_ratio = (phase_df["gpu_energy_j"] + phase_df["cpu_dram_energy_j"]) / phase_df["total_energy_j"]
    max_abs_err = float((component_ratio - 1.0).abs().max()) if not component_ratio.empty else 0.0
    if max_abs_err > 0.02:
        raise ValueError("Component closure check failed; explicit 'other' component handling required.")
    print(f"component_closure_max_abs_error={max_abs_err:.6f}")

    agg = (
        phase_df.groupby(["platform", "policy_norm", "phase_bucket"], dropna=False)[
            ["gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]
        ]
        .sum()
        .reset_index()
    )
    agg["gpu_share_mean"] = agg["gpu_energy_j"] / agg["total_energy_j"]
    agg["cpu_dram_share_mean"] = agg["cpu_dram_energy_j"] / agg["total_energy_j"]

    print("selected runs by platform/policy:")
    print(
        phase_df.groupby(["platform", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["platform", "policy_norm"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    for ax, platform in zip(axes, PLATFORM_ORDER):
        sub = agg[agg["platform"] == platform]
        x_positions = list(range(len(PHASE_ORDER)))
        for policy in POLICY_ORDER:
            psub = sub[sub["policy_norm"] == policy]
            if psub.empty:
                continue

            vals_gpu = []
            vals_cpu = []
            for phase in PHASE_ORDER:
                row = psub[psub["phase_bucket"] == phase]
                vals_gpu.append(float(row["gpu_share_mean"].iloc[0]) if not row.empty else 0.0)
                vals_cpu.append(float(row["cpu_dram_share_mean"].iloc[0]) if not row.empty else 0.0)

            xpos = [x + POLICY_OFFSETS[policy] for x in x_positions]
            barw = 0.20
            ax.bar(
                xpos,
                vals_gpu,
                width=barw,
                color=POLICY_COLORS[policy],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.95,
                zorder=2,
            )
            ax.bar(
                xpos,
                vals_cpu,
                width=barw,
                bottom=vals_gpu,
                color=POLICY_COLORS[policy],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.35,
                zorder=2,
            )

        ax.set_title(platform)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(PHASE_ORDER, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("phase_bucket")
    axes[0].set_ylabel("energy share")

    component_handles = [
        Patch(facecolor="#777777", edgecolor="black", alpha=0.95, label="GPU"),
        Patch(facecolor="#777777", edgecolor="black", alpha=0.35, label="CPU/DRAM"),
    ]
    policy_handles = [
        Patch(facecolor=POLICY_COLORS["ppo"], edgecolor="black", label="PPO"),
        Patch(facecolor=POLICY_COLORS["remax"], edgecolor="black", label="ReMax"),
        Patch(facecolor=POLICY_COLORS["grpo"], edgecolor="black", label="GRPO"),
    ]

    fig.suptitle("Phase-Level Component Energy Shares (Selected Runs)", y=0.99)
    fig.legend(
        component_handles,
        ["GPU", "CPU/DRAM"],
        title="component (fill)",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.30, 0.96),
    )
    fig.legend(
        policy_handles,
        ["PPO", "ReMax", "GRPO"],
        title="policy (color)",
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.73, 0.96),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
