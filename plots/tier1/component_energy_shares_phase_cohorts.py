"""Component-level phase energy accounting (baseline only, condensed view).

Shows energy-weighted GPU vs CPU/DRAM shares by phase, with policies grouped in each model panel.
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


OUTPATH = Path("plots/out/figures/tier1/component_energy_shares_phase_cohorts.png")

TARGET_MODELS = ("Llama", "Qwen")
TARGET_POLICIES = ("ppo", "remax", "grpo")
PHASE_ORDER = ("rollout", "training", "rl_policy")
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_OFFSETS = {"ppo": -0.22, "remax": 0.0, "grpo": 0.22}

COMPONENT_COLORS = {
    "gpu_share_mean": "#222222",
    "cpu_dram_share_mean": "#bbbbbb",
}
POLICY_COLORS = {"ppo": "#1f77b4", "remax": "#ff7f0e", "grpo": "#2ca02c"}
MODEL_HATCH = {"Llama": "", "Qwen": "//"}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _phase_bucket(phase_name: str) -> str:
    key = str(phase_name).strip().lower()
    if key in {"rollout", "training", "rl_policy", "validation"}:
        return key
    return "other"


def _is_baseline_group(logical_run_group: str) -> bool:
    grp = str(logical_run_group).strip().lower()
    return grp.startswith(("stage1_llama8b_", "qwen_sys_3b_"))


def _select_runs() -> pd.DataFrame:
    runs, _ = load_view("run_summary_view")
    required = [
        "run_id",
        "logical_run_group",
        "policy",
        "model",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")

    runs = runs.copy()
    runs["run_id"] = runs["run_id"].astype(str)
    runs["policy_norm"] = runs["policy"].astype(str).str.lower().replace({"remx": "remax"})
    runs["model_facet"] = runs["model"].map(_model_facet)
    runs["is_checkpoint_continuation"] = runs["is_checkpoint_continuation"].fillna(False).astype(bool)
    runs["join_coverage_rate"] = pd.to_numeric(runs["join_coverage_rate"], errors="coerce")
    runs["phase_boundary_integrity_rate"] = pd.to_numeric(runs["phase_boundary_integrity_rate"], errors="coerce")
    runs["is_baseline"] = runs["logical_run_group"].map(_is_baseline_group)

    mask = (
        runs["model_facet"].isin(TARGET_MODELS)
        & runs["policy_norm"].isin(TARGET_POLICIES)
        & runs["is_baseline"]
        & (~runs["is_checkpoint_continuation"])
        & (runs["join_coverage_rate"] == 1.0)
        & (runs["phase_boundary_integrity_rate"] == 1.0)
    )
    selected = runs.loc[mask, ["run_id", "logical_run_group", "model_facet", "policy_norm"]].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected after integrity gates.")
    return selected


def main() -> None:
    selected_runs = _select_runs()
    print("selected baseline runs by (model, policy):")
    print(
        selected_runs.groupby(["model_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["model_facet", "policy_norm"])
        .to_string(index=False)
    )

    phase_df, _ = load_view("phase_fact_view")
    required_pf = ["run_id", "phase_name", "gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]
    missing_pf = [c for c in required_pf if c not in phase_df.columns]
    if missing_pf:
        raise ValueError(f"phase_fact_view missing required columns: {missing_pf}")

    phase_df = phase_df[phase_df["run_id"].astype(str).isin(selected_runs["run_id"])].copy()
    before = phase_df.copy()
    phase_df = apply_analysis_ok(phase_df)
    print(f"filtering={explain_filtering(before, phase_df)}")

    phase_df = phase_df.merge(selected_runs[["run_id", "model_facet", "policy_norm"]], how="inner", on="run_id")
    phase_df["phase_bucket"] = phase_df["phase_name"].map(_phase_bucket)
    phase_df = phase_df[phase_df["phase_bucket"].isin(PHASE_ORDER)].copy()

    for c in ["gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]:
        phase_df[c] = pd.to_numeric(phase_df[c], errors="coerce")
    phase_df = phase_df.dropna(subset=["gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]).copy()
    phase_df = phase_df[phase_df["total_energy_j"] > 0].copy()

    component_ratio = (phase_df["gpu_energy_j"] + phase_df["cpu_dram_energy_j"]) / phase_df["total_energy_j"]
    max_abs_err = float((component_ratio - 1.0).abs().max()) if not component_ratio.empty else 0.0
    if max_abs_err > 0.02:
        raise ValueError("Component closure check failed; explicit 'other' component handling required.")
    print(f"component_closure_max_abs_error={max_abs_err:.6f}")

    agg = (
        phase_df.groupby(["model_facet", "policy_norm", "phase_bucket"], dropna=False)[
            ["gpu_energy_j", "cpu_dram_energy_j", "total_energy_j"]
        ]
        .sum()
        .reset_index()
    )
    agg["gpu_share_mean"] = agg["gpu_energy_j"] / agg["total_energy_j"]
    agg["cpu_dram_share_mean"] = agg["cpu_dram_energy_j"] / agg["total_energy_j"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, model in zip(axes, TARGET_MODELS):
        sub = agg[agg["model_facet"] == model]
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
            hatch = MODEL_HATCH[model]
            ax.bar(
                xpos,
                vals_gpu,
                width=barw,
                color=POLICY_COLORS[policy],
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
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
                hatch=hatch,
                alpha=0.35,
                zorder=2,
            )

        ax.set_title(model)
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
    model_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=MODEL_HATCH["Llama"], label="Llama"),
        Patch(facecolor="white", edgecolor="black", hatch=MODEL_HATCH["Qwen"], label="Qwen"),
    ]

    fig.suptitle("Phase-Level Component Energy Shares (Baseline Cohort)", y=0.99)
    fig.legend(component_handles, ["GPU", "CPU/DRAM"], title="component (fill)", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.25, 0.96))
    fig.legend(policy_handles, ["PPO", "ReMax", "GRPO"], title="policy (color)", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.58, 0.96))
    fig.legend(model_handles, ["Llama", "Qwen"], title="model (hatch)", loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.87, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
