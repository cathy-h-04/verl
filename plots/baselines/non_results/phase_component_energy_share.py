"""Baseline component-level phase energy accounting by model and policy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/baselines/non_results/phase_component_energy_share.png")

TARGET_MODELS = ("Llama", "Qwen")
TARGET_POLICIES = ("ppo", "remax", "grpo")
PHASE_ORDER = ("rollout", "rl_policy", "training")
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_OFFSETS = {"ppo": -0.22, "remax": 0.0, "grpo": 0.22}

MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
PHASE_DISPLAY = {
    "rollout": "Rollout",
    "rl_policy": "Preparation",
    "training": "Training",
}
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


def _annotation_color(fill_alpha: float) -> str:
    return "black" if fill_alpha <= 0.4 else "white"


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


def _annotate_stack(ax: plt.Axes, xpos: float, bottom: float, height: float, text: str, fill_alpha: float) -> None:
    if height < 0.06:
        return
    ax.text(
        xpos,
        bottom + (height / 2.0),
        text,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=_annotation_color(fill_alpha),
    )


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

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
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
            for xval, gpu_val, cpu_val in zip(xpos, vals_gpu, vals_cpu):
                _annotate_stack(ax, xval, 0.0, gpu_val, f"{gpu_val * 100:.0f}%", fill_alpha=0.95)
                _annotate_stack(ax, xval, gpu_val, cpu_val, f"{cpu_val * 100:.0f}%", fill_alpha=0.35)

        ax.set_title(MODEL_DISPLAY[model], fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([PHASE_DISPLAY[phase] for phase in PHASE_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Phase")
    axes[0].set_ylabel("Energy share")

    component_handles = [
        Patch(facecolor="#777777", edgecolor="black", alpha=0.95, label="GPU"),
        Patch(facecolor="#777777", edgecolor="black", alpha=0.35, label="CPU/DRAM"),
    ]
    policy_handles = [
        Patch(facecolor=POLICY_COLORS["ppo"], edgecolor="black", label="PPO"),
        Patch(facecolor=POLICY_COLORS["remax"], edgecolor="black", label="ReMax"),
        Patch(facecolor=POLICY_COLORS["grpo"], edgecolor="black", label="GRPO"),
    ]

    fig.suptitle("Component Energy Shares by Phase, Model, and Policy", y=0.99, fontweight="bold")
    fig.legend(
        component_handles,
        ["GPU", "CPU/DRAM"],
        title="Component",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.29, 0.94),
    )
    fig.legend(
        policy_handles,
        ["PPO", "ReMax", "GRPO"],
        title="Policy",
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.72, 0.94),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
