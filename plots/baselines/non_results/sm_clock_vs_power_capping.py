"""SM clock distribution versus software power capping state."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/baselines/non_results/sm_clock_vs_power_capping.png")
TARGET_POLICIES = {"ppo", "remax", "grpo"}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_PANEL_TITLE = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
PHASE_ORDER = ("rollout", "training")
PHASE_LABEL = {"rollout": "Rollout", "training": "Training"}
PHASE_COLORS = {"rollout": "#4C78A8", "training": "#F58518"}
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
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
    ][["run_id", "model_facet"]].drop_duplicates()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def _build_plot_df(selected_runs: pd.DataFrame) -> pd.DataFrame:
    periodic, _ = load_view("hardware_periodic")
    needed = [
        "run_id",
        "phase_name",
        "sm_clock_MHz",
        "thr_sw_power_cap",
        "record_type",
        "source",
    ]
    df = periodic[needed].copy()
    df = df[df["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))].copy()
    df = df[df["record_type"].astype(str).str.upper() == "PERIODIC"].copy()
    df = df[df["source"].astype(str).str.lower() == "nvml"].copy()
    df["phase_name"] = df["phase_name"].astype(str).str.lower()
    df = df[df["phase_name"].isin(PHASE_ORDER)].copy()
    df["sm_clock_MHz"] = pd.to_numeric(df["sm_clock_MHz"], errors="coerce")
    df["thr_sw_power_cap"] = df["thr_sw_power_cap"].fillna(False).astype(bool)
    df = df.dropna(subset=["sm_clock_MHz"]).copy()
    df = apply_analysis_ok(df)
    df = df.merge(selected_runs, on="run_id", how="inner")
    df["phase_label"] = df["phase_name"].map(PHASE_LABEL)
    df["cap_label"] = df["thr_sw_power_cap"].map({False: "False", True: "True"})
    return df


def main() -> None:
    selected_runs = _select_baseline_runs()
    plot_df = _build_plot_df(selected_runs)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1), sharey=True)
    for ax, model_facet in zip(axes, TARGET_MODEL_FACETS):
        sub = plot_df[plot_df["model_facet"] == model_facet].copy()
        sns.violinplot(
            data=sub,
            x="cap_label",
            y="sm_clock_MHz",
            hue="phase_label",
            palette=[PHASE_COLORS[p] for p in PHASE_ORDER],
            cut=0,
            inner="box",
            linewidth=1.0,
            ax=ax,
        )
        ax.set_title(MODEL_PANEL_TITLE[model_facet], fontweight="bold")
        ax.set_xlabel("thr_sw_power_cap")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("sm_clock_MHz")
        else:
            ax.set_ylabel("")
        legend = ax.get_legend()
        if legend is not None:
            if ax is axes[0]:
                legend.set_title("Phase")
            else:
                legend.remove()

    fig.suptitle("SM Clock Frequency Distribution vs. Software Power Capping", y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.4)
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
