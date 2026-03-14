"""Reserved VRAM over training step for reward-mechanism runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/reward_model/non_results/vram_reserved_over_time.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
POLICY_COLORS = {"ppo": "#5B2A86", "remax": "#FF5C7A", "grpo": "#0097A7"}
TARGET_EXPERIMENT_FACETS = ("Llama Reward Function", "Llama Reward Model")
EXPERIMENT_DISPLAY = {
    "Llama Reward Function": "Llama-3.1-8B-Inst | reward function",
    "Llama Reward Model": "Llama-3.1-8B-Inst | reward model",
}
EXPERIMENT_LINESTYLE = {
    "Llama Reward Function": "-",
    "Llama Reward Model": "--",
}
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama Reward Function": "llama_new_baseline",
    "Llama Reward Model": "llama_rm_gsm8k",
}
LOGICAL_GROUP_PREFIXES_BY_FACET = {
    "Llama Reward Function": ("stage1_llama8b_",),
    "Llama Reward Model": ("llama8b_",),
}


def _experiment_facet(slurm_job_name: str, logical_run_group: str) -> str:
    slurm_text = str(slurm_job_name).strip().lower()
    logical_text = str(logical_run_group).strip().lower()
    for facet in TARGET_EXPERIMENT_FACETS:
        if (
            slurm_text == TARGET_SLURM_JOB_NAME_BY_FACET[facet]
            and logical_text.startswith(LOGICAL_GROUP_PREFIXES_BY_FACET[facet])
        ):
            return facet
    return "Other"


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")
    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()
    runs_df["experiment_facet"] = [
        _experiment_facet(slurm_job_name=slurm_job_name, logical_run_group=logical_run_group_value)
        for slurm_job_name, logical_run_group_value in zip(runs_df["slurm_job_name"], runs_df["logical_run_group"])
    ]
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["experiment_facet"].isin(TARGET_EXPERIMENT_FACETS)
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )
    selected = runs_df[non_rollout_knob_mask & target_mask & checkpoint_mask][
        ["run_id", "policy_norm", "experiment_facet"]
    ].drop_duplicates()
    if selected.empty:
        raise ValueError("No reward-mechanism VRAM runs selected.")
    return selected


def main() -> None:
    selected_runs = _select_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    step_fact, _ = load_view("step_fact_view")
    needed = ["run_id", "global_step_canonical", "max_memory_reserved_gb"]
    steps = step_fact[step_fact["run_id"].astype(str).isin(selected_run_ids)][needed].copy()
    steps = steps.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    steps["global_step_canonical"] = pd.to_numeric(steps["global_step_canonical"], errors="coerce")
    steps["max_memory_reserved_gb"] = pd.to_numeric(steps["max_memory_reserved_gb"], errors="coerce")
    steps = steps.dropna(subset=["global_step_canonical", "max_memory_reserved_gb"]).copy()
    steps["global_step_canonical"] = steps["global_step_canonical"].astype(int)
    steps = steps.sort_values(["experiment_facet", "policy_norm", "global_step_canonical"])

    print("Reserved VRAM rows used:")
    print(
        steps[
            ["run_id", "experiment_facet", "policy_norm", "global_step_canonical", "max_memory_reserved_gb"]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.4))
    for facet in TARGET_EXPERIMENT_FACETS:
        for policy in TARGET_POLICIES:
            sub = steps[(steps["experiment_facet"] == facet) & (steps["policy_norm"] == policy)].copy()
            if sub.empty:
                continue
            ax.plot(
                sub["global_step_canonical"],
                sub["max_memory_reserved_gb"],
                linewidth=2.0,
                color=POLICY_COLORS[policy],
                linestyle=EXPERIMENT_LINESTYLE[facet],
                alpha=0.95,
            )

    ax.set_title("Reserved VRAM by Reward Mechanism", fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reserved VRAM (GB)")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    policy_handles = [
        Line2D([0], [0], color=POLICY_COLORS[policy], linewidth=2.4, label=POLICY_DISPLAY[policy])
        for policy in TARGET_POLICIES
    ]
    experiment_handles = [
        Line2D([0], [0], color="#444444", linestyle=EXPERIMENT_LINESTYLE[facet], linewidth=2.4, label=EXPERIMENT_DISPLAY[facet])
        for facet in TARGET_EXPERIMENT_FACETS
    ]
    fig.legend(policy_handles, [h.get_label() for h in policy_handles], title="Policy", frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.34, 0.94))
    fig.legend(experiment_handles, [h.get_label() for h in experiment_handles], title="Experiment", frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.74, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
