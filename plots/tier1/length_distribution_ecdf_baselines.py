"""Length distribution impact (ECDF) for baseline runs.

Data source:
- tokens_and_steps: rollout_mean_output_len (fallback: rollout_output_tokens_total / rollout_num_sequences)

Plot:
- ECDF of step-level output length by policy (PPO/ReMax/GRPO), baseline runs only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/figures/tier1/length_distribution_ecdf_baselines_by_model.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
POLICY_COLORS = {
    "ppo": "#1f77b4",
    "remax": "#ff7f0e",
    "grpo": "#2ca02c",
}


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
    required = [
        "run_id",
        "policy",
        "model",
        "logical_run_group",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    df = run_summary.merge(runs[["run_id", "slurm_job_name"]], on="run_id", how="left", validate="one_to_one").copy()
    df["policy_norm"] = df["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    df["model_facet"] = df["model"].map(_model_facet)
    df["logical_run_group"] = df["logical_run_group"].astype(str).str.lower()
    df["slurm_job_name"] = df["slurm_job_name"].astype(str).str.lower()
    df["is_checkpoint_continuation"] = df["is_checkpoint_continuation"].fillna(False).astype(bool)
    df["join_coverage_rate"] = pd.to_numeric(df["join_coverage_rate"], errors="coerce")
    df["phase_boundary_integrity_rate"] = pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce")

    baseline_label_mask = df["logical_run_group"].str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    expected_slurm = df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_mask = df["slurm_job_name"] == expected_slurm
    integrity_mask = (df["join_coverage_rate"] == 1.0) & (df["phase_boundary_integrity_rate"] == 1.0)
    mask = (
        (~df["is_checkpoint_continuation"])
        & df["policy_norm"].isin(TARGET_POLICIES)
        & baseline_label_mask
        & slurm_mask
        & integrity_mask
    )
    selected = df.loc[mask, ["run_id", "policy_norm", "model_facet"]].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def _ecdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if vals.size == 0:
        return np.array([]), np.array([])
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def main() -> None:
    selected_runs = _select_baseline_runs()
    print("selected runs by policy:")
    print(
        selected_runs.groupby("policy_norm", dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values("policy_norm")
        .to_string(index=False)
    )

    # Step-level eligibility filtering.
    step_fact, _ = load_view("step_fact_view")
    eligible = step_fact[step_fact["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))].copy()
    eligible = apply_analysis_ok(eligible)
    eligible = eligible[["run_id", "global_step_canonical"]].drop_duplicates()

    tok, _ = load_view("tokens_and_steps")
    needed = [
        "run_id",
        "global_step_canonical",
        "rollout_num_sequences",
        "rollout_output_tokens_total",
        "rollout_mean_output_len",
    ]
    missing = [c for c in needed if c not in tok.columns]
    if missing:
        raise ValueError(f"tokens_and_steps missing required columns: {missing}")

    df = tok[tok["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    df = df.merge(eligible, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(selected_runs[["run_id", "policy_norm", "model_facet"]], on="run_id", how="inner")
    for c in ["rollout_num_sequences", "rollout_output_tokens_total", "rollout_mean_output_len"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["rollout_num_sequences"] > 0].copy()
    # Prefer explicit mean length; fallback to total/num_sequences.
    fallback = df["rollout_output_tokens_total"] / df["rollout_num_sequences"]
    df["output_len_step"] = df["rollout_mean_output_len"].where(df["rollout_mean_output_len"].notna(), fallback)
    df = df.dropna(subset=["output_len_step"]).copy()

    print("step samples by (model, policy):")
    print(
        df.groupby(["model_facet", "policy_norm"], dropna=False)
        .agg(
            n_steps=("output_len_step", "size"),
            mean_output_len=("output_len_step", "mean"),
            p95_output_len=("output_len_step", lambda s: s.quantile(0.95)),
        )
        .reset_index()
        .sort_values(["model_facet", "policy_norm"])
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, model in zip(axes, ("Llama", "Qwen")):
        mdf = df[df["model_facet"] == model].copy()
        for policy in TARGET_POLICIES:
            vals = mdf.loc[mdf["policy_norm"] == policy, "output_len_step"].to_numpy(dtype=float)
            x, y = _ecdf(vals)
            if x.size == 0:
                continue
            ax.step(x, y, where="post", linewidth=2.2, color=POLICY_COLORS[policy], label=policy.upper())

        ax.set_xlabel("output length (tokens per step; rollout mean output len)")
        ax.set_title(f"Length Distribution Impact (ECDF) - {model} Baseline Runs")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1.01)
        ax.legend(title="policy", frameon=False, loc="lower right")

    axes[0].set_ylabel("ECDF")
    fig.tight_layout()
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
