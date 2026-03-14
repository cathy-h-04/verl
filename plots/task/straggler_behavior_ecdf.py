"""Straggler behavior comparison under fixed reward-model setup.

Creates policy-faceted ECDF plots of rollout straggler ratio for:
- gsm8k
- full-hh-rlhf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.task.line_style import DATASET_ALPHAS, DATASET_COLORS, DATASET_LINESTYLES

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/task/straggler_behavior_ecdf.png")
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
TARGET_SLURM_TO_DATASET = {
    "llama_rm_gsm8k": "gsm8k",
    "llama_rm_rlhf": "full-hh-rlhf",
}
DATASET_ORDER = ("gsm8k", "full-hh-rlhf")
DATASET_COLOR = {"gsm8k": "#1f77b4", "full-hh-rlhf": "#d62728"}


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    required = ["run_id", "policy", "logical_run_group"]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")
    if "slurm_job_name" not in runs.columns:
        raise ValueError("runs missing required column: slurm_job_name")

    df = run_summary.merge(runs[["run_id", "slurm_job_name"]], on="run_id", how="left", validate="one_to_one").copy()
    df["policy_norm"] = df["policy"].astype(str).str.lower()
    df["slurm_job_name_lc"] = df["slurm_job_name"].astype(str).str.lower()
    df["dataset_group"] = df["slurm_job_name_lc"].map(TARGET_SLURM_TO_DATASET)
    logical_group = df["logical_run_group"].astype(str).str.lower()

    target_mask = df["policy_norm"].isin(TARGET_POLICIES) & df["dataset_group"].isin(DATASET_ORDER)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    checkpoint_mask = (
        ~df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in df.columns
        else True
    )
    integrity_mask = (
        (pd.to_numeric(df["join_coverage_rate"], errors="coerce") == 1.0)
        & (pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce") == 1.0)
        if {"join_coverage_rate", "phase_boundary_integrity_rate"}.issubset(df.columns)
        else True
    )

    selected = df[target_mask & non_rollout_knob_mask & checkpoint_mask & integrity_mask].copy()
    if selected.empty:
        raise ValueError("No target runs selected for straggler comparison.")

    expected = {(d, p) for d in DATASET_ORDER for p in TARGET_POLICIES}
    observed = set(zip(selected["dataset_group"], selected["policy_norm"]))
    missing_pairs = expected - observed
    if missing_pairs:
        raise ValueError(f"Missing dataset-policy combinations: {sorted(missing_pairs)}")

    return selected[["run_id", "policy_norm", "dataset_group"]].drop_duplicates()


def _load_straggler_rows(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step, _ = load_view("step_fact_view")
    needed = ["run_id", "global_step_canonical", "straggler_ratio"]
    missing = [c for c in needed if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    out = step[step["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed].copy()
    before = out.copy()
    out = apply_analysis_ok(out)
    print(f"step filtering={explain_filtering(before, out)}")

    out["global_step_canonical"] = pd.to_numeric(out["global_step_canonical"], errors="coerce")
    out["straggler_ratio"] = pd.to_numeric(out["straggler_ratio"], errors="coerce")
    out = out.dropna(subset=["global_step_canonical", "straggler_ratio"]).copy()
    out["global_step_canonical"] = out["global_step_canonical"].astype(int)
    out = out[np.isfinite(out["straggler_ratio"])].copy()
    out = out[out["straggler_ratio"] >= 0].copy()

    out = out.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")
    return out


def _ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def main() -> None:
    selected_runs = _select_runs()
    print("selected runs:")
    print(selected_runs.sort_values(["dataset_group", "policy_norm", "run_id"]).to_string(index=False))

    df = _load_straggler_rows(selected_runs)
    if df.empty:
        raise ValueError("No straggler rows available after filters.")

    summary = (
        df.groupby(["policy_norm", "dataset_group"], dropna=False)["straggler_ratio"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p90=lambda s: float(s.quantile(0.90)),
            p95=lambda s: float(s.quantile(0.95)),
            p99=lambda s: float(s.quantile(0.99)),
        )
        .reset_index()
    )
    print("\nstraggler summary by policy,dataset:")
    print(summary.sort_values(["policy_norm", "dataset_group"]).to_string(index=False))

    fig, axes = plt.subplots(1, len(TARGET_POLICIES), figsize=(17.6, 5.2), sharey=True)
    if len(TARGET_POLICIES) == 1:
        axes = [axes]

    xmax = float(df["straggler_ratio"].quantile(0.995))
    xmax = max(xmax, 1e-6)

    for ax, policy in zip(axes, TARGET_POLICIES):
        psub = df[df["policy_norm"] == policy].copy()
        for dataset in DATASET_ORDER:
            sub = psub[psub["dataset_group"] == dataset]["straggler_ratio"].to_numpy(dtype=float)
            if sub.size == 0:
                continue
            x, y = _ecdf(sub)
            ax.plot(
                x,
                y,
                color=DATASET_COLORS[dataset],
                linestyle=DATASET_LINESTYLES[dataset],
                linewidth=2.6,
                alpha=DATASET_ALPHAS[dataset],
            )

        ax.set_title(POLICY_DISPLAY.get(policy, policy.upper()), fontweight="bold")
        ax.set_xlabel("Rollout Straggler Ratio")
        ax.set_xlim(0, xmax)
        ax.grid(alpha=0.23)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Empirical CDF")
    handles = [
        Line2D([0], [0], color=DATASET_COLORS["gsm8k"], linestyle=DATASET_LINESTYLES["gsm8k"], linewidth=2.8, alpha=DATASET_ALPHAS["gsm8k"], label="gsm8k"),
        Line2D([0], [0], color=DATASET_COLORS["full-hh-rlhf"], linestyle=DATASET_LINESTYLES["full-hh-rlhf"], linewidth=2.8, alpha=DATASET_ALPHAS["full-hh-rlhf"], label="full-hh-rlhf"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Straggler Behavior by Policy and Dataset", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
