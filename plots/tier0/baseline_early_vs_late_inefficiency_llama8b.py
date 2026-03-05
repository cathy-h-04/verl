"""Baseline early-vs-late comparison (Llama 8B, 8192 rollout cap).

Windows are computed per run on eligible non-validation steps:
- early = first 10 iterations
- late = last 10 iterations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/baseline_early_vs_late_inefficiency_llama8b.png")

MODEL_EXACT = "meta-llama/Llama-3.1-8B-Instruct"
POLICY_ORDER = ("ppo", "remax", "grpo")
ROLLOUT_MAX_BATCHED_TOKENS = 8192
WINDOW_SIZE = 10
BASELINE_RUN_IDS = {
    "stage1_llama8b_ppo_20260301_075906",
    "stage1_llama8b_remax_20260301_083423",
    "stage1_llama8b_grpo_20260301_090832",
}

WINDOW_ORDER = ("early", "late")
WINDOW_COLORS = {"early": "#4c78a8", "late": "#f58518"}
REQUESTED_IMBALANCE_METRIC = "timing_dist_s/update_actor/imbalance"
FALLBACK_IMBALANCE_METRIC = "sync_efficiency"
TOTAL_OUTPUT_TOKENS_METRIC = "window_total_output_tokens"

METRICS = [
    ("step_total_energy_j", "step_total_energy_j"),
    ("step_rollout_output_tokens", "step_rollout_output_tokens"),
    ("step_train_tokens_est", "step_train_tokens_est"),
    (TOTAL_OUTPUT_TOKENS_METRIC, "total_output_tokens (window sum)"),
    ("straggler_ratio", "straggler_ratio (lower better)"),
    (REQUESTED_IMBALANCE_METRIC, "timing_dist_s/update_actor/imbalance (lower better)"),
]


def _norm_policy(policy: str) -> str:
    return str(policy).strip().lower().replace("remx", "remax")


def _load_selected_runs() -> pd.DataFrame:
    runs, _ = load_view("run_summary_view")
    required = [
        "run_id",
        "policy",
        "model",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")

    runs = runs.copy()
    runs["run_id"] = runs["run_id"].astype(str)
    runs["policy_norm"] = runs["policy"].map(_norm_policy)
    runs["rollout_max_batched_tokens"] = pd.to_numeric(runs["rollout_max_batched_tokens"], errors="coerce")
    runs["is_checkpoint_continuation"] = runs["is_checkpoint_continuation"].fillna(False).astype(bool)
    runs["join_coverage_rate"] = pd.to_numeric(runs["join_coverage_rate"], errors="coerce")
    runs["phase_boundary_integrity_rate"] = pd.to_numeric(runs["phase_boundary_integrity_rate"], errors="coerce")

    integrity_ok = (runs["join_coverage_rate"] == 1.0) & (runs["phase_boundary_integrity_rate"] == 1.0)
    mask = (
        (runs["model"] == MODEL_EXACT)
        & runs["policy_norm"].isin(POLICY_ORDER)
        & (runs["rollout_max_batched_tokens"] == float(ROLLOUT_MAX_BATCHED_TOKENS))
        & (~runs["is_checkpoint_continuation"])
        & runs["run_id"].isin(BASELINE_RUN_IDS)
        & integrity_ok
    )
    selected = runs.loc[mask, ["run_id", "policy_norm"]].copy()
    if selected.empty:
        raise ValueError("No baseline Llama 8B runs selected with required integrity and knob filters.")

    missing_policies = sorted(set(POLICY_ORDER) - set(selected["policy_norm"].unique().tolist()))
    if missing_policies:
        raise ValueError(f"Missing baseline runs for policies: {missing_policies}")
    missing_runs = sorted(BASELINE_RUN_IDS - set(selected["run_id"].tolist()))
    if missing_runs:
        raise ValueError(f"Missing required baseline run_ids after filtering: {missing_runs}")
    return selected


def main() -> None:
    selected_runs = _load_selected_runs()
    print("selected runs (baseline llama8b, 8192):")
    print(
        selected_runs.groupby("policy_norm", dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values("policy_norm")
        .to_string(index=False)
    )

    step, _ = load_view("step_fact_view")
    if REQUESTED_IMBALANCE_METRIC not in step.columns and FALLBACK_IMBALANCE_METRIC in step.columns:
        step[REQUESTED_IMBALANCE_METRIC] = step[FALLBACK_IMBALANCE_METRIC]
        print(
            "warning: requested metric 'timing_dist_s/update_actor/imbalance' not found in step_fact_view; "
            "using sync_efficiency as fallback source."
        )
    base_metric_cols = [m for m, _ in METRICS if m != TOTAL_OUTPUT_TOKENS_METRIC]
    required_cols = ["run_id", "policy", "model", "global_step_canonical", *base_metric_cols]
    missing = [c for c in required_cols if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    step = step[step["run_id"].astype(str).isin(selected_runs["run_id"])].copy()
    before_filter = step.copy()
    step = apply_analysis_ok(step)
    print(f"filtering={explain_filtering(before_filter, step)}")

    # Explicit non-validation guard.
    if "is_validation_step" in step.columns:
        step = step[~step["is_validation_step"].fillna(False).astype(bool)].copy()

    step["global_step_canonical"] = pd.to_numeric(step["global_step_canonical"], errors="coerce")
    for metric in base_metric_cols:
        step[metric] = pd.to_numeric(step[metric], errors="coerce")
    step = step.dropna(subset=["global_step_canonical"]).copy()
    step = step.merge(selected_runs, on="run_id", how="inner")

    per_run_records: list[dict[str, object]] = []
    dropped_runs: list[dict[str, object]] = []

    for run_id, run_df in step.groupby("run_id", dropna=False):
        run_df = run_df.sort_values("global_step_canonical").copy()
        n_steps = len(run_df)
        policy = str(run_df["policy_norm"].iloc[0])

        if n_steps < WINDOW_SIZE:
            dropped_runs.append({"run_id": str(run_id), "policy_norm": policy, "eligible_steps": int(n_steps)})
            continue

        early_df = run_df.head(WINDOW_SIZE)
        late_df = run_df.tail(WINDOW_SIZE)

        early_row: dict[str, object] = {"run_id": str(run_id), "policy_norm": policy, "window": "early", "n_steps": int(len(early_df))}
        late_row: dict[str, object] = {"run_id": str(run_id), "policy_norm": policy, "window": "late", "n_steps": int(len(late_df))}
        for metric, _ in METRICS:
            if metric == TOTAL_OUTPUT_TOKENS_METRIC:
                early_row[metric] = float(early_df["step_rollout_output_tokens"].sum())
                late_row[metric] = float(late_df["step_rollout_output_tokens"].sum())
            else:
                early_row[metric] = float(early_df[metric].mean())
                late_row[metric] = float(late_df[metric].mean())
        per_run_records.append(early_row)
        per_run_records.append(late_row)

    if dropped_runs:
        dropped_df = pd.DataFrame(dropped_runs)
        print(f"warning: dropped {len(dropped_df)} runs due to insufficient eligible steps (<{WINDOW_SIZE}):")
        print(dropped_df.sort_values(["policy_norm", "run_id"]).to_string(index=False))
    else:
        print("dropped_runs_due_to_insufficient_steps=0")

    if not per_run_records:
        raise ValueError("No runs left for aggregation after step-count checks.")

    per_run = pd.DataFrame(per_run_records)
    agg = (
        per_run.groupby(["policy_norm", "window"], dropna=False)[[m for m, _ in METRICS]]
        .mean()
        .reset_index()
    )
    counts = (
        per_run.groupby(["policy_norm", "window"], dropna=False)
        .agg(run_count=("run_id", "nunique"), step_count=("n_steps", "sum"))
        .reset_index()
    )
    agg = agg.merge(counts, on=["policy_norm", "window"], how="left")

    expected = {(p, w) for p in POLICY_ORDER for w in WINDOW_ORDER}
    got = set(zip(agg["policy_norm"], agg["window"]))
    if expected - got:
        raise ValueError(f"Missing policy/window groups after aggregation: {sorted(expected-got)}")

    agg = agg.sort_values(["policy_norm", "window"]).copy()
    print("policy_window_summary:")
    print(
        agg[
            [
                "policy_norm",
                "window",
                "run_count",
                "step_count",
                *[m for m, _ in METRICS],
            ]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    x_positions = {p: i for i, p in enumerate(POLICY_ORDER)}
    width = 0.34
    offsets = {"early": -width / 2, "late": width / 2}

    for idx, (metric, title) in enumerate(METRICS):
        ax = axes[idx]
        vals_by_window: dict[str, list[float]] = {}
        xs_by_window: dict[str, list[float]] = {}
        for window in WINDOW_ORDER:
            vals = []
            xs = []
            for policy in POLICY_ORDER:
                row = agg[(agg["policy_norm"] == policy) & (agg["window"] == window)].iloc[0]
                vals.append(float(row[metric]))
                xs.append(x_positions[policy] + offsets[window])
            vals_by_window[window] = vals
            xs_by_window[window] = xs
            ax.bar(
                xs,
                vals,
                width=width,
                color=WINDOW_COLORS[window],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                label=window if idx == 0 else None,
            )

        # Annotate late bars with percent delta relative to early (inside bar).
        early_vals = vals_by_window["early"]
        late_vals = vals_by_window["late"]
        late_xs = xs_by_window["late"]
        for x, early_v, late_v in zip(late_xs, early_vals, late_vals):
            if early_v == 0:
                label = "n/a"
            else:
                delta_pct = (late_v - early_v) / abs(early_v) * 100.0
                label = f"{delta_pct:+.1f}%"
            y_text = late_v * 0.55
            ax.text(x, y_text, label, fontsize=8, ha="center", va="center", color="white", fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks([x_positions[p] for p in POLICY_ORDER])
        ax.set_xticklabels([p.upper() for p in POLICY_ORDER])

    axes[0].legend(title="window", frameon=False)
    for ax in axes[3:]:
        ax.set_xlabel("policy")

    fig.suptitle("Baseline Early vs Late (Llama 8B): Joules and Tokens Kept Separate", y=0.99)
    fig.text(
        0.5,
        0.95,
        "Baseline only, rollout_max_batched_tokens=8192, windows = first/last 10 non-validation iterations per run",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
