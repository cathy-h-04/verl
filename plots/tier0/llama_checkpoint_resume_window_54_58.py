"""Llama checkpoint continuation vs baseline at late-epoch matched window (54-58).

Selection:
- model: meta-llama/Llama-3.1-8B-Instruct
- policy: ppo, remax, grpo (accepts remx alias)
- rollout_max_batched_tokens: 8192
- cohorts: baseline (is_checkpoint_continuation=False), continuation (True)

Computation:
- Step-level window filter: global_step_canonical in [54, 58]
- Apply standard step-level analysis filter (outlier/validation/integrity masks)
- Per-run mean over window, then cohort x policy mean of per-run means.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/llama_checkpoint_resume_window_54_58.png")
OUTPATH_ACCURACY = Path("plots/out/figures/tier0/llama_checkpoint_resume_accuracy_path_to_58.png")

MODEL_EXACT = "meta-llama/Llama-3.1-8B-Instruct"
POLICY_ORDER = ("ppo", "remax", "grpo")
ROLLOUT_MAX_BATCHED_TOKENS = 8192
WINDOW_START = 54
WINDOW_END = 58
BASELINE_RUN_IDS = {
    "stage1_llama8b_ppo_20260301_075906",
    "stage1_llama8b_remax_20260301_083423",
    "stage1_llama8b_grpo_20260301_090832",
}
CONTINUATION_RUN_IDS = {
    "stage2_llama8b_ppo_20260301_120523",
    "stage2_llama8b_remax_20260301_121545",
    "stage2_llama8b_grpo_20260301_122602",
}

COHORT_LABELS = {False: "baseline", True: "continuation"}
COHORT_ORDER = ("baseline", "continuation")
COHORT_COLORS = {"baseline": "#4c78a8", "continuation": "#f58518"}
REQUESTED_IMBALANCE_METRIC = "timing_dist_s/update_actor/imbalance"
FALLBACK_IMBALANCE_METRIC = "sync_efficiency"

METRIC_COLUMNS = [
    "step_j_per_output_token",
    "rollout_j_per_output_token",
    "train_j_per_effective_token",
    "mfu_actor",
    "straggler_ratio",
    REQUESTED_IMBALANCE_METRIC,
]
ACCURACY_METRIC_KEY = "val-core/openai/gsm8k/reward/mean@1"
METRIC_LABELS = {
    "step_j_per_output_token": "overall_j_per_output_token (step mean)",
    "rollout_j_per_output_token": "rollout_j_per_output_token",
    "train_j_per_effective_token": "train_j_per_effective_token",
    "mfu_actor": "mfu_actor",
    "straggler_ratio": "straggler_ratio (lower better)",
    REQUESTED_IMBALANCE_METRIC: "timing_dist_s/update_actor/imbalance (lower better)",
}


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

    integrity_ok = (
        (runs["join_coverage_rate"] == 1.0)
        & (runs["phase_boundary_integrity_rate"] == 1.0)
    )

    base_mask = (
        (runs["model"] == MODEL_EXACT)
        & runs["policy_norm"].isin(POLICY_ORDER)
        & (runs["rollout_max_batched_tokens"] == float(ROLLOUT_MAX_BATCHED_TOKENS))
        & integrity_ok
        & (~runs["is_checkpoint_continuation"])
        & runs["run_id"].isin(BASELINE_RUN_IDS)
    )
    continuation_mask = (
        (runs["model"] == MODEL_EXACT)
        & runs["policy_norm"].isin(POLICY_ORDER)
        & (runs["rollout_max_batched_tokens"] == float(ROLLOUT_MAX_BATCHED_TOKENS))
        & integrity_ok
        & runs["is_checkpoint_continuation"]
        & runs["run_id"].isin(CONTINUATION_RUN_IDS)
    )
    mask = base_mask | continuation_mask

    selected = runs.loc[mask, ["run_id", "policy_norm", "is_checkpoint_continuation"]].copy()
    selected["cohort"] = selected["is_checkpoint_continuation"].map(COHORT_LABELS)

    expected = {(p, c) for p in POLICY_ORDER for c in COHORT_ORDER}
    got = set(zip(selected["policy_norm"], selected["cohort"]))
    if expected - got:
        available = (
            runs[(runs["model"] == MODEL_EXACT)][["run_id", "policy_norm", "rollout_max_batched_tokens", "is_checkpoint_continuation"]]
            .sort_values(["policy_norm", "rollout_max_batched_tokens", "is_checkpoint_continuation"])
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Missing required policy/cohort groups: {sorted(expected-got)}. "
            f"Available Llama configs: {available}"
        )
    missing_baselines = sorted(BASELINE_RUN_IDS - set(selected.loc[selected["cohort"] == "baseline", "run_id"].tolist()))
    missing_continuations = sorted(
        CONTINUATION_RUN_IDS - set(selected.loc[selected["cohort"] == "continuation", "run_id"].tolist())
    )
    if missing_baselines:
        raise ValueError(f"Missing required baseline run_ids after filtering: {missing_baselines}")
    if missing_continuations:
        raise ValueError(f"Missing required continuation run_ids after filtering: {missing_continuations}")

    return selected


def main() -> None:
    selected_runs = _load_selected_runs()
    print("selected runs (policy, cohort):")
    print(
        selected_runs.groupby(["policy_norm", "cohort"], dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values(["policy_norm", "cohort"])
        .to_string(index=False)
    )

    step, _ = load_view("step_fact_view")
    if REQUESTED_IMBALANCE_METRIC not in step.columns and FALLBACK_IMBALANCE_METRIC in step.columns:
        step[REQUESTED_IMBALANCE_METRIC] = step[FALLBACK_IMBALANCE_METRIC]
        print(
            "warning: requested metric 'timing_dist_s/update_actor/imbalance' not found in step_fact_view; "
            "using sync_efficiency as fallback source."
        )
    required_step = ["run_id", "policy", "model", "global_step_canonical", *METRIC_COLUMNS]
    missing_step = [c for c in required_step if c not in step.columns]
    if missing_step:
        raise ValueError(f"step_fact_view missing required columns: {missing_step}")

    step = step[step["run_id"].astype(str).isin(selected_runs["run_id"])].copy()
    before_filter = step.copy()
    step = apply_analysis_ok(step)
    print(f"filtering={explain_filtering(before_filter, step)}")
    eligible_steps_all = step[["run_id", "global_step_canonical"]].drop_duplicates().copy()

    step["global_step_canonical"] = pd.to_numeric(step["global_step_canonical"], errors="coerce")
    step = step[(step["global_step_canonical"] >= WINDOW_START) & (step["global_step_canonical"] <= WINDOW_END)].copy()
    if step.empty:
        raise ValueError("No step rows remain in the target window [54,58] after filtering.")

    step = step.merge(selected_runs[["run_id", "policy_norm", "cohort"]], on="run_id", how="inner")
    for c in METRIC_COLUMNS:
        step[c] = pd.to_numeric(step[c], errors="coerce")

    # Per-run window means, then aggregate by policy x cohort.
    run_window = (
        step.groupby(["run_id", "policy_norm", "cohort"], dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    step_counts = (
        step.groupby(["run_id", "policy_norm", "cohort"], dropna=False)["global_step_canonical"]
        .nunique()
        .rename("n_window_steps")
        .reset_index()
    )
    run_window = run_window.merge(step_counts, on=["run_id", "policy_norm", "cohort"], how="left")

    if (run_window["n_window_steps"] < (WINDOW_END - WINDOW_START + 1)).any():
        print("warning: some runs have <5 window steps after filtering:")
        print(
            run_window[run_window["n_window_steps"] < (WINDOW_END - WINDOW_START + 1)][
                ["run_id", "policy_norm", "cohort", "n_window_steps"]
            ].to_string(index=False)
        )

    agg = (
        run_window.groupby(["policy_norm", "cohort"], dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    counts = (
        run_window.groupby(["policy_norm", "cohort"], dropna=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_steps_total=("n_window_steps", "sum"),
            n_steps_per_run_mean=("n_window_steps", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(counts, on=["policy_norm", "cohort"], how="left")

    expected = {(p, c) for p in POLICY_ORDER for c in COHORT_ORDER}
    got = set(zip(agg["policy_norm"], agg["cohort"]))
    if expected - got:
        raise ValueError(f"Missing aggregated policy/cohort groups after window filter: {sorted(expected-got)}")

    agg = agg.sort_values(["policy_norm", "cohort"]).copy()
    print("aggregated windowed values (used for plotting):")
    print(
        agg[
            [
                "policy_norm",
                "cohort",
                "n_runs",
                "n_steps_total",
                "n_steps_per_run_mean",
                *METRIC_COLUMNS,
            ]
        ].to_string(index=False)
    )

    # Plot: paired bars within each policy, separate subplot per metric.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()
    x_positions = {p: i for i, p in enumerate(POLICY_ORDER)}
    width = 0.34
    offsets = {"baseline": -width / 2, "continuation": width / 2}

    for idx, metric in enumerate(METRIC_COLUMNS):
        ax = axes[idx]
        vals_by_cohort: dict[str, list[float]] = {}
        xs_by_cohort: dict[str, list[float]] = {}
        for cohort in COHORT_ORDER:
            vals = []
            xs = []
            for policy in POLICY_ORDER:
                row = agg[(agg["policy_norm"] == policy) & (agg["cohort"] == cohort)].iloc[0]
                vals.append(float(row[metric]))
                xs.append(x_positions[policy] + offsets[cohort])
            vals_by_cohort[cohort] = vals
            xs_by_cohort[cohort] = xs
            ax.bar(
                xs,
                vals,
                width=width,
                color=COHORT_COLORS[cohort],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                label=cohort if idx == 0 else None,
            )

        # Annotate continuation bars with percent delta vs baseline (inside orange bar).
        base_vals = vals_by_cohort["baseline"]
        cont_vals = vals_by_cohort["continuation"]
        cont_xs = xs_by_cohort["continuation"]
        for x, base_v, cont_v in zip(cont_xs, base_vals, cont_vals):
            if base_v == 0:
                label = "n/a"
            else:
                delta_pct = (cont_v - base_v) / abs(base_v) * 100.0
                label = f"{delta_pct:+.1f}%"
            y_text = cont_v * 0.55
            ax.text(x, y_text, label, fontsize=8, ha="center", va="center", color="white", fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks([x_positions[p] for p in POLICY_ORDER])
        ax.set_xticklabels([p.upper() for p in POLICY_ORDER])

    axes[0].legend(title="cohort", frameon=False)
    for ax in axes[3:]:
        ax.set_xlabel("policy")

    fig.suptitle("Checkpoint Resume vs Baseline (Late-Epoch Matched Window)", y=0.99)
    fig.text(0.5, 0.95, "Window = iterations 54–58 (late-epoch matched)", ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.925, "Llama 8B, rollout_max_batched_tokens=8192", ha="center", va="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")

    # Accuracy path figure (up to step 58): baseline trajectory vs continuation checkpoint-step point.
    step_long, _ = load_view("step_metrics_long")
    required_long = ["run_id", "global_step_canonical", "metric_key", "metric_value_float"]
    missing_long = [c for c in required_long if c not in step_long.columns]
    if missing_long:
        raise ValueError(f"step_metrics_long missing required columns for accuracy plot: {missing_long}")

    acc = step_long[step_long["metric_key"] == ACCURACY_METRIC_KEY][required_long].copy()
    acc = acc[acc["run_id"].astype(str).isin(selected_runs["run_id"])].copy()
    acc["global_step_canonical"] = pd.to_numeric(acc["global_step_canonical"], errors="coerce")
    acc["metric_value_float"] = pd.to_numeric(acc["metric_value_float"], errors="coerce")
    acc = acc.dropna(subset=["global_step_canonical", "metric_value_float"]).copy()
    acc = acc[acc["global_step_canonical"] <= WINDOW_END].copy()
    acc = acc.merge(selected_runs[["run_id", "policy_norm", "cohort"]], on="run_id", how="inner")

    if acc.empty:
        print("warning: no accuracy rows found after filtering; skipping accuracy path figure.")
        return

    acc_summary = (
        acc.groupby(["policy_norm", "cohort", "global_step_canonical"], dropna=False)["metric_value_float"]
        .mean()
        .reset_index()
        .sort_values(["policy_norm", "cohort", "global_step_canonical"])
    )
    print("accuracy rows by policy/cohort (n_runs, n_steps):")
    acc_counts = (
        acc.groupby(["policy_norm", "cohort"], dropna=False)
        .agg(n_runs=("run_id", "nunique"), n_steps=("global_step_canonical", "nunique"))
        .reset_index()
        .sort_values(["policy_norm", "cohort"])
    )
    print(acc_counts.to_string(index=False))
    print("accuracy series used for plotting:")
    print(acc_summary.to_string(index=False))

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    policy_to_ax = dict(zip(POLICY_ORDER, axes2))
    for policy in POLICY_ORDER:
        ax = policy_to_ax[policy]
        for cohort in COHORT_ORDER:
            sub = acc_summary[(acc_summary["policy_norm"] == policy) & (acc_summary["cohort"] == cohort)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("global_step_canonical")
            ax.plot(
                sub["global_step_canonical"],
                sub["metric_value_float"],
                marker="o",
                linewidth=1.8,
                markersize=5.0,
                color=COHORT_COLORS[cohort],
                label=cohort if policy == POLICY_ORDER[0] else None,
            )
            # Label final point for quick readout at/near iteration 58.
            last = sub.iloc[-1]
            ax.text(
                float(last["global_step_canonical"]) + 0.3,
                float(last["metric_value_float"]),
                f"{float(last['metric_value_float']):.3f}",
                fontsize=8,
                ha="left",
                va="center",
                color=COHORT_COLORS[cohort],
            )
        ax.set_title(policy.upper())
        ax.set_xlabel("global_step_canonical")
        ax.grid(alpha=0.2)
        ax.set_xlim(9, WINDOW_END + 2)

    axes2[0].set_ylabel("validation accuracy (gsm8k reward@1)")
    axes2[0].legend(title="cohort", frameon=False, loc="lower right")
    fig2.suptitle("Accuracy path to iteration 58: baseline vs checkpoint continuation", y=0.99)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPATH_ACCURACY.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(OUTPATH_ACCURACY, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {OUTPATH_ACCURACY}")


if __name__ == "__main__":
    main()
