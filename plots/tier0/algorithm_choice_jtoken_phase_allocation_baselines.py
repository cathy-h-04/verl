"""Algorithm choice effects on J/token and phase energy allocation (baseline strict knobs).

Panel A:
- grouped bars of overall_j_per_output_token_mean by policy, color by model.

Panel B:
- stacked bars of phase_energy_share_mean_* for each (model, policy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier0/algorithm_choice_jtoken_phase_allocation_baselines.png")

TARGET_MODELS = ("Llama", "Qwen")
TARGET_POLICIES = ("ppo", "remax", "grpo")
ROLLOUT_MAX_BATCHED_TOKENS = 8192
REQUIRE_CHECKPOINT_CONTINUATION_FALSE = True

# Keep phase colors consistent with phase_dominance_map_baselines.py.
PHASE_COLORS = {
    "rollout": "#1f77b4",
    "training": "#ff7f0e",
    "rl_policy": "#2ca02c",
    "validation": "#d62728",
    "other": "#7f7f7f",
}
MODEL_COLORS = {
    "Llama": "#4c78a8",
    "Qwen": "#f58518",
}
PHASE_STACK_ORDER = ["rollout", "training", "rl_policy"]


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _load_comparison_baselines() -> pd.DataFrame:
    df, _ = load_view("comparison_view")

    needed = [
        "policy",
        "model",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "n_runs",
        "overall_j_per_output_token_mean",
        "phase_energy_share_mean_rollout",
        "phase_energy_share_mean_training",
        "phase_energy_share_mean_rl_policy",
        "phase_energy_share_mean_validation",
        "phase_energy_share_mean_other",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"comparison_view missing required columns: {missing}")

    out = df.copy()
    out["policy_norm"] = out["policy"].astype(str).str.lower().replace({"remx": "remax"})
    out["model_facet"] = out["model"].map(_model_facet)
    out["is_checkpoint_continuation"] = out["is_checkpoint_continuation"].fillna(False).astype(bool)
    out["rollout_max_batched_tokens"] = pd.to_numeric(out["rollout_max_batched_tokens"], errors="coerce")
    out["overall_j_per_output_token_mean"] = pd.to_numeric(out["overall_j_per_output_token_mean"], errors="coerce")
    out["n_runs"] = pd.to_numeric(out["n_runs"], errors="coerce")

    mask = (
        out["model_facet"].isin(TARGET_MODELS)
        & out["policy_norm"].isin(TARGET_POLICIES)
        & (out["rollout_max_batched_tokens"] == float(ROLLOUT_MAX_BATCHED_TOKENS))
    )
    if REQUIRE_CHECKPOINT_CONTINUATION_FALSE:
        mask &= ~out["is_checkpoint_continuation"]

    out = out.loc[mask].copy()
    return out


def _load_integrity_gated_counts() -> pd.DataFrame:
    runs, _ = load_view("run_summary_view")
    needed = [
        "run_id",
        "policy",
        "model",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
    ]
    missing = [c for c in needed if c not in runs.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns for integrity gating: {missing}")

    runs = runs.copy()
    runs["policy_norm"] = runs["policy"].astype(str).str.lower().replace({"remx": "remax"})
    runs["model_facet"] = runs["model"].map(_model_facet)
    runs["rollout_max_batched_tokens"] = pd.to_numeric(runs["rollout_max_batched_tokens"], errors="coerce")
    runs["is_checkpoint_continuation"] = runs["is_checkpoint_continuation"].fillna(False).astype(bool)
    runs["join_coverage_rate"] = pd.to_numeric(runs["join_coverage_rate"], errors="coerce")
    runs["phase_boundary_integrity_rate"] = pd.to_numeric(runs["phase_boundary_integrity_rate"], errors="coerce")

    mask = (
        runs["model_facet"].isin(TARGET_MODELS)
        & runs["policy_norm"].isin(TARGET_POLICIES)
        & (runs["rollout_max_batched_tokens"] == float(ROLLOUT_MAX_BATCHED_TOKENS))
        & (runs["join_coverage_rate"] == 1.0)
        & (runs["phase_boundary_integrity_rate"] == 1.0)
    )
    if REQUIRE_CHECKPOINT_CONTINUATION_FALSE:
        mask &= ~runs["is_checkpoint_continuation"]
    gated = runs.loc[mask].copy()

    # Outlier-count field is not present in run_summary_view schema; log explicitly.
    print("note: run_summary_view has no outlier-count column; skipping high-outlier-count exclusion gate.")

    grouped = (
        gated.groupby(["model_facet", "policy_norm"], dropna=False)["run_id"]
        .nunique()
        .rename("integrity_gated_run_count")
        .reset_index()
    )
    return grouped


def main() -> None:
    comp = _load_comparison_baselines()
    if comp.empty:
        raise ValueError("No rows in comparison_view after strict baseline filters.")

    integrity_counts = _load_integrity_gated_counts()
    merged = comp.merge(integrity_counts, on=["model_facet", "policy_norm"], how="left")
    merged["integrity_gated_run_count"] = merged["integrity_gated_run_count"].fillna(0).astype(int)

    # Keep rows where aggregated comparison row run-count matches integrity-gated count.
    merged["n_runs_int"] = merged["n_runs"].fillna(0).astype(int)
    merged = merged[merged["n_runs_int"] == merged["integrity_gated_run_count"]].copy()

    if merged.empty:
        raise ValueError(
            "All comparison_view rows were dropped by integrity gating alignment. "
            "Check whether comparison_view aggregation includes runs failing run_summary integrity gates."
        )

    expected_pairs = {(m, p) for m in TARGET_MODELS for p in TARGET_POLICIES}
    actual_pairs = set(zip(merged["model_facet"], merged["policy_norm"]))
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if missing_pairs:
        raise ValueError(f"Missing required baseline pairs after filtering: {missing_pairs}")

    merged = merged.sort_values(["model_facet", "policy_norm"])
    print("rows used (model, policy, n_runs):")
    print(merged[["model_facet", "policy_norm", "n_runs_int"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: grouped bars.
    ax_a = axes[0]
    x = list(range(len(TARGET_POLICIES)))
    width = 0.36
    for i, model in enumerate(TARGET_MODELS):
        vals = []
        for policy in TARGET_POLICIES:
            row = merged[(merged["model_facet"] == model) & (merged["policy_norm"] == policy)].iloc[0]
            vals.append(float(row["overall_j_per_output_token_mean"]))
        offset = -width / 2 if i == 0 else width / 2
        ax_a.bar(
            [v + offset for v in x],
            vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.7,
            label=model,
            alpha=0.9,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([p.upper() for p in TARGET_POLICIES])
    ax_a.set_xlabel("policy")
    ax_a.set_ylabel("overall_j_per_output_token_mean")
    ax_a.set_title("A) Overall J/token by Algorithm")
    ax_a.grid(axis="y", alpha=0.2)
    ax_a.legend(title="model", frameon=False)

    # Panel B: stacked bars by (model, policy).
    ax_b = axes[1]
    bar_labels = [f"{row.model_facet}\n{row.policy_norm.upper()}" for row in merged.itertuples()]
    xpos = list(range(len(bar_labels)))
    bottom = [0.0] * len(bar_labels)
    for phase in PHASE_STACK_ORDER:
        col = f"phase_energy_share_mean_{phase}"
        vals = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).tolist()
        ax_b.bar(
            xpos,
            vals,
            bottom=bottom,
            color=PHASE_COLORS[phase],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.92,
            label=phase,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels(bar_labels)
    ax_b.set_xlabel("(model, policy)")
    ax_b.set_ylabel("phase energy share")
    ax_b.set_title("B) Phase Energy Allocation")
    ax_b.set_ylim(0, 1.02)
    ax_b.grid(axis="y", alpha=0.2)
    phase_handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="black", label=p) for p in PHASE_STACK_ORDER]
    ax_b.legend(handles=phase_handles, title="phase", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    fig.suptitle("Algorithm choice impacts: overall J/token and phase energy share", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
