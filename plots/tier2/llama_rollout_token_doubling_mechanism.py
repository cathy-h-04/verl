"""Llama rollout-token doubling test: outcome + mechanism evidence (bar charts).

Panel A:
- rollout_j_per_output_token_mean vs rollout_max_batched_tokens (grouped bars by policy)

Panel B:
- sync_efficiency_mean and straggler_ratio_mean together (dual-axis grouped bars)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier2/llama_rollout_token_doubling_mechanism.png")

TARGET_POLICIES = ("ppo", "remax")
TARGET_TOKENS = (8192, 16384)
POLICY_COLORS = {"ppo": "#1f77b4", "remax": "#ff7f0e"}
POLICY_OFFSETS = {"ppo": -0.18, "remax": 0.18}


def _norm_policy(policy: str) -> str:
    return str(policy).strip().lower().replace("remx", "remax")


def _is_llama(model: str) -> bool:
    return "llama" in str(model).lower()


def _load_integrity_gated_counts() -> pd.DataFrame:
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
    runs["policy_norm"] = runs["policy"].map(_norm_policy)
    runs["is_llama"] = runs["model"].map(_is_llama)
    runs["rollout_max_batched_tokens"] = pd.to_numeric(runs["rollout_max_batched_tokens"], errors="coerce").astype("Int64")
    runs["is_checkpoint_continuation"] = runs["is_checkpoint_continuation"].fillna(False).astype(bool)
    runs["join_coverage_rate"] = pd.to_numeric(runs["join_coverage_rate"], errors="coerce")
    runs["phase_boundary_integrity_rate"] = pd.to_numeric(runs["phase_boundary_integrity_rate"], errors="coerce")

    outlier_cols = [c for c in runs.columns if "outlier" in c.lower()]
    if outlier_cols:
        print(f"outlier-related columns in run_summary_view: {outlier_cols}")
    else:
        print("note: no outlier-count columns in run_summary_view; skipping high-outlier exclusion gate.")

    mask = (
        runs["is_llama"]
        & runs["policy_norm"].isin(TARGET_POLICIES)
        & runs["rollout_max_batched_tokens"].isin(TARGET_TOKENS)
        & (~runs["is_checkpoint_continuation"])
        & (runs["join_coverage_rate"] == 1.0)
        & (runs["phase_boundary_integrity_rate"] == 1.0)
    )
    gated = runs.loc[mask].copy()
    grouped = (
        gated.groupby(["policy_norm", "rollout_max_batched_tokens"], dropna=False)["run_id"]
        .nunique()
        .rename("integrity_gated_run_count")
        .reset_index()
    )
    return grouped


def _load_comparison_data() -> pd.DataFrame:
    comp, _ = load_view("comparison_view")
    required = [
        "policy",
        "model",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "n_runs",
        "rollout_j_per_output_token_mean",
        "sync_efficiency_mean",
        "straggler_ratio_mean",
    ]
    missing = [c for c in required if c not in comp.columns]
    if missing:
        raise ValueError(f"comparison_view missing required columns: {missing}")

    out = comp.copy()
    out["policy_norm"] = out["policy"].map(_norm_policy)
    out["is_llama"] = out["model"].map(_is_llama)
    out["rollout_max_batched_tokens"] = pd.to_numeric(out["rollout_max_batched_tokens"], errors="coerce").astype("Int64")
    out["is_checkpoint_continuation"] = out["is_checkpoint_continuation"].fillna(False).astype(bool)
    out["n_runs_int"] = pd.to_numeric(out["n_runs"], errors="coerce").fillna(0).astype(int)
    out["rollout_j_per_output_token_mean"] = pd.to_numeric(out["rollout_j_per_output_token_mean"], errors="coerce")
    out["sync_efficiency_mean"] = pd.to_numeric(out["sync_efficiency_mean"], errors="coerce")
    out["straggler_ratio_mean"] = pd.to_numeric(out["straggler_ratio_mean"], errors="coerce")
    if "mfu_actor_mean" in out.columns:
        out["mfu_actor_mean"] = pd.to_numeric(out["mfu_actor_mean"], errors="coerce")
    if "phase_energy_share_mean_rollout" in out.columns:
        out["phase_energy_share_mean_rollout"] = pd.to_numeric(out["phase_energy_share_mean_rollout"], errors="coerce")

    out = out[
        out["is_llama"]
        & out["policy_norm"].isin(TARGET_POLICIES)
        & out["rollout_max_batched_tokens"].isin(TARGET_TOKENS)
        & (~out["is_checkpoint_continuation"])
    ].copy()

    bad = out[~out["sync_efficiency_mean"].between(0.0, 1.0, inclusive="both")]
    if not bad.empty:
        print("warning: sync_efficiency_mean has out-of-range values:")
        print(bad[["policy_norm", "rollout_max_batched_tokens", "sync_efficiency_mean"]].to_string(index=False))

    return out


def main() -> None:
    gated_counts = _load_integrity_gated_counts()
    comp = _load_comparison_data()

    merged = comp.merge(gated_counts, on=["policy_norm", "rollout_max_batched_tokens"], how="left")
    merged["integrity_gated_run_count"] = merged["integrity_gated_run_count"].fillna(0).astype(int)

    merged = merged[merged["n_runs_int"] == merged["integrity_gated_run_count"]].copy()
    if merged.empty:
        raise ValueError("All rows dropped after integrity-count alignment with run_summary_view.")

    expected = {(p, t) for p in TARGET_POLICIES for t in TARGET_TOKENS}
    got = set(zip(merged["policy_norm"], merged["rollout_max_batched_tokens"].astype(int)))
    if expected - got:
        available = (
            comp[["policy_norm", "rollout_max_batched_tokens", "n_runs_int"]]
            .sort_values(["policy_norm", "rollout_max_batched_tokens"])
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Missing required (policy, rollout_max_batched_tokens) groups: {sorted(expected-got)}. "
            f"Available after base filter: {available}"
        )

    merged = merged.sort_values(["policy_norm", "rollout_max_batched_tokens"]).copy()
    print("aggregated values used:")
    cols = [
        "policy_norm",
        "rollout_max_batched_tokens",
        "rollout_j_per_output_token_mean",
        "sync_efficiency_mean",
        "straggler_ratio_mean",
        "n_runs_int",
    ]
    for opt in ("mfu_actor_mean", "phase_energy_share_mean_rollout"):
        if opt in merged.columns:
            cols.append(opt)
    print(merged[cols].to_string(index=False))

    x_order = list(TARGET_TOKENS)
    x_map = {v: i for i, v in enumerate(x_order)}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), sharex=True)
    ax_a, ax_b = axes
    ax_b2 = ax_b.twinx()

    # Panel A: outcome grouped bars by policy.
    for policy in TARGET_POLICIES:
        sub = merged[merged["policy_norm"] == policy].sort_values("rollout_max_batched_tokens")
        xs = [x_map[int(v)] + POLICY_OFFSETS[policy] for v in sub["rollout_max_batched_tokens"].tolist()]
        ys = sub["rollout_j_per_output_token_mean"].tolist()
        ax_a.bar(
            xs,
            ys,
            width=0.32,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=policy.upper(),
            zorder=3,
        )

    ax_a.set_title("A) Rollout J / Output Token")
    ax_a.set_ylabel("rollout_j_per_output_token_mean")
    ax_a.set_xlabel("rollout_max_batched_tokens")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(title="policy", frameon=False, loc="upper left")

    # Panel B: mechanism grouped bars (sync + straggler together).
    sync_width = 0.12
    strag_width = 0.12
    for policy in TARGET_POLICIES:
        sub = merged[merged["policy_norm"] == policy].sort_values("rollout_max_batched_tokens")
        base_xs = [x_map[int(v)] + POLICY_OFFSETS[policy] for v in sub["rollout_max_batched_tokens"].tolist()]
        xs_sync = [x - 0.07 for x in base_xs]
        xs_strag = [x + 0.07 for x in base_xs]

        ax_b.bar(
            xs_sync,
            sub["sync_efficiency_mean"].tolist(),
            width=sync_width,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
            zorder=3,
        )
        ax_b2.bar(
            xs_strag,
            sub["straggler_ratio_mean"].tolist(),
            width=strag_width,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.35,
            hatch="//",
            zorder=2,
        )

    ax_b.set_title("B) Mechanism: Sync + Straggler")
    ax_b.set_ylabel("sync_efficiency_mean (higher is better)")
    ax_b2.set_ylabel("straggler_ratio_mean (lower is better)")
    ax_b.set_xlabel("rollout_max_batched_tokens")
    ax_b.grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_xticks([x_map[v] for v in x_order])
        ax.set_xticklabels([str(v) for v in x_order])

    policy_handles = [
        Patch(facecolor=POLICY_COLORS["ppo"], edgecolor="black", label="PPO"),
        Patch(facecolor=POLICY_COLORS["remax"], edgecolor="black", label="ReMax"),
    ]
    metric_handles = [
        Patch(facecolor="#666666", edgecolor="black", alpha=0.92, label="sync_efficiency_mean"),
        Patch(facecolor="#666666", edgecolor="black", alpha=0.35, hatch="//", label="straggler_ratio_mean"),
    ]
    ax_b.legend(handles=policy_handles + metric_handles, title="color=policy, style=metric", frameon=False, loc="upper right")

    fig.suptitle("Llama rollout token knob: outcome + mechanism", y=0.99)
    fig.text(
        0.5,
        0.94,
        "Llama, checkpoint_continuation=false; rollout_max_batched_tokens ∈ {8192, 16384}",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
