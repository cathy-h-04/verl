"""Rollout output-token distribution comparison for PPO Llama:
GSM8K dataset run vs RLHF dataset run.

Figure:
- Left: ECDF of rollout_output_tokens_total per step
- Right: Box plot (with jitter) + mean/p50/p95 annotations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/rollout_output_token_distribution_ppo_llama_dataset_compare.png")
GSM8K_RUN_ID = "stage1_llama8b_ppo_20260301_075906"
RLHF_RUN_ID = "llama31_8b_smoke_test_rlhf_ff_20260304_192533"
RUN_ORDER = [GSM8K_RUN_ID, RLHF_RUN_ID]
RUN_LABELS = {
    GSM8K_RUN_ID: "GSM8K dataset",
    RLHF_RUN_ID: "RLHF dataset",
}
RUN_COLORS = {
    GSM8K_RUN_ID: "#1f77b4",
    RLHF_RUN_ID: "#d62728",
}


def _ecdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if vals.size == 0:
        return np.array([]), np.array([])
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def main() -> None:
    # Step eligibility from canonical analysis filter.
    step_fact, _ = load_view("step_fact_view")
    eligible = step_fact[step_fact["run_id"].astype(str).isin(RUN_ORDER)].copy()
    before = eligible.copy()
    eligible = apply_analysis_ok(eligible)
    print(f"filtering={explain_filtering(before, eligible)}")
    eligible = eligible[["run_id", "global_step_canonical"]].drop_duplicates()

    # Use tokens view for rollout output-token total signal.
    tok, _ = load_view("tokens_and_steps")
    needed = [
        "run_id",
        "global_step_canonical",
        "rollout_output_tokens_total",
    ]
    missing = [c for c in needed if c not in tok.columns]
    if missing:
        raise ValueError(f"tokens_and_steps missing required columns: {missing}")

    df = tok[tok["run_id"].astype(str).isin(RUN_ORDER)][needed].copy()
    df = df.merge(eligible, on=["run_id", "global_step_canonical"], how="inner")
    df["rollout_output_tokens_total"] = pd.to_numeric(df["rollout_output_tokens_total"], errors="coerce")
    df = df.dropna(subset=["rollout_output_tokens_total"]).copy()
    df = df[df["rollout_output_tokens_total"] >= 0].copy()
    if df.empty:
        raise ValueError("No rollout_output_tokens_total rows after filtering.")

    df["dataset"] = df["run_id"].map(RUN_LABELS)

    print("sample counts and tail stats:")
    stats = (
        df.groupby("run_id", dropna=False)["rollout_output_tokens_total"]
        .agg(
            n_steps="size",
            mean_tokens="mean",
            p50_tokens=lambda s: s.quantile(0.5),
            p95_tokens=lambda s: s.quantile(0.95),
            std="std",
        )
        .reset_index()
    )
    stats["dataset"] = stats["run_id"].map(RUN_LABELS)
    print(stats[["run_id", "dataset", "n_steps", "mean_tokens", "p50_tokens", "p95_tokens", "std"]].to_string(index=False))

    fig, (ax_ecdf, ax_box) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: ECDF lines.
    for run_id in RUN_ORDER:
        vals = df.loc[df["run_id"] == run_id, "rollout_output_tokens_total"].to_numpy(dtype=float)
        x, y = _ecdf(vals)
        if x.size == 0:
            continue
        ax_ecdf.step(
            x,
            y,
            where="post",
            linewidth=2.2,
            color=RUN_COLORS[run_id],
            label=RUN_LABELS[run_id],
        )
    ax_ecdf.set_xlabel("rollout_output_tokens_total (tokens per step)")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.set_ylim(0, 1.01)
    ax_ecdf.grid(alpha=0.2)
    ax_ecdf.set_title("Output Token Total ECDF")
    ax_ecdf.legend(frameon=False, loc="lower right")

    # Right: box + jitter + mean/p50/p95 annotations.
    box_data = [df.loc[df["run_id"] == rid, "rollout_output_tokens_total"].to_numpy(dtype=float) for rid in RUN_ORDER]
    bp = ax_box.boxplot(
        box_data,
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.1},
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35, "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    for patch, rid in zip(bp["boxes"], RUN_ORDER):
        patch.set_facecolor(RUN_COLORS[rid])
        patch.set_alpha(0.25)

    rng = np.random.default_rng(7)
    for i, rid in enumerate(RUN_ORDER, start=1):
        vals = df.loc[df["run_id"] == rid, "rollout_output_tokens_total"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        x_jitter = i + rng.uniform(-0.08, 0.08, size=vals.size)
        ax_box.scatter(x_jitter, vals, s=10, alpha=0.25, color=RUN_COLORS[rid], edgecolors="none")
        s = stats[stats["run_id"] == rid].iloc[0]
        ax_box.text(
            i,
            float(s["mean_tokens"]) * 1.02,
            f"mean={s['mean_tokens']:.0f}\np50={s['p50_tokens']:.0f}\np95={s['p95_tokens']:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=RUN_COLORS[rid],
        )

    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels([RUN_LABELS[r] for r in RUN_ORDER])
    ax_box.set_ylabel("rollout_output_tokens_total (tokens per step)")
    ax_box.grid(axis="y", alpha=0.2)
    ax_box.set_title("Output Token Total Distribution (Box)")

    fig.suptitle("Rollout Output Token Distribution (PPO Llama): GSM8K vs RLHF dataset", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
