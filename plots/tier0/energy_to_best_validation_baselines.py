"""Combined energy + validation view for baseline cohort.

Bars encode energy_to_best_validation_j.
Overlaid markers encode best_validation_metric on a zoomed secondary axis.
Per-policy deltas are annotated for quick comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.data.loader import load_view


OUTPATH = Path("plots/out/figures/tier0/energy_to_best_validation_baselines.png")

TARGET_MODELS = ("Llama", "Qwen")
TARGET_POLICIES = ("ppo", "remax", "grpo")
ROLLOUT_MAX_BATCHED_TOKENS = 8192

MODEL_COLORS = {
    "Llama": "#4c78a8",
    "Qwen": "#f58518",
}


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def main() -> None:
    df, _ = load_view("run_summary_view")

    required = [
        "policy",
        "model",
        "rollout_max_batched_tokens",
        "is_checkpoint_continuation",
        "join_coverage_rate",
        "phase_boundary_integrity_rate",
        "energy_to_best_validation_j",
        "best_validation_metric",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")

    df = df.copy()
    df["policy_norm"] = df["policy"].astype(str).str.lower().replace({"remx": "remax"})
    df["model_facet"] = df["model"].map(_model_facet)
    df["rollout_max_batched_tokens"] = pd.to_numeric(df["rollout_max_batched_tokens"], errors="coerce")
    df["join_coverage_rate"] = pd.to_numeric(df["join_coverage_rate"], errors="coerce")
    df["phase_boundary_integrity_rate"] = pd.to_numeric(df["phase_boundary_integrity_rate"], errors="coerce")
    df["energy_to_best_validation_j"] = pd.to_numeric(df["energy_to_best_validation_j"], errors="coerce")
    df["best_validation_metric"] = pd.to_numeric(df["best_validation_metric"], errors="coerce")
    df["is_checkpoint_continuation"] = df["is_checkpoint_continuation"].fillna(False).astype(bool)

    mask = (
        df["model_facet"].isin(TARGET_MODELS)
        & df["policy_norm"].isin(TARGET_POLICIES)
        & (df["rollout_max_batched_tokens"] == float(ROLLOUT_MAX_BATCHED_TOKENS))
        & (~df["is_checkpoint_continuation"])
        & (df["join_coverage_rate"] == 1.0)
        & (df["phase_boundary_integrity_rate"] == 1.0)
    )
    plot_df = df.loc[mask].copy()
    plot_df = plot_df.dropna(subset=["energy_to_best_validation_j"])

    expected_pairs = {(m, p) for m in TARGET_MODELS for p in TARGET_POLICIES}
    got_pairs = set(zip(plot_df["model_facet"], plot_df["policy_norm"]))
    missing_pairs = sorted(expected_pairs - got_pairs)
    if missing_pairs:
        raise ValueError(f"Missing baseline pairs in run_summary_view: {missing_pairs}")

    print("rows used (model, policy, energy_to_best_validation_j, best_validation_metric):")
    print(
        plot_df[["model_facet", "policy_norm", "energy_to_best_validation_j", "best_validation_metric"]]
        .sort_values(["model_facet", "policy_norm"])
        .to_string(index=False)
    )

    x_positions = {policy: i for i, policy in enumerate(TARGET_POLICIES)}
    width = 0.36
    offsets = {"Llama": -width / 2, "Qwen": width / 2}

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax2 = ax.twinx()
    energy_max = 0.0
    validation_vals: list[float] = []

    for model in TARGET_MODELS:
        model_df = plot_df[plot_df["model_facet"] == model]
        xs: list[float] = []
        energy_vals: list[float] = []
        metric_vals: list[float] = []
        for policy in TARGET_POLICIES:
            row = model_df[model_df["policy_norm"] == policy].iloc[0]
            x = x_positions[policy] + offsets[model]
            e = float(row["energy_to_best_validation_j"])
            m = float(row["best_validation_metric"]) if pd.notna(row["best_validation_metric"]) else float("nan")
            xs.append(x)
            energy_vals.append(e)
            metric_vals.append(m)
            energy_max = max(energy_max, e)
            if pd.notna(m):
                validation_vals.append(m)

        bars = ax.bar(
            xs,
            energy_vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.82,
            label=f"{model} energy",
            zorder=2,
        )
        ax2.plot(
            xs,
            metric_vals,
            linestyle="None",
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=MODEL_COLORS[model],
            markeredgewidth=1.6,
            label=f"{model} validation",
            zorder=4,
        )
        for bar, mv in zip(bars, metric_vals):
            if pd.notna(mv):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    mv,
                    f"{mv:.3f}",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color="black",
                )

    ax.set_xticks([x_positions[p] for p in TARGET_POLICIES])
    ax.set_xticklabels([p.upper() for p in TARGET_POLICIES])
    ax.set_xlabel("policy")
    ax.set_ylabel("energy_to_best_validation_j")
    ax.set_ylim(0, energy_max * 1.20 if energy_max > 0 else 1.0)
    ax2.set_ylabel("best_validation_metric")
    if validation_vals:
        vmin, vmax = min(validation_vals), max(validation_vals)
        vpad = max((vmax - vmin) * 0.25, 0.01)
        ax2.set_ylim(vmin - vpad, vmax + vpad)
    ax.set_title("Energy to Best Validation and Validation Score (Baseline Cohort)")
    ax.grid(axis="y", alpha=0.2)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, title="series", frameon=False, loc="upper right")
    fig.tight_layout()

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
