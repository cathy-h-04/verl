"""Mean rollout response length over time by policy and dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from plots.plotting.style import savefig_paper
from plots.task.line_style import (
    DATASET_ALPHAS,
    DATASET_COLORS,
    DATASET_LINESTYLES,
    DATASET_MARKERS,
)
from plots.task.non_results.response_and_prompt_length import (
    DATASET_DISPLAY,
    POLICY_DISPLAY,
    POLICY_ORDER,
    TARGET_DATASETS,
    _load_length_df,
    _select_runs,
)


OUTPATH = Path("plots/out/task/rollout_response_length_over_time.png")


def main() -> None:
    selected_runs = _select_runs()
    df = _load_length_df(selected_runs)

    summary = (
        df.groupby(["policy_norm", "dataset_group", "global_step_canonical"], dropna=False)["response_length_tokens"]
        .mean()
        .rename("mean_response_length_tokens")
        .reset_index()
        .sort_values(["policy_norm", "dataset_group", "global_step_canonical"])
    )
    print("mean rollout response length by policy, dataset, iteration:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(15.0, 4.8), sharey=True)
    if len(POLICY_ORDER) == 1:
        axes = [axes]

    for ax, policy in zip(axes, POLICY_ORDER):
        psub = summary[summary["policy_norm"] == policy].copy()
        for dataset in TARGET_DATASETS:
            sub = psub[psub["dataset_group"] == dataset].copy()
            if sub.empty:
                continue
            ax.plot(
                sub["global_step_canonical"],
                sub["mean_response_length_tokens"],
                color=DATASET_COLORS[dataset],
                linewidth=2.2,
                linestyle=DATASET_LINESTYLES[dataset],
                marker=DATASET_MARKERS[dataset],
                markersize=3.0,
                alpha=DATASET_ALPHAS[dataset],
            )

        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.grid(axis="both", alpha=0.22, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Mean rollout response length (tokens)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            linewidth=2.4,
            linestyle=DATASET_LINESTYLES[dataset],
            marker=DATASET_MARKERS[dataset],
            markersize=4.0,
            label=DATASET_DISPLAY[dataset],
            alpha=DATASET_ALPHAS[dataset],
        )
        for dataset in TARGET_DATASETS
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=9,
    )
    fig.suptitle(
        "Mean Rollout Response Length Over Time by Policy and Dataset",
        y=1.02,
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
