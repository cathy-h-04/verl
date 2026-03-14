"""Rollout response-length ECDF by policy and dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plots.plotting.style import savefig_paper
from plots.task.line_style import DATASET_ALPHAS, DATASET_LINESTYLES
from plots.task.non_results.response_and_prompt_length import (
    DATASET_DISPLAY,
    POLICY_DISPLAY,
    POLICY_ORDER,
    TARGET_DATASETS,
    _load_length_df,
    _select_runs,
)


OUTPATH = Path("plots/out/task/response_length_ecdf_by_policy.png")
POLICY_COLORS = {"ppo": "#5B2A86", "remax": "#FF5C7A", "grpo": "#0097A7"}


def _ecdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def main() -> None:
    selected_runs = _select_runs()
    df = _load_length_df(selected_runs)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.0))

    for policy in POLICY_ORDER:
        psub = df[df["policy_norm"] == policy].copy()
        for dataset in TARGET_DATASETS:
            vals = psub[psub["dataset_group"] == dataset]["response_length_tokens"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            x, y = _ecdf(vals)
            ax.step(
                x,
                y,
                where="post",
                linewidth=2.2,
                color=POLICY_COLORS[policy],
                linestyle=DATASET_LINESTYLES[dataset],
                alpha=DATASET_ALPHAS[dataset],
            )

    ax.set_xlabel("Mean rollout response length (tokens)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(axis="both", alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)

    policy_handles = [
        Line2D([0], [0], color=POLICY_COLORS[policy], linewidth=2.4, linestyle="-", label=POLICY_DISPLAY[policy])
        for policy in POLICY_ORDER
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.2,
            linestyle=DATASET_LINESTYLES[dataset],
            alpha=DATASET_ALPHAS[dataset],
            label=DATASET_DISPLAY[dataset],
        )
        for dataset in TARGET_DATASETS
    ]
    fig.legend(
        policy_handles,
        [h.get_label() for h in policy_handles],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.42, 0.97),
        fontsize=9,
    )
    fig.legend(
        dataset_handles,
        [h.get_label() for h in dataset_handles],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.82, 0.97),
        fontsize=9,
    )
    fig.suptitle(
        "Rollout Response Length ECDF by Policy and Dataset",
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
