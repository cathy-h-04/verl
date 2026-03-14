"""Reward vs relative response-length bucket for task-comparison runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plots.plotting.style import savefig_paper
from plots.task.line_style import (
    DATASET_ALPHAS,
    DATASET_COLORS,
    DATASET_LINESTYLES,
    DATASET_MARKERS,
)
from plots.task.non_results.response_and_prompt_length import (
    DATASET_COLORS,
    DATASET_DISPLAY,
    POLICY_DISPLAY,
    POLICY_ORDER,
    TARGET_DATASETS,
    _select_runs,
)


OUTPATH = Path("plots/out/task/reward_by_response_length_bucket.png")
RESULTS_ROOTS = (
    Path("results/monitoring_val/reward_models_gsm8k"),
    Path("results/monitoring_val/reward_models_rlhf"),
)
QUARTILE_ORDER = ("q1", "q2", "q3", "q4")
QUARTILE_DISPLAY = {
    "q1": "Shortest\nQ1",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Longest\nQ4",
}
POINT_ALPHA = 0.20


def _resolve_run_dir(run_id: str) -> Path:
    for root in RESULTS_ROOTS:
        candidate = root / run_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find raw run directory for {run_id}")


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _load_plot_df(selected_runs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in selected_runs.itertuples(index=False):
        run_id = str(row.run_id)
        run_dir = _resolve_run_dir(run_id)
        jsonl_path = run_dir / f"{run_id}.jsonl"
        df = _load_jsonl(jsonl_path)
        if df.empty or "data" not in df.columns:
            continue

        flat_rows: list[dict[str, object]] = []
        for entry in df.itertuples(index=False):
            payload = getattr(entry, "data", None)
            if not isinstance(payload, dict):
                continue
            if str(payload.get("logging/record_scope", "")) != "iteration_summary":
                continue
            if bool(payload.get("logging/validation_logged", False)):
                continue
            flat_rows.append(
                {
                    "run_id": run_id,
                    "global_step_canonical": pd.to_numeric(
                        payload.get("training/global_step", getattr(entry, "step", None)),
                        errors="coerce",
                    ),
                    "response_length_mean": pd.to_numeric(payload.get("response_length/mean"), errors="coerce"),
                    "reward_mean": pd.to_numeric(payload.get("critic/rewards/mean"), errors="coerce"),
                    "policy_norm": row.policy_norm,
                    "dataset_group": row.dataset_group,
                }
            )
        if flat_rows:
            frames.append(pd.DataFrame(flat_rows))

    if not frames:
        raise ValueError("No raw iteration-summary reward rows found for the selected task runs.")

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df = plot_df.dropna(
        subset=["global_step_canonical", "response_length_mean", "reward_mean"]
    ).copy()
    plot_df["global_step_canonical"] = plot_df["global_step_canonical"].astype(int)
    return plot_df


def _assign_relative_length_quartiles(plot_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = plot_df.copy()
    ranks = plot_df.groupby(["policy_norm", "dataset_group"])["response_length_mean"].transform(
        lambda s: s.rank(method="first", pct=True)
    )
    plot_df["length_bucket"] = pd.cut(
        ranks,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=list(QUARTILE_ORDER),
        include_lowest=True,
    ).astype(str)
    plot_df = plot_df[plot_df["length_bucket"].isin(QUARTILE_ORDER)].copy()
    if plot_df.empty:
        raise ValueError("No rows remained after assigning response-length quartiles.")
    return plot_df


def main() -> None:
    selected_runs = _select_runs()
    plot_df = _assign_relative_length_quartiles(_load_plot_df(selected_runs))

    summary = (
        plot_df.groupby(["policy_norm", "dataset_group", "length_bucket"], dropna=False)
        .agg(
            n=("reward_mean", "size"),
            reward_mean=("reward_mean", "mean"),
            reward_median=("reward_mean", "median"),
            response_len_mean=("response_length_mean", "mean"),
            response_len_median=("response_length_mean", "median"),
        )
        .reset_index()
        .sort_values(["policy_norm", "dataset_group", "length_bucket"])
    )
    print("reward by within-dataset response-length quartile:")
    print(summary.to_string(index=False))

    trend_rows: list[dict[str, object]] = []
    for (policy, dataset), g in plot_df.groupby(["policy_norm", "dataset_group"], dropna=False):
        trend_rows.append(
            {
                "policy_norm": policy,
                "dataset_group": dataset,
                "corr": g["response_length_mean"].corr(g["reward_mean"]),
                "reward_delta_q4_q1": g.loc[g["length_bucket"] == "q4", "reward_mean"].mean()
                - g.loc[g["length_bucket"] == "q1", "reward_mean"].mean(),
            }
        )
    trend_summary = pd.DataFrame(trend_rows)
    print("within-dataset reward-length trend summary:")
    print(trend_summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(15.0, 4.9), sharey=True)
    if len(POLICY_ORDER) == 1:
        axes = [axes]
    x = np.arange(len(QUARTILE_ORDER), dtype=float)

    for ax, policy in zip(axes, POLICY_ORDER):
        psub = plot_df[plot_df["policy_norm"] == policy].copy()
        ssub = summary[summary["policy_norm"] == policy].copy()

        for dataset in TARGET_DATASETS:
            dsub = psub[psub["dataset_group"] == dataset].copy()
            dsum = ssub[ssub["dataset_group"] == dataset].set_index("length_bucket")
            if dsub.empty or dsum.empty:
                continue

            rng = np.random.default_rng(42 + hash((policy, dataset)) % 1000)
            for i, quartile in enumerate(QUARTILE_ORDER):
                pts = dsub[dsub["length_bucket"] == quartile]["reward_mean"].to_numpy(dtype=float)
                if pts.size == 0:
                    continue
                jx = x[i] + rng.uniform(-0.08, 0.08, size=pts.size)
                ax.scatter(
                    jx,
                    pts,
                    s=10.0,
                    color=DATASET_COLORS[dataset],
                    alpha=POINT_ALPHA,
                    linewidths=0,
                    zorder=2,
                )

            yvals = [float(dsum.loc[q, "reward_mean"]) for q in QUARTILE_ORDER if q in dsum.index]
            xvals = [x[i] for i, q in enumerate(QUARTILE_ORDER) if q in dsum.index]
            ax.plot(
                xvals,
                yvals,
                color=DATASET_COLORS[dataset],
                linestyle=DATASET_LINESTYLES[dataset],
                marker=DATASET_MARKERS[dataset],
                linewidth=2.0,
                markersize=5.5,
                alpha=DATASET_ALPHAS[dataset],
                zorder=4,
            )

            if "q4" in dsum.index:
                delta = float(dsum.loc["q4", "reward_mean"] - dsum.loc["q1", "reward_mean"])
                sign = "+" if delta >= 0 else ""
                ax.text(
                    x[-1] + 0.08,
                    float(dsum.loc["q4", "reward_mean"]),
                    f"{sign}{delta:.2f}",
                    color=DATASET_COLORS[dataset],
                    fontsize=8,
                    fontweight="bold",
                    va="center",
                    alpha=DATASET_ALPHAS[dataset],
                )

        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([QUARTILE_DISPLAY[q] for q in QUARTILE_ORDER])
        ax.set_xlabel("Response length quartile within dataset")
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Mean reward score")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_COLORS[dataset],
            linestyle=DATASET_LINESTYLES[dataset],
            marker=DATASET_MARKERS[dataset],
            linewidth=2.0,
            markersize=5.5,
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
        "Reward vs Response Length by Policy and Dataset",
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
