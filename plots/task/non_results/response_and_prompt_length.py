"""Rollout response/prompt length variation by policy and dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/task/non_results/response_and_prompt_length.png")
TARGET_DATASETS = ("gsm8k", "rlhf-ff")
POLICY_ORDER = ("ppo", "remax", "grpo")
DATASET_DISPLAY = {"gsm8k": "gsm8k", "rlhf-ff": "full-hh-rlhf"}
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
DATASET_COLORS = {"gsm8k": "#295894", "rlhf-ff": "#D04A1C"}
BAR_WIDTH = 0.32
JITTER_ALPHA = 0.30
JITTER_SIZE = 3.5


def _select_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    required = ["run_id", "policy", "dataset"]
    missing = [c for c in required if c not in run_summary.columns]
    if missing:
        raise ValueError(f"run_summary_view missing required columns: {missing}")

    runs = run_summary[required].drop_duplicates().copy()
    runs["policy_norm"] = runs["policy"].astype(str).str.lower()
    runs["dataset_group"] = runs["dataset"].astype(str).str.lower()
    if "is_checkpoint_continuation" in run_summary.columns:
        runs = runs.merge(
            run_summary[["run_id", "is_checkpoint_continuation"]].drop_duplicates(),
            on="run_id",
            how="left",
        )
        runs = runs[~runs["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()

    runs = runs[
        runs["policy_norm"].isin(POLICY_ORDER) & runs["dataset_group"].isin(TARGET_DATASETS)
    ][["run_id", "policy_norm", "dataset_group"]].drop_duplicates()
    if runs.empty:
        raise ValueError("No task-comparison runs selected for response-length plot.")
    return runs


def _load_length_df(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step_fact, _ = load_view("step_fact_view")
    needed_step = ["run_id", "global_step_canonical"]
    missing_step = [c for c in needed_step if c not in step_fact.columns]
    if missing_step:
        raise ValueError(f"step_fact_view missing required columns: {missing_step}")

    eligible = step_fact[step_fact["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))].copy()
    before = eligible.copy()
    eligible = apply_analysis_ok(eligible)
    print(f"step filtering={explain_filtering(before, eligible)}")
    eligible = eligible[["run_id", "global_step_canonical"]].drop_duplicates()

    tokens, _ = load_view("tokens_and_steps")
    needed_tok = [
        "run_id",
        "global_step_canonical",
        "rollout_num_sequences",
        "rollout_prompt_tokens_total",
        "rollout_output_tokens_total",
        "rollout_mean_output_len",
    ]
    missing_tok = [c for c in needed_tok if c not in tokens.columns]
    if missing_tok:
        raise ValueError(f"tokens_and_steps missing required columns: {missing_tok}")

    df = tokens[tokens["run_id"].astype(str).isin(selected_runs["run_id"].astype(str))][needed_tok].copy()
    df = df.merge(eligible, on=["run_id", "global_step_canonical"], how="inner")
    df = df.merge(selected_runs, on="run_id", how="inner", validate="many_to_one")

    for col in ["rollout_num_sequences", "rollout_prompt_tokens_total", "rollout_output_tokens_total", "rollout_mean_output_len"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["rollout_num_sequences"] > 0].copy()

    fallback = df["rollout_output_tokens_total"] / df["rollout_num_sequences"]
    df["response_length_tokens"] = df["rollout_mean_output_len"].where(df["rollout_mean_output_len"].notna(), fallback)
    df["prompt_length_tokens"] = df["rollout_prompt_tokens_total"] / df["rollout_num_sequences"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["response_length_tokens", "prompt_length_tokens"]).copy()
    df = df[(df["response_length_tokens"] > 0) & (df["prompt_length_tokens"] > 0)].copy()
    if df.empty:
        raise ValueError("No response/prompt-length rows remained after filtering.")
    return df


def _draw_box_row(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    positions = np.arange(len(POLICY_ORDER), dtype=float)
    offset = BAR_WIDTH / 2.0

    for dataset_i, dataset in enumerate(TARGET_DATASETS):
        pos = positions + (-offset if dataset_i == 0 else offset)
        grouped = []
        valid_pos = []
        valid_policies = []
        for i, policy in enumerate(POLICY_ORDER):
            vals = df[(df["policy_norm"] == policy) & (df["dataset_group"] == dataset)][metric].to_list()
            if vals:
                grouped.append(vals)
                valid_pos.append(pos[i])
                valid_policies.append(policy)
        if not grouped:
            continue

        bp = ax.boxplot(
            grouped,
            positions=valid_pos,
            widths=BAR_WIDTH,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
            boxprops={"edgecolor": "black", "linewidth": 0.7},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(DATASET_COLORS[dataset])
            patch.set_alpha(1.0)

        rng = np.random.default_rng(42 + dataset_i)
        for xpos, policy in zip(valid_pos, valid_policies):
            pts = df[(df["policy_norm"] == policy) & (df["dataset_group"] == dataset)][metric].to_numpy()
            if pts.size == 0:
                continue
            jx = xpos + rng.uniform(-BAR_WIDTH * 0.4, BAR_WIDTH * 0.4, size=len(pts))
            ax.scatter(jx, pts, s=JITTER_SIZE, color=DATASET_COLORS[dataset], alpha=JITTER_ALPHA, zorder=4, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([POLICY_DISPLAY[p] for p in POLICY_ORDER])
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)


def main() -> None:
    selected_runs = _select_runs()
    df = _load_length_df(selected_runs)

    response_summary = (
        df.groupby(["policy_norm", "dataset_group"], dropna=False)["response_length_tokens"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
        .sort_values(["policy_norm", "dataset_group"])
    )
    prompt_summary = (
        df.groupby(["policy_norm", "dataset_group"], dropna=False)["prompt_length_tokens"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
        .sort_values(["policy_norm", "dataset_group"])
    )
    print("prompt-length summary by policy and dataset:")
    print(prompt_summary.to_string(index=False))
    print("response-length summary by policy and dataset:")
    print(response_summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0))
    _draw_box_row(axes[0], df, "prompt_length_tokens", "Mean rollout prompt length (tokens)")
    _draw_box_row(axes[1], df, "response_length_tokens", "Mean rollout response length (tokens)")
    axes[0].set_title("Prompt Length", fontweight="bold")
    axes[1].set_title("Response Length", fontweight="bold")

    legend_handles = [
        Patch(facecolor=DATASET_COLORS[dataset], edgecolor="black", label=DATASET_DISPLAY[dataset])
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
        "Rollout Response and Prompt Length by Policy and Dataset",
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
