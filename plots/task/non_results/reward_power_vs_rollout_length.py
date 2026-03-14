"""Reward subphase power versus rollout response length for task runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/task/non_results/reward_power_vs_rollout_length.png")
RESULTS_ROOTS = (
    Path("results/monitoring_val/reward_models_gsm8k"),
    Path("results/monitoring_val/reward_models_rlhf"),
)
TARGET_DATASETS = ("gsm8k", "rlhf-ff")
POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
DATASET_DISPLAY = {"gsm8k": "gsm8k", "rlhf-ff": "full-hh-rlhf"}
DATASET_COLORS = {"gsm8k": "#295894", "rlhf-ff": "#D04A1C"}
POINT_ALPHA = 0.78
POINT_SIZE = 26.0


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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
        raise ValueError("No task-comparison runs selected.")
    return runs


def _resolve_run_dir(run_id: str) -> Path:
    for root in RESULTS_ROOTS:
        candidate = root / run_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find raw run directory for {run_id}")


def _load_tokens(run_id: str) -> pd.DataFrame:
    df = _load_jsonl(_resolve_run_dir(run_id) / "tokens_and_steps.jsonl")
    if df.empty:
        return pd.DataFrame(columns=["iteration", "rollout_mean_output_len"])
    for col in ["iteration", "rollout_mean_output_len"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["phase_name"] = df.get("phase_name", "").astype(str).str.lower()
    df["record_type"] = df.get("record_type", "").astype(str).str.upper()
    df = df[
        (df["phase_name"] == "rollout")
        & (df["record_type"] == "PERIODIC")
    ][["iteration", "rollout_mean_output_len"]].dropna().copy()
    df["iteration"] = df["iteration"].astype(int)
    return df.groupby("iteration", as_index=False)["rollout_mean_output_len"].last()


def _load_reward_durations(run_id: str) -> pd.DataFrame:
    path = _resolve_run_dir(run_id) / f"phase_timings_{run_id}.jsonl"
    df = _load_jsonl(path)
    if df.empty:
        return pd.DataFrame(columns=["iteration", "reward_duration_s", "rl_policy_subphase_time_s"])

    for col in ["iteration", "value"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["phase_name"] = df.get("phase_name", "").astype(str).str.lower()
    df["subphase_name"] = df.get("subphase_name", "").astype(str)
    df["metric_unit"] = df.get("metric_unit", "").astype(str)
    df = df[
        (df["phase_name"] == "rl_policy")
        & (df["metric_unit"] == "s")
        & df["subphase_name"].isin(["reward", "old_log_prob", "values", "adv"])
    ][["iteration", "subphase_name", "value"]].dropna().copy()
    df["iteration"] = df["iteration"].astype(int)

    reward = (
        df[df["subphase_name"] == "reward"]
        .groupby("iteration", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "reward_duration_s"})
    )
    total = (
        df.groupby("iteration", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "rl_policy_subphase_time_s"})
    )
    return reward.merge(total, on="iteration", how="outer")


def _load_rl_policy_energy(run_id: str) -> pd.DataFrame:
    df = _load_jsonl(_resolve_run_dir(run_id) / "nvml_boundary.jsonl")
    if df.empty:
        return pd.DataFrame(columns=["iteration", "rl_policy_energy_j", "rl_policy_phase_time_s"])

    for col in ["iteration", "phase_gpu_energy_delta_J", "phase_duration_s"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["phase_name"] = df.get("phase_name", "").astype(str).str.lower()
    df["phase_event"] = df.get("phase_event", "").astype(str).str.upper()
    df = df[
        (df["phase_name"] == "rl_policy")
        & (df["phase_event"] == "END")
    ][["iteration", "phase_gpu_energy_delta_J", "phase_duration_s"]].dropna().copy()
    df["iteration"] = df["iteration"].astype(int)
    return (
        df.groupby("iteration", as_index=False)
        .agg(
            rl_policy_energy_j=("phase_gpu_energy_delta_J", "sum"),
            rl_policy_phase_time_s=("phase_duration_s", "max"),
        )
    )


def _build_plot_df() -> pd.DataFrame:
    selected_runs = _select_runs()
    frames: list[pd.DataFrame] = []

    for row in selected_runs.itertuples(index=False):
        tokens = _load_tokens(str(row.run_id))
        reward_durations = _load_reward_durations(str(row.run_id))
        rl_energy = _load_rl_policy_energy(str(row.run_id))

        run_df = tokens.merge(reward_durations, on="iteration", how="inner").merge(rl_energy, on="iteration", how="inner")
        if run_df.empty:
            continue

        run_df["reward_duration_s"] = pd.to_numeric(run_df["reward_duration_s"], errors="coerce")
        run_df["rl_policy_subphase_time_s"] = pd.to_numeric(run_df["rl_policy_subphase_time_s"], errors="coerce")
        run_df["rl_policy_energy_j"] = pd.to_numeric(run_df["rl_policy_energy_j"], errors="coerce")
        run_df["reward_energy_share"] = (
            run_df["reward_duration_s"] / run_df["rl_policy_subphase_time_s"].replace(0.0, np.nan)
        )
        run_df["reward_subphase_energy_j"] = run_df["rl_policy_energy_j"] * run_df["reward_energy_share"]
        run_df["reward_subphase_power_w"] = run_df["reward_subphase_energy_j"] / run_df["reward_duration_s"].replace(0.0, np.nan)
        run_df["run_id"] = str(row.run_id)
        run_df["policy_norm"] = row.policy_norm
        run_df["dataset_group"] = row.dataset_group
        frames.append(run_df)

    if not frames:
        raise ValueError("No reward-subphase scatter rows could be built from the raw task runs.")

    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "rollout_mean_output_len",
            "reward_duration_s",
            "rl_policy_subphase_time_s",
            "rl_policy_energy_j",
            "reward_subphase_energy_j",
            "reward_subphase_power_w",
        ]
    ).copy()
    df = df[
        (df["rollout_mean_output_len"] > 0.0)
        & (df["reward_duration_s"] > 0.0)
        & (df["rl_policy_subphase_time_s"] > 0.0)
        & (df["rl_policy_energy_j"] > 0.0)
        & (df["reward_subphase_energy_j"] > 0.0)
        & (df["reward_subphase_power_w"] > 0.0)
    ].copy()
    if df.empty:
        raise ValueError("No valid reward-subphase scatter rows remained after filtering.")
    return df


def _add_trend_line(ax: plt.Axes, subdf: pd.DataFrame, dataset: str) -> None:
    if len(subdf) < 2:
        return
    x = subdf["rollout_mean_output_len"].to_numpy(dtype=float)
    y = subdf["reward_subphase_power_w"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ys = slope * xs + intercept
    ax.plot(xs, ys, color=DATASET_COLORS[dataset], linewidth=2.0, alpha=0.95, zorder=4)


def main() -> None:
    df = _build_plot_df()

    summary = (
        df.groupby(["policy_norm", "dataset_group"], dropna=False)
        .agg(
            n_points=("iteration", "size"),
            mean_output_len=("rollout_mean_output_len", "mean"),
            mean_reward_power_w=("reward_subphase_power_w", "mean"),
            median_reward_power_w=("reward_subphase_power_w", "median"),
            reward_share_mean=("reward_energy_share", "mean"),
        )
        .reset_index()
        .sort_values(["policy_norm", "dataset_group"])
    )
    print("reward-subphase scatter summary by policy and dataset:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, len(POLICY_ORDER), figsize=(14.0, 4.8), sharey=True)
    global_xmax = float(df["rollout_mean_output_len"].max())

    for ax, policy in zip(axes, POLICY_ORDER):
        policy_df = df[df["policy_norm"] == policy].copy()
        for dataset in TARGET_DATASETS:
            subdf = policy_df[policy_df["dataset_group"] == dataset].copy()
            if subdf.empty:
                continue
            ax.scatter(
                subdf["rollout_mean_output_len"],
                subdf["reward_subphase_power_w"],
                s=POINT_SIZE,
                color=DATASET_COLORS[dataset],
                alpha=POINT_ALPHA,
                edgecolors="black",
                linewidths=0.35,
                label=DATASET_DISPLAY[dataset] if policy == POLICY_ORDER[0] else None,
                zorder=3,
            )
            _add_trend_line(ax, subdf, dataset)

        ax.set_title(POLICY_DISPLAY[policy], fontweight="bold")
        ax.set_xlabel("Mean rollout response length (tokens)")
        ax.grid(axis="both", alpha=0.22, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=9)
        ax.set_xlim(left=0.0, right=max(global_xmax * 1.04, 1.0))

    axes[0].set_ylabel("Reward subphase GPU power (W)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=9,
    )
    fig.suptitle(
        "Reward Subphase Power vs. Rollout Response Length",
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
