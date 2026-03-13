"""Marginal learning gain per energy over validation intervals for baseline runs.

x-axis: cumulative step energy at interval midpoint (MJ)
y-axis: delta(validation score) / delta(energy) over each interval (score per MJ)
hue: policy
facet: model (Llama vs Qwen)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/baselines/marginal_learning_gain_per_joule_baselines.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

TARGET_SLURM_JOB_NAME_BY_FACET = {
    "Llama": "llama_new_baseline",
    "Qwen": "qwen_new_baseline",
}
TARGET_POLICIES = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {"ppo": "PPO", "remax": "ReMax", "grpo": "GRPO"}
POLICY_COLORS = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}
TARGET_MODEL_FACETS = ("Llama", "Qwen")
MODEL_DISPLAY = {
    "Llama": "Llama-3.1-8B-Inst",
    "Qwen": "Qwen2.5-3B-Inst",
}
BASELINE_GROUP_PREFIXES = ("stage1_llama8b_", "qwen_sys_3b_")
ENERGY_TO_MJ = 1e6


def _model_facet(model: str) -> str:
    text = str(model).lower()
    if "llama" in text:
        return "Llama"
    if "qwen" in text:
        return "Qwen"
    return "Other"


def _select_baseline_runs() -> pd.DataFrame:
    run_summary, _ = load_view("run_summary_view")
    runs, _ = load_view("runs")

    runs_df = run_summary.merge(
        runs[["run_id", "slurm_job_name"]],
        on="run_id",
        how="left",
        validate="one_to_one",
    ).copy()
    runs_df["policy_norm"] = runs_df["policy"].astype(str).str.lower()
    runs_df["model_facet"] = runs_df["model"].map(_model_facet)
    logical_group = runs_df["logical_run_group"].astype(str).str.lower()

    baseline_label_mask = logical_group.str.startswith(BASELINE_GROUP_PREFIXES, na=False)
    non_rollout_knob_mask = ~logical_group.str.contains(r"rollout|knob|cap", na=False)
    target_pair_mask = runs_df["policy_norm"].isin(TARGET_POLICIES) & runs_df["model_facet"].isin(TARGET_MODEL_FACETS)
    expected_slurm = runs_df["model_facet"].map(TARGET_SLURM_JOB_NAME_BY_FACET).astype(str).str.lower()
    slurm_job_mask = runs_df["slurm_job_name"].astype(str).str.lower() == expected_slurm
    checkpoint_mask = (
        ~runs_df["is_checkpoint_continuation"].fillna(False).astype(bool)
        if "is_checkpoint_continuation" in runs_df.columns
        else True
    )

    selected = runs_df[
        baseline_label_mask & non_rollout_knob_mask & target_pair_mask & slurm_job_mask & checkpoint_mask
    ].copy()
    if selected.empty:
        raise ValueError("No baseline runs selected.")
    return selected


def _resolve_run_dir(run_id: str) -> Path:
    root = Path("results/monitoring_val")
    matches = [p for p in root.glob(f"**/{run_id}") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find run directory for run_id={run_id} under {root}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: len(str(p)))
    return matches[0]


def _load_validation_series_from_run_json(run_id: str) -> pd.DataFrame:
    run_dir = _resolve_run_dir(run_id)
    run_json = run_dir / f"{run_id}.jsonl"
    if not run_json.exists():
        raise FileNotFoundError(f"Missing primary run jsonl: {run_json}")

    rows: list[dict[str, float]] = []
    with run_json.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            step = payload.get("step")
            data = payload.get("data", {}) or {}
            if not data.get("logging/validation_logged", False):
                continue
            score = data.get("critic/score/mean")
            if step is None or score is None:
                continue
            rows.append({"global_step_canonical": int(step), "validation_score": float(score)})

    out = pd.DataFrame(rows).sort_values("global_step_canonical").drop_duplicates("global_step_canonical", keep="last")
    if out.empty:
        raise ValueError(f"No validation score records found in {run_json}")
    return out


def _build_interval_df(selected_runs: pd.DataFrame) -> pd.DataFrame:
    step_fact, _ = load_view("step_fact_view")

    all_rows: list[pd.DataFrame] = []
    for _, run_row in selected_runs.iterrows():
        run_id = str(run_row["run_id"])
        model_facet = str(run_row["model_facet"])
        policy_norm = str(run_row["policy_norm"])

        run_steps = step_fact[step_fact["run_id"].astype(str) == run_id].copy()
        run_steps["global_step_canonical"] = pd.to_numeric(run_steps["global_step_canonical"], errors="coerce")
        run_steps["step_total_energy_j"] = pd.to_numeric(run_steps["step_total_energy_j"], errors="coerce")
        run_steps = run_steps.dropna(subset=["global_step_canonical", "step_total_energy_j"]).copy()
        run_steps["global_step_canonical"] = run_steps["global_step_canonical"].astype(int)
        run_steps = run_steps.sort_values("global_step_canonical")
        run_steps["cumulative_energy_j"] = run_steps["step_total_energy_j"].cumsum()

        val_df = _load_validation_series_from_run_json(run_id)
        merged = val_df.merge(
            run_steps[["global_step_canonical", "cumulative_energy_j"]],
            on="global_step_canonical",
            how="inner",
            validate="one_to_one",
        ).sort_values("global_step_canonical")

        if len(merged) < 2:
            continue

        merged["next_step"] = merged["global_step_canonical"].shift(-1)
        merged["next_score"] = merged["validation_score"].shift(-1)
        merged["next_energy_j"] = merged["cumulative_energy_j"].shift(-1)
        merged = merged.iloc[:-1].copy()

        merged["delta_score"] = merged["next_score"] - merged["validation_score"]
        merged["delta_energy_j"] = merged["next_energy_j"] - merged["cumulative_energy_j"]
        merged = merged[merged["delta_energy_j"] > 0].copy()

        merged["mid_energy_mj"] = (merged["cumulative_energy_j"] + merged["next_energy_j"]) / (2.0 * ENERGY_TO_MJ)
        merged["gain_per_mj"] = merged["delta_score"] / (merged["delta_energy_j"] / ENERGY_TO_MJ)
        merged["run_id"] = run_id
        merged["model_facet"] = model_facet
        merged["policy_norm"] = policy_norm

        all_rows.append(
            merged[
                [
                    "run_id",
                    "model_facet",
                    "policy_norm",
                    "global_step_canonical",
                    "next_step",
                    "mid_energy_mj",
                    "gain_per_mj",
                    "delta_score",
                    "delta_energy_j",
                ]
            ].copy()
        )

    if not all_rows:
        raise ValueError("No interval rows could be built. Check validation records and step energy coverage.")
    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    selected_runs = _select_baseline_runs()
    selected_run_ids = selected_runs["run_id"].astype(str).tolist()

    interval_df = _build_interval_df(selected_runs)
    print("interval rows used:")
    print(interval_df.sort_values(["model_facet", "policy_norm", "global_step_canonical"]).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    facet_axes = dict(zip(TARGET_MODEL_FACETS, axes))

    for model_facet in TARGET_MODEL_FACETS:
        ax = facet_axes[model_facet]
        sub = interval_df[interval_df["model_facet"] == model_facet].copy()

        for policy in TARGET_POLICIES:
            line = sub[sub["policy_norm"] == policy].sort_values("mid_energy_mj")
            if line.empty:
                continue
            ax.plot(
                line["mid_energy_mj"],
                line["gain_per_mj"],
                marker="o",
                markersize=5,
                linewidth=2.0,
                color=POLICY_COLORS[policy],
                label=POLICY_DISPLAY[policy],
                zorder=3,
            )

        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(MODEL_DISPLAY[model_facet], fontweight="bold")
        ax.set_xlabel("Cumulative Energy at Interval Midpoint (MJ)")
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.6)

    axes[0].set_ylabel("Marginal Validation Gain per Energy (score/MJ)")
    fig.suptitle("Marginal Learning Gain per Joule by Policy and Model", fontweight="bold", y=1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="marginal_learning_gain_per_joule_baselines",
        run_ids=selected_run_ids,
        data_sources={"views": ["run_summary_view", "runs", "step_fact_view"], "raw_files": ["<run_id>.jsonl"]},
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
