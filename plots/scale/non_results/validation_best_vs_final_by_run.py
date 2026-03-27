"""Best versus final validation score by run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plots.data.loader import load_view
from plots.data.manifest import build_run_manifest, save_manifest
from plots.plotting.style import savefig_paper


OUTPATH = Path("plots/out/scale/validation_best_vs_final_by_run.png")
MANIFEST_PATH = OUTPATH.with_suffix(".manifest.json")

POLICY_ORDER = ("ppo", "remax", "grpo")
POLICY_DISPLAY = {
    "ppo": "PPO",
    "remax": "ReMax",
    "grpo": "GRPO",
}
POLICY_COLORS = {
    "ppo": "#5B2A86",
    "remax": "#FF5C7A",
    "grpo": "#0097A7",
}
CONFIG_ORDER = ("2xA100", "2xH200", "4xA100", "4xH200")

FIGURE_TITLE_SIZE = 18
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 10


def _config_from_run_id(run_id: str) -> str:
    rid = str(run_id).lower()
    if "2gpu_a100" in rid:
        return "2xA100"
    if "2gpu_h200" in rid:
        return "2xH200"
    if "4gpu_a100" in rid:
        return "4xA100"
    if "4gpu_h200" in rid:
        return "4xH200"
    return "Unknown"


def _short_label(row: pd.Series) -> str:
    return f"{POLICY_DISPLAY[row['policy_norm']]} {row['config']}"


def main() -> None:
    runs_df, _ = load_view("runs")
    summary_df, _ = load_view("run_summary_view")

    selected = runs_df[runs_df["run_dir"].astype(str).str.contains("/llama_scaling/", regex=False)][["run_id"]].copy()
    selected = selected.merge(
        summary_df[["run_id", "policy", "best_validation_metric", "final_validation_metric"]],
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    selected["policy_norm"] = selected["policy"].astype(str).str.lower().str.replace("remx", "remax", regex=False)
    selected["config"] = selected["run_id"].map(_config_from_run_id)
    selected = selected[
        selected["policy_norm"].isin(POLICY_ORDER)
        & selected["config"].isin(CONFIG_ORDER)
        & selected["best_validation_metric"].notna()
        & selected["final_validation_metric"].notna()
    ].copy()
    if "is_checkpoint_continuation" in summary_df.columns:
        flags = summary_df[["run_id", "is_checkpoint_continuation"]].copy()
        selected = selected.merge(flags, on="run_id", how="left", validate="one_to_one")
        selected = selected[~selected["is_checkpoint_continuation"].fillna(False).astype(bool)].copy()
        selected = selected.drop(columns=["is_checkpoint_continuation"])

    selected["drop_best_to_final"] = selected["best_validation_metric"] - selected["final_validation_metric"]
    selected["label"] = selected.apply(_short_label, axis=1)
    selected["policy_order"] = selected["policy_norm"].map({p: i for i, p in enumerate(POLICY_ORDER)})
    selected["config_order"] = selected["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)})
    selected = selected.sort_values(["policy_order", "config_order"]).reset_index(drop=True)
    print("validation stability summary:")
    print(
        selected[["run_id", "label", "best_validation_metric", "final_validation_metric", "drop_best_to_final"]]
        .to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    y = range(len(selected))
    for idx, row in selected.iterrows():
        color = POLICY_COLORS[row["policy_norm"]]
        ax.plot(
            [row["final_validation_metric"], row["best_validation_metric"]],
            [idx, idx],
            color="0.7",
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            row["final_validation_metric"],
            idx,
            s=80,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax.scatter(
            row["best_validation_metric"],
            idx,
            s=95,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(selected["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Validation score", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("Best vs Final Validation by Run", fontsize=FIGURE_TITLE_SIZE, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_axisbelow(True)
    fig.tight_layout()
    saved = savefig_paper(fig, OUTPATH)
    plt.close(fig)
    print(f"wrote {saved}")

    manifest = build_run_manifest(
        plot_name="validation_best_vs_final_by_run",
        run_ids=selected["run_id"].astype(str).tolist(),
        data_sources={
            "root": "results/monitoring_val/llama_scaling",
            "views": ["runs", "run_summary_view"],
            "metrics": ["best_validation_metric", "final_validation_metric"],
        },
    )
    save_manifest(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
