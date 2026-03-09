"""Communication-only step decomposition: Llama baseline vs small microbatch.

Matches late-step window across all requested runs.
Prefers steps [54,55,56,57,58], falls back to the latest common steps.
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
from plots.plotting.filters import apply_analysis_ok


OUTPATH = Path("plots/out/figures/tier1/step_time_decomposition_smallmb_vs_baseline_llama_stage2.png")

RUNS = [
    # Baselines
    {"run_id": "stage2_llama8b_ppo_20260301_120523", "policy": "PPO", "variant": "Baseline"},
    {"run_id": "stage2_llama8b_grpo_20260301_122602", "policy": "GRPO", "variant": "Baseline"},
    {"run_id": "stage2_llama8b_remax_20260301_121545", "policy": "ReMax", "variant": "Baseline"},
    # Small microbatch
    {"run_id": "stage2_smallmb_llama8b_ppo_20260304_162933", "policy": "PPO", "variant": "SmallMB"},
    {"run_id": "stage2_smallmb_llama8b_grpo_20260304_171751", "policy": "GRPO", "variant": "SmallMB"},
    {"run_id": "stage2_smallmb_llama8b_remax_20260304_165430", "policy": "ReMax", "variant": "SmallMB"},
]

PANELS = ["PPO", "GRPO", "ReMax"]
VARIANT_ORDER = ["Baseline", "SmallMB"]

COMM_COLS = [
    "comm_s/gen",
    "comm_s/values",
    "comm_s/old_log_prob",
    "comm_s/update_actor",
    "comm_s/update_critic",
]
COMM_COLORS = {
    "comm_s/gen": "#4e79a7",
    "comm_s/values": "#f28e2b",
    "comm_s/old_log_prob": "#59a14f",
    "comm_s/update_actor": "#e15759",
    "comm_s/update_critic": "#76b7b2",
}


def _choose_window(common_steps: list[int]) -> list[int]:
    preferred = [54, 55, 56, 57, 58]
    if all(s in common_steps for s in preferred):
        return preferred
    # Fallback: latest 4 common steps (these runs have 51..57, so this picks 54..57).
    return sorted(common_steps)[-4:]


def _load_run_jsonl_steps(run_id: str) -> pd.DataFrame:
    path = Path("results/monitoring_val") / run_id / f"{run_id}.jsonl"
    rows: list[dict[str, float | bool | str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            data = rec.get("data", {})
            if data.get("logging/record_scope") != "iteration_summary":
                continue
            step = data.get("training/global_step", rec.get("step"))
            if step is None:
                continue
            row: dict[str, float | bool | str] = {
                "run_id": run_id,
                "global_step_canonical": float(step),
                "validation_logged": bool(data.get("logging/validation_logged", False)),
                "timing_s/step": data.get("timing_s/step", np.nan),
                "timing_s/gen": data.get("timing_s/gen", np.nan),
                "comm_s/step": data.get("comm_s/step", np.nan),
            }
            for c in COMM_COLS:
                row[c] = data.get(c, np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    num_cols = ["global_step_canonical", "timing_s/step", "timing_s/gen", "comm_s/step"] + COMM_COLS
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    run_meta = pd.DataFrame(RUNS)
    run_ids = run_meta["run_id"].tolist()

    sf, _ = load_view("step_fact_view")
    req = ["run_id", "global_step_canonical", "validation_logged"]
    filt = sf[sf["run_id"].astype(str).isin(run_ids)][req].copy()
    filt = apply_analysis_ok(filt)
    filt["global_step_canonical"] = pd.to_numeric(filt["global_step_canonical"], errors="coerce")
    filt["validation_logged"] = filt["validation_logged"].fillna(False).astype(bool)
    filt = filt.dropna(subset=["global_step_canonical"]).copy()
    filt = filt[~filt["validation_logged"]].copy()
    filt = filt[["run_id", "global_step_canonical"]].drop_duplicates()

    raw = pd.concat([_load_run_jsonl_steps(rid) for rid in run_ids], ignore_index=True)
    raw = raw.dropna(subset=["global_step_canonical", "timing_s/step"]).copy()
    raw = raw.merge(filt, on=["run_id", "global_step_canonical"], how="inner")
    raw = raw.merge(run_meta, on="run_id", how="inner")

    # Ensure a single matched late-step window across all six runs.
    steps_by_run = {
        rid: set(raw.loc[raw["run_id"] == rid, "global_step_canonical"].dropna().astype(int).tolist()) for rid in run_ids
    }
    common_steps = sorted(set.intersection(*steps_by_run.values()))
    if not common_steps:
        raise ValueError("No common non-validation steps across all six runs.")
    chosen_steps = _choose_window(common_steps)
    raw = raw[raw["global_step_canonical"].astype(int).isin(chosen_steps)].copy()

    for c in COMM_COLS:
        raw[c] = raw[c].fillna(0.0)
    raw["comm_sum_components"] = raw[COMM_COLS].sum(axis=1)
    raw["comm_step_ref"] = pd.to_numeric(raw["comm_s/step"], errors="coerce")
    raw["comm_total_used"] = raw["comm_step_ref"].where(raw["comm_step_ref"].notna(), raw["comm_sum_components"])

    agg_cols = COMM_COLS + ["timing_s/step", "timing_s/gen", "comm_total_used"]
    agg = raw.groupby(["policy", "variant"], as_index=False)[agg_cols].mean()

    print("common steps across all runs:", common_steps)
    print("chosen matched late-step window:", chosen_steps)
    print("mean step decomposition (s/step):")
    print(
        agg.sort_values(["policy", "variant"])[
            ["policy", "variant"] + COMM_COLS + ["comm_total_used", "timing_s/step"]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    y_max = float((agg["comm_total_used"].max() if not agg.empty else 1.0) * 1.35)

    legend_handles = None
    legend_labels = None
    seen_legend_labels: set[str] = set()

    for ax, policy in zip(axes, PANELS):
        sub = agg[agg["policy"] == policy].copy().set_index("variant")
        x_map = {"Baseline": 0.0, "SmallMB": 1.0}
        for variant in VARIANT_ORDER:
            x = x_map[variant]
            if variant not in sub.index:
                ax.text(x, 0.02 * y_max, "NA", ha="center", va="bottom", fontsize=10)
                continue
            row = sub.loc[variant]
            bottom = 0.0
            for c in COMM_COLS:
                val = float(row[c])
                ax.bar(
                    x,
                    val,
                    bottom=bottom,
                    width=0.62,
                    color=COMM_COLORS[c],
                    edgecolor="black",
                    linewidth=0.5,
                    label=c if c not in seen_legend_labels else "_nolegend_",
                )
                seen_legend_labels.add(c)
                bottom += val
            ax.text(x, bottom + 0.02 * y_max, f"{bottom:.3f}s", ha="center", va="bottom", fontsize=9)

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Baseline", "SmallMB"])
        ax.set_title(policy)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, y_max)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Communication seconds per step (mean)")
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Step Communication Decomposition: Llama Baseline vs SmallMB\n"
        f"Matched common late steps: {chosen_steps}",
        y=1.08,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()

