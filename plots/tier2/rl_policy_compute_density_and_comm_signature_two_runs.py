"""Communication-signature plot for two PPO Llama runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GSM8K_RUN_ID = "stage1_llama8b_ppo_20260301_075906"
RLHF_RUN_ID = "llama31_8b_smoke_test_rlhf_ff_20260304_192533"
RUN_ORDER = [GSM8K_RUN_ID, RLHF_RUN_ID]
RUN_LABELS = {
    GSM8K_RUN_ID: "GSM8K dataset",
    RLHF_RUN_ID: "RLHF dataset",
}
OUT_B_COMM = Path("plots/out/figures/tier2/comm_overhead_signature_two_runs.png")


def _load_run_step_metrics(run_id: str) -> pd.DataFrame:
    run_jsonl = Path("results/monitoring_val") / run_id / f"{run_id}.jsonl"
    rows: list[dict[str, float | bool | str]] = []
    with run_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data", {})
            step = data.get("training/global_step", rec.get("step"))
            if step is None:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "global_step_canonical": float(step),
                    "validation_logged": bool(data.get("logging/validation_logged", False)),
                    "comm_fraction_step": float(data.get("comm_fraction/step", np.nan)),
                    "comm_s_gen": float(data.get("comm_s/gen", np.nan)),
                    "comm_s_values": float(data.get("comm_s/values", np.nan)),
                    "comm_s_update_actor": float(data.get("comm_s/update_actor", np.nan)),
                    "comm_s_update_critic": float(data.get("comm_s/update_critic", np.nan)),
                    "perf_total_num_tokens": float(data.get("perf/total_num_tokens", np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for c in [
        "global_step_canonical",
        "comm_fraction_step",
        "comm_s_gen",
        "comm_s_values",
        "comm_s_update_actor",
        "comm_s_update_critic",
        "perf_total_num_tokens",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["run_id", "global_step_canonical"]).drop_duplicates(
        subset=["run_id", "global_step_canonical"], keep="last"
    )
    return out


def _plot_comm(step: pd.DataFrame) -> None:
    comm_components = ["comm_s_gen", "comm_s_values", "comm_s_update_actor", "comm_s_update_critic"]
    comm_labels = ["comm_s/gen", "comm_s/values", "comm_s/update_actor", "comm_s/update_critic"]
    comm_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for col, rid in enumerate(RUN_ORDER):
        sub_step = step[step["run_id"] == rid].copy()
        sub_step = sub_step.sort_values("global_step_canonical")
        x = sub_step["global_step_canonical"].to_numpy(dtype=float)
        bottom = np.zeros_like(x, dtype=float)
        ax_top = axes[col]
        for c, label, clr in zip(comm_components, comm_labels, comm_colors):
            y = sub_step[c].fillna(0.0).to_numpy(dtype=float)
            ax_top.bar(x, y, bottom=bottom, width=0.85, color=clr, alpha=0.85, label=label)
            bottom = bottom + y
        ax_top.set_title(RUN_LABELS[rid])
        ax_top.set_ylabel("comm_s/* (seconds)")
        ax_top.set_xlabel("global_step_canonical")
        ax_top.grid(alpha=0.2, axis="y")

    handles0, labels0 = axes[0].get_legend_handles_labels()
    fig.legend(handles0, labels0, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Communication Overhead Signature: Per-Step comm_s/*", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_B_COMM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_B_COMM, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    step = pd.concat([_load_run_step_metrics(rid) for rid in RUN_ORDER], ignore_index=True)
    step = step.dropna(subset=["global_step_canonical"]).copy()
    # Keep only non-validation training records for communication signatures.
    step = step[~step["validation_logged"].fillna(False)].copy()
    _plot_comm(step)
    print(f"wrote {OUT_B_COMM}")


if __name__ == "__main__":
    main()
