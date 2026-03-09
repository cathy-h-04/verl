"""Rollout J/token and straggler ratio: stage2 Llama baseline vs rollcap224 on matched mature steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.data.loader import load_view
from plots.plotting.filters import apply_analysis_ok, explain_filtering


OUTPATH = Path("plots/out/figures/tier0/rollcap224_vs_baseline_jtoken_straggler_llama_stage2.png")

PAIRS = [
    {
        "policy": "PPO",
        "baseline_run_id": "stage2_llama8b_ppo_20260301_120523",
        "cap_run_id": "stage2_llama8b_ppo_rollcap224_20260304_173955",
    },
    {
        "policy": "ReMax",
        "baseline_run_id": "stage2_llama8b_remax_20260301_121545",
        "cap_run_id": "stage2_llama8b_remax_rollcap224_20260304_175727",
    },
]

COLORS = {"baseline": "#4e79a7", "cap": "#e15759"}


def _choose_mature_common_steps(common_steps: list[int], width: int = 5) -> list[int]:
    """Pick mature matched steps; prefer [54..58] if available, else last width common steps."""
    preferred = [54, 55, 56, 57, 58]
    if all(s in common_steps for s in preferred):
        return preferred
    return sorted(common_steps)[-width:]


def main() -> None:
    step, _ = load_view("step_fact_view")
    required = [
        "run_id",
        "global_step_canonical",
        "validation_logged",
        "rollout_j_per_output_token",
        "straggler_ratio",
    ]
    missing = [c for c in required if c not in step.columns]
    if missing:
        raise ValueError(f"step_fact_view missing required columns: {missing}")

    run_ids = [x["baseline_run_id"] for x in PAIRS] + [x["cap_run_id"] for x in PAIRS]
    df = step[step["run_id"].astype(str).isin(run_ids)][required].copy()
    before = df.copy()
    df = apply_analysis_ok(df)
    print(f"step_filtering={explain_filtering(before, df)}")

    df["global_step_canonical"] = pd.to_numeric(df["global_step_canonical"], errors="coerce")
    df["rollout_j_per_output_token"] = pd.to_numeric(df["rollout_j_per_output_token"], errors="coerce")
    df["straggler_ratio"] = pd.to_numeric(df["straggler_ratio"], errors="coerce")
    df["validation_logged"] = df["validation_logged"].fillna(False).astype(bool)
    df = df.dropna(subset=["global_step_canonical"]).copy()
    df = df[~df["validation_logged"]].copy()

    rows: list[dict[str, object]] = []
    steps_used: dict[str, list[int]] = {}
    for pair in PAIRS:
        pol = pair["policy"]
        b = pair["baseline_run_id"]
        c = pair["cap_run_id"]
        b_steps = set(
            df.loc[df["run_id"] == b, "global_step_canonical"]
            .dropna()
            .astype(int)
            .tolist()
        )
        c_steps = set(
            df.loc[df["run_id"] == c, "global_step_canonical"]
            .dropna()
            .astype(int)
            .tolist()
        )
        common = sorted(b_steps.intersection(c_steps))
        if not common:
            raise ValueError(f"No common steps for policy={pol}")
        chosen = _choose_mature_common_steps(common, width=5)
        steps_used[pol] = chosen
        sub = df[df["global_step_canonical"].astype(int).isin(chosen)].copy()
        for kind, rid in [("Baseline", b), ("Rollcap224", c)]:
            rsub = sub[sub["run_id"] == rid]
            rows.append(
                {
                    "policy": pol,
                    "kind": kind,
                    "run_id": rid,
                    "n_steps": int(rsub["global_step_canonical"].nunique()),
                    "rollout_j_per_output_token_mean": float(rsub["rollout_j_per_output_token"].mean()),
                    "straggler_ratio_mean": float(rsub["straggler_ratio"].mean()),
                }
            )

    out = pd.DataFrame(rows)
    print("matched mature step windows:")
    for pol, s in steps_used.items():
        print(f"{pol}: {s}")
    print("summary:")
    print(out.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    metrics = [
        ("rollout_j_per_output_token_mean", "rollout_j_per_output_token", "Rollout J per Output Token"),
        ("straggler_ratio_mean", "straggler_ratio", "Straggler Ratio"),
    ]

    x = np.arange(len(PAIRS), dtype=float)
    w = 0.36

    for ax, (col, ylabel, title) in zip(axes, metrics):
        b_vals, c_vals = [], []
        for pair in PAIRS:
            pol = pair["policy"]
            b_vals.append(float(out[(out["policy"] == pol) & (out["kind"] == "Baseline")][col].iloc[0]))
            c_vals.append(float(out[(out["policy"] == pol) & (out["kind"] == "Rollcap224")][col].iloc[0]))
        b_vals = np.array(b_vals, dtype=float)
        c_vals = np.array(c_vals, dtype=float)

        bars_b = ax.bar(x - w / 2, b_vals, width=w, color=COLORS["baseline"], edgecolor="black", linewidth=0.7, label="Baseline (Cap=8192)")
        bars_c = ax.bar(x + w / 2, c_vals, width=w, color=COLORS["cap"], edgecolor="black", linewidth=0.7, label="Rollcap224")

        for i in range(len(PAIRS)):
            pct = (c_vals[i] - b_vals[i]) / b_vals[i] * 100.0 if b_vals[i] > 0 else np.nan
            txt = f"{pct:+.1f}%" if np.isfinite(pct) else "NA"
            y = max(b_vals[i], c_vals[i])
            ax.text(x[i], y * 1.03, txt, ha="center", va="bottom", fontsize=9)

        for bars in [bars_b, bars_c]:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h * 1.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([p["policy"] for p in PAIRS])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, loc="best")

    step_note = "; ".join([f"{k}: {v}" for k, v in steps_used.items()])
    fig.suptitle(
        "Stage2 Llama: Baseline (Cap=8192) vs Rollcap224 on Matched Mature Steps\n"
        f"Steps used: {step_note}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
