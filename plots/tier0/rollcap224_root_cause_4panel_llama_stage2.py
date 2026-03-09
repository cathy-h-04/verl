"""4-panel root-cause figure for Stage2 Llama baseline vs rollcap224 (PPO/ReMax).

Panels:
1) Rollout cost normalization sensitivity.
2) Effective throughput.
3) Straggler ratio.
4) Phase time/energy deltas (cap vs baseline).
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


OUTPATH = Path("plots/out/figures/tier0/rollcap224_root_cause_4panel_llama_stage2.png")

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

COLORS = {
    "baseline": "#4e79a7",
    "cap": "#e15759",
    "rollout_total_j": "#9c755f",
    "rollout_j_out": "#59a14f",
    "rollout_j_total": "#f28e2b",
    "phase_time": "#f28e2b",
    "phase_energy": "#76b7b2",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _choose_steps(common_steps: list[int], width: int = 5) -> list[int]:
    if len(common_steps) <= width:
        return sorted(common_steps)
    return sorted(common_steps)[-width:]


def _load_run_config(run_id: str) -> dict:
    p = REPO_ROOT / "results" / "monitoring_val" / run_id / "run_config.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_iter_summary(run_id: str) -> pd.DataFrame:
    p = REPO_ROOT / "results" / "monitoring_val" / run_id / f"{run_id}.jsonl"
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            step = rec.get("step")
            data = rec.get("data", {})
            if step is None:
                continue
            if data.get("logging/record_scope") != "iteration_summary":
                continue
            if bool(data.get("logging/validation_logged", False)):
                continue
            rows.append(
                {
                    "global_step_canonical": int(step),
                    "response_length_mean": float(data.get("response_length/mean", np.nan)),
                    "response_length_p95": float(data.get("rollout/response_length_p95", np.nan)),
                    "straggler_ratio": float(data.get("rollout/straggler_ratio", np.nan)),
                    "throughput_tokens_s": float(data.get("perf/throughput", np.nan)),
                    "step_time_s": float(data.get("timing_s/step", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    step, _ = load_view("step_fact_view")
    phase, _ = load_view("phase_fact_view")

    run_ids = [x["baseline_run_id"] for x in PAIRS] + [x["cap_run_id"] for x in PAIRS]
    step = step[step["run_id"].astype(str).isin(run_ids)].copy()
    phase = phase[phase["run_id"].astype(str).isin(run_ids)].copy()
    step = step[~step["validation_logged"].fillna(False)].copy()
    phase = phase[~phase["is_validation_step"].fillna(False)].copy()

    iter_tables = {rid: _load_iter_summary(rid) for rid in run_ids}
    cfg = {rid: _load_run_config(rid) for rid in run_ids}
    summary_rows = []
    rollout_norm_rows = []
    phase_delta_rows = []
    steps_used = {}

    for pair in PAIRS:
        pol = pair["policy"]
        b = pair["baseline_run_id"]
        c = pair["cap_run_id"]
        b_steps = set(step.loc[step["run_id"] == b, "global_step_canonical"].dropna().astype(int).tolist())
        c_steps = set(step.loc[step["run_id"] == c, "global_step_canonical"].dropna().astype(int).tolist())
        common = sorted(b_steps.intersection(c_steps))
        if not common:
            raise ValueError(f"No common non-validation steps for policy={pol}")
        chosen = _choose_steps(common, width=5)
        steps_used[pol] = chosen

        for kind, rid in [("Baseline", b), ("Rollcap224", c)]:
            s = step[(step["run_id"] == rid) & (step["global_step_canonical"].astype(int).isin(chosen))].copy()
            i = iter_tables[rid]
            i = i[i["global_step_canonical"].astype(int).isin(chosen)].copy()
            train_cfg = cfg[rid].get("train", {})
            summary_rows.append(
                {
                    "policy": pol,
                    "kind": kind,
                    "run_id": rid,
                    "n_steps": int(s["global_step_canonical"].nunique()),
                    "throughput_tokens_s_mean": float(s["throughput_tokens_s"].mean()),
                    "step_time_s_mean": float(s["step_time_s"].mean()),
                    "step_total_energy_j_mean": float(s["step_total_energy_j"].mean()),
                    "straggler_ratio_mean": float(s["straggler_ratio"].mean()),
                    "response_length_mean": float(i["response_length_mean"].mean()),
                    "response_length_p95": float(i["response_length_p95"].mean()),
                    "cfg_rollout_max_batched_tokens": float(train_cfg.get("rollout_max_batched_tokens", np.nan)),
                }
            )

        psub = phase[
            phase["global_step_canonical"].astype(int).isin(chosen)
            & (phase["run_id"].isin([b, c]))
            & (phase["phase_name"].isin(["rollout", "rl_policy", "training"]))
        ].copy()
        for phase_name in ["rollout", "rl_policy", "training"]:
            pb = psub[(psub["run_id"] == b) & (psub["phase_name"] == phase_name)]
            pc = psub[(psub["run_id"] == c) & (psub["phase_name"] == phase_name)]
            if pb.empty or pc.empty:
                continue
            t_b = float(pb["phase_time_s"].mean())
            t_c = float(pc["phase_time_s"].mean())
            e_b = float(pb["total_energy_j"].mean())
            e_c = float(pc["total_energy_j"].mean())
            phase_delta_rows.append(
                {
                    "policy": pol,
                    "phase_name": phase_name,
                    "time_pct_delta": (t_c - t_b) / t_b * 100.0 if t_b > 0 else np.nan,
                    "energy_pct_delta": (e_c - e_b) / e_b * 100.0 if e_b > 0 else np.nan,
                }
            )

        for kind, rid in [("Baseline", b), ("Rollcap224", c)]:
            pr = psub[(psub["run_id"] == rid) & (psub["phase_name"] == "rollout")].copy()
            if pr.empty:
                continue
            rollout_norm_rows.append(
                {
                    "policy": pol,
                    "kind": kind,
                    "rollout_total_energy_j_mean": float(pr["total_energy_j"].mean()),
                    "rollout_time_ms_mean": float(pr["phase_time_s"].mean() * 1000.0),
                    "rollout_avg_power_w_mean": float(pr["avg_power_w"].mean()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    rollout_norm = pd.DataFrame(rollout_norm_rows)
    phase_delta = pd.DataFrame(phase_delta_rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # Panel A: rollout total energy/time/power (bar chart).
    metrics = [
        ("rollout_total_energy_j_mean", "Total energy (J)", COLORS["rollout_total_j"]),
        ("rollout_time_ms_mean", "Time (ms)", COLORS["rollout_j_out"]),
        ("rollout_avg_power_w_mean", "Average power (W)", COLORS["rollout_j_total"]),
    ]
    labels_a = []
    base_vals = []
    cap_vals = []
    bar_colors = []
    for pol in ["PPO", "ReMax"]:
        for metric, metric_label, color in metrics:
            labels_a.append(f"{pol}\n{metric_label}")
            base_vals.append(
                float(rollout_norm[(rollout_norm["policy"] == pol) & (rollout_norm["kind"] == "Baseline")][metric].iloc[0])
            )
            cap_vals.append(
                float(rollout_norm[(rollout_norm["policy"] == pol) & (rollout_norm["kind"] == "Rollcap224")][metric].iloc[0])
            )
            bar_colors.append(color)
    xa = np.arange(len(labels_a), dtype=float)
    wa = 0.38
    ax_a.bar(xa - wa / 2, base_vals, width=wa, color=bar_colors, alpha=0.45, edgecolor="black", linewidth=0.6, label="Baseline")
    ax_a.bar(xa + wa / 2, cap_vals, width=wa, color=bar_colors, alpha=0.90, edgecolor="black", linewidth=0.6, label="Rollcap224")
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels(labels_a, fontsize=8)
    ax_a.set_ylabel("Metric value (native units)")
    ax_a.set_title("A) Rollout Total Energy, Time, and Power")
    ax_a.grid(axis="y", alpha=0.2)
    ax_a.legend(frameon=False, fontsize=8, loc="best")

    # Panel B: throughput.
    xb = np.arange(2, dtype=float)
    base_t = [float(summary[(summary["policy"] == p) & (summary["kind"] == "Baseline")]["throughput_tokens_s_mean"].iloc[0]) for p in ["PPO", "ReMax"]]
    cap_t = [float(summary[(summary["policy"] == p) & (summary["kind"] == "Rollcap224")]["throughput_tokens_s_mean"].iloc[0]) for p in ["PPO", "ReMax"]]
    wb = 0.35
    ax_b.bar(xb - wb / 2, base_t, width=wb, color=COLORS["baseline"], edgecolor="black", linewidth=0.7, label="Baseline")
    ax_b.bar(xb + wb / 2, cap_t, width=wb, color=COLORS["cap"], edgecolor="black", linewidth=0.7, label="Rollcap224")
    for i in range(2):
        pct = (cap_t[i] - base_t[i]) / base_t[i] * 100.0 if base_t[i] > 0 else np.nan
        y = max(base_t[i], cap_t[i])
        ax_b.text(i, y * 1.03, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["PPO", "ReMax"])
    ax_b.set_ylabel("Tokens / second")
    ax_b.set_title("B) Effective Throughput (Matched Mature Steps)")
    ax_b.grid(axis="y", alpha=0.2)
    ax_b.legend(frameon=False)

    # Panel C: straggler only.
    base_s = [float(summary[(summary["policy"] == p) & (summary["kind"] == "Baseline")]["straggler_ratio_mean"].iloc[0]) for p in ["PPO", "ReMax"]]
    cap_s = [float(summary[(summary["policy"] == p) & (summary["kind"] == "Rollcap224")]["straggler_ratio_mean"].iloc[0]) for p in ["PPO", "ReMax"]]

    wc = 0.35
    ax_c.bar(xb - wc / 2, base_s, width=wc, color=COLORS["baseline"], edgecolor="black", linewidth=0.7, label="Baseline straggler")
    ax_c.bar(xb + wc / 2, cap_s, width=wc, color=COLORS["cap"], edgecolor="black", linewidth=0.7, label="Rollcap224 straggler")
    for i in range(2):
        pct = (cap_s[i] - base_s[i]) / base_s[i] * 100.0 if base_s[i] > 0 else np.nan
        y = max(base_s[i], cap_s[i])
        ax_c.text(i, y * 1.03, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax_c.set_xticks(xb)
    ax_c.set_xticklabels(["PPO", "ReMax"])
    ax_c.set_ylabel("Straggler ratio")
    ax_c.set_title("C) Straggler Ratio (Higher Is Worse)")
    ax_c.grid(axis="y", alpha=0.2)
    ax_c.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel D: phase deltas (%).
    d_order = []
    for pol in ["PPO", "ReMax"]:
        for ph in ["rollout", "rl_policy", "training"]:
            d_order.append((pol, ph))
    xd = np.arange(len(d_order), dtype=float)
    time_delta = []
    energy_delta = []
    labels_d = []
    for pol, ph in d_order:
        r = phase_delta[(phase_delta["policy"] == pol) & (phase_delta["phase_name"] == ph)]
        if r.empty:
            time_delta.append(np.nan)
            energy_delta.append(np.nan)
        else:
            time_delta.append(float(r["time_pct_delta"].iloc[0]))
            energy_delta.append(float(r["energy_pct_delta"].iloc[0]))
        labels_d.append(f"{pol}\n{ph}")
    wd = 0.38
    ax_d.bar(xd - wd / 2, time_delta, width=wd, color=COLORS["phase_time"], edgecolor="black", linewidth=0.7, label="Time delta %")
    ax_d.bar(xd + wd / 2, energy_delta, width=wd, color=COLORS["phase_energy"], edgecolor="black", linewidth=0.7, label="Energy delta %")
    ax_d.axhline(0.0, color="black", linewidth=0.8)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(labels_d, fontsize=8)
    ax_d.set_ylabel("% change (Rollcap224 vs Baseline)")
    ax_d.set_title("D) Phase Cost Shift (+ means rollcap is slower/more energy)")
    ax_d.grid(axis="y", alpha=0.2)
    ax_d.legend(frameon=False)
    ax_d.text(
        0.01,
        0.98,
        "+% = increase under Rollcap224 (worse)\n-% = decrease under Rollcap224 (better)",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="gray"),
    )

    steps_note = "; ".join(f"{k}: {v}" for k, v in steps_used.items())
    fig.suptitle(
        "Stage2 Llama Rollcap224 Root-Cause Diagnostic (Matched Non-Validation Mature Steps)\n"
        f"Steps used: {steps_note}",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("matched steps:", steps_used)
    print(f"wrote {OUTPATH}")


if __name__ == "__main__":
    main()
