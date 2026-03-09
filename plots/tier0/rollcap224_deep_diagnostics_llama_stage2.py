"""Deep diagnostics for Stage2 Llama baseline vs rollcap224 (PPO, ReMax).

Computes four checks on matched mature steps:
1) Absolute step energy/time.
2) Phase decomposition (absolute + normalized).
3) Normalization by total tokens (prompt + output).
4) Occupancy/packing proxies (tokens per batch, throughput, straggler/sync).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plots.data.loader import load_view


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

OUT_MD = Path("plots/out/tables/tier0/rollcap224_deep_diagnostics_llama_stage2.md")


def _choose_mature_common_steps(common_steps: list[int], width: int = 5) -> list[int]:
    preferred = [54, 55, 56, 57, 58]
    if all(s in common_steps for s in preferred):
        return preferred
    return sorted(common_steps)[-width:]


def _pct(a: float, b: float) -> float:
    if not np.isfinite(a) or abs(a) < 1e-12:
        return np.nan
    return (b - a) / a * 100.0


def _safe_div(x: pd.Series, y: pd.Series) -> pd.Series:
    y = y.replace(0, np.nan)
    return x / y


def main() -> None:
    step, _ = load_view("step_fact_view")
    phase, _ = load_view("phase_fact_view")
    step = step.copy()
    phase = phase.copy()

    run_ids = [p["baseline_run_id"] for p in PAIRS] + [p["cap_run_id"] for p in PAIRS]
    step = step[step["run_id"].astype(str).isin(run_ids)].copy()
    phase = phase[phase["run_id"].astype(str).isin(run_ids)].copy()

    # Exclude validation rows for apples-to-apples training behavior.
    step = step[~step["validation_logged"].fillna(False)].copy()
    phase = phase[~phase["is_validation_step"].fillna(False)].copy()

    # Determine matched mature windows per policy.
    steps_used: dict[str, list[int]] = {}
    rows_step = []
    rows_phase = []

    for pair in PAIRS:
        pol = pair["policy"]
        b = pair["baseline_run_id"]
        c = pair["cap_run_id"]

        b_steps = set(step.loc[step["run_id"] == b, "global_step_canonical"].dropna().astype(int).tolist())
        c_steps = set(step.loc[step["run_id"] == c, "global_step_canonical"].dropna().astype(int).tolist())
        common = sorted(b_steps.intersection(c_steps))
        if not common:
            raise ValueError(f"No common steps for policy={pol}")
        chosen = _choose_mature_common_steps(common, width=5)
        steps_used[pol] = chosen

        step_sub = step[step["global_step_canonical"].astype(int).isin(chosen)].copy()
        phase_sub = phase[phase["global_step_canonical"].astype(int).isin(chosen)].copy()

        for label, run_id in [("Baseline", b), ("Rollcap224", c)]:
            s = step_sub[step_sub["run_id"] == run_id].copy()
            # 1 + 4: absolute step metrics and occupancy proxies.
            rows_step.append(
                {
                    "policy": pol,
                    "run_kind": label,
                    "run_id": run_id,
                    "n_steps": int(s["global_step_canonical"].nunique()),
                    "step_time_s_mean": float(s["step_time_s"].mean()),
                    "step_total_energy_j_mean": float(s["step_total_energy_j"].mean()),
                    "throughput_tokens_s_mean": float(s["throughput_tokens_s"].mean()),
                    "rollout_total_tokens_per_step_mean": float(s["step_rollout_total_tokens"].mean()),
                    "rollout_output_tokens_per_step_mean": float(s["step_rollout_output_tokens"].mean()),
                    "output_token_fraction_mean": float(
                        _safe_div(s["step_rollout_output_tokens"], s["step_rollout_total_tokens"]).mean()
                    ),
                    "straggler_ratio_mean": float(s["straggler_ratio"].mean()),
                    "sync_efficiency_mean": float(s["sync_efficiency"].mean()),
                    "idle_sync_proxy_mean": float((1.0 - s["sync_efficiency"]).mean()),
                    "rollout_total_tokens_cv": float(s["step_rollout_total_tokens"].std(ddof=0) / s["step_rollout_total_tokens"].mean()),
                }
            )

            # 2 + 3: phase decomposition with absolute and normalized metrics.
            p = phase_sub[(phase_sub["run_id"] == run_id) & (phase_sub["phase_name"].isin(["rollout", "rl_policy", "training"]))].copy()
            p["j_per_total_token_proxy"] = np.where(
                p["phase_name"] == "rollout",
                _safe_div(p["total_energy_j"], p["rollout_total_tokens"]),
                _safe_div(p["total_energy_j"], p["train_tokens_effective_estimated"]),
            )
            p["tokens_per_s_total_proxy"] = np.where(
                p["phase_name"] == "rollout",
                _safe_div(p["rollout_total_tokens"], p["phase_time_s"]),
                _safe_div(p["train_tokens_effective_estimated"], p["phase_time_s"]),
            )
            p["tokens_per_s_output_rollout"] = np.where(
                p["phase_name"] == "rollout",
                _safe_div(p["rollout_output_tokens_total"], p["phase_time_s"]),
                np.nan,
            )
            p["j_per_token_phase_native"] = np.where(
                p["phase_name"] == "rollout",
                p["j_per_output_token_rollout"],
                p["j_per_train_token_est"],
            )

            g = (
                p.groupby("phase_name", dropna=False)[
                    [
                        "phase_time_s",
                        "total_energy_j",
                        "avg_power_w",
                        "j_per_token_phase_native",
                        "j_per_total_token_proxy",
                        "tokens_per_s_total_proxy",
                        "tokens_per_s_output_rollout",
                    ]
                ]
                .mean(numeric_only=True)
                .reset_index()
            )
            for _, r in g.iterrows():
                rows_phase.append(
                    {
                        "policy": pol,
                        "run_kind": label,
                        "run_id": run_id,
                        "phase_name": r["phase_name"],
                        "phase_time_s_mean": float(r["phase_time_s"]),
                        "total_energy_j_mean": float(r["total_energy_j"]),
                        "avg_power_w_mean": float(r["avg_power_w"]),
                        "j_per_token_phase_native_mean": float(r["j_per_token_phase_native"]),
                        "j_per_total_token_proxy_mean": float(r["j_per_total_token_proxy"]),
                        "tokens_per_s_total_proxy_mean": float(r["tokens_per_s_total_proxy"]),
                        "tokens_per_s_output_rollout_mean": float(r["tokens_per_s_output_rollout"]),
                    }
                )

    step_out = pd.DataFrame(rows_step).sort_values(["policy", "run_kind"]).reset_index(drop=True)
    phase_out = pd.DataFrame(rows_phase).sort_values(["policy", "phase_name", "run_kind"]).reset_index(drop=True)

    # Add delta blocks (Rollcap224 vs Baseline).
    step_deltas = []
    for pol in step_out["policy"].unique():
        b = step_out[(step_out["policy"] == pol) & (step_out["run_kind"] == "Baseline")].iloc[0]
        c = step_out[(step_out["policy"] == pol) & (step_out["run_kind"] == "Rollcap224")].iloc[0]
        rec = {"policy": pol}
        for col in [
            "step_time_s_mean",
            "step_total_energy_j_mean",
            "throughput_tokens_s_mean",
            "rollout_total_tokens_per_step_mean",
            "rollout_output_tokens_per_step_mean",
            "output_token_fraction_mean",
            "straggler_ratio_mean",
            "sync_efficiency_mean",
            "idle_sync_proxy_mean",
            "rollout_total_tokens_cv",
        ]:
            rec[f"{col}_pct_change"] = _pct(float(b[col]), float(c[col]))
        step_deltas.append(rec)
    step_delta_out = pd.DataFrame(step_deltas)

    phase_deltas = []
    for pol in phase_out["policy"].unique():
        for ph in ["rollout", "rl_policy", "training"]:
            b = phase_out[(phase_out["policy"] == pol) & (phase_out["phase_name"] == ph) & (phase_out["run_kind"] == "Baseline")].iloc[0]
            c = phase_out[(phase_out["policy"] == pol) & (phase_out["phase_name"] == ph) & (phase_out["run_kind"] == "Rollcap224")].iloc[0]
            rec = {"policy": pol, "phase_name": ph}
            for col in [
                "phase_time_s_mean",
                "total_energy_j_mean",
                "avg_power_w_mean",
                "j_per_token_phase_native_mean",
                "j_per_total_token_proxy_mean",
                "tokens_per_s_total_proxy_mean",
                "tokens_per_s_output_rollout_mean",
            ]:
                rec[f"{col}_pct_change"] = _pct(float(b[col]), float(c[col]))
            phase_deltas.append(rec)
    phase_delta_out = pd.DataFrame(phase_deltas)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Stage2 Llama Rollcap224 Deep Diagnostics\n\n")
        f.write("## Matched mature steps\n")
        for pol, s in steps_used.items():
            f.write(f"- {pol}: {s}\n")
        f.write("\n## 1 + 4) Step-level absolute + occupancy proxies\n\n")
        f.write(step_out.to_markdown(index=False))
        f.write("\n\n### Step deltas (Rollcap224 vs Baseline), %\n\n")
        f.write(step_delta_out.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n## 2 + 3) Phase decomposition (absolute + normalized)\n\n")
        f.write(phase_out.to_markdown(index=False))
        f.write("\n\n### Phase deltas (Rollcap224 vs Baseline), %\n\n")
        f.write(phase_delta_out.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n")

    print(f"wrote {OUT_MD}")
    print("\nMatched mature steps:")
    for pol, s in steps_used.items():
        print(f"  {pol}: {s}")
    print("\nStep deltas (%):")
    print(step_delta_out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nPhase deltas (%):")
    print(phase_delta_out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
