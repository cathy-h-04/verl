#!/usr/bin/env python3
"""
Compute phase-level energy and efficiency metrics from JSONL telemetry logs.

Inputs expected in --monitor-dir:
- nvml_boundary.jsonl
- nvml_periodic.jsonl
- rapl_boundary.jsonl
- rapl_periodic.jsonl (optional for this script)
- tokens_and_steps.jsonl
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _phase_key(rec: Dict[str, Any]) -> Tuple[Any, Any]:
    return (
        rec.get("phase_name"),
        rec.get("iteration"),
    )


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _pearson(x_vals: Iterable[Any], y_vals: Iterable[Any]) -> Optional[float]:
    pairs = []
    for x, y in zip(x_vals, y_vals):
        xf = _as_float(x)
        yf = _as_float(y)
        if xf is None or yf is None:
            continue
        pairs.append((xf, yf))
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom <= 0:
        return None
    return cov / denom


def _bool_fraction(values: Iterable[Any]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    trues = sum(1 for v in filtered if bool(v))
    return trues / float(len(filtered))


def compute_phase_metrics(monitor_dir: Path) -> List[Dict[str, Any]]:
    nvml_boundary = _read_jsonl(monitor_dir / "nvml_boundary.jsonl")
    nvml_periodic = _read_jsonl(monitor_dir / "nvml_periodic.jsonl")
    rapl_boundary = _read_jsonl(monitor_dir / "rapl_boundary.jsonl")
    tokens_logs = _read_jsonl(monitor_dir / "tokens_and_steps.jsonl")

    phase_windows: Dict[Tuple[Any, Any], Dict[str, Any]] = defaultdict(dict)
    phase_energy: Dict[Tuple[Any, Any], Dict[str, float]] = defaultdict(
        lambda: {
            "phase_gpu_energy_J_total": 0.0,
            "phase_cpu_energy_J": 0.0,
            "phase_dram_energy_J": 0.0,
            "phase_duration_s": 0.0,
        }
    )
    tokens_by_phase: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

    # Phase windows + GPU energy.
    for rec in nvml_boundary:
        key = _phase_key(rec)
        event = rec.get("phase_event")
        mono = rec.get("ts_monotonic_ns")
        if event == "START":
            prev = phase_windows[key].get("start_ns")
            phase_windows[key]["start_ns"] = mono if prev is None else min(prev, mono)
        elif event == "END":
            prev = phase_windows[key].get("end_ns")
            phase_windows[key]["end_ns"] = mono if prev is None else max(prev, mono)

            delta_j = _as_float(rec.get("phase_gpu_energy_delta_J"))
            if delta_j is not None:
                phase_energy[key]["phase_gpu_energy_J_total"] += delta_j
            duration = _as_float(rec.get("phase_duration_s"))
            if duration is not None and duration > phase_energy[key]["phase_duration_s"]:
                phase_energy[key]["phase_duration_s"] = duration

    # CPU + DRAM energy.
    for rec in rapl_boundary:
        key = _phase_key(rec)
        event = rec.get("phase_event")
        mono = rec.get("ts_monotonic_ns")
        if event == "START":
            prev = phase_windows[key].get("start_ns")
            phase_windows[key]["start_ns"] = mono if prev is None else min(prev, mono)
        elif event == "END":
            prev = phase_windows[key].get("end_ns")
            phase_windows[key]["end_ns"] = mono if prev is None else max(prev, mono)

            delta_uj = _as_float(rec.get("phase_domain_energy_delta_uJ"))
            if delta_uj is None:
                continue
            delta_j = delta_uj / 1_000_000.0
            rapl_domain = str(rec.get("rapl_domain") or "").lower()
            if rapl_domain.startswith("dram"):
                phase_energy[key]["phase_dram_energy_J"] += delta_j
            else:
                phase_energy[key]["phase_cpu_energy_J"] += delta_j

            duration = _as_float(rec.get("phase_duration_s"))
            if duration is not None and duration > phase_energy[key]["phase_duration_s"]:
                phase_energy[key]["phase_duration_s"] = duration

    for rec in tokens_logs:
        if rec.get("metric_scope") != "tokens_and_steps":
            continue
        tokens_by_phase[_phase_key(rec)] = rec

    # Diagnostics from periodic NVML samples.
    periodic_by_phase: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for rec in nvml_periodic:
        key = _phase_key(rec)
        win = phase_windows.get(key)
        if not win:
            continue
        start_ns = win.get("start_ns")
        end_ns = win.get("end_ns")
        t_ns = rec.get("ts_monotonic_ns")
        if start_ns is None or end_ns is None or t_ns is None:
            continue
        if start_ns <= t_ns <= end_ns:
            periodic_by_phase[key].append(rec)

    results: List[Dict[str, Any]] = []
    all_keys = set(phase_windows.keys()) | set(phase_energy.keys()) | set(tokens_by_phase.keys())
    for key in sorted(all_keys, key=lambda k: (k[1] if k[1] is not None else -1, str(k[0]))):
        phase_name, iteration = key
        e = phase_energy.get(key, {})
        duration_s = _as_float(e.get("phase_duration_s")) or 0.0
        gpu_j = _as_float(e.get("phase_gpu_energy_J_total")) or 0.0
        cpu_j = _as_float(e.get("phase_cpu_energy_J")) or 0.0
        dram_j = _as_float(e.get("phase_dram_energy_J")) or 0.0
        total_j = gpu_j + cpu_j + dram_j

        samples = periodic_by_phase.get(key, [])
        sw_cap_fraction = _bool_fraction([s.get("thr_sw_power_cap") for s in samples])
        corr_power_util = _pearson([s.get("gpu_power_mW") for s in samples], [s.get("gpu_util_pct") for s in samples])
        corr_power_clk = _pearson([s.get("gpu_power_mW") for s in samples], [s.get("sm_clock_MHz") for s in samples])
        corr_power_temp = _pearson([s.get("gpu_power_mW") for s in samples], [s.get("temp_gpu_C") for s in samples])

        out: Dict[str, Any] = {
            "phase_name": phase_name,
            "iteration": iteration,
            "phase_duration_s": duration_s if duration_s > 0 else None,
            "phase_gpu_energy_J_total": gpu_j,
            "phase_cpu_energy_J": cpu_j,
            "phase_dram_energy_J": dram_j,
            "phase_total_energy_J": total_j,
            "avg_phase_gpu_power_W": (gpu_j / duration_s) if duration_s > 0 else None,
            "avg_phase_total_power_W": (total_j / duration_s) if duration_s > 0 else None,
            "fraction_phase_time_thr_sw_power_cap": sw_cap_fraction,
            "corr_power_vs_util": corr_power_util,
            "corr_power_vs_clocks": corr_power_clk,
            "corr_power_vs_temp": corr_power_temp,
            "samples_in_phase_window": len(samples),
        }

        tok = tokens_by_phase.get(key, {})
        if phase_name == "rollout":
            output_tokens = _as_float(tok.get("rollout_output_tokens_total"))
            total_tokens = _as_float(tok.get("rollout_total_tokens"))
            out.update(
                {
                    "rollout_num_sequences": tok.get("rollout_num_sequences"),
                    "rollout_prompt_tokens_total": tok.get("rollout_prompt_tokens_total"),
                    "rollout_output_tokens_total": tok.get("rollout_output_tokens_total"),
                    "rollout_total_tokens": tok.get("rollout_total_tokens"),
                    "J_per_output_token": (total_j / output_tokens) if output_tokens and output_tokens > 0 else None,
                    "J_per_total_token": (total_j / total_tokens) if total_tokens and total_tokens > 0 else None,
                    "tokens_per_s_rollout": (output_tokens / duration_s) if duration_s > 0 and output_tokens else None,
                }
            )
        elif phase_name == "training":
            effective_tokens = _as_float(
                tok.get("train_tokens_effective_estimated", tok.get("train_tokens_effective"))
            )
            out.update(
                {
                    "train_batch_tokens": tok.get("train_batch_tokens"),
                    "train_microbatch_tokens_estimated": tok.get(
                        "train_microbatch_tokens_estimated", tok.get("train_microbatch_tokens")
                    ),
                    "train_epochs": tok.get("train_epochs"),
                    "train_minibatch_count_estimated": tok.get(
                        "train_minibatch_count_estimated", tok.get("train_minibatch_count")
                    ),
                    "train_minibatch_passes_estimated": tok.get(
                        "train_minibatch_passes_estimated", tok.get("train_minibatch_passes")
                    ),
                    "train_tokens_effective_estimated": tok.get(
                        "train_tokens_effective_estimated", tok.get("train_tokens_effective")
                    ),
                    "J_per_effective_train_token": (
                        (total_j / effective_tokens) if effective_tokens and effective_tokens > 0 else None
                    ),
                    "tokens_per_s_train_effective": (
                        (effective_tokens / duration_s) if duration_s > 0 and effective_tokens else None
                    ),
                }
            )

        results.append(out)

    return results


def main():
    parser = argparse.ArgumentParser(description="Postprocess NVML/RAPL phase telemetry into energy metrics.")
    parser.add_argument("--monitor-dir", required=True, help="Directory containing telemetry JSONL files.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <monitor-dir>/phase_energy_metrics.jsonl)",
    )
    args = parser.parse_args()

    monitor_dir = Path(args.monitor_dir)
    output_path = Path(args.output) if args.output else (monitor_dir / "phase_energy_metrics.jsonl")

    results = compute_phase_metrics(monitor_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(results)} phase metric rows to {output_path}")


if __name__ == "__main__":
    main()
