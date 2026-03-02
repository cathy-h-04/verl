#!/usr/bin/env python3
"""Build canonical Parquet datasets from experiment run artifacts.

This script scans a results root for run folders that contain `experiment_name.txt`,
validates required artifacts, excludes incomplete runs, and writes normalized tables.

Default output: /n/home08/chou/verl_research/DATASETS
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


REQUIRED_STATIC_FILES = [
    "experiment_name.txt",
    "slurm_job_ids.txt",
    "slurm_config.json",
    "run_config.json",
    "nvml_boundary.jsonl",
    "nvml_periodic.jsonl",
    "rapl_boundary.jsonl",
    "rapl_periodic.jsonl",
    "tokens_and_steps.jsonl",
]

CRITICAL_JSONL_FILES = [
    "nvml_boundary.jsonl",
    "nvml_periodic.jsonl",
    "rapl_boundary.jsonl",
    "rapl_periodic.jsonl",
    "tokens_and_steps.jsonl",
]

# Curated convenience view keys.
CURATED_METRIC_KEYS = [
    "training/global_step",
    "training/epoch",
    "logging/validation_logged",
    "val-core/openai/gsm8k/reward/mean@1",
    "timing_s/step",
    "perf/throughput",
    "perf/total_num_tokens",
    "comm_s/step",
    "comm_s/update_actor",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/entropy",
    "actor/lr",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/returns/mean",
    "response_length/mean",
    "prompt_length/mean",
    "rollout/straggler_ratio",
    "rollout/sync_efficiency",
    "perf/mfu/actor",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "logging/wall_time",
]

TABLE_FILES = {
    "runs": "runs.parquet",
    "run_lineage": "run_lineage.parquet",
    "step_metrics_long": "step_metrics_long.parquet",
    "step_metrics_wide_curated": "step_metrics_wide_curated.parquet",
    "phase_timings_long": "phase_timings_long.parquet",
    "tokens_and_steps": "tokens_and_steps.parquet",
    "hardware_boundary": "hardware_boundary.parquet",
    "hardware_periodic": "hardware_periodic.parquet",
    "phase_summary": "phase_summary.parquet",
    "ingestion_report": "ingestion_report.parquet",
}


@dataclass
class JsonlReadResult:
    records: List[Dict[str, Any]]
    parse_errors: int
    nonempty_lines: int


@dataclass
class RunQuality:
    run_dir: str
    run_id: Optional[str]
    status: str
    reason: str
    missing_files: List[str]
    zero_line_files: List[str]
    parse_error_files: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run datasets from experiment artifacts.")
    parser.add_argument(
        "--results-root",
        default="/n/home08/chou/verl_research/results",
        help="Root directory containing experiment result folders.",
    )
    parser.add_argument(
        "--output-root",
        default="/n/home08/chou/verl_research/DATASETS",
        help="Directory where output Parquet tables will be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallel parsing. Current implementation is single-process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove existing output directory first.",
    )
    return parser.parse_args()


def _sanitize_col(metric_key: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", metric_key).strip("_").lower()
    return f"metric_{name}" if name else "metric_unknown"


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {"_value": obj}


def _read_jsonl(path: Path) -> JsonlReadResult:
    records: List[Dict[str, Any]] = []
    parse_errors = 0
    nonempty = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nonempty += 1
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    records.append({"_value": obj})
            except Exception:
                parse_errors += 1

    return JsonlReadResult(records=records, parse_errors=parse_errors, nonempty_lines=nonempty)


def _parse_slurm_job_ids(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            out[key.strip()] = val.strip()
    return out


def _discover_run_dirs(results_root: Path) -> List[Path]:
    run_dirs = sorted({p.parent for p in results_root.rglob("experiment_name.txt")})
    return [p for p in run_dirs if p.is_dir()]


def _required_files_for_run(run_id: str) -> List[str]:
    return REQUIRED_STATIC_FILES + [
        f"{run_id}.jsonl",
        f"{run_id}_config.json",
        f"phase_timings_{run_id}.jsonl",
    ]


def _extract_lineage(resume_path: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not resume_path:
        return None, None
    m = re.search(r"/checkpoints/(.+?)/global_step_(\d+)$", resume_path)
    if not m:
        return None, None
    parent_run_name, step = m.group(1), m.group(2)
    return parent_run_name, _safe_int(step)


def _json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _ensure_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df


def _classify_metric(value: Any) -> Tuple[str, Optional[float], Optional[bool], Optional[str]]:
    if value is None:
        return "null", None, None, None
    if isinstance(value, bool):
        return "bool", None, value, None
    if isinstance(value, (int, float)):
        return "number", float(value), None, None
    if isinstance(value, str):
        return "string", None, None, value
    try:
        encoded = json.dumps(value, ensure_ascii=True, default=_json_default)
    except Exception:
        encoded = str(value)
    return "other", None, None, encoded


def build_datasets(results_root: Path, output_root: Path, overwrite: bool) -> None:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(results_root)

    runs_rows: List[Dict[str, Any]] = []
    lineage_rows: List[Dict[str, Any]] = []
    step_metrics_long_rows: List[Dict[str, Any]] = []
    phase_timings_rows: List[Dict[str, Any]] = []
    tokens_rows: List[Dict[str, Any]] = []
    hw_boundary_rows: List[Dict[str, Any]] = []
    hw_periodic_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    wide_rows_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    curated_metric_cols = {_k: _sanitize_col(_k) for _k in CURATED_METRIC_KEYS}

    for run_dir in run_dirs:
        run_id: Optional[str] = None
        quality = RunQuality(
            run_dir=str(run_dir),
            run_id=None,
            status="excluded",
            reason="unknown",
            missing_files=[],
            zero_line_files=[],
            parse_error_files={},
        )

        try:
            run_id = _read_text(run_dir / "experiment_name.txt")
            quality.run_id = run_id

            required_files = _required_files_for_run(run_id)
            missing = [f for f in required_files if not (run_dir / f).exists()]
            if missing:
                quality.reason = "missing_required_files"
                quality.missing_files = missing
                report_rows.append(
                    {
                        **quality.__dict__,
                        "included": False,
                    }
                )
                continue

            exp_metrics_file = f"{run_id}.jsonl"
            phase_timings_file = f"phase_timings_{run_id}.jsonl"
            full_config_file = f"{run_id}_config.json"

            critical_files = CRITICAL_JSONL_FILES + [exp_metrics_file, phase_timings_file]
            nonempty_counts: Dict[str, int] = {}
            for fname in critical_files:
                read_res = _read_jsonl(run_dir / fname)
                nonempty_counts[fname] = read_res.nonempty_lines
                if read_res.nonempty_lines == 0:
                    quality.zero_line_files.append(fname)

            if quality.zero_line_files:
                quality.reason = "incomplete_zero_line_critical_files"
                report_rows.append(
                    {
                        **quality.__dict__,
                        "included": False,
                    }
                )
                continue

            run_config = _read_json(run_dir / "run_config.json")
            slurm_config = _read_json(run_dir / "slurm_config.json")
            full_config = _read_json(run_dir / full_config_file)
            slurm_ids = _parse_slurm_job_ids(run_dir / "slurm_job_ids.txt")

            logical_run_group = (
                run_config.get("run", {}).get("name") if isinstance(run_config.get("run"), dict) else None
            )
            resume_path = run_config.get("run", {}).get("resume_path") if isinstance(run_config.get("run"), dict) else None
            parent_run_name, resume_from_global_step = _extract_lineage(resume_path)

            # Parse core JSONL files once for included runs.
            read_map = {
                "metrics": _read_jsonl(run_dir / exp_metrics_file),
                "phase_timings": _read_jsonl(run_dir / phase_timings_file),
                "tokens": _read_jsonl(run_dir / "tokens_and_steps.jsonl"),
                "nvml_boundary": _read_jsonl(run_dir / "nvml_boundary.jsonl"),
                "nvml_periodic": _read_jsonl(run_dir / "nvml_periodic.jsonl"),
                "rapl_boundary": _read_jsonl(run_dir / "rapl_boundary.jsonl"),
                "rapl_periodic": _read_jsonl(run_dir / "rapl_periodic.jsonl"),
            }

            for key, val in read_map.items():
                if val.parse_errors > 0:
                    quality.parse_error_files[key] = val.parse_errors

            if quality.parse_error_files:
                quality.reason = "json_parse_errors"
                report_rows.append(
                    {
                        **quality.__dict__,
                        "included": False,
                    }
                )
                continue

            # Included run.
            quality.status = "included"
            quality.reason = "ok"

            runs_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "results_root": str(results_root),
                    "logical_run_group": logical_run_group,
                    "run_name": run_config.get("run", {}).get("name") if isinstance(run_config.get("run"), dict) else None,
                    "model": run_config.get("run", {}).get("model") if isinstance(run_config.get("run"), dict) else None,
                    "dataset": run_config.get("run", {}).get("dataset") if isinstance(run_config.get("run"), dict) else None,
                    "policy": run_config.get("run", {}).get("policy") if isinstance(run_config.get("run"), dict) else None,
                    "use_validation": run_config.get("run", {}).get("use_validation")
                    if isinstance(run_config.get("run"), dict)
                    else None,
                    "val_freq": run_config.get("run", {}).get("val_freq") if isinstance(run_config.get("run"), dict) else None,
                    "poll_interval": run_config.get("run", {}).get("poll_interval")
                    if isinstance(run_config.get("run"), dict)
                    else None,
                    "total_steps": run_config.get("run", {}).get("total_steps")
                    if isinstance(run_config.get("run"), dict)
                    else None,
                    "total_epochs": run_config.get("run", {}).get("total_epochs")
                    if isinstance(run_config.get("run"), dict)
                    else None,
                    "resume_path": resume_path,
                    "is_resumed_run": bool(resume_path),
                    "resume_parent_run_name": parent_run_name,
                    "resume_from_global_step": resume_from_global_step,
                    "meta_source": run_config.get("meta", {}).get("source") if isinstance(run_config.get("meta"), dict) else None,
                    "meta_index": run_config.get("meta", {}).get("index") if isinstance(run_config.get("meta"), dict) else None,
                    "meta_name": run_config.get("meta", {}).get("name") if isinstance(run_config.get("meta"), dict) else None,
                    "slurm_job_id": slurm_ids.get("slurm_job_id"),
                    "slurm_array_task_id": slurm_ids.get("slurm_array_task_id"),
                    "slurm_job_name": slurm_ids.get("slurm_job_name"),
                    "slurm_timestamp": slurm_ids.get("timestamp"),
                    "slurm_partition": slurm_config.get("partition"),
                    "slurm_nodes": slurm_config.get("nodes"),
                    "slurm_gpus_per_node": slurm_config.get("gpus_per_node"),
                    "slurm_cpus_per_task": slurm_config.get("cpus_per_task"),
                    "slurm_mem": slurm_config.get("mem"),
                    "run_config_json": json.dumps(run_config, ensure_ascii=True, default=_json_default),
                    "slurm_config_json": json.dumps(slurm_config, ensure_ascii=True, default=_json_default),
                    "full_config_json": json.dumps(full_config, ensure_ascii=True, default=_json_default),
                }
            )

            lineage_rows.append(
                {
                    "run_id": run_id,
                    "is_resumed_run": bool(resume_path),
                    "resume_path": resume_path,
                    "resume_parent_run_name": parent_run_name,
                    "resume_from_global_step": resume_from_global_step,
                }
            )

            # Metrics table(s).
            for rec in read_map["metrics"].records:
                step = _safe_int(rec.get("step"))
                data = rec.get("data")
                if step is None or not isinstance(data, dict):
                    continue

                validation_logged = data.get("logging/validation_logged")
                if not isinstance(validation_logged, bool):
                    validation_logged = None

                wide_key = (run_id, step)
                if wide_key not in wide_rows_map:
                    wide_rows_map[wide_key] = {
                        "run_id": run_id,
                        "logical_run_group": logical_run_group,
                        "global_step": step,
                        "validation_logged": validation_logged,
                    }
                elif validation_logged is not None:
                    wide_rows_map[wide_key]["validation_logged"] = validation_logged

                for metric_key, metric_val in data.items():
                    metric_type, metric_value_float, metric_value_bool, metric_value_str = _classify_metric(metric_val)
                    step_metrics_long_rows.append(
                        {
                            "run_id": run_id,
                            "logical_run_group": logical_run_group,
                            "global_step": step,
                            "metric_key": metric_key,
                            "metric_type": metric_type,
                            "metric_value_float": metric_value_float,
                            "metric_value_bool": metric_value_bool,
                            "metric_value_str": metric_value_str,
                            "validation_logged": validation_logged,
                        }
                    )

                    if metric_key in curated_metric_cols:
                        col = curated_metric_cols[metric_key]
                        if isinstance(metric_val, bool):
                            wide_rows_map[wide_key][col] = metric_val
                        elif isinstance(metric_val, (int, float)):
                            wide_rows_map[wide_key][col] = float(metric_val)
                        elif metric_val is None:
                            wide_rows_map[wide_key][col] = None
                        else:
                            wide_rows_map[wide_key][col] = json.dumps(metric_val, ensure_ascii=True, default=_json_default)

            # Phase timings table.
            for rec in read_map["phase_timings"].records:
                row = dict(rec)
                row["run_id"] = run_id
                row["logical_run_group"] = logical_run_group
                row["global_step"] = _safe_int(rec.get("iteration"))
                phase_timings_rows.append(row)

            # Tokens and steps table.
            for rec in read_map["tokens"].records:
                row = dict(rec)
                row["run_id"] = run_id
                row["logical_run_group"] = logical_run_group
                row["global_step"] = _safe_int(rec.get("iteration"))
                tokens_rows.append(row)

            # Hardware tables.
            def add_hw_rows(records: List[Dict[str, Any]], source: str, record_kind: str) -> None:
                target = hw_boundary_rows if record_kind == "boundary" else hw_periodic_rows
                for rec in records:
                    global_step = _safe_int(rec.get("iteration"))
                    phase_name = rec.get("phase_name")

                    # Drop warmup idle rows by request.
                    if record_kind == "periodic" and global_step == 0 and phase_name == "idle":
                        continue

                    row = dict(rec)
                    row["run_id"] = run_id
                    row["logical_run_group"] = logical_run_group
                    row["global_step"] = global_step
                    row["source"] = source
                    row["record_kind"] = record_kind

                    if source == "nvml":
                        row["device_kind"] = "gpu"
                        row["device_id"] = row.get("gpu_uuid") or (f"gpu_index:{row.get('gpu_index')}" if row.get("gpu_index") is not None else None)
                    else:
                        row["device_kind"] = "rapl"
                        row["device_id"] = row.get("rapl_domain") or row.get("domain_path")

                    # Convenience converted column while keeping raw units.
                    row["phase_domain_energy_delta_j"] = None
                    if "phase_domain_energy_delta_uJ" in row:
                        uj = _safe_float(row.get("phase_domain_energy_delta_uJ"))
                        row["phase_domain_energy_delta_j"] = (uj / 1_000_000.0) if uj is not None else None

                    target.append(row)

            add_hw_rows(read_map["nvml_boundary"].records, "nvml", "boundary")
            add_hw_rows(read_map["rapl_boundary"].records, "rapl", "boundary")
            add_hw_rows(read_map["nvml_periodic"].records, "nvml", "periodic")
            add_hw_rows(read_map["rapl_periodic"].records, "rapl", "periodic")

            report_rows.append(
                {
                    **quality.__dict__,
                    "included": True,
                }
            )

        except Exception as exc:
            quality.reason = f"unexpected_error:{type(exc).__name__}"
            report_rows.append(
                {
                    **quality.__dict__,
                    "included": False,
                }
            )

    # Build DataFrames.
    runs_df = pd.DataFrame(runs_rows)
    lineage_df = pd.DataFrame(lineage_rows)
    metrics_long_df = pd.DataFrame(step_metrics_long_rows)
    phase_timings_df = pd.DataFrame(phase_timings_rows)
    tokens_df = pd.DataFrame(tokens_rows)
    hw_boundary_df = pd.DataFrame(hw_boundary_rows)
    hw_periodic_df = pd.DataFrame(hw_periodic_rows)
    report_df = pd.DataFrame(report_rows)

    wide_rows = list(wide_rows_map.values())
    wide_df = pd.DataFrame(wide_rows)

    # Derive phase summary from hardware tables.
    phase_group_cols = ["run_id", "global_step", "phase_name", "phase_id", "source"]

    boundary_summary = pd.DataFrame(columns=phase_group_cols)
    if not hw_boundary_df.empty:
        b = hw_boundary_df.copy()
        for col in ["phase_duration_s", "phase_gpu_energy_delta_J", "phase_domain_energy_delta_uJ", "phase_domain_energy_delta_j"]:
            if col in b.columns:
                b[col] = pd.to_numeric(b[col], errors="coerce")

        agg_spec: Dict[str, Any] = {"device_id": "nunique"}
        rename_map = {"device_id": "boundary_device_count"}
        if "phase_duration_s" in b.columns:
            agg_spec["phase_duration_s"] = "max"
            rename_map["phase_duration_s"] = "boundary_phase_duration_s_max"
        if "phase_gpu_energy_delta_J" in b.columns:
            agg_spec["phase_gpu_energy_delta_J"] = "sum"
            rename_map["phase_gpu_energy_delta_J"] = "boundary_gpu_energy_delta_j_sum"
        if "phase_domain_energy_delta_uJ" in b.columns:
            agg_spec["phase_domain_energy_delta_uJ"] = "sum"
            rename_map["phase_domain_energy_delta_uJ"] = "boundary_rapl_energy_delta_uj_sum"
        if "phase_domain_energy_delta_j" in b.columns:
            agg_spec["phase_domain_energy_delta_j"] = "sum"
            rename_map["phase_domain_energy_delta_j"] = "boundary_rapl_energy_delta_j_sum"

        boundary_summary = b.groupby(phase_group_cols, dropna=False).agg(agg_spec).reset_index().rename(columns=rename_map)
        boundary_counts = b.groupby(phase_group_cols, dropna=False).size().reset_index(name="boundary_row_count")
        boundary_summary = boundary_summary.merge(boundary_counts, on=phase_group_cols, how="outer")

    periodic_summary = pd.DataFrame(columns=phase_group_cols)
    if not hw_periodic_df.empty:
        p = hw_periodic_df.copy()
        numeric_candidates = [
            "gpu_power_mW",
            "gpu_util_pct",
            "sm_util_pct",
            "mem_util_pct",
            "temp_gpu_C",
            "cpu_energy_uJ",
        ]
        for col in numeric_candidates:
            if col in p.columns:
                p[col] = pd.to_numeric(p[col], errors="coerce")

        agg_spec_p: Dict[str, Any] = {"device_id": "nunique"}
        rename_map_p = {"device_id": "periodic_device_count"}
        for col in numeric_candidates:
            if col in p.columns:
                agg_spec_p[col] = "mean"
                rename_map_p[col] = f"periodic_{col}_mean"

        periodic_summary = p.groupby(phase_group_cols, dropna=False).agg(agg_spec_p).reset_index().rename(columns=rename_map_p)
        periodic_counts = p.groupby(phase_group_cols, dropna=False).size().reset_index(name="periodic_row_count")
        periodic_summary = periodic_summary.merge(periodic_counts, on=phase_group_cols, how="outer")

    if not boundary_summary.empty and not periodic_summary.empty:
        phase_summary_df = boundary_summary.merge(periodic_summary, on=phase_group_cols, how="outer")
    elif not boundary_summary.empty:
        phase_summary_df = boundary_summary
    elif not periodic_summary.empty:
        phase_summary_df = periodic_summary
    else:
        phase_summary_df = pd.DataFrame(columns=phase_group_cols)

    # Ensure key columns exist even for empty outputs.
    runs_df = _ensure_columns(runs_df, ["run_id", "run_dir", "logical_run_group", "is_resumed_run"])
    lineage_df = _ensure_columns(lineage_df, ["run_id", "is_resumed_run", "resume_parent_run_name", "resume_from_global_step"])
    metrics_long_df = _ensure_columns(
        metrics_long_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "metric_key",
            "metric_type",
            "metric_value_float",
            "metric_value_bool",
            "metric_value_str",
            "validation_logged",
        ],
    )
    wide_df = _ensure_columns(wide_df, ["run_id", "logical_run_group", "global_step", "validation_logged"])
    for key in CURATED_METRIC_KEYS:
        wide_df = _ensure_columns(wide_df, [curated_metric_cols[key]])

    phase_timings_df = _ensure_columns(
        phase_timings_df,
        ["run_id", "logical_run_group", "global_step", "phase_name", "phase_id", "subphase_name", "value", "metric_unit"],
    )
    tokens_df = _ensure_columns(
        tokens_df,
        ["run_id", "logical_run_group", "global_step", "phase_name", "phase_id", "metric_scope"],
    )
    hw_boundary_df = _ensure_columns(
        hw_boundary_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "phase_name",
            "phase_id",
            "source",
            "record_kind",
            "device_kind",
            "device_id",
            "phase_event",
            "ts_monotonic_ns",
            "ts_wall_ms",
        ],
    )
    hw_periodic_df = _ensure_columns(
        hw_periodic_df,
        [
            "run_id",
            "logical_run_group",
            "global_step",
            "phase_name",
            "phase_id",
            "source",
            "record_kind",
            "device_kind",
            "device_id",
            "ts_monotonic_ns",
            "ts_wall_ms",
        ],
    )
    phase_summary_df = _ensure_columns(phase_summary_df, phase_group_cols)
    report_df = _ensure_columns(
        report_df,
        ["run_dir", "run_id", "status", "reason", "included", "missing_files", "zero_line_files", "parse_error_files"],
    )

    # Stable sort for reproducibility.
    if not runs_df.empty:
        runs_df = runs_df.sort_values(["run_id"]).reset_index(drop=True)
    if not lineage_df.empty:
        lineage_df = lineage_df.sort_values(["run_id"]).reset_index(drop=True)
    if not metrics_long_df.empty:
        metrics_long_df = metrics_long_df.sort_values(["run_id", "global_step", "metric_key"]).reset_index(drop=True)
    if not wide_df.empty:
        wide_df = wide_df.sort_values(["run_id", "global_step"]).reset_index(drop=True)
    if not phase_timings_df.empty:
        phase_timings_df = phase_timings_df.sort_values(
            ["run_id", "global_step", "phase_name", "subphase_name"]
        ).reset_index(drop=True)
    if not tokens_df.empty:
        tokens_df = tokens_df.sort_values(["run_id", "global_step", "phase_name"]).reset_index(drop=True)
    if not hw_boundary_df.empty:
        hw_boundary_df = hw_boundary_df.sort_values(
            ["run_id", "global_step", "phase_name", "phase_event", "source", "device_id", "ts_monotonic_ns"]
        ).reset_index(drop=True)
    if not hw_periodic_df.empty:
        hw_periodic_df = hw_periodic_df.sort_values(
            ["run_id", "global_step", "phase_name", "source", "device_id", "ts_monotonic_ns"]
        ).reset_index(drop=True)
    if not phase_summary_df.empty:
        phase_summary_df = phase_summary_df.sort_values(phase_group_cols).reset_index(drop=True)
    if not report_df.empty:
        report_df = report_df.sort_values(["included", "run_id", "run_dir"], ascending=[False, True, True]).reset_index(drop=True)

    # Serialize list/dict report fields as JSON strings for Parquet compatibility.
    for col in ["missing_files", "zero_line_files", "parse_error_files"]:
        report_df[col] = report_df[col].apply(lambda x: json.dumps(x, ensure_ascii=True, default=_json_default))

    outputs = {
        "runs": runs_df,
        "run_lineage": lineage_df,
        "step_metrics_long": metrics_long_df,
        "step_metrics_wide_curated": wide_df,
        "phase_timings_long": phase_timings_df,
        "tokens_and_steps": tokens_df,
        "hardware_boundary": hw_boundary_df,
        "hardware_periodic": hw_periodic_df,
        "phase_summary": phase_summary_df,
        "ingestion_report": report_df,
    }

    for key, df in outputs.items():
        out_path = output_root / TABLE_FILES[key]
        df.to_parquet(out_path, index=False)

    included_runs = int(report_df["included"].astype(bool).sum()) if not report_df.empty else 0
    excluded_runs = int((~report_df["included"].astype(bool)).sum()) if not report_df.empty else 0

    print("Dataset build complete")
    print(f"  Results root: {results_root}")
    print(f"  Output root:  {output_root}")
    print(f"  Included runs: {included_runs}")
    print(f"  Excluded runs: {excluded_runs}")
    print("  Wrote tables:")
    for key in TABLE_FILES:
        print(f"    - {TABLE_FILES[key]}")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not results_root.exists() or not results_root.is_dir():
        raise FileNotFoundError(f"results root does not exist or is not a directory: {results_root}")

    # Currently single-process even if workers > 1; keep interface stable.
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    build_datasets(results_root=results_root, output_root=output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
