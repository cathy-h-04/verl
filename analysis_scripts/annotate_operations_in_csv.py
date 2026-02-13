#!/usr/bin/env python3
"""Heuristically annotate phased GPU CSV rows with operation labels.

This script operates on a cleaned folder. For each run folder, it:
1) Filters out idle rows from the phased CSV.
2) Aligns CSV wall-clock timestamps to cleaned phase timing timestamps
   using the first non-idle (iteration, phase) anchor.
3) Builds approximate phase windows from cleaned phase timings.
4) Slices each phase window into operation sub-windows proportional to
   per-operation durations and assigns an operation label per CSV row.

Outputs an annotated CSV next to the original within each run folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Default repo root (do not depend on current working directory)
DEFAULT_ROOT = Path("/home/cathxhou/projects/verl_research")
# Default cleaned folder name (used when no name/dir args are provided)
DEFAULT_CLEANED_DIR = "monitoring_small_cleaned"

# Phase-specific operation ordering for proportional slicing
PHASE_OPERATION_ORDER: Dict[str, List[str]] = {
    "rollout": [
        "start_profile",
        "generate_sequences",
        "generation_timing/max",
        "generation_timing/min",
        "generation_timing/topk_ratio",
        "gen",
        "gen_max",
    ],
    "rl_policy": [
        "reward",
        "old_log_prob",
        "Role.RefPolicy",
        "values",
        "adv",
    ],
    "training": [
        "update_critic",
        "update_actor",
    ],
}


@dataclass(frozen=True)
class PhaseTiming:
    iteration: int
    phase: str
    timestamp_unix: float  # logged near phase completion
    operations: Dict[str, float]


def parse_csv_timestamp(ts: str) -> float:
    """Parse CSV timestamp (YYYY-MM-DD_HH:MM:SS) to unix seconds (UTC)."""
    dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_cleaned_phase_timings(path: Path) -> List[PhaseTiming]:
    timings: List[PhaseTiming] = []
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iteration = rec.get("iteration")
            phase = rec.get("phase")
            timestamp = rec.get("timestamp")
            if not isinstance(iteration, int) or not isinstance(phase, str):
                raise ValueError(f"Invalid phase timing at {path}:{line_no}")
            if not isinstance(timestamp, (int, float)):
                raise ValueError(f"Missing timestamp at {path}:{line_no}")
            ops = {
                k: float(v)
                for k, v in rec.items()
                if k not in {"iteration", "phase", "timestamp"}
                and isinstance(v, (int, float))
            }
            timings.append(
                PhaseTiming(
                    iteration=iteration,
                    phase=phase,
                    timestamp_unix=float(timestamp),
                    operations=ops,
                )
            )
    return timings


def group_phase_timings(timings: Iterable[PhaseTiming]) -> Dict[Tuple[int, str], PhaseTiming]:
    return {(t.iteration, t.phase): t for t in timings}


def find_first_non_idle_row(rows: Sequence[Mapping[str, str]]) -> Optional[Mapping[str, str]]:
    for row in rows:
        if row.get("phase_name") != "idle" and row.get("iteration") not in {"", "0", 0}:
            return row
    return None


def compute_time_offset(first_csv_row: Mapping[str, str], phase_map: Dict[Tuple[int, str], PhaseTiming]) -> Optional[float]:
    try:
        iteration = int(first_csv_row["iteration"])
        phase = first_csv_row["phase_name"]
        csv_ts = parse_csv_timestamp(first_csv_row["timestamp"])
    except Exception:
        return None

    key = (iteration, phase)
    timing = phase_map.get(key)
    if timing is None:
        return None
    return timing.timestamp_unix - csv_ts


def build_phase_windows(timings: List[PhaseTiming]) -> Dict[Tuple[int, str], Tuple[float, float]]:
    """Build approximate [start,end] windows using phase completion timestamps.

    For each iteration, windows are chained in timestamp order. We treat the
    phase timestamp as the phase end, and the previous phase timestamp as the
    start.
    """
    by_iter: Dict[int, List[PhaseTiming]] = {}
    for t in timings:
        by_iter.setdefault(t.iteration, []).append(t)

    windows: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for iteration, items in by_iter.items():
        items_sorted = sorted(items, key=lambda x: x.timestamp_unix)
        prev_end: Optional[float] = None
        for t in items_sorted:
            end = t.timestamp_unix
            start = prev_end if prev_end is not None else end
            windows[(iteration, t.phase)] = (start, end)
            prev_end = end
    return windows


def operation_slices(
    phase: str,
    operations: Mapping[str, float],
    start: float,
    end: float,
) -> List[Tuple[float, float, str]]:
    """Slice a phase window into operation sub-windows by duration proportion."""
    ordered_ops = PHASE_OPERATION_ORDER.get(phase, [])
    present_ops = [(op, operations.get(op, 0.0)) for op in ordered_ops]
    present_ops = [(op, dur) for op, dur in present_ops if dur and dur > 0]
    if not present_ops:
        return [(start, end, "unknown")]

    total_dur = sum(d for _, d in present_ops)
    if total_dur <= 0 or end <= start:
        return [(start, end, present_ops[-1][0])]

    window = end - start
    slices: List[Tuple[float, float, str]] = []
    cursor = start
    for i, (op, dur) in enumerate(present_ops):
        if i == len(present_ops) - 1:
            op_end = end
        else:
            op_end = cursor + window * (dur / total_dur)
        slices.append((cursor, op_end, op))
        cursor = op_end
    return slices


def assign_operation(ts: float, slices: Sequence[Tuple[float, float, str]]) -> str:
    for start, end, op in slices:
        if start <= ts <= end:
            return op
    return slices[-1][2] if slices else "unknown"


def annotate_run(run_dir: Path) -> Tuple[int, int, int]:
    phased_csvs = sorted(run_dir.glob("*_phased_*.csv"))
    cleaned_timings = sorted(run_dir.glob("cleaned_phase_timings_*.jsonl"))

    if not phased_csvs or not cleaned_timings:
        print(f"⚠ Skipping {run_dir.name}: missing phased CSV or cleaned timings")
        return (0, 0, 0)

    csv_path = phased_csvs[0]
    timings_path = cleaned_timings[0]

    timings = load_cleaned_phase_timings(timings_path)
    phase_map = group_phase_timings(timings)
    windows = build_phase_windows(timings)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    first_non_idle = find_first_non_idle_row(rows)
    if first_non_idle is None:
        print(f"⚠ Skipping {run_dir.name}: no non-idle rows found")
        return (0, 0, 0)

    offset = compute_time_offset(first_non_idle, phase_map)
    if offset is None:
        print(
            f"⚠ Skipping {run_dir.name}: could not align first non-idle row to timings"
        )
        return (0, 0, 0)

    kept_rows = 0
    tagged_rows = 0

    extra_cols = [
        "timestamp_aligned_unix",
        "operation",
    ]
    out_fieldnames = fieldnames + [c for c in extra_cols if c not in fieldnames]

    annotated_rows: List[Dict[str, str]] = []
    gpu_ids: set[str] = set()

    for row in rows:
        if row.get("phase_name") == "idle":
            continue
        try:
            iteration = int(row.get("iteration", "0"))
        except ValueError:
            continue
        if iteration <= 0:
            continue

        kept_rows += 1
        phase = row.get("phase_name", "")
        key = (iteration, phase)
        window = windows.get(key)
        timing = phase_map.get(key)

        aligned_ts = parse_csv_timestamp(row["timestamp"]) + offset
        row["timestamp_aligned_unix"] = f"{aligned_ts:.6f}"

        if window and timing:
            start, end = window
            slices = operation_slices(phase, timing.operations, start, end)
            op = assign_operation(aligned_ts, slices)
            row["operation"] = op
            tagged_rows += 1
        else:
            row["operation"] = "unknown"

        gpu_id = str(row.get("gpu_id", "")).strip()
        if gpu_id:
            gpu_ids.add(gpu_id)
        annotated_rows.append(row)

    out_path = run_dir / f"annotated_{csv_path.name}"
    with out_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in annotated_rows:
            writer.writerow(row)

    if len(gpu_ids) > 1:
        def _safe_gpu_id(raw: str) -> str:
            return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)

        for gpu_id in sorted(gpu_ids):
            suffix = _safe_gpu_id(gpu_id)
            gpu_out = run_dir / f"annotated_{csv_path.stem}_gpu{suffix}.csv"
            with gpu_out.open("w", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
                writer.writeheader()
                for row in annotated_rows:
                    if str(row.get("gpu_id", "")).strip() == gpu_id:
                        writer.writerow(row)

    return (kept_rows, tagged_rows, len(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate phased CSV rows with operation labels."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Repo root containing monitoring folders (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        help="Base folder name. Uses <name>_cleaned under --root.",
    )
    parser.add_argument(
        "--cleaned-dir",
        help="Explicit cleaned directory (absolute or relative to --root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.name is None and args.cleaned_dir is None:
        print("❌ Please provide --name or --cleaned-dir to avoid operating on the wrong dataset.")
        return
    root = Path(args.root).expanduser().resolve()
    if args.cleaned_dir:
        cleaned_root = Path(args.cleaned_dir).expanduser()
        cleaned_root = cleaned_root if cleaned_root.is_absolute() else root / cleaned_root
    elif args.name:
        cleaned_root = root / f"{args.name}_cleaned"
    else:
        cleaned_root = root / DEFAULT_CLEANED_DIR

    if not cleaned_root.exists():
        print(f"❌ CLEANED_FOLDER does not exist: {cleaned_root}")
        return

    run_dirs = sorted([p for p in cleaned_root.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"⚠ No run directories found under {cleaned_root}")
        return

    total_kept = 0
    total_tagged = 0
    total_rows = 0

    for run_dir in run_dirs:
        kept, tagged, rows = annotate_run(run_dir)
        total_kept += kept
        total_tagged += tagged
        total_rows += rows
        if kept:
            print(
                f"✓ {run_dir.name}: kept={kept}, tagged={tagged}, total_rows={rows}"
            )

    print("-" * 80)
    print(f"Rows total (including idle): {total_rows}")
    print(f"Rows kept (non-idle): {total_kept}")
    print(f"Rows tagged with operation: {total_tagged}/{total_kept}")
    print(f"Output: annotated_*.csv next to each phased CSV under {cleaned_root}")


if __name__ == "__main__":
    main()
