#!/usr/bin/env python3
"""
Clean up phase timing data by removing irrelevant fields from each phase.

This script reads phase_timings_*.jsonl files and filters each phase log entry
to only contain operations that belong to that specific phase, removing the
accumulated timing data from previous phases.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set

# Default repo root (do not depend on current working directory)
DEFAULT_ROOT = Path("/home/cathxhou/projects/verl_research")
# Default monitoring directory names (used when no name/dir args are provided)
DEFAULT_DATA_DIR = "monitoring_small_data"
DEFAULT_VAL_DIR = "monitoring_small_val"
DEFAULT_OUT_DIR = "monitoring_small_cleaned"

# Define which operations belong to each phase
PHASE_OPERATIONS: Dict[str, Set[str]] = {
    "rollout": {
        "start_profile",
        "generate_sequences",
        "generation_timing/max",
        "generation_timing/min",
        "generation_timing/median",
        "generation_timing/p95",
        "generation_timing/imbalance",
        "generation_timing/topk_ratio",
        "comm_s/gen",
        "gen",
        "gen_max",
    },
    "rl_policy": {
        "reward", "old_log_prob", "Role.RefPolicy", "values", "adv"
    },
    "training": {
        "update_critic", "update_actor"
    }
}

# Metadata fields to always keep
METADATA_FIELDS = {"iteration", "phase", "timestamp"}


def _normalize_dir(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def resolve_input_dirs(
    root: Path,
    name: str | None,
    data_dir: str | None,
    val_dir: str | None,
    include_val: bool,
) -> list[Path]:
    """
    Resolve input directories with precedence:
      1) explicit --data-dir/--val-dir
      2) --name (data=<name>, val=<name>_val)
      3) defaults (monitoring_small_data / monitoring_small_val)
    """
    if data_dir or val_dir:
        dirs = []
        if data_dir:
            dirs.append(_normalize_dir(root, data_dir))
        if include_val and val_dir:
            dirs.append(_normalize_dir(root, val_dir))
        return dirs
    if name:
        return [root / name] + ([root / f"{name}_val"] if include_val else [])
    return [root / DEFAULT_DATA_DIR] + ([root / DEFAULT_VAL_DIR] if include_val else [])


def resolve_output_root(root: Path, name: str | None, out_dir: str | None) -> Path:
    if out_dir:
        return _normalize_dir(root, out_dir)
    if name:
        return root / f"{name}_cleaned"
    return root / DEFAULT_OUT_DIR


def clean_phase_timings(input_file: Path, output_file: Path) -> None:
    """
    Clean a phase timings file by filtering operations per phase.
    
    Args:
        input_file: Path to input phase_timings_*.jsonl file
        output_file: Path to output cleaned file
    """
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cleaned_count = 0
    total_count = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            total_count += 1
            entry = json.loads(line.strip())
            
            phase_name = entry.get("phase", "")
            relevant_ops = PHASE_OPERATIONS.get(phase_name, set())
            
            # Create cleaned entry with metadata
            cleaned_entry = {k: v for k, v in entry.items() if k in METADATA_FIELDS}
            
            # Add only relevant timing fields for this phase
            for key, value in entry.items():
                if key not in METADATA_FIELDS and key in relevant_ops:
                    cleaned_entry[key] = value
            
            # Write cleaned entry
            fout.write(json.dumps(cleaned_entry) + '\n')
            cleaned_count += 1
    
    print(f"✓ Cleaned {input_file.name}: {total_count} entries processed")
    return cleaned_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean phase_timings_*.jsonl by keeping only phase-specific operations."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Repo root containing monitoring folders (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        help="Base folder name. Uses <name> and <name>_val under --root.",
    )
    parser.add_argument(
        "--data-dir",
        help="Explicit data directory (absolute or relative to --root).",
    )
    parser.add_argument(
        "--val-dir",
        help="Explicit val directory (absolute or relative to --root).",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also clean validation folders (default: data only).",
    )
    parser.add_argument(
        "--out-dir",
        help="Output cleaned directory (absolute or relative to --root).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for data cleanup."""
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    input_dirs = resolve_input_dirs(
        root,
        args.name,
        args.data_dir,
        args.val_dir,
        include_val=args.include_val,
    )
    output_root = resolve_output_root(root, args.name, args.out_dir)
    
    total_files = 0
    processed_files = 0
    
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"⚠ Skipping {input_dir.name} (does not exist)")
            continue
        
        # Find all phase_timings_*.jsonl files recursively (files live in sweep subfolders)
        phase_timing_files = list(input_dir.rglob("phase_timings_*.jsonl"))
        total_files += len(phase_timing_files)
        
        if not phase_timing_files:
            print(f"⚠ No phase_timings_*.jsonl files found in {input_dir.name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {input_dir.name}/ ({len(phase_timing_files)} files)")
        print(f"{'='*80}")
        
        for input_file in sorted(phase_timing_files):
            # Write into the cleaned output tree, mirroring run folder names.
            cleaned_name = input_file.name.replace(
                "phase_timings_", "cleaned_phase_timings_", 1
            )
            output_dir = output_root / input_file.parent.name
            output_file = output_dir / cleaned_name
            
            try:
                clean_phase_timings(input_file, output_file)
                processed_files += 1
            except Exception as e:
                print(f"✗ Error processing {input_file.name}: {e}", file=sys.stderr)
    
    print(f"\n{'='*80}")
    print(f"Summary: {processed_files}/{total_files} files cleaned successfully")
    print(f"Output root: {output_root}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
