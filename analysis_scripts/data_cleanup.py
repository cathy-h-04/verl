#!/usr/bin/env python3
"""
Clean up phase timing data by removing irrelevant fields from each phase.

This script reads phase_timings_*.jsonl files and filters each phase log entry
to only contain operations that belong to that specific phase, removing the
accumulated timing data from previous phases.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set

# Monitoring directory constants (easy to change later)
MONITORING_DATA_DIR = "monitoring_small_data"
MONITORING_VAL_DIR = "monitoring_small_val"

# Define which operations belong to each phase
PHASE_OPERATIONS: Dict[str, Set[str]] = {
    "rollout": {
        "start_profile", "generate_sequences", "generation_timing/max", 
        "generation_timing/min", "generation_timing/topk_ratio", "gen", "gen_max"
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


def main():
    """Main entry point for data cleanup."""
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    # Define input and output directories
    input_dirs = [
        project_root / MONITORING_DATA_DIR,
        project_root / MONITORING_VAL_DIR,
    ]
    
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
            # Write back next to the source file with a cleaned_ prefix
            cleaned_name = input_file.name.replace(
                "phase_timings_", "cleaned_phase_timings_", 1
            )
            output_file = input_file.with_name(cleaned_name)
            
            try:
                clean_phase_timings(input_file, output_file)
                processed_files += 1
            except Exception as e:
                print(f"✗ Error processing {input_file.name}: {e}", file=sys.stderr)
    
    print(f"\n{'='*80}")
    print(f"Summary: {processed_files}/{total_files} files cleaned successfully")
    print("Output location: next to each source phase timings file")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
