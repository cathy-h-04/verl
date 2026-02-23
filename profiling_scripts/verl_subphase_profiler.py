#!/usr/bin/env python3
"""
Phase + sub-phase profiler for verl RLHF training.
Provides IPC between controller and monitoring script, with optional sub-phase timing logs.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Literal

PhaseType = Literal["idle", "rollout", "rl_policy", "training", "validation", "other"]

PHASE_IDS = {
    "idle": 0,
    "rollout": 1,
    "rl_policy": 2,
    "training": 3,
    "validation": 4,
    "other": 5,
}


class PhaseProfiler:
    """Writer class - used by verl trainer to signal phase transitions."""

    def __init__(self, experiment_name: str, enable: bool = True):
        self.experiment_name = experiment_name
        self.enabled = enable
        self.current_phase: PhaseType = "idle"
        self.current_iteration = 0
        self.phase_start_time = None

        if not self.enabled:
            return

        # Use file-based IPC in monitoring directory
        scratch_dir = os.environ.get("SCRATCH_DIR", "/n/netscratch/yu_lab/Lab/chou")
        monitoring_dir = (
            os.environ.get("MONITORING_DIR")
            or os.environ.get("VERL_FILE_LOGGER_ROOT")
            or f"{scratch_dir}/logs"
        )
        self.state_file = Path(monitoring_dir) / f"phase_state_{experiment_name}.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with idle state
        self._write_state(
            {
                "phase_id": PHASE_IDS["idle"],
                "phase_name": "idle",
                "iteration": 0,
                "timestamp": time.time(),
            }
        )
        print(f"✓ Phase profiler initialized: {self.state_file}")

    def _write_state(self, state: Dict):
        """Write state to file atomically."""
        if not self.enabled:
            return
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state, f)
        temp_file.replace(self.state_file)

    def mark_phase_start(self, phase_name: PhaseType, iteration: int = None):
        """Mark the start of a training phase."""
        if not self.enabled:
            return

        self.current_phase = phase_name
        if iteration is not None:
            self.current_iteration = iteration
        self.phase_start_time = time.time()

        self._write_state(
            {
                "phase_id": PHASE_IDS[phase_name],
                "phase_name": phase_name,
                "iteration": self.current_iteration,
                "timestamp": self.phase_start_time,
            }
        )

    def mark_phase_end(self, phase_name: PhaseType = None):
        """Mark the end of a training phase."""
        if not self.enabled:
            return 0.0

        if self.phase_start_time:
            duration = time.time() - self.phase_start_time
            return duration
        return 0.0

    def cleanup(self):
        """Clean up resources."""
        if self.enabled and self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")


class PhaseReader:
    """Reader class - used by monitoring script to query current phase."""

    def __init__(self, experiment_name: str):
        scratch_dir = os.environ.get("SCRATCH_DIR", "/n/netscratch/yu_lab/Lab/chou")
        monitoring_dir = (
            os.environ.get("MONITORING_DIR")
            or os.environ.get("VERL_FILE_LOGGER_ROOT")
            or f"{scratch_dir}/logs"
        )
        self.state_file = Path(monitoring_dir) / f"phase_state_{experiment_name}.json"

    def get_current_phase(self) -> Dict:
        """Read the current phase state."""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "phase_id": 0,
                "phase_name": "idle",
                "iteration": 0,
                "timestamp": time.time(),
            }


class SubPhaseProfiler(PhaseProfiler):
    """
    Enhanced profiler that captures sub-phase timings in addition to phase transitions.

    Inherits all functionality from PhaseProfiler and adds timing log capability.
    Can operate in two modes:
    - granularity='phase': Only track phase-level (same as PhaseProfiler)
    - granularity='operation': Track operation-level timings (sub-phase)
    """

    def __init__(self, experiment_name: str, enable: bool = True, granularity: str = "phase"):
        """
        Initialize sub-phase profiler.

        Args:
            experiment_name: Unique name for this experiment
            enable: Whether profiling is enabled
            granularity: 'phase' for phase-level only, 'operation' for sub-phase tracking
        """
        # Initialize parent class (handles phase state file)
        super().__init__(experiment_name, enable)

        if not self.enabled:
            return

        self.granularity = granularity

        # Only create timing log if we're doing operation-level profiling
        if self.granularity == "operation":
            # Create timing log file (JSONL format - one JSON object per line)
            scratch_dir = os.environ.get("SCRATCH_DIR", "/n/netscratch/yu_lab/Lab/chou")
            monitoring_dir_env = os.environ.get("MONITORING_DIR")
            if monitoring_dir_env:
                monitoring_dir = Path(monitoring_dir_env)
            else:
                monitoring_root = os.environ.get("VERL_FILE_LOGGER_ROOT") or f"{scratch_dir}/logs"
                monitoring_dir = Path(monitoring_root) / experiment_name
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            self.timing_log_file = monitoring_dir / f"phase_timings_{experiment_name}.jsonl"

            # Clear any existing log file
            if self.timing_log_file.exists():
                self.timing_log_file.unlink()

            print(f"✓ Sub-phase profiler initialized (granularity: {granularity})")
            print(f"  Phase state: {self.state_file}")
            print(f"  Timing log: {self.timing_log_file}")
        else:
            print(f"✓ Phase profiler initialized (granularity: {granularity})")
            print(f"  Phase state: {self.state_file}")

    def log_timings(self, timing_dict: Dict[str, float], phase_name: str, iteration: int):
        """
        Log timing data for sub-phase analysis.

        This captures the timing_raw dictionary from verl's marked_timer instrumentation
        and associates it with the current phase and iteration.

        Only logs if granularity is set to 'operation'.

        Args:
            timing_dict: Dictionary of operation names to durations (from marked_timer)
            phase_name: Current phase name (rollout, rl_policy, training, validation, other)
            iteration: Current training iteration
        """
        if not self.enabled:
            return

        # Only log timings if we're doing operation-level profiling
        if self.granularity != "operation":
            return

        # Create timing entry
        timing_entry = {
            "iteration": iteration,
            "phase": phase_name,
            "timestamp": time.time(),
        }

        # Add all timing measurements
        timing_entry.update(timing_dict)

        # Append to JSONL file (one line per phase completion)
        with open(self.timing_log_file, "a") as f:
            f.write(json.dumps(timing_entry) + "\n")

    def cleanup(self):
        """Clean up resources including timing log file."""
        super().cleanup()
        if self.enabled and hasattr(self, "timing_log_file") and self.timing_log_file.exists():
            try:
                # Don't delete timing log - it's valuable data!
                # self.timing_log_file.unlink()
                pass
            except Exception as e:
                print(f"Warning: Timing log cleanup issue: {e}")


__all__ = [
    "SubPhaseProfiler",
    "PhaseProfiler",
    "PhaseReader",
    "PHASE_IDS",
    "PhaseType",
]
