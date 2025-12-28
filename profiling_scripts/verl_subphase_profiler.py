#!/usr/bin/env python3
"""
Sub-phase profiler for verl RLHF training.
Extends the basic phase profiler to capture fine-grained timing data.

This module provides backwards-compatible enhancement to phase-level profiling
by capturing the timing_raw dictionary from marked_timer instrumentation.
"""

import time
import json
from pathlib import Path
from typing import Dict

# Import the original profiler
import sys
sys.path.insert(0, '/home/cathxhou/projects/verl_research')
from verl_phase_profiler import PhaseProfiler, PhaseReader, PHASE_IDS, PhaseType


class SubPhaseProfiler(PhaseProfiler):
    """
    Enhanced profiler that captures sub-phase timings in addition to phase transitions.
    
    Inherits all functionality from PhaseProfiler and adds timing log capability.
    Can operate in two modes:
    - granularity='phase': Only track phase-level (same as PhaseProfiler)
    - granularity='operation': Track operation-level timings (sub-phase)
    """
    
    def __init__(self, experiment_name: str, enable: bool = True, granularity: str = 'phase'):
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
        if self.granularity == 'operation':
            # Create timing log file (JSONL format - one JSON object per line)
            monitoring_dir = Path("/home/cathxhou/projects/verl_research/monitoring")
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
            phase_name: Current phase name (rollout, rl_policy, training)
            iteration: Current training iteration
        """
        if not self.enabled:
            return
        
        # Only log timings if we're doing operation-level profiling
        if self.granularity != 'operation':
            return
        
        # Create timing entry
        timing_entry = {
            "iteration": iteration,
            "phase": phase_name,
            "timestamp": time.time()
        }
        
        # Add all timing measurements
        timing_entry.update(timing_dict)
        
        # Append to JSONL file (one line per phase completion)
        with open(self.timing_log_file, 'a') as f:
            f.write(json.dumps(timing_entry) + '\n')
    
    def cleanup(self):
        """Clean up resources including timing log file."""
        super().cleanup()
        if self.enabled and hasattr(self, 'timing_log_file') and self.timing_log_file.exists():
            try:
                # Don't delete timing log - it's valuable data!
                # self.timing_log_file.unlink()
                pass
            except Exception as e:
                print(f"Warning: Timing log cleanup issue: {e}")


# Re-export PhaseReader for convenience
__all__ = ['SubPhaseProfiler', 'PhaseReader', 'PHASE_IDS', 'PhaseType']