#!/usr/bin/env python3
"""
Phase profiler for verl RLHF training.
Provides IPC between controller and monitoring script.
"""

import time
import json
from pathlib import Path
from typing import Dict, Literal

PhaseType = Literal["idle", "rollout", "rl_policy", "training"]

PHASE_IDS = {
    "idle": 0,
    "rollout": 1,
    "rl_policy": 2,
    "training": 3
}

class PhaseProfiler:
    """Writer class - used by verl trainer to signal phase transitions."""
    
    def __init__(self, experiment_name: str, enable: bool = True):
        self.experiment_name = experiment_name
        self.enabled = enable
        self.current_phase = "idle"
        self.current_iteration = 0
        self.phase_start_time = None
        
        if not self.enabled:
            return
        
        # Use file-based IPC in monitoring directory
        self.state_file = Path("/home/cathxhou/projects/verl_research/monitoring") / f"phase_state_{experiment_name}.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize with idle state
        self._write_state({
            "phase_id": PHASE_IDS["idle"],
            "phase_name": "idle",
            "iteration": 0,
            "timestamp": time.time()
        })
        print(f"✓ Phase profiler initialized: {self.state_file}")
    
    def _write_state(self, state: Dict):
        """Write state to file atomically."""
        if not self.enabled:
            return
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
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
        
        self._write_state({
            "phase_id": PHASE_IDS[phase_name],
            "phase_name": phase_name,
            "iteration": self.current_iteration,
            "timestamp": self.phase_start_time
        })
    
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
        self.state_file = Path("/home/cathxhou/projects/verl_research/monitoring") / f"phase_state_{experiment_name}.json"
    
    def get_current_phase(self) -> Dict:
        """Read the current phase state."""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "phase_id": 0,
                "phase_name": "idle",
                "iteration": 0,
                "timestamp": time.time()
            }