"""Alias entrypoint for the older DRAM-per-token non-results filename."""

from __future__ import annotations

from pathlib import Path

import plots.scale.dram_power_by_configuration as target


def main() -> None:
    target.OUTPATH = Path("plots/out/scale/non_results/cpu_gpu_interaction_dram_energy_by_configuration.png")
    target.MANIFEST_PATH = target.OUTPATH.with_suffix(".manifest.json")
    target.main()


if __name__ == "__main__":
    main()
