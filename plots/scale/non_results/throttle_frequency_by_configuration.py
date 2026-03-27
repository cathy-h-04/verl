"""Alias entrypoint for the older throttle-by-configuration non-results filename."""

from __future__ import annotations

from pathlib import Path

import plots.scale.non_results.throttle_frequency as target


def main() -> None:
    target.OUTPATH = Path("plots/out/scale/non_results/throttle_frequency_by_configuration.png")
    target.MANIFEST_PATH = target.OUTPATH.with_suffix(".manifest.json")
    target.main()


if __name__ == "__main__":
    main()
