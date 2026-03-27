from __future__ import annotations

from pathlib import Path

from plots.scale.non_results._phase_transition_h200_common import build_phase_transition_plot


OUTPATH = Path("plots/out/scale/non_results/phase_transition_temperature_h200.png")


def main() -> None:
    build_phase_transition_plot(
        metric_col="temp_gpu_C",
        metric_name="temp",
        ylabel_suffix="GPU Temp (C)",
        figure_title="GPU Temperature Recovery Across Phase Transitions, H200 Scaling Runs",
        outpath=OUTPATH,
    )


if __name__ == "__main__":
    main()
