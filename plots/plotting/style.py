"""Paper-oriented style helpers with stable global invariants."""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Invariants we keep globally stable to avoid style churn.
_TITLE_SIZE = 12
_LABEL_SIZE = 11
_TICK_SIZE = 10
_LEGEND_SIZE = 9
_GRID_ALPHA = 0.22
_GRID_LINESTYLE = "--"
_GRID_LINEWIDTH = 0.6
_SAVE_DPI = 300


def scatter_paper(ax: Axes) -> Axes:
    """Apply stable defaults for paper scatter plots."""
    _apply_common_axis_style(ax)
    _configure_legend_defaults(ax)
    return ax


def line_paper(ax: Axes) -> Axes:
    """Apply stable defaults for paper line plots."""
    _apply_common_axis_style(ax)
    _configure_legend_defaults(ax)
    return ax


def bar_paper(ax: Axes) -> Axes:
    """Apply stable defaults for paper bar plots."""
    _apply_common_axis_style(ax)
    _configure_legend_defaults(ax)
    return ax


def heatmap_paper(ax: Axes) -> Axes:
    """Apply stable defaults for paper heatmap plots."""
    _apply_common_axis_style(ax)
    _configure_legend_defaults(ax)
    return ax


def savefig_paper(fig: Figure, outpath: str | Path) -> Path:
    """Always save paper figures as PNG with tight bounding box."""
    target = Path(outpath)
    if target.suffix.lower() != ".png":
        target = target.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="png", dpi=_SAVE_DPI, bbox_inches="tight")
    return target


def _apply_common_axis_style(ax: Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", linestyle=_GRID_LINESTYLE, linewidth=_GRID_LINEWIDTH, alpha=_GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=_TICK_SIZE)

    ax.xaxis.label.set_size(_LABEL_SIZE)
    ax.yaxis.label.set_size(_LABEL_SIZE)
    ax.title.set_size(_TITLE_SIZE)


def _configure_legend_defaults(ax: Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_fontsize(_LEGEND_SIZE)
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(_LEGEND_SIZE)
