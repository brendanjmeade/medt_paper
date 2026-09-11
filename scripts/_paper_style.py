"""Common matplotlib style for manuscript figures.

Importing this module sets a paper-grade rcParams configuration. No emojis,
no decorative elements; figures are produced as PDF (vector) for the LaTeX
build and as PNG for fast iteration.
"""

from __future__ import annotations

import matplotlib as mpl

PT_PER_IN = 72.27
COL_WIDTH_PT = 246.0
TEXT_WIDTH_PT = 510.0


def set_paper_style() -> None:
    """Set rcParams matching the manuscript house style.

    Per `.claude/skills/matplotlib-figure-style/SKILL.md`:
    - keep top + right spines (full box)
    - tick direction "out", short ticks, no minor ticks
    - no grid by default
    - serif text for math compatibility (Times/STIX)
    - high dpi, tight bbox
    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "axes.grid": False,
            "savefig.dpi": 500,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def panel_letter(ax, letter: str, loc: str = "upper right") -> None:
    """Place a panel letter inside the axes per SKILL.md (top-right plain
    text, no parens, no bbox)."""
    if loc == "upper right":
        x, y, ha, va = 0.95, 0.95, "right", "top"
    elif loc == "upper left":
        x, y, ha, va = 0.05, 0.95, "left", "top"
    elif loc == "lower right":
        x, y, ha, va = 0.95, 0.05, "right", "bottom"
    else:
        x, y, ha, va = 0.05, 0.05, "left", "bottom"
    ax.text(x, y, letter, transform=ax.transAxes, ha=ha, va=va,
             fontsize=10, fontweight="bold")


def col_size(height_in: float = 2.6) -> tuple[float, float]:
    """Single-column figure size: width = 246 pt."""
    return (COL_WIDTH_PT / PT_PER_IN, height_in)


def text_size(height_in: float = 4.0) -> tuple[float, float]:
    """Full-text-width figure size: width = 510 pt."""
    return (TEXT_WIDTH_PT / PT_PER_IN, height_in)


set_paper_style()
