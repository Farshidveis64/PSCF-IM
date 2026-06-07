"""A single, publication-ready matplotlib style used by every figure script."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette mirrors the LaTeX/TikZ figures in the manuscript.
GREEN = "#1E783C"   # fair channel
RED = "#E53935"     # unfair channel
GRAY = "#828282"    # off-axis / neutral
TEAL = "#1E786E"    # "ours"
BLUE = "#3478B4"    # baseline


def use_paper_style() -> None:
    """Apply a compact, serif, vector-friendly style for camera-ready figures."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,   # editable text in the PDF
        "ps.fonttype": 42,
    })


def savefig(fig: "plt.Figure", stem: str) -> None:
    """Save a figure as both ``.pdf`` (vector, for LaTeX) and ``.png`` (preview)."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
