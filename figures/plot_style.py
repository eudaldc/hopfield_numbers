"""Shared house style for every figure in the paper (main text + SM).

Single source of truth for font scaling, rcParams and the paper's palette, so
that the main-text figures (``fig1_conceptual_wells_ens``,
``fig3_comparison_ncrit``, ``trial_level_cofit_figure``) and the SM figures
(``trial_level_pexact_decomposition``, ``trial_level_weber_fraction``,
``decarli_dd_summary_figure``) stay visually consistent. Import from here
instead of redefining these constants locally.

    from figures.plot_style import get_style, PALETTE, C_ATTR, C_STATIC, C_BENCH, panel_label
"""

from __future__ import annotations

import matplotlib as mpl

# Seaborn "deep" palette — the paper's one discrete palette. Pick colors from
# here (or a tint/shade of one of them) rather than inventing new hues.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# Roles reused across figures: the winning attractor model is always blue, its
# static (no clean-up) twin a lighter tint of that same blue, and benchmark /
# comparison models a neutral grey.
C_ATTR = PALETTE[0]     # "#4C72B0" — the winning attractor (clean-up) model
C_STATIC = "#A3C0DE"    # light tint of C_ATTR — same wells, static readout
C_BENCH = "#9A9A9A"     # grey — Weber/ANS, normative and other benchmarks
C_WARN = PALETTE[3]     # "#C44E52" — the palette's red, for contrast/alerts


def get_style(font_scale: float) -> dict:
    """rcParams for a given font_scale. fig1/fig3 (large single-column canvases,
    2-3 stacked panels) use font_scale=2.1; dense multi-panel grids should pass
    a smaller value tuned so text does not crowd the smaller panels — both still
    render at the same physical size once the paper shrinks the figure to
    \\columnwidth, since canvas width in inches is comparable across figures."""
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 8.5 * font_scale,
        "font.size": 10 * font_scale,   # default for ax.text/annotate without explicit size
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }


def apply_style(font_scale: float) -> None:
    mpl.rcParams.update(get_style(font_scale))


def panel_label(ax, letter: str, x: float = 0.015, **kwargs) -> None:
    """Bold lowercase panel letter ("a", "b", "c", ...) at the top-left of ax,
    matching the convention used throughout the paper's figures."""
    style = dict(fontweight="bold", loc="left", x=x)
    style.update(kwargs)
    ax.set_title(letter, **style)
