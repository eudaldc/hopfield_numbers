"""Visualization suite for the Numerical Attractors PoC.

Shared style configuration, saving helpers, and optional W&B logging.
All plot functions follow the convention::

    fig = plot_something(data, ..., save_dir=None, wandb_run=None)

When *save_dir* is not None the figure is written to disk.
When *wandb_run* is not None the figure is logged as a W&B image.
Optional *file_suffix* appends ``_{suffix}`` to the PNG stem; *run_label*
draws a short caption on the figure.
"""

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── house style ──────────────────────────────────────────────────────
STYLE: dict = {
    "figure.figsize": (7, 5),
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def apply_style() -> None:
    """Apply the project-wide matplotlib style."""
    mpl.rcParams.update(STYLE)


def save_figure(
    fig: plt.Figure,
    name: str,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    close: bool = True,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> None:
    """Save *fig* to disk and optionally log to W&B.

    Parameters
    ----------
    fig       : matplotlib figure
    name      : filename stem (without extension)
    save_dir  : directory for the PNG; skipped when None
    wandb_run : active ``wandb.Run`` instance; skipped when None
    close     : close the figure after saving (default True)
    run_label : optional short line drawn on the figure (e.g. checkpoint preset)
    wandb_prefix : optional sub-key under ``output/`` for W&B (e.g. preset slug)
    file_suffix : optional slug appended to the stem: ``{name}_{file_suffix}.png``
    """
    if run_label:
        fig.text(
            0.99,
            0.01,
            run_label,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.45",
            transform=fig.transFigure,
        )

    stem = f"{name}_{file_suffix}" if file_suffix else name

    if save_dir is not None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{stem}.png")

    if wandb_run is not None:
        try:
            import wandb

            wb_name = f"{wandb_prefix}/{stem}" if wandb_prefix else stem
            wandb_run.log({f"output/{wb_name}": wandb.Image(fig)})
        except Exception:
            pass

    if close:
        plt.close(fig)


def compose_panel(
    figures: list[plt.Figure],
    suptitle: str | None = None,
    max_cols: int = 3,
    cell_size: tuple[float, float] = (6, 5),
    dpi: int = 200,
) -> plt.Figure:
    """Rasterize *figures* and tile them into a single summary panel.

    Each figure is rendered to a PNG buffer, placed as an image in a grid
    cell, then closed.  Returns a new Figure containing the assembled panel.
    """
    import io
    import matplotlib.image as mpimg

    rasters: list = []
    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        rasters.append(mpimg.imread(buf))
        buf.close()
        plt.close(fig)

    n = len(rasters)
    if n == 0:
        panel = plt.figure()
        panel.text(0.5, 0.5, "No plots to show", ha="center", va="center")
        return panel

    ncols = min(n, max_cols)
    nrows = -(-n // ncols)

    w, h = cell_size
    panel, axes = plt.subplots(
        nrows, ncols,
        figsize=(w * ncols, h * nrows),
        squeeze=False,
    )

    for idx, img in enumerate(rasters):
        r, c = divmod(idx, ncols)
        axes[r][c].imshow(img, aspect="auto")
        axes[r][c].axis("off")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    if suptitle:
        panel.suptitle(suptitle, fontsize=14, fontweight="bold")

    panel.tight_layout(pad=0.5)
    return panel


apply_style()
