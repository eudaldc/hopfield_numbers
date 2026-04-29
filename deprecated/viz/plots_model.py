"""H2/H3 — Stimulus diagnostics and model-geometry plots.

Works with any core that exposes ``.centers()``, ``.widths()``,
``.pairwise_separation()``, and ``.estimate_kstar()`` (AttractorCore,
ANSCore, AccumulatorCore all do).
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.viz import PALETTE, save_figure


# =====================================================================
# H3 — geometry of the unified model
# =====================================================================


@torch.no_grad()
def plot_centers(
    core,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """mu_n vs n — well centres on the latent axis."""
    ints = core.integers.cpu().numpy()
    mu = core.centers().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ints, mu, "o-", color=PALETTE[0], markersize=5)
    ax.set_xlabel("integer n")
    ax.set_ylabel("μ_n  (latent position)")
    ax.set_title("Well centres")
    fig.tight_layout()
    save_figure(
        fig,
        "model_centers",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_widths(
    core,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """sigma_n vs n — population-level well widths."""
    ints = core.integers.cpu().numpy()
    sigma = core.widths().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ints, sigma, "s-", color=PALETTE[1], markersize=5)
    ax.set_xlabel("integer n")
    ax.set_ylabel("σ_n")
    ax.set_title("Well widths")
    fig.tight_layout()
    save_figure(
        fig,
        "model_widths",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_separation(
    core,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Pairwise separation between neighbouring wells."""
    ints = core.integers.cpu().numpy()
    sep = core.pairwise_separation().cpu().numpy()

    fig, ax = plt.subplots()
    midpoints = 0.5 * (ints[:-1] + ints[1:])
    ax.bar(midpoints, sep, width=0.7, color=PALETTE[2], edgecolor="white")
    ax.axhline(1.0, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("n  (midpoint of consecutive pair)")
    ax.set_ylabel("separation  Δμ / σ̄")
    ax.set_title("Neighbouring-well separation")
    fig.tight_layout()
    save_figure(
        fig,
        "model_separation",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_posterior_heatmap(
    core,
    subject_idx: int = 0,
    n_y_points: int = 200,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of p(n | y) across a range of evidence values y.

    Uses the geometry of a single representative subject.
    """
    device = core.integers.device
    mu = core.centers()
    y_lo = mu.min().item() - 3.0
    y_hi = mu.max().item() + 3.0
    y_grid = torch.linspace(y_lo, y_hi, n_y_points, device=device)

    idx = torch.full((n_y_points,), subject_idx, dtype=torch.long, device=device)
    probs = core(y_grid, idx)["probs"].cpu().numpy()          # [n_y, K]

    ints = core.integers.cpu().numpy()
    y_np = y_grid.cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(
        ints, y_np, probs,
        shading="nearest", cmap="viridis",
    )
    ax.set_xlabel("integer n")
    ax.set_ylabel("latent evidence y")
    ax.set_title("p(n | y) — posterior heatmap")
    fig.colorbar(im, ax=ax, label="p(n | y)")
    fig.tight_layout()
    save_figure(
        fig,
        "model_posterior_heatmap",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_energy_landscape(
    core,
    subject_idx: int = 0,
    n_y_points: int = 300,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Landscape: logit energy (un-normalised) as a function of y.

    Shows one curve per integer, illustrating well shape.
    """
    device = core.integers.device
    mu = core.centers()
    y_lo = mu.min().item() - 3.0
    y_hi = mu.max().item() + 3.0
    y_grid = torch.linspace(y_lo, y_hi, n_y_points, device=device)

    idx = torch.full((n_y_points,), subject_idx, dtype=torch.long, device=device)
    logits = core(y_grid, idx)["logits"].cpu().numpy()        # [n_y, K]

    ints = core.integers.cpu().numpy()
    y_np = y_grid.cpu().numpy()
    K = len(ints)

    fig, ax = plt.subplots(figsize=(9, 5))
    step = max(1, K // 10)
    for j in range(0, K, step):
        ax.plot(y_np, logits[:, j], color=PALETTE[j % len(PALETTE)],
                alpha=0.7, label=f"n={ints[j]:.0f}")

    ax.set_xlabel("latent evidence y")
    ax.set_ylabel("logit energy")
    ax.set_title("Energy landscape (selected wells)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(
        fig,
        "model_energy_landscape",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_wells_gaussian(
    core,
    subject_idx: int = 0,
    n_y_points: int = 500,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Gaussian density for each well, showing overlap growth with n."""
    device = core.integers.device
    mu = core.centers().cpu().numpy()
    sigma_pop = core.widths()

    idx_t = torch.tensor([subject_idx], device=device)
    width_s = core.subject_params.width_scale(idx_t).item()
    sigma = (sigma_pop * width_s).cpu().numpy()

    ints = core.integers.cpu().numpy()
    K = len(ints)

    y_lo = mu[0] - 4 * sigma[0]
    y_hi = mu[-1] + 4 * sigma[-1]
    y_grid = np.linspace(y_lo, y_hi, n_y_points)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ints[0], vmax=ints[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(K):
        g = np.exp(-0.5 * ((y_grid - mu[j]) / sigma[j]) ** 2) / (
            sigma[j] * np.sqrt(2 * np.pi)
        )
        colour = cmap(norm(ints[j]))
        ax.plot(y_grid, g, color=colour, alpha=0.75, lw=1.3)
        ax.fill_between(y_grid, g, alpha=0.12, color=colour)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="integer n", pad=0.02)

    ax.set_xlabel("latent position  y")
    ax.set_ylabel("density  p(y | n)")
    ax.set_title("Attractor wells (Gaussian densities)")
    fig.tight_layout()
    save_figure(
        fig,
        "model_wells_gaussian",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_precision(
    core,
    subject_idx: int = 0,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Precision (1/σ_n) vs n for a single subject."""
    device = core.integers.device
    sigma_pop = core.widths()

    idx_t = torch.tensor([subject_idx], device=device)
    width_s = core.subject_params.width_scale(idx_t).item()
    sigma = (sigma_pop * width_s).cpu().numpy()

    ints = core.integers.cpu().numpy()
    precision = 1.0 / sigma

    fig, ax = plt.subplots()
    ax.plot(ints, precision, "o-", color=PALETTE[0], markersize=4)
    ax.set_xlabel("integer n")
    ax.set_ylabel("precision  1/σ_n")
    ax.set_title("Precision decay")
    fig.tight_layout()
    save_figure(
        fig,
        "model_precision",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


@torch.no_grad()
def plot_kstar_distribution(
    core,
    n_subjects: int,
    q: float = 0.75,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Histogram of the useful-discreteness threshold K* across subjects."""
    device = core.integers.device
    idx = torch.arange(n_subjects, device=device)
    kstar = core.estimate_kstar(idx, q=q).cpu().numpy()

    fig, ax = plt.subplots()
    bins = np.arange(kstar.min() - 0.5, kstar.max() + 1.5, 1.0)
    ax.hist(kstar, bins=bins, color=PALETTE[4], edgecolor="white")
    ax.axvline(np.median(kstar), ls="--", color=PALETTE[3], lw=1.3,
               label=f"median={np.median(kstar):.1f}")
    ax.set_xlabel("K*")
    ax.set_ylabel("n subjects")
    ax.set_title(f"K* distribution  (q={q})")
    ax.legend()
    fig.tight_layout()
    save_figure(
        fig,
        "model_kstar_distribution",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


# =====================================================================
# H2 — stimulus diagnostics
# =====================================================================

def plot_feature_correlations(
    feature_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Correlation matrix between true numerosity and visual cue features."""
    if feature_cols is None:
        from src.data.schemas import FEATURE_COLUMNS
        feature_cols = FEATURE_COLUMNS

    cols = ["true_n"] + [c for c in feature_cols if c in feature_df.columns]
    corr = feature_df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature × numerosity correlations")
    fig.tight_layout()
    save_figure(fig, "stim_feature_correlations", save_dir, wandb_run, close=False)
    return fig


def plot_feature_distributions(
    feature_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Histograms of each stimulus feature, one panel per feature."""
    if feature_cols is None:
        from src.data.schemas import FEATURE_COLUMNS
        feature_cols = FEATURE_COLUMNS

    cols = [c for c in feature_cols if c in feature_df.columns]
    n = len(cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature columns found", ha="center",
                va="center", transform=ax.transAxes)
        save_figure(fig, "stim_feature_distributions", save_dir, wandb_run, close=False)
        return fig

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax, col in zip(axes.ravel(), cols):
        vals = feature_df[col].dropna()
        ax.hist(vals, bins=40, color=PALETTE[0], edgecolor="white", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle("Stimulus feature distributions", y=1.02)
    fig.tight_layout()
    save_figure(fig, "stim_feature_distributions", save_dir, wandb_run, close=False)
    return fig
