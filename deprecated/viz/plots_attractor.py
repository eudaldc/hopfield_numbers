"""Dynamic-attractor visualization suite (WP-6).

Six plot families
-----------------
A  Cleanup map      — y → x*(y) plateau structure
B  Basin map        — y → n̂(y) decision boundary as a step function
C  Trajectories     — x_t and E(x_t | y) over cleanup iterations
D  RT proxy curves  — accuracy, RT, and error vs true numerosity N
E  Phase diagrams   — 1-D curves and 2-D heatmaps from a parameter scan
F  Comparison panel — side-by-side accuracy / RT / MAE + cleanup maps

All plot functions follow the project convention::

    fig = plot_something(..., save_dir=None, wandb_run=None)

Helper
------
evaluate_comparison — evaluates a dict of (model, rt_proxy) pairs on
                      synthetic Task-A data and returns per-N metrics
                      ready for D and F.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.viz import PALETTE, save_figure


# ── A. Cleanup map ─────────────────────────────────────────────────────────


@torch.no_grad()
def plot_cleanup_map(
    core,
    n_points: int = 500,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Plot y → x*(y): the attractor cleanup transformation.

    Plateau regions where many y values collapse to the same well confirm
    that the model is a genuine attractor rather than a static readout.

    Parameters
    ----------
    core     : DynamicAttractorCore
    n_points : resolution of the y grid
    """
    mu, sigma = core.compute_mu_sigma()
    y_lo = (mu.min() - 2.5 * sigma.max()).item()
    y_hi = (mu.max() + 2.5 * sigma.max()).item()
    y_grid = torch.linspace(y_lo, y_hi, n_points, device=mu.device)

    out = core(y_grid)
    x_star = out["x_star"].cpu().numpy()
    y_np = y_grid.cpu().numpy()
    mu_np = mu.cpu().numpy()
    ints_np = core.integers.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(y_np, y_np, "--", color="0.7", lw=1.0, label="y  (no cleanup)", zorder=1)
    ax.plot(y_np, x_star, color=PALETTE[0], lw=1.8, label="x*(y)", zorder=2)

    x_margin = 0.05 * (y_hi - y_lo)
    for m, n_int in zip(mu_np, ints_np):
        ax.axhline(m, color=PALETTE[2], lw=0.6, ls=":", alpha=0.55)
        ax.text(y_hi + x_margin, m, f"n={n_int:.0f}",
                va="center", fontsize=7, color=PALETTE[2])

    ax.set_xlabel("sensory evidence  y")
    ax.set_ylabel("cleaned state  x*")
    ax.set_title("Cleanup map  y → x*(y)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    save_figure(
        fig, "attractor_cleanup_map",
        save_dir, wandb_run, close=False,
        run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
    )
    return fig


# ── B. Basin map ────────────────────────────────────────────────────────────


@torch.no_grad()
def plot_basin_map(
    core,
    n_points: int = 500,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Plot y → n̂(y): integer decision boundary as a step function.

    Wider plateaus indicate a larger basin of attraction for that integer.
    The lower panel shows the posterior basin margin (top1 − top2).

    Parameters
    ----------
    core     : DynamicAttractorCore
    n_points : resolution of the y grid
    """
    mu, sigma = core.compute_mu_sigma()
    y_lo = (mu.min() - 2.5 * sigma.max()).item()
    y_hi = (mu.max() + 2.5 * sigma.max()).item()
    y_grid = torch.linspace(y_lo, y_hi, n_points, device=mu.device)

    out = core(y_grid)
    n_hat_np = out["n_hat"].cpu().numpy()
    margin_np = out["basin_margin"].cpu().numpy()
    y_np = y_grid.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.step(y_np, n_hat_np, where="mid", color=PALETTE[0], lw=1.5)
    ax1.set_ylabel("decoded integer  n̂")
    ax1.set_title("Basin map  y → n̂(y)")
    unique_ints = sorted(set(n_hat_np.astype(int).tolist()))
    ax1.set_yticks(unique_ints)

    ax2.fill_between(y_np, margin_np, alpha=0.45, color=PALETTE[1])
    ax2.plot(y_np, margin_np, color=PALETTE[1], lw=0.8)
    ax2.axhline(0, color="0.6", lw=0.5)
    ax2.set_ylabel("basin margin")
    ax2.set_xlabel("sensory evidence  y")

    fig.tight_layout()

    save_figure(
        fig, "attractor_basin_map",
        save_dir, wandb_run, close=False,
        run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
    )
    return fig


# ── C. Example trajectories ─────────────────────────────────────────────────


@torch.no_grad()
def plot_trajectories(
    core,
    y_selected: Optional[torch.Tensor] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Plot cleanup trajectories x_t and energy E(x_t | y) over iterations.

    Parameters
    ----------
    core       : DynamicAttractorCore (cleanup_mode should be "gradient")
    y_selected : [M] starting evidence values; defaults to a mix of well
                 centres and inter-well midpoints (at most 8 traces)
    """
    mu, sigma = core.compute_mu_sigma()
    mu_np = mu.cpu().numpy()

    if y_selected is None:
        step = max(1, len(mu_np) // 3)
        selected: list[float] = []
        for i in range(0, len(mu_np), step):
            selected.append(float(mu_np[i]))
        for i in range(0, len(mu_np) - 1, step):
            selected.append(float(0.5 * (mu_np[i] + mu_np[i + 1])))
        y_selected = torch.tensor(selected[:8], device=mu.device)
    else:
        y_selected = y_selected.to(mu.device)

    out = core(y_selected, return_traj=True)
    x_traj = out["x_traj"].cpu().numpy()       # [M, T]
    e_traj = out["energy_trace"].cpu().numpy()  # [M, T]
    T = x_traj.shape[1]
    M = x_traj.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(M):
        col = PALETTE[i % len(PALETTE)]
        y_val = y_selected[i].item()
        ax1.plot(range(T), x_traj[i], color=col, lw=1.4, alpha=0.85,
                 label=f"y={y_val:.2f}")
        ax2.plot(range(T), e_traj[i], color=col, lw=1.4, alpha=0.85)

    for m in mu_np:
        ax1.axhline(m, color="0.75", lw=0.5, ls=":")

    ax1.set_xlabel("cleanup step  t")
    ax1.set_ylabel("latent state  x_t")
    ax1.set_title("Cleanup trajectories")
    if M <= 10:
        ax1.legend(fontsize=7, ncol=2)

    ax2.set_xlabel("cleanup step  t")
    ax2.set_ylabel("energy  E(x_t | y)")
    ax2.set_title("Energy over cleanup")

    fig.tight_layout()

    save_figure(
        fig, "attractor_trajectories",
        save_dir, wandb_run, close=False,
        run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
    )
    return fig


# ── D. RT proxy curves ──────────────────────────────────────────────────────


def plot_rt_curves(
    results: dict[str, dict[str, float]],
    n_min: int = 1,
    n_max: int = 9,
    small_n_max: int = 4,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Plot accuracy, RT proxy, and MAE vs N for multiple evaluation modes.

    Parameters
    ----------
    results     : {mode_name: metrics_dict}
                  Each metrics_dict must contain acc_n{N}, rt_n{N}, mae_n{N}.
    small_n_max : upper bound of the shaded small-N region
    """
    ns = list(range(n_min, n_max + 1))
    fig, (ax_acc, ax_rt, ax_mae) = plt.subplots(1, 3, figsize=(14, 5))

    for idx, (mode, metrics) in enumerate(results.items()):
        col = PALETTE[idx % len(PALETTE)]
        kw = dict(color=col, lw=1.5, label=mode, marker="o", markersize=4)
        ax_acc.plot(ns, [metrics.get(f"acc_n{n}", float("nan")) for n in ns], **kw)
        ax_rt.plot(ns, [metrics.get(f"rt_n{n}", float("nan")) for n in ns], **kw)
        ax_mae.plot(ns, [metrics.get(f"mae_n{n}", float("nan")) for n in ns], **kw)

    for ax in (ax_acc, ax_rt, ax_mae):
        ax.axvspan(n_min - 0.5, small_n_max + 0.5, alpha=0.07, color=PALETTE[4])
        ax.set_xlabel("true numerosity  N")
        ax.set_xticks(ns)

    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy vs N")
    ax_acc.set_ylim(0, 1.05)
    ax_rt.set_ylabel("mean RT proxy")
    ax_rt.set_title("RT proxy vs N")
    ax_mae.set_ylabel("mean |error|")
    ax_mae.set_title("MAE vs N")

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(results),
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    save_figure(
        fig, "attractor_rt_curves",
        save_dir, wandb_run, close=False,
        run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
    )
    return fig


# ── E. Phase diagrams ───────────────────────────────────────────────────────


_METRIC_LABELS: dict[str, str] = {
    "acc_small": "Accuracy  (N≤4)",
    "mae_large": "MAE  (N>4)",
    "rt_slope_small": "RT slope  (small N)",
    "rt_slope_large": "RT slope  (large N)",
    "rt_slope_ratio": "RT slope ratio  (large / small)",
    "k_star": "K* geometry",
    "mean_steps": "Mean cleanup steps",
    "frac_converged": "Fraction converged",
    "mean_margin": "Mean basin margin",
    "sep_ratio_mean": "Mean separation ratio",
    "occ_entropy": "Basin occupancy entropy",
}

_DEFAULT_1D_METRICS = [
    "acc_small", "mae_large", "rt_slope_large",
    "k_star", "mean_steps", "occ_entropy",
]

_DEFAULT_2D_METRICS = [
    "acc_small", "mae_large", "rt_slope_ratio",
    "k_star", "mean_steps", "occ_entropy",
]


def plot_phase_1d(
    df: "pd.DataFrame",
    param: str,
    metrics: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple metrics as functions of a single scanned parameter.

    Parameters
    ----------
    df      : DataFrame from run_phase_scan (1-D scan)
    param   : column name of the scanned parameter (x-axis)
    metrics : metric columns to plot; defaults to key diagnostics
    """
    if metrics is None:
        metrics = [m for m in _DEFAULT_1D_METRICS if m in df.columns]

    n_metrics = len(metrics)
    ncols = min(n_metrics, 3)
    nrows = -(-n_metrics // ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )
    x_vals = df[param].values

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if metric in df.columns:
            ax.plot(x_vals, df[metric].values, "o-",
                    color=PALETTE[idx % len(PALETTE)], lw=1.5)
        label = _METRIC_LABELS.get(metric, metric)
        ax.set_xlabel(param)
        ax.set_ylabel(label)
        ax.set_title(label)

    for idx in range(n_metrics, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Phase scan — {param}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_figure(
        fig, f"phase_1d_{param}",
        save_dir, wandb_run, close=False,
        run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
    )
    return fig


def plot_phase_2d(
    df: "pd.DataFrame",
    param1: str,
    param2: str,
    metrics: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> dict[str, plt.Figure]:
    """Plot 2-D heatmaps for a two-parameter phase scan.

    Parameters
    ----------
    df      : DataFrame from run_phase_scan (2-D scan)
    param1  : column name for the x-axis parameter
    param2  : column name for the y-axis parameter
    metrics : metric columns to plot; defaults to key diagnostics

    Returns
    -------
    {metric_name: Figure} — one heatmap figure per metric
    """
    if metrics is None:
        metrics = [m for m in _DEFAULT_2D_METRICS if m in df.columns]

    p1_vals = sorted(df[param1].unique())
    p2_vals = sorted(df[param2].unique())

    figs: dict[str, plt.Figure] = {}
    for metric in metrics:
        if metric not in df.columns:
            continue

        grid = df.pivot_table(
            index=param2, columns=param1, values=metric, aggfunc="mean"
        ).reindex(index=p2_vals, columns=p1_vals)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(
            p1_vals, p2_vals, grid.values,
            shading="nearest", cmap="viridis",
        )
        label = _METRIC_LABELS.get(metric, metric)
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f"{label}  ({param1} × {param2})")
        fig.tight_layout()

        stem = f"phase_2d_{param1}x{param2}_{metric}"
        save_figure(
            fig, stem,
            save_dir, wandb_run, close=False,
            run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
        )
        figs[metric] = fig

    return figs


# ── F. Comparison panel ─────────────────────────────────────────────────────


def plot_comparison_panel(
    results: dict[str, dict[str, float]],
    cores: Optional[dict[str, object]] = None,
    n_min: int = 1,
    n_max: int = 9,
    small_n_max: int = 4,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of modes on accuracy, RT, and MAE.

    Optionally adds a row of individual cleanup maps when *cores* is provided.

    Parameters
    ----------
    results : {mode_name: metrics_dict} from evaluate_comparison
    cores   : optional {mode_name: DynamicAttractorCore} for cleanup map row;
              only modes present in this dict get a cleanup map
    """
    ns = list(range(n_min, n_max + 1))
    modes = list(results.keys())
    n_modes = len(modes)

    map_modes = list(cores.keys()) if cores else []
    n_map_cols = min(len(map_modes), 4)
    has_maps = n_map_cols > 0

    if has_maps:
        n_grid_cols = max(n_map_cols, 3)
        fig = plt.figure(figsize=(4.5 * n_grid_cols, 18))
        gs = fig.add_gridspec(4, n_grid_cols, hspace=0.5, wspace=0.35)
        ax_acc = fig.add_subplot(gs[0, :])
        ax_rt = fig.add_subplot(gs[1, :])
        ax_mae = fig.add_subplot(gs[2, :])
        map_axes = [fig.add_subplot(gs[3, c]) for c in range(n_map_cols)]
    else:
        fig, axes_row = plt.subplots(1, 3, figsize=(14, 5))
        ax_acc, ax_rt, ax_mae = axes_row
        map_axes = []

    for idx, (mode, metrics) in enumerate(results.items()):
        col = PALETTE[idx % len(PALETTE)]
        kw = dict(color=col, lw=1.5, label=mode, marker="o", markersize=4)
        ax_acc.plot(ns, [metrics.get(f"acc_n{n}", float("nan")) for n in ns], **kw)
        ax_rt.plot(ns, [metrics.get(f"rt_n{n}", float("nan")) for n in ns], **kw)
        ax_mae.plot(ns, [metrics.get(f"mae_n{n}", float("nan")) for n in ns], **kw)

    for ax in (ax_acc, ax_rt, ax_mae):
        ax.axvspan(n_min - 0.5, small_n_max + 0.5, alpha=0.07, color=PALETTE[4])
        ax.set_xlabel("true N")
        ax.set_xticks(ns)

    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy vs N")
    ax_acc.set_ylim(0, 1.05)
    ax_rt.set_ylabel("RT proxy")
    ax_rt.set_title("RT proxy vs N")
    ax_mae.set_ylabel("|error|")
    ax_mae.set_title("MAE vs N")
    ax_acc.legend(fontsize=8, ncol=min(n_modes, 4))

    if has_maps and cores is not None:
        for c, (mode_name, core) in enumerate(cores.items()):
            if c >= len(map_axes):
                break
            ax = map_axes[c]
            with torch.no_grad():
                mu_t, sigma_t = core.compute_mu_sigma()
                y_lo = (mu_t.min() - 2.0 * sigma_t.max()).item()
                y_hi = (mu_t.max() + 2.0 * sigma_t.max()).item()
                y_g = torch.linspace(y_lo, y_hi, 300, device=mu_t.device)
                out_c = core(y_g)
                x_star_c = out_c["x_star"].cpu().numpy()
                y_g_np = y_g.cpu().numpy()
                mu_np = mu_t.cpu().numpy()
            ax.plot(y_g_np, y_g_np, "--", color="0.7", lw=0.8)
            ax.plot(y_g_np, x_star_c, color=PALETTE[0], lw=1.4)
            for m in mu_np:
                ax.axhline(m, color=PALETTE[2], lw=0.4, ls=":", alpha=0.5)
            ax.set_title(mode_name, fontsize=9)
            ax.set_xlabel("y₀")
            ax.set_ylabel("x*")

    fig.suptitle("Model comparison", fontsize=13, fontweight="bold")
    if not has_maps:
        fig.tight_layout()

    save_figure(
        fig, "comparison_panel",
        save_dir, wandb_run, close=False,
        run_label=run_label, wandb_prefix=wandb_prefix, file_suffix=file_suffix,
    )
    return fig


# ── G. Effective-model recovery plots ──────────────────────────────────────


def plot_effective_recovery(
    per_n_results: dict[int, dict],
    n_min: int = 1,
    n_max: int = 9,
    small_n_max: int = 4,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    wandb_prefix: Optional[str] = None,
) -> plt.Figure:
    """Plot per-N metrics for attractor, accumulator-eff, and ANS-eff models.

    Parameters
    ----------
    per_n_results : {N: {attractor_acc, attractor_mae, acc_nll, ans_nll}}
    """
    ns = sorted(per_n_results.keys())
    attr_acc = [per_n_results[n]["attractor_acc"] for n in ns]
    attr_mae = [per_n_results[n]["attractor_mae"] for n in ns]
    acc_nll = [per_n_results[n]["acc_nll"] for n in ns]
    ans_nll = [per_n_results[n]["ans_nll"] for n in ns]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    ax1.plot(ns, attr_acc, "o-", color=PALETTE[0], lw=1.5, label="attractor")
    ax1.axvspan(n_min - 0.5, small_n_max + 0.5, alpha=0.07, color=PALETTE[4])
    ax1.set_ylabel("accuracy")
    ax1.set_title("Attractor accuracy vs N")
    ax1.set_xlabel("true N")
    ax1.set_xticks(ns)
    ax1.legend(fontsize=8)

    ax2.plot(ns, attr_mae, "o-", color=PALETTE[0], lw=1.5, label="attractor")
    ax2.axvspan(n_min - 0.5, small_n_max + 0.5, alpha=0.07, color=PALETTE[4])
    ax2.set_ylabel("MAE")
    ax2.set_title("Attractor MAE vs N")
    ax2.set_xlabel("true N")
    ax2.set_xticks(ns)

    ax3.plot(ns, acc_nll, "s-", color=PALETTE[1], lw=1.5, label="accumulator-eff")
    ax3.plot(ns, ans_nll, "^-", color=PALETTE[3], lw=1.5, label="ANS-eff")
    ax3.axvspan(n_min - 0.5, small_n_max + 0.5, alpha=0.07, color=PALETTE[4])
    ax3.set_ylabel("NLL (per sample)")
    ax3.set_title("Effective baseline NLL vs N")
    ax3.set_xlabel("true N")
    ax3.set_xticks(ns)
    ax3.legend(fontsize=8)

    fig.suptitle("Effective-model recovery", fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_figure(
        fig, "effective_recovery",
        save_dir, wandb_run, close=False,
        wandb_prefix=wandb_prefix,
    )
    return fig


# ── H. Confusion matrix ───────────────────────────────────────────────────


def plot_confusion_matrix(
    n_true: "np.ndarray",
    n_hat: "np.ndarray",
    n_min: int = 1,
    n_max: int = 9,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    wandb_prefix: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix for Task A (absolute estimation)."""
    ns = list(range(n_min, n_max + 1))
    K = len(ns)
    mat = np.zeros((K, K), dtype=float)
    for i, n_t in enumerate(ns):
        mask = n_true == n_t
        total = mask.sum()
        if total == 0:
            continue
        for j, n_h in enumerate(ns):
            mat[i, j] = ((n_true == n_t) & (n_hat == n_h)).sum() / total

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(K))
    ax.set_xticklabels(ns)
    ax.set_yticks(range(K))
    ax.set_yticklabels(ns)
    ax.set_xlabel("predicted n̂")
    ax.set_ylabel("true N")
    ax.set_title("Confusion matrix  (Task A)")
    fig.colorbar(im, ax=ax, label="P(n̂ | N)")
    fig.tight_layout()

    save_figure(
        fig, "confusion_matrix",
        save_dir, wandb_run, close=False,
        wandb_prefix=wandb_prefix,
    )
    return fig


# ── I. Same/different heatmap ──────────────────────────────────────────────


def plot_same_different_heatmap(
    n_true: "np.ndarray",
    target: "np.ndarray",
    pred_same: "np.ndarray",
    n_min: int = 1,
    n_max: int = 9,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    wandb_prefix: Optional[str] = None,
) -> plt.Figure:
    """Plot accuracy heatmap over (N, target) for Task C."""
    ns = list(range(n_min, n_max + 1))
    K = len(ns)
    acc_mat = np.full((K, K), float("nan"))
    label = (n_true == target).astype(int)

    for i, n_t in enumerate(ns):
        for j, tgt in enumerate(ns):
            mask = (n_true == n_t) & (target == tgt)
            if mask.sum() == 0:
                continue
            correct = (pred_same[mask] == label[mask]).mean()
            acc_mat[i, j] = correct

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(acc_mat, cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(K))
    ax.set_xticklabels(ns)
    ax.set_yticks(range(K))
    ax.set_yticklabels(ns)
    ax.set_xlabel("target m")
    ax.set_ylabel("true N")
    ax.set_title("Same/different accuracy  (Task C)")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()

    save_figure(
        fig, "same_different_heatmap",
        save_dir, wandb_run, close=False,
        wandb_prefix=wandb_prefix,
    )
    return fig


# ── helper: evaluate multiple modes ────────────────────────────────────────


@torch.no_grad()
def evaluate_comparison(
    modes: dict[str, tuple],
    n_min: int = 1,
    n_max: int = 9,
    n_samples: int = 5_000,
    seed: int = 7,
    small_n_max: int = 4,
    device: Optional[torch.device] = None,
    channel=None,
) -> dict[str, dict[str, float]]:
    """Evaluate a set of (model, rt_proxy) pairs on synthetic Task-A data.

    Parameters
    ----------
    modes    : {mode_name: (model, rt_proxy)}
    n_min, n_max, n_samples, seed, small_n_max : evaluation settings
    device   : computation device
    channel  : NeutralEvidenceChannel; uses default if None

    Returns
    -------
    {mode_name: metrics_dict} suitable for plot_rt_curves and
    plot_comparison_panel.  Each metrics_dict contains acc_n{N}, rt_n{N},
    mae_n{N} for all N, plus aggregate metrics.
    """
    from src.data.synth_benchmarks import generate_task_a
    from src.analysis.phase_scan import compute_metrics

    if device is None:
        device = torch.device("cpu")

    data = generate_task_a(
        n_min=n_min, n_max=n_max, n_samples=n_samples, seed=seed,
        channel=channel,
    )

    results: dict[str, dict[str, float]] = {}
    for mode_name, (model, rt_proxy) in modes.items():
        results[mode_name] = compute_metrics(
            model, rt_proxy, data,
            n_min, n_max, small_n_max, device,
        )
    return results
