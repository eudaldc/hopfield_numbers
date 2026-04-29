"""H7 — Subject-level analysis plots.

Parameter distributions, scatterplot matrices, and correlations with
downstream measures (arithmetic, number line, Raven, working memory).
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import PALETTE, save_figure

_PARAM_COLS = ["temperature", "width_scale", "tau_sym", "count_bias", "kstar"]


def _safe_pearson(a, b) -> float:
    """Pearson r that returns NaN silently when either input is constant."""
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
_DOWNSTREAM_COLS = [
    "arithmetic_score", "number_line_score", "raven_score", "wm_score",
]


# =====================================================================
# Parameter distributions
# =====================================================================

def plot_subject_param_histograms(
    subject_df: pd.DataFrame,
    param_cols: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Histogram of each subject-level learned parameter."""
    if param_cols is None:
        param_cols = [c for c in _PARAM_COLS if c in subject_df.columns]

    n = len(param_cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No parameter columns", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(
            fig,
            "subj_param_histograms",
            save_dir,
            wandb_run,
            close=False,
            run_label=run_label,
            wandb_prefix=wandb_prefix,
            file_suffix=file_suffix,
        )
        return fig

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax, col in zip(axes.ravel(), param_cols):
        vals = subject_df[col].dropna()
        ax.hist(vals, bins=30, color=PALETTE[0], edgecolor="white")
        ax.axvline(vals.median(), color=PALETTE[3], ls="--", lw=1.2,
                   label=f"median={vals.median():.3f}")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("n subjects")
        ax.legend(fontsize=7)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle("Subject parameter distributions", y=1.02)
    fig.tight_layout()
    save_figure(
        fig,
        "subj_param_histograms",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


# =====================================================================
# Scatterplot matrix
# =====================================================================

def plot_subject_param_scattermatrix(
    subject_df: pd.DataFrame,
    param_cols: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Pairs plot of subject parameters (lower triangle: scatter,
    diagonal: histogram, upper triangle: correlation coefficient).
    """
    if param_cols is None:
        param_cols = [c for c in _PARAM_COLS if c in subject_df.columns]

    n = len(param_cols)
    if n < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need ≥ 2 parameters", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(
            fig,
            "subj_param_scattermatrix",
            save_dir,
            wandb_run,
            close=False,
            run_label=run_label,
            wandb_prefix=wandb_prefix,
            file_suffix=file_suffix,
        )
        return fig

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    data = subject_df[param_cols].dropna()

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[param_cols[i]], bins=25, color=PALETTE[0],
                        edgecolor="white")
            elif i > j:
                ax.scatter(data[param_cols[j]], data[param_cols[i]],
                           s=5, alpha=0.4, color=PALETTE[0])
            else:
                r = _safe_pearson(
                    data[param_cols[i]].values, data[param_cols[j]].values,
                )
                label = f"r={r:.2f}" if np.isfinite(r) else "r=—"
                ax.text(0.5, 0.5, label, transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=PALETTE[3] if np.isfinite(r) and abs(r) > 0.3
                        else PALETTE[7])
                ax.set_xticks([])
                ax.set_yticks([])

            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param_cols[j], fontsize=8)
            if j > 0 and i != j:
                ax.set_yticklabels([])
            elif i == j:
                pass
            else:
                ax.set_ylabel(param_cols[i], fontsize=8)

    fig.suptitle("Subject parameter pairs", y=1.01)
    fig.tight_layout()
    save_figure(
        fig,
        "subj_param_scattermatrix",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


# =====================================================================
# Parameter vs downstream measures
# =====================================================================

def plot_param_vs_downstream(
    subject_df: pd.DataFrame,
    param_col: str,
    downstream_col: str,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Scatter + linear-fit of a subject parameter against a downstream
    measure (arithmetic, number line, Raven, or WM).
    """
    data = subject_df[[param_col, downstream_col]].dropna()
    if data.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No overlapping data", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(
            fig,
            f"subj_{param_col}_vs_{downstream_col}",
            save_dir,
            wandb_run,
            close=False,
            run_label=run_label,
            wandb_prefix=wandb_prefix,
            file_suffix=file_suffix,
        )
        return fig

    x, y = data[param_col].values, data[downstream_col].values
    r = _safe_pearson(x, y)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=8, alpha=0.4, color=PALETTE[0])

    if len(x) > 2 and np.isfinite(r):
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(z, xs), color=PALETTE[3], lw=1.5,
                label=f"r={r:.3f}")

    ax.set_xlabel(param_col)
    ax.set_ylabel(downstream_col)
    ax.set_title(f"{param_col} vs {downstream_col}")
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    fig.tight_layout()
    save_figure(
        fig,
        f"subj_{param_col}_vs_{downstream_col}",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


def plot_all_param_vs_downstream(
    subject_df: pd.DataFrame,
    param_cols: Optional[list[str]] = None,
    downstream_cols: Optional[list[str]] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Grid of scatter plots: each param vs each downstream measure.

    Cells show the Pearson correlation.
    """
    if param_cols is None:
        param_cols = [c for c in _PARAM_COLS if c in subject_df.columns]
    if downstream_cols is None:
        downstream_cols = [c for c in _DOWNSTREAM_COLS if c in subject_df.columns]

    nr, nc = len(param_cols), len(downstream_cols)
    if nr == 0 or nc == 0:
        fig, ax = plt.subplots()
        reasons = []
        if nr == 0:
            reasons.append("no parameter columns in dataframe")
        if nc == 0:
            reasons.append(
                "no downstream scores (arithmetic, number line, Raven, WM); "
                "merge subjects.parquet using task subject index"
            )
        ax.text(
            0.5,
            0.5,
            "Cannot plot grid:\n" + "\n".join(reasons),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        if save_dir is None and wandb_run is None:
            return fig
        save_figure(
            fig,
            "subj_param_downstream_grid",
            save_dir,
            wandb_run,
            close=False,
            run_label=run_label,
            wandb_prefix=wandb_prefix,
            file_suffix=file_suffix,
        )
        return fig

    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.5 * nr), squeeze=False)

    for i, pc in enumerate(param_cols):
        for j, dc in enumerate(downstream_cols):
            ax = axes[i, j]
            data = subject_df[[pc, dc]].dropna()
            if data.empty:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            x, y = data[pc].values, data[dc].values
            ax.scatter(x, y, s=6, alpha=0.3, color=PALETTE[0])
            r = _safe_pearson(x, y)
            ax.set_title(
                f"r={r:.2f}" if np.isfinite(r) else "r=—",
                fontsize=9,
            )
            if i == nr - 1:
                ax.set_xlabel(dc, fontsize=8)
            if j == 0:
                ax.set_ylabel(pc, fontsize=8)

    fig.suptitle("Subject parameters vs downstream measures", y=1.02)
    fig.tight_layout()
    save_figure(
        fig,
        "subj_param_downstream_grid",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig


# =====================================================================
# K* vs grade
# =====================================================================

def plot_kstar_vs_grade(
    subject_df: pd.DataFrame,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    run_label: Optional[str] = None,
    wandb_prefix: Optional[str] = None,
    file_suffix: Optional[str] = None,
) -> plt.Figure:
    """Box/strip plot of K* by grade."""
    if "kstar" not in subject_df.columns or "grade" not in subject_df.columns:
        fig, ax = plt.subplots()
        msg = []
        if "kstar" not in subject_df.columns:
            msg.append("kstar missing from export")
        if "grade" not in subject_df.columns:
            msg.append(
                "grade missing (align subject_idx to subject_id via task parquets, "
                "then merge subjects.parquet)"
            )
        ax.text(
            0.5,
            0.5,
            "\n".join(msg),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        save_figure(
            fig,
            "subj_kstar_grade",
            save_dir,
            wandb_run,
            close=False,
            run_label=run_label,
            wandb_prefix=wandb_prefix,
            file_suffix=file_suffix,
        )
        return fig

    data = subject_df[["kstar", "grade"]].dropna()
    grades = sorted(data["grade"].unique())

    fig, ax = plt.subplots()
    positions = list(range(len(grades)))
    bp_data = [data[data["grade"] == g]["kstar"].values for g in grades]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE[4])
        patch.set_alpha(0.5)

    for i, g in enumerate(grades):
        ys = data[data["grade"] == g]["kstar"].values
        xs = np.random.default_rng(73).normal(i, 0.08, len(ys))
        ax.scatter(xs, ys, s=8, alpha=0.35, color=PALETTE[0])

    ax.set_xticks(positions)
    ax.set_xticklabels([str(g) for g in grades])
    ax.set_xlabel("grade")
    ax.set_ylabel("K*")
    ax.set_title("K* by grade")
    fig.tight_layout()
    save_figure(
        fig,
        "subj_kstar_grade",
        save_dir,
        wandb_run,
        close=False,
        run_label=run_label,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )
    return fig
