"""H5/H6 — Posterior predictive checks and model comparison.

Observed-vs-predicted summaries, calibration curves, and
cross-family comparison bar charts.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import PALETTE, save_figure


# =====================================================================
# H5 — posterior predictive checks
# =====================================================================

def plot_obs_vs_pred_accuracy(
    df: pd.DataFrame,
    pred_probs: np.ndarray,
    group_col: str,
    task_name: str,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Observed vs predicted accuracy, binned by *group_col*.

    Parameters
    ----------
    df         : trial-level DataFrame (must contain 'correct' and *group_col*)
    pred_probs : predicted probabilities aligned with *df*
    group_col  : column to group by (e.g. 'true_n', 'distance', …)
    task_name  : used in titles and filenames
    """
    df = df.copy()
    df["pred_correct"] = (pred_probs >= 0.5).astype(float)
    obs = df.groupby(group_col)["correct"].mean()
    pred = df.groupby(group_col)["pred_correct"].mean()

    fig, ax = plt.subplots()
    ax.plot(obs.index, obs.values, "o-", color=PALETTE[0], label="observed")
    ax.plot(pred.index, pred.values, "s--", color=PALETTE[1], label="predicted")
    ax.set_xlabel(group_col)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{task_name} — obs vs pred accuracy by {group_col}")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"ppc_{task_name}_acc_{group_col}", save_dir, wandb_run,
                close=False)
    return fig


def plot_obs_vs_pred_rt(
    df: pd.DataFrame,
    pred_rt: Optional[np.ndarray],
    task_name: str,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    rt_col: str = "rt_ms",
) -> plt.Figure:
    """Overlaid histograms of observed and predicted RT.

    *pred_rt* may be None when no RT model is fitted.
    """
    valid = df[df[rt_col] > 0]
    fig, ax = plt.subplots()
    ax.hist(valid[rt_col], bins=60, density=True, alpha=0.6,
            color=PALETTE[0], label="observed")

    if pred_rt is not None:
        pred_valid = pred_rt[df[rt_col].values > 0]
        ax.hist(pred_valid, bins=60, density=True, alpha=0.5,
                color=PALETTE[1], label="predicted")

    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("density")
    ax.set_title(f"{task_name} — observed vs predicted RT")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"ppc_{task_name}_rt", save_dir, wandb_run, close=False)
    return fig


def plot_obs_vs_pred_same_heatmap(
    df: pd.DataFrame,
    pred_probs: np.ndarray,
    row_col: str,
    col_col: str,
    task_name: str,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Side-by-side heatmaps: observed vs predicted P("same").

    Useful for dot verification (row=true_n, col=shown_digit) or
    match-to-sample (row=sample_n, col=target_n).
    """
    df = df.copy()
    df["pred_same"] = pred_probs

    response_col = "response_same" if "response_same" in df.columns else "label"

    obs_pivot = df.pivot_table(index=row_col, columns=col_col,
                               values=response_col, aggfunc="mean")
    pred_pivot = df.pivot_table(index=row_col, columns=col_col,
                                values="pred_same", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pivot, label in [
        (axes[0], obs_pivot, "Observed"),
        (axes[1], pred_pivot, "Predicted"),
    ]:
        im = ax.imshow(
            pivot.values, aspect="auto", cmap="YlOrRd",
            vmin=0, vmax=1, origin="lower",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int), fontsize=7)
        ax.set_xlabel(col_col)
        ax.set_ylabel(row_col)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(f"{task_name} — obs vs pred P(\"same\")", y=1.03)
    fig.tight_layout()
    save_figure(fig, f"ppc_{task_name}_same_heatmap", save_dir, wandb_run,
                close=False)
    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    task_name: str,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """2×2 confusion matrix for binary predictions."""
    from collections import Counter

    pairs = list(zip(labels.astype(int), preds.astype(int)))
    counts = Counter(pairs)
    mat = np.array([
        [counts.get((0, 0), 0), counts.get((0, 1), 0)],
        [counts.get((1, 0), 0), counts.get((1, 1), 0)],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"{task_name} — confusion matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_figure(fig, f"ppc_{task_name}_confusion", save_dir, wandb_run, close=False)
    return fig


def plot_calibration_curve(
    pred_probs: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    n_bins: int = 10,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Reliability diagram (calibration curve)."""
    edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (pred_probs >= lo) & (pred_probs <= hi)
        else:
            mask = (pred_probs >= lo) & (pred_probs < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = labels[mask].mean()
            bin_confs[i] = pred_probs[mask].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    valid = bin_counts > 0
    ax.plot(bin_confs[valid], bin_accs[valid], "o-", color=PALETTE[0],
            label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title(f"{task_name} — reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    ax2 = axes[1]
    ax2.bar(range(n_bins), bin_counts, color=PALETTE[2], edgecolor="white")
    ax2.set_xlabel("probability bin")
    ax2.set_ylabel("count")
    ax2.set_title("Prediction histogram")

    fig.tight_layout()
    save_figure(fig, f"ppc_{task_name}_calibration", save_dir, wandb_run,
                close=False)
    return fig


# =====================================================================
# H6 — model comparison
# =====================================================================

def plot_model_comparison_nll(
    results: dict[str, float],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Bar chart: held-out NLL by model family.

    Parameters
    ----------
    results : {family_name: nll_value}
    """
    families = sorted(results.keys())
    vals = [results[f] for f in families]

    fig, ax = plt.subplots()
    bars = ax.bar(families, vals, color=PALETTE[: len(families)], edgecolor="white")
    ax.set_ylabel("held-out NLL")
    ax.set_title("Model comparison — NLL")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "compare_nll", save_dir, wandb_run, close=False)
    return fig


def plot_model_comparison_accuracy(
    results: dict[str, dict[str, float]],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Grouped bar chart: per-task accuracy by model family.

    Parameters
    ----------
    results : {family: {task: accuracy}}
    """
    families = sorted(results.keys())
    tasks = sorted({t for r in results.values() for t in r})
    n_tasks = len(tasks)
    x = np.arange(n_tasks)
    width = 0.8 / max(len(families), 1)

    fig, ax = plt.subplots(figsize=(max(7, n_tasks * 2), 5))
    for i, fam in enumerate(families):
        vals = [results[fam].get(t, 0.0) for t in tasks]
        ax.bar(x + i * width, vals, width=width, label=fam,
               color=PALETTE[i % len(PALETTE)], edgecolor="white")

    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison — accuracy by task")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "compare_accuracy", save_dir, wandb_run, close=False)
    return fig


def plot_model_comparison_transfer(
    results: dict[str, dict[str, float]],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Transfer-performance comparison across model families.

    Same structure as ``plot_model_comparison_accuracy`` but titled for
    transfer tasks.
    """
    fig = plot_model_comparison_accuracy(results, save_dir=None, wandb_run=None)
    fig.axes[0].set_title("Transfer performance by model family")
    save_figure(fig, "compare_transfer", save_dir, wandb_run, close=False)
    return fig
