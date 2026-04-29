"""H4 — Task-level fit plots.

Behavioural psychometric curves, heatmaps, and RT plots for each task,
overlaying observed data with (optional) model predictions.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import PALETTE, save_figure


# =====================================================================
# Symbolic comparison
# =====================================================================

def plot_symbolic_accuracy_by_distance(
    df: pd.DataFrame,
    pred_probs: Optional[np.ndarray] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Accuracy (and optional model prediction) vs numerical distance.

    Parameters
    ----------
    df         : trial-level DataFrame with left_digit, right_digit, correct
    pred_probs : optional array of predicted P(right) aligned with *df*
    """
    df = df.copy()
    df["distance"] = (df["right_digit"] - df["left_digit"]).abs()

    obs = df.groupby("distance")["correct"].mean()
    fig, ax = plt.subplots()
    ax.plot(obs.index, obs.values, "o-", color=PALETTE[0], label="observed")

    if pred_probs is not None:
        df["pred_correct"] = (pred_probs >= 0.5).astype(float)
        pred = df.groupby("distance")["pred_correct"].mean()
        ax.plot(pred.index, pred.values, "s--", color=PALETTE[1], label="predicted")

    ax.set_xlabel("numerical distance |d_R − d_L|")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Symbolic comparison — accuracy by distance")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "task_symbolic_acc_distance", save_dir, wandb_run, close=False)
    return fig


def plot_symbolic_rt_by_distance(
    df: pd.DataFrame,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    rt_col: str = "rt_ms",
) -> plt.Figure:
    """Median RT vs numerical distance for symbolic comparison."""
    df = df.copy()
    df["distance"] = (df["right_digit"] - df["left_digit"]).abs()
    valid = df[df[rt_col] > 0]
    if valid.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No RT data", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(fig, "task_symbolic_rt_distance", save_dir, wandb_run, close=False)
        return fig

    stats = valid.groupby("distance")[rt_col].agg(["median", "mean", "std"])
    fig, ax = plt.subplots()
    ax.errorbar(stats.index, stats["median"], yerr=stats["std"] / 2,
                fmt="o-", color=PALETTE[0], capsize=3)
    ax.set_xlabel("numerical distance")
    ax.set_ylabel("median RT (ms)")
    ax.set_title("Symbolic comparison — RT by distance")
    fig.tight_layout()
    save_figure(fig, "task_symbolic_rt_distance", save_dir, wandb_run, close=False)
    return fig


# =====================================================================
# Match-to-sample
# =====================================================================

def plot_mts_psychometric(
    df: pd.DataFrame,
    pred_probs: Optional[np.ndarray] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """P("same") as a function of |sample_n − target_n|."""
    df = df.copy()
    df["num_diff"] = (df["sample_true_n"] - df["target_true_n"]).abs()

    obs = df.groupby("num_diff")["response_same"].mean()
    fig, ax = plt.subplots()
    ax.plot(obs.index, obs.values, "o-", color=PALETTE[0], label="observed")

    if pred_probs is not None:
        df["pred_same"] = pred_probs
        pred = df.groupby("num_diff")["pred_same"].mean()
        ax.plot(pred.index, pred.values, "s--", color=PALETTE[1], label="predicted")

    ax.set_xlabel("|sample N − target N|")
    ax.set_ylabel('P("same")')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Match-to-sample — psychometric curve")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "task_mts_psychometric", save_dir, wandb_run, close=False)
    return fig


# =====================================================================
# Dot verification
# =====================================================================

def plot_dot_verification_heatmap(
    df: pd.DataFrame,
    pred_probs: Optional[np.ndarray] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Heatmap of P("same") by (true N, shown digit).

    Left panel: observed.  Right panel (optional): predicted.
    """
    n_panels = 1 if pred_probs is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                             squeeze=False)
    df = df.copy()

    obs_pivot = df.pivot_table(
        index="true_n", columns="shown_digit",
        values="response_same", aggfunc="mean",
    )
    _draw_heatmap(axes[0, 0], obs_pivot, "Observed P(\"same\")")

    if pred_probs is not None:
        df["pred_same"] = pred_probs
        pred_pivot = df.pivot_table(
            index="true_n", columns="shown_digit",
            values="pred_same", aggfunc="mean",
        )
        _draw_heatmap(axes[0, 1], pred_pivot, "Predicted P(\"same\")")

    fig.suptitle("Dot verification — same-probability heatmap", y=1.02)
    fig.tight_layout()
    save_figure(fig, "task_dotver_heatmap", save_dir, wandb_run, close=False)
    return fig


def _draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str) -> None:
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="YlOrRd",
        vmin=0, vmax=1, origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=8)
    ax.set_xlabel("shown digit")
    ax.set_ylabel("true N")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_dot_verification_rt(
    df: pd.DataFrame,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    rt_col: str = "rt_ms",
) -> plt.Figure:
    """RT by true N, split by response type (same / different)."""
    valid = df[df[rt_col] > 0].copy()
    if valid.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No RT data", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(fig, "task_dotver_rt", save_dir, wandb_run, close=False)
        return fig

    valid["resp_label"] = valid["response_same"].map({1: "same", 0: "different"})
    fig, ax = plt.subplots()
    for i, (resp, grp) in enumerate(valid.groupby("resp_label")):
        means = grp.groupby("true_n")[rt_col].median()
        ax.plot(means.index, means.values, "o-", color=PALETTE[i], label=resp)

    ax.set_xlabel("true N")
    ax.set_ylabel("median RT (ms)")
    ax.set_title("Dot verification — RT by true N")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "task_dotver_rt", save_dir, wandb_run, close=False)
    return fig


def plot_counting_route_probability(
    df: pd.DataFrame,
    pi_count: np.ndarray,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Estimated counting-route probability by N and (optionally) grade.

    Parameters
    ----------
    pi_count : array of estimated pi_count aligned with *df*
    """
    df = df.copy()
    df["pi_count"] = pi_count

    fig, ax = plt.subplots()
    if "grade" in df.columns and df["grade"].nunique() > 1:
        for i, (g, grp) in enumerate(df.groupby("grade")):
            means = grp.groupby("true_n")["pi_count"].mean()
            ax.plot(means.index, means.values, "o-", color=PALETTE[i],
                    label=f"grade {g}")
    else:
        means = df.groupby("true_n")["pi_count"].mean()
        ax.plot(means.index, means.values, "o-", color=PALETTE[0])

    ax.set_xlabel("true N")
    ax.set_ylabel("P(counting route)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Estimated counting-route probability")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "task_dotver_counting_route", save_dir, wandb_run, close=False)
    return fig


# =====================================================================
# Panamath
# =====================================================================

def plot_panamath_psychometric_ratio(
    df: pd.DataFrame,
    pred_probs: Optional[np.ndarray] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Accuracy as a function of the larger/smaller ratio."""
    df = df.copy()
    bigger = df[["left_true_n", "right_true_n"]].max(axis=1)
    smaller = df[["left_true_n", "right_true_n"]].min(axis=1).clip(lower=1)
    df["ratio"] = bigger / smaller
    df["ratio_bin"] = pd.cut(df["ratio"], bins=8)

    obs = df.groupby("ratio_bin", observed=True)["correct"].mean()
    x_pos = np.arange(len(obs))

    fig, ax = plt.subplots()
    ax.plot(x_pos, obs.values, "o-", color=PALETTE[0], label="observed")

    if pred_probs is not None:
        df["pred_correct"] = (pred_probs >= 0.5).astype(float)
        pred = df.groupby("ratio_bin", observed=True)["pred_correct"].mean()
        ax.plot(x_pos, pred.values, "s--", color=PALETTE[1], label="predicted")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in obs.index], rotation=30, fontsize=7)
    ax.set_xlabel("ratio (larger / smaller)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Panamath — psychometric by ratio")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "task_panamath_ratio", save_dir, wandb_run, close=False)
    return fig


def plot_panamath_congruency(
    df: pd.DataFrame,
    pred_probs: Optional[np.ndarray] = None,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Accuracy split by congruency condition."""
    if "congruency" not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No congruency column", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(fig, "task_panamath_congruency", save_dir, wandb_run, close=False)
        return fig

    df = df.copy()
    obs = df.groupby("congruency")["correct"].mean()

    fig, ax = plt.subplots()
    x = np.arange(len(obs))
    ax.bar(x - 0.15, obs.values, width=0.3, color=PALETTE[0], label="observed")

    if pred_probs is not None:
        df["pred_correct"] = (pred_probs >= 0.5).astype(float)
        pred = df.groupby("congruency")["pred_correct"].mean()
        ax.bar(x + 0.15, pred.values, width=0.3, color=PALETTE[1], label="predicted")

    ax.set_xticks(x)
    ax.set_xticklabels(obs.index)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Panamath — accuracy by congruency")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "task_panamath_congruency", save_dir, wandb_run, close=False)
    return fig
