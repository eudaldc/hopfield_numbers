"""H1 — Data QC plots.

Subject counts, missingness, trial distributions, RT histograms,
and accuracy summaries.  All functions accept plain DataFrames.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import PALETTE, save_figure


# ── H1.1  subject counts by cohort / grade / task ────────────────────

def plot_subject_counts(
    subjects_df: pd.DataFrame,
    task_dfs: dict[str, pd.DataFrame],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Bar chart: number of unique subjects by cohort × grade, plus per-task."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left panel: cohort × grade from subjects table
    if "grade" in subjects_df.columns and "cohort" in subjects_df.columns:
        ct = (
            subjects_df.groupby(["cohort", "grade"])["subject_id"]
            .nunique()
            .unstack(fill_value=0)
        )
        ct.plot.bar(ax=axes[0], color=PALETTE[: ct.shape[1]])
        axes[0].set_title("Subjects by cohort × grade")
        axes[0].set_ylabel("n subjects")
        axes[0].set_xlabel("")
        axes[0].legend(title="grade")

    # right panel: per-task subject counts
    task_names = sorted(task_dfs.keys())
    counts = [task_dfs[t]["subject_id"].nunique() for t in task_names]
    bars = axes[1].barh(task_names, counts, color=PALETTE[: len(task_names)])
    axes[1].set_title("Subjects per task")
    axes[1].set_xlabel("n subjects")
    for bar, c in zip(bars, counts):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(c), va="center", fontsize=9)

    fig.tight_layout()
    save_figure(fig, "data_subject_counts", save_dir, wandb_run, close=False)
    return fig


# ── H1.2  missingness heatmap ────────────────────────────────────────

def plot_missingness_heatmap(
    subjects_df: pd.DataFrame,
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Heatmap of missing-value fraction per column in the subject table."""
    frac_missing = subjects_df.isnull().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, len(frac_missing) * 0.35)))
    colors = [PALETTE[3] if v > 0.1 else PALETTE[0] for v in frac_missing.values]
    ax.barh(frac_missing.index, frac_missing.values, color=colors)
    ax.set_xlim(0, max(1.0, frac_missing.max() * 1.1))
    ax.set_xlabel("Fraction missing")
    ax.set_title("Subject-table missingness")
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "data_missingness", save_dir, wandb_run, close=False)
    return fig


# ── H1.3  trials per child ──────────────────────────────────────────

def plot_trials_per_child(
    task_dfs: dict[str, pd.DataFrame],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Histogram of trial counts per child, one panel per task."""
    n_tasks = len(task_dfs)
    ncols = min(n_tasks, 3)
    nrows = int(np.ceil(n_tasks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for ax, (task, df) in zip(axes.ravel(), sorted(task_dfs.items())):
        counts = df.groupby("subject_id").size()
        ax.hist(counts, bins=30, color=PALETTE[0], edgecolor="white")
        ax.set_title(task)
        ax.set_xlabel("trials")
        ax.set_ylabel("n children")

    for ax in axes.ravel()[n_tasks:]:
        ax.set_visible(False)

    fig.suptitle("Trials per child by task", y=1.02)
    fig.tight_layout()
    save_figure(fig, "data_trials_per_child", save_dir, wandb_run, close=False)
    return fig


# ── H1.4  RT distributions ──────────────────────────────────────────

def plot_rt_distributions(
    task_dfs: dict[str, pd.DataFrame],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
    rt_col: str = "rt_ms",
    max_rt: float = 10_000.0,
) -> plt.Figure:
    """RT histograms by task (filtered to valid positive RTs)."""
    tasks_with_rt = {
        t: df for t, df in task_dfs.items()
        if rt_col in df.columns and (df[rt_col] > 0).any()
    }
    n = len(tasks_with_rt)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No RT data available", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(fig, "data_rt_distributions", save_dir, wandb_run, close=False)
        return fig

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax, (task, df) in zip(axes.ravel(), sorted(tasks_with_rt.items())):
        rts = df[rt_col]
        rts = rts[(rts > 0) & (rts <= max_rt)]
        ax.hist(rts, bins=60, color=PALETTE[1], edgecolor="white", alpha=0.85)
        ax.axvline(rts.median(), color=PALETTE[3], ls="--", lw=1.2,
                   label=f"median={rts.median():.0f}")
        ax.set_title(task)
        ax.set_xlabel("RT (ms)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle("RT distributions by task", y=1.02)
    fig.tight_layout()
    save_figure(fig, "data_rt_distributions", save_dir, wandb_run, close=False)
    return fig


# ── H1.5  accuracy by task and grade ────────────────────────────────

def plot_accuracy_by_task_grade(
    task_dfs: dict[str, pd.DataFrame],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Grouped bar chart of mean accuracy by task, split by grade."""
    records = []
    for task, df in task_dfs.items():
        if "correct" not in df.columns:
            continue
        grp = df.groupby("grade")["correct"].mean()
        for grade, acc in grp.items():
            records.append({"task": task, "grade": grade, "accuracy": acc})

    if not records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No accuracy data", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(fig, "data_accuracy_grade", save_dir, wandb_run, close=False)
        return fig

    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="task", columns="grade", values="accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot.bar(ax=ax, color=PALETTE[: pivot.shape[1]])
    ax.set_ylabel("mean accuracy")
    ax.set_title("Accuracy by task and grade")
    ax.set_ylim(0, 1.05)
    ax.legend(title="grade")
    ax.set_xlabel("")
    fig.tight_layout()
    save_figure(fig, "data_accuracy_grade", save_dir, wandb_run, close=False)
    return fig


# ── H1.6  exclusion / filtering summary ─────────────────────────────

def plot_filtering_summary(
    raw_counts: dict[str, int],
    clean_counts: dict[str, int],
    save_dir: Optional[str | Path] = None,
    wandb_run=None,
) -> plt.Figure:
    """Horizontal stacked bar: raw vs retained trial counts per task."""
    tasks = sorted(raw_counts.keys())
    raw = [raw_counts[t] for t in tasks]
    clean = [clean_counts.get(t, 0) for t in tasks]
    excluded = [r - c for r, c in zip(raw, clean)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(tasks) * 0.6)))
    y = np.arange(len(tasks))
    ax.barh(y, clean, color=PALETTE[0], label="retained")
    ax.barh(y, excluded, left=clean, color=PALETTE[3], alpha=0.6,
            label="excluded")
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("trials")
    ax.set_title("Filtering summary")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "data_filtering_summary", save_dir, wandb_run, close=False)
    return fig
