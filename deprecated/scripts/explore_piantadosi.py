"""Exploratory plots for the Piantadosi numerosity perception dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

OUTPUT_DIR = Path("output/explore_piantadosi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv("data/data_piantadosi.csv")

# Collapse to trial level (one row per trial per participant)
trial = (
    df_raw.groupby(["uniqueid", "cond", "pid", "trial_id"])
    .agg(
        exp_dur=("exp_dur", "first"),
        opacity=("opacity", "first"),
        n_shown=("n_shown", "first"),
        n_guess=("n_guess", "first"),
        rt=("rt", "first"),
        control=("control", "first"),
    )
    .reset_index()
)

trial["error"] = trial["n_guess"] - trial["n_shown"]
trial["ratio"] = trial["n_guess"] / trial["n_shown"]
trial["log_n_shown"] = np.log2(trial["n_shown"])
trial["log_n_guess"] = np.log2(trial["n_guess"])

COND_COLORS = {"DURATION": "#2171b5", "CONTRAST": "#cb181d"}
CTRL_COLORS = {"NONE": "#525252", "DENSITY": "#31a354", "AREA": "#756bb1"}

# ── 1. n_guess vs n_shown heatmap ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, cond in zip(axes, ["DURATION", "CONTRAST"]):
    sub = trial[trial["cond"] == cond]
    h, xedges, yedges = np.histogram2d(
        sub["n_shown"], sub["n_guess"],
        bins=[np.arange(0.5, 16.5, 1), np.arange(0.5, 40.5, 1)],
    )
    ax.pcolormesh(xedges, yedges, h.T, cmap="YlOrRd", norm=LogNorm(vmin=1))
    ax.plot([1, 15], [1, 15], "k--", lw=1.5, alpha=0.7, label="perfect")
    ax.set_xlabel("n shown")
    ax.set_ylabel("n guessed")
    ax.set_title(f"Condition: {cond}")
    ax.legend()
fig.suptitle("Joint distribution of shown vs guessed numerosity", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_heatmap_shown_vs_guess.png", dpi=150)
plt.close(fig)

# ── 2. Mean guess ± SD vs n_shown, by condition ───────────────────────────────
agg = (
    trial.groupby(["cond", "n_shown"])["n_guess"]
    .agg(["mean", "std", "median"])
    .reset_index()
)
fig, ax = plt.subplots(figsize=(8, 6))
for cond, g in agg.groupby("cond"):
    ax.errorbar(
        g["n_shown"], g["mean"], yerr=g["std"],
        fmt="o-", capsize=3, label=cond, color=COND_COLORS[cond],
    )
ax.plot([1, 15], [1, 15], "k--", lw=1, alpha=0.5, label="perfect")
ax.set_xlabel("n shown")
ax.set_ylabel("n guessed (mean ± SD)")
ax.set_title("Estimation accuracy by condition")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_mean_guess_by_condition.png", dpi=150)
plt.close(fig)

# ── 3. Effect of exposure duration (DURATION condition) ────────────────────────
dur_sub = trial[trial["cond"] == "DURATION"]
agg_dur = (
    dur_sub.groupby(["exp_dur", "n_shown"])["n_guess"]
    .agg(["mean", "std"])
    .reset_index()
)
durations = sorted(agg_dur["exp_dur"].unique())
cmap_dur = plt.cm.viridis(np.linspace(0.15, 0.95, len(durations)))

fig, ax = plt.subplots(figsize=(9, 6))
for color, dur in zip(cmap_dur, durations):
    g = agg_dur[agg_dur["exp_dur"] == dur]
    ax.plot(g["n_shown"], g["mean"], "o-", color=color, label=f"{dur} ms", markersize=4)
ax.plot([1, 15], [1, 15], "k--", lw=1, alpha=0.4)
ax.set_xlabel("n shown")
ax.set_ylabel("mean n guessed")
ax.set_title("Effect of exposure duration on estimation")
ax.legend(title="Exposure", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_effect_duration.png", dpi=150)
plt.close(fig)

# ── 4. Effect of opacity (CONTRAST condition) ─────────────────────────────────
con_sub = trial[trial["cond"] == "CONTRAST"]
agg_con = (
    con_sub.groupby(["opacity", "n_shown"])["n_guess"]
    .agg(["mean", "std"])
    .reset_index()
)
opacities = sorted(agg_con["opacity"].unique())
cmap_op = plt.cm.plasma(np.linspace(0.15, 0.85, len(opacities)))

fig, ax = plt.subplots(figsize=(9, 6))
for color, op in zip(cmap_op, opacities):
    g = agg_con[agg_con["opacity"] == op]
    ax.plot(g["n_shown"], g["mean"], "o-", color=color, label=f"{op:.2f}", markersize=4)
ax.plot([1, 15], [1, 15], "k--", lw=1, alpha=0.4)
ax.set_xlabel("n shown")
ax.set_ylabel("mean n guessed")
ax.set_title("Effect of contrast (opacity) on estimation")
ax.legend(title="Opacity", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_effect_opacity.png", dpi=150)
plt.close(fig)

# ── 5. Coefficient of variation (Weber-like) by n_shown & condition ────────────
cv = (
    trial.groupby(["cond", "n_shown"])["n_guess"]
    .agg(["mean", "std"])
    .reset_index()
)
cv["cv"] = cv["std"] / cv["mean"]

fig, ax = plt.subplots(figsize=(8, 5))
for cond, g in cv.groupby("cond"):
    ax.plot(g["n_shown"], g["cv"], "o-", color=COND_COLORS[cond], label=cond)
ax.set_xlabel("n shown")
ax.set_ylabel("CV (std / mean)")
ax.set_title("Coefficient of variation by numerosity")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_cv_by_numerosity.png", dpi=150)
plt.close(fig)

# ── 6. Response time distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, cond in zip(axes, ["DURATION", "CONTRAST"]):
    sub = trial[trial["cond"] == cond]
    ax.hist(sub["rt"] / 1000, bins=80, range=(0, 20), alpha=0.8,
            color=COND_COLORS[cond], edgecolor="white", linewidth=0.3)
    med = sub["rt"].median() / 1000
    ax.axvline(med, color="black", ls="--", lw=1.2, label=f"median = {med:.1f}s")
    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{cond}")
    ax.legend()
fig.suptitle("Response time distributions", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_rt_distribution.png", dpi=150)
plt.close(fig)

# ── 7. RT vs n_shown ──────────────────────────────────────────────────────────
rt_agg = (
    trial.groupby(["cond", "n_shown"])["rt"]
    .agg(["mean", "median"])
    .reset_index()
)
fig, ax = plt.subplots(figsize=(8, 5))
for cond, g in rt_agg.groupby("cond"):
    ax.plot(g["n_shown"], g["median"] / 1000, "o-", color=COND_COLORS[cond], label=cond)
ax.set_xlabel("n shown")
ax.set_ylabel("Median RT (s)")
ax.set_title("Response time by numerosity")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "07_rt_vs_nshown.png", dpi=150)
plt.close(fig)

# ── 8. Control condition comparison ───────────────────────────────────────────
agg_ctrl = (
    trial.groupby(["control", "n_shown"])["n_guess"]
    .agg(["mean", "std"])
    .reset_index()
)
fig, ax = plt.subplots(figsize=(8, 6))
for ctrl, g in agg_ctrl.groupby("control"):
    ax.errorbar(
        g["n_shown"] + {"NONE": -0.1, "DENSITY": 0, "AREA": 0.1}[ctrl],
        g["mean"], yerr=g["std"],
        fmt="o-", capsize=2, label=ctrl, color=CTRL_COLORS[ctrl], markersize=4,
    )
ax.plot([1, 15], [1, 15], "k--", lw=1, alpha=0.4)
ax.set_xlabel("n shown")
ax.set_ylabel("n guessed (mean ± SD)")
ax.set_title("Estimation by control condition (AREA / DENSITY / NONE)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "08_control_conditions.png", dpi=150)
plt.close(fig)

# ── 9. Participant-level bias: mean(n_guess - n_shown) ────────────────────────
subj = (
    trial.groupby("uniqueid")
    .agg(
        mean_error=("error", "mean"),
        mean_ratio=("ratio", "mean"),
        n_trials=("trial_id", "count"),
        cond=("cond", "first"),
    )
    .reset_index()
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(subj["mean_error"], bins=50, color="#636363", edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="red", ls="--", lw=1.2)
axes[0].set_xlabel("Mean error (guess − shown)")
axes[0].set_ylabel("Participants")
axes[0].set_title("Participant-level mean bias")

axes[1].hist(subj["mean_ratio"], bins=50, range=(0, 4), color="#636363", edgecolor="white", linewidth=0.3)
axes[1].axvline(1, color="red", ls="--", lw=1.2)
axes[1].set_xlabel("Mean ratio (guess / shown)")
axes[1].set_ylabel("Participants")
axes[1].set_title("Participant-level mean ratio")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "09_participant_bias.png", dpi=150)
plt.close(fig)

# ── 10. Subitizing range: accuracy for small numbers ──────────────────────────
small = trial[trial["n_shown"] <= 6].copy()
small["correct"] = (small["n_guess"] == small["n_shown"]).astype(int)
acc = small.groupby(["cond", "n_shown"])["correct"].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
for cond, g in acc.groupby("cond"):
    ax.plot(g["n_shown"], g["correct"] * 100, "o-", color=COND_COLORS[cond], label=cond)
ax.set_xlabel("n shown")
ax.set_ylabel("% exact correct")
ax.set_title("Exact accuracy for small numerosities (subitizing range)")
ax.set_xticks(range(1, 7))
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "10_subitizing_accuracy.png", dpi=150)
plt.close(fig)

print(f"All plots saved to {OUTPUT_DIR}/")
