"""Accuracy and RT by n_shown, colored by exposure duration (DURATION condition only)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output/explore_piantadosi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv("data/data_piantadosi.csv")

trial = (
    df_raw.groupby(["uniqueid", "cond", "pid", "trial_id"])
    .agg(
        exp_dur=("exp_dur", "first"),
        n_shown=("n_shown", "first"),
        n_guess=("n_guess", "first"),
        rt=("rt", "first"),
    )
    .reset_index()
)

dur = trial[trial["cond"] == "DURATION"].copy()
dur["error"] = (dur["n_guess"] - dur["n_shown"]).abs()
dur["ratio"] = dur["n_guess"] / dur["n_shown"]

durations = sorted(dur["exp_dur"].unique())
cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(durations)))

# ── 1. Accuracy (mean absolute error) by n_shown, colored by duration ─────────
agg_acc = (
    dur.groupby(["exp_dur", "n_shown"])
    .agg(
        mean_abs_error=("error", "mean"),
        mean_ratio=("ratio", "mean"),
        pct_exact=("error", lambda x: (x == 0).mean() * 100),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for color, d in zip(cmap, durations):
    g = agg_acc[agg_acc["exp_dur"] == d]
    axes[0].plot(g["n_shown"], g["mean_abs_error"], "o-", color=color,
                 label=f"{d} ms", markersize=5)
    axes[1].plot(g["n_shown"], g["pct_exact"], "o-", color=color,
                 label=f"{d} ms", markersize=5)
    axes[2].plot(g["n_shown"], g["mean_ratio"], "o-", color=color,
                 label=f"{d} ms", markersize=5)

axes[0].set_xlabel("n shown")
axes[0].set_ylabel("Mean absolute error")
axes[0].set_title("Absolute error")
axes[0].legend(title="Exposure", fontsize=8)

axes[1].set_xlabel("n shown")
axes[1].set_ylabel("% exact correct")
axes[1].set_title("Exact accuracy")
axes[1].legend(title="Exposure", fontsize=8)

axes[2].set_xlabel("n shown")
axes[2].set_ylabel("Mean ratio (guess / shown)")
axes[2].axhline(1, color="black", ls="--", lw=1, alpha=0.4)
axes[2].set_title("Estimation ratio (1 = perfect)")
axes[2].legend(title="Exposure", fontsize=8)

fig.suptitle("Accuracy by numerosity and exposure duration", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "11_accuracy_by_duration.png", dpi=150)
plt.close(fig)

# ── 2. RT by n_shown, colored by duration ─────────────────────────────────────
agg_rt = (
    dur.groupby(["exp_dur", "n_shown"])["rt"]
    .agg(["mean", "median"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 6))
for color, d in zip(cmap, durations):
    g = agg_rt[agg_rt["exp_dur"] == d]
    ax.plot(g["n_shown"], g["median"] / 1000, "o-", color=color,
            label=f"{d} ms", markersize=5)

ax.set_xlabel("n shown")
ax.set_ylabel("Median RT (s)")
ax.set_title("Response time by numerosity and exposure duration", fontsize=14)
ax.legend(title="Exposure", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "12_rt_by_duration.png", dpi=150)
plt.close(fig)

print("Done: 11_accuracy_by_duration.png, 12_rt_by_duration.png")
