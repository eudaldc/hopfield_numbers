#!/usr/bin/env python
"""Figure 3 (revision): model comparison, RT channel, and the predicted limit.

Poisson-free throughout. The model comparison is shown at two levels: a joint
fit (one fit shared across all five durations, on the left) and the per-duration
fit (each duration fitted independently, on the right). Heatmap cells are dAIC
relative to the best row of each block, viridis, darker is better; the joint and
per-duration blocks are separate fits, each colour-scaled on its own range.

  a — Accuracy channel, joint fit: one landscape shared across all durations
      (only the sensory channel varies). The attractor wins and beats even the
      fully free per-duration Weber/ANS.
  b — Accuracy channel, separate fits per duration, over the ~53,000 individual
      responses. The clean-up advantage is concentrated at brief exposures; the
      static readouts catch up at 320-640 ms, once the clean-up has largely run
      during the exposure itself.
  c — RT channel, joint fit: one RT mapping shared across durations; the
      dynamics-based hybrid link still beats the static entropy proxy.
  d — RT channel, separate fits per duration on the winning landscape: the
      hybrid convergence-time + entropy link wins at every exposure. Response to
      the "RT is grafted on" point.
  e — The fitted width law sigma_n per exposure duration, the sigma_n = 1
      resolvability line, and the emerging n_crit — the model predicts its own
      limit, which grows with exposure.

    .venv/bin/python figures/fig3_comparison_ncrit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.plot_style import apply_style, panel_label  # noqa: E402

OUT_DIR = ROOT / "output" / "trial_level_model_comparison"
OUT_STEM = ROOT / "figures" / "fig3_comparison_ncrit"

FONT_SCALE = 2.1
apply_style(FONT_SCALE)

CMAP = mpl.colormaps["viridis"]
DUR_KEYS = [40, 80, 160, 320, 640]
DUR_LABELS = ["40 ms", "80", "160", "320", "640"]

# ── Response channel: per-duration dAIC (each duration fitted on its own) and
#    the joint fit (one landscape shared across all five durations). They are
#    separate fits, each drawn in its own panel and colour-scaled on its own
#    range, so the per-duration structure is not washed out. ────────────────
resp = pd.read_csv(OUT_DIR / "response_per_duration_aic.csv")
piv_a = resp.pivot_table(index="model_key", columns="duration_ms", values="resp_AIC")
piv_a = piv_a - piv_a.min(axis=0)                       # dAIC vs best in each column
ladder = pd.read_csv(OUT_DIR / "shared_landscape_ladder.csv").set_index("model_key")

ROWS_A = [
    ("Frequency wells + clean-up", "gaussian_freebeta_attractor"),
    ("Frequency wells, static", "gaussian_benford_free_beta"),
    ("Power-law wells + clean-up", "gaussian_pow_attractor"),
    ("Power-law wells, static", "gaussian_pow_readout"),
    ("Weber/ANS", "weber_ans"),
    ("Normative (log-prior)", "cp_log_prior_proxy"),
]
dur_a = [[float(piv_a.loc[k, d]) for d in DUR_KEYS] for _, k in ROWS_A]
joint_a = [[float(ladder.loc[k, "joint_dAIC"])] for _, k in ROWS_A]

# ── RT channel: per-duration dAIC on the winning landscape, and the joint fit
#    (one RT mapping shared across all five durations). ─────────────────────
rt = pd.read_csv(OUT_DIR / "rt_exploration.csv")
rt = rt[rt.model_key == "gaussian_freebeta_attractor"]
piv_b = rt.pivot_table(index="metric", columns="duration_ms", values="AIC")
piv_b = piv_b.loc[["hybrid", "steps", "entropy"]]
piv_b = piv_b - piv_b.min(axis=0)
rtj = pd.read_csv(OUT_DIR / "rt_joint.csv").set_index("metric")

ROWS_B = [
    ("Convergence time + entropy", "hybrid"),
    ("Convergence time", "steps"),
    ("Readout entropy", "entropy"),
]
dur_b = [[float(piv_b.loc[k, d]) for d in DUR_KEYS] for _, k in ROWS_B]
joint_b = [[float(rtj.loc[k, "dAIC"])] for _, k in ROWS_B]


# ── Heatmap drawing helper ─────────────────────────────────────────────
def draw_heat(ax, rows_vals, col_labels, row_labels, framed=False):
    """dAIC heatmap into `ax` (viridis, sqrt scale, dark = best), one value per
    cell. Colour range is local to the axes. Empty strings in `col_labels` draw
    no top header; `row_labels=None` hides the y labels (shared with the panel
    on the left); `framed` boxes the block to mark it as the joint fit. The
    training label goes below the block (added by the caller)."""
    nrow, ncol = len(rows_vals), len(col_labels)
    vmax = max(v for row in rows_vals for v in row) or 1.0
    for i, row in enumerate(rows_vals):
        y0 = nrow - 1 - i
        for j, v in enumerate(row):
            frac = float(np.sqrt(v / vmax))
            ax.add_patch(Rectangle((j, y0), 1.0, 1.0, fc=CMAP(frac),
                                   ec="white", lw=3, zorder=2))
            ax.text(j + 0.5, y0 + 0.5, f"{v:.0f}", ha="center", va="center",
                    fontsize=9.5 * FONT_SCALE,
                    color="white" if frac < 0.55 else "0.15", zorder=3)
    for j, lab in enumerate(col_labels):
        if lab:
            ax.text(j + 0.5, nrow + 0.12, lab, ha="center", va="bottom",
                    fontsize=8.5 * FONT_SCALE, color="0.15")
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_xticks([])
    if row_labels is None:
        ax.set_yticks([])
    else:
        ax.set_yticks([nrow - 0.5 - i for i in range(nrow)])
        ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if framed:
        ax.add_patch(Rectangle((0.03, 0.0), ncol - 0.06, nrow, fill=False,
                               ec="0.4", lw=2.2, zorder=4))


# ── Panel c data: width law per duration ──────────────────────────────
theta = pd.read_csv(OUT_DIR / "winner_theta.csv").set_index("duration_ms")
DURATIONS = list(theta.index)


def sigma_of_n(d: int, n: np.ndarray) -> np.ndarray:
    f = np.log10(1.0 + 1.0 / np.asarray(n, float))
    return (float(theta.loc[d, "sigma_c"]) / f) ** float(theta.loc[d, "beta"])


def ncrit(d: int) -> float:
    ns = np.linspace(1.0, 40.0, 40000)
    s = sigma_of_n(d, ns)
    i = int(np.argmax(s >= 1.0))
    return float(ns[i - 1] + (1.0 - s[i - 1]) * (ns[i] - ns[i - 1]) / (s[i] - s[i - 1]))


# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 17.5))

CELL_W, CELL_H = 0.082, 0.046
LAB_LEFT = 0.315                      # joint panel left edge (row labels to its left)
JW = CELL_W                          # joint panel = 1 column
GAP = 0.35 * CELL_W
DUR_LEFT = LAB_LEFT + JW + GAP        # per-duration panel left edge
DW = 5 * CELL_W                      # per-duration panel = 5 columns
ROW1_TOP, ROW2_TOP = 0.955, 0.585    # top edges of the response and RT blocks

ax_aj = fig.add_axes((LAB_LEFT, ROW1_TOP - 6 * CELL_H, JW, 6 * CELL_H))
ax_ad = fig.add_axes((DUR_LEFT, ROW1_TOP - 6 * CELL_H, DW, 6 * CELL_H))
ax_cj = fig.add_axes((LAB_LEFT, ROW2_TOP - 3 * CELL_H, JW, 3 * CELL_H))
ax_cd = fig.add_axes((DUR_LEFT, ROW2_TOP - 3 * CELL_H, DW, 3 * CELL_H))
ax_e = fig.add_axes((0.10, 0.035, 0.87, 0.315))

# training labels sit below each block; the channel dAIC label sits lower still,
# centred between the joint and per-duration panels
XC_JOINT = LAB_LEFT + JW / 2
XC_DUR = DUR_LEFT + DW / 2
XC_BOTH = (LAB_LEFT + DUR_LEFT + DW) / 2


def channel_labels(y_block_bottom, joint_txt, dur_txt, daic_txt):
    fig.text(XC_JOINT, y_block_bottom - 0.010, joint_txt, ha="center", va="top",
             fontsize=9 * FONT_SCALE, color="0.15")
    fig.text(XC_DUR, y_block_bottom - 0.010, dur_txt, ha="center", va="top",
             fontsize=9 * FONT_SCALE, color="0.15")
    fig.text(XC_BOTH, y_block_bottom - 0.030, daic_txt, ha="center", va="top",
             fontsize=11 * FONT_SCALE, color="0.0")


# response channel: a = joint (one shared landscape), b = per-duration
draw_heat(ax_aj, joint_a, [""], [r for r, _ in ROWS_A], framed=True)
draw_heat(ax_ad, dur_a, DUR_LABELS, None)
channel_labels(ROW1_TOP - 6 * CELL_H, "Joint fit", "Separate fits",
               r"$\Delta$AIC (accuracy)")
panel_label(ax_aj, "a", x=-2.4, pad=24)
panel_label(ax_ad, "b", x=-0.04, pad=24)

# RT channel: c = joint (one shared mapping), d = per-duration
draw_heat(ax_cj, joint_b, [""], [r for r, _ in ROWS_B], framed=True)
draw_heat(ax_cd, dur_b, DUR_LABELS, None)
channel_labels(ROW2_TOP - 3 * CELL_H, "Joint fit", "Separate fits",
               r"$\Delta$AIC (response time)")
panel_label(ax_cj, "c", x=-2.4, pad=24)
panel_label(ax_cd, "d", x=-0.04, pad=24)

# Panel e — width law and n_crit
n_cont = np.linspace(1.0, 15.55, 800)
dur_colors = {d: CMAP(i / (len(DURATIONS) - 1)) for i, d in enumerate(DURATIONS)}
ax_e.axhspan(1.0, 40.0, color="0.55", alpha=0.10, zorder=0)
ax_e.axhline(1.0, color="0.25", lw=1.6, ls="--", zorder=2)
for d in DURATIONS:
    ax_e.plot(n_cont, sigma_of_n(d, n_cont), lw=2.4, color=dur_colors[d], zorder=3,
              label=f"{d} ms")
    ax_e.plot(ncrit(d), 1.0, "o", ms=12, color=dur_colors[d], zorder=5,
              markeredgecolor="white", markeredgewidth=1.5)

ax_e.text(1.45, 3.6, "Approximate regime",
          fontsize=9.5 * FONT_SCALE, ha="left", va="center", color="0.30")
ax_e.text(5.2, 0.028, "Exact regime",
          fontsize=9.5 * FONT_SCALE, ha="left", va="bottom", color="0.30")
ax_e.text(15.35, 1.08, r"$\sigma_n = 1$",
          fontsize=9 * FONT_SCALE, ha="right", va="bottom", color="0.25")
ax_e.set_yscale("log")
ax_e.set_xlim(0.45, 15.55)
ax_e.set_ylim(0.008, 12)
ax_e.set_xticks(np.arange(1, 16))
ax_e.set_xlabel("Number line  $n$")
ax_e.set_ylabel(r"Well width  $\sigma_n$")
panel_label(ax_e, "e", x=-0.075)
ax_e.legend(loc="lower right", ncols=2, framealpha=0.9, edgecolor="0.75",
            title="Exposure", title_fontsize=8.5 * FONT_SCALE)

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(f"Saved {OUT_STEM}.{{png,pdf}}")
plt.close(fig)
