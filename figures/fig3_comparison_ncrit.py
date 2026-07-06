#!/usr/bin/env python
"""Figure 3 (revision): model comparison, RT channel, and the predicted limit.

Poisson-free throughout. Panels a and b are two column-aligned heatmaps with
exposure durations as columns (plus a separated Total column); cells are dAIC
relative to the best of each column, viridis, darker is better, shading scaled
within each panel. Panel c is the width law.

  a — Response-likelihood dAIC over the ~53,000 individual responses, per
      duration. Rows are model x dynamics: each well family split into its
      clean-up and static variants, plus the Weber/ANS and normative
      benchmarks. Shows the model comparison AND that the clean-up advantage
      is concentrated at brief exposures (statics catch up at 320-640 ms).
  b — RT-channel dAIC on the winning landscape, per duration: the hybrid
      convergence-time + entropy link wins at every exposure, convergence
      time alone stays close, the static entropy proxy degrades at longer
      exposures. This is the response to the "RT is grafted on" objection.
  c — The fitted width law sigma_n per exposure duration, the sigma_n = 1
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
DUR_KEYS = [40, 80, 160, 320, 640, "total"]
DUR_COLS = ["40 ms", "80", "160", "320", "640", "Total"]

# ── Panel a: response dAIC per duration, model x dynamics ─────────────
resp = pd.read_csv(OUT_DIR / "response_per_duration_aic.csv")
piv_a = resp.pivot_table(index="model_key", columns="duration_ms", values="resp_AIC")
piv_a["total"] = piv_a.sum(axis=1)
piv_a = piv_a - piv_a.min(axis=0)

ROWS_A = [
    ("Frequency wells + clean-up", "gaussian_freebeta_attractor"),
    ("Frequency wells, static", "gaussian_benford_free_beta"),
    ("Power-law wells + clean-up", "gaussian_pow_attractor"),
    ("Power-law wells, static", "gaussian_pow_readout"),
    ("Weber/ANS", "weber_ans"),
    ("Normative (log-prior)", "cp_log_prior_proxy"),
]
vals_a = [[float(piv_a.loc[k, d]) for d in DUR_KEYS] for _, k in ROWS_A]

# ── Panel b: RT-channel dAIC per duration on the winning landscape ────
rt = pd.read_csv(OUT_DIR / "rt_exploration.csv")
rt = rt[rt.model_key == "gaussian_freebeta_attractor"]
piv_b = rt.pivot_table(index="metric", columns="duration_ms", values="AIC")
piv_b = piv_b.loc[["hybrid", "steps", "entropy"]]
piv_b["total"] = piv_b.sum(axis=1)
piv_b = piv_b - piv_b.min(axis=0)

ROWS_B = [
    ("Convergence time + entropy", "hybrid"),
    ("Convergence time", "steps"),
    ("Readout entropy", "entropy"),
]
vals_b = [[float(piv_b.loc[k, d]) for d in DUR_KEYS] for _, k in ROWS_B]


# ── Heatmap drawing helper ─────────────────────────────────────────────
def heatmap(ax, vals, row_labels, xlabel, x_edges):
    """dAIC heatmap: viridis on a sqrt scale (dark = best), values in the
    cells, duration headers on top, channel label below the grid."""
    nrow = len(vals)
    vmax = max(v for row in vals for v in row)
    for i, row in enumerate(vals):
        y0 = nrow - 1 - i
        for j, v in enumerate(row):
            x0, x1 = x_edges[j]
            frac = float(np.sqrt(v / vmax))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, 1.0, fc=CMAP(frac),
                                   ec="white", lw=3, zorder=2))
            ax.text((x0 + x1) / 2, y0 + 0.5, f"{v:.0f}", ha="center",
                    va="center", fontsize=9.5 * FONT_SCALE,
                    color="white" if frac < 0.55 else "0.15", zorder=3)
    for (x0, x1), lab in zip(x_edges, DUR_COLS):
        ax.text((x0 + x1) / 2, nrow + 0.13, lab, ha="center", va="bottom",
                fontsize=8.5 * FONT_SCALE, color="0.15")
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_xlim(x_edges[0][0], x_edges[-1][1])
    ax.set_ylim(0, nrow)
    ax.set_xticks([])
    ax.set_yticks([nrow - 0.5 - i for i in range(nrow)])
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


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
GRID_LEFT = 0.364     # centres the heatmap block (labels + cells) on panel c
X_EDGES = [(float(j), j + 1.0) for j in range(5)] + [(5.3, 6.3)]
ax_a = fig.add_axes((GRID_LEFT, 0.955 - 6 * CELL_H, 6.3 * CELL_W, 6 * CELL_H))
ax_b = fig.add_axes((GRID_LEFT, 0.60 - 3 * CELL_H, 6.3 * CELL_W, 3 * CELL_H))
ax_c = fig.add_axes((0.10, 0.04, 0.87, 0.345))

heatmap(ax_a, vals_a, [r for r, _ in ROWS_A],
        xlabel=r"$\Delta$AIC (response)", x_edges=X_EDGES)
panel_label(ax_a, "a", x=-0.64, pad=30)

heatmap(ax_b, vals_b, [r for r, _ in ROWS_B],
        xlabel=r"$\Delta$AIC (response time)", x_edges=X_EDGES)
panel_label(ax_b, "b", x=-0.64, pad=30)

# Panel c — width law and n_crit
n_cont = np.linspace(1.0, 15.55, 800)
dur_colors = {d: CMAP(i / (len(DURATIONS) - 1)) for i, d in enumerate(DURATIONS)}
ax_c.axhspan(1.0, 40.0, color="0.55", alpha=0.10, zorder=0)
ax_c.axhline(1.0, color="0.25", lw=1.6, ls="--", zorder=2)
for d in DURATIONS:
    ax_c.plot(n_cont, sigma_of_n(d, n_cont), lw=2.4, color=dur_colors[d], zorder=3,
              label=f"{d} ms")
    ax_c.plot(ncrit(d), 1.0, "o", ms=12, color=dur_colors[d], zorder=5,
              markeredgecolor="white", markeredgewidth=1.5)

ax_c.text(1.45, 3.6, "Approximate regime",
          fontsize=9.5 * FONT_SCALE, ha="left", va="center", color="0.30")
ax_c.text(5.2, 0.028, "Exact regime",
          fontsize=9.5 * FONT_SCALE, ha="left", va="bottom", color="0.30")
ax_c.text(15.35, 1.08, r"$\sigma_n = 1$",
          fontsize=9 * FONT_SCALE, ha="right", va="bottom", color="0.25")
ax_c.set_yscale("log")
ax_c.set_xlim(0.45, 15.55)
ax_c.set_ylim(0.008, 12)
ax_c.set_xticks(np.arange(1, 16))
ax_c.set_xlabel("Number line  $n$")
ax_c.set_ylabel(r"Well width  $\sigma_n$")
panel_label(ax_c, "c", x=-0.075)
ax_c.legend(loc="lower right", ncols=2, framealpha=0.9, edgecolor="0.75",
            title="Exposure", title_fontsize=8.5 * FONT_SCALE)

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(f"Saved {OUT_STEM}.{{png,pdf}}")
plt.close(fig)
