#!/usr/bin/env python
"""Figure 3 (revision): trial-level model comparison + the width law and n_crit.

Panel a — Poisson-free response-AIC ladder from the trial-level comparison
(consolidated_summary.csv). The width argument is carried by the nested
fixed-sqrt(n) (beta = 1/2) Gaussian variants; the clean-up argument by the
attractor vs static pairs at equal parameter count.

Panel b — the fitted width law sigma_n per exposure duration, the sigma_n = 1
(adjacent-well d' = 1) resolvability line, and the emerging n_crit per duration:
the model predicts its own limit, and the limit grows with exposure.

    .venv/bin/python figures/fig3_comparison_ncrit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "trial_level_model_comparison"
OUT_STEM = ROOT / "figures" / "fig3_comparison_ncrit"

FONT_SCALE = 2.1
STYLE: dict = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 13 * FONT_SCALE,
    "axes.labelsize": 11 * FONT_SCALE,
    "xtick.labelsize": 9 * FONT_SCALE,
    "ytick.labelsize": 9 * FONT_SCALE,
    "legend.fontsize": 8.5 * FONT_SCALE,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}
mpl.rcParams.update(STYLE)

# ── Panel a data: Poisson-free ladder ─────────────────────────────────
LADDER = {
    # model_key -> (label, category)
    "gaussian_freebeta_attractor": ("Super-linear width attractor (free $\\beta$) $\\bigstar$", "attractor"),
    "gaussian_pow_attractor": ("Power-law width attractor ($\\sigma_n \\propto n^p$)", "attractor"),
    "gaussian_benford_free_beta": ("Super-linear width readout (no clean-up)", "static"),
    "gaussian_pow_readout": ("Power-law width readout (no clean-up)", "static"),
    "weber_ans": ("Weber/ANS ($\\sigma_n \\propto n$)", "benchmark"),
    "gaussian_benford_attractor": ("Fixed $\\sqrt{n}$-width attractor ($\\beta = 1/2$)", "attractor"),
    "gaussian_benford_no_cleanup": ("Fixed $\\sqrt{n}$-width readout (no clean-up)", "static"),
    "cp_log_prior_proxy": ("C&P log-prior proxy", "benchmark"),
}
COLORS = {"attractor": "#4C72B0", "static": "#A3C0DE", "benchmark": "#9A9A9A"}

summary = pd.read_csv(OUT_DIR / "consolidated_summary.csv").set_index("model_key")
rows = []
for key, (label, cat) in LADDER.items():
    rows.append({"key": key, "label": label, "cat": cat,
                 "AIC": float(summary.loc[key, "resp_AIC"])})
lad = pd.DataFrame(rows)
lad["dAIC"] = lad["AIC"] - lad["AIC"].min()
lad = lad.sort_values("dAIC", ascending=True).reset_index(drop=True)

# ── Panel b data: width law per duration ─────────────────────────────
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
fig = plt.figure(figsize=(16, 15))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.85], hspace=0.35)
ax_lad = fig.add_subplot(gs[0])
ax_w = fig.add_subplot(gs[1])

# Panel a — ladder
ypos = np.arange(len(lad))[::-1]
bars = ax_lad.barh(ypos, lad["dAIC"], height=0.62,
                   color=[COLORS[c] for c in lad["cat"]],
                   edgecolor=["0.1" if k == "gaussian_freebeta_attractor" else "none"
                              for k in lad["key"]],
                   linewidth=2.0, zorder=3)
for y, (_, r) in zip(ypos, lad.iterrows()):
    ax_lad.text(r["dAIC"] + 90, y, f"{r['dAIC']:.0f}" if r["dAIC"] > 0 else "best",
                va="center", ha="left", fontsize=9 * FONT_SCALE, color="0.15")
ax_lad.set_yticks(ypos)
ax_lad.set_yticklabels(lad["label"])
ax_lad.set_xlabel(r"$\Delta$AIC vs best (trial-level response likelihood)")
ax_lad.set_xlim(0, lad["dAIC"].max() * 1.14)
ax_lad.set_title("a", fontweight="bold", loc="left", x=-0.32)

handles = [mpl.patches.Patch(color=COLORS[c], label=l) for c, l in
           [("attractor", "attractor (clean-up)"), ("static", "static readout"),
            ("benchmark", "benchmark")]]
ax_lad.legend(handles=handles, loc="upper right", framealpha=0.9, edgecolor="0.75")

# Panel b — width law and n_crit
n_cont = np.linspace(1.0, 15.55, 800)
dur_cmap = mpl.colormaps["viridis"]
dur_colors = {d: dur_cmap(i / (len(DURATIONS) - 1)) for i, d in enumerate(DURATIONS)}

ax_w.axhspan(1.0, 40.0, color="0.55", alpha=0.10, zorder=0)
ax_w.axhline(1.0, color="0.25", lw=1.6, ls="--", zorder=2)

for d in DURATIONS:
    s = sigma_of_n(d, n_cont)
    ax_w.plot(n_cont, s, lw=2.4, color=dur_colors[d], zorder=3, label=f"{d} ms")
    ax_w.plot(ncrit(d), 1.0, "o", ms=12, color=dur_colors[d], zorder=5,
              markeredgecolor="white", markeredgewidth=1.5)

nc_min, nc_max = ncrit(DURATIONS[0]), ncrit(DURATIONS[-1])
ax_w.annotate(
    "", xy=(nc_max + 0.35, 1.0), xytext=(nc_min - 0.35, 1.0),
    arrowprops=dict(arrowstyle="-|>", color="0.25", lw=2.0,
                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=-0.35"),
    zorder=4,
)
ax_w.text(12.9, 5.4, "more exposure $\\rightarrow$\nexact range extends",
          fontsize=9.5 * FONT_SCALE, ha="center", va="center", color="0.15",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
ax_w.text(1.45, 3.6, "wells merge:\napproximate regime",
          fontsize=9.5 * FONT_SCALE, ha="left", va="center", color="0.30")
ax_w.text(5.2, 0.028, "wells resolvable:\nexact regime",
          fontsize=9.5 * FONT_SCALE, ha="left", va="bottom", color="0.30")
ax_w.text(15.35, 1.08, r"$\sigma_n = 1$  ($d' = 1$)",
          fontsize=9 * FONT_SCALE, ha="right", va="bottom", color="0.25")

ax_w.set_yscale("log")
ax_w.set_xlim(0.45, 15.55)
ax_w.set_ylim(0.008, 12)
ax_w.set_xticks(np.arange(1, 16))
ax_w.set_xlabel("Number line  $n$")
ax_w.set_ylabel(r"Well width  $\sigma_n$")
ax_w.set_title("b", fontweight="bold", loc="left", x=-0.32)
ax_w.legend(loc="lower right", ncols=2, framealpha=0.9, edgecolor="0.75",
            title="exposure", title_fontsize=8.5 * FONT_SCALE)

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(f"Saved {OUT_STEM}.{{png,pdf}}")
plt.close(fig)
