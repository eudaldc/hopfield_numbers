#!/usr/bin/env python
"""Figure 2: model comparison (accuracy + RT) and individual width-acuity link.

Single shared-landscape (joint) fit throughout; the per-duration fits are dropped.

  a - Accuracy channel, joint fit: dAIC of every model relative to the winning
      power-law attractor with clean-up. Bars are coloured by model role
      (attractor + clean-up / static readout / external benchmark).
  b - RT channel, joint fit: dAIC of the three RT links on the winning landscape
      (one shared mapping across durations). Single neutral colour - the three
      are variants of the same model, not a model-role category.
  c - Individual differences: empirical-Bayes n_width vs Weber noise from the
      bivariate hierarchical model (non-overlapping n splits; partial pooling).
      Data from ``scripts/trial_level_bivariate_id.py`` (primary cutoff 8/9).

    .venv/bin/python figures/fig3_comparison_ncrit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.plot_style import apply_style, panel_label  # noqa: E402

OUT_DIR = ROOT / "output" / "trial_level_model_comparison"
CP_DIR = ROOT / "output" / "cp_exact"
DRV_DIR = ROOT / "driven_attractor" / "output" / "fits"

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, choices=(100, 120), default=120)
args = parser.parse_args()
TAG = "_G100" if args.budget == 100 else ""
OUT_STEM = ROOT / "figures" / f"fig3_comparison_ncrit{TAG}"


def result_csv(stem: str) -> Path:
    return OUT_DIR / f"{stem}{TAG}.csv"


FONT_SCALE = 2.1
apply_style(FONT_SCALE)

# three model-role colours (NPG): cleanup/static share a cool family, bench accents
CLEANUP = "#3C5488"  # attractor with clean-up dynamics
STATIC = "#4DBBD5"  # matched static readout (same wells, no dynamics)
BENCH = "#E64B35"  # external benchmark
NEUTRAL = CLEANUP  # panel b: same blue as attractor (RT links are variants of it)

# ── Accuracy channel, joint fit: dAIC vs the winning power-law attractor ─────
ladder = pd.read_csv(result_csv("shared_landscape_ladder")).set_index("model_key")
cp_fit = pd.read_csv(CP_DIR / "cp_exact_fits.csv").set_index("variant")
_pure = json.load(open(DRV_DIR / "M2_tether_g0.json"))
PURE_JOINT_AIC = 2 * _pure["k"] - 2 * _pure["logL"]
# The shared-ladder AIC counts G; the driven cache's k excludes it.
G_AIC = 2.0

# (display label, model key, role colour)
ROWS_A = [
    ("Power-law wells + clean-up", "gaussian_pow_attractor", CLEANUP),
    ("Power-law wells, static", "gaussian_pow_readout", STATIC),
    ("Constant-width + clean-up", "gaussian_const_attractor", CLEANUP),
    ("Constant-width, static", "gaussian_const_readout", STATIC),
    ("Power-law wells + driven accumulator", "pure_driven", CLEANUP),
    ("Weber/ANS", "weber_ans", BENCH),
    ("Normative (C&P)", "cp_exact", BENCH),
]


def _raw_aic(key: str) -> float:
    if key == "cp_exact":
        return float(cp_fit.loc["uniform_duration_lapse", "AIC"])
    if key == "pure_driven":
        return PURE_JOINT_AIC
    return float(ladder.loc[key, "joint_AIC"])


def joint_daic(key: str) -> float:
    aic = _raw_aic(key) + (G_AIC if key == "pure_driven" else 0.0)
    ref = _raw_aic("gaussian_pow_attractor")
    return aic - ref


ACC = [(lab, joint_daic(key), col) for lab, key, col in ROWS_A]

# ── RT channel, joint fit: dAIC of the three RT links (one shared mapping) ───
rtj = pd.read_csv(result_csv("rt_joint")).set_index("metric")
RT = [
    ("Convergence time + entropy", float(rtj.loc["hybrid", "dAIC"]), NEUTRAL),
    ("Convergence time", float(rtj.loc["steps", "dAIC"]), NEUTRAL),
    ("Entropy", float(rtj.loc["entropy", "dAIC"]), NEUTRAL),
]


def sqrt_barh(ax, items, xlabel, *, decimal_labels=False):
    """Horizontal dAIC bars on a sqrt scale (huge range), best model at the top."""
    items = sorted(items, key=lambda t: t[1])
    labs = [t[0] for t in items][::-1]
    vals = np.array([t[1] for t in items])[::-1]
    cols = [t[2] for t in items][::-1]
    y = np.arange(len(labs))
    ax.barh(y, np.sqrt(vals), color=cols, edgecolor="black", lw=1.0, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    for yi, v in zip(y, vals):
        value_label = f"{v:.2f}" if v < 1 else f"{v:.1f}"
        ax.text(
            np.sqrt(v) + np.sqrt(vals.max()) * 0.015,
            yi,
            "best" if v == 0 else value_label if decimal_labels else f"{v:,.0f}",
            va="center",
            ha="left",
            fontsize=8 * FONT_SCALE,
        )
    ticks = [0, 500, 2000, 5000, 11000]
    ax.set_xticks(np.sqrt(ticks))
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.set_xlim(0, np.sqrt(vals.max()) * 1.18)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)


# ── Figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 15))
gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.72, 1.45], hspace=0.34)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

sqrt_barh(ax_a, ACC, r"$\Delta$AIC (accuracy)")
sqrt_barh(ax_b, RT, r"$\Delta$AIC (response time)", decimal_labels=True)
ax_a.legend(
    handles=[
        Patch(fc=CLEANUP, ec="black", lw=0.6, label="Attractor + clean-up"),
        Patch(fc=STATIC, ec="black", lw=0.6, label="Static readout"),
        Patch(fc=BENCH, ec="black", lw=0.6, label="Benchmark"),
    ],
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.02),
    fontsize=8 * FONT_SCALE,
    handlelength=1.1,
)

# panel c - hierarchical EB n_width vs Weber noise (primary cutoff 8/9)
ind = pd.read_csv(result_csv("bivariate_id_participants"))
ind = ind[ind["cutoff"] == "8/9"]
summary = pd.read_csv(result_csv("bivariate_id_summary"))
rho_hat = float(summary.loc[summary.cutoff == "8/9", "rho"].iloc[0])
d = ind[["weber_mean", "n_width_mean"]].dropna()
ax_c.scatter(
    d.weber_mean, d.n_width_mean, s=34, alpha=0.45, color=CLEANUP, edgecolor="none"
)
ax_c.set_xlabel("Weber noise $w$")
ax_c.set_ylabel(r"$n_{\mathrm{width}}$")
ax_c.set_xlim(d.weber_mean.min() - 0.01, d.weber_mean.quantile(0.99) + 0.02)
ax_c.text(
    0.96,
    0.95,
    "Latent participant correlation\n"
    rf"$\rho = {rho_hat:.2f}$, $N = {len(d)}$",
    transform=ax_c.transAxes,
    ha="right",
    va="top",
    fontsize=8 * FONT_SCALE,
)
ax_c.spines[["top", "right"]].set_visible(False)

for ax in (ax_a, ax_b, ax_c):
    for s in ax.spines.values():
        s.set_color("black")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

panel_label(ax_a, "a")
panel_label(ax_b, "b")
panel_label(ax_c, "c")

# Panels a/b have long y-tick labels that push their axes right; panel c shares
# that same axes left edge, leaving empty space. Expand c leftward so its full
# content (ylabel + ticks + plot) aligns with the left edge of a/b's labels.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
left_a = ax_a.get_tightbbox(renderer).transformed(inv).x0
left_c = ax_c.get_tightbbox(renderer).transformed(inv).x0
delta = left_c - left_a
if delta > 0:
    pos = ax_c.get_position()
    ax_c.set_position([pos.x0 - delta, pos.y0, pos.width + delta, pos.height])

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(f"Saved {OUT_STEM}.{{png,pdf}}  (panel c: rho={rho_hat:.3f}, n={len(d)})")
plt.close(fig)
