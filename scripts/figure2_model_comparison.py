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
      all-duration hierarchical model (non-overlapping n splits; partial
      pooling). The inset reports the latent correlation conditional on the
      participant lapse effect.

    .venv/bin/python -m scripts.figure2_model_comparison
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

from scripts.plot_style import apply_style, panel_label  # noqa: E402

OUT_DIR = ROOT / "output" / "trial_level_model_comparison"
CP_DIR = ROOT / "output" / "cp_exact"
DRV_DIR = ROOT / "driven_attractor" / "output" / "fits"

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, choices=(100, 120), default=120)
args = parser.parse_args()
TAG = "_G100" if args.budget == 100 else ""
OUT_STEM = (
    ROOT / "figures" / "figure2_model_comparison"
    if args.budget == 120
    else OUT_DIR / "figure2_model_comparison_G100"
)


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
# a and b span the full width; the bottom row splits into c (individual
# differences) and d (effective Weber fraction), side by side — both read out
# the same Weber-noise story from different angles.
fig = plt.figure(figsize=(12, 15))
gs = fig.add_gridspec(
    3, 2, height_ratios=[1.3, 0.72, 1.45], hspace=0.34, wspace=0.30
)
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, :])
ax_c = fig.add_subplot(gs[2, 0])
ax_d = fig.add_subplot(gs[2, 1])

sqrt_barh(ax_a, ACC, r"$\Delta$AIC (categorical-response)")
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
hierarchical = pd.read_csv(result_csv("bivariate_id_summary")).iloc[0]
rho_hat = float(hierarchical["rho"])
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
    "Landscape resolution\nWeber noise association\n"
    rf"$\rho = {rho_hat:.2f}$, $N = {len(d)}$",
    transform=ax_c.transAxes,
    ha="right",
    va="top",
    fontsize=7.5 * FONT_SCALE,
)
ax_c.spines[["top", "right"]].set_visible(False)

# panel d - effective Weber fraction: CoV of the fitted response distribution
# per numerosity and exposure, with the empirical CoV (dots). n=1 has no lower
# neighbour, so its CoV is an edge artefact and is dropped, as in the SM figure.
web = pd.read_csv(OUT_DIR / "weber_fraction.csv")
web_durs = sorted(web.duration_ms.unique())
dur_colors = plt.cm.magma(np.linspace(0.15, 0.82, len(web_durs)))
n_width_shared = float(web.n_width.mean())
for d_ms, col in zip(web_durs, dur_colors):
    sub = web[(web.duration_ms == d_ms) & (web.n >= 2)].sort_values("n")
    ax_d.plot(sub.n, sub.model_cov, "-", color=col, lw=2.0, label=f"{d_ms} ms")
    ax_d.plot(sub.n, sub.empirical_cov, "o", color=col, ms=4, alpha=0.9)
ax_d.axvline(n_width_shared, color="0.5", ls=":", lw=1.2)
ax_d.set_xlabel("Numerosity $n$")
ax_d.set_ylabel(r"Response CoV $w_{\mathrm{eff}}$")
ax_d.set_xlim(1.6, 15.4)
ax_d.set_xticks(np.arange(2, 16, 3))
ax_d.legend(
    frameon=False,
    ncol=2,
    loc="upper right",
    fontsize=7 * FONT_SCALE,
    handlelength=1.1,
    columnspacing=1.0,
    labelspacing=0.3,
)
ax_d.spines[["top", "right"]].set_visible(False)

for ax in (ax_a, ax_b, ax_c, ax_d):
    for s in ax.spines.values():
        s.set_color("black")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

panel_label(ax_a, "a")
panel_label(ax_b, "b")
panel_label(ax_c, "c")
panel_label(ax_d, "d")

# Stretch the bottom row (c, d) across the full width that panels a/b occupy —
# their long left-hand model labels included — so no horizontal space is wasted.
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_inv = fig.transFigure.inverted()
_left = min(ax.get_tightbbox(_r).transformed(_inv).x0 for ax in (ax_a, ax_b))
_right = max(ax.get_tightbbox(_r).transformed(_inv).x1 for ax in (ax_a, ax_b))
_gap = 0.10   # horizontal separation between c and d
_drop = 0.03  # nudge the bottom row down, clear of panel b's x-axis label
_c_left = _left + 0.045  # leave room for panel c's own y-axis label
_w = (_right - _c_left - _gap) / 2
for _ax, _x0 in ((ax_c, _c_left), (ax_d, _c_left + _w + _gap)):
    _p = _ax.get_position()
    _ax.set_position([_x0, _p.y0 - _drop, _w, _p.height])

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(
    f"Saved {OUT_STEM}.{{png,pdf}}  "
    f"(panel c: rho={rho_hat:.3f}, n={len(d)}; panel d: {len(web_durs)} durations)"
)
plt.close(fig)
