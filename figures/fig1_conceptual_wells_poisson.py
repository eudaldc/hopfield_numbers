#!/usr/bin/env python
"""Conceptual figure: Poisson attractor wells with entropy-based recognition.

Faithful representation of the model (well_type=poisson, rt_mode=entropy):
  - Poisson wells: f_n(x) = (x/c) log(n/c) - n/c - lgamma(x/c + 1)
  - Variance grows linearly with n: Var(x|n) = c · n
  - No Benford frequency — well shape is fully determined by the Poisson distribution
  - Posterior: p(n|x) ∝ exp(f_n(x))
  - RT = t0 + kappa * H(posterior)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as pe
from scipy.special import gammaln

# ── Project house style ───────────────────────────────────────────────
FONT_SCALE = 2.1

STYLE: dict = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 13 * FONT_SCALE,
    "axes.labelsize": 11 * FONT_SCALE,
    "xtick.labelsize": 9 * FONT_SCALE,
    "ytick.labelsize": 9 * FONT_SCALE,
    "legend.fontsize": 9 * FONT_SCALE,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

mpl.rcParams.update(STYLE)

# ── Model parameters (poisson wells, entropy RT) ──────────────────────
RHO = 1              # number-line compression (1 = linear)
SIGMA_C = 0.042      # Poisson scale  (fitted at 160 ms)

N_DISPLAY = 15       # visible range
N_SUPPORT = 18       # wells in computation (padding beyond display)

# ── Annotation positions — all (x, y) in axes-fraction coords ────────
DEEP_WELL_POS      = (0.15, 0.10)
DEEP_WELL_ARROW    = (0.02, 0.29)
SHALLOW_WELL_POS   = (0.60, 0.40)
SHALLOW_WELL_ARROW = (0.91, 0.83)
LOW_ENTROPY_POS    = (0.35, 0.05)
HIGH_ENTROPY_POS   = (0.85, 0.23)

PANEL_B_HEADROOM = 2.2


# ── Poisson well function ────────────────────────────────────────────
def poisson_log_pmf(x_arr: np.ndarray, n: float, c: float) -> np.ndarray:
    """Scaled Poisson log-PMF: Poisson(n/c) evaluated at x/c."""
    x_s = np.clip(x_arr / c, 0.1, None)
    n_s = n / c
    return x_s * np.log(n_s) - n_s - gammaln(x_s + 1)


# ── Derived quantities ────────────────────────────────────────────────
integers = np.arange(1, N_SUPPORT + 1, dtype=float)
integers_display = integers[:N_DISPLAY].astype(int)

mu = integers ** RHO
mu_display = mu[:N_DISPLAY]

# Well standard deviation: std_n = sqrt(c · n)
well_std = np.sqrt(SIGMA_C * integers)

# Visual strength: 1/n gives steep enough decay to offset wide-well overlap
peak_prob = 1.0 / integers
peak_prob_norm = peak_prob / peak_prob.max()

# Continuous x-axis
X_PAD = 0.3
x = np.linspace(mu[0] - X_PAD - 0.3, mu[-1] + 0.5, 6000)
XLIM = (mu[0] - X_PAD, mu[N_DISPLAY - 1] + X_PAD)

# ── Visual wells (Panel A) ───────────────────────────────────────────
DEPTH_MIN, DEPTH_MAX = 0.10, 1.0
visual_depth = DEPTH_MIN + (DEPTH_MAX - DEPTH_MIN) * peak_prob_norm

wells_viz = np.zeros((N_SUPPORT, len(x)))
for i in range(N_SUPPORT):
    raw = poisson_log_pmf(x, integers[i], SIGMA_C)
    peak_val = poisson_log_pmf(mu[i], integers[i], SIGMA_C)
    bell = np.exp(np.clip(raw - peak_val, -30, 0))
    wells_viz[i] = -visual_depth[i] * bell

E_total = wells_viz.sum(axis=0)

# ── Posterior entropy (pure Poisson likelihood, no Benford prior) ────
log_post = np.zeros((len(x), N_SUPPORT))
for i in range(N_SUPPORT):
    log_post[:, i] = poisson_log_pmf(x, integers[i], SIGMA_C)

log_post -= log_post.max(axis=1, keepdims=True)
posterior = np.exp(log_post)
posterior /= posterior.sum(axis=1, keepdims=True)

entropy = -np.sum(posterior * np.log(posterior + 1e-12), axis=1)
H_at_int = np.array(
    [entropy[np.argmin(np.abs(x - m))] for m in mu_display]
)

# ── Colourmap ─────────────────────────────────────────────────────────
cmap_wells = mpl.colormaps["plasma"]
norm_c = Normalize(vmin=-0.10, vmax=1.15)


def well_color(i: int) -> tuple:
    return cmap_wells(norm_c(peak_prob_norm[i]))


# ── Figure layout ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.30)

ax_land = fig.add_subplot(gs[0])
ax_ent = fig.add_subplot(gs[1])

# ──────────────────────────────────────────────────────────────────────
#  Panel A — Energy landscape (Poisson wells)
# ──────────────────────────────────────────────────────────────────────
for i in range(N_SUPPORT):
    ax_land.fill_between(
        x, 0, wells_viz[i], alpha=0.40, color=well_color(i), lw=0,
    )

ax_land.plot(x, E_total, color="0.10", lw=2.2, zorder=4)
ax_land.axhline(0, color="0.70", lw=0.5, ls="--", zorder=1)

for i in range(N_DISPLAY):
    ax_land.text(
        mu[i], -visual_depth[i] - 0.03, str(int(integers[i])),
        ha="center", va="top",
        fontsize=9 * FONT_SCALE, fontweight="bold",
        color=well_color(i),
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        zorder=5,
    )

ax_land.annotate(
    "Deep, narrow well\n(small $n$)",
    xy=DEEP_WELL_ARROW, xycoords="axes fraction",
    xytext=DEEP_WELL_POS, textcoords="axes fraction",
    fontsize=10 * FONT_SCALE, color="0.15", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.5,
                    connectionstyle="arc3,rad=0.35"),
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)
ax_land.annotate(
    "Shallow, wide well\n(large $n$)",
    xy=SHALLOW_WELL_ARROW, xycoords="axes fraction",
    xytext=SHALLOW_WELL_POS, textcoords="axes fraction",
    fontsize=10 * FONT_SCALE, color="0.15", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.5,
                    connectionstyle="arc3,rad=0.25"),
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)

ax_land.set_xlim(*XLIM)
ax_land.set_ylim(min(E_total) * 1.15, 0.12)
ax_land.set_ylabel("Energy  $E(x)$")

ax_land.set_title("a", fontweight="bold", loc="left", x=0.025)
ax_land.set_xticks(mu_display)
ax_land.set_xticklabels([str(n) for n in integers_display])
ax_land.set_yticks([])


# ──────────────────────────────────────────────────────────────────────
#  Panel B — Posterior entropy landscape
# ──────────────────────────────────────────────────────────────────────

ax_ent.fill_between(x, 0, entropy, alpha=0.07, color="0.5", zorder=0)
ax_ent.plot(x, entropy, color="0.10", lw=2.2, zorder=3)

for i in range(N_DISPLAY):
    ax_ent.axvline(mu[i], color="0.92", lw=0.3, zorder=0)

for i in range(N_DISPLAY):
    ax_ent.plot(
        mu[i], H_at_int[i], "o",
        color=well_color(i), ms=10, zorder=6,
        markeredgecolor="white", markeredgewidth=1.5,
    )

# Annotations
ax_ent.annotate(
    "Low entropy\n(certain $\\rightarrow$ fast RT)",
    xy=(mu[1], H_at_int[1]),
    xytext=LOW_ENTROPY_POS,
    textcoords="axes fraction",
    fontsize=10 * FONT_SCALE, color=PALETTE[2], ha="center", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=PALETTE[2], lw=1.5,
                    connectionstyle="arc3,rad=-0.1"),
    zorder=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
)
ax_ent.annotate(
    "High entropy\n(uncertain $\\rightarrow$ slow RT)",
    xy=(mu[13], H_at_int[13]),
    xytext=HIGH_ENTROPY_POS,
    textcoords="axes fraction",
    fontsize=10 * FONT_SCALE, color=PALETTE[3], ha="center", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=PALETTE[3], lw=1.5,
                    connectionstyle="arc3,rad=-0.25"),
    zorder=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
)

# ── Inset: example posterior distributions ────────────────────────────
INSET_HALF = 3
INSET_PAD = 0.8

inset_ns = [2, 8, 13]
inset_labels = ["sharp", "moderate", "broad"]
inset_bounds = [
    [0.028, 0.37, 0.16, 0.44],
    [0.42, 0.48, 0.16, 0.44],
    [0.76, 0.55, 0.16, 0.44],
]
inset_title_y = [1.02, 0.80, 0.66]

# Shared ylim so bar heights are comparable
global_max_p = 0.0
for n_pos in inset_ns:
    idx = np.argmin(np.abs(x - mu[n_pos - 1]))
    lo, hi = n_pos - INSET_HALF, n_pos + INSET_HALF
    m = (integers >= lo) & (integers <= hi)
    global_max_p = max(global_max_p, posterior[idx, m].max())
INSET_YLIM = global_max_p * 1.10

for n_pos, lbl, bounds, ty in zip(
    inset_ns, inset_labels, inset_bounds, inset_title_y,
):
    ins = ax_ent.inset_axes(bounds)
    ins.set_facecolor("white")

    i_pos = n_pos - 1
    x_idx = np.argmin(np.abs(x - mu[i_pos]))
    post_n = posterior[x_idx, :]

    lo_n = n_pos - INSET_HALF
    hi_n = n_pos + INSET_HALF
    mask = (integers >= lo_n) & (integers <= hi_n)
    ns_show = integers[mask].astype(int)
    ps_show = post_n[mask]
    colors_show = [well_color(j) for j in np.where(mask)[0]]

    ins.bar(
        ns_show, ps_show, color=colors_show,
        edgecolor="white", linewidth=0.4, width=0.7,
    )
    ins.set_xlim(lo_n - INSET_PAD, hi_n + INSET_PAD)
    ins.set_ylim(0, INSET_YLIM)
    ins.set_xticks(ns_show)
    if n_pos == 2:
        xtick_lbls = [str(int(n)) for n in ns_show]
    elif n_pos == 8:
        xtick_lbls = [str(int(n)) if int(n) in (6, 8, 10) else "" for n in ns_show]
    else:
        xtick_lbls = [str(int(n)) if int(n) in (11, 13, 15) else "" for n in ns_show]
    ins.set_xticklabels(xtick_lbls, fontsize=9 * FONT_SCALE)
    ins.set_yticks([])
    ins.set_title(
        f"$p(n \\mid x\\!=\\!\\mu_{{{n_pos}}})$ — {lbl}\n"
        f"$H = {H_at_int[i_pos]:.2f}$",
        fontsize=8.5 * FONT_SCALE, y=ty,
    )
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)

# ── Gradient legend ───────────────────────────────────────────────────
gradient_vals = np.linspace(1.0, 0.05, 5)
legend_dots = []
for v in gradient_vals:
    d, = ax_ent.plot(
        [], [], "o", color=cmap_wells(norm_c(v)), ms=10,
        markeredgecolor="white", markeredgewidth=0.8,
    )
    legend_dots.append(d)

ax_ent.legend(
    [tuple(legend_dots)], [r"Posterior entropy $H(n)$"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    loc="lower right", framealpha=0.9, edgecolor="0.75",
    handletextpad=0.8,
)

H_max = entropy[np.isfinite(entropy)].max()
ax_ent.set_xlim(*XLIM)
ax_ent.set_ylim(-0.03, H_max * PANEL_B_HEADROOM)

if RHO == 1:
    ax_ent.set_xlabel("Number line  $n$")
else:
    ax_ent.set_xlabel(
        r"Compressed number line  $\mu_n = n^{\rho}$"
        f"  ($\\rho = {RHO}$)",
    )

ax_ent.set_ylabel("Posterior entropy landscape $H(x)$")
ax_ent.set_title("b", fontweight="bold", loc="left", x=0.025)
ax_ent.set_xticks(mu_display)
ax_ent.set_xticklabels([str(n) for n in integers_display])
ax_ent.set_yticks([])

# ── Save ──────────────────────────────────────────────────────────────
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/conceptual_attractor_wells_poisson.png")
fig.savefig(f"{out_dir}/conceptual_attractor_wells_poisson.pdf")
print(f"Saved to {out_dir}/conceptual_attractor_wells_poisson.{{png,pdf}}")
plt.close(fig)
