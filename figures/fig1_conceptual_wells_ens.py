#!/usr/bin/env python
"""Figure 1 (revision): conceptual figure of the winning super-linear Gaussian attractor.

Two stacked panels sharing the number-line x-axis (n = 1..15), 160 ms fitted
parameters:
  a — energy landscape: deep, isolated wells for small n and broader,
      overlapping components at larger n. Well widths are drawn to scale,
      floored at 0.1 for visibility
      (the n = 1 well is 0.02 wide); depths are the true well depths
      -log sigma_n, floored at 0.05 so broad wells remain visible. Lattice
      n = 1..19 as in the fitted model.
  b — posterior entropy BEFORE clean-up (grey: the raw readout of a position x,
      with high ridges between the narrow small-n wells, where a flat readout
      pushes mass onto the broad distant wells) and AFTER clean-up (black: the
      attractor resolves those ambiguous positions, quantising the line into
      basins; the residual entropy staircase characterises readout ambiguity).
      For this
      illustration the lattice is extended to n = 30 so that the finite-lattice
      truncation does not bend the curve down near n = 15 (fits use n <= 19
      with the contamination channel absorbing larger responses).

    .venv/bin/python figures/fig1_conceptual_wells_ens.py
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.plot_style import C_STATIC, PALETTE, apply_style, panel_label  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT_STEM = ROOT / "figures" / "conceptual_attractor_wells_ens"

# ── Project house style ───────────────────────────────────────────────
FONT_SCALE = 2.1
apply_style(FONT_SCALE)

N_DISPLAY = 15
N_LATTICE_MODEL = 19    # panel a: the fitted model's lattice (stimuli 1..15 + padding)
N_LATTICE_ILLUS = 30    # panel b: extended lattice, avoids edge truncation in H
SHOW_DURATION = 160

theta = T.load_shared_theta()
SIGMA_C = float(theta.loc[SHOW_DURATION, "sigma_c"])
POW = float(theta.loc[SHOW_DURATION, "pow"])


def widths(wells: np.ndarray) -> np.ndarray:
    # power-law wells: sigma_n = sigma_c * n^p
    return SIGMA_C * np.asarray(wells, float) ** POW
wells_a = np.arange(1, N_LATTICE_MODEL + 1, dtype=float)
sigma_a = widths(wells_a)
wells_b = np.arange(1, N_LATTICE_ILLUS + 1, dtype=float)
sigma_b = widths(wells_b)

# Number frequency declines as a power of magnitude (Dehaene & Mehler 1992): this
# is the motivation for wider wells at larger n. Shown illustratively as ~1/n
# (the width law itself is parameterised directly as n^p, not through f_n).
num_freq = 1.0 / wells_a
num_freq_norm = num_freq / num_freq.max()
integers_display = wells_a[:N_DISPLAY].astype(int)

X_PAD = 0.55
x = np.linspace(wells_a[0] - X_PAD - 0.2, N_DISPLAY + X_PAD + 0.2, 6000)
XLIM = (wells_a[0] - X_PAD, N_DISPLAY + X_PAD)

# ── Panel a: stylized wells (true depths/widths, floored for visibility) ──
depth = np.maximum(-np.log(sigma_a), 0.05)
width_viz = np.maximum(sigma_a, 0.10)

wells_viz = np.zeros((N_LATTICE_MODEL, len(x)))
for i in range(N_LATTICE_MODEL):
    wells_viz[i] = -depth[i] * np.exp(-0.5 * ((x - wells_a[i]) / width_viz[i]) ** 2)
E_total = wells_viz.sum(axis=0)

# ── Panel b: entropy before vs after clean-up (extended lattice) ────────


def posterior_of(xv: np.ndarray) -> np.ndarray:
    logp = -np.log(sigma_b)[None, :] - (xv[:, None] - wells_b[None, :]) ** 2 / (
        2.0 * (sigma_b ** 2)[None, :])
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def entropy_of(xv: np.ndarray) -> np.ndarray:
    p = posterior_of(xv)
    return -np.sum(p * np.log(p + 1e-12), axis=1)


def cleanup(y0: np.ndarray, eta: float = 1.0, max_steps: int = 120,
            tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Clamped gradient descent on the winner landscape, with the same step
    clamp, tolerance, oscillation guard and 120-step budget as the fitted
    module (T._cleanup) — the figure shows what the model actually does.
    Returns (x_star, n_steps)."""
    xv = y0.copy()
    max_step = 0.5 * float(sigma_b.min())
    prev = np.zeros_like(xv)
    active = np.ones(xv.shape, dtype=bool)
    n_steps = np.full(xv.shape, float(max_steps))
    for t in range(max_steps):
        diff = xv[:, None] - wells_b[None, :]
        logw = -np.log(sigma_b)[None, :] - diff ** 2 / (2.0 * (sigma_b ** 2)[None, :])
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        pull = (w * diff / (sigma_b ** 2)[None, :]).sum(axis=1)
        step = np.clip(eta * pull, -max_step, max_step)
        osc = active & (t > 0) & (step * prev < -1e-12)
        done = osc | (active & (np.abs(step) < tol))
        n_steps = np.where(done, t + 1, n_steps)
        active &= ~done
        xv = np.where(active, xv - step, xv)
        prev = np.where(active, step, prev)
        if not active.any():
            break
    return xv, n_steps


# Expected entropy over the sensory channel y0 ~ N(nu, sd(nu)) at the fitted
# 160 ms sensory parameters. The deterministic basin map alone would be a step
# function, whereas sensory averaging gives the smooth ambiguity curve shown.
S00 = 0.05
S0_D = float(theta.loc[SHOW_DURATION, "s0"])
LAMBDA0_D = float(theta.loc[SHOW_DURATION, "lambda0"])


def sensory_sd(nu: np.ndarray) -> np.ndarray:
    return np.sqrt(S00 ** 2 + S0_D * np.power(nu, LAMBDA0_D))


x_map = np.linspace(-2.0, 33.0, 14000)
H_pre_map = entropy_of(x_map)
x_star_map, steps_map = cleanup(x_map.copy())
H_post_map = entropy_of(x_star_map)

T_NODES = np.linspace(-4.0, 4.0, 81)
W_NODES = np.exp(-0.5 * T_NODES ** 2)
W_NODES /= W_NODES.sum()


def expected_of(nu_grid: np.ndarray, value_map: np.ndarray) -> np.ndarray:
    sd = sensory_sd(nu_grid)
    y = nu_grid[:, None] + sd[:, None] * T_NODES[None, :]
    vy = np.interp(y.ravel(), x_map, value_map).reshape(y.shape)
    return (vy * W_NODES[None, :]).sum(axis=1)


# Presented numerosities are integers — curves are drawn through n = 1..15 only
n_int = wells_a[:N_DISPLAY]
H_pre_i = expected_of(n_int, H_pre_map)
H_at_int = expected_of(n_int, H_post_map)    # = the model's E[H] per n
ET_at_int = expected_of(n_int, steps_map)    # = the model's E[T] per n
mu_star, _ = cleanup(wells_b[:N_DISPLAY].copy())
H_single = entropy_of(mu_star)               # single-trial insets

# Selected RT channel: RT = t0 + tau * E[T].
rt_theta = pd.read_csv(T.OUT_DIR / "rt_joint.csv")
rt_row = rt_theta[rt_theta.metric == "steps"].iloc[0]
RT_T0 = 1.30
RT_TAU = float(rt_row["scale"])
RT_at_int = RT_T0 + RT_TAU * ET_at_int

# Magma is the paper's one continuous scale (fig. 2c durations, SM recovery);
# here it encodes number frequency (higher for small n), which motivates the
# narrow-to-wide well profile
cmap_wells = mpl.colormaps["magma"]
norm_c = Normalize(vmin=-0.10, vmax=1.15)


def well_color(i: int) -> tuple:
    return cmap_wells(norm_c(num_freq_norm[i]))


# ── Figure layout ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 19))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 0.9, 0.72], hspace=0.36)
ax_land = fig.add_subplot(gs[0])
ax_ent = fig.add_subplot(gs[1])
ax_rt = fig.add_subplot(gs[2])

# ──────────────────────────────────────────────────────────────────────
#  Panel a — Energy landscape (160 ms fitted parameters)
# ──────────────────────────────────────────────────────────────────────
for i in range(N_LATTICE_MODEL):
    ax_land.fill_between(x, 0, wells_viz[i], alpha=0.40, color=well_color(i), lw=0)
ax_land.plot(x, E_total, color="0.10", lw=2.2, zorder=4)
ax_land.axhline(0, color="0.70", lw=0.5, ls="--", zorder=1)

for i in range(N_DISPLAY):
    ax_land.text(
        wells_a[i], -depth[i] - (0.10 if depth[i] > 0.3 else 0.42),
        str(int(wells_a[i])),
        ha="center", va="top", fontsize=9 * FONT_SCALE, fontweight="bold",
        color=well_color(i),
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        zorder=5,
    )

ax_land.text(
    0.2, 0.05, "Deep, isolated wells:\nexact readout",
    transform=ax_land.transAxes,
    fontsize=10 * FONT_SCALE, color="0.15", ha="center", va="bottom",
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)
ax_land.text(
    0.8, 0.52, "Broad, overlapping components:\ncoarser readout",
    transform=ax_land.transAxes,
    fontsize=10 * FONT_SCALE, color="0.15", ha="center", va="bottom",
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)

ax_land.set_xlim(*XLIM)
ax_land.set_ylim(-4.95, 0.55)   # extra room below so the annotation box clears the frequency line
ax_land.set_xlabel("Evidence axis  $x$")
ax_land.set_ylabel("Energy")
panel_label(ax_land, "a")
ax_land.set_xticks(integers_display)
ax_land.set_xticklabels([str(n) for n in integers_display])
ax_land.set_yticks([])

# Secondary y-axis: number frequency, which declines with magnitude and motivates
# the narrow-to-wide well profile
FREQ_COLOR = PALETTE[1]   # deep-palette orange — the number-frequency decline
ax_fq = ax_land.twinx()
ax_fq.plot(
    integers_display, num_freq[:N_DISPLAY], "-o",
    color=FREQ_COLOR, ms=5, lw=1.8, alpha=0.85,
    markeredgecolor="white", markeredgewidth=0.8, zorder=7,
)
ax_fq.set_ylabel("Number frequency", color=FREQ_COLOR, labelpad=10)
ax_fq.tick_params(axis="y", colors=FREQ_COLOR)
ax_fq.set_ylim(-num_freq[0] * 0.08, num_freq[0] * 1.15)   # bottom pad matched to ax_land's
ax_fq.spines["right"].set_visible(True)
ax_fq.spines["right"].set_color(FREQ_COLOR)
ax_fq.spines["right"].set_alpha(0.5)

# ──────────────────────────────────────────────────────────────────────
#  Panel b — Posterior entropy before vs after clean-up
# ──────────────────────────────────────────────────────────────────────
ax_ent.plot(n_int, H_at_int, "-", color="0.10", lw=2.6, zorder=7)

for i in range(N_DISPLAY):
    ax_ent.plot(wells_a[i], H_at_int[i], "o", color=well_color(i), ms=10,
                zorder=8, markeredgecolor="white", markeredgewidth=1.5)

# fixed vertical scale (kept independent of the plotted curve so the panel
# layout and the inset placements do not shift)
H_max = float(np.max(H_pre_i))

# Insets: example posteriors after clean-up (sharp / moderate / broad)
INSET_HALF = 3
INSET_PAD = 0.8
inset_ns = [2, 6, 11]
inset_labels = ["sharp", "moderate", "broad"]   # kept for per-inset scaling/axis logic; not shown in titles
inset_bounds = [
    [0.028, 0.392, 0.15, 0.36],   # sharp (y0=2): centred above n=2, lowered 0.3h
    [0.325, 0.52, 0.15, 0.36],   # moderate (y0=6): axis trimmed to 5-9, box shifted so n=6 sits above n=6
    [0.624, 0.70, 0.15, 0.24],   # broad (y0=11): centred above n=11, raised above the E[H] curve; title to its right
]
post_at = posterior_of(mu_star)
global_max_p = 0.0
for n_pos in inset_ns:
    m = (wells_b >= n_pos - INSET_HALF) & (wells_b <= n_pos + INSET_HALF)
    global_max_p = max(global_max_p, post_at[n_pos - 1, m].max())
INSET_YLIM = global_max_p * 1.10

for n_pos, lbl, bounds in zip(inset_ns, inset_labels, inset_bounds):
    ins = ax_ent.inset_axes(bounds)
    ins.set_facecolor("none")   # transparent so the readout curve reads on top, uncut
    post_n = post_at[n_pos - 1, :]
    mask = (wells_b >= n_pos - INSET_HALF) & (wells_b <= n_pos + INSET_HALF)
    ns_show = wells_b[mask].astype(int)
    ps_show = post_n[mask]
    colors_show = [well_color(min(j, N_LATTICE_MODEL - 1)) for j in np.where(mask)[0]]
    ins.bar(ns_show, ps_show, color=colors_show, edgecolor="white",
            linewidth=0.4, width=0.7)
    # moderate/broad readouts have low peaks; give them their own scale so the bars stay legible
    if lbl == "broad":
        ins.set_ylim(0, ps_show.max() * 1.05)   # tight headroom: small gap between bars and title
    elif lbl == "moderate":
        ins.set_ylim(0, ps_show.max() * 1.20)
    else:
        ins.set_ylim(0, INSET_YLIM)
    if lbl == "sharp":
        # single bar at n_pos: trim the empty 4-5 tail, label every integer shown (1-3)
        ins.set_xlim(n_pos - 1.6, n_pos + 1.6)
        ins.set_xticks([n_pos - 1, n_pos, n_pos + 1])
        ins.set_xticklabels([str(n_pos - 1), str(n_pos), str(n_pos + 1)], fontsize=8.5 * FONT_SCALE)
        ins.tick_params(axis="x", pad=2.0)   # raise the labels a point toward the axis
    elif lbl == "moderate":
        # bars at 3-4 are ~0: start the axis at 5, label every integer 5..9
        ticks = [n for n in ns_show if n >= 5]
        ins.set_xlim(5 - 0.5, ns_show[-1] + INSET_PAD)
        ins.set_xticks(ticks)
        ins.set_xticklabels([str(n) for n in ticks], fontsize=8.5 * FONT_SCALE)
    else:  # broad: label the odd numbers so that 11 (the presented value) is shown
        ins.set_xlim(ns_show[0] - INSET_PAD, ns_show[-1] + INSET_PAD)
        ins.set_xticks(ns_show)
        ins.set_xticklabels([str(n) if n % 2 == 1 else "" for n in ns_show],
                            fontsize=8.5 * FONT_SCALE)
    ins.set_yticks([])
    title_txt = f"$y_0\\!=\\!{n_pos}$\n$H = {H_single[n_pos - 1]:.2f}$"
    if lbl == "broad":
        # title placed to the right of the bars (not above) so the whole
        # inset can sit above the E[H] curve without the title crossing it
        ins.text(1.08, 0.5, title_txt, transform=ins.transAxes,
                  fontsize=8 * FONT_SCALE, ha="left", va="center")
    else:
        ins.set_title(title_txt, fontsize=8 * FONT_SCALE, y=0.97)
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)

# Single label explaining what the three insets show, with a concave arrow to each
LABEL_ANCHOR = ((10.0 - XLIM[0]) / (XLIM[1] - XLIM[0]), 0.06)   # above the "10" tick
ARROW_COLOR = "0.35"
label_text = ax_ent.text(
    *LABEL_ANCHOR, "Readout distribution\nover integers",
    transform=ax_ent.transAxes, fontsize=9.5 * FONT_SCALE,
    ha="center", va="bottom", color=ARROW_COLOR, zorder=9,
)
fig.canvas.draw()   # so the label's rendered extent is known before placing arrows above it
label_top = ax_ent.transAxes.inverted().transform(
    label_text.get_window_extent(renderer=fig.canvas.get_renderer()))[1, 1]

PANEL_DIP = 1.0 / 12    # arrows sag this fraction of the panel height below the chord
ARROW_GAP = 1.0 / 40    # arrows start this far above the label's top, clear of the text
ARROW_START = (LABEL_ANCHOR[0], label_top + ARROW_GAP)
ARROW_TARGET_ADJUST = {   # (dx, dy) in units of the inset's own height h
    "sharp": (0.18, 0.08),
    "moderate": (0.16, 0.04),
    "broad": (0.0, 0.0),
}
for lbl, (x0, y0, w, h) in zip(inset_labels, inset_bounds):
    dx, dy = ARROW_TARGET_ADJUST[lbl]
    target = (x0 + w / 2 + dx * h, y0 - 0.32 * h + dy * h)
    control = ((ARROW_START[0] + target[0]) / 2,
               (ARROW_START[1] + target[1]) / 2 - PANEL_DIP)
    path = MplPath([ARROW_START, control, target],
                   [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax_ent.add_patch(mpatches.FancyArrowPatch(
        path=path, transform=ax_ent.transAxes,
        arrowstyle="->", color=ARROW_COLOR, lw=1.1,
        shrinkA=4, shrinkB=6, mutation_scale=14,
        zorder=3,   # under the E[H] curve/dots (zorder 7-8): arrows read as pointers, not clutter
    ))

ax_ent.set_xlim(*XLIM)
ax_ent.set_ylim(-0.03, H_max * 1.75)
ax_ent.set_ylabel("Entropy")
panel_label(ax_ent, "b")
ax_ent.set_xticks(integers_display)
ax_ent.set_xticklabels([str(n) for n in integers_display])
ax_ent.set_yticks([])

# ──────────────────────────────────────────────────────────────────────
#  Panel c — RT from fitted convergence time (160 ms)
# ──────────────────────────────────────────────────────────────────────
band_t0 = np.full_like(n_int, RT_T0)
band_steps = RT_TAU * ET_at_int
ax_rt.stackplot(n_int, band_t0, band_steps,
                colors=["0.92", C_STATIC], zorder=1)
RT_line = band_t0 + band_steps
ax_rt.plot(n_int, RT_line, color="0.10", lw=2.6, zorder=4)
for i in range(N_DISPLAY):
    ax_rt.plot(wells_a[i], RT_at_int[i], "o", color=well_color(i), ms=10,
               zorder=6, markeredgecolor="white", markeredgewidth=1.5)
handles = [
    mpl.patches.Patch(color="0.92", label="Basal latency  $t_0$"),
    mpl.patches.Patch(color=C_STATIC, label=r"Attractor time  $\tau\,E[T]$"),
]
ax_rt.legend(handles=handles, loc="upper left", framealpha=0.95, edgecolor="0.75")
ax_rt.text(0.90, 0.20, r"$\mathrm{RT} = t_0 + \tau\,E[T]$",
           transform=ax_rt.transAxes, ha="right", va="bottom",
           fontsize=10 * FONT_SCALE, color="0.15")
ax_rt.text(1.35, RT_at_int[0] + 0.38, "Fast, accurate", fontsize=9.5 * FONT_SCALE,
           color=PALETTE[2], fontweight="bold", ha="left")
ax_rt.text(14.55, RT_line.max() - 0.50, "Slow, approximate",
           fontsize=9.5 * FONT_SCALE, color=PALETTE[3], fontweight="bold", ha="right")

ax_rt.set_xlim(*XLIM)
ax_rt.set_ylim(1.05, RT_line.max() * 1.06)
ax_rt.set_ylabel("Predicted RT  (s)")
ax_rt.set_xlabel("Presented numerosity  $n$")
panel_label(ax_rt, "c")
ax_rt.set_xticks(integers_display)
ax_rt.set_xticklabels([str(n) for n in integers_display])

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(f"Saved {OUT_STEM}.{{png,pdf}}")
plt.close(fig)
