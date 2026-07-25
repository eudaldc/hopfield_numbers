#!/usr/bin/env python
"""Figure 1: conceptual figure of the winning super-linear Gaussian attractor.

Three stacked panels, 160 ms fitted parameters:
  a — energy landscape: deep, isolated wells for small n and broader,
      overlapping components at larger n. Well widths are drawn to scale,
      floored at 0.1 for visibility
      (the n = 1 well is 0.02 wide); depths are the true well depths
      -log sigma_n, floored at 0.05 so broad wells remain visible. Lattice
      n = 1..19 as in the fitted model.
  b — attractor clean-up map: initial estimate y0 versus the bounded final
      state x*(y0) under the fitted dynamics (black), with the matched static
      readout x*=y0 as a light-grey diagonal. Insets show p(r|x*) at
      y0 = 2, 6, 13. The map uses the fitted lattice (n = 1..19) and is not
      smoothed across basin discontinuities.
  c — selected RT link RT = t0 + tau E[T], stacked basal latency and
      convergence time. Expected settling time uses an extended lattice
      (n = 1..30) so finite-support truncation does not bend E[T] near n = 15.

    .venv/bin/python -m scripts.figure1_conceptual_attractor_wells
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_style import C_STATIC, PALETTE, apply_style, panel_label  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT_STEM = ROOT / "figures" / "figure1_conceptual_attractor_wells"

# ── Project house style ───────────────────────────────────────────────
FONT_SCALE = 2.1
apply_style(FONT_SCALE)

N_DISPLAY = 15
N_LATTICE_MODEL = 19    # panels a/b: the fitted model's lattice (stimuli 1..15 + padding)
N_LATTICE_ILLUS = 30    # panel c: extended lattice, avoids edge truncation in E[T]
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

# ── Clean-up dynamics (same clamp / tolerance / G=120 budget as T._cleanup) ──


def posterior_of(xv: np.ndarray, wells: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    logp = -np.log(sigma)[None, :] - (xv[:, None] - wells[None, :]) ** 2 / (
        2.0 * (sigma ** 2)[None, :])
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def cleanup(y0: np.ndarray, wells: np.ndarray, sigma: np.ndarray,
            eta: float = 1.0, max_steps: int = 120,
            tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Clamped gradient descent on the winner landscape. Returns (x_star, n_steps)."""
    xv = y0.copy()
    max_step = 0.5 * float(sigma.min())
    prev = np.zeros_like(xv)
    active = np.ones(xv.shape, dtype=bool)
    n_steps = np.full(xv.shape, float(max_steps))
    for t in range(max_steps):
        diff = xv[:, None] - wells[None, :]
        logw = -np.log(sigma)[None, :] - diff ** 2 / (2.0 * (sigma ** 2)[None, :])
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        pull = (w * diff / (sigma ** 2)[None, :]).sum(axis=1)
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


# Panel b: deterministic clean-up map on the fitted lattice (no sensory averaging)
n_int = wells_a[:N_DISPLAY]
y0_map = np.linspace(XLIM[0], XLIM[1], 8000)
x_star_map, _ = cleanup(y0_map.copy(), wells_a, sigma_a)
x_star_at_int, _ = cleanup(n_int.copy(), wells_a, sigma_a)

# Panel c: expected settling time over the sensory channel at the fitted 160 ms
# parameters (extended lattice so E[T] is not bent by edge truncation near n=15)
S00 = 0.05
S0_D = float(theta.loc[SHOW_DURATION, "s0"])
LAMBDA0_D = float(theta.loc[SHOW_DURATION, "lambda0"])


def sensory_sd(nu: np.ndarray) -> np.ndarray:
    return np.sqrt(S00 ** 2 + S0_D * np.power(nu, LAMBDA0_D))


x_rt = np.linspace(-2.0, 33.0, 14000)
_, steps_map = cleanup(x_rt.copy(), wells_b, sigma_b)

T_NODES = np.linspace(-4.0, 4.0, 81)
W_NODES = np.exp(-0.5 * T_NODES ** 2)
W_NODES /= W_NODES.sum()


def expected_of(nu_grid: np.ndarray, value_map: np.ndarray) -> np.ndarray:
    sd = sensory_sd(nu_grid)
    y = nu_grid[:, None] + sd[:, None] * T_NODES[None, :]
    vy = np.interp(y.ravel(), x_rt, value_map).reshape(y.shape)
    return (vy * W_NODES[None, :]).sum(axis=1)


ET_at_int = expected_of(n_int, steps_map)

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
    0.2, 0.05, "Deep, isolated wells:\nnear-exact readout",
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
# OLD: ax_land.set_ylabel("Energy")
ax_land.set_ylabel("Schematic energy")
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
# OLD: ax_fq.set_ylabel("Number-word frequency", color=FREQ_COLOR, labelpad=10)
ax_fq.set_ylabel("Illustrative frequency  ($1/n$)", color=FREQ_COLOR, labelpad=10)
ax_fq.tick_params(axis="y", colors=FREQ_COLOR)
ax_fq.set_ylim(-num_freq[0] * 0.08, num_freq[0] * 1.15)   # bottom pad matched to ax_land's
ax_fq.spines["right"].set_visible(True)
ax_fq.spines["right"].set_color(FREQ_COLOR)
ax_fq.spines["right"].set_alpha(0.5)

# ──────────────────────────────────────────────────────────────────────
#  Panel b — Attractor clean-up map y0 ↦ x*
# ──────────────────────────────────────────────────────────────────────
ax_map = ax_ent  # keep the middle-row axis handle name used by the layout

# Matched static readout reference, then the bounded attractor map (unsmoothed)
ax_map.plot(XLIM, XLIM, "-", color="0.78", lw=2.0, zorder=2)
ax_map.plot(y0_map, x_star_map, "-", color="0.10", lw=2.6, zorder=7)

for i in range(N_DISPLAY):
    ax_map.plot(wells_a[i], x_star_at_int[i], "o", color=well_color(i), ms=10,
                zorder=8, markeredgecolor="white", markeredgewidth=1.5)

# Insets: example posteriors after clean-up (sharp / moderate / broad)
INSET_HALF = 3
INSET_PAD = 0.8
inset_ns = [2, 6, 13]
inset_labels = ["sharp", "moderate", "broad"]
_ins_w, _ins_h = 0.16, 0.26
_x13 = (13.0 - XLIM[0]) / (XLIM[1] - XLIM[0])
inset_bounds = [
    [0.028, 0.392, 0.15, 0.36],                # sharp (y0=2)
    [0.325, 0.68, 0.15, 0.30],                 # moderate (y0=6)
    [_x13 - _ins_w / 2, 0.30, _ins_w, _ins_h],  # broad (y0=13): lower third, clear of the axis
]
y0_inset = np.asarray(inset_ns, float)
x_star_inset, _ = cleanup(y0_inset.copy(), wells_a, sigma_a)
post_at = posterior_of(x_star_inset, wells_a, sigma_a)
global_max_p = 0.0
for i_ins, n_pos in enumerate(inset_ns):
    m = (wells_a >= n_pos - INSET_HALF) & (wells_a <= n_pos + INSET_HALF)
    global_max_p = max(global_max_p, post_at[i_ins, m].max())
INSET_YLIM = global_max_p * 1.10

for i_ins, (n_pos, lbl, bounds) in enumerate(zip(inset_ns, inset_labels, inset_bounds)):
    ins = ax_map.inset_axes(bounds)
    ins.set_facecolor("none")
    post_n = post_at[i_ins, :]
    mask = (wells_a >= n_pos - INSET_HALF) & (wells_a <= n_pos + INSET_HALF)
    ns_show = wells_a[mask].astype(int)
    ps_show = post_n[mask]
    colors_show = [well_color(j) for j in np.where(mask)[0]]
    ins.bar(ns_show, ps_show, color=colors_show, edgecolor="white",
            linewidth=0.4, width=0.7)
    if lbl == "broad":
        ins.set_ylim(0, ps_show.max() * 1.20)   # same title–bar gap as the other insets
    elif lbl == "moderate":
        ins.set_ylim(0, ps_show.max() * 1.20)
    else:
        ins.set_ylim(0, INSET_YLIM)
    if lbl == "sharp":
        ins.set_xlim(n_pos - 1.6, n_pos + 1.6)
        ins.set_xticks([n_pos - 1, n_pos, n_pos + 1])
        ins.set_xticklabels([str(n_pos - 1), str(n_pos), str(n_pos + 1)],
                            fontsize=8.5 * FONT_SCALE)
        ins.tick_params(axis="x", pad=2.0)
    elif lbl == "moderate":
        ticks = [n for n in ns_show if n >= 5]
        ins.set_xlim(5 - 0.5, ns_show[-1] + INSET_PAD)
        ins.set_xticks(ticks)
        ins.set_xticklabels([str(n) for n in ticks], fontsize=8.5 * FONT_SCALE)
    else:
        ins.set_xlim(ns_show[0] - INSET_PAD, ns_show[-1] + INSET_PAD)
        ins.set_xticks(ns_show)
        ins.set_xticklabels([str(n) if n % 2 == 1 else "" for n in ns_show],
                            fontsize=8.5 * FONT_SCALE)
    ins.set_yticks([])
    ins.set_title(f"$y_0\\!=\\!{n_pos}$", fontsize=8 * FONT_SCALE, y=0.97)
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)

# Static-readout label: right of centre, above the y0=6 inset
ax_map.text(
    0.68, 0.86,
    "Static readout\nreference",
    transform=ax_map.transAxes, fontsize=9.5 * FONT_SCALE,
    ha="center", va="center", color="0.45", zorder=9,
)

# Clean-up readout label: slightly right of centre, above the bottom margin
ax_map.text(
    0.55, 0.20,
    "Readout distribution\nover integers",
    transform=ax_map.transAxes, fontsize=9.5 * FONT_SCALE,
    ha="center", va="bottom", color="black", zorder=9,
)

ax_map.set_xlim(*XLIM)
ax_map.set_ylim(*XLIM)
ax_map.set_xlabel(r"Initial estimate  $y_0$")
ax_map.set_ylabel(r"Final state  $x^*$")
panel_label(ax_map, "b")
ax_map.set_xticks(integers_display)
ax_map.set_xticklabels([str(n) for n in integers_display])
ax_map.set_yticks(integers_display)
ax_map.set_yticklabels([str(n) for n in integers_display])

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
# Anchor the legend so its left edge sits at n=1
_leg_x = (1.0 - XLIM[0]) / (XLIM[1] - XLIM[0])
ax_rt.legend(handles=handles, loc="upper left",
             bbox_to_anchor=(_leg_x, 1.0),
             framealpha=0.95, edgecolor="0.75")
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
