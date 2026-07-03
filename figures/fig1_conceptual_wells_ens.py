#!/usr/bin/env python
"""Figure 1 (revision): conceptual figure of the winning super-linear Gaussian attractor.

Two stacked panels sharing the number-line x-axis (n = 1..15), 160 ms fitted
parameters:
  a — energy landscape: deep, isolated wells for small n; shallow, merged wells
      beyond n_crit. Well widths drawn to scale, floored at 0.1 for visibility
      (the n = 1 well is 0.02 wide); depths are the true well depths
      -log sigma_n, floored at 0.05 so merged wells remain visible. Lattice
      n = 1..19 as in the fitted model.
  b — posterior entropy BEFORE clean-up (grey: the raw readout of a position x,
      with high ridges between the narrow small-n wells, where a flat readout
      pushes mass onto the broad distant wells) and AFTER clean-up (black: the
      attractor resolves those ambiguous positions, quantising the line into
      basins; the residual entropy staircase is what drives RT). For this
      illustration the lattice is extended to n = 30 so that the finite-lattice
      truncation does not bend the curve down near n = 15 (fits use n <= 19
      with the contamination channel absorbing larger responses).

    .venv/bin/python figures/fig1_conceptual_wells_ens.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
THETA_CSV = ROOT / "output" / "trial_level_model_comparison" / "winner_theta.csv"
OUT_STEM = ROOT / "figures" / "conceptual_attractor_wells_ens"

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
    "legend.fontsize": 8.5 * FONT_SCALE,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}
mpl.rcParams.update(STYLE)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

N_DISPLAY = 15
N_LATTICE_MODEL = 19    # panel a: the fitted model's lattice (stimuli 1..15 + padding)
N_LATTICE_ILLUS = 30    # panel b: extended lattice, avoids edge truncation in H
SHOW_DURATION = 160

theta = pd.read_csv(THETA_CSV).set_index("duration_ms")
SIGMA_C = float(theta.loc[SHOW_DURATION, "sigma_c"])
BETA = float(theta.loc[SHOW_DURATION, "beta"])


def widths(wells: np.ndarray) -> np.ndarray:
    freq = np.log10(1.0 + 1.0 / wells)
    return (SIGMA_C / freq) ** BETA


def ncrit() -> float:
    ns = np.linspace(1.0, 40.0, 40000)
    s = widths(ns)
    i = int(np.argmax(s >= 1.0))
    return float(ns[i - 1] + (1.0 - s[i - 1]) * (ns[i] - ns[i - 1]) / (s[i] - s[i - 1]))


NCRIT_SHOW = ncrit()

wells_a = np.arange(1, N_LATTICE_MODEL + 1, dtype=float)
sigma_a = widths(wells_a)
wells_b = np.arange(1, N_LATTICE_ILLUS + 1, dtype=float)
sigma_b = widths(wells_b)

benford_freq = np.log10(1.0 + 1.0 / wells_a)
benford_freq_norm = benford_freq / benford_freq.max()
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


def cleanup(y0: np.ndarray, eta: float = 1.0, max_steps: int = 100,
            tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Clamped gradient descent on the winner landscape, with the same step
    clamp, tolerance, oscillation guard and 100-step budget as the fitted
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


# Behaviourally relevant curves: EXPECTED entropy over the sensory channel
# y0 ~ N(nu, sd(nu)) (160 ms fitted sensory parameters) — the deterministic
# basin map alone would be a step function (basin boundaries are razor-sharp),
# but what drives behaviour/RT is E[H], averaged over the sensory noise.
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

# Hybrid RT channel: RT = t0 + alpha * E[T] + beta_H * E[H]  (fitted, 160 ms)
rt_theta = pd.read_csv(ROOT / "output" / "trial_level_model_comparison" /
                       "joint_refit_theta.csv")
rt_row = rt_theta[(rt_theta.model_key == "gaussian_freebeta_attractor") &
                  (rt_theta.duration_ms == SHOW_DURATION)].iloc[0]
RT_T0 = 1.30
RT_ALPHA = float(rt_row["rt0"])
RT_BETA = float(rt_row["rt1"])
RT_at_int = RT_T0 + RT_ALPHA * ET_at_int + RT_BETA * H_at_int

cmap_wells = mpl.colormaps["plasma"]
norm_c = Normalize(vmin=-0.10, vmax=1.15)


def well_color(i: int) -> tuple:
    return cmap_wells(norm_c(benford_freq_norm[i]))


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

ax_land.axvline(NCRIT_SHOW, color="0.25", lw=1.6, ls=":", zorder=3)
ax_land.text(
    NCRIT_SHOW + 0.15, -3.35, r"$n_{\rm crit}$  ($\sigma_n = 1$)",
    fontsize=9.5 * FONT_SCALE, color="0.25", ha="left", va="bottom",
)

ax_land.annotate(
    "Deep, isolated wells:\nexact readout",
    xy=(0.10, 0.34), xycoords="axes fraction",
    xytext=(0.22, 0.10), textcoords="axes fraction",
    fontsize=10 * FONT_SCALE, color="0.15", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.5,
                    connectionstyle="arc3,rad=0.30"),
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)
ax_land.annotate(
    "Shallow, merged wells:\napproximate readout",
    xy=(0.90, 0.78), xycoords="axes fraction",
    xytext=(0.70, 0.44), textcoords="axes fraction",
    fontsize=10 * FONT_SCALE, color="0.15", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.5,
                    connectionstyle="arc3,rad=-0.25"),
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
)

ax_land.set_xlim(*XLIM)
ax_land.set_ylim(-4.6, 0.55)
ax_land.set_xlabel("Evidence axis  $x$  (the number line a trial's state moves on)")
ax_land.set_ylabel("Energy  $E(x)$")
ax_land.set_title("a", fontweight="bold", loc="left", x=0.015)
ax_land.set_xticks(integers_display)
ax_land.set_xticklabels([str(n) for n in integers_display])
ax_land.set_yticks([])

# Secondary y-axis: number frequency (drives the well widths)
FREQ_COLOR = "#CC6677"
ax_fq = ax_land.twinx()
ax_fq.plot(
    integers_display, benford_freq[:N_DISPLAY], "-o",
    color=FREQ_COLOR, ms=5, lw=1.8, alpha=0.85,
    markeredgecolor="white", markeredgewidth=0.8, zorder=7,
)
ax_fq.set_ylabel(r"Number frequency  $f_n$", color=FREQ_COLOR, labelpad=10)
ax_fq.tick_params(axis="y", colors=FREQ_COLOR)
ax_fq.set_ylim(0, benford_freq[0] * 1.15)
ax_fq.spines["right"].set_visible(True)
ax_fq.spines["right"].set_color(FREQ_COLOR)
ax_fq.spines["right"].set_alpha(0.5)

# ──────────────────────────────────────────────────────────────────────
#  Panel b — Posterior entropy before vs after clean-up
# ──────────────────────────────────────────────────────────────────────
ax_ent.plot(n_int, H_pre_i, "-", color="0.68", lw=1.8, zorder=2)
ax_ent.plot(n_int, H_at_int, "-", color="0.10", lw=2.6, zorder=3)

for i in range(N_DISPLAY):
    ax_ent.axvline(wells_a[i], color="0.92", lw=0.3, zorder=0)
    ax_ent.plot(wells_a[i], H_at_int[i], "o", color=well_color(i), ms=10,
                zorder=6, markeredgecolor="white", markeredgewidth=1.5)

ax_ent.axvline(NCRIT_SHOW, color="0.25", lw=1.6, ls=":", zorder=1)

H_max = float(np.max(H_pre_i))

# inline labels for the two curves
ax_ent.annotate(
    "before clean-up",
    xy=(3.0, H_pre_i[2]), xytext=(3.4, H_max * 1.10),
    fontsize=9.5 * FONT_SCALE, color="0.45", ha="center", fontstyle="italic",
    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.4,
                    connectionstyle="arc3,rad=-0.2"),
    zorder=8,
)
ax_ent.annotate(
    "after clean-up",
    xy=(11.0, H_at_int[10]),
    xytext=(11.6, H_max * 0.52),
    fontsize=9.5 * FONT_SCALE, color="0.10", ha="center", fontstyle="italic",
    arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.4,
                    connectionstyle="arc3,rad=0.25"),
    zorder=8,
)

# twin axis: the attractor's convergence time (the second RT ingredient)
STEPS_COLOR = "#4C72B0"
ax_st = ax_ent.twinx()
ax_st.plot(n_int, ET_at_int, "-", color=STEPS_COLOR, lw=2.4, alpha=0.9, zorder=2)
ax_st.set_ylabel("Convergence steps  $E[T]$", color=STEPS_COLOR, labelpad=10)
ax_st.tick_params(axis="y", colors=STEPS_COLOR)
ax_st.set_ylim(0, 215)
ax_st.spines["right"].set_visible(True)
ax_st.spines["right"].set_color(STEPS_COLOR)
ax_st.spines["right"].set_alpha(0.5)
ax_st.annotate(
    "clean-up steps",
    xy=(6.0, ET_at_int[5]), xytext=(7.9, 52),
    fontsize=9.5 * FONT_SCALE, color=STEPS_COLOR, ha="center", fontstyle="italic",
    arrowprops=dict(arrowstyle="-|>", color=STEPS_COLOR, lw=1.4,
                    connectionstyle="arc3,rad=0.25"),
    zorder=8,
)

# Insets: example posteriors after clean-up (sharp / moderate / broad)
INSET_HALF = 3
INSET_PAD = 0.8
inset_ns = [2, 6, 12]
inset_labels = ["sharp", "moderate", "broad"]
inset_bounds = [
    [0.028, 0.50, 0.15, 0.36],
    [0.40, 0.58, 0.15, 0.36],
    [0.78, 0.62, 0.15, 0.36],
]
post_at = posterior_of(mu_star)
global_max_p = 0.0
for n_pos in inset_ns:
    m = (wells_b >= n_pos - INSET_HALF) & (wells_b <= n_pos + INSET_HALF)
    global_max_p = max(global_max_p, post_at[n_pos - 1, m].max())
INSET_YLIM = global_max_p * 1.10

for n_pos, lbl, bounds in zip(inset_ns, inset_labels, inset_bounds):
    ins = ax_ent.inset_axes(bounds)
    ins.set_facecolor("white")
    post_n = post_at[n_pos - 1, :]
    mask = (wells_b >= n_pos - INSET_HALF) & (wells_b <= n_pos + INSET_HALF)
    ns_show = wells_b[mask].astype(int)
    ps_show = post_n[mask]
    colors_show = [well_color(min(j, N_LATTICE_MODEL - 1)) for j in np.where(mask)[0]]
    ins.bar(ns_show, ps_show, color=colors_show, edgecolor="white",
            linewidth=0.4, width=0.7)
    ins.set_xlim(ns_show[0] - INSET_PAD, ns_show[-1] + INSET_PAD)
    ins.set_ylim(0, INSET_YLIM)
    ins.set_xticks(ns_show)
    step_lbl = ns_show[::2]
    ins.set_xticklabels([str(n) if n in step_lbl else "" for n in ns_show],
                        fontsize=8.5 * FONT_SCALE)
    ins.set_yticks([])
    ins.set_title(f"$p(n \\mid x^*)$, $y_0\\!=\\!{n_pos}$ — {lbl}\n"
                  f"$H = {H_single[n_pos - 1]:.2f}$",
                  fontsize=8 * FONT_SCALE, y=0.97)
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)

ax_ent.set_xlim(*XLIM)
ax_ent.set_ylim(-0.03, H_max * 1.75)
ax_ent.set_ylabel("Expected readout entropy  $E[H]$")
ax_ent.set_title("b", fontweight="bold", loc="left", x=0.015)
ax_ent.set_xticks(integers_display)
ax_ent.set_xticklabels([str(n) for n in integers_display])
ax_ent.set_yticks([])

# ──────────────────────────────────────────────────────────────────────
#  Panel c — RT as the sum of the two mechanism signals (fitted, 160 ms)
# ──────────────────────────────────────────────────────────────────────
band_t0 = np.full_like(n_int, RT_T0)
band_steps = RT_ALPHA * ET_at_int
band_H = RT_BETA * H_at_int
ax_rt.stackplot(n_int, band_t0, band_steps, band_H,
                colors=["0.92", "#A3C0DE", "#C9B8D8"], zorder=1)
RT_line = band_t0 + band_steps + band_H
ax_rt.plot(n_int, RT_line, color="0.10", lw=2.6, zorder=4)
for i in range(N_DISPLAY):
    ax_rt.plot(wells_a[i], RT_at_int[i], "o", color=well_color(i), ms=10,
               zorder=6, markeredgecolor="white", markeredgewidth=1.5)
ax_rt.axvline(NCRIT_SHOW, color="0.25", lw=1.6, ls=":", zorder=3)

handles = [
    mpl.patches.Patch(color="0.92", label="basal latency  $t_0$"),
    mpl.patches.Patch(color="#A3C0DE", label=r"attractor time  $\alpha\,E[T]$"),
    mpl.patches.Patch(color="#C9B8D8", label=r"readout uncertainty  $\beta_H\,E[H]$"),
]
ax_rt.legend(handles=handles, loc="upper left", framealpha=0.95, edgecolor="0.75")
ax_rt.text(0.985, 0.06, r"$\mathrm{RT} = t_0 + \alpha\,E[T] + \beta_H\,E[H]$",
           transform=ax_rt.transAxes, ha="right", va="bottom",
           fontsize=10 * FONT_SCALE, color="0.15")
ax_rt.text(1.7, RT_at_int[0] + 0.22, "fast, exact", fontsize=9.5 * FONT_SCALE,
           color=PALETTE[2], fontweight="bold", ha="left")
ax_rt.text(14.9, RT_line.max() - 0.30, "slow, approximate",
           fontsize=9.5 * FONT_SCALE, color=PALETTE[3], fontweight="bold", ha="right")

ax_rt.set_xlim(*XLIM)
ax_rt.set_ylim(1.05, RT_line.max() * 1.06)
ax_rt.set_ylabel("Predicted RT  (s)")
ax_rt.set_xlabel("Presented numerosity  $n$  (one clean-up episode per trial)")
ax_rt.set_title("c", fontweight="bold", loc="left", x=0.015)
ax_rt.set_xticks(integers_display)
ax_rt.set_xticklabels([str(n) for n in integers_display])

fig.savefig(f"{OUT_STEM}.png")
fig.savefig(f"{OUT_STEM}.pdf")
print(f"Saved {OUT_STEM}.{{png,pdf}}")
plt.close(fig)
