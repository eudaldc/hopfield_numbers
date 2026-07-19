"""Pure driven model (the 'song completion' idea) + trajectory figure.

Pure driven = NO completed-estimate y0 and NO sigma(n) sensory channel. Items
are registered over the exposure (a count K(t) climbing 0 -> n), and the
attractor is driven by this running count while the landscape pulls the state
toward stored integers:

    tau_x dx/dt = -dV/dx + beta (K(t) - x),   gated until the first item.

The only noise is the registration process (item latencies). The state is read
out at offset. This is the literal 'partial cue progressively retrieves the
stored pattern' model.

Part 1: its fit vs M0 (uses the round-1 fitted pure-driven model, M2_tether).
Part 2: trajectories on the WINNER's (sharp) landscape, showing that driving the
        attractor with a partial count captures the wrong (small) integer early
        -- the melody disanalogy: a partial count is not diagnostic of the total.

    .venv/bin/python driven_attractor/pure_driven_trajectories.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D, model as M  # noqa: E402
import driven_attractor.driven_sibling as S  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402
from scripts.plot_style import apply_style, panel_label  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
FIGURE_STEM = ROOT / "figures" / "supfigure1_pure_driven_trajectories"

# Shared winner landscape with the 320 ms sensory channel for the illustration.
WIN = S.m0_params("shared")[320]
SIG = M.sigma_wells(WIN["sigma_c"], WIN["p"])
D_ILLUS = 0.32          # 320 ms
RHO = 30.0              # item registration rate (items/s): ~1 every 33 ms
BETA_ILLUS = 120.0      # moderate coupling: landscape competes with the drive
TAU_ILLUS = 0.12


def fit_number():
    """Pure driven (round-1 M2_tether) response logL vs M0, from cached fits."""
    dat = D.load(); r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)
    # M0 references
    ladder = ROOT / "output/trial_level_model_comparison/shared_landscape_ladder.csv"
    import pandas as pd
    L = pd.read_csv(ladder).set_index("model_key")
    m0_shared = float(L.loc["gaussian_pow_attractor", "shared_logL"])
    try:
        pd_fit = json.load(open(OUT / "fits" / "M2_tether_g0.json"))
        pure = pd_fit["logL"]
        print("=== Pure driven model fit (round-1 M2_tether, K(t)-driven, no sigma channel) ===")
        print(f"  pure driven logL       = {pure:.1f}")
        print(f"  M0 shared-landscape    = {m0_shared:.1f}   (pure driven is {pure-m0_shared:+.0f})")
    except FileNotFoundError:
        print("(round-1 M2_tether_g0.json not found; skipping fit number)")


def traj_one(n, n_paths, dt=M.DT_EXPO):
    """Directly integrate the pure-driven dynamics to D_ILLUS for one n.
    Returns (tgrid_ms, x_t (steps+1, paths), arrivals_ms (paths, list))."""
    E = M.exp_draws(4096)
    arr = M.arrival_times(n, RHO, 0.0, E)[:n_paths]        # (paths, N_MAX)
    n_steps = int(round(D_ILLUS / dt))
    x = np.zeros(n_paths)
    xs = [x.copy()]
    Ks = [np.zeros(n_paths)]
    for s in range(1, n_steps + 1):
        t = s * dt
        K = (arr <= t).sum(axis=1).astype(float)          # running count (paths,)
        gate = K > 0
        x = M._guarded_step(x, K, SIG, BETA_ILLUS, dt / TAU_ILLUS,
                            attractor=True, gate=gate)
        x = np.where(gate, x, 0.0)
        xs.append(x.copy())
        Ks.append(K.copy())
    tgrid = np.arange(n_steps + 1) * dt * 1000
    return tgrid, np.asarray(xs), np.asarray(Ks), arr[:, :n] * 1000


def trajectories():
    apply_style(1.6)
    ns = [2, 4, 8, 12]
    labels = ["a", "b", "c", "d"]
    n_paths = 5
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, n, lab in zip(axes.flat, ns, labels):
        tgrid, xt, Kt, arr_ms = traj_one(n, n_paths)
        for w in range(1, 16):
            ax.axhline(w, color="0.92", lw=0.6, zorder=0)
        ax.axhline(n, color="black", lw=1.3, ls="--", zorder=1, label="true $n$")
        ax.step(tgrid, Kt[:, 0], where="post", color="0.55", lw=1.1, ls=":",
                zorder=1, label="count $K(t)$")
        for p in range(n_paths):
            ax.plot(tgrid, xt[:, p], lw=1.3, alpha=0.85)
            a = arr_ms[p][np.isfinite(arr_ms[p])]
            a = a[a <= tgrid[-1]]
            ax.plot(a, np.interp(a, tgrid, xt[:, p]), "k.", ms=4.5, zorder=5)
        ax.text(0.97, 0.94, f"$n = {n}$", transform=ax.transAxes, ha="right",
                va="top", fontsize=plt.rcParams["axes.titlesize"])
        ax.set_ylim(0, 15.5)
        panel_label(ax, lab)
        if lab in ("a", "c"):
            ax.set_ylabel("attractor state $x$")
        if lab in ("c", "d"):
            ax.set_xlabel("time from onset (ms)")
    axes[0, 0].legend(loc="upper left", frameon=False,
                      fontsize=plt.rcParams["legend.fontsize"])
    fig.tight_layout()
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nsaved {FIGURE_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    fit_number()
    trajectories()
