"""Winner joint co-fit — predicted vs observed summaries (accuracy + RT).

Draws the fit we actually report: the winning super-linear Gaussian attractor with
the **joint co-fit** parameters (`joint_refit_theta.csv`, landscape + lapse + RT
optimised together), overlaid on the data, per exposure duration. Three rows:

  1. mean guess (lapse-free 'core' model mean vs in-support empirical mean),
  2. exact accuracy  P(r = n),
  3. median RT  (hybrid link  t0 + alpha*E[T_conv] + beta*E[H]).

This is the accuracy counterpart the existing `predictive_summaries.png` lacks (that
one plots the older model set, not the winner) and uses the co-fit parameters rather
than the response-only fit, so it is faithful to "these fits".

    .venv/bin/python -m scripts.trial_level_cofit_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

WINNER = "gaussian_freebeta_attractor"
THETA_CSV = T.OUT_DIR / "joint_refit_theta.csv"      # co-fit params (fallback below)
OUT_FIG = T.OUT_DIR / "winner_cofit_summaries.png"


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def main() -> None:
    spec = T.MODEL_BY_KEY[WINNER]
    quad = T.make_quadrature(801, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    theta_df = pd.read_csv(THETA_CSV)
    theta_df = theta_df[theta_df["model_key"] == WINNER].set_index("duration_ms")

    n = T.N_SHOWN
    fig, axes = plt.subplots(3, len(T.DURATIONS),
                             figsize=(3.0 * len(T.DURATIONS), 8.6), squeeze=False)
    r2s = {"mean": [], "exact": [], "rt": []}

    for ci, d in enumerate(T.DURATIONS):
        row = theta_df.loc[d]
        theta = {p: float(row[f"theta_{p}"]) if f"theta_{p}" in row
                 else float(row[p]) for p in spec.param_names}
        lapse = float(row["lapse"])
        # hybrid RT params: rt0 = alpha (steps/E[T_conv]), rt1 = beta (entropy)
        alpha, beta = float(row["rt0"]), float(row["rt1"])

        grid = T.compute_grid(spec, theta, quad)
        pf = T.apply_lapse(grid["p_resp"], lapse, r_max)
        psum = T.predicted_summaries(pf)
        pred_exact = psum["pct_exact"].to_numpy()
        pred_core_mean = (grid["p_resp"] * T.WELLS[None, :]).sum(axis=1)  # lapse-free
        pred_rt = T.T0 + alpha * grid["ET"] + beta * grid["EH"]           # hybrid median

        # empirical. RT is compared against the geometric mean (= the log-normal
        # RT MLE target, exp(mean log rt)); the median is kept faint for reference
        # (log RT is right-skewed, so median < geometric mean by ~0.1-0.2 s).
        data = data_by_dur[d]
        emp_mean, emp_exact, emp_rt, emp_rt_med = [], [], [], []
        for nn in n:
            sel = data.n_trial == nn
            g = data.r_trial[sel]
            lr = data.logrt_trial[sel]
            insup = g[g <= T.N_SUPPORT_MAX]
            emp_mean.append(np.mean(insup) if insup.size else np.nan)
            emp_exact.append(100.0 * np.mean(g == nn) if g.size else np.nan)
            emp_rt.append(np.exp(np.mean(lr)) if lr.size else np.nan)        # geom. mean
            emp_rt_med.append(np.median(np.exp(lr)) if lr.size else np.nan)  # median (ref)
        emp_mean = np.array(emp_mean); emp_exact = np.array(emp_exact)
        emp_rt = np.array(emp_rt); emp_rt_med = np.array(emp_rt_med)

        r2s["mean"].append(r2(emp_mean, pred_core_mean))
        r2s["exact"].append(r2(emp_exact, pred_exact))
        r2s["rt"].append(r2(emp_rt, pred_rt))

        ax0, ax1, ax2 = axes[0][ci], axes[1][ci], axes[2][ci]
        ax0.plot([1, 15], [1, 15], color="0.8", lw=0.8, zorder=0)
        ax0.plot(n, emp_mean, "k.", ms=8)
        ax0.plot(n, pred_core_mean, color="#c0392b", lw=1.8)
        ax1.plot(n, emp_exact, "k.", ms=8)
        ax1.plot(n, pred_exact, color="#c0392b", lw=1.8)
        ax2.plot(n, emp_rt_med, ".", color="0.72", ms=5)
        ax2.plot(n, emp_rt, "k.", ms=8)
        ax2.plot(n, pred_rt, color="#2c7fb8", lw=1.8)

        ax0.set_title(f"{d} ms")
        for ax, key in ((ax0, "mean"), (ax1, "exact"), (ax2, "rt")):
            ax.text(0.04, 0.93, f"R²={r2s[key][-1]:.2f}", transform=ax.transAxes,
                    fontsize=8, va="top")
        if ci == 0:
            ax0.set_ylabel("mean guess (core)")
            ax1.set_ylabel("% exact  P(r=n)")
            ax2.set_ylabel("geom.-mean RT (s)")
        ax2.set_xlabel("n shown")

    # legend proxies
    from matplotlib.lines import Line2D
    axes[0][0].legend(handles=[
        Line2D([], [], marker=".", color="k", ls="", ms=9, label="data"),
        Line2D([], [], color="#c0392b", lw=2, label="winner (co-fit)")],
        fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("Winner joint co-fit — predicted (lines) vs data (dots; RT = geometric "
                 "mean, median faint).  "
                 f"Pooled R²: mean {np.nanmean(r2s['mean']):.2f}, "
                 f"exact {np.nanmean(r2s['exact']):.2f}, RT {np.nanmean(r2s['rt']):.2f}",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=190)
    plt.close(fig)

    print(f"Saved {OUT_FIG}")
    for key, lab in (("mean", "mean guess"), ("exact", "% exact"), ("rt", "median RT")):
        vals = ", ".join(f"{d}:{v:.2f}" for d, v in zip(T.DURATIONS, r2s[key]))
        print(f"  R² {lab:11s} -> {vals}")


if __name__ == "__main__":
    main()
