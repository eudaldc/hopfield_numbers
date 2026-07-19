"""Winner two-stage fit — predicted vs observed summaries (accuracy + RT).

Draws the fit we actually report: the winning power-law-width Gaussian attractor
(sigma_n = sigma_c * n^p) with the **two-stage** parameters — the landscape + lapse
fitted to the responses only (`pow_theta.csv`), and the hybrid RT link fitted on top
with the landscape held fixed (`pow_winner_rt.csv`, metric="hybrid") — overlaid on
the data, per exposure duration. Three rows:

  1. mean guess (lapse-free 'core' model mean vs in-support empirical mean),
  2. exact accuracy  P(r = n),
  3. geometric-mean RT  (hybrid link  t0 + alpha*E[T_conv] + beta*E[H]).

The RT row is therefore a cross-prediction of the response-fitted landscape, which
is the paper's logic (fit on responses, predict RT).

    .venv/bin/python -m scripts.trial_level_fit_figure
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
from scripts.plot_style import C_ATTR, apply_style, panel_label

# Same house style as fig1/fig3 (comparable ~15-16in canvas width, so point
# sizes end up the same physical size once the paper shrinks all three to
# \columnwidth), but this figure packs 5x3 small panels rather than 2-3 full
# width ones, so it uses a smaller font_scale tuned for that density.
FONT_SCALE = 1.5
apply_style(FONT_SCALE)

WINNER = T.WINNER_KEY                                 # gaussian_pow_attractor
THETA_CSV = T.LEGACY_OUT_DIR / "pow_theta.csv"              # response-only landscape + lapse
RT_CSV = T.LEGACY_OUT_DIR / "pow_winner_rt.csv"             # RT link fitted on the fixed landscape
OUT_FIG = T.LEGACY_OUT_DIR / "winner_fit_summaries.png"
OUT_FIG_PDF = T.LEGACY_OUT_DIR / "winner_fit_summaries.pdf"


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

    theta_df = pd.read_csv(THETA_CSV).set_index("duration_ms")
    rt_df = pd.read_csv(RT_CSV)
    rt_df = rt_df[(rt_df["model_key"] == WINNER)
                  & (rt_df["metric"] == "hybrid")].set_index("duration_ms")

    n = T.N_SHOWN
    fig, axes = plt.subplots(3, len(T.DURATIONS),
                             figsize=(3.0 * len(T.DURATIONS), 8.6), squeeze=False)
    r2s = {"mean": [], "exact": [], "rt": []}
    pooled = {k: {"obs": [], "pred": []} for k in r2s}

    for ci, d in enumerate(T.DURATIONS):
        row = theta_df.loc[d]
        theta = {p: float(row[p]) for p in spec.param_names}
        lapse = float(row["lapse"])
        # hybrid RT link fitted on this fixed landscape: alpha (steps), beta (entropy)
        rt_row = rt_df.loc[d]
        alpha, beta = float(rt_row["alpha"]), float(rt_row["beta"])

        grid = T.compute_grid(spec, theta, quad)
        pf = T.apply_lapse(grid["p_resp"], lapse, r_max)
        psum = T.predicted_summaries(pf)
        pred_exact = psum["pct_exact"].to_numpy()
        pred_core_mean = (grid["p_resp"] * T.WELLS[None, :]).sum(axis=1)  # lapse-free
        pred_rt = T.T0 + alpha * grid["ET"] + beta * grid["EH"]           # hybrid median

        data = data_by_dur[d]
        emp_mean, emp_exact, emp_rt = [], [], []
        for nn in n:
            sel = data.n_trial == nn
            g = data.r_trial[sel]
            lr = data.logrt_trial[sel]
            insup = g[g <= T.N_SUPPORT_MAX]
            emp_mean.append(np.mean(insup) if insup.size else np.nan)
            emp_exact.append(100.0 * np.mean(g == nn) if g.size else np.nan)
            emp_rt.append(np.exp(np.mean(lr)) if lr.size else np.nan)
        emp_mean = np.array(emp_mean); emp_exact = np.array(emp_exact)
        emp_rt = np.array(emp_rt)

        r2s["mean"].append(r2(emp_mean, pred_core_mean))
        r2s["exact"].append(r2(emp_exact, pred_exact))
        r2s["rt"].append(r2(emp_rt, pred_rt))
        for key, obs, pred in (("mean", emp_mean, pred_core_mean),
                               ("exact", emp_exact, pred_exact),
                               ("rt", emp_rt, pred_rt)):
            pooled[key]["obs"].append(obs)
            pooled[key]["pred"].append(pred)

        ax0, ax1, ax2 = axes[0][ci], axes[1][ci], axes[2][ci]
        ax0.plot([1, 15], [1, 15], color="0.8", lw=0.8, zorder=0)
        ax0.plot(n, emp_mean, "k.", ms=8)
        ax0.plot(n, pred_core_mean, color=C_ATTR, lw=1.8)
        ax1.plot(n, emp_exact, "k.", ms=8)
        ax1.plot(n, pred_exact, color=C_ATTR, lw=1.8)
        ax2.plot(n, emp_rt, "k.", ms=8)
        ax2.plot(n, pred_rt, color=C_ATTR, lw=1.8)

        ax0.set_title(f"{d} ms")
        for ax, key, hx, ha in ((ax0, "mean", 0.04, "left"),
                                 (ax1, "exact", 0.96, "right"),
                                 (ax2, "rt", 0.04, "left")):
            ax.text(hx, 0.93, f"R²={r2s[key][-1]:.2f}", transform=ax.transAxes,
                    fontsize=8.5 * FONT_SCALE, va="top", ha=ha)
        if ci == 0:
            ax0.set_ylabel("Mean guess")
            ax1.set_ylabel("% exact")
            ax2.set_ylabel("Mean RT (s)")
            panel_label(ax0, "a")
            panel_label(ax1, "b")
            panel_label(ax2, "c")
        ax2.set_xlabel("n shown")

    # legend proxies
    from matplotlib.lines import Line2D
    axes[0][0].legend(handles=[
        Line2D([], [], marker=".", color="k", ls="", ms=9, label="data"),
        Line2D([], [], color=C_ATTR, lw=2, label="model")],
        frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_FIG_PDF, dpi=300)
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)

    print(f"Saved {OUT_FIG}")
    for key, lab in (("mean", "mean guess"), ("exact", "% exact"), ("rt", "geomean RT")):
        vals = ", ".join(f"{d}:{v:.2f}" for d, v in zip(T.DURATIONS, r2s[key]))
        pool = r2(np.concatenate(pooled[key]["obs"]), np.concatenate(pooled[key]["pred"]))
        print(f"  R² {lab:11s} -> {vals}  | pooled {pool:.2f}")


if __name__ == "__main__":
    main()
