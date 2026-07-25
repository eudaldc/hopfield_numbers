"""Winner joint fit — predicted vs observed summaries (accuracy + RT).

Same layout as ``trial_level_fit_figure.py``, but uses the primary joint fit:
the power-law-width Gaussian attractor landscape shared across all five
exposure durations (``shared_landscape_ladder_theta.csv``), with only the
sensory channel and lapse free per duration, and the selected convergence-time
RT mapping shared across durations (``rt_joint.csv``).

    .venv/bin/python -m scripts.trial_level_cofit_figure
"""

from __future__ import annotations

import argparse
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

FONT_SCALE = 1.5
apply_style(FONT_SCALE)

WINNER = T.WINNER_KEY


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, choices=(100, 120), default=120)
    args = parser.parse_args(argv)
    tag = "_G100" if args.budget == 100 else ""
    theta_csv = T.OUT_DIR / f"shared_landscape_ladder_theta{tag}.csv"
    rt_csv = T.OUT_DIR / f"rt_joint{tag}.csv"
    out_stem = (
        _ROOT / "figures" / "figure3_joint_model_fit"
        if args.budget == 120
        else T.OUT_DIR / "winner_cofit_summaries_G100"
    )
    out_fig = out_stem.with_suffix(".png")
    out_fig_pdf = out_stem.with_suffix(".pdf")
    T.MAX_STEPS = args.budget

    spec = T.MODEL_BY_KEY[WINNER]
    quad = T.make_quadrature(801, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)

    theta_df = pd.read_csv(theta_csv)
    theta_df = theta_df[theta_df["model_key"] == WINNER].set_index("duration_ms")
    rt_table = pd.read_csv(rt_csv)
    rt_row = rt_table.loc[rt_table["AIC"].idxmin()]
    rt_metric = str(rt_row["metric"])
    rt_scale = np.array([float(value) for value in str(rt_row["scale"]).split(",")])

    n = T.N_SHOWN
    fig, axes = plt.subplots(3, len(T.DURATIONS),
                             figsize=(3.0 * len(T.DURATIONS), 8.6), squeeze=False)
    r2s = {"mean": [], "exact": [], "rt": []}
    pooled = {k: {"obs": [], "pred": []} for k in r2s}

    for ci, d in enumerate(T.DURATIONS):
        row = theta_df.loc[d]
        theta = {p: float(row[p]) for p in spec.param_names}
        lapse = float(row["lapse"])

        grid = T.compute_grid(spec, theta, quad)
        pf = T.apply_lapse(grid["p_resp"], lapse, r_max)
        psum = T.predicted_summaries(pf)
        pred_exact = psum["pct_exact"].to_numpy()
        pred_core_mean = (grid["p_resp"] * T.WELLS[None, :]).sum(axis=1)
        pred_rt = T.T0 + T._rt_link(
            rt_metric, rt_scale, grid["EH"], grid["ET"], grid["ET_flow"]
        )
        # Give-up onset: the smallest n at which the clean-up would run past the
        # budget K = MAX_STEPS (the give-up cap selected by the profile
        # likelihood). We read it off the same landscape
        # run to a larger budget: where the unbounded settling time first exceeds
        # K, the cap starts to bind. Past this point RT would diverge without the
        # give-up (explained in the manuscript); we mark the onset with a line.
        _cap = T.MAX_STEPS
        T.MAX_STEPS = 500
        _et_unc = T.compute_grid(spec, theta, quad)["ET"]
        T.MAX_STEPS = _cap
        _over = np.where(_et_unc > _cap)[0]
        n_cut = int(n[_over[0]]) if _over.size else None

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
        emp_mean = np.array(emp_mean)
        emp_exact = np.array(emp_exact)
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
        if n_cut is not None:
            ax2.axvline(n_cut, color="0.55", lw=1.0, ls=":", zorder=0)
            if ci == len(T.DURATIONS) - 1:   # rightmost panel: label the onset
                ax2.text(n_cut + 0.3, 0.30, "Give-up\nengages",
                         transform=ax2.get_xaxis_transform(), va="center",
                         ha="left", fontsize=7.5 * FONT_SCALE, color="0.4")

        ax0.set_title(f"{d} ms")
        for ax, key, hx, ha in ((ax0, "mean", 0.04, "left"),
                                 (ax1, "exact", 0.96, "right"),
                                 (ax2, "rt", 0.04, "left")):
            ax.text(hx, 0.93, f"R²={r2s[key][-1]:.2f}", transform=ax.transAxes,
                    fontsize=8.5 * FONT_SCALE, va="top", ha=ha)
        if ci == 0:
            ax0.set_ylabel("Mean guess")
            ax1.set_ylabel("% exact")
            # OLD: ax2.set_ylabel("Mean RT (s)")
            ax2.set_ylabel("Geometric mean RT (s)")
            panel_label(ax0, "a")
            panel_label(ax1, "b")
            panel_label(ax2, "c")
        ax2.set_xlabel("n shown")

    from matplotlib.lines import Line2D
    axes[0][0].legend(handles=[
        Line2D([], [], marker=".", color="k", ls="", ms=9, label="Data"),
        Line2D([], [], color=C_ATTR, lw=2, label="Model")],
        frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_fig_pdf, dpi=300)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    print(f"Saved {out_fig} (G={args.budget}, RT metric={rt_metric})")
    for key, lab in (("mean", "mean guess"), ("exact", "% exact"), ("rt", "geomean RT")):
        vals = ", ".join(f"{d}:{v:.2f}" for d, v in zip(T.DURATIONS, r2s[key]))
        pool = r2(np.concatenate(pooled[key]["obs"]), np.concatenate(pooled[key]["pred"]))
        print(f"  R² {lab:11s} -> {vals}  | pooled {pool:.2f}")


if __name__ == "__main__":
    main()
