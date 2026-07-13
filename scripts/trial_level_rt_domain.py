"""RT sensitivity analysis using the width-based window n < n_width.

The continuous index n_width is defined by sigma_n = 1. It is used here only
to specify a reproducible lower-numerosity fitting window, not as a model-validity,
exact-range, or topological boundary.

Procedure (two-stage, RT never touches the landscape):
  * landscape + lapse are taken from the accuracy-only fit (`winner_theta.csv`);
  * n_width(duration) = sigma_n = 1 crossing of that landscape;
  * the RT link (latent effort -> seconds; alpha, beta, sigma) is fit on trials with
    n <= floor(n_width) only, then the shape fixed by the accuracy landscape is
    compared inside and outside that fitting window.

We report R^2 of the median-RT curve (a) inside the width-based window and
(b) over all n, for both the entropy and dynamics
(steps+entropy hybrid) read-outs.

    .venv/bin/python -m scripts.trial_level_rt_domain
"""

from __future__ import annotations

import sys
from math import floor
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
from scripts.trial_level_model_comparison import DurationData
from figures.plot_style import C_ATTR, apply_style

WINNER = T.WINNER_KEY
METRICS = ("hybrid", "entropy")
OUT_CSV = T.OUT_DIR / "rt_domain.csv"
OUT_FIG = T.OUT_DIR / "rt_domain.png"


def sigma_n(theta: dict, n: np.ndarray) -> np.ndarray:
    # power-law wells: sigma_n = sigma_c * n^p
    return theta["sigma_c"] * np.power(np.asarray(n, float), theta["pow"])


def n_width_crossing(theta: dict, level: float = 1.0) -> float:
    ns = np.linspace(1.0, 40.0, 40000)
    s = sigma_n(theta, ns)
    idx = np.where(s >= level)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan")
    i = idx[0]
    return float(ns[i - 1] + (level - s[i - 1]) * (ns[i] - ns[i - 1]) /
                 (s[i] - s[i - 1]))


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 and obs.size > 1 else np.nan


def predicted_median_rt(metric: str, scale: dict, grid: dict) -> np.ndarray:
    """Median of the log-normal RT = t0 + link(n)."""
    if metric == "hybrid":
        return T.T0 + scale["alpha"] * grid["ET"] + scale["beta"] * grid["EH"]
    return T.T0 + scale["rt_scale"] * grid["EH"]


def restrict_data(trial: pd.DataFrame, d: int, nmax: int, r_max: int) -> DurationData:
    sub = trial[(trial["exp_dur"] == d) & (trial["n_shown"] <= nmax)]
    n_t = sub["n_shown"].to_numpy(int)
    r_t = sub["n_guess"].to_numpy(int)
    logrt = np.log(sub["rt_s"].to_numpy(float))
    counts = np.zeros((T.N_SHOWN.size, r_max), dtype=float)
    np.add.at(counts, (n_t - T.N_TRUE_MIN, np.clip(r_t, 1, r_max) - 1), 1.0)
    return DurationData(int(d), n_t, r_t, logrt, counts, int(n_t.size))


def main() -> None:
    apply_style(1.5)   # rendered at \textwidth (figure*), 15.5 in canvas
    spec = T.MODEL_BY_KEY[WINNER]
    quad = T.make_quadrature(801, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_full = T.build_duration_data(trial, r_max)
    wt = pd.read_csv(T.OUT_DIR / "pow_theta.csv").set_index("duration_ms")
    n = T.N_SHOWN.astype(float)

    rows = []
    fig, axes = plt.subplots(1, len(T.DURATIONS),
                             figsize=(3.1 * len(T.DURATIONS), 3.8), squeeze=False,
                             sharey=False)

    for ci, d in enumerate(T.DURATIONS):
        theta = {p: float(wt.loc[d, p]) for p in spec.param_names}
        lapse = float(wt.loc[d, "lapse"])
        resp_fit = {"theta": theta, "lapse": lapse}
        n_width = n_width_crossing(theta)
        n_width_max = int(floor(n_width))
        grid = T.compute_grid(spec, theta, quad)    # all n = 1..15

        # empirical RT per n (all n): the geometric mean, i.e. the log-normal RT
        # MLE target (= exp(mean log rt)) — labelled simply "mean" in the figure
        lrt, ntr = data_full[d].logrt_trial, data_full[d].n_trial
        emp = np.array([np.exp(np.mean(lrt[ntr == int(nn)])) for nn in n])
        in_window = n <= n_width_max

        data_window = restrict_data(trial, d, n_width_max, r_max)
        pred_by_metric = {}
        for metric in METRICS:
            rt_window = T.fit_rt(spec, data_window, resp_fit, quad, metric=metric,
                                 seed=T.SEED + d)
            rt_full = T.fit_rt(spec, data_full[d], resp_fit, quad, metric=metric,
                               seed=T.SEED + d)
            pred_window = predicted_median_rt(metric, rt_window["scale"], grid)
            pred_full = predicted_median_rt(metric, rt_full["scale"], grid)
            pred_by_metric[metric] = pred_window
            rows.append({
                "duration_ms": int(d), "metric": metric,
                "n_width": round(n_width, 2), "n_width_window_max": n_width_max,
                "n_width_window_levels": int(in_window.sum()),
                "R2_width_fit_in_window": r2(emp[in_window], pred_window[in_window]),
                "R2_full_fit_in_window": r2(emp[in_window], pred_full[in_window]),
                "R2_full_fit_all_n": r2(emp, pred_full),
                **{f"scale_{k}": v for k, v in rt_window["scale"].items()},
                "sigma_rt_width_window": rt_window["sigma_rt"],
            })

        # figure: hybrid domain fit -> solid within domain, dashed extrapolation
        ax = axes[0][ci]
        ph = pred_by_metric["hybrid"]
        ax.plot(n, emp, "k.", ms=9, label="Data")
        ax.plot(n[in_window], ph[in_window], "-", color=C_ATTR, lw=2.4, label="Fit window")
        ax.plot(n[~in_window], ph[~in_window], "--", color=C_ATTR, lw=1.6, alpha=0.8,
                label="Outside window")
        ax.axvline(n_width, color="0.25", ls=":", lw=1.2)
        rr = rows[-2]  # the hybrid row for this duration
        ax.set_title(f"{d} ms")
        ax.text(0.97, 0.05, f"$R^2 = {rr['R2_width_fit_in_window']:.2f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
                zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=3))
        ax.text(n_width + 0.25, ax.get_ylim()[0] + 0.25 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                r"$n_{\rm width}$", fontsize=17, color="0.25")
        if ci == 0:
            ax.set_ylabel("Mean RT (s)")

    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 0.90, 1))
    fig.supxlabel("Number shown  $n$", fontsize=19, x=0.45)
    fig.legend(handles, leg_labels, frameon=False, loc="center right",
               bbox_to_anchor=(0.998, 0.4))
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("=== RT fit in the width-based window (n <= floor(n_width)) ===")
    show = df[["duration_ms", "metric", "n_width", "n_width_window_max",
               "n_width_window_levels", "R2_width_fit_in_window",
               "R2_full_fit_in_window", "R2_full_fit_all_n"]]
    print(show.round(3).to_string(index=False))
    print(f"\nSaved {OUT_CSV.name}, {OUT_FIG.name}")


if __name__ == "__main__":
    main()
