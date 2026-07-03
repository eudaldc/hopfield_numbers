"""RT prediction restricted to the model's domain of validity (n < n_crit).

Framing test. The winner is a small-range (exact) mechanism valid up to the
resolvability limit n_crit (well width sigma_n = 1 <=> adjacent-well d' = 1). Above
n_crit people leave the exact regime (fast approximate guessing -> the RT dip the
monotone model cannot capture). So the honest RT question is not "does it fit all
n?" but "does it predict RT *within its domain* (n < n_crit)?".

Procedure (two-stage, RT never touches the landscape):
  * landscape + lapse are taken from the accuracy-only fit (`winner_theta.csv`);
  * n_crit(duration) = sigma_n = 1 crossing of that landscape;
  * the RT link (latent effort -> seconds; alpha, beta, sigma) is fit on trials with
    n <= floor(n_crit) only, then the SHAPE (fixed by the accuracy landscape) is
    compared to data within the domain and extrapolated above n_crit.

We report R^2 of the median-RT curve (a) within the domain for the domain-restricted
fit, and (b) the full-range fit over all n, for both the entropy and dynamics
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

WINNER = "gaussian_freebeta_attractor"
METRICS = ("hybrid", "entropy")
OUT_CSV = T.OUT_DIR / "rt_domain.csv"
OUT_FIG = T.OUT_DIR / "rt_domain.png"


def sigma_n(theta: dict, n: np.ndarray) -> np.ndarray:
    freq = np.log10(1.0 + 1.0 / np.asarray(n, float))
    return np.power(theta["sigma_c"] / freq, theta["beta"])


def ncrit_crossing(theta: dict, level: float = 1.0) -> float:
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
    spec = T.MODEL_BY_KEY[WINNER]
    quad = T.make_quadrature(801, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_full = T.build_duration_data(trial, r_max)
    wt = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    n = T.N_SHOWN.astype(float)

    rows = []
    fig, axes = plt.subplots(1, len(T.DURATIONS),
                             figsize=(3.0 * len(T.DURATIONS), 3.8), squeeze=False)

    for ci, d in enumerate(T.DURATIONS):
        theta = {p: float(wt.loc[d, p]) for p in spec.param_names}
        lapse = float(wt.loc[d, "lapse"])
        resp_fit = {"theta": theta, "lapse": lapse}
        ncrit = ncrit_crossing(theta)
        nmax = int(floor(ncrit))                    # last fully-resolvable integer
        grid = T.compute_grid(spec, theta, quad)    # all n = 1..15

        # empirical RT per n (all n). Compare against the geometric mean (the
        # log-normal RT MLE target = exp(mean log rt)); the median sits ~0.1-0.2 s
        # below it because log RT is right-skewed, and would make the line look
        # spuriously high. Median kept for reference.
        lrt, ntr = data_full[d].logrt_trial, data_full[d].n_trial
        emp = np.array([np.exp(np.mean(lrt[ntr == int(nn)])) for nn in n])
        emp_med = np.array([np.median(np.exp(lrt[ntr == int(nn)])) for nn in n])
        in_dom = n <= nmax

        data_dom = restrict_data(trial, d, nmax, r_max)
        pred_by_metric = {}
        for metric in METRICS:
            rt_dom = T.fit_rt(spec, data_dom, resp_fit, quad, metric=metric,
                              seed=T.SEED + d)
            rt_full = T.fit_rt(spec, data_full[d], resp_fit, quad, metric=metric,
                               seed=T.SEED + d)
            pred_dom = predicted_median_rt(metric, rt_dom["scale"], grid)
            pred_full = predicted_median_rt(metric, rt_full["scale"], grid)
            pred_by_metric[metric] = pred_dom
            rows.append({
                "duration_ms": int(d), "metric": metric,
                "n_crit": round(ncrit, 2), "n_domain_max": nmax,
                "n_domain_levels": int(in_dom.sum()),
                "R2_domain_fit_on_domain": r2(emp[in_dom], pred_dom[in_dom]),
                "R2_full_fit_on_domain": r2(emp[in_dom], pred_full[in_dom]),
                "R2_full_fit_all_n": r2(emp, pred_full),
                **{f"scale_{k}": v for k, v in rt_dom["scale"].items()},
                "sigma_rt_domain": rt_dom["sigma_rt"],
            })

        # figure: hybrid domain fit -> solid within domain, dashed extrapolation
        ax = axes[0][ci]
        ph = pred_by_metric["hybrid"]
        ax.plot(n, emp_med, ".", color="0.72", ms=5, label="data (median)")
        ax.plot(n, emp, "k.", ms=8, label="data (geom. mean)")
        ax.plot(n[in_dom], ph[in_dom], "-", color="#c0392b", lw=2,
                label="fit (n < n_crit)")
        ax.plot(n[~in_dom], ph[~in_dom], "--", color="#c0392b", lw=1.4, alpha=0.7,
                label="extrapolation")
        ax.axvline(ncrit, color="k", ls=":", lw=1.0)
        ax.text(ncrit + 0.15, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n_crit≈{ncrit:.1f}", fontsize=7)
        rr = rows[-2]  # the hybrid row for this duration
        ax.set_title(f"{d} ms  (R²|dom={rr['R2_domain_fit_on_domain']:.2f})", fontsize=9)
        ax.set_xlabel("n shown")
        if ci == 0:
            ax.set_ylabel("geometric-mean RT (s)")
            ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    fig.suptitle("RT predicted within the model's domain (n < n_crit), extrapolated above "
                 "(hybrid: steps + entropy)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=190)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("=== RT fit restricted to the exact regime (n <= floor(n_crit)) ===")
    show = df[["duration_ms", "metric", "n_crit", "n_domain_max", "n_domain_levels",
               "R2_domain_fit_on_domain", "R2_full_fit_on_domain", "R2_full_fit_all_n"]]
    print(show.round(3).to_string(index=False))
    print(f"\nSaved {OUT_CSV.name}, {OUT_FIG.name}")


if __name__ == "__main__":
    main()
