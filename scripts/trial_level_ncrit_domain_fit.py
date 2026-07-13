"""Sensitivity analysis fitted within the width-based window n <= n_width.

The main-manuscript version (``trial_level_rt_figure.py``) trains the winner
landscape on the full range n = 1..15 and predicts RT at every n. This supplement
uses a lower-numerosity window defined from the fitted width profile:

  (1) **retrains** the winner landscape (super-linear Gaussian attractor, free
      beta) on trials with n <= n_width, and
  (2) **fits and predicts** the RT link (latent effort -> seconds) on those same
      n <= n_width trials.

Here n_width(duration) is the continuous sigma_n = 1 index of the
**main full-range fit**
(``winner_theta.csv``), so the training window is fixed by the main model and is
*not* circular. It matches the main-fit values 4.6 / 7.5 / 8.7 / 9.2 / 10.0
(floor -> n <= 4 / 7 / 8 / 9 / 9) for 40 / 80 / 160 / 320 / 640 ms.

Because the clean-up is a zero-parameter change (deterministic given theta), the
static twin (``gaussian_freebeta_no_cleanup``) is refit in the same window to
report the clean-up ΔAIC there. This use of n_width defines an analysis window,
not a model-validity, exact-range, or topological boundary.

**Caveat (flagged in the output).** At 40 ms n_width ~ 4.6, so only n in {1,2,3,4}
enter the landscape refit -- 4 numerosity levels for 4 landscape params, i.e. a
near-saturated landscape. The row is kept but flagged ``landscape_saturated``; the
width exponent beta there is not well identified.

    .venv/bin/python -m scripts.trial_level_ncrit_domain_fit            # fit + plot
    .venv/bin/python -m scripts.trial_level_ncrit_domain_fit --replot   # figure only
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
TWIN = "gaussian_freebeta_no_cleanup"          # equal-k static twin (clean-up off)
METRICS = ("hybrid", "entropy")                # dynamics (steps+entropy) and static

# Optimiser budget for the landscape refit (matches the real-run settings) and the
# quadrature grids (fit cheap, evaluate accurate; RT grid validated to match MC
# to <=2e-3, as in trial_level_rt_domain).
FIT_NODES, EVAL_NODES, RT_NODES = 201, 1201, 801
MAXITER, POPSIZE = 35, 10

OUT_LANDSCAPE = T.OUT_DIR / "ncrit_domain_landscape.csv"
OUT_FIT = T.OUT_DIR / "ncrit_domain_fit.csv"
OUT_FIG = T.OUT_DIR / "ncrit_domain_rt.png"


def sigma_n(theta: dict, n: np.ndarray) -> np.ndarray:
    """Gaussian-Benford well width sigma_n = (sigma_c / freq_n)^beta."""
    freq = np.log10(1.0 + 1.0 / np.asarray(n, dtype=float))
    return np.power(theta["sigma_c"] / freq, theta["beta"])


def n_width_crossing(theta: dict, level: float = 1.0) -> float:
    """Smallest n where sigma_n(n) >= level (linear interpolation)."""
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


def predicted_rt(metric: str, scale: dict, grid: dict) -> np.ndarray:
    """Median (= geometric-mean-targeting) log-normal RT = t0 + link(n)."""
    if metric == "hybrid":
        return T.T0 + scale["alpha"] * grid["ET"] + scale["beta"] * grid["EH"]
    return T.T0 + scale["rt_scale"] * grid["EH"]


def restrict_data(trial: pd.DataFrame, d: int, nmax: int, r_max: int) -> DurationData:
    """DurationData for one duration keeping only n_shown <= nmax.

    Both channels are restricted: counts are zero for n > nmax (so the response
    refit ignores them), and n_trial / logrt_trial only carry the domain trials
    (so the RT fit uses only them)."""
    sub = trial[(trial["exp_dur"] == d) & (trial["n_shown"] <= nmax)]
    n_t = sub["n_shown"].to_numpy(int)
    r_t = sub["n_guess"].to_numpy(int)
    logrt = np.log(sub["rt_s"].to_numpy(float))
    counts = np.zeros((T.N_SHOWN.size, r_max), dtype=float)
    np.add.at(counts, (n_t - T.N_TRUE_MIN, np.clip(r_t, 1, r_max) - 1), 1.0)
    return DurationData(int(d), n_t, r_t, logrt, counts, int(n_t.size))


def empirical_rt(data_full: dict, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-n geometric-mean RT (log-normal MLE target) and median (reference)."""
    lrt, ntr = data_full[d].logrt_trial, data_full[d].n_trial
    n = T.N_SHOWN.astype(float)
    geo = np.array([np.exp(np.mean(lrt[ntr == int(nn)])) for nn in n])
    med = np.array([np.median(np.exp(lrt[ntr == int(nn)])) for nn in n])
    return geo, med


def rt_scale_from_row(metric: str, row: pd.Series) -> dict:
    if metric == "hybrid":
        return {"alpha": float(row["scale_alpha"]), "beta": float(row["scale_beta"])}
    return {"rt_scale": float(row["scale_rt_scale"])}


# ── Fitting ──────────────────────────────────────────────────────────────


def fit_all(trial: pd.DataFrame, data_full: dict, wt_main: pd.DataFrame,
            r_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = T.MODEL_BY_KEY[WINNER]
    twin = T.MODEL_BY_KEY[TWIN]
    n_landscape_params = len(spec.param_names)          # 4 (sigma_c, s0, lambda0, beta)
    quad_fit = T.make_quadrature(FIT_NODES, kind="grid")
    quad_eval = T.make_quadrature(EVAL_NODES, kind="grid")
    quad_rt = T.make_quadrature(RT_NODES, kind="grid")
    n = T.N_SHOWN.astype(float)

    land_rows: list[dict] = []
    fit_rows: list[dict] = []
    for d in T.DURATIONS:
        # n_width from the main full-range fit fixes the training window
        theta_main = {p: float(wt_main.loc[d, p]) for p in spec.param_names}
        n_width = n_width_crossing(theta_main)
        nmax = int(floor(n_width))
        n_levels = nmax
        saturated = n_levels <= n_landscape_params
        data_dom = restrict_data(trial, d, nmax, r_max)
        print(f"[{d} ms] n_width={n_width:.2f} -> n<={nmax} ({n_levels} levels, "
              f"{data_dom.n_total} trials){' [SATURATED]' if saturated else ''}",
              flush=True)

        # (1) retrain the winner landscape (and its static twin) within domain
        rf = T.fit_response(spec, data_dom, quad_fit, r_max, maxiter=MAXITER,
                            popsize=POPSIZE, seed=T.SEED + d, quad_eval=quad_eval)
        print(f"    winner refit: AIC={rf['AIC']:.1f} beta={rf['theta']['beta']:.3f} "
              f"lapse={rf['lapse']:.3f}", flush=True)
        rf_twin = T.fit_response(twin, data_dom, quad_fit, r_max, maxiter=MAXITER,
                                 popsize=POPSIZE, seed=T.SEED + d, quad_eval=quad_eval)
        print(f"    twin refit:   AIC={rf_twin['AIC']:.1f} "
              f"(clean-up dAIC={rf_twin['AIC'] - rf['AIC']:+.1f})", flush=True)
        theta_dom = rf["theta"]
        n_width_refit = n_width_crossing(theta_dom)

        land_rows.append({
            "duration_ms": int(d),
            "n_width_main": round(n_width, 3), "n_width_window_max": nmax,
            "n_width_window_levels": n_levels, "landscape_saturated": bool(saturated),
            **{p: theta_dom[p] for p in spec.param_names},
            "lapse": rf["lapse"], "resp_logL": rf["logL"], "resp_AIC": rf["AIC"],
            "n_width_refit": round(n_width_refit, 3),
            "twin_resp_AIC": rf_twin["AIC"],
            "cleanup_dAIC_domain": rf_twin["AIC"] - rf["AIC"],
        })

        # (2) fit + predict the RT link on the same domain, retrained landscape
        resp_fit = {"theta": theta_dom, "lapse": rf["lapse"]}
        grid = T.compute_grid(spec, theta_dom, quad_rt)  # over n = 1..15
        in_dom = n <= nmax
        emp_geo, _ = empirical_rt(data_full, d)
        for metric in METRICS:
            rt = T.fit_rt(spec, data_dom, resp_fit, quad_rt, metric=metric,
                          seed=T.SEED + d)
            pred = predicted_rt(metric, rt["scale"], grid)
            r2_dom = r2(emp_geo[in_dom], pred[in_dom])
            print(f"    RT {metric:7s}: R2|dom={r2_dom:.3f}", flush=True)
            fit_rows.append({
                "duration_ms": int(d), "metric": metric,
                "n_width": round(n_width, 3), "n_width_window_max": nmax,
                "n_width_window_levels": n_levels, "landscape_saturated": bool(saturated),
                "R2_domain": r2_dom,
                **{f"scale_{k}": v for k, v in rt["scale"].items()},
                "sigma_rt": rt["sigma_rt"], "rt_logL": rt["logL"], "rt_AIC": rt["AIC"],
            })
    return pd.DataFrame(land_rows), pd.DataFrame(fit_rows)


# ── Figure (rebuildable from the saved CSVs; no refitting) ───────────────


def render_figure(data_full: dict, land_df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    spec = T.MODEL_BY_KEY[WINNER]
    quad_rt = T.make_quadrature(RT_NODES, kind="grid")
    n = T.N_SHOWN.astype(float)
    land_df = land_df.set_index("duration_ms")

    fig, axes = plt.subplots(1, len(T.DURATIONS),
                             figsize=(3.0 * len(T.DURATIONS), 3.6), squeeze=False)
    for ci, d in enumerate(T.DURATIONS):
        lrow = land_df.loc[d]
        theta_dom = {p: float(lrow[p]) for p in spec.param_names}
        n_width = float(lrow["n_width_main"])
        nmax = int(lrow["n_width_window_max"])
        saturated = bool(lrow["landscape_saturated"])
        grid = T.compute_grid(spec, theta_dom, quad_rt)
        in_dom = n <= nmax
        emp_geo, emp_med = empirical_rt(data_full, d)

        ax = axes[0][ci]
        ax.plot(n, emp_med, ".", color="0.72", ms=5, label="data (median)")
        ax.plot(n, emp_geo, "k.", ms=8, label="data (geom. mean)")
        for metric, col, lbl in (("hybrid", "#c0392b", "dynamics"),
                                 ("entropy", "#2c7fb8", "entropy")):
            frow = fit_df[(fit_df.duration_ms == d) & (fit_df.metric == metric)].iloc[0]
            pred = predicted_rt(metric, rt_scale_from_row(metric, frow), grid)
            ax.plot(n[in_dom], pred[in_dom], "-", color=col, lw=1.8,
                    label=f"{lbl} (R²|dom={float(frow['R2_domain']):.2f})")
            ax.plot(n[~in_dom], pred[~in_dom], "--", color=col, lw=1.2, alpha=0.6)
        ax.axvline(n_width, color="k", ls=":", lw=1.0)
        y0, y1 = ax.get_ylim()
        ax.text(n_width + 0.2, y1 - 0.04 * (y1 - y0), f"n_width≈{n_width:.1f}",
                fontsize=7, va="top", ha="left")
        flag = "  [saturated]" if saturated else ""
        ax.set_title(f"{d} ms{flag}", fontsize=9)
        ax.set_xlabel("n shown")
        if ci == 0:
            ax.set_ylabel("geometric-mean RT (s)")
        ax.legend(fontsize=6.3, frameon=False, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Sensitivity analysis in the n_width fitting window "
                 "(solid: inside; dashed: outside)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    replot = "--replot" in sys.argv[1:]
    T.OUT_DIR.mkdir(parents=True, exist_ok=True)
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_full = T.build_duration_data(trial, r_max)

    if replot and OUT_LANDSCAPE.exists() and OUT_FIT.exists():
        land_df = pd.read_csv(OUT_LANDSCAPE)
        fit_df = pd.read_csv(OUT_FIT)
        print(f"Replotting from cached {OUT_LANDSCAPE.name} / {OUT_FIT.name}.")
    else:
        wt_main = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
        land_df, fit_df = fit_all(trial, data_full, wt_main, r_max)
        land_df.to_csv(OUT_LANDSCAPE, index=False)
        fit_df.to_csv(OUT_FIT, index=False)
        print("\n=== landscape retrained in the n_width(duration) window ===")
        print(land_df[["duration_ms", "n_width_main", "n_width_window_max",
                       "n_width_window_levels", "landscape_saturated", "beta", "n_width_refit",
                       "cleanup_dAIC_domain"]].round(3).to_string(index=False))
        print("\n=== RT predicted in the n_width(duration) window, geometric-mean R^2 ===")
        print(fit_df[["duration_ms", "metric", "n_width", "n_width_window_max",
                      "landscape_saturated", "R2_domain"]].round(3).to_string(index=False))

    render_figure(data_full, land_df, fit_df)
    print(f"\nSaved {OUT_LANDSCAPE.name}, {OUT_FIT.name}, {OUT_FIG.name}")


if __name__ == "__main__":
    main()
