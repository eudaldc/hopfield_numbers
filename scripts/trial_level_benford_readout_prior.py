"""Winner + Benford readout prior: does frequency help at the readout too?

The trial-level 2x2 deliberately used a FLAT readout prior for every family so
that the well-family and clean-up factors stayed isolated; in the winner the
Benford frequency enters only through the well WIDTH, sigma_n = (sigma_c/f_n)^beta.
The old SM1 Gaussian-Benford variant instead let the frequency multiply the
readout, p(n | x*) proportional to f_n * exp(g_n(x*)).

This script asks whether the winner improves when the same frequency prior
multiplies its readout — a ZERO-extra-parameter change, so a plain AIC comparison
against the registered winner is fair. If it helps, the "numbers as
frequency-weighted memories" story gains a second, independent foothold; if it
does not, the width channel already carries the frequency effect.

    .venv/bin/python -m scripts.trial_level_benford_readout_prior
    .venv/bin/python -m scripts.trial_level_benford_readout_prior --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT_CSV = T.OUT_DIR / "benford_readout_prior.csv"

SPEC = T.MODEL_BY_KEY[T.WINNER_KEY]
BOUNDS = list(SPEC.bounds) + [T.LAPSE_BOUNDS]

_W: dict = {}


def _init_worker(fit_nodes: int) -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    _W["r_max"] = r_max
    _W["data"] = T.build_duration_data(trial, r_max)
    _W["quad_fit"] = T.make_quadrature(fit_nodes, kind="grid")
    _W["quad_eval"] = T.make_quadrature(1601, kind="grid")


def p_resp_prior(theta: dict, quad) -> np.ndarray:
    """Winner response grid with a Benford readout prior, p(n|x*) ~ f_n exp(g_n)."""
    n_values = T.N_SHOWN.astype(float)
    sd = T.evidence_sd(n_values, theta["s0"], theta["lambda0"])
    y = n_values[:, None] + np.sqrt(2.0) * sd[:, None] * quad.t[None, :]
    y_flat = y.reshape(-1)
    sigma = np.power(theta["sigma_c"] / T.FREQ, theta["beta"])
    x_star, _, _ = T._cleanup(y_flat, family="gaussian", sigma=sigma,
                              max_steps=T.MAX_STEPS)
    logf = T._gauss_logwells(x_star, sigma) + np.log(T.FREQ)[None, :]
    probs = T._softmax(logf).reshape(n_values.size, quad.n_nodes, T.N_WELLS)
    return (probs * quad.w[None, :, None]).sum(axis=1)


def loglik_prior(vec: np.ndarray, quad, counts: np.ndarray, r_max: int) -> float:
    theta = dict(zip(SPEC.param_names, vec[:-1]))
    lapse = float(vec[-1])
    try:
        p_resp = p_resp_prior(theta, quad)
    except (FloatingPointError, ValueError):
        return -np.inf
    pf = T.apply_lapse(p_resp, lapse, r_max)
    if not np.all(np.isfinite(pf)) or np.any(pf <= 0):
        return -np.inf
    ll = float(np.sum(counts * np.log(pf)))
    return ll if np.isfinite(ll) else -np.inf


def _fit_one(args) -> dict:
    d, maxiter, popsize = args
    data = _W["data"][d]

    def neg(vec: np.ndarray) -> float:
        return -loglik_prior(vec, _W["quad_fit"], data.counts, _W["r_max"])

    de = differential_evolution(neg, bounds=BOUNDS, maxiter=maxiter, popsize=popsize,
                                tol=1e-4, mutation=(0.5, 1.0), recombination=0.85,
                                seed=T.SEED + d, init="latinhypercube", polish=False,
                                updating="deferred", workers=1)
    pol = minimize(neg, de.x, method="Nelder-Mead", bounds=BOUNDS,
                   options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-4})
    vec = pol.x if pol.fun <= de.fun else de.x
    logl = loglik_prior(vec, _W["quad_eval"], data.counts, _W["r_max"])
    k = len(SPEC.param_names) + 1
    row = {"duration_ms": d, "k_params": k, "logL": float(logl),
           "AIC": float(2 * k - 2 * logl)}
    row |= {p: float(v) for p, v in zip(SPEC.param_names, vec)}
    row["lapse"] = float(vec[-1])
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    fit_nodes = 101 if args.smoke else 201
    maxiter = 6 if args.smoke else 35
    popsize = 5 if args.smoke else 10
    durs = T.DURATIONS[2:3] if args.smoke else T.DURATIONS

    t0 = time.time()
    with Pool(len(durs), initializer=_init_worker, initargs=(fit_nodes,)) as pool:
        rows = pool.map(_fit_one, [(d, maxiter, popsize) for d in durs])
    df = pd.DataFrame(rows).sort_values("duration_ms")

    winner = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    df["winner_AIC_flat"] = [float(winner.loc[d, "resp_AIC"]) for d in df["duration_ms"]]
    df["dAIC_prior_minus_flat"] = df["AIC"] - df["winner_AIC_flat"]

    if args.smoke:
        print(df.round(3).to_string(index=False))
        return
    df.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 200)
    print("\n=== Winner with Benford readout prior vs registered winner (flat) ===")
    print(df[["duration_ms", "k_params", "logL", "AIC", "winner_AIC_flat",
              "dAIC_prior_minus_flat"]].round(1).to_string(index=False))
    print(f"\nSummed dAIC (prior - flat): {df['dAIC_prior_minus_flat'].sum():+.1f} "
          "(negative => the readout prior helps; equal k, so AIC == 2*dlogL)")
    print(f"Elapsed {(time.time() - t0) / 60:.1f} min. Saved {OUT_CSV}")
    print("\nNote: the prior enters the READOUT only (as in the old SM1), not the "
          "energy landscape; the width channel keeps its own frequency dependence.")


if __name__ == "__main__":
    main()
