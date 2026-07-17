"""Fitting for the round-2 evidence families (see evidence2.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from . import evidence2 as E

FIT_SEED = 7301
M_FIT = 2048
M_FINAL = 16384
# The clean-up (120 clamped iterations) dominates each likelihood evaluation,
# so the optimiser uses the registered analysis's own fitting grid (201 nodes)
# and the reported likelihoods are recomputed on the accurate 1601-node grid.
QUAD_FIT = 201
QUAD_FINAL = 1601

# (lo, hi, log?)
BOUNDS: dict[str, tuple[float, float, bool]] = {
    "sigma_c": (1e-3, 0.5, True),
    "p": (0.3, 3.0, False),
    # ddm / kalman sensory channel: variance coefficient and exponent
    "s0": (1e-6, 2.0, True),
    "lam": (0.0, 4.0, False),
    # kalman prior
    "mu0": (0.0, 12.0, False),
    "v0": (1e-3, 1e3, True),
    # imperfect pulses
    "rho": (2.0, 400.0, True),
    "pi_hit": (0.3, 1.0, False),
    "cv_a": (0.01, 1.5, False),
    "phi": (0.01, 20.0, True),
    # coupling. The landscape's restoring force is ~1/sigma_n^2 (up to ~2e3
    # for the sharpest wells), so the tether must be allowed well past that
    # for the batch-equivalent corner (beta -> inf) to be reachable at all.
    "r_tether": (0.5, 1e5, True),
    "tau_x": (0.005, 2.0, True),
    "beta": (0.1, 1e5, True),
}


def to_native(name: str, v: float) -> float:
    return float(np.exp(v)) if BOUNDS[name][2] else float(v)


def opt_bounds(names) -> list[tuple[float, float]]:
    out = []
    for n in names:
        lo, hi, lg = BOUNDS[n]
        out.append((np.log(lo), np.log(hi)) if lg else (lo, hi))
    return out


def vec_to_theta(names, vec) -> dict:
    return {n: to_native(n, v) for n, v in zip(names, np.asarray(vec, float))}


def theta_to_vec(names, theta) -> np.ndarray:
    return np.array([np.log(theta[n]) if BOUNDS[n][2] else theta[n]
                     for n in names], float)


def _negll(vec, family, variant, names, counts, r_max, m_paths, quad_nodes):
    theta = vec_to_theta(names, vec)
    try:
        ll, _ = E.total_loglik(family, variant, theta, counts, r_max,
                               m_paths=m_paths, quad_nodes=quad_nodes)
    except (FloatingPointError, ValueError):
        return 1e18
    return -ll if np.isfinite(ll) else 1e18


def fit(family: str, variant: str, counts: dict, r_max: int, *,
        maxiter: int = 40, popsize: int = 9, seed: int = FIT_SEED,
        workers: int = 10, m_fit: int = M_FIT, m_final: int = M_FINAL,
        x0: np.ndarray | None = None) -> dict:
    names = E.param_names(family, variant)
    bnds = opt_bounds(names)
    args = (family, variant, names, counts, r_max, m_fit, QUAD_FIT)
    t0 = time.time()
    if x0 is None:
        de = differential_evolution(
            _negll, bounds=bnds, args=args, maxiter=maxiter, popsize=popsize,
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.85, seed=seed,
            init="sobol", polish=False, updating="deferred", workers=workers)
        x, f = de.x, float(de.fun)
    else:
        x = np.asarray(x0, float)
        f = _negll(x, *args)
    pol = minimize(_negll, x, args=args, method="Nelder-Mead", bounds=bnds,
                   options={"maxiter": 300, "xatol": 1e-4, "fatol": 0.05})
    if pol.fun <= f:
        x, f = pol.x, float(pol.fun)

    theta = vec_to_theta(names, x)
    ll, lapses = E.total_loglik(family, variant, theta, counts, r_max,
                                m_paths=m_final, quad_nodes=QUAD_FINAL)
    k = E.n_params(family, variant)
    n_tot = int(sum(c.sum() for c in counts.values()))
    return {
        "family": family, "variant": variant, "theta": theta,
        "lapses": {int(d): float(v) for d, v in lapses.items()},
        "x_opt": [float(v) for v in x], "param_names": list(names),
        "logL_fit": -f, "logL": float(ll), "k": k, "n": n_tot,
        "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n_tot) - 2 * ll,
        "m_final": m_final, "wall_min": (time.time() - t0) / 60.0,
    }


def save(fit_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fit_dict, f, indent=1, default=float)
