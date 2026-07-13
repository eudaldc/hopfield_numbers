"""Fitting machinery for the driven-attractor model family.

Free parameters are optimised in transformed space (logs for scale
parameters). The five per-duration lapses are profiled out inside the
likelihood (concave 1-D profiles), so the outer search is 4-6 dimensional.

Model variants (response channel):
  M1_offset  : accumulation-at-offset control. x(d) = K(d), then relaxation.
               free: sigma_c, p, rho, tau_x            (tau_x sets the cap)
  M2_tether  : primary continuously driven attractor (soft tether).
               free: sigma_c, p, rho, tau_x, beta
  M2_pulse   : robustness variant, unit pulses dx = dK, a = 1 fixed.
               free: sigma_c, p, rho, tau_x
  M3_static  : driven no-attractor control (tether only; the response
               depends on beta and tau_x only through beta/tau_x, so the
               identified parameter is the tether rate r_t = beta/tau_x).
               free: sigma_c, p, rho, r_tether
Each variant exists with gamma = 0 (serial) or gamma = 1 (parallel), and
optionally free gamma or an onset delay delta (tier 2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from . import model as M

FIT_SEED = 730
M_FIT = 2048
M_FINAL = 16384


@dataclass(frozen=True)
class ParamDef:
    name: str
    lo: float
    hi: float
    log: bool = False

    def to_native(self, v: float) -> float:
        return float(np.exp(v)) if self.log else float(v)

    def bounds(self) -> tuple[float, float]:
        return (np.log(self.lo), np.log(self.hi)) if self.log else (self.lo, self.hi)


P_SIGMA_C = ParamDef("sigma_c", 1e-3, 0.5, log=True)
P_P = ParamDef("p", 0.3, 3.0)
P_RHO = ParamDef("rho", 2.0, 400.0, log=True)
P_TAU = ParamDef("tau_x", 0.02, 2.0, log=True)
P_BETA = ParamDef("beta", 0.1, 2000.0, log=True)
P_RT = ParamDef("r_tether", 0.5, 2000.0, log=True)   # beta/tau_x for M3
P_GAMMA = ParamDef("gamma", 0.0, 1.0)
P_DELTA = ParamDef("delta", 0.0, 0.035)


def param_defs(model: str, free_gamma: bool = False,
               free_delta: bool = False) -> list[ParamDef]:
    base = {
        "M1_offset": [P_SIGMA_C, P_P, P_RHO, P_TAU],
        "M2_tether": [P_SIGMA_C, P_P, P_RHO, P_TAU, P_BETA],
        "M2_pulse": [P_SIGMA_C, P_P, P_RHO, P_TAU],
        "M3_static": [P_SIGMA_C, P_P, P_RHO, P_RT],
    }[model]
    out = list(base)
    if free_gamma:
        out.append(P_GAMMA)
    if free_delta:
        out.append(P_DELTA)
    return out


def vec_to_theta(model: str, vec: np.ndarray, defs: list[ParamDef],
                 gamma: float) -> dict:
    theta = {"gamma": gamma}
    for d, v in zip(defs, vec):
        theta[d.name] = d.to_native(v)
    if model == "M3_static":
        # r_tether = beta / tau_x; the integrator wants (tau_x, beta).
        theta["tau_x"] = 1.0
        theta["beta"] = theta.pop("r_tether")
    return theta


def _negll(vec, model, defs, gamma, m_paths, counts_by_dur, r_max,
           dt=M.DT_EXPO):
    theta = vec_to_theta(model, np.asarray(vec, float), defs, gamma)
    try:
        ll, _ = M.total_loglik(model, theta, counts_by_dur, r_max,
                               m_paths=m_paths, dt=dt)
    except (FloatingPointError, ValueError):
        return 1e18
    return -ll if np.isfinite(ll) else 1e18


def fit_model(model: str, counts_by_dur: dict, r_max: int, *,
              gamma: float = 0.0, free_gamma: bool = False,
              free_delta: bool = False,
              maxiter: int = 45, popsize: int = 9, seed: int = FIT_SEED,
              m_fit: int = M_FIT, m_final: int = M_FINAL,
              workers: int = 12, x0: np.ndarray | None = None) -> dict:
    defs = param_defs(model, free_gamma=free_gamma, free_delta=free_delta)
    bounds = [d.bounds() for d in defs]
    args = (model, defs, gamma if not free_gamma else np.nan, m_fit,
            counts_by_dur, r_max)

    # DE at coarse settings (M=1024, dt=5 ms); Nelder-Mead polish and the
    # final score at dt=2.5 ms (validated: 2.5 -> 1.25 ms changes logL by ~3).
    de_args = args[:3] + (min(m_fit, 1024), args[4], args[5], 0.005)
    nm_args = args[:3] + (m_fit, args[4], args[5], 0.0025)

    t0 = time.time()
    if x0 is None:
        de = differential_evolution(
            _negll, bounds=bounds, args=de_args, maxiter=maxiter, popsize=popsize,
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.85, seed=seed,
            init="sobol", polish=False, updating="deferred", workers=workers,
        )
        x, f_de = de.x, _negll(de.x, *nm_args)
    else:
        x = np.asarray(x0, float)
        f_de = _negll(x, *nm_args)

    pol = minimize(_negll, x, args=nm_args, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 200, "xatol": 1e-4, "fatol": 0.05})
    x_best, f_best = (pol.x, float(pol.fun)) if pol.fun <= f_de else (x, f_de)

    # final high-accuracy evaluation (>= 4x the optimisation sample count)
    theta = vec_to_theta(model, x_best, defs, gamma if not free_gamma else np.nan)
    ll_final, lapses = M.total_loglik(model, theta, counts_by_dur, r_max,
                                      m_paths=m_final, dt=0.0025)
    ll_fit = -f_best
    k = len(defs) + len(M.DURATIONS)           # mechanism params + 5 lapses
    n_tot = sum(int(c.sum()) for c in counts_by_dur.values())
    return {
        "model": model, "gamma": gamma, "free_gamma": free_gamma,
        "free_delta": free_delta,
        "theta": theta, "lapses": lapses, "x_opt": x_best.tolist(),
        "param_names": [d.name for d in defs],
        "logL_fit": ll_fit, "logL": ll_final, "k": k, "n": n_tot,
        "AIC": 2 * k - 2 * ll_final, "BIC": k * np.log(n_tot) - 2 * ll_final,
        "m_fit": m_fit, "m_final": m_final,
        "wall_min": (time.time() - t0) / 60.0,
    }


def save_fit(fit: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fit, f, indent=1, default=float)
