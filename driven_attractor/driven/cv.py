"""Participant-blocked cross-validation for M0 (current model) and the
driven family, on identical folds.

Folds: participants (uniqueid) split into K folds with a fixed seed. For
each fold, every model is refit on the training participants' count tables
(warm-started from its full-data optimum) and scored on the held-out
participants' trials. Reported: held-out log predictive density (lpd) in
total, per fold, and per participant (so "how many participants does each
model predict better" can be answered, not only the pooled sum).

The lapse (per duration) is a fitted parameter: it is estimated on the
training fold and applied to the test fold, for every model alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import fitting as F
from . import model as M

import scripts.trial_level_model_comparison as T

CV_SEED = 20260711
N_FOLDS = 5


def make_folds(trial: pd.DataFrame, n_folds: int = N_FOLDS,
               seed: int = CV_SEED) -> pd.Series:
    ids = np.sort(trial["uniqueid"].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids.size)
    fold_of = {uid: int(perm[i] % n_folds) for i, uid in enumerate(ids)}
    return trial["uniqueid"].map(fold_of)


def counts_from(trial: pd.DataFrame, r_max: int) -> dict[int, np.ndarray]:
    out = {}
    for d in M.DURATIONS:
        sub = trial[trial["exp_dur"] == d]
        c = np.zeros((M.N_SHOWN.size, r_max))
        np.add.at(c, (sub["n_shown"].to_numpy(int) - 1,
                      np.clip(sub["n_guess"].to_numpy(int), 1, r_max) - 1), 1.0)
        out[d] = c
    return out


# ── per-trial log-probability lookup ──────────────────────────────────────

def trial_lpd(trial_test: pd.DataFrame, pf_by_dur: dict[int, np.ndarray]) -> np.ndarray:
    """log p(r | n, d) per test trial from full (15, r_max) tables."""
    lp = np.empty(len(trial_test))
    for i, (d, n, r) in enumerate(zip(trial_test["exp_dur"].to_numpy(int),
                                      trial_test["n_shown"].to_numpy(int),
                                      trial_test["n_guess"].to_numpy(int))):
        lp[i] = np.log(pf_by_dur[d][n - 1, min(r, pf_by_dur[d].shape[1]) - 1])
    return lp


# ── M0 (current model) fold refits ────────────────────────────────────────

def m0_tables(theta_by_dur: dict[int, dict], lapse_by_dur: dict[int, float],
              r_max: int, spec=None, nodes: int = 1601) -> dict[int, np.ndarray]:
    spec = spec or T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(nodes, kind="grid")
    out = {}
    for d in M.DURATIONS:
        grid = T.compute_grid(spec, theta_by_dur[d], quad)
        out[d] = T.apply_lapse(grid["p_resp"], lapse_by_dur[d], r_max)
    return out


def fit_m0_perdur(counts: dict[int, np.ndarray], r_max: int,
                  warm: dict[int, np.ndarray], budget: int = 400) -> dict:
    """Per-duration M0 refit (NM from the full-data optimum)."""
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(201, kind="grid")
    quad_hi = T.make_quadrature(1601, kind="grid")
    bounds = list(spec.bounds) + [T.LAPSE_BOUNDS]
    theta_by_dur, lapse_by_dur, ll = {}, {}, 0.0
    for d in M.DURATIONS:
        def neg(v):
            return -T.response_loglik(spec, v, quad, counts[d], r_max)
        res = minimize(neg, warm[d], method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": budget, "xatol": 1e-5, "fatol": 1e-3})
        v = res.x if res.fun <= neg(warm[d]) else np.asarray(warm[d])
        theta_by_dur[d] = dict(zip(spec.param_names, v[:-1]))
        lapse_by_dur[d] = float(v[-1])
        ll += T.response_loglik(spec, v, quad_hi, counts[d], r_max)
    return {"theta": theta_by_dur, "lapse": lapse_by_dur, "logL_train": ll}


def fit_m0_shared(counts: dict[int, np.ndarray], r_max: int,
                  warm_shared: np.ndarray, warm_nuis: dict[int, np.ndarray],
                  rounds: int = 1, budget: int = 150) -> dict:
    """Shared-landscape M0 (sigma_c, p shared; s0, lambda0, lapse per dur)."""
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(201, kind="grid")
    quad_hi = T.make_quadrature(1601, kind="grid")
    sh_bounds = [spec.bounds[0], spec.bounds[3]]           # sigma_c, pow
    nu_bounds = [spec.bounds[1], spec.bounds[2], T.LAPSE_BOUNDS]

    shared = np.asarray(warm_shared, float)                # [sigma_c, pow]
    nuis = {d: np.asarray(warm_nuis[d], float) for d in M.DURATIONS}

    def full_vec(sh, nu):
        return np.array([sh[0], nu[0], nu[1], sh[1], nu[2]])

    def ll_d(sh, nu, d, q):
        return T.response_loglik(spec, full_vec(sh, nu), q, counts[d], r_max)

    for _ in range(rounds):
        for d in M.DURATIONS:
            res = minimize(lambda v: -ll_d(shared, v, d, quad), nuis[d],
                           method="Nelder-Mead", bounds=nu_bounds,
                           options={"maxiter": budget, "xatol": 1e-5, "fatol": 1e-3})
            if -res.fun >= ll_d(shared, nuis[d], d, quad):
                nuis[d] = res.x
        res = minimize(lambda s: -sum(ll_d(s, nuis[d], d, quad)
                                      for d in M.DURATIONS), shared,
                       method="Nelder-Mead", bounds=sh_bounds,
                       options={"maxiter": budget, "xatol": 1e-6, "fatol": 1e-3})
        if -res.fun >= sum(ll_d(shared, nuis[d], d, quad) for d in M.DURATIONS):
            shared = res.x

    theta_by_dur = {d: {"sigma_c": shared[0], "s0": nuis[d][0],
                        "lambda0": nuis[d][1], "pow": shared[1]}
                    for d in M.DURATIONS}
    lapse_by_dur = {d: float(nuis[d][2]) for d in M.DURATIONS}
    ll = sum(ll_d(shared, nuis[d], d, quad_hi) for d in M.DURATIONS)
    return {"theta": theta_by_dur, "lapse": lapse_by_dur, "logL_train": ll,
            "shared": shared.tolist()}


# ── driven-family fold refits ─────────────────────────────────────────────

def fit_driven_fold(model: str, gamma: float, counts: dict, r_max: int,
                    x0: np.ndarray, budget: int = 250,
                    m_fit: int = 1024) -> dict:
    defs = F.param_defs(model)
    bounds = [d.bounds() for d in defs]
    args = (model, defs, gamma, m_fit, counts, r_max, 0.0025)
    res = minimize(F._negll, x0, args=args, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": budget, "xatol": 1e-4, "fatol": 0.05})
    x = res.x if res.fun <= F._negll(x0, *args) else np.asarray(x0, float)
    theta = F.vec_to_theta(model, x, defs, gamma)
    return {"theta": theta, "x_opt": x.tolist()}


def driven_tables(model: str, theta: dict, counts_train: dict, r_max: int,
                  m_paths: int = F.M_FINAL) -> dict[int, np.ndarray]:
    """Final response tables with lapse profiled on the TRAIN counts."""
    ev = M.evaluate_model(model, theta, m_paths=m_paths, dt=0.0025)
    out = {}
    for i, d in enumerate(M.DURATIONS):
        lapse, _ = M.profile_lapse(ev.p_wells[i], ev.p_guess[i],
                                   counts_train[d], r_max)
        out[d] = M.p_final_full(ev.p_wells[i], ev.p_guess[i], lapse, r_max)
    return out
