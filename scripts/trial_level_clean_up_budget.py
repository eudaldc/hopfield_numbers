"""Profile the clean-up budget K (= MAX_STEPS) under the primary joint fit.

The clean-up is run for a fixed budget of K steps (the give-up point). On the
flat large-numerosity landscape, some trajectories do not settle within a
behaviourally plausible iteration budget, so K implements a finite stopping
policy and prevents implausibly long predicted response times. This script
identifies K from the RESPONSE channel of the joint (shared-landscape) fit: for a
grid of K it re-fits the winner's shared landscape (sigma_c, p shared across the
five durations; sensory channel and lapse per duration) and records the total
response log-likelihood and the width index n_width.

Result (see ``output/.../clean_up_budget_profile.csv``, quoted in SM2): the joint
response likelihood has a broad, shallow optimum near K~120; K=100 lies within
~5 log-likelihood units, whereas K<=60 and K>=250 are disfavoured by tens to
hundreds of units. n_width barely moves (within ~0.3). Because the budget shapes
the large-n readout, it is pinned by accuracy, not by the response times, so
using it to cap the settling time is not circular.

    .venv/bin/python -m scripts.trial_level_clean_up_budget

Slow (~one joint re-fit per K); the committed CSV is the cached result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

SPEC = T.MODEL_BY_KEY[T.WINNER_KEY]        # params: sigma_c, s0, lambda0, pow
K_GRID = [40, 60, 100, 120, 140, 160, 250]
NODES = 201
SC_B, S0_B, LAM_B, POW_B = SPEC.bounds
LAP_B = T.LAPSE_BOUNDS
OUT_CSV = T.OUT_DIR / "clean_up_budget_profile.csv"


def _vec(shared, nuis):  # theta_vec order = (sigma_c, s0, lambda0, pow, lapse)
    return np.array([shared[0], nuis[0], nuis[1], shared[1], nuis[2]])


def n_width_of(sigma_c, pow_):
    return float(sigma_c ** (-1.0 / pow_))


def _fit_nuisance(shared, x0n, quad, counts, r_max):
    def neg(v):
        return -T.response_loglik(SPEC, _vec(shared, v), quad, counts, r_max)
    pol = minimize(neg, x0n, method="Nelder-Mead", bounds=[S0_B, LAM_B, LAP_B],
                   options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-4})
    return pol.x


def _fit_shared(shared0, nuis, quad, data, r_max, durs):
    def neg(s):
        return -sum(T.response_loglik(SPEC, _vec(s, nuis[d]), quad, data[d].counts, r_max)
                    for d in durs)
    pol = minimize(neg, shared0, method="Nelder-Mead", bounds=[SC_B, POW_B],
                   options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-3})
    return pol.x, -float(pol.fun)


def joint_fit(K, shared0, nuis0, quad, data, r_max, durs, rounds=4):
    """Coordinate descent: nuisance per duration, then the shared landscape."""
    T.MAX_STEPS = K
    shared = np.asarray(shared0, float)
    nuis = {d: np.asarray(nuis0[d], float) for d in durs}
    prev, tot = -np.inf, -np.inf
    for _ in range(rounds):
        for d in durs:
            nuis[d] = _fit_nuisance(shared, nuis[d], quad, data[d].counts, r_max)
        shared, tot = _fit_shared(shared, nuis, quad, data, r_max, durs)
        if abs(tot - prev) < 0.5:
            break
        prev = tot
    return shared, nuis, tot


def main() -> None:
    durs = tuple(T.DURATIONS)
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    quad = T.make_quadrature(NODES, kind="grid")

    th = pd.read_csv(T.OUT_DIR / "shared_landscape_ladder_theta.csv")
    th = th[th.model_key == T.WINNER_KEY].set_index("duration_ms")
    shared0 = np.array([th.iloc[0]["sigma_c"], th.iloc[0]["pow"]])
    nuis0 = {d: np.array([th.loc[d]["s0"], th.loc[d]["lambda0"], th.loc[d]["lapse"]]) for d in durs}

    # continuation warm-start: solve the selected K=120 budget, then sweep out
    sol = {}
    anchor = 120
    sh, nu, tot = joint_fit(anchor, shared0, nuis0, quad, data, r_max, durs)
    sol[anchor] = (sh, nu, tot)
    for K in [k for k in K_GRID if k > anchor]:
        base = sol[max(sol)]
        sol[K] = joint_fit(K, base[0], base[1], quad, data, r_max, durs)
    for K in [k for k in reversed(K_GRID) if k < anchor]:
        base = sol[min(sol)]
        sol[K] = joint_fit(K, base[0], base[1], quad, data, r_max, durs)

    rows = [dict(K=K, logL=sol[K][2], sigma_c=float(sol[K][0][0]), pow=float(sol[K][0][1]),
                 n_width=n_width_of(*sol[K][0])) for K in K_GRID]
    df = pd.DataFrame(rows)
    df["dlogL"] = df["logL"] - df["logL"].max()
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
