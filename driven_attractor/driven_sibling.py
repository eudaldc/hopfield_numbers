"""The minimal driven sibling of M0 (matched sensory channel).

M0 (winner): y0 ~ N(n, sigma(n,d)); x(0)=y0; relax on V; softmax readout.
Sibling:     evidence accumulates over [0,d] to an offset value with the SAME
             law N(n, sigma(n,d)), but drives the attractor DURING exposure via
             a soft tether; after offset the tether is removed and the state
             relaxes on the same V. One extra parameter, the tether strength
             beta (with tau_x = 1 absorbed). As beta -> inf it reproduces M0.

Drive: a Brownian accumulation whose running estimate is
    u(t) = n + sigma_B(n) * W(t)/t + s00 * xi,   sigma_B(n) = sqrt(s0_d n^lam_d d)
so that Var(u(d)) = s00^2 + s0_d n^lam_d = sigma(n,d)^2  (M0's per-duration
noise exactly), with a static component s00*xi shared across the path. Same
fixed Sobol common random numbers as the rest of driven_attractor/.

    .venv/bin/python driven_attractor/driven_sibling.py --validate
    .venv/bin/python driven_attractor/driven_sibling.py --fit-beta --params perdur
    .venv/bin/python driven_attractor/driven_sibling.py --fit-beta --params shared
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

S00 = T.S00
DT = 0.0025
DUR_S = M.DUR_S
N_SHOWN = M.N_SHOWN


def m0_params(which: str) -> dict[int, dict]:
    if which == "perdur":
        df = pd.read_csv(ROOT / "legacy/trial_level_per_duration/output/pow_theta.csv")
        return {int(r.duration_ms): {"sigma_c": r.sigma_c, "p": r["pow"],
                                     "s0": r.s0, "lam": r.lambda0}
                for _, r in df.iterrows()}
    df = pd.read_csv(ROOT / "output/trial_level_model_comparison/"
                     "shared_landscape_ladder_theta.csv")
    df = df[df.model_key == "gaussian_pow_attractor"]
    return {int(r.duration_ms): {"sigma_c": r.sigma_c, "p": r["pow"],
                                 "s0": r.s0, "lam": r.lambda0}
            for _, r in df.iterrows()}


def sibling_pwells(par: dict, beta: float, d_ms: int, m_paths: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """p_wells (15, 19) and settle info for one duration under the driven sibling."""
    d = d_ms / 1000.0
    sig = M.sigma_wells(par["sigma_c"], par["p"])
    n = N_SHOWN.astype(float)
    sigma_B = np.sqrt(np.maximum(1e-12, par["s0"] * np.power(n, par["lam"]) * d))  # (15,)
    # common random numbers
    rng = np.random.default_rng(seed)
    half = m_paths // 2
    z = rng.standard_normal((int(round(d / DT)), half))
    z = np.concatenate([z, -z], axis=1)
    W = np.cumsum(z * np.sqrt(DT), axis=0)              # (T, m)
    xi = rng.standard_normal(half); xi = np.concatenate([xi, -xi])   # (m,)
    n_steps = W.shape[0]
    m = m_paths
    x = np.zeros((N_SHOWN.size, m))
    h = DT  # tau_x = 1
    for s in range(1, n_steps + 1):
        t = (s - 0.5) * DT
        u = n[:, None] + sigma_B[:, None] * (W[s - 1][None, :] / t) + S00 * xi[None, :]
        u = np.clip(u, -5.0, 60.0)
        x = M._guarded_step(x.reshape(-1), u.reshape(-1), sig, beta, h,
                            attractor=True,
                            gate=np.ones(x.size, bool)).reshape(N_SHOWN.size, m)
    # post-offset: M0's exact clean-up operator (120 clamped steps), so that
    # beta -> inf reproduces M0 (x(d) -> y0 -> _cleanup(y0) = M0 readout).
    x_star, _, _ = T._cleanup(x.reshape(-1), family="gaussian", sigma=sig,
                              max_steps=T.MAX_STEPS)
    probs = M.readout_probs(x_star, sig).reshape(N_SHOWN.size, m, -1).mean(axis=1)
    return probs, np.zeros(N_SHOWN.size)


def total_loglik(params: dict[int, dict], beta: float, counts: dict, r_max: int,
                 m_paths: int, seed: int = 909) -> tuple[float, dict]:
    tot, lapses = 0.0, {}
    for d in M.DURATIONS:
        pw, pg = sibling_pwells(params[d], beta, d, m_paths, seed + d)
        lapse, ll = M.profile_lapse(pw, pg, counts[d], r_max)
        lapses[d] = lapse
        tot += ll
    return tot, lapses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fit-beta", action="store_true")
    ap.add_argument("--params", default="perdur", choices=["perdur", "shared"])
    ap.add_argument("--m", type=int, default=8192)
    args = ap.parse_args()

    dat = D.load(); r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)
    par = m0_params(args.params)

    # M0 reference logL at these params (batch, exact quadrature)
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")
    m0_ll = {}
    for d in M.DURATIONS:
        th = {"sigma_c": par[d]["sigma_c"], "s0": par[d]["s0"],
              "lambda0": par[d]["lam"], "pow": par[d]["p"]}
        grid = T.compute_grid(spec, th, quad)
        lapse, ll = M.profile_lapse(grid["p_resp"], np.zeros(15), counts[d], r_max)
        m0_ll[d] = ll
    m0_tot = sum(m0_ll.values())
    print(f"M0 ({args.params}) reference logL = {m0_tot:.1f}", flush=True)

    if args.validate:
        for beta in (50.0, 500.0, 5e3, 5e4):
            tot, _ = total_loglik(par, beta, counts, r_max, args.m)
            print(f"  sibling beta={beta:8.0f}: logL {tot:.1f}  "
                  f"(vs M0 {tot - m0_tot:+.1f})", flush=True)
        return

    if args.fit_beta:
        def neg(logbeta):
            tot, _ = total_loglik(par, float(np.exp(logbeta)), counts, r_max, args.m)
            return -tot
        res = minimize_scalar(neg, bounds=(np.log(1.0), np.log(1e5)),
                              method="bounded", options={"xatol": 0.05})
        beta = float(np.exp(res.x))
        tot, lapses = total_loglik(par, beta, counts, r_max, args.m)
        k_m0 = 25 if args.params == "perdur" else 17
        k_sib = k_m0 + 1
        d_aic = (2 * k_sib - 2 * tot) - (2 * k_m0 - 2 * m0_tot)
        print(f"\n=== driven sibling, {args.params} params, beta fitted ===")
        print(f"  fitted beta = {beta:.1f}")
        print(f"  sibling logL = {tot:.1f}  (M0 {m0_tot:.1f}; dlogL {tot - m0_tot:+.1f})")
        print(f"  dAIC (sibling - M0, +1 param) = {d_aic:+.1f}")
        pd.DataFrame([{"params": args.params, "beta": beta, "logL": tot,
                       "m0_logL": m0_tot, "dlogL": tot - m0_tot,
                       "k_sibling": k_sib, "dAIC": d_aic}]).to_csv(
            ROOT / f"driven_attractor/output/driven_sibling_{args.params}.csv",
            index=False)


if __name__ == "__main__":
    main()
