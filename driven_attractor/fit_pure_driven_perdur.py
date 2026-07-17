"""Per-duration fit of the pure driven model (M2_tether, K(t)-driven, no sigma
channel) for Figure 2 panel b. One duration per process (run in parallel).

    .venv/bin/python driven_attractor/fit_pure_driven_perdur.py --dur 160
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D, model as M, fitting as F  # noqa: E402

OUT = ROOT / "driven_attractor" / "output" / "fits"
# free params of the pure-driven (serial) model: sigma_c, p, rho, tau_x, beta
DEFS = F.param_defs("M2_tether")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=int, required=True)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    d = args.dur
    di = list(M.DURATIONS).index(d)

    dat = D.load(); r_max = D.r_max(dat)
    counts_d = D.counts_by_duration(dat, r_max)[d]

    def negll(vec, m_paths, dt):
        theta = F.vec_to_theta("M2_tether", np.asarray(vec, float), DEFS, 0.0)
        try:
            ev = M.evaluate_model("M2_tether", theta, m_paths=m_paths, dt=dt)
        except (FloatingPointError, ValueError):
            return 1e18
        _, ll = M.profile_lapse(ev.p_wells[di], ev.p_guess[di], counts_d, r_max)
        return -ll if np.isfinite(ll) else 1e18

    bounds = [dd.bounds() for dd in DEFS]
    de = differential_evolution(lambda v: negll(v, 1024, 0.005), bounds=bounds,
                                maxiter=args.maxiter, popsize=args.popsize, tol=1e-6,
                                mutation=(0.5, 1.0), recombination=0.85, seed=7 + d,
                                init="sobol", polish=False, updating="deferred",
                                workers=1)
    pol = minimize(lambda v: negll(v, 2048, 0.0025), de.x, method="Nelder-Mead",
                   bounds=bounds, options={"maxiter": 150, "xatol": 1e-4, "fatol": 0.05})
    x = pol.x if pol.fun <= negll(de.x, 2048, 0.0025) else de.x
    ll = -negll(x, 16384, 0.0025)
    theta = F.vec_to_theta("M2_tether", x, DEFS, 0.0)
    k = len(DEFS) + 1                                  # mechanism + lapse
    n = int(counts_d.sum())
    rec = {"model": "pure_driven", "duration_ms": d, "logL": ll, "k": k,
           "AIC": 2 * k - 2 * ll, "theta": theta, "x_opt": x.tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"pure_driven_perdur_{d}.json", "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(f"[done] {d}ms: logL {ll:.1f} AIC {rec['AIC']:.1f}  "
          f"theta { {kk: round(vv,4) for kk,vv in theta.items()} }", flush=True)


if __name__ == "__main__":
    main()
