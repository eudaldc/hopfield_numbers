"""Constrained trial-level fits: shared attractor landscape across durations.

Reviewer 1 (point 8) asked whether the landscape is a stable property of the
representational system: the direct test is a constrained model in which the
landscape parameters are shared across exposure durations, compared against the
per-duration fit. For the winner (``gaussian_freebeta_attractor``) the landscape is
(sigma_c, beta); the sensory channel (s0, lambda0) and the lapse stay per-duration
in every variant.

Variants: share (sigma_c, beta) [both duration sets]; share beta only; share
sigma_c only [all-durations set]. The 80-640 ms subset excludes the saturated
40 ms flash, whose landscape is the clear outlier.

Strategy: warm-start candidates for the shared parameters (each duration's
per-duration fitted value + the mean), inner per-duration differential evolution
over the nuisance parameters, then monotone alternation (Nelder-Mead on the shared
parameters given the nuisances / per-duration Nelder-Mead on the nuisances given
the shared) until the summed fit-grid logL improves by < 0.5. Reported logL/AIC/BIC
use the accurate 1601-node grid, matching the headline ladder.

    .venv/bin/python -m scripts.trial_level_shared_landscape           # full run
    .venv/bin/python -m scripts.trial_level_shared_landscape --smoke   # tiny budget, one variant
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

OUT_CSV = T.OUT_DIR / "shared_landscape.csv"
OUT_THETA = T.OUT_DIR / "shared_landscape_theta.csv"

SPEC = T.MODEL_BY_KEY[T.WINNER_KEY]
ORDER = SPEC.param_names + ("lapse",)  # canonical theta-vector order
BOUNDS = dict(zip(SPEC.param_names, SPEC.bounds)) | {"lapse": T.LAPSE_BOUNDS}

VARIANTS: dict[str, tuple[str, ...]] = {
    "shared_sigma_c_beta": ("sigma_c", "beta"),
    "shared_beta": ("beta",),
    "shared_sigma_c": ("sigma_c",),
}
DURSETS: dict[str, tuple[int, ...]] = {
    "all5": T.DURATIONS,
    "80_640": tuple(d for d in T.DURATIONS if d >= 80),
}
# (variant, durset) runs; single-parameter shares only on the full set
RUNS = (
    ("shared_sigma_c_beta", "all5"),
    ("shared_sigma_c_beta", "80_640"),
    ("shared_beta", "all5"),
    ("shared_sigma_c", "all5"),
)

N_WORKERS = 20
_W: dict = {}


def _init_worker(fit_nodes: int) -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    _W["r_max"] = r_max
    _W["data"] = T.build_duration_data(trial, r_max)
    _W["quad_fit"] = T.make_quadrature(fit_nodes, kind="grid")


def _vec(shared: dict, nuis: np.ndarray, nuis_names: tuple[str, ...]) -> np.ndarray:
    th = dict(shared) | dict(zip(nuis_names, np.asarray(nuis, float)))
    return np.array([th[p] for p in ORDER])


def _ll(shared: dict, nuis: np.ndarray, nuis_names: tuple[str, ...], d: int) -> float:
    return T.response_loglik(SPEC, _vec(shared, nuis, nuis_names),
                             _W["quad_fit"], _W["data"][d].counts, _W["r_max"])


def _fit_nuis_de(args) -> tuple[int, int, np.ndarray, float]:
    cand_idx, shared, nuis_names, d, seed, maxiter, popsize = args
    bounds = [BOUNDS[p] for p in nuis_names]

    def neg(v: np.ndarray) -> float:
        return -_ll(shared, v, nuis_names, d)

    de = differential_evolution(neg, bounds=bounds, maxiter=maxiter, popsize=popsize,
                                tol=1e-4, mutation=(0.5, 1.0), recombination=0.85,
                                seed=seed, init="latinhypercube", polish=False,
                                updating="deferred", workers=1)
    pol = minimize(neg, de.x, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-4})
    x, f = (pol.x, pol.fun) if pol.fun <= de.fun else (de.x, de.fun)
    return cand_idx, d, np.asarray(x, float), -float(f)


def _polish_nuis(args) -> tuple[int, np.ndarray, float]:
    shared, nuis_names, d, x0 = args
    bounds = [BOUNDS[p] for p in nuis_names]

    def neg(v: np.ndarray) -> float:
        return -_ll(shared, v, nuis_names, d)

    pol = minimize(neg, x0, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-4})
    if pol.fun <= neg(np.asarray(x0, float)):
        return d, np.asarray(pol.x, float), -float(pol.fun)
    return d, np.asarray(x0, float), float(_ll(shared, x0, nuis_names, d))


def _ncrit(sigma_c: float, beta: float) -> float:
    ns = np.linspace(1.0, 60.0, 60000)
    s = np.power(sigma_c / np.log10(1.0 + 1.0 / ns), beta)
    above = np.where(s >= 1.0)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return 1.0
    x0, x1, y0, y1 = ns[i - 1], ns[i], s[i - 1], s[i]
    return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))


def run_one(variant: str, dsname: str, pool: Pool, th_free: pd.DataFrame,
            *, de_maxiter: int, de_popsize: int, max_rounds: int) -> tuple[dict, list[dict]]:
    shared_names = VARIANTS[variant]
    durs = DURSETS[dsname]
    nuis_names = tuple(p for p in ORDER if p not in shared_names)

    cands = [{p: float(th_free.loc[d, p]) for p in shared_names} for d in durs]
    cands.append({p: float(th_free.loc[list(durs), p].mean()) for p in shared_names})

    tasks = [(ci, cand, nuis_names, d, T.SEED + 101 * ci + d, de_maxiter, de_popsize)
             for ci, cand in enumerate(cands) for d in durs]
    t0 = time.time()
    results = pool.map(_fit_nuis_de, tasks)
    per_cand: dict[int, dict] = {}
    for ci, d, x, ll in results:
        per_cand.setdefault(ci, {})[d] = (x, ll)
    totals = {ci: sum(v[1] for v in dd.values()) for ci, dd in per_cand.items()}
    best_ci = max(totals, key=totals.get)
    shared = dict(cands[best_ci])
    nuis = {d: per_cand[best_ci][d][0] for d in durs}
    total = totals[best_ci]
    print(f"  [{variant}/{dsname}] warm-start best candidate {best_ci} "
          f"shared={ {k: round(v, 4) for k, v in shared.items()} } logL={total:.1f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    for rnd in range(max_rounds):
        prev = total
        sh_bounds = [BOUNDS[p] for p in shared_names]

        def neg_shared(sv: np.ndarray) -> float:
            sh = dict(zip(shared_names, np.asarray(sv, float)))
            return -sum(_ll(sh, nuis[d], nuis_names, d) for d in durs)

        res = minimize(neg_shared, np.array([shared[p] for p in shared_names]),
                       method="Nelder-Mead", bounds=sh_bounds,
                       options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-3})
        if -res.fun >= total:
            shared = dict(zip(shared_names, np.asarray(res.x, float)))
            total = -float(res.fun)

        polished = pool.map(_polish_nuis, [(shared, nuis_names, d, nuis[d]) for d in durs])
        nuis = {d: x for d, x, _ in polished}
        total = sum(ll for _, _, ll in polished)
        print(f"  [{variant}/{dsname}] round {rnd + 1}: logL={total:.1f} "
              f"(+{total - prev:.2f})", flush=True)
        if total - prev < 0.5:
            break

    # score at the accurate grid, matching the headline ladder
    quad_eval = T.make_quadrature(1601, kind="grid")
    data = _W["data"]
    logl_eval = sum(
        T.response_loglik(SPEC, _vec(shared, nuis[d], nuis_names),
                          quad_eval, data[d].counts, _W["r_max"])
        for d in durs
    )
    k = len(shared_names) + len(nuis_names) * len(durs)
    n_tot = sum(data[d].n_total for d in durs)
    aic = 2 * k - 2 * logl_eval
    bic = k * np.log(n_tot) - 2 * logl_eval

    free_ll = float(th_free.loc[list(durs), "resp_logL"].sum())
    free_k = (len(SPEC.param_names) + 1) * len(durs)
    free_aic = 2 * free_k - 2 * free_ll
    free_bic = free_k * np.log(n_tot) - 2 * free_ll

    row = {
        "variant": variant, "durations": dsname, "shared_params": "+".join(shared_names),
        "k": k, "logL": logl_eval, "AIC": aic, "BIC": bic,
        "free_k": free_k, "free_logL": free_ll, "free_AIC": free_aic, "free_BIC": free_bic,
        "dAIC_constrained_minus_free": aic - free_aic,
        "dBIC_constrained_minus_free": bic - free_bic,
    }
    for p in shared_names:
        row[f"shared_{p}"] = float(shared[p])
    if set(shared_names) == {"sigma_c", "beta"}:
        row["ncrit_shared"] = _ncrit(shared["sigma_c"], shared["beta"])

    theta_rows = []
    for d in durs:
        tr = {"variant": variant, "durations": dsname, "duration_ms": d}
        full = dict(shared) | dict(zip(nuis_names, nuis[d]))
        tr |= {p: float(full[p]) for p in ORDER}
        tr["ncrit"] = _ncrit(full["sigma_c"], full["beta"])
        theta_rows.append(tr)
    return row, theta_rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    fit_nodes = 101 if args.smoke else 201
    de_maxiter = 6 if args.smoke else 25
    de_popsize = 5 if args.smoke else 10
    max_rounds = 2 if args.smoke else 8
    runs = RUNS[:1] if args.smoke else RUNS

    _init_worker(fit_nodes)  # parent needs data/quad for the shared step
    th_free = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")

    rows, theta_rows = [], []
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(fit_nodes,)) as pool:
        for variant, dsname in runs:
            row, trs = run_one(variant, dsname, pool, th_free,
                               de_maxiter=de_maxiter, de_popsize=de_popsize,
                               max_rounds=max_rounds)
            rows.append(row)
            theta_rows.extend(trs)
            print(f"  => {variant}/{dsname}: k={row['k']} vs free {row['free_k']}, "
                  f"dAIC={row['dAIC_constrained_minus_free']:+.1f} "
                  f"dBIC={row['dBIC_constrained_minus_free']:+.1f}", flush=True)

    out = pd.DataFrame(rows)
    theta = pd.DataFrame(theta_rows)
    if args.smoke:
        print(out.round(2).to_string(index=False))
        print(theta.round(4).to_string(index=False))
        return
    out.to_csv(OUT_CSV, index=False)
    theta.to_csv(OUT_THETA, index=False)
    pd.set_option("display.width", 250)
    print("\n=== Constrained (shared-landscape) vs per-duration fits ===")
    cols = ["variant", "durations", "k", "free_k", "logL", "free_logL",
            "dAIC_constrained_minus_free", "dBIC_constrained_minus_free"]
    print(out[cols].round(1).to_string(index=False))
    print("\n=== Fitted parameters per duration (constrained runs) ===")
    print(theta.round(4).to_string(index=False))
    print(f"\nSaved {OUT_CSV} and {OUT_THETA}")
    print("\nReading: dAIC > 0 means the constrained model fits worse, i.e. the "
          "data reject a duration-independent landscape; the interesting question "
          "is by how much, and whether beta or sigma_c carries the change.")


if __name__ == "__main__":
    main()
