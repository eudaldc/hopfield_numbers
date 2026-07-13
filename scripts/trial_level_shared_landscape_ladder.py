"""Joint-training model comparison: one shared landscape across all durations.

Both reviewers press on the per-duration fitting strategy. Reviewer 1 (point 8)
asks directly for a constrained model in which the well-scale is shared across
exposure durations, with only the sensory channel varying, compared to the free
fit. Reviewer 2 (point 4) asks whether we are claiming the mechanism itself
changes with exposure time. This script answers both at the level of the whole
model ladder shown in Figure 3.

For every well-model (the two well families x clean-up/static) we fit a single
representational landscape shared across the five durations, leaving only the
sensory channel (s0, lambda0) and the lapse free per duration. The landscape
parameters are the well-shape parameters: (sigma_c, beta) for the frequency
(Benford) wells and (sigma_c, pow) for the power-law wells. Exposure enters only
through the sensory channel, so the model is one mechanism whose input sharpens
with viewing time, not a different mechanism per duration.

Weber/ANS and the normative log-prior proxy have no landscape separable from
their sensory noise (the Weber fraction *is* the noise), so their most-constrained
sensible form already fits a per-duration scale. We carry their free per-duration
fit unchanged. This is the conservative comparison: our models are handicapped
with a single landscape while the benchmarks keep full per-duration freedom.

    .venv/bin/python -m scripts.trial_level_shared_landscape_ladder
    .venv/bin/python -m scripts.trial_level_shared_landscape_ladder --smoke
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
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT_CSV = T.OUT_DIR / "shared_landscape_ladder.csv"
OUT_THETA = T.OUT_DIR / "shared_landscape_ladder_theta.csv"

# ── The six Figure-3 models ───────────────────────────────────────────────
# Well-models get a constrained shared-landscape fit; the two benchmarks keep
# their free per-duration fit (no separable landscape).
SPECS: dict[str, T.ModelSpec] = {
    "gaussian_freebeta_attractor": T.MODEL_BY_KEY["gaussian_freebeta_attractor"],
    "gaussian_benford_free_beta": T.ModelSpec(
        key="gaussian_benford_free_beta", label="Frequency wells, static",
        family="gaussian_freebeta", cleanup=False,
        param_names=("sigma_c", "s0", "lambda0", "beta"),
        bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0))),
    "gaussian_pow_attractor": T.ModelSpec(
        key="gaussian_pow_attractor", label="Power-law wells + clean-up",
        family="gaussian_pow", cleanup=True,
        param_names=("sigma_c", "s0", "lambda0", "pow"),
        bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0))),
    "gaussian_pow_readout": T.ModelSpec(
        key="gaussian_pow_readout", label="Power-law wells, static",
        family="gaussian_pow", cleanup=False,
        param_names=("sigma_c", "s0", "lambda0", "pow"),
        bounds=((1e-4, 1.0), (1e-6, 0.1), (0.1, 4.0), (0.1, 4.0))),
    "gaussian_const_attractor": T.MODEL_BY_KEY["gaussian_const_attractor"],
    "gaussian_const_readout": T.MODEL_BY_KEY["gaussian_const_readout"],
}
LANDSCAPE: dict[str, tuple[str, ...]] = {
    "gaussian_freebeta_attractor": ("sigma_c", "beta"),
    "gaussian_benford_free_beta": ("sigma_c", "beta"),
    "gaussian_pow_attractor": ("sigma_c", "pow"),
    "gaussian_pow_readout": ("sigma_c", "pow"),
    "gaussian_const_attractor": ("sigma_c",),
    "gaussian_const_readout": ("sigma_c",),
}
WELL_LABELS = {
    "gaussian_freebeta_attractor": "Frequency wells + clean-up",
    "gaussian_benford_free_beta": "Frequency wells, static",
    "gaussian_pow_attractor": "Power-law wells + clean-up",
    "gaussian_pow_readout": "Power-law wells, static",
    "gaussian_const_attractor": "Constant-width wells + clean-up",
    "gaussian_const_readout": "Constant-width wells, static",
}
BENCHMARKS = {
    "weber_ans": "Weber/ANS",
    "cp_log_prior_proxy": "Normative (log-prior)",
}

N_WORKERS = 20
_W: dict = {}


def _init_worker(fit_nodes: int) -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    _W["r_max"] = r_max
    _W["data"] = T.build_duration_data(trial, r_max)
    _W["quad_fit"] = T.make_quadrature(fit_nodes, kind="grid")
    _W["quad_eval"] = T.make_quadrature(1601, kind="grid")


# ── Free per-duration fit (one pool task per model x duration) ────────────
def _fit_free(args) -> tuple[str, int, dict, float, float, float]:
    key, d, seed, maxiter, popsize = args
    spec = SPECS[key]
    rf = T.fit_response(spec, _W["data"][d], _W["quad_fit"], _W["r_max"],
                        maxiter=maxiter, popsize=popsize, seed=seed,
                        quad_eval=_W["quad_eval"])
    return key, d, rf["theta"], rf["lapse"], rf["logL"], rf["BIC"]


# ── Constrained shared-landscape helpers ──────────────────────────────────
def _order(spec: T.ModelSpec) -> tuple[str, ...]:
    return spec.param_names + ("lapse",)


def _vec(spec: T.ModelSpec, shared: dict, nuis: np.ndarray,
         nuis_names: tuple[str, ...]) -> np.ndarray:
    th = dict(shared) | dict(zip(nuis_names, np.asarray(nuis, float)))
    return np.array([th[p] for p in _order(spec)])


def _ll(key: str, shared: dict, nuis: np.ndarray, nuis_names: tuple[str, ...],
        d: int, quad) -> float:
    spec = SPECS[key]
    return T.response_loglik(spec, _vec(spec, shared, nuis, nuis_names),
                             quad, _W["data"][d].counts, _W["r_max"])


def _bounds(spec: T.ModelSpec) -> dict:
    return dict(zip(spec.param_names, spec.bounds)) | {"lapse": T.LAPSE_BOUNDS}


def _fit_nuis_de(args) -> tuple[int, int, np.ndarray, float]:
    key, cand_idx, shared, nuis_names, d, seed, maxiter, popsize = args
    bounds = [_bounds(SPECS[key])[p] for p in nuis_names]

    def neg(v: np.ndarray) -> float:
        return -_ll(key, shared, v, nuis_names, d, _W["quad_fit"])

    de = differential_evolution(neg, bounds=bounds, maxiter=maxiter, popsize=popsize,
                                tol=1e-4, mutation=(0.5, 1.0), recombination=0.85,
                                seed=seed, init="latinhypercube", polish=False,
                                updating="deferred", workers=1)
    pol = minimize(neg, de.x, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-4})
    x, f = (pol.x, pol.fun) if pol.fun <= de.fun else (de.x, de.fun)
    return cand_idx, d, np.asarray(x, float), -float(f)


def _polish_nuis(args) -> tuple[int, np.ndarray, float]:
    key, shared, nuis_names, d, x0 = args
    bounds = [_bounds(SPECS[key])[p] for p in nuis_names]

    def neg(v: np.ndarray) -> float:
        return -_ll(key, shared, v, nuis_names, d, _W["quad_fit"])

    pol = minimize(neg, x0, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-4})
    if pol.fun <= neg(np.asarray(x0, float)):
        return d, np.asarray(pol.x, float), -float(pol.fun)
    return d, np.asarray(x0, float), float(_ll(key, shared, x0, nuis_names, d, _W["quad_fit"]))


def _n_width(shared_names: tuple[str, ...], shared: dict) -> float:
    """Continuous width index at which sigma_n crosses one."""
    if len(shared_names) == 1:
        return 1.0 if shared["sigma_c"] >= 1.0 else float("nan")
    ns = np.linspace(1.0, 60.0, 60000)
    if "beta" in shared_names:  # frequency (Benford) wells
        s = np.power(shared["sigma_c"] / np.log10(1.0 + 1.0 / ns), shared["beta"])
    else:  # power-law wells
        s = shared["sigma_c"] * np.power(ns, shared["pow"])
    above = np.where(s >= 1.0)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return 1.0
    x0, x1, y0, y1 = ns[i - 1], ns[i], s[i - 1], s[i]
    return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))


def fit_shared(key: str, pool: Pool, free_theta: dict, durs: tuple[int, ...],
               *, de_maxiter: int, de_popsize: int, max_rounds: int) -> tuple[dict, list[dict]]:
    """Constrained fit: share the landscape params across durs; sensory + lapse
    stay per duration. free_theta[d] is the per-duration free theta (warm start)."""
    spec = SPECS[key]
    shared_names = LANDSCAPE[key]
    nuis_names = tuple(p for p in _order(spec) if p not in shared_names)

    cands = [{p: float(free_theta[d][p]) for p in shared_names} for d in durs]
    cands.append({p: float(np.mean([free_theta[d][p] for d in durs])) for p in shared_names})

    tasks = [(key, ci, cand, nuis_names, d, T.SEED + 101 * ci + d, de_maxiter, de_popsize)
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
    print(f"  [{key}] warm-start cand {best_ci} shared="
          f"{ {k: round(v, 4) for k, v in shared.items()} } logL={total:.1f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    for rnd in range(max_rounds):
        prev = total
        sh_bounds = [_bounds(spec)[p] for p in shared_names]

        def neg_shared(sv: np.ndarray) -> float:
            sh = dict(zip(shared_names, np.asarray(sv, float)))
            return -sum(_ll(key, sh, nuis[d], nuis_names, d, _W["quad_fit"]) for d in durs)

        res = minimize(neg_shared, np.array([shared[p] for p in shared_names]),
                       method="Nelder-Mead", bounds=sh_bounds,
                       options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-3})
        if -res.fun >= total:
            shared = dict(zip(shared_names, np.asarray(res.x, float)))
            total = -float(res.fun)

        polished = pool.map(_polish_nuis, [(key, shared, nuis_names, d, nuis[d]) for d in durs])
        nuis = {d: x for d, x, _ in polished}
        total = sum(ll for _, _, ll in polished)
        print(f"  [{key}] round {rnd + 1}: logL={total:.1f} (+{total - prev:.2f})", flush=True)
        if total - prev < 0.5:
            break

    data = _W["data"]
    logl = sum(_ll(key, shared, nuis[d], nuis_names, d, _W["quad_eval"]) for d in durs)
    k = len(shared_names) + len(nuis_names) * len(durs)
    n_tot = sum(data[d].n_total for d in durs)
    row = {
        "model_key": key, "label": WELL_LABELS[key], "has_landscape": True,
        "shared_params": "+".join(shared_names),
        "shared_k": k, "shared_logL": logl,
        "shared_AIC": 2 * k - 2 * logl, "shared_BIC": k * np.log(n_tot) - 2 * logl,
        "n_width_shared": _n_width(shared_names, shared),
    }
    for p in shared_names:
        row[f"shared_{p}"] = float(shared[p])

    theta_rows = []
    for d in durs:
        full = dict(shared) | dict(zip(nuis_names, nuis[d]))
        tr = {"model_key": key, "duration_ms": d}
        tr |= {p: float(full[p]) for p in _order(spec)}
        theta_rows.append(tr)
    return row, theta_rows


def benchmark_row(key: str, resp: pd.DataFrame, n_by_dur: dict) -> dict:
    """Free per-duration AIC/BIC for a no-landscape benchmark, from the ladder.

    The benchmark has no landscape separable from its sensory noise, so its
    most-constrained sensible form already fits a per-duration scale; we carry
    its free per-duration fit unchanged. It is one model of the whole dataset
    (k per-duration parameters over all trials), so its BIC uses the total N,
    matching the joint fit of the well models (not a per-duration N)."""
    sub = resp[resp.model_key == key]
    spec = T.MODEL_BY_KEY[key]
    k_d = len(spec.param_names) + 1
    k = len(sub) * k_d
    n_tot = sum(n_by_dur[int(d)] for d in sub["duration_ms"])
    aic = float(sub["resp_AIC"].sum())
    logl = sum((2 * k_d - a) / 2 for a in sub["resp_AIC"])
    bic = k * np.log(n_tot) - 2 * logl
    return {"model_key": key, "label": BENCHMARKS[key], "has_landscape": False,
            "shared_params": "", "shared_k": k, "shared_logL": logl,
            "shared_AIC": aic, "shared_BIC": bic, "n_width_shared": np.nan}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--constant-width-only",
        action="store_true",
        help="Fit the two p=0 models and merge them into the existing ladder outputs.",
    )
    args = ap.parse_args(argv)

    fit_nodes = 31 if args.smoke and args.constant_width_only else (101 if args.smoke else 201)
    free_maxiter = 2 if args.smoke and args.constant_width_only else (8 if args.smoke else 35)
    de_maxiter = 1 if args.smoke and args.constant_width_only else (6 if args.smoke else 25)
    de_popsize = 3 if args.smoke and args.constant_width_only else (5 if args.smoke else 10)
    max_rounds = 1 if args.smoke and args.constant_width_only else (2 if args.smoke else 8)
    durs = (
        tuple(T.DURATIONS[:1])
        if args.smoke and args.constant_width_only
        else (tuple(T.DURATIONS[:2]) if args.smoke else tuple(T.DURATIONS))
    )
    if args.smoke:
        well_keys = [
            "gaussian_const_attractor"
            if args.constant_width_only
            else "gaussian_freebeta_attractor"
        ]
    elif args.constant_width_only:
        well_keys = ["gaussian_const_attractor", "gaussian_const_readout"]
    else:
        well_keys = list(SPECS)

    _init_worker(fit_nodes)  # parent needs data/quad for the shared step
    n_by_dur = {d: _W["data"][d].n_total for d in T.DURATIONS}

    rows, theta_rows = [], []
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(fit_nodes,)) as pool:
        # 1. Free per-duration fits (warm-starts + the free column of the table)
        free_tasks = [(key, d, T.SEED + d, free_maxiter, de_popsize)
                      for key in well_keys for d in durs]
        free_res = list(
            tqdm(
                pool.imap_unordered(_fit_free, free_tasks),
                total=len(free_tasks),
                desc="Free per-duration fits",
                unit="fit",
            )
        )
        free_theta: dict[str, dict] = {key: {} for key in well_keys}
        free_ll: dict[str, float] = {key: 0.0 for key in well_keys}
        free_bic: dict[str, float] = {key: 0.0 for key in well_keys}
        for key, d, theta, lapse, logl, bic in free_res:
            free_theta[key][d] = dict(theta) | {"lapse": lapse}
            free_ll[key] += logl
            free_bic[key] += bic

        # 2. Constrained shared-landscape fits
        for key in tqdm(well_keys, desc="Shared-landscape fits", unit="model"):
            row, trs = fit_shared(key, pool, free_theta[key], durs,
                                  de_maxiter=de_maxiter, de_popsize=de_popsize,
                                  max_rounds=max_rounds)
            free_k = (len(SPECS[key].param_names) + 1) * len(durs)
            row |= {"free_k": free_k, "free_logL": free_ll[key],
                    "free_AIC": 2 * free_k - 2 * free_ll[key], "free_BIC": free_bic[key]}
            rows.append(row)
            theta_rows.extend(trs)
            print(f"  => {key}: shared k={row['shared_k']} AIC={row['shared_AIC']:.1f} "
                  f"BIC={row['shared_BIC']:.1f} | free k={free_k} "
                  f"AIC={row['free_AIC']:.1f}  (dAIC shared-free={row['shared_AIC'] - row['free_AIC']:+.1f})",
                  flush=True)

    # 3. Benchmarks (no landscape): free = joint-training form
    resp = pd.read_csv(T.OUT_DIR / "response_per_duration_aic.csv")
    if not args.smoke and not args.constant_width_only:
        for key in BENCHMARKS:
            rows.append(benchmark_row(key, resp, n_by_dur))

    out = pd.DataFrame(rows)
    theta = pd.DataFrame(theta_rows)
    if args.constant_width_only and not args.smoke:
        fitted = set(well_keys)
        previous = pd.read_csv(OUT_CSV)
        previous = previous[~previous["model_key"].isin(fitted)]
        out = pd.concat([previous, out], ignore_index=True, sort=False)
        previous_theta = pd.read_csv(OUT_THETA)
        previous_theta = previous_theta[~previous_theta["model_key"].isin(fitted)]
        theta = pd.concat([previous_theta, theta], ignore_index=True, sort=False)

    # joint-training score: the constrained shared-landscape fit for the well
    # models, the (unconstrained) free fit for the no-landscape benchmarks. Both
    # already live in the shared_AIC/BIC columns.
    out["joint_AIC"] = out["shared_AIC"]
    out["joint_BIC"] = out["shared_BIC"]
    out["joint_dAIC"] = out["joint_AIC"] - out["joint_AIC"].min()
    out["joint_dBIC"] = out["joint_BIC"] - out["joint_BIC"].min()
    out = out.sort_values("joint_AIC").reset_index(drop=True)

    if args.smoke:
        pd.set_option("display.width", 250)
        print(out.round(2).to_string(index=False))
        print(theta.round(4).to_string(index=False))
        return

    out.to_csv(OUT_CSV, index=False)
    theta.to_csv(OUT_THETA, index=False)
    pd.set_option("display.width", 250)
    print("\n=== Joint training: one shared landscape across the five durations ===")
    cols = ["model_key", "has_landscape", "shared_k", "shared_AIC", "shared_BIC",
            "joint_dAIC", "joint_dBIC", "free_AIC", "n_width_shared"]
    print(out[[c for c in cols if c in out.columns]].round(1).to_string(index=False))
    print(f"\nSaved {OUT_CSV} and {OUT_THETA}")


if __name__ == "__main__":
    main()
