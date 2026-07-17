"""Joint model comparison with one landscape shared across all durations.

Each well model shares its representational parameters across the five exposure
durations, while the sensory channel and lapse vary by duration. Weber/ANS and
the normative log-prior proxy have no separable landscape, so their
duration-specific parameters are scored together as one joint model.

    .venv/bin/python -m scripts.trial_level_shared_landscape_ladder
    .venv/bin/python -m scripts.trial_level_shared_landscape_ladder --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
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

# ── Shared-landscape models and no-landscape benchmarks ─────────────────
SPECS: dict[str, T.ModelSpec] = {
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
    "gaussian_pow_attractor": ("sigma_c", "pow"),
    "gaussian_pow_readout": ("sigma_c", "pow"),
    "gaussian_const_attractor": ("sigma_c",),
    "gaussian_const_readout": ("sigma_c",),
}
WELL_LABELS = {
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


# ── Duration-level component of a no-landscape benchmark ─────────────────
def _fit_benchmark(args) -> tuple[str, int, float]:
    key, d, seed, maxiter, popsize = args
    spec = T.MODEL_BY_KEY[key]
    rf = T.fit_response(spec, _W["data"][d], _W["quad_fit"], _W["r_max"],
                        maxiter=maxiter, popsize=popsize, seed=seed,
                        quad_eval=_W["quad_eval"])
    return key, d, rf["logL"]


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
    s = shared["sigma_c"] * np.power(ns, shared["pow"])
    above = np.where(s >= 1.0)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return 1.0
    x0, x1, y0, y1 = ns[i - 1], ns[i], s[i - 1], s[i]
    return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))


def _shared_candidates(key: str, warm_shared: dict | None) -> list[dict]:
    """Broad deterministic starts for the shared landscape optimization."""
    spec = SPECS[key]
    bounds = _bounds(spec)
    values = []
    for name in LANDSCAPE[key]:
        lo, hi = bounds[name]
        if name == "sigma_c":
            values.append(np.exp(np.linspace(np.log(lo), np.log(hi), 3)))
        else:
            values.append(np.linspace(lo, hi, 3))
    candidates = [dict(zip(LANDSCAPE[key], point)) for point in product(*values)]
    if warm_shared is not None:
        candidates.insert(0, warm_shared)
    return candidates


def fit_shared(key: str, pool: Pool, durs: tuple[int, ...],
               warm_shared: dict | None,
               *, de_maxiter: int, de_popsize: int, max_rounds: int) -> tuple[dict, list[dict]]:
    """Share landscape parameters while sensory parameters and lapse vary."""
    spec = SPECS[key]
    shared_names = LANDSCAPE[key]
    nuis_names = tuple(p for p in _order(spec) if p not in shared_names)
    cands = _shared_candidates(key, warm_shared)

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
    k = len(shared_names) + len(nuis_names) * len(durs) + int(spec.cleanup)
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


def benchmark_row(key: str, logl: float, n_tot: int) -> dict:
    """Pooled score for a benchmark whose parameters vary by duration."""
    spec = T.MODEL_BY_KEY[key]
    k_d = len(spec.param_names) + 1
    k = len(T.DURATIONS) * k_d
    aic = 2 * k - 2 * logl
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

    fit_nodes = 31 if args.smoke else 201
    benchmark_maxiter = 2 if args.smoke else 35
    de_maxiter = 1 if args.smoke else 25
    de_popsize = 3 if args.smoke else 10
    max_rounds = 1 if args.smoke else 8
    durs = tuple(T.DURATIONS[:1]) if args.smoke else tuple(T.DURATIONS)
    if args.smoke:
        well_keys = [
            "gaussian_const_attractor"
            if args.constant_width_only
            else "gaussian_pow_attractor"
        ]
    elif args.constant_width_only:
        well_keys = ["gaussian_const_attractor", "gaussian_const_readout"]
    else:
        well_keys = list(SPECS)

    _init_worker(fit_nodes)  # parent needs data/quad for the shared step
    previous_theta = pd.read_csv(OUT_THETA) if OUT_THETA.exists() else pd.DataFrame()

    rows, theta_rows = [], []
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(fit_nodes,)) as pool:
        # 1. Shared-landscape fits
        for key in tqdm(well_keys, desc="Shared-landscape fits", unit="model"):
            warm = None
            if not previous_theta.empty:
                old = previous_theta[previous_theta["model_key"] == key]
                if not old.empty:
                    warm = {name: float(old.iloc[0][name]) for name in LANDSCAPE[key]}
            row, trs = fit_shared(key, pool, durs, warm,
                                  de_maxiter=de_maxiter, de_popsize=de_popsize,
                                  max_rounds=max_rounds)
            rows.append(row)
            theta_rows.extend(trs)
            print(f"  => {key}: shared k={row['shared_k']} AIC={row['shared_AIC']:.1f} "
                  f"BIC={row['shared_BIC']:.1f}", flush=True)

        # 2. Benchmarks have no separable landscape, so their duration-specific
        # components are fitted and scored together as one joint model.
        if not args.smoke and not args.constant_width_only:
            tasks = [(key, d, T.SEED + d, benchmark_maxiter, de_popsize)
                     for key in BENCHMARKS for d in durs]
            benchmark_res = list(
                tqdm(
                    pool.imap_unordered(_fit_benchmark, tasks),
                    total=len(tasks),
                    desc="Joint benchmark components",
                    unit="fit",
                )
            )
            benchmark_ll = {key: 0.0 for key in BENCHMARKS}
            for key, _, logl in benchmark_res:
                benchmark_ll[key] += logl

    # 3. Benchmarks have no separable landscape, so their duration-specific
    # parameters are scored together as one pooled model.
    if not args.smoke and not args.constant_width_only:
        n_tot = sum(_W["data"][d].n_total for d in T.DURATIONS)
        for key in BENCHMARKS:
            rows.append(benchmark_row(key, benchmark_ll[key], n_tot))

    out = pd.DataFrame(rows)
    theta = pd.DataFrame(theta_rows)
    if args.constant_width_only and not args.smoke:
        fitted = set(well_keys)
        previous = pd.read_csv(OUT_CSV)
        active_keys = set(SPECS) | set(BENCHMARKS)
        previous = previous[
            previous["model_key"].isin(active_keys)
            & ~previous["model_key"].isin(fitted)
        ]
        out = pd.concat([previous, out], ignore_index=True, sort=False)
        previous_theta = previous_theta[
            previous_theta["model_key"].isin(SPECS)
            & ~previous_theta["model_key"].isin(fitted)
        ]
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
            "joint_dAIC", "joint_dBIC", "n_width_shared"]
    print(out[[c for c in cols if c in out.columns]].round(1).to_string(index=False))
    print(f"\nSaved {OUT_CSV} and {OUT_THETA}")


if __name__ == "__main__":
    main()
