"""Head-to-head audit: gaussian_freebeta vs gaussian_pow at G=120.

Fits both under the shared-landscape pipeline with identical:
  well support 1..19, sensory channel, lapse, quadrature (201 fit / 1601 eval),
  optimiser (coordinate-descent Nelder-Mead from ladder warm-starts), data,
  and MAX_STEPS = G = 120.

Also runs (a) multi-seed restarts of the shared step, (b) participant-held-out
scoring with landscape held at the full-data fit and lapse re-profiled per fold,
(c) a short parametric recovery for freebeta (and optionally pow).

    .venv/bin/python -m scripts.trial_level_freebeta_audit
    .venv/bin/python -m scripts.trial_level_freebeta_audit --skip-recovery
    .venv/bin/python -m scripts.trial_level_freebeta_audit --smoke

Outputs under output/trial_level_model_comparison/:
  freebeta_vs_pow_G120.csv
  freebeta_vs_pow_G120_theta.csv
  freebeta_vs_pow_G120_seeds.csv
  freebeta_vs_pow_G120_cv.csv
  freebeta_vs_pow_G120_recovery.csv   (unless --skip-recovery)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

G = 120
FIT_NODES = 201
EVAL_NODES = 1601
ROUNDS = 4
N_SEEDS = 5
N_FOLDS = 5
CV_SEED = 20260711
RECOVERY_REPS = 8
OUT = T.OUT_DIR

KEYS = ("gaussian_freebeta_attractor", "gaussian_pow_attractor")
SHARED_NAMES = {
    "gaussian_freebeta_attractor": ("sigma_c", "beta"),
    "gaussian_pow_attractor": ("sigma_c", "pow"),
}


def _print_equations() -> None:
    print("=== Width laws (only difference) ===")
    print("  freebeta:  f_n = log10(1 + 1/n)   [analytic; NOT empirical word freqs]")
    print("             sigma_n = (sigma_c / f_n)^beta")
    print("  pow:       sigma_n = sigma_c * n^p")
    print("  both:      g_n(x) = -log sigma_n - (x-n)^2 / (2 sigma_n^2)")
    print("             V = -log sum exp(g);  readout = softmax(g); flat prior")
    print("  neither equation is Benford's law as a response distribution.")
    print()


def _bounds(key: str):
    spec = T.MODEL_BY_KEY[key]
    b = dict(zip(spec.param_names, spec.bounds))
    b["lapse"] = T.LAPSE_BOUNDS
    return b


def _vec(key: str, shared: dict, nuis: dict) -> np.ndarray:
    """Theta vector in ModelSpec.param_names + lapse order for response_loglik."""
    spec = T.MODEL_BY_KEY[key]
    full = dict(shared) | dict(nuis)
    return np.array([full[p] for p in spec.param_names] + [full["lapse"]], float)


def _split_shared_nuis(key: str, theta_row: pd.Series) -> tuple[dict, dict]:
    shared_names = SHARED_NAMES[key]
    shared = {p: float(theta_row[p]) for p in shared_names}
    nuis = {p: float(theta_row[p]) for p in ("s0", "lambda0", "lapse")}
    return shared, nuis


def _fit_nuisance(key, shared, x0n, quad, counts, r_max):
    b = _bounds(key)
    bounds = [b["s0"], b["lambda0"], b["lapse"]]

    def neg(v):
        nuis = {"s0": v[0], "lambda0": v[1], "lapse": v[2]}
        return -T.response_loglik(
            T.MODEL_BY_KEY[key], _vec(key, shared, nuis), quad, counts, r_max
        )

    pol = minimize(neg, x0n, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-4})
    return pol.x


def _fit_shared(key, shared0, nuis, quad, data, r_max, durs):
    names = SHARED_NAMES[key]
    b = _bounds(key)
    bounds = [b[p] for p in names]

    def neg(s):
        shared = dict(zip(names, s))
        return -sum(
            T.response_loglik(
                T.MODEL_BY_KEY[key],
                _vec(key, shared, dict(zip(("s0", "lambda0", "lapse"), nuis[d]))),
                quad, data[d].counts, r_max,
            )
            for d in durs
        )

    pol = minimize(neg, shared0, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-3})
    return dict(zip(names, pol.x)), -float(pol.fun)


def joint_fit(key, shared0, nuis0, quad, data, r_max, durs, rounds=ROUNDS,
              jitter=0.0, seed=0):
    """Coordinate descent at current T.MAX_STEPS."""
    names = SHARED_NAMES[key]
    shared = dict(shared0)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        b = _bounds(key)
        for p in names:
            lo, hi = b[p]
            shared[p] = float(np.clip(
                shared[p] * np.exp(rng.uniform(-jitter, jitter)), lo * 1.02, hi * 0.98
            ))
    nuis = {d: np.asarray(nuis0[d], float).copy() for d in durs}
    prev, tot = -np.inf, -np.inf
    for _ in range(rounds):
        for d in durs:
            nuis[d] = _fit_nuisance(key, shared, nuis[d], quad, data[d].counts, r_max)
        shared, tot = _fit_shared(
            key, np.array([shared[p] for p in names]), nuis, quad, data, r_max, durs
        )
        if abs(tot - prev) < 0.5:
            break
        prev = tot
    return shared, nuis, tot


def eval_logL(key, shared, nuis, quad, data, r_max, durs) -> float:
    return sum(
        T.response_loglik(
            T.MODEL_BY_KEY[key],
            _vec(key, shared, dict(zip(("s0", "lambda0", "lapse"), nuis[d]))),
            quad, data[d].counts, r_max,
        )
        for d in durs
    )


def n_width_of(key: str, shared: dict) -> float:
    ns = np.linspace(1.0, 60.0, 60000)
    if "beta" in shared:
        s = np.power(shared["sigma_c"] / np.log10(1.0 + 1.0 / ns), shared["beta"])
    else:
        s = shared["sigma_c"] * np.power(ns, shared["pow"])
    above = np.where(s >= 1.0)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return 1.0
    x0, x1, y0, y1 = ns[i - 1], ns[i], s[i - 1], s[i]
    return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))


def load_warmstarts(durs):
    th = pd.read_csv(OUT / "shared_landscape_ladder_theta.csv")
    out = {}
    for key in KEYS:
        sub = th[th.model_key == key].set_index("duration_ms")
        shared, _ = _split_shared_nuis(key, sub.iloc[0])
        nuis = {
            d: np.array([sub.loc[d, "s0"], sub.loc[d, "lambda0"], sub.loc[d, "lapse"]])
            for d in durs
        }
        out[key] = (shared, nuis)
    return out


def run_fits(durs, data, r_max, smoke: bool):
    T.MAX_STEPS = G
    quad_fit = T.make_quadrature(31 if smoke else FIT_NODES, kind="grid")
    quad_eval = T.make_quadrature(101 if smoke else EVAL_NODES, kind="grid")
    warm = load_warmstarts(durs)
    rounds = 1 if smoke else ROUNDS

    rows, theta_rows = [], []
    fits = {}
    for key in KEYS:
        t0 = time.time()
        shared0, nuis0 = warm[key]
        shared, nuis, tot_fit = joint_fit(
            key, shared0, nuis0, quad_fit, data, r_max, durs, rounds=rounds
        )
        tot = eval_logL(key, shared, nuis, quad_eval, data, r_max, durs)
        k = 2 + 3 * len(durs) + 1  # shared landscape (2) + per-dur (s0,lam,lapse) + G
        n = sum(data[d].n_total for d in durs)
        row = {
            "model_key": key,
            "G": G,
            "shared_k_excl_G": 2 + 3 * len(durs),
            "k_incl_G": k,
            "logL_fitgrid": tot_fit,
            "logL": tot,
            "AIC": 2 * k - 2 * tot,
            "BIC": k * np.log(n) - 2 * tot,
            "n_width": n_width_of(key, shared),
            "minutes": (time.time() - t0) / 60.0,
            **{f"shared_{p}": float(shared[p]) for p in SHARED_NAMES[key]},
        }
        rows.append(row)
        fits[key] = (shared, nuis)
        for d in durs:
            tr = {"model_key": key, "duration_ms": d, **shared,
                  "s0": float(nuis[d][0]), "lambda0": float(nuis[d][1]),
                  "lapse": float(nuis[d][2])}
            theta_rows.append(tr)
        print(f"  [{key}] logL={tot:.2f} AIC={row['AIC']:.1f} "
              f"n_width={row['n_width']:.3f} ({row['minutes']:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df["dAIC"] = df["AIC"] - df["AIC"].min()
    df["dBIC"] = df["BIC"] - df["BIC"].min()
    df["dlogL"] = df["logL"] - df["logL"].max()
    return df, pd.DataFrame(theta_rows), fits, quad_fit, quad_eval


def run_seeds(fits, durs, data, r_max, quad_fit, quad_eval, smoke: bool):
    T.MAX_STEPS = G
    n_seeds = 2 if smoke else N_SEEDS
    rows = []
    for key in KEYS:
        shared0, nuis0 = fits[key]
        for s in tqdm(range(n_seeds), desc=f"seeds {key}", leave=False):
            shared, nuis, _ = joint_fit(
                key, shared0, nuis0, quad_fit, data, r_max, durs,
                rounds=1 if smoke else 3, jitter=0.15, seed=1103 + 17 * s,
            )
            tot = eval_logL(key, shared, nuis, quad_eval, data, r_max, durs)
            rows.append({
                "model_key": key, "seed": s, "logL": tot,
                **{f"shared_{p}": float(shared[p]) for p in SHARED_NAMES[key]},
                "n_width": n_width_of(key, shared),
            })
    return pd.DataFrame(rows)


def _participant_folds(trial: pd.DataFrame) -> np.ndarray:
    ids = trial["uniqueid"].astype("category")
    codes = ids.cat.codes.to_numpy()
    n = int(codes.max()) + 1
    rng = np.random.default_rng(CV_SEED)
    assign = rng.permutation(n) % N_FOLDS
    return assign[codes]


def run_cv(fits, trial, durs, r_max, quad_eval, smoke: bool):
    """Landscape fixed at full-data fit; lapse re-profiled on train; score holdout."""
    T.MAX_STEPS = G
    folds = _participant_folds(trial)
    n_folds = 2 if smoke else N_FOLDS
    rows = []
    for key in KEYS:
        shared, nuis_full = fits[key]
        spec = T.MODEL_BY_KEY[key]
        # precompute p_resp (no lapse) per duration on the eval grid
        p_resp = {}
        for d in durs:
            theta = {p: float(shared[p]) for p in SHARED_NAMES[key]}
            theta["s0"] = float(nuis_full[d][0])
            theta["lambda0"] = float(nuis_full[d][1])
            p_resp[d] = T.compute_grid(spec, theta, quad_eval)["p_resp"]

        for fold in range(n_folds):
            train = folds != fold
            test = folds == fold
            hold_ll = 0.0
            n_hold = 0
            for d in durs:
                sub = trial[trial["exp_dur"] == d].reset_index(drop=True)
                dur_mask = trial["exp_dur"].to_numpy() == d
                tr_m = train[dur_mask]
                te_m = test[dur_mask]
                n_t = sub.loc[tr_m, "n_shown"].to_numpy(int)
                r_t = sub.loc[tr_m, "n_guess"].to_numpy(int)
                counts = np.zeros((T.N_SHOWN.size, r_max), float)
                np.add.at(counts, (n_t - T.N_TRUE_MIN, np.clip(r_t, 1, r_max) - 1), 1.0)

                def neg_lapse(ell, _pr=p_resp[d], _c=counts):
                    pf = T.apply_lapse(_pr, float(np.clip(ell[0], 1e-4, 0.5)), r_max)
                    p = np.clip(pf, 1e-300, None)
                    return -float(np.sum(_c * np.log(p)))

                pol = minimize(neg_lapse, [float(nuis_full[d][2])], method="Nelder-Mead",
                               bounds=[T.LAPSE_BOUNDS],
                               options={"maxiter": 80, "xatol": 1e-6, "fatol": 1e-4})
                ell = float(np.clip(pol.x[0], *T.LAPSE_BOUNDS))
                pf = T.apply_lapse(p_resp[d], ell, r_max)
                n_te = sub.loc[te_m, "n_shown"].to_numpy(int)
                r_te = sub.loc[te_m, "n_guess"].to_numpy(int)
                for n, r in zip(n_te, r_te):
                    hold_ll += np.log(max(pf[n - T.N_TRUE_MIN, min(r, r_max) - 1], 1e-300))
                n_hold += int(te_m.sum())
            rows.append({
                "model_key": key, "fold": fold, "holdout_logL": hold_ll,
                "n_holdout": n_hold, "holdout_logL_per_trial": hold_ll / max(n_hold, 1),
            })
    return pd.DataFrame(rows)


def run_recovery(key, shared, nuis, durs, data, r_max, smoke: bool):
    """Per-duration recovery around the G=120 shared fit (sensory free; landscape free)."""
    T.MAX_STEPS = G
    spec = T.MODEL_BY_KEY[key]
    quad_fit = T.make_quadrature(31 if smoke else 201, kind="grid")
    quad_gen = T.make_quadrature(101 if smoke else 1601, kind="grid")
    reps = 2 if smoke else RECOVERY_REPS
    b = _bounds(key)
    rows = []
    jitter = {"sigma_c": 0.35, "s0": 0.50, "lambda0": 0.20, "lapse": 0.40,
              "pow": 0.25, "beta": 0.25}
    for d in durs:
        base = {p: float(shared[p]) for p in SHARED_NAMES[key]}
        base |= {"s0": float(nuis[d][0]), "lambda0": float(nuis[d][1]),
                 "lapse": float(nuis[d][2])}
        for rep in range(reps):
            rng = np.random.default_rng(1000 * d + rep + (0 if "pow" in shared else 50000))
            gen = {}
            for p, val in base.items():
                lo, hi = b[p]
                gen[p] = float(np.clip(
                    val * np.exp(rng.uniform(-jitter[p], jitter[p])), lo * 1.02, hi * 0.98
                ))
            grid = T.compute_grid(spec, {p: gen[p] for p in spec.param_names}, quad_gen)
            pf = T.apply_lapse(grid["p_resp"], gen["lapse"], r_max)
            real = data[d]
            counts = np.zeros_like(real.counts)
            for i in range(real.counts.shape[0]):
                pi = pf[i] / pf[i].sum()
                counts[i] = rng.multinomial(int(real.counts[i].sum()), pi)
            synth = replace(real, counts=counts)
            # fit from a jittered start
            x0 = np.array([gen[p] * np.exp(rng.uniform(-0.1, 0.1)) for p in list(spec.param_names) + ["lapse"]])
            bounds = [b[p] for p in list(spec.param_names) + ["lapse"]]

            def neg(v):
                return -T.response_loglik(spec, v, quad_fit, synth.counts, r_max)

            pol = minimize(neg, x0, method="Nelder-Mead", bounds=bounds,
                           options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-4})
            hat = dict(zip(list(spec.param_names) + ["lapse"], pol.x))
            row = {"model_key": key, "duration_ms": d, "rep": rep}
            for p in hat:
                row[f"gen_{p}"] = gen[p]
                row[f"hat_{p}"] = float(hat[p])
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-recovery", action="store_true")
    args = ap.parse_args(argv)

    _print_equations()
    durs = tuple(T.DURATIONS[:2] if args.smoke else T.DURATIONS)
    trial = T.load_trials()
    if args.smoke:
        # keep a few participants for speed
        keep = trial["uniqueid"].drop_duplicates().head(40)
        trial = trial[trial["uniqueid"].isin(keep)].reset_index(drop=True)
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)

    print(f"=== Shared-landscape fits at G={G} ===", flush=True)
    fit_df, theta_df, fits, quad_fit, quad_eval = run_fits(durs, data, r_max, args.smoke)
    fit_df.to_csv(OUT / "freebeta_vs_pow_G120.csv", index=False)
    theta_df.to_csv(OUT / "freebeta_vs_pow_G120_theta.csv", index=False)
    print(fit_df.to_string(index=False))
    print(f"Saved {OUT / 'freebeta_vs_pow_G120.csv'}", flush=True)

    print("\n=== Multi-seed stability ===", flush=True)
    seed_df = run_seeds(fits, durs, data, r_max, quad_fit, quad_eval, args.smoke)
    seed_df.to_csv(OUT / "freebeta_vs_pow_G120_seeds.csv", index=False)
    summary = (seed_df.groupby("model_key")["logL"]
               .agg(["mean", "std", "min", "max"]).reset_index())
    print(summary.to_string(index=False))
    print(f"Saved {OUT / 'freebeta_vs_pow_G120_seeds.csv'}", flush=True)

    print("\n=== Participant-held-out CV (lapse re-profiled) ===", flush=True)
    # rebuild full trial folds on the (possibly smoke-filtered) trial table
    cv_df = run_cv(fits, trial, durs, r_max, quad_eval, args.smoke)
    cv_df.to_csv(OUT / "freebeta_vs_pow_G120_cv.csv", index=False)
    cv_sum = (cv_df.groupby("model_key")[["holdout_logL", "holdout_logL_per_trial"]]
              .sum().reset_index())
    # per-trial mean across folds
    cv_sum = (cv_df.groupby("model_key")
              .agg(holdout_logL=("holdout_logL", "sum"),
                   n_holdout=("n_holdout", "sum"),
                   mean_per_trial=("holdout_logL_per_trial", "mean"))
              .reset_index())
    print(cv_sum.to_string(index=False))
    print(f"Saved {OUT / 'freebeta_vs_pow_G120_cv.csv'}", flush=True)

    if not args.skip_recovery:
        print("\n=== Parameter recovery (short) ===", flush=True)
        rec_rows = []
        for key in KEYS:
            shared, nuis = fits[key]
            rec_rows.append(run_recovery(key, shared, nuis, durs, data, r_max, args.smoke))
        rec = pd.concat(rec_rows, ignore_index=True)
        rec.to_csv(OUT / "freebeta_vs_pow_G120_recovery.csv", index=False)
        for key in KEYS:
            sub = rec[rec.model_key == key]
            land = "beta" if "beta" in SHARED_NAMES[key] else "pow"
            for p in ("sigma_c", land, "s0", "lambda0", "lapse"):
                g, h = sub[f"gen_{p}"].to_numpy(), sub[f"hat_{p}"].to_numpy()
                if p in ("sigma_c", "s0", "lapse"):
                    g, h = np.log10(g), np.log10(h)
                r2 = 1 - np.sum((h - g) ** 2) / np.sum((g - g.mean()) ** 2)
                rho = pd.Series(g).corr(pd.Series(h), method="spearman")
                print(f"  [{key}] {p}: R2={r2:.3f}  Spearman={rho:.3f}")
        print(f"Saved {OUT / 'freebeta_vs_pow_G120_recovery.csv'}", flush=True)


if __name__ == "__main__":
    main()
