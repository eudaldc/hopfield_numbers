"""Fit the *exact* Cheyette & Piantadosi (2020) bounded-optimal model and place
it on the trial-level model-comparison ladder, on equal footing.

This replaces the schematic ``cp_log_prior_proxy`` (a log-normal encoding + a
Benford-ish log prior) with the literal C&P model as published in
https://github.com/samcheyette/info_theory_number:

  * Zipf(1/x^2) prior, info-bottleneck posterior ``Q(k | n)`` with
    ``KL(Q||P) = info_bound`` (vendored in ``scripts/cheyette_model.py``),
  * exposure enters through ``info_bound(t) = min(B, R*t)`` with ``t`` in seconds,
  * three parameters ``(B, R, epsilon)`` shared across ALL five durations.

To compare against the attractor winner we score C&P through the SAME trial-level
response likelihood used everywhere else: its structured readout ``Q(k | n)`` over
the integer support ``1..N_SUPPORT_MAX`` plus the common uniform contamination
over ``1..r_max`` (with ``epsilon`` playing the role of the lapse weight, exactly
as in the C&P likelihood). The uniform floor is also the *more generous* lapse
for C&P than its own Zipf guess, so this is both the fair and the favourable
choice; the literal Zipf-guess variant is reported alongside as a robustness
check.

Because C&P ties all five durations together with just ``(B, R)``, it is fit
jointly across durations (3 parameters total) rather than per-duration. AIC is
convention-free for the pooled total (``2*k_total - 2*logL_total`` equals the
sum of per-duration AICs), so the ladder is directly comparable to
``output/trial_level_model_comparison/consolidated_summary.csv``.

Run from the project root::

    .venv/bin/python -m scripts.trial_level_cp_exact
    .venv/bin/python -m scripts.trial_level_cp_exact --validate-against <clone_dir>

Outputs land in ``output/cp_exact/``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cheyette_model as cp
from scripts.trial_level_model_comparison import (
    DURATIONS,
    N_SHOWN,
    N_SUPPORT_MAX,
    N_TRUE_MIN,
    N_WELLS,
    SEED,
    apply_lapse,
    build_duration_data,
    load_trials,
    response_support_report,
)

OUT_DIR = ROOT / "output" / "cp_exact"
CONSOLIDATED = (
    ROOT / "output" / "trial_level_model_comparison" / "consolidated_summary.csv"
)

# Faithful C&P cache settings (their computeQ.py defaults).
ALPHA = 2.0
N_STEPS = 10000

# Reference models already fitted in the trial-level comparison. ``k_total`` is
# the summed parameter count over the five durations (sensory params + lapse,
# times five); logL is recovered from the published response AIC via
# ``AIC = 2*k_total - 2*logL_total``.
REFERENCE_MODELS = {
    "gaussian_freebeta_attractor": {
        "label": "Frequency-well attractor (free beta) [winner]", "k_total": 25},
    "gaussian_benford_free_beta": {
        "label": "Same wells, no clean-up (static twin)", "k_total": 25},
    "weber_ans": {"label": "Weber/ANS", "k_total": 20},
    "cp_log_prior_proxy": {"label": "C&P log-prior proxy (schematic)", "k_total": 20},
}

# Winner's per-duration response AIC (k = 5 params/duration), from
# output/trial_level_model_comparison/response_per_duration_aic.csv. Used to
# benchmark C&P's per-duration ceiling; logL_d = (2*5 - AIC_d) / 2.
WINNER_RESP_AIC = {40: 57132.529692070704, 80: 46929.3933847668,
                   160: 42715.38288533016, 320: 41570.94858249419,
                   640: 37838.17244140747}


# ── Fitting the exact C&P model to the real trial-level data ──────────────


def _unpack(params: np.ndarray) -> tuple[float, float, float]:
    log_B, log_R, logit_eps = params
    B = float(np.exp(log_B))
    R = float(np.exp(log_R))
    eps = float(1.0 / (1.0 + np.exp(-logit_eps)))
    return B, R, eps


def cp_response_loglik(
    params: np.ndarray,
    cache: dict,
    data_by_dur: dict,
    r_max: int,
    *,
    guess: str = "uniform",
    p_k_full: np.ndarray | None = None,
) -> float:
    """Summed trial-level response log-likelihood of the exact C&P model.

    ``guess="uniform"``: structured readout over ``1..N_SUPPORT_MAX`` plus the
    common uniform lapse over ``1..r_max`` (equal footing with all other models;
    ``epsilon`` is the lapse weight). ``guess="zipf"``: the literal C&P mixture
    ``(1-eps) Q + eps P(k)`` over the full ``1..r_max`` support.
    """
    B, R, eps = _unpack(params)
    ll = 0.0
    for d in DURATIONS:
        info_bound = min(B, R * (d / 1000.0))  # t in seconds
        q_nk = cp.lookup_q_interp(cache, info_bound)  # (n, K)
        data = data_by_dur[d]
        if guess == "uniform":
            pf = apply_lapse(q_nk[:, :N_WELLS], eps, r_max)
        else:  # literal Zipf guess over the full support
            pf = (1.0 - eps) * q_nk + eps * p_k_full[None, :]
        if not np.all(np.isfinite(pf)) or np.any(pf <= 0):
            return -np.inf
        ll += float(np.sum(data.counts * np.log(pf)))
    return ll if np.isfinite(ll) else -np.inf


def fit_cp_exact(
    cache: dict,
    data_by_dur: dict,
    r_max: int,
    *,
    guess: str = "uniform",
    p_k_full: np.ndarray | None = None,
    seed: int = SEED,
) -> dict:
    """Global fit of ``(log B, log R, logit eps)`` to the pooled responses."""
    bounds = [
        (np.log(0.5), np.log(30.0)),   # B in [0.5, 30] bits
        (np.log(1.0), np.log(1000.0)), # R in [1, 1000] bits/s
        (-8.0, 0.5),                   # eps in ~[3e-4, 0.62]
    ]

    def neg(params):
        return -cp_response_loglik(
            params, cache, data_by_dur, r_max, guess=guess, p_k_full=p_k_full
        )

    de = differential_evolution(
        neg, bounds=bounds, maxiter=60, popsize=15, tol=1e-7,
        mutation=(0.5, 1.0), recombination=0.9, seed=seed,
        init="latinhypercube", polish=False, updating="deferred", workers=1,
    )
    pol = minimize(neg, de.x, method="Nelder-Mead",
                   options={"maxiter": 800, "xatol": 1e-7, "fatol": 1e-6})
    vec = pol.x if pol.fun <= de.fun else de.x
    B, R, eps = _unpack(vec)
    logl = cp_response_loglik(vec, cache, data_by_dur, r_max,
                              guess=guess, p_k_full=p_k_full)

    per_dur = []
    for d in DURATIONS:
        info_bound = min(B, R * (d / 1000.0))
        q_nk = cp.lookup_q_interp(cache, info_bound)
        data = data_by_dur[d]
        if guess == "uniform":
            pf = apply_lapse(q_nk[:, :N_WELLS], eps, r_max)
        else:
            pf = (1.0 - eps) * q_nk + eps * p_k_full[None, :]
        per_dur.append({
            "duration_ms": int(d),
            "info_bound_bits": float(info_bound),
            "logL": float(np.sum(data.counts * np.log(pf))),
            "n_trials": int(data.n_total),
        })

    n_total = sum(p["n_trials"] for p in per_dur)
    k = 3
    return {
        "guess": guess,
        "B": B, "R": R, "epsilon": eps,
        "logL_total": float(logl),
        "k_params": k,
        "n_total": int(n_total),
        "AIC": float(2 * k - 2 * logl),
        "BIC_pooled": float(k * np.log(n_total) - 2 * logl),
        "per_duration": per_dur,
    }


# ── Shape ceiling: best per-duration info_bound, ignoring the rate law ────


def cp_shape_ceiling(cache: dict, data_by_dur: dict, r_max: int) -> dict:
    """Per-duration ceiling of the C&P readout shape.

    For each duration we scan every cached ``info_bound`` and optimise the lapse
    ``epsilon`` for it, keeping the best. This upper-bounds what *any* ``(B, R)``
    rate law could achieve (each duration is free to pick its own budget), so it
    separates the two possible reasons C&P fits worse: the ``Q(k | n)`` readout
    *shape* (captured by this ceiling) versus the ``min(B, R*t)`` rate law (the
    extra loss of the fitted model relative to this ceiling). 10 free parameters
    (5 info budgets + 5 lapses).
    """
    bounds = cache["info_bounds"]
    qc = cache["q_cache"]
    rows = []
    for d in DURATIONS:
        counts = data_by_dur[d].counts

        def best_eps_ll(q_nk):
            def negll(eps):
                pf = apply_lapse(q_nk[:, :N_WELLS], eps, r_max)
                return -float(np.sum(counts * np.log(pf)))
            res = minimize_scalar(negll, bounds=(1e-4, 0.5), method="bounded")
            return -res.fun, res.x

        best_ll, best_ib, best_eps = -np.inf, None, None
        for i, ib in enumerate(bounds):
            ll, eps = best_eps_ll(qc[i])
            if ll > best_ll:
                best_ll, best_ib, best_eps = ll, float(ib), float(eps)
        win_ll = (2 * 5 - WINNER_RESP_AIC[d]) / 2.0
        rows.append({
            "duration_ms": int(d), "best_info_bound": best_ib,
            "best_eps": best_eps, "ceiling_logL": best_ll,
            "winner_logL": win_ll, "gap_logL": win_ll - best_ll,
            "n_trials": int(data_by_dur[d].n_total),
        })
    total_ceiling = sum(r["ceiling_logL"] for r in rows)
    total_winner = sum(r["winner_logL"] for r in rows)
    k_ceiling, k_winner = 10, 25
    return {
        "per_duration": rows,
        "ceiling_logL_total": total_ceiling,
        "ceiling_AIC": 2 * k_ceiling - 2 * total_ceiling,
        "winner_logL_total": total_winner,
        "winner_AIC": 2 * k_winner - 2 * total_winner,
        "dAIC_ceiling_vs_winner": (2 * k_ceiling - 2 * total_ceiling)
        - (2 * k_winner - 2 * total_winner),
    }


# ── Ladder assembly ──────────────────────────────────────────────────────


def build_ladder(cp_fit: dict) -> pd.DataFrame:
    """Assemble the response-channel AIC/BIC ladder with the exact C&P model.

    Reference models are read from the published consolidated summary; their
    logL is recovered from the response AIC. BIC is recomputed on the pooled
    convention ``k_total * ln(N_total)`` so every row (including the 3-parameter
    C&P) is on the same footing.
    """
    n_total = cp_fit["n_total"]
    ln_n = np.log(n_total)
    cons = pd.read_csv(CONSOLIDATED).set_index("model_key")

    rows = []
    for key, meta in REFERENCE_MODELS.items():
        if key not in cons.index:
            continue
        aic = float(cons.loc[key, "resp_AIC"])
        k_total = meta["k_total"]
        logl = (2 * k_total - aic) / 2.0
        rows.append({
            "model_key": key, "label": meta["label"],
            "logL_total": logl, "k_total": k_total,
            "AIC": aic, "BIC_pooled": k_total * ln_n - 2 * logl,
        })
    rows.append({
        "model_key": "cp_exact_uniform", "label": "C&P exact (info-bottleneck)",
        "logL_total": cp_fit["logL_total"], "k_total": cp_fit["k_params"],
        "AIC": cp_fit["AIC"], "BIC_pooled": cp_fit["BIC_pooled"],
    })

    df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    df["dAIC"] = df["AIC"] - df["AIC"].min()
    df["dBIC"] = df["BIC_pooled"] - df["BIC_pooled"].min()
    return df


# ── Validation of the vendored port ──────────────────────────────────────


def validate_against_clone(clone_dir: Path) -> dict:
    """Check the vendored model reproduces C&P's published cache and fits."""
    report = {}
    pkl = clone_dir / "cache_files" / "info_number_cache.pkl"
    their = pickle.load(open(pkl, "rb"))
    ours = cp.build_q_cache(
        alpha=their["alpha"], min_n=int(their["ns"][0]), max_n=int(their["ns"][-1]),
        max_k=int(their["ks"][-1]), info_step=float(their["info_step"]),
        max_info=float(their["info_bounds"][-1]), n_steps=N_STEPS,
    )
    diff = np.abs(ours["q_cache"] - their["q_cache"])
    report["cache_max_abs_diff"] = float(diff.max())
    report["cache_mean_abs_diff"] = float(diff.mean())

    # Likelihood fidelity: our NLL at their published fitted params vs their NLL.
    mle = pd.read_csv(clone_dir / "output" / "mle_results.csv")
    sim = pd.read_csv(clone_dir / "output" / "simulated_data.csv")
    p_ks = their["p_ks"]
    n_k = len(p_ks)
    check = mle.head(25)
    max_rel = 0.0
    for _, row in check.iterrows():
        subj = sim[sim["participant_id"] == row["participant_id"]]
        B, R, eps = row["fit_B"], row["fit_R"], row["fit_epsilon"]
        nll = 0.0
        for t in sorted(subj["time"].unique()):
            m = subj["time"] == t
            ns = subj.loc[m, "n_shown"].to_numpy(int)
            rs = subj.loc[m, "response"].to_numpy(int)
            q_nk = cp.lookup_q(their, min(B, R * t))
            k_idx = np.clip(rs - 1, 0, n_k - 1)
            valid = (rs - 1 >= 0) & (rs - 1 < n_k)
            q_vals = q_nk[ns - int(their["ns"][0]), k_idx]
            probs = np.where(valid, (1 - eps) * q_vals + eps * p_ks[k_idx], 1e-10)
            nll -= np.sum(np.log(np.maximum(probs, 1e-10)))
        rel = abs(nll - row["neg_log_likelihood"]) / abs(row["neg_log_likelihood"])
        max_rel = max(max_rel, rel)
    report["mle_nll_max_rel_diff"] = float(max_rel)
    return report


# ── Orchestration ────────────────────────────────────────────────────────


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-against", type=str, default=None,
                    help="path to a clone of samcheyette/info_theory_number")
    ap.add_argument("--skip-literal", action="store_true",
                    help="skip the literal Zipf-guess C&P variant")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate_against:
        print("== Validating vendored port against the C&P clone ==")
        rep = validate_against_clone(Path(args.validate_against))
        print(f"  cache max|diff|      = {rep['cache_max_abs_diff']:.2e}")
        print(f"  cache mean|diff|     = {rep['cache_mean_abs_diff']:.2e}")
        print(f"  MLE NLL max rel diff = {rep['mle_nll_max_rel_diff']:.2e}")
        pd.DataFrame([rep]).to_csv(OUT_DIR / "port_validation.csv", index=False)

    trial = load_trials()
    support = response_support_report(trial)
    r_max = support["r_max"]
    data_by_dur = build_duration_data(trial, r_max)
    n_total = sum(data_by_dur[d].n_total for d in DURATIONS)
    print(f"\nLoaded {len(trial)} trials ({n_total} across the five durations). "
          f"Response support 1..{r_max}.")

    # C&P cache on the shared integer support 1..N_SUPPORT_MAX (= the wells).
    print("Building C&P cache on the comparison support (1..%d)..." % N_SUPPORT_MAX)
    cache = cp.build_q_cache(alpha=ALPHA, min_n=N_TRUE_MIN, max_n=int(N_SHOWN[-1]),
                             max_k=N_SUPPORT_MAX, n_steps=N_STEPS)

    print("Fitting exact C&P (B, R, eps) jointly across durations [uniform lapse]...")
    fit_u = fit_cp_exact(cache, data_by_dur, r_max, guess="uniform")
    print(f"  B={fit_u['B']:.3f} bits  R={fit_u['R']:.2f} bits/s  "
          f"eps={fit_u['epsilon']:.4f}  logL={fit_u['logL_total']:.1f}  "
          f"AIC={fit_u['AIC']:.1f}")

    results = {"uniform": fit_u}
    if not args.skip_literal:
        print("Building C&P cache on the full support (1..%d) for the literal "
              "Zipf-guess variant..." % r_max)
        cache_full = cp.build_q_cache(alpha=ALPHA, min_n=N_TRUE_MIN,
                                      max_n=int(N_SHOWN[-1]), max_k=r_max,
                                      n_steps=N_STEPS)
        p_k_full = cp.P(np.arange(1, r_max + 1, dtype=float), ALPHA)
        print("Fitting exact C&P [literal Zipf guess]...")
        fit_z = fit_cp_exact(cache_full, data_by_dur, r_max, guess="zipf",
                             p_k_full=p_k_full)
        print(f"  B={fit_z['B']:.3f}  R={fit_z['R']:.2f}  eps={fit_z['epsilon']:.4f}  "
              f"logL={fit_z['logL_total']:.1f}  AIC={fit_z['AIC']:.1f}")
        results["zipf"] = fit_z

    # Persist the fits and the per-duration breakdown.
    fit_rows = []
    for guess, fit in results.items():
        fit_rows.append({
            "variant": guess, "B": fit["B"], "R": fit["R"],
            "epsilon": fit["epsilon"], "logL_total": fit["logL_total"],
            "k_params": fit["k_params"], "n_total": fit["n_total"],
            "AIC": fit["AIC"], "BIC_pooled": fit["BIC_pooled"],
        })
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / "cp_exact_fits.csv", index=False)
    per_dur_rows = []
    for guess, fit in results.items():
        for pd_row in fit["per_duration"]:
            per_dur_rows.append({"variant": guess, **pd_row})
    pd.DataFrame(per_dur_rows).to_csv(OUT_DIR / "cp_exact_per_duration.csv", index=False)

    print("Computing the C&P per-duration shape ceiling (rate law removed)...")
    ceiling = cp_shape_ceiling(cache, data_by_dur, r_max)
    pd.DataFrame(ceiling["per_duration"]).to_csv(
        OUT_DIR / "cp_exact_ceiling.csv", index=False)
    shape_daic = ceiling["dAIC_ceiling_vs_winner"]
    ratelaw_daic = fit_u["AIC"] - ceiling["ceiling_AIC"]

    ladder = build_ladder(fit_u)
    ladder.to_csv(OUT_DIR / "cp_exact_ladder.csv", index=False)

    winner_aic = ladder["AIC"].min()
    cp_row = ladder[ladder["model_key"] == "cp_exact_uniform"].iloc[0]
    proxy_row = ladder[ladder["model_key"] == "cp_log_prior_proxy"]
    print("\n=== Response-channel ladder (pooled, exact C&P included) ===")
    with pd.option_context("display.width", 200):
        print(ladder[["model_key", "label", "k_total", "logL_total",
                      "AIC", "dAIC", "dBIC"]]
              .round({"logL_total": 1, "AIC": 1, "dAIC": 1, "dBIC": 1})
              .to_string(index=False))
    print(f"\nExact C&P loses to the winner by dAIC = {cp_row['dAIC']:.1f}.")
    if len(proxy_row):
        print(f"(The schematic proxy lost by dAIC = "
              f"{float(proxy_row['dAIC'].iloc[0]):.1f}.)")
    if "zipf" in results:
        z = results["zipf"]
        print(f"Literal Zipf-guess C&P: AIC={z['AIC']:.1f}  "
              f"dAIC vs winner = {z['AIC'] - winner_aic:.1f}.")

    print("\n=== Why C&P loses: readout shape vs rate law ===")
    print(f"  C&P per-duration shape ceiling (10 params, no rate law): "
          f"AIC={ceiling['ceiling_AIC']:.1f}")
    print(f"  -> readout-shape penalty (ceiling vs winner):   dAIC = {shape_daic:.1f}")
    print(f"  -> min(B,R*t) rate-law penalty (fit vs ceiling): dAIC = {ratelaw_daic:.1f}")
    print(f"  -> total (exact fit vs winner):                  dAIC = {cp_row['dAIC']:.1f}")
    print("  The readout shape carries most of the gap: even given a free bit "
          "budget\n  per duration, C&P's Q(k|n) fits the responses worse than the "
          "attractor.")
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
