"""Analyses supporting the response to reviewer comments (i-Perception).

Three self-contained analyses, each writing a CSV that feeds the revised
manuscript / supplementary material:

  A. Point 2 -- per-duration AIC, Delta-AIC and Akaike weights for the
     five summary-level models. Gives the honest picture of *how* modest
     the Hopfield-Poisson advantage is and how consistent it is.

  B. Point 4 -- shared-(c, kappa) joint fit across the five durations,
     with only the sensory-noise parameters (s0, lambda0) free per
     duration. Tests whether the attractor *mechanism* is duration
     invariant and only the sensory input changes with exposure.

  C. Point 3 -- DD vs control differences re-expressed as recoverable
     derived quantities (n_crit = 1/c and the noise profile sigma(n)),
     rather than as individual, partly trading-off parameters.

Run from the project root:

    .venv/bin/python -m scripts.reviewer_response_analyses
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.fit_helpers as fh

DURATIONS: tuple[int, ...] = (40, 80, 160, 320, 640)
MC_DIR = Path("output/model_comparison")
PIANT_DIR = Path("output/piantadosi_fit")
DECARLI_DIR = Path("output/decarli_fit")
T0_PIANT = 1.30


# ──────────────────────────────────────────────────────────────────────────
# A. Per-duration AIC / Delta-AIC / Akaike weights  (reviewer point 2)
# ──────────────────────────────────────────────────────────────────────────

MODEL_ORDER = (
    "hopfield_poisson",
    "no_cleanup_poisson",
    "gaussian_benford",
    "weber_ans",
    "cp_log_prior_proxy",
)
MODEL_LABEL = {
    "hopfield_poisson": "Hopfield-Poisson",
    "no_cleanup_poisson": "No-cleanup Poisson",
    "gaussian_benford": "Gaussian/Benford",
    "weber_ans": "Weber/ANS",
    "cp_log_prior_proxy": "C&P log-prior proxy",
}


def akaike_weights(aic: np.ndarray) -> np.ndarray:
    d = aic - np.min(aic)
    w = np.exp(-0.5 * d)
    return w / w.sum()


def analysis_a_aic_weights() -> pd.DataFrame:
    fit = pd.read_csv(MC_DIR / "per_duration_fit_metrics.csv")
    piv = fit.pivot_table(index="duration_ms", columns="model_key", values="AIC")
    piv = piv[list(MODEL_ORDER)]

    rows: list[dict] = []
    for duration, aic_row in piv.iterrows():
        aic = aic_row.to_numpy(float)
        d_aic = aic - aic.min()
        w = akaike_weights(aic)
        winner = MODEL_LABEL[aic_row.idxmin()]
        for key, a, da, wi in zip(MODEL_ORDER, aic, d_aic, w):
            rows.append(
                {
                    "duration_ms": int(duration),
                    "model": MODEL_LABEL[key],
                    "model_key": key,
                    "AIC": float(a),
                    "delta_AIC": float(da),
                    "akaike_weight": float(wi),
                    "duration_winner": winner,
                }
            )

    # Across-duration mean AIC -> delta + weights (matches the manuscript's
    # "Global" bar) plus per-duration win counts vs the no-cleanup control.
    mean_aic = piv.mean(axis=0).to_numpy(float)
    d_mean = mean_aic - mean_aic.min()
    w_mean = akaike_weights(mean_aic)
    beats_nocleanup = int((piv["hopfield_poisson"] < piv["no_cleanup_poisson"]).sum())
    for key, a, da, wi in zip(MODEL_ORDER, mean_aic, d_mean, w_mean):
        rows.append(
            {
                "duration_ms": -1,  # sentinel: across-duration mean
                "model": MODEL_LABEL[key],
                "model_key": key,
                "AIC": float(a),
                "delta_AIC": float(da),
                "akaike_weight": float(wi),
                "duration_winner": "mean",
            }
        )

    out = pd.DataFrame(rows)
    MC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MC_DIR / "aic_per_duration_weights.csv", index=False)

    print("\n[A] Per-duration AIC table:")
    print(piv.round(1).to_string())
    print("\n[A] Mean-AIC Delta and Akaike weights:")
    summ = (
        out[out.duration_ms == -1][["model", "AIC", "delta_AIC", "akaike_weight"]]
        .round({"AIC": 1, "delta_AIC": 2, "akaike_weight": 3})
    )
    print(summ.to_string(index=False))
    print(f"\n[A] Hopfield beats no-cleanup Poisson in {beats_nocleanup}/5 durations.")
    print(f"[A] Saved {MC_DIR / 'aic_per_duration_weights.csv'}")
    return out


# ──────────────────────────────────────────────────────────────────────────
# B. Shared-(c, kappa) joint fit across durations  (reviewer point 4)
# ──────────────────────────────────────────────────────────────────────────

# The live config must match the published main model for the simulator to
# reproduce it: Poisson wells, entropy RT, fixed t0=1.3, free [sigma_c, s0,
# lambda0, kappa].
assert fh.WELL_TYPE == "poisson", "config.yaml must use well_type: poisson"
assert fh.RT_MODE == "entropy", "config.yaml must use rt_mode: entropy"
assert list(fh.PARAM_NAMES) == ["sigma_c", "s0", "lambda0", "kappa"], (
    f"unexpected PARAM_NAMES {fh.PARAM_NAMES}"
)

SIGMA_C_BOUNDS = (0.01, 0.50)
KAPPA_BOUNDS = (0.1, 3.0)
S0_BOUNDS = (1e-6, 0.1)
LAMBDA0_BOUNDS = (0.1, 4.0)


def _emp_arrays_by_duration() -> dict[int, dict]:
    return {d: fh._extract_empirical_arrays(fh.extract_empirical(d)) for d in DURATIONS}


def _theta_vec(sigma_c: float, s0: float, lambda0: float, kappa: float) -> np.ndarray:
    # order follows fh.PARAM_NAMES
    return np.array([sigma_c, s0, lambda0, kappa], dtype=float)


def _shared_objective(
    vec: np.ndarray,
    emp_by_dur: dict[int, dict],
    n_samples: int,
    seed: int,
) -> float:
    sigma_c, kappa = vec[0], vec[1]
    total = 0.0
    for i, d in enumerate(DURATIONS):
        s0 = vec[2 + 2 * i]
        lambda0 = vec[3 + 2 * i]
        tv = _theta_vec(sigma_c, s0, lambda0, kappa)
        total += fh._single_loss(tv, emp_by_dur[d], n_samples, seed)
    return float(total)


def analysis_b_shared_fit(
    n_samples_fit: int = 600,
    n_samples_eval: int = 5000,
    maxiter: int = 60,
    seed: int = 73,
) -> pd.DataFrame:
    emp_by_dur = _emp_arrays_by_duration()
    bounds = [SIGMA_C_BOUNDS, KAPPA_BOUNDS]
    for _ in DURATIONS:
        bounds += [S0_BOUNDS, LAMBDA0_BOUNDS]

    print("\n[B] Fitting shared-(c, kappa) joint model (12 params) ...")
    result = differential_evolution(
        _shared_objective,
        bounds=bounds,
        args=(emp_by_dur, n_samples_fit, seed),
        maxiter=maxiter,
        popsize=8,
        seed=17,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.85,
        polish=True,
        init="latinhypercube",
        updating="immediate",
        workers=1,
        disp=True,
    )
    vec = result.x
    sigma_c_shared, kappa_shared = float(vec[0]), float(vec[1])
    print(f"[B] shared sigma_c = {sigma_c_shared:.4f}, shared kappa = {kappa_shared:.4f}")

    # Per-duration goodness for the constrained model (final high-sample eval).
    rows: list[dict] = []
    logL_total = 0.0
    n_total = 0
    for i, d in enumerate(DURATIONS):
        s0 = float(vec[2 + 2 * i])
        lambda0 = float(vec[3 + 2 * i])
        theta = fh.complete_piantadosi_theta(
            {"sigma_c": sigma_c_shared, "s0": s0, "lambda0": lambda0, "kappa": kappa_shared}
        )
        emp = fh.extract_empirical(d)
        model = fh.simulate(theta, n_samples_per_n=n_samples_eval, seed=seed)
        # k=0 here: we account for parameters at the pooled level, not per duration
        g = fh.compute_goodness(emp, model, k=0)
        logL_total += float(g["logL"])
        n_total += int(g["n_data"])
        rows.append(
            {
                "duration_ms": d,
                "sigma_c_shared": sigma_c_shared,
                "kappa_shared": kappa_shared,
                "s0": s0,
                "lambda0": lambda0,
                "R2_mean_guess": g["R2_mean_guess"],
                "R2_pct_exact": g["R2_pct_exact"],
                "R2_rt": g["R2_rt"],
                "logL": float(g["logL"]),
                "n_data": int(g["n_data"]),
            }
        )
    per_dur = pd.DataFrame(rows)

    # Pooled information criteria: constrained k=12 vs free per-duration k=20.
    k_constrained = 2 + 2 * len(DURATIONS)
    aic_constrained = 2 * k_constrained - 2 * logL_total
    bic_constrained = k_constrained * np.log(n_total) - 2 * logL_total

    free = pd.read_csv(PIANT_DIR / "summary_all_durations.csv")
    logL_free = float(free["logL"].sum())
    n_free = int(free["n_data"].sum())
    k_free = int(free["k_params"].sum())  # 4 per duration * 5 = 20
    aic_free = 2 * k_free - 2 * logL_free
    bic_free = k_free * np.log(n_free) - 2 * logL_free

    summary = pd.DataFrame(
        [
            {
                "model": "Per-duration (free c, kappa)",
                "k_params_total": k_free,
                "logL_total": logL_free,
                "AIC": aic_free,
                "BIC": bic_free,
                "n_data": n_free,
            },
            {
                "model": "Shared c & kappa (sensory free)",
                "k_params_total": k_constrained,
                "logL_total": logL_total,
                "AIC": aic_constrained,
                "BIC": bic_constrained,
                "n_data": n_total,
            },
        ]
    )
    summary["delta_AIC"] = summary["AIC"] - summary["AIC"].min()
    summary["delta_BIC"] = summary["BIC"] - summary["BIC"].min()

    DECARLI_DIR  # noqa  (keep imports tidy)
    PIANT_DIR.mkdir(parents=True, exist_ok=True)
    per_dur.to_csv(PIANT_DIR / "shared_ckappa_per_duration.csv", index=False)
    summary.to_csv(PIANT_DIR / "shared_ckappa_model_comparison.csv", index=False)

    print("\n[B] Constrained per-duration fit (shared c, kappa):")
    print(
        per_dur[["duration_ms", "s0", "lambda0", "R2_mean_guess", "R2_pct_exact", "R2_rt"]]
        .round(4)
        .to_string(index=False)
    )
    print("\n[B] Pooled model comparison (free per-duration vs shared c & kappa):")
    print(
        summary[["model", "k_params_total", "logL_total", "AIC", "BIC", "delta_AIC", "delta_BIC"]]
        .round({"logL_total": 1, "AIC": 1, "BIC": 1, "delta_AIC": 1, "delta_BIC": 1})
        .to_string(index=False)
    )
    print(f"\n[B] Saved {PIANT_DIR / 'shared_ckappa_per_duration.csv'}")
    print(f"[B] Saved {PIANT_DIR / 'shared_ckappa_model_comparison.csv'}")
    return summary


# ──────────────────────────────────────────────────────────────────────────
# C. DD vs control derived quantities  (reviewer point 3)
# ──────────────────────────────────────────────────────────────────────────

DERIVED_N = (1, 4, 7)  # representative numerosities within the fitted DD range


def _derived_from_draws(draws: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["n_crit"] = 1.0 / draws["sigma_c"]
    for n in DERIVED_N:
        out[f"sigma_n{n}"] = np.sqrt(draws["s0"] * np.power(float(n), draws["lambda0"]))
    # mean noise SD across the fitted range n = 1..7
    ns = np.arange(1, 8)
    sig = np.sqrt(
        draws["s0"].to_numpy()[:, None] * np.power(ns[None, :], draws["lambda0"].to_numpy()[:, None])
    )
    out["sigma_mean_1to7"] = sig.mean(axis=1)
    return out


def _ci(x: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.median(x)),
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    )


def analysis_c_dd_derived(seed: int = 7) -> pd.DataFrame:
    dd = pd.read_csv(DECARLI_DIR / "bootstrap_draws_DD.csv")
    ct = pd.read_csv(DECARLI_DIR / "bootstrap_draws_controls.csv")
    der_dd = _derived_from_draws(dd)
    der_ct = _derived_from_draws(ct)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for q in der_dd.columns:
        m_dd, lo_dd, hi_dd = _ci(der_dd[q].to_numpy())
        m_ct, lo_ct, hi_ct = _ci(der_ct[q].to_numpy())
        # Difference distribution via random pairing of independent draws.
        a = der_dd[q].to_numpy()
        b = der_ct[q].to_numpy()
        n = min(len(a), len(b))
        diff = rng.permutation(a)[:n] - rng.permutation(b)[:n]
        m_d, lo_d, hi_d = _ci(diff)
        frac_dd_gt = float(np.mean(diff > 0))
        rows.append(
            {
                "quantity": q,
                "DD_median": m_dd,
                "DD_ci_lo": lo_dd,
                "DD_ci_hi": hi_dd,
                "Control_median": m_ct,
                "Control_ci_lo": lo_ct,
                "Control_ci_hi": hi_ct,
                "diff_median": m_d,
                "diff_ci_lo": lo_d,
                "diff_ci_hi": hi_d,
                "P(DD>Control)": frac_dd_gt,
            }
        )
    out = pd.DataFrame(rows)
    DECARLI_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DECARLI_DIR / "derived_quantities.csv", index=False)

    print("\n[C] DD vs control derived quantities (bootstrap medians, 95% CI):")
    show = out.copy()
    for c in show.columns:
        if c != "quantity":
            show[c] = show[c].round(4)
    print(show.to_string(index=False))
    print(f"\n[C] Saved {DECARLI_DIR / 'derived_quantities.csv'}")
    return out


def main() -> None:
    analysis_a_aic_weights()
    analysis_b_shared_fit()
    analysis_c_dd_derived()


if __name__ == "__main__":
    main()
