"""Tests for the trial-level (multinomial + RT) model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import trial_level_model_comparison as T


QUAD = T.make_quadrature(31)
THETA_POISSON = dict(sigma_c=0.043, s0=0.0018, lambda0=3.4)


def test_quadrature_weights_sum_to_one() -> None:
    for n in (21, 41, 61):
        q = T.make_quadrature(n)
        assert abs(float(q.w.sum()) - 1.0) < 1e-12


def test_response_probs_sum_to_one_all_models() -> None:
    thetas = {
        "poisson": THETA_POISSON,
        "poisson_pow": dict(sigma_c=0.05, s0=0.0018, lambda0=3.4, qvar=2.0),
        "gaussian": dict(sigma_c=0.05, s0=0.0018, lambda0=3.4),
        "gaussian_freebeta": dict(sigma_c=0.05, s0=0.0018, lambda0=3.4, beta=1.8),
        "gaussian_pow": dict(sigma_c=0.1, s0=0.0018, lambda0=3.4, pow=1.6),
        "gaussian_const": dict(sigma_c=0.1, s0=0.0018, lambda0=3.4),
        "weber": dict(sigma0=0.2, exponent=0.5, decision_width=0.15),
        "cp": dict(sigma_log=0.3, alpha=0.5, readout_temp=1.0),
    }
    for spec in T.MODEL_SPECS:
        grid = T.compute_grid(spec, thetas[spec.family], QUAD)
        rowsums = grid["p_resp"].sum(axis=1)
        assert np.allclose(rowsums, 1.0, atol=1e-9), spec.key


def test_quadrature_matches_monte_carlo() -> None:
    nsub = np.array([1, 3, 5, 8, 12])  # representative numerosities (keeps it fast)
    # No-cleanup models integrate a smooth softmax -> Gauss-Hermite is exact.
    spec = T.MODEL_BY_KEY["no_cleanup_poisson"]
    gh = T.compute_grid(spec, THETA_POISSON, T.make_quadrature(61), n_values=nsub)["p_resp"]
    mc = T.simulate_response_mc(spec, THETA_POISSON, n_values=nsub, n_samples=150_000, seed=3)
    # threshold allows for Monte-Carlo sampling noise (~1.3e-3 per cell)
    assert np.abs(gh - mc).max() < 8e-3
    # Attractor clean-up makes the integrand step-like; the robust grid scheme
    # (not Gauss-Hermite) must match a high-sample Monte-Carlo run.
    spec_a = T.MODEL_BY_KEY["hopfield_poisson"]
    grid = T.compute_grid(spec_a, THETA_POISSON,
                          T.make_quadrature(801, kind="grid"), n_values=nsub)["p_resp"]
    mc_a = T.simulate_response_mc(spec_a, THETA_POISSON, n_values=nsub,
                                  n_samples=150_000, seed=4)
    assert np.abs(grid - mc_a).max() < 8e-3


def test_no_cleanup_equals_attractor_with_cleanup_disabled() -> None:
    attractor = T.MODEL_BY_KEY["hopfield_poisson"]
    static = T.MODEL_BY_KEY["no_cleanup_poisson"]
    g_off = T.compute_grid(attractor, THETA_POISSON, QUAD, force_no_cleanup=True)
    g_static = T.compute_grid(static, THETA_POISSON, QUAD)
    assert np.allclose(g_off["p_resp"], g_static["p_resp"], atol=1e-12)
    assert np.allclose(g_off["ET"], 0.0)
    assert np.allclose(g_static["ET"], 0.0)


def test_attractor_has_positive_convergence_steps() -> None:
    grid = T.compute_grid(T.MODEL_BY_KEY["hopfield_poisson"], THETA_POISSON, QUAD)
    assert np.all(grid["ET"] >= 0.0)
    assert grid["ET"].max() > 0.0  # the dynamics actually run


def test_flow_time_present_and_consistent() -> None:
    spec = T.MODEL_BY_KEY["hopfield_poisson"]
    grid = T.compute_grid(spec, THETA_POISSON, QUAD)
    assert "ET_flow" in grid and "flowtime_nodes" in grid
    assert np.all(grid["ET_flow"] >= 0.0)
    # flow-time discounts clamped steps, so it never exceeds the raw step count
    assert np.all(grid["flowtime_nodes"] <= grid["steps_nodes"] + 1e-9)
    # no-clean-up -> both zero
    g0 = T.compute_grid(spec, THETA_POISSON, QUAD, force_no_cleanup=True)
    assert np.allclose(g0["ET_flow"], 0.0) and np.allclose(g0["ET"], 0.0)


def test_poisson_pow_reduces_to_poisson_at_qvar_one() -> None:
    pois = T.MODEL_BY_KEY["hopfield_poisson"]
    ppow = T.ModelSpec("pp", "PP", "poisson_pow", True,
                       ("sigma_c", "s0", "lambda0", "qvar"), ((0, 1),) * 4)
    g_p = T.compute_grid(pois, THETA_POISSON, QUAD)
    g_pp = T.compute_grid(ppow, dict(THETA_POISSON, qvar=1.0), QUAD)
    assert np.abs(g_p["p_resp"] - g_pp["p_resp"]).max() < 1e-10
    assert np.allclose(g_p["ET"], g_pp["ET"])


def test_constant_width_reduces_to_power_law_at_p_zero() -> None:
    theta = dict(sigma_c=0.1, s0=0.0018, lambda0=3.4)
    const = T.MODEL_BY_KEY["gaussian_const_attractor"]
    power = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    g_const = T.compute_grid(const, theta, QUAD)
    g_power = T.compute_grid(power, dict(theta, pow=0.0), QUAD)
    assert np.allclose(g_const["p_resp"], g_power["p_resp"], atol=1e-12)
    assert np.allclose(g_const["ET"], g_power["ET"])


def test_winner_is_registered_with_dynamics() -> None:
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    assert spec.family == "gaussian_pow" and spec.cleanup
    assert spec.has_dynamics  # so steps/hybrid RT metrics apply


def test_lapse_keeps_all_probabilities_positive() -> None:
    grid = T.compute_grid(T.MODEL_BY_KEY["hopfield_poisson"], THETA_POISSON, QUAD)
    r_max = 99
    pf = T.apply_lapse(grid["p_resp"], lapse=0.03, r_max=r_max)
    assert pf.shape == (T.N_SHOWN.size, r_max)
    assert pf.min() > 0.0                         # nothing is exactly zero
    assert np.allclose(pf.sum(axis=1), 1.0)       # still a distribution
    # out-of-well responses carry only the lapse mass
    assert np.allclose(pf[:, T.N_WELLS:], 0.03 / r_max)


def _synthetic_counts(spec, theta, lapse, r_max, n_per_n=400) -> np.ndarray:
    grid = T.compute_grid(spec, theta, QUAD)
    pf = T.apply_lapse(grid["p_resp"], lapse, r_max)
    return pf * n_per_n  # expected (fractional) counts -- deterministic


def test_response_support_handles_out_of_range_without_crashing() -> None:
    spec = T.MODEL_BY_KEY["hopfield_poisson"]
    r_max = 99
    counts = np.zeros((T.N_SHOWN.size, r_max))
    # deliberately place mass on far out-of-support responses (r = 50, 99)
    counts[0, 49] = 5.0
    counts[5, 98] = 3.0
    counts[7, 14] = 10.0
    vec = np.array([0.043, 0.0018, 3.4, 0.05])
    ll = T.response_loglik(spec, vec, QUAD, counts, r_max)
    assert np.isfinite(ll)


def test_likelihood_is_maximised_near_generating_parameters() -> None:
    spec = T.MODEL_BY_KEY["hopfield_poisson"]
    r_max = 99
    lapse_true = 0.04
    counts = _synthetic_counts(spec, THETA_POISSON, lapse_true, r_max)
    vec_true = np.array([THETA_POISSON["sigma_c"], THETA_POISSON["s0"],
                         THETA_POISSON["lambda0"], lapse_true])
    ll_true = T.response_loglik(spec, vec_true, QUAD, counts, r_max)
    # perturb each parameter -> log-likelihood should not improve
    for scale in (0.7, 1.3):
        vec_bad = vec_true.copy()
        vec_bad[0] *= scale
        vec_bad[2] *= scale
        ll_bad = T.response_loglik(spec, vec_bad, QUAD, counts, r_max)
        assert ll_true > ll_bad


def test_fit_recovers_better_likelihood_than_start_and_is_reproducible() -> None:
    spec = T.MODEL_BY_KEY["no_cleanup_poisson"]   # fast (no cleanup loop)
    r_max = 99
    counts = _synthetic_counts(spec, THETA_POISSON, 0.05, r_max, n_per_n=600)
    data = T.DurationData(
        duration_ms=160,
        n_trial=np.repeat(T.N_SHOWN, 1),
        r_trial=np.ones(T.N_SHOWN.size, dtype=int),
        logrt_trial=np.zeros(T.N_SHOWN.size),
        counts=counts,
        n_total=int(counts.sum()),
    )
    fit1 = T.fit_response(spec, data, QUAD, r_max, maxiter=10, popsize=6, seed=11)
    fit2 = T.fit_response(spec, data, QUAD, r_max, maxiter=10, popsize=6, seed=11)
    # reproducible across identical runs
    assert fit1["logL"] == fit2["logL"]
    assert np.allclose(fit1["theta_vec"], fit2["theta_vec"])
    # the fit should beat a deliberately-wrong starting point
    bad = np.array([0.2, 0.05, 1.0, 0.3])
    ll_bad = T.response_loglik(spec, bad, QUAD, counts, r_max)
    assert fit1["logL"] > ll_bad
    assert np.isfinite(fit1["AIC"]) and np.isfinite(fit1["BIC"])


def test_load_trials_shape_and_support() -> None:
    trial = T.load_trials()
    assert {"exp_dur", "n_shown", "n_guess", "rt_s"}.issubset(trial.columns)
    assert trial["n_shown"].between(1, 15).all()
    assert (trial["rt_s"] > 0).all()
    report = T.response_support_report(trial)
    assert report["r_max"] >= T.N_SUPPORT_MAX
    assert 0.0 < report["frac_out_of_support_overall"] < 0.5
