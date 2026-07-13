"""Smoke tests for scripts.density_transition.

Assertions focus on three things:

  1. The extended-support simulation returns consistent shapes and
     produces bounded, normalised posteriors.
  2. The analytical ``n_width = 1/c`` and the numerical ``d' < 1``
     criterion agree to within a single integer (the numerical rule
     can only land on an integer grid point).
  3. The four per-n metrics behave monotonically in the expected
     direction: entropy grows with n, mode probability decays with n,
     and the Bhattacharyya coefficient between adjacent decision
     distributions grows with n.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.density_transition import (
    compute_metrics,
    critical_n_from_metrics,
    load_fitted_params,
    reynvoet_weber_validation,
    simulate_extended,
)


SEED = 11


@pytest.fixture(scope="module")
def theta_ref() -> dict:
    return load_fitted_params(640)


@pytest.fixture(scope="module")
def sim_small(theta_ref: dict) -> dict:
    return simulate_extended(
        theta_ref,
        n_values=range(1, 41),
        m_trials=1500,
        seed=SEED,
    )


def test_simulate_shapes_and_normalisation(sim_small: dict) -> None:
    N = sim_small["n_values"].size
    M = sim_small["posteriors"].shape[1]
    K = sim_small["integers"].size
    assert sim_small["posteriors"].shape == (N, M, K)
    assert sim_small["n_hat"].shape == (N, M)
    row_sums = sim_small["posteriors"].sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)
    assert (sim_small["posteriors"] >= 0).all()


def test_metrics_monotonicity(sim_small: dict, theta_ref: dict) -> None:
    metrics = compute_metrics(sim_small, c=theta_ref["sigma_c"])
    H = metrics["H_mean"].values
    p_max = metrics["p_max"].values
    bc = metrics["BC_adjacent"].values[:-1]
    n = metrics["n"].values
    # Allow small Monte-Carlo wobble via Spearman-style monotonicity:
    # the ranks of low-n vs high-n bins should differ in the expected
    # direction by a wide margin.
    low_mask, high_mask = (n <= 5), (n >= 25)
    assert H[high_mask].mean() > H[low_mask].mean() + 0.5
    assert p_max[low_mask].mean() > p_max[high_mask].mean() + 0.3
    assert bc[high_mask[:-1]].mean() > bc[low_mask[:-1]].mean() + 0.3


def test_dprime_matches_one_over_c(sim_small: dict, theta_ref: dict) -> None:
    metrics = compute_metrics(sim_small, c=theta_ref["sigma_c"])
    crit = critical_n_from_metrics(metrics, c=theta_ref["sigma_c"])
    # d' < 1 must fire on the integer grid nearest to 1/c, within one step.
    assert abs(crit["n_width_dprime"] - crit["n_width_analytical"]) <= 1.0


def test_reynvoet_sqrt_model_closed_form(theta_ref: dict) -> None:
    rv = reynvoet_weber_validation(theta_ref, c=theta_ref["sigma_c"])
    # a_sqrt_theory is the raw-decoder prediction; a_sqrt_fit is LS on w_eff.
    # They should both be positive and finite, and the AIC summary should
    # be finite.
    assert rv["a_sqrt_fit"] > 0 and np.isfinite(rv["a_sqrt_fit"])
    assert rv["a_sqrt_theory"] > 0 and np.isfinite(rv["a_sqrt_theory"])
    assert np.isfinite(rv["delta_aic"])
