"""Tests for lightweight effective baselines (WP-3)."""

import numpy as np
import pytest

from src.models.effective_baselines import (
    AccumulatorEffective,
    ANSEffective,
    compute_aic,
    compute_bic,
)


RNG = np.random.default_rng(23)
N_TRUE = RNG.integers(1, 10, size=500)


class TestAccumulatorEffective:
    def test_predict_shape(self):
        model = AccumulatorEffective()
        preds = model.predict(N_TRUE, rng=np.random.default_rng(5))
        assert preds.shape == N_TRUE.shape

    def test_log_likelihood_finite(self):
        model = AccumulatorEffective()
        n_hat = N_TRUE + RNG.normal(0, 0.5, size=len(N_TRUE))
        ll = model.log_likelihood(N_TRUE, n_hat)
        assert np.isfinite(ll)

    def test_fit_returns_self(self):
        model = AccumulatorEffective()
        n_hat = N_TRUE + RNG.normal(0, 0.3, size=len(N_TRUE))
        result = model.fit(N_TRUE, n_hat)
        assert result is model

    def test_fit_improves_likelihood(self):
        n_hat = N_TRUE.astype(float) + RNG.normal(0, 0.5, size=len(N_TRUE))
        model_before = AccumulatorEffective(sigma_count=2.0, sigma_floor=2.0)
        ll_before = model_before.log_likelihood(N_TRUE, n_hat)
        model_after = AccumulatorEffective(sigma_count=2.0, sigma_floor=2.0).fit(N_TRUE, n_hat)
        ll_after = model_after.log_likelihood(N_TRUE, n_hat)
        assert ll_after >= ll_before

    def test_accuracy_in_unit_interval(self):
        model = AccumulatorEffective()
        acc = model.accuracy(N_TRUE, N_TRUE)
        assert 0.0 <= acc <= 1.0


class TestANSEffective:
    def test_predict_y_shape(self):
        model = ANSEffective()
        y = model.predict_y(N_TRUE, rng=np.random.default_rng(5))
        assert y.shape == N_TRUE.shape

    def test_log_likelihood_finite(self):
        model = ANSEffective()
        y_obs = model.predict_y(N_TRUE, rng=np.random.default_rng(5))
        ll = model.log_likelihood(N_TRUE, y_obs)
        assert np.isfinite(ll)

    def test_fit_returns_self(self):
        model = ANSEffective()
        y_obs = model.predict_y(N_TRUE, rng=np.random.default_rng(5))
        result = model.fit(N_TRUE, y_obs)
        assert result is model

    def test_fit_improves_likelihood(self):
        model_gen = ANSEffective(nu=1.0, rho=0.5, sigma=0.5, lam=0.5)
        y_obs = model_gen.predict_y(N_TRUE, rng=np.random.default_rng(5))
        model_bad = ANSEffective(nu=3.0, rho=0.1, sigma=2.0, lam=0.1)
        ll_before = model_bad.log_likelihood(N_TRUE, y_obs)
        model_fitted = ANSEffective(nu=3.0, rho=0.1, sigma=2.0, lam=0.1).fit(N_TRUE, y_obs)
        ll_after = model_fitted.log_likelihood(N_TRUE, y_obs)
        assert ll_after >= ll_before


class TestMetricHelpers:
    def test_aic_larger_with_more_params(self):
        nll = 1.0
        n = 100
        aic_2 = compute_aic(nll, 2, n)
        aic_4 = compute_aic(nll, 4, n)
        assert aic_4 > aic_2

    def test_bic_larger_with_more_params(self):
        nll = 1.0
        n = 100
        bic_2 = compute_bic(nll, 2, n)
        bic_4 = compute_bic(nll, 4, n)
        assert bic_4 > bic_2
