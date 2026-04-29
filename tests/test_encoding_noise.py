"""Sensory encoding noise: variance vs standard deviation convention."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import fit_helpers
from scripts.fit_helpers import (
    encoding_sd,
    encoding_variance,
    evidence_noise_std,
)


def test_encoding_variance_reference_n40():
    v = encoding_variance(40, 0.010, 2.19)
    assert float(v) == pytest.approx(32.2486382815, rel=1e-6)


def test_encoding_sd_reference_n40():
    sd = encoding_sd(40, 0.010, 2.19)
    assert float(sd) == pytest.approx(5.6787884519, rel=1e-6)
    assert float(sd) == pytest.approx(np.sqrt(32.2486382815), rel=1e-6)


def test_encoding_helpers_numpy_array():
    n = np.array([1.0, 10.0, 40.0])
    v = encoding_variance(n, 0.02, 1.5)
    assert v.shape == (3,)
    assert np.all(v > 0)
    assert np.all(np.isfinite(v))
    sd = encoding_sd(n, 0.02, 1.5)
    assert np.allclose(sd, np.sqrt(v))


def test_encoding_helpers_scalar_types():
    v = encoding_variance(np.float64(3.0), 0.1, 1.0)
    assert float(v) == pytest.approx(0.3)
    sd = encoding_sd(3.0, 0.1, 1.0)
    assert float(sd) == pytest.approx(np.sqrt(0.3))


def test_encoding_variance_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n > 0"):
        encoding_variance(0, 0.1, 1.0)
    with pytest.raises(ValueError, match="n > 0"):
        encoding_variance(np.array([1.0, -1.0]), 0.1, 1.0)


def test_evidence_noise_std_matches_sqrt_sum_variances():
    if fit_helpers.CONSTANT_NOISE:
        pytest.skip("constant_noise uses only s00")
    s00 = float(fit_helpers.FIXED["s00"])
    theta = {"s0": 0.01, "lambda0": 2.19}
    v_pl = float(encoding_variance(40, theta["s0"], theta["lambda0"]))
    expected = float(np.sqrt(s00 * s00 + v_pl))
    assert evidence_noise_std(40, theta) == pytest.approx(expected)


def test_simulate_uses_std_not_variance_as_torch_scale():
    theta = {
        "sigma_c": 0.1,
        "s0": 0.01,
        "lambda0": 2.19,
        "t0": 1.0,
        "tau_in": 1.0,
        "beta": 0.5,
        "kappa_lse": 1.0,
        "eta": 1.0,
        "kappa": 0.5,
        "c_steps": 0.1,
        "rho": 1.0,
    }
    theta = fit_helpers.complete_piantadosi_theta(theta)
    n = 40
    std = evidence_noise_std(n, theta)
    n_draws = 50_000
    g = torch.Generator().manual_seed(17)
    noise = std * torch.randn(n_draws, generator=g)
    emp = float(noise.std(unbiased=True))
    assert emp == pytest.approx(std, rel=0.02)

