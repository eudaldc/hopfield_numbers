"""Tests for legacy synthetic front-ends (diagnostics only)."""

import pytest
import torch

from src.diagnostics.legacy_frontends import (
    ANSFrontend,
    AccumulatorFrontend,
    ExactSmallFrontend,
    build_frontend,
    FRONTEND_CODE,
    FRONTEND_REGISTRY,
)


N_MIN, N_MAX = 1, 9
B = 200


def _n(b: int = B) -> torch.Tensor:
    return torch.randint(N_MIN, N_MAX + 1, (b,))


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class TestANSFrontend:
    def test_sample_shape(self):
        y = ANSFrontend().sample(_n())
        assert y.shape == (B,)

    def test_sample_dtype(self):
        y = ANSFrontend().sample(_n())
        assert y.dtype == torch.float32

    def test_reproducible_with_same_seed(self):
        fe = ANSFrontend()
        n = _n()
        y1 = fe.sample(n, generator=_gen(11))
        y2 = fe.sample(n, generator=_gen(11))
        assert torch.allclose(y1, y2)

    def test_mean_monotone(self):
        fe = ANSFrontend()
        mu = fe.mean(torch.arange(N_MIN, N_MAX + 1))
        assert (mu[1:] > mu[:-1]).all()


class TestAccumulatorFrontend:
    def test_sample_shape(self):
        y = AccumulatorFrontend().sample(_n())
        assert y.shape == (B,)

    def test_linear_mean(self):
        fe = AccumulatorFrontend(alpha=2.0, beta=1.0, scale=1e-4)
        n = torch.arange(N_MIN, N_MAX + 1)
        y = fe.sample(n)
        expected = 2.0 * n.float() + 1.0
        assert torch.allclose(y, expected, atol=0.01)


class TestExactSmallFrontend:
    def test_exact_small_n_no_noise(self):
        fe = ExactSmallFrontend(exact_threshold=4, small_noise=0.0)
        n = torch.arange(1, 5)
        y = fe.sample(n)
        assert torch.allclose(y, n.float())


class TestBuildFrontend:
    def test_ans(self):
        assert isinstance(build_frontend("ans"), ANSFrontend)

    def test_accumulator(self):
        assert isinstance(build_frontend("accumulator"), AccumulatorFrontend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_frontend("bogus_frontend")

    def test_frontend_codes_unique(self):
        assert len(set(FRONTEND_CODE.values())) == len(FRONTEND_CODE)
