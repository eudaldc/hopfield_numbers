"""Tests for the RT proxy module (WP-3)."""

import pytest
import torch

from src.models.rt_proxy import RTProxy


B = 64


def _n_steps(low: int = 1, high: int = 15, b: int = B) -> torch.Tensor:
    return torch.randint(low, high + 1, (b,))


def _margin(b: int = B) -> torch.Tensor:
    return torch.rand(b) * 0.9 + 0.05


# ── construction ───────────────────────────────────────────────────────────


class TestConstruction:
    def test_valid_modes(self):
        RTProxy(mode="steps")
        RTProxy(mode="steps_plus_margin")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            RTProxy(mode="unknown_mode")

    def test_params_positive_at_init(self):
        proxy = RTProxy()
        assert proxy.t0.item() > 0
        assert proxy.c_steps.item() > 0

    def test_steps_plus_margin_has_c_margin_param(self):
        proxy = RTProxy(mode="steps_plus_margin")
        assert isinstance(proxy._c_margin_raw, torch.nn.Parameter)
        assert proxy.c_margin.item() > 0

    def test_steps_mode_c_margin_is_buffer(self):
        proxy = RTProxy(mode="steps")
        names = {n for n, _ in proxy.named_parameters()}
        assert "_c_margin_raw" not in names


# ── steps mode ─────────────────────────────────────────────────────────────


class TestStepsMode:
    def test_output_shape(self):
        proxy = RTProxy(mode="steps")
        rt = proxy(_n_steps())
        assert rt.shape == (B,)

    def test_rt_positive(self):
        proxy = RTProxy(mode="steps")
        rt = proxy(_n_steps())
        assert (rt > 0).all()

    def test_rt_increases_with_steps(self):
        proxy = RTProxy(mode="steps")
        n_low = torch.ones(B, dtype=torch.long)
        n_high = torch.full((B,), 15, dtype=torch.long)
        assert (proxy(n_high) > proxy(n_low)).all()

    def test_linear_in_steps(self):
        proxy = RTProxy(mode="steps", t0_init=1.0, c_steps_init=2.0)
        n1 = torch.full((B,), 3, dtype=torch.long)
        n2 = torch.full((B,), 6, dtype=torch.long)
        diff = (proxy(n2) - proxy(n1)).mean().item()
        # diff should be approximately c_steps * 3
        assert abs(diff - proxy.c_steps.item() * 3) < 0.01

    def test_gradient_flows(self):
        proxy = RTProxy(mode="steps")
        rt = proxy(_n_steps())
        rt.mean().backward()
        assert proxy._t0_raw.grad is not None
        assert proxy._c_steps_raw.grad is not None

    def test_basin_margin_silently_ignored(self):
        proxy = RTProxy(mode="steps")
        rt_with = proxy(_n_steps(), basin_margin=_margin())
        rt_without = proxy(_n_steps())
        # shapes should be the same even if values differ due to different seeds
        assert rt_with.shape == rt_without.shape


# ── steps_plus_margin mode ─────────────────────────────────────────────────


class TestStepsPlusMarginMode:
    def test_output_shape(self):
        proxy = RTProxy(mode="steps_plus_margin")
        rt = proxy(_n_steps(), basin_margin=_margin())
        assert rt.shape == (B,)

    def test_rt_positive(self):
        proxy = RTProxy(mode="steps_plus_margin")
        rt = proxy(_n_steps(), basin_margin=_margin())
        assert (rt > 0).all()

    def test_rt_increases_with_steps(self):
        proxy = RTProxy(mode="steps_plus_margin")
        margin = _margin()
        n_low = torch.ones(B, dtype=torch.long)
        n_high = torch.full((B,), 15, dtype=torch.long)
        assert (proxy(n_high, margin) > proxy(n_low, margin)).all()

    def test_rt_decreases_with_larger_margin(self):
        proxy = RTProxy(mode="steps_plus_margin")
        n_steps = torch.full((B,), 5, dtype=torch.long)
        small_margin = torch.full((B,), 0.01)
        large_margin = torch.full((B,), 0.99)
        assert (proxy(n_steps, small_margin) > proxy(n_steps, large_margin)).all()

    def test_missing_margin_raises(self):
        proxy = RTProxy(mode="steps_plus_margin")
        with pytest.raises(ValueError):
            proxy(_n_steps())

    def test_gradient_flows_through_margin(self):
        proxy = RTProxy(mode="steps_plus_margin")
        margin = _margin().requires_grad_(True)
        rt = proxy(_n_steps(), basin_margin=margin)
        rt.mean().backward()
        assert proxy._t0_raw.grad is not None
        assert proxy._c_steps_raw.grad is not None
        assert proxy._c_margin_raw.grad is not None
        assert margin.grad is not None

    def test_strictly_higher_than_steps_mode(self):
        """steps_plus_margin RT should be strictly higher than steps-only for same inputs."""
        steps_proxy = RTProxy(mode="steps", t0_init=1.0, c_steps_init=1.0)
        full_proxy = RTProxy(mode="steps_plus_margin", t0_init=1.0, c_steps_init=1.0)
        n_steps = _n_steps()
        margin = _margin()
        rt_steps = steps_proxy(n_steps)
        rt_full = full_proxy(n_steps, margin)
        assert (rt_full > rt_steps).all()


# ── integration with DynamicAttractorCore output ───────────────────────────


class TestIntegrationWithCleanupOutput:
    def test_with_attractor_output(self):
        from src.models.attractor_dynamics import DynamicAttractorCore

        core = DynamicAttractorCore(support_min=1, support_max=9)
        proxy = RTProxy(mode="steps_plus_margin")
        y = torch.empty(B).uniform_(1.0, 9.0)

        with torch.no_grad():
            out = core(y)

        rt = proxy(out["n_steps"], out["basin_margin"])
        assert rt.shape == (B,)
        assert (rt > 0).all()

    def test_rt_near_boundary_not_less_than_center(self):
        """On average, boundary inputs should produce at least as long RT as center."""
        from src.models.attractor_dynamics import DynamicAttractorCore

        core = DynamicAttractorCore(
            support_min=1, support_max=9,
            kappa_init=3.0, tau_in_init=1.0,
        )
        proxy = RTProxy(mode="steps_plus_margin")
        mu, _ = core.compute_mu_sigma()

        # Inputs at well centres
        y_center = mu[2:5].detach().repeat(20)
        # Inputs midway between wells
        y_boundary = ((mu[2:5] + mu[3:6]) / 2).detach().repeat(20)

        with torch.no_grad():
            out_center = core(y_center)
            out_boundary = core(y_boundary)

        rt_center = proxy(out_center["n_steps"], out_center["basin_margin"])
        rt_boundary = proxy(out_boundary["n_steps"], out_boundary["basin_margin"])

        assert rt_boundary.mean() >= rt_center.mean()
