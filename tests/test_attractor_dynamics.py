"""Tests for the DynamicAttractorCore (WP-1)."""

import pytest
import torch

from src.models.attractor_dynamics import DynamicAttractorCore


B = 32
N_MIN, N_MAX = 1, 9


def _make_core(cleanup_mode="gradient", **kwargs) -> DynamicAttractorCore:
    return DynamicAttractorCore(
        support_min=N_MIN,
        support_max=N_MAX,
        cleanup_mode=cleanup_mode,
        **kwargs,
    )


def _make_y(b: int = B, low: float = 1.0, high: float = 9.0) -> torch.Tensor:
    return torch.empty(b).uniform_(low, high)


# ── geometry ───────────────────────────────────────────────────────────────


class TestGeometry:
    def test_centers_shape(self):
        core = _make_core()
        mu, sigma = core.compute_mu_sigma()
        K = N_MAX - N_MIN + 1
        assert mu.shape == (K,)
        assert sigma.shape == (K,)

    def test_centers_monotone(self):
        core = _make_core()
        mu, _ = core.compute_mu_sigma()
        assert (mu[1:] > mu[:-1]).all()

    def test_widths_positive(self):
        core = _make_core()
        _, sigma = core.compute_mu_sigma()
        assert (sigma > 0).all()

    def test_constrained_params_positive(self):
        core = _make_core()
        assert core.nu.item() > 0
        assert core.sigma0.item() > 0
        assert core.tau_in.item() > 0
        assert core.kappa.item() > 0
        assert core.eta.item() > 0

    def test_separation_ratio_shape(self):
        core = _make_core()
        sr = core.separation_ratio()
        assert sr.shape == (N_MAX - N_MIN,)

    def test_k_star_geometry_scalar(self):
        core = _make_core()
        ks = core.k_star_geometry()
        assert ks.ndim == 0


# ── energy ─────────────────────────────────────────────────────────────────


class TestEnergy:
    def test_energy_shape(self):
        core = _make_core()
        y = _make_y()
        x = _make_y()
        e = core.energy(x, y)
        assert e.shape == (B,)

    def test_energy_lower_at_well_center(self):
        core = _make_core(kappa_init=5.0, tau_in_init=10.0)
        mu, _ = core.compute_mu_sigma()
        y = torch.full((1,), mu[3].item())
        e_center = core.energy(mu[3:4], y)
        e_off = core.energy(mu[3:4] + 0.5, y)
        assert e_center < e_off

    def test_posterior_sums_to_one(self):
        core = _make_core()
        x = _make_y()
        probs = core.posterior_from_x(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)


# ── cleanup — static mode ──────────────────────────────────────────────────


class TestStaticMode:
    def test_x_star_equals_y(self):
        core = _make_core(cleanup_mode="none")
        y = _make_y()
        out = core(y)
        assert torch.allclose(out["x_star"], y)

    def test_zero_steps(self):
        core = _make_core(cleanup_mode="none")
        y = _make_y()
        out = core(y)
        assert (out["n_steps"] == 0).all()

    def test_all_converged(self):
        core = _make_core(cleanup_mode="none")
        y = _make_y()
        out = core(y)
        assert out["converged"].all()

    def test_output_keys(self):
        core = _make_core(cleanup_mode="none")
        y = _make_y()
        out = core(y)
        required = {"y0", "x0", "x_star", "n_steps", "converged",
                    "probs_clean", "n_hat", "expected_n_clean", "basin_margin"}
        assert required <= out.keys()

    def test_traj_returned_when_requested(self):
        core = _make_core(cleanup_mode="none")
        y = _make_y()
        out = core(y, return_traj=True)
        assert "x_traj" in out
        assert "energy_trace" in out


# ── cleanup — gradient mode ────────────────────────────────────────────────


class TestGradientMode:
    def test_output_keys(self):
        core = _make_core()
        y = _make_y()
        out = core(y)
        required = {"y0", "x0", "x_star", "n_steps", "converged",
                    "probs_clean", "n_hat", "expected_n_clean", "basin_margin"}
        assert required <= out.keys()

    def test_shapes(self):
        core = _make_core()
        y = _make_y()
        out = core(y)
        K = N_MAX - N_MIN + 1
        assert out["x_star"].shape == (B,)
        assert out["probs_clean"].shape == (B, K)
        assert out["n_hat"].shape == (B,)
        assert out["n_steps"].shape == (B,)
        assert out["converged"].shape == (B,)
        assert out["basin_margin"].shape == (B,)

    def test_n_hat_in_support(self):
        core = _make_core()
        y = _make_y()
        out = core(y)
        assert (out["n_hat"] >= N_MIN).all()
        assert (out["n_hat"] <= N_MAX).all()

    def test_probs_sum_to_one(self):
        core = _make_core()
        y = _make_y()
        out = core(y)
        assert torch.allclose(out["probs_clean"].sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_strong_attraction_collapses_to_nearest(self):
        """With high kappa, nearby y values should produce the same n_hat."""
        core = _make_core(kappa_init=10.0, tau_in_init=0.1, eta_init=0.5)
        mu, _ = core.compute_mu_sigma()
        center = mu[4].item()
        y_near = torch.full((5,), center) + torch.linspace(-0.05, 0.05, 5)
        with torch.no_grad():
            out = core(y_near)
        assert out["n_hat"].unique().numel() == 1

    def test_weak_attraction_stays_near_y(self):
        """With tiny kappa, x* should stay close to y."""
        core = _make_core(kappa_init=0.001, tau_in_init=0.1, max_steps=16)
        y = _make_y()
        with torch.no_grad():
            out = core(y)
        assert (out["x_star"] - out["y0"]).abs().max() < 0.5

    def test_n_steps_bounded(self):
        core = _make_core(max_steps=8)
        y = _make_y()
        out = core(y)
        assert (out["n_steps"] <= 8).all()
        assert (out["n_steps"] >= 1).all()

    def test_traj_shape(self):
        core = _make_core(max_steps=4)
        y = _make_y(b=4)
        out = core(y, return_traj=True)
        T = out["x_traj"].shape[-1]
        assert T <= 5   # x0 + up to max_steps
        assert out["energy_trace"].shape == (4, T)

    def test_gradients_flow_through_x_star(self):
        core = _make_core()
        y = _make_y()
        out = core(y)
        loss = out["x_star"].mean()
        loss.backward()
        for name, p in core.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_through_probs(self):
        core = _make_core()
        y = _make_y()
        out = core(y)
        loss = out["probs_clean"].sum()
        loss.backward()
        assert core._kappa_raw.grad is not None

    def test_static_and_gradient_agree_at_zero_kappa(self):
        """At effectively zero kappa the dynamic core should stay near y."""
        core_dyn = _make_core(kappa_init=1e-4, tau_in_init=0.1, max_steps=16)
        core_sta = _make_core(cleanup_mode="none")
        y = _make_y()
        with torch.no_grad():
            out_dyn = core_dyn(y)
            out_sta = core_sta(y)
        assert torch.allclose(out_dyn["x_star"], out_sta["x_star"], atol=0.1)
