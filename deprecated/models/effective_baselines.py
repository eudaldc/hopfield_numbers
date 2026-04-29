"""Lightweight effective baselines for regime-recovery analyses.

These are NOT full nn.Module model cores for empirical training.
They are simple parametric models fitted via MLE to attractor-generated
synthetic data, used to test whether:
  - an accumulator-like model fits the small-N regime
  - an ANS-like model fits the large-N regime
  - the attractor model fits the full range better than either
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


# ── Box-Cox helper (numpy) ─────────────────────────────────────────────────


def _boxcox_np(z: np.ndarray, rho: float) -> np.ndarray:
    if abs(rho) < 1e-6:
        return np.log(z)
    return (z ** rho - 1.0) / rho


# ── Accumulator-like effective model ───────────────────────────────────────


@dataclass
class AccumulatorEffective:
    """Accumulator-like: c ~ Normal(N, sigma_count * sqrt(N) + sigma_floor).

    Fittable params: sigma_count, sigma_floor.
    Response: nearest-integer from c.
    """

    sigma_count: float = 0.5
    sigma_floor: float = 0.1

    def _std(self, n: np.ndarray) -> np.ndarray:
        return self.sigma_count * np.sqrt(n.astype(float)) + self.sigma_floor

    def log_likelihood(self, n_true: np.ndarray, n_hat: np.ndarray) -> float:
        """Total log-likelihood of observing n_hat given n_true."""
        sigma = self._std(n_true)
        ll = -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-12) \
             - 0.5 * ((n_hat - n_true) / (sigma + 1e-12)) ** 2
        return float(ll.sum())

    def predict(self, n_true: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample predicted n_hat."""
        if rng is None:
            rng = np.random.default_rng()
        sigma = self._std(n_true)
        c = rng.normal(n_true.astype(float), sigma)
        return np.round(c).astype(int)

    def accuracy(self, n_true: np.ndarray, n_hat: np.ndarray) -> float:
        sigma = self._std(n_true)
        from scipy.stats import norm
        p_correct = norm.cdf(0.5, loc=0, scale=sigma) - norm.cdf(-0.5, loc=0, scale=sigma)
        return float(p_correct.mean())

    def fit(self, n_true: np.ndarray, n_hat: np.ndarray) -> "AccumulatorEffective":
        """Fit sigma_count and sigma_floor via MLE."""
        def _neg_ll(params):
            sc, sf = params
            sigma = sc * np.sqrt(n_true.astype(float)) + sf
            sigma = np.maximum(sigma, 1e-6)
            return -float((-0.5 * np.log(2 * np.pi) - np.log(sigma)
                           - 0.5 * ((n_hat - n_true) / sigma) ** 2).sum())

        result = optimize.minimize(
            _neg_ll, [self.sigma_count, self.sigma_floor],
            method="Nelder-Mead",
            bounds=[(0.01, 10.0), (0.001, 5.0)],
        )
        self.sigma_count, self.sigma_floor = result.x
        return self


# ── ANS-like effective model ───────────────────────────────────────────────


@dataclass
class ANSEffective:
    """ANS-like: z ~ Normal(BoxCox(N+nu, rho), sigma * (N+nu)^lam).

    Fittable params: nu, rho, sigma, lam.
    Response: inverse-BoxCox of z, rounded.
    """

    nu: float = 1.0
    rho: float = 0.5
    sigma: float = 0.5
    lam: float = 0.5

    def _mean(self, n: np.ndarray) -> np.ndarray:
        z = n.astype(float) + self.nu
        return _boxcox_np(z, self.rho)

    def _std(self, n: np.ndarray) -> np.ndarray:
        z = n.astype(float) + self.nu
        return self.sigma * z ** self.lam

    def log_likelihood(self, n_true: np.ndarray, y_observed: np.ndarray) -> float:
        """Log-likelihood in the latent (y) space."""
        mu = self._mean(n_true)
        sigma = self._std(n_true)
        ll = -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-12) \
             - 0.5 * ((y_observed - mu) / (sigma + 1e-12)) ** 2
        return float(ll.sum())

    def predict_y(self, n_true: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        mu = self._mean(n_true)
        sigma = self._std(n_true)
        return rng.normal(mu, sigma)

    def fit(self, n_true: np.ndarray, y_observed: np.ndarray) -> "ANSEffective":
        """Fit nu, rho, sigma, lam via MLE."""
        def _neg_ll(params):
            nu, rho, sigma, lam = params
            if nu < 0.01 or sigma < 0.01:
                return 1e12
            z = n_true.astype(float) + nu
            mu = _boxcox_np(z, rho)
            s = sigma * z ** lam
            s = np.maximum(s, 1e-6)
            return -float((-0.5 * np.log(2 * np.pi) - np.log(s)
                           - 0.5 * ((y_observed - mu) / s) ** 2).sum())

        result = optimize.minimize(
            _neg_ll, [self.nu, self.rho, self.sigma, self.lam],
            method="Nelder-Mead",
        )
        self.nu, self.rho, self.sigma, self.lam = result.x
        return self


# ── Metrics helpers ────────────────────────────────────────────────────────


def compute_nll(n_true: np.ndarray, n_hat: np.ndarray,
                sigma: np.ndarray) -> float:
    """Negative log-likelihood under Gaussian assumption."""
    ll = -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-12) \
         - 0.5 * ((n_hat - n_true) / (sigma + 1e-12)) ** 2
    return -float(ll.mean())


def compute_aic(nll: float, n_params: int, n_samples: int) -> float:
    return 2 * n_params + 2 * nll * n_samples


def compute_bic(nll: float, n_params: int, n_samples: int) -> float:
    return n_params * np.log(n_samples) + 2 * nll * n_samples
