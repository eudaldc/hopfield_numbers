"""Synthetic sensory front-ends for numerosity generation.

Each front-end maps a true numerosity N to a noisy scalar evidence y,
mimicking different hypothesised representational schemes.

Usage
-----
    fe = ANSFrontend(sigma0=0.3)
    n_true = torch.randint(1, 10, (1000,))
    y = fe.sample(n_true)

Reproducibility
---------------
Pass a seeded ``torch.Generator`` to ``sample`` for deterministic output:

    g = torch.Generator(); g.manual_seed(7)
    y = fe.sample(n_true, generator=g)
"""

import math
from dataclasses import dataclass, field

import torch


# ── shared helper ──────────────────────────────────────────────────────────


def _boxcox(z: torch.Tensor, rho: float) -> torch.Tensor:
    """Box-Cox transform, continuous through rho = 0.

    boxcox(z, rho) = (z^rho - 1) / rho   (rho != 0)
                   = log(z)               (rho = 0)
    """
    if abs(rho) < 1e-6:
        return torch.log(z)
    return (z.pow(rho) - 1.0) / rho


# ── ANS front-end ──────────────────────────────────────────────────────────


@dataclass
class ANSFrontend:
    """ANS-like front-end: Box-Cox compressed mean with power-law noise.

    y ~ Normal( BoxCox(N + nu, rho),  sigma0 * (N + nu)^lambda )

    This matches the prototype geometry used by AttractorCore / DynamicAttractorCore,
    so sensory noise and attractor width share the same parametric family.
    """

    rho: float = 0.5
    nu: float = 1.0
    sigma0: float = 0.5
    lam: float = 0.5

    def mean(self, n_true: torch.Tensor) -> torch.Tensor:
        """Expected evidence mu(N).  Shape same as n_true (float)."""
        return _boxcox((n_true.float() + self.nu), self.rho)

    def std(self, n_true: torch.Tensor) -> torch.Tensor:
        """Standard deviation sigma(N).  Shape same as n_true (float)."""
        return self.sigma0 * (n_true.float() + self.nu).pow(self.lam)

    def sample(
        self,
        n_true: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noisy evidence y ~ Normal(mu(N), sigma(N)).

        Parameters
        ----------
        n_true    : integer tensor of true numerosities
        generator : optional seeded Generator for reproducibility

        Returns
        -------
        y : float tensor, same shape as n_true
        """
        mu = self.mean(n_true)
        sigma = self.std(n_true)
        eps = torch.empty_like(mu).normal_(generator=generator)
        return mu + sigma * eps


# ── Accumulator front-end ──────────────────────────────────────────────────


@dataclass
class AccumulatorFrontend:
    """Accumulator-like front-end: linear mean with square-root noise.

    y ~ Normal( alpha * N + beta,  scale * sqrt(N + offset) )

    This produces linear spacing of means (unlike the compressed ANS spacing),
    and noise that grows more slowly with N.
    """

    alpha: float = 1.0
    beta: float = 0.0
    scale: float = 1.0
    offset: float = 0.5

    def mean(self, n_true: torch.Tensor) -> torch.Tensor:
        """Expected evidence mu(N)."""
        return self.alpha * n_true.float() + self.beta

    def std(self, n_true: torch.Tensor) -> torch.Tensor:
        """Standard deviation sigma(N)."""
        return self.scale * (n_true.float() + self.offset).sqrt()

    def sample(
        self,
        n_true: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        mu = self.mean(n_true)
        sigma = self.std(n_true)
        eps = torch.empty_like(mu).normal_(generator=generator)
        return mu + sigma * eps


# ── Exact-small front-end ──────────────────────────────────────────────────


@dataclass
class ExactSmallFrontend:
    """Near-exact for N <= exact_threshold, ANS-like above.

    Useful for ablations: tests whether the attractor can simultaneously
    preserve near-exact small-N representations and handle noisier large-N.
    """

    exact_threshold: int = 4
    small_noise: float = 0.05
    _ans: ANSFrontend = field(default_factory=ANSFrontend, repr=False)

    def sample(
        self,
        n_true: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        n_float = n_true.float()
        eps_small = torch.empty_like(n_float).normal_(generator=generator)
        y_exact = n_float + self.small_noise * eps_small
        y_ans = self._ans.sample(n_true, generator=generator)
        return torch.where(n_true <= self.exact_threshold, y_exact, y_ans)


# ── registry ───────────────────────────────────────────────────────────────


FRONTEND_REGISTRY: dict[str, type] = {
    "ans": ANSFrontend,
    "accumulator": AccumulatorFrontend,
    "exact_small": ExactSmallFrontend,
}

FRONTEND_CODE: dict[str, int] = {
    "ans": 0,
    "accumulator": 1,
    "exact_small": 2,
}


def build_frontend(family: str, **kwargs):
    """Instantiate a front-end by name.

    Parameters
    ----------
    family : "ans" | "accumulator" | "exact_small"
    **kwargs : overrides for the front-end dataclass fields

    Returns
    -------
    One of ANSFrontend, AccumulatorFrontend, ExactSmallFrontend
    """
    if family not in FRONTEND_REGISTRY:
        raise ValueError(
            f"Unknown frontend family {family!r}. "
            f"Choose from: {list(FRONTEND_REGISTRY)}"
        )
    return FRONTEND_REGISTRY[family](**kwargs)
