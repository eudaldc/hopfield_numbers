"""ANS baseline: continuous Gaussian code without attractor wells.

Replaces the attractor core with a pure Gaussian likelihood model.
The key differences from the unified attractor model:

  1. No per-subject temperature — decisiveness is tied to noise level
  2. Proper Gaussian normalisation (-log σ) in the posterior

Centres follow a power-law compression:
    mu_n = (n + nu)^rho

Widths follow Weber-like scaling:
    sigma_n = sigma0 · (n + nu)^lambda

Posterior:
    log p(n | y) ∝ -0.5 · ((y - mu_n) / sigma_n_i)² - log(sigma_n_i),
    where sigma_n_i = sigma_n · width_subject_i

This module is a drop-in replacement for AttractorCore: same forward
signature, same output dict keys, and compatible with SymbolicHead
and all existing task heads via the ``modules`` dict.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import SupportConfig
from src.models.subject_params import SubjectParams, inverse_softplus


@dataclass
class ANSConfig:
    """Parameters for the ANS baseline (continuous Gaussian code)."""

    rho_init: float = 0.5
    nu_init: float = 1.0
    sigma0_init: float = 1.0
    lam_init: float = 0.5
    support: SupportConfig = field(default_factory=SupportConfig)


class ANSCore(nn.Module):
    """Continuous Gaussian code — no attractor dynamics.

    Parameters
    ----------
    cfg : ANSConfig
    subject_params : SubjectParams
    """

    def __init__(self, cfg: ANSConfig, subject_params: SubjectParams):
        super().__init__()
        self.support_cfg = cfg.support
        K = cfg.support.size
        n_min = cfg.support.n_min

        self.register_buffer(
            "integers",
            torch.arange(n_min, n_min + K, dtype=torch.float32),
        )

        self._rho_raw = nn.Parameter(
            torch.tensor(inverse_softplus(cfg.rho_init))
        )
        self._nu_raw = nn.Parameter(
            torch.tensor(inverse_softplus(cfg.nu_init))
        )
        self._sigma0_raw = nn.Parameter(
            torch.tensor(inverse_softplus(cfg.sigma0_init))
        )
        self.lam = nn.Parameter(torch.tensor(cfg.lam_init))

        self.subject_params = subject_params

    # -- constrained accessors ---------------------------------------------

    @property
    def rho(self) -> torch.Tensor:
        """Positive exponent for the power-law compression."""
        return F.softplus(self._rho_raw) + 1e-6

    @property
    def nu(self) -> torch.Tensor:
        """Positive offset (keeps n + nu > 0 across support)."""
        return F.softplus(self._nu_raw) + 1e-6

    @property
    def sigma0(self) -> torch.Tensor:
        """Positive base width."""
        return F.softplus(self._sigma0_raw) + 1e-6

    # -- geometry ----------------------------------------------------------

    def centers(self) -> torch.Tensor:
        """Power-law compressed centres  mu_n = (n + nu)^rho.  Shape [K]."""
        return (self.integers + self.nu).pow(self.rho)

    def widths(self) -> torch.Tensor:
        """Weber-like noise widths  sigma_n = sigma0 · (n + nu)^lam.  [K]."""
        return self.sigma0 * (self.integers + self.nu).pow(self.lam)

    def pairwise_separation(self) -> torch.Tensor:
        """Centre distance / average width for consecutive wells.  [K−1]."""
        mu = self.centers()
        sigma = self.widths()
        return (mu[1:] - mu[:-1]) / (0.5 * (sigma[:-1] + sigma[1:]))

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        y: torch.Tensor,
        subject_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map scalar evidence to a posterior over integers.

        Parameters
        ----------
        y           : [B]  scalar evidence from visual encoder
        subject_idx : [B]  integer subject indices

        Returns
        -------
        dict with ``logits`` [B,K], ``probs`` [B,K],
        ``expected_n`` [B], ``entropy`` [B].
        """
        mu = self.centers()                                            # [K]
        sigma = self.widths()                                          # [K]
        width_s = self.subject_params.width_scale(subject_idx)         # [B]

        sigma_i = sigma.unsqueeze(0) * width_s.unsqueeze(1)           # [B, K]
        z = (y.unsqueeze(1) - mu.unsqueeze(0)) / sigma_i              # [B, K]

        logits = -0.5 * z.pow(2) - sigma_i.log()                      # [B, K]

        probs = F.softmax(logits, dim=-1)
        expected_n = (probs * self.integers.unsqueeze(0)).sum(dim=-1)

        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "expected_n": expected_n,
            "entropy": entropy,
        }

    # -- diagnostics -------------------------------------------------------

    @torch.no_grad()
    def estimate_kstar(
        self,
        subject_idx: torch.Tensor,
        q: float = 0.75,
    ) -> torch.Tensor:
        """Useful-discreteness threshold per subject.

        K*_i = max n such that P_i(n | y = mu_n) >= q
        """
        mu = self.centers()
        K = mu.shape[0]
        S = subject_idx.shape[0]
        device = mu.device

        y_rep = mu.unsqueeze(0).expand(S, K).reshape(S * K)
        idx_rep = subject_idx.unsqueeze(1).expand(S, K).reshape(S * K)

        probs = self.forward(y_rep, idx_rep)["probs"].view(S, K, K)

        diag = torch.arange(K, device=device)
        correct_prob = probs[:, diag, diag]                            # [S, K]

        above = correct_prob >= q
        k_range = torch.arange(K, device=device).unsqueeze(0).expand(S, K)
        masked = torch.where(above, k_range, torch.full_like(k_range, -1))
        best_idx = masked.max(dim=1).values                            # [S]

        return torch.where(
            best_idx >= 0,
            self.integers[best_idx.clamp(min=0)],
            self.integers[0] - 1,
        )
