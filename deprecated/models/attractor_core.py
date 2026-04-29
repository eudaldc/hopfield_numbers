"""Attractor core: continuous scalar evidence → posterior over integer wells.

Static posterior (no cleanup dynamics):

    mu_n       = boxcox(n + nu, rho)             — well centres
    sigma_n    = sigma0 · (n + nu)^lambda         — well widths
    sigma_n_i  = sigma_n · exp(width_subject_i)   — per-subject widths
    temp_i     = softplus(temp_subject_i) + eps    — per-subject temperature
    logit_n    = (-0.5·((y₀ - mu_n) / sigma_n_i)² - log(sigma_n_i)) / temp_i
    p_i(n | y₀) = softmax(logit_n)

The -log(sigma) normalisation term ensures that wider wells (large n)
are penalised for spreading probability mass (Gaussian log-likelihood).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import AttractorConfig
from src.models.subject_params import SubjectParams, inverse_softplus


class AttractorCore(nn.Module):
    """Continuous scalar evidence → posterior over integer support.

    Parameters
    ----------
    cfg : AttractorConfig
    subject_params : SubjectParams
    """

    def __init__(self, cfg: AttractorConfig, subject_params: SubjectParams):
        super().__init__()
        self.support_cfg = cfg.support
        K = cfg.support.size
        n_min = cfg.support.n_min

        self.register_buffer(
            "integers",
            torch.arange(n_min, n_min + K, dtype=torch.float32),
        )

        self.rho = nn.Parameter(torch.tensor(cfg.rho_init))
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
    def nu(self) -> torch.Tensor:
        """Positive offset (ensures n + nu > 0 for all n in support)."""
        return F.softplus(self._nu_raw) + 1e-6

    @property
    def sigma0(self) -> torch.Tensor:
        """Positive base width."""
        return F.softplus(self._sigma0_raw) + 1e-6

    # -- Box-Cox -----------------------------------------------------------

    @staticmethod
    def _boxcox(z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Box-Cox transform, continuous through rho = 0.

        boxcox(z, rho) = (z^rho - 1) / rho   (rho ≠ 0)
                       = log(z)                (rho = 0)
        """
        log_z = torch.log(z)
        return torch.where(
            rho.abs() > 1e-6,
            torch.expm1(rho * log_z) / rho,
            log_z,
        )

    # -- geometry ----------------------------------------------------------

    def centers(self) -> torch.Tensor:
        """Well centres mu_n.  Shape [K]."""
        return self._boxcox(self.integers + self.nu, self.rho)

    def widths(self) -> torch.Tensor:
        """Population-level well widths sigma_n.  Shape [K]."""
        return self.sigma0 * (self.integers + self.nu).pow(self.lam)

    def pairwise_separation(self) -> torch.Tensor:
        """Centre distance / average width for consecutive wells.  [K-1].

        Higher values mean better-separated neighbouring wells.
        """
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
        y           : [B]  scalar evidence
        subject_idx : [B]  integer subject indices

        Returns
        -------
        dict with ``logits`` [B,K], ``probs`` [B,K],
        ``expected_n`` [B], ``entropy`` [B].
        """
        mu = self.centers()
        sigma = self.widths()
        width_s = self.subject_params.width_scale(subject_idx)
        temp = self.subject_params.temperature(subject_idx)

        sigma_i = sigma.unsqueeze(0) * width_s.unsqueeze(1)       # [B, K]
        z = (y.unsqueeze(1) - mu.unsqueeze(0)) / sigma_i          # [B, K]
        logits = (-0.5 * z.pow(2) - sigma_i.log()) / temp.unsqueeze(1)  # [B, K]

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

        Parameters
        ----------
        subject_idx : [S]
        q           : probability threshold

        Returns
        -------
        kstar : [S]  float tensor with integer values; n_min − 1 when
                      no integer reaches the threshold.
        """
        mu = self.centers()
        K = mu.shape[0]
        S = subject_idx.shape[0]
        device = mu.device

        y_rep = mu.unsqueeze(0).expand(S, K).reshape(S * K)
        idx_rep = subject_idx.unsqueeze(1).expand(S, K).reshape(S * K)

        probs = self.forward(y_rep, idx_rep)["probs"].view(S, K, K)

        diag = torch.arange(K, device=device)
        correct_prob = probs[:, diag, diag]                        # [S, K]

        above = correct_prob >= q
        k_range = torch.arange(K, device=device).unsqueeze(0).expand(S, K)
        masked = torch.where(above, k_range, torch.full_like(k_range, -1))
        best_idx = masked.max(dim=1).values                        # [S]

        return torch.where(
            best_idx >= 0,
            self.integers[best_idx.clamp(min=0)],
            self.integers[0] - 1,
        )
