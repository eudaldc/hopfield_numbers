"""Subject-level learned embeddings for the unified attractor model.

Each subject has four parameters accessed via integer indices:
  - temperature (posterior sharpness)
  - width scale (well-width multiplier)
  - symbolic access noise (digit → distribution spread)
  - counting tendency bias (dot-verification mixture intercept)

All embeddings are regularised toward zero with a configurable L2 penalty.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import SubjectConfig


def inverse_softplus(x: float) -> float:
    """Compute y such that softplus(y) ≈ x."""
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


class SubjectParams(nn.Module):
    """Learned per-subject parameter bank.

    Parameters
    ----------
    cfg : SubjectConfig
        Contains ``n_subjects``, initial values, and L2 penalty weight.
    """

    def __init__(self, cfg: SubjectConfig):
        super().__init__()
        n = cfg.n_subjects
        self._eps = 1e-6
        self.l2_weight = cfg.l2_penalty

        self.temp_raw = nn.Embedding(n, 1)
        nn.init.constant_(
            self.temp_raw.weight, inverse_softplus(cfg.temp_init_mean)
        )

        self.width_raw = nn.Embedding(n, 1)
        nn.init.normal_(self.width_raw.weight, 0.0, cfg.width_init_std)

        self.tau_sym_raw = nn.Embedding(n, 1)
        nn.init.constant_(
            self.tau_sym_raw.weight, inverse_softplus(cfg.tau_sym_init)
        )

        self.count_bias_raw = nn.Embedding(n, 1)
        nn.init.constant_(self.count_bias_raw.weight, 0.0)

    def temperature(self, idx: torch.Tensor) -> torch.Tensor:
        """[B] → [B] positive temperature."""
        return F.softplus(self.temp_raw(idx).squeeze(-1)) + self._eps

    def width_scale(self, idx: torch.Tensor) -> torch.Tensor:
        """[B] → [B] positive width multiplier (≈ 1.0 at init)."""
        return torch.exp(self.width_raw(idx).squeeze(-1))

    def tau_sym(self, idx: torch.Tensor) -> torch.Tensor:
        """[B] → [B] positive symbolic access noise."""
        return F.softplus(self.tau_sym_raw(idx).squeeze(-1)) + self._eps

    def count_bias(self, idx: torch.Tensor) -> torch.Tensor:
        """[B] → [B] logit-scale counting tendency intercept."""
        return self.count_bias_raw(idx).squeeze(-1)

    def l2_penalty(self) -> torch.Tensor:
        """Weighted sum of squared embedding weights."""
        total = sum(
            emb.weight.pow(2).sum()
            for emb in [
                self.temp_raw,
                self.width_raw,
                self.tau_sym_raw,
                self.count_bias_raw,
            ]
        )
        return self.l2_weight * total
