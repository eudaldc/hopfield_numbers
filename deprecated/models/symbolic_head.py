"""Symbolic head: digit identity → distribution over integer support.

Uses the attractor core's well centres and per-subject symbolic access
noise to produce a soft distribution:

    q_i(n | d) ∝ exp( −0.5 · ((mu_n − mu_d) / tau_sym_i)² )

Contract (from WORK_PACKAGES.md):
    out = symbolic_head(digits, subject_idx)
    # out["digit_probs"]: [B, K]
    # out["expected_n"] : [B]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attractor_core import AttractorCore


class SymbolicHead(nn.Module):
    """Maps a presented digit to a distribution over the integer support."""

    def __init__(self, attractor_core: AttractorCore):
        super().__init__()
        self.attractor = attractor_core

    def forward(
        self,
        digits: torch.Tensor,
        subject_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        digits      : [B]  integer digit values (within support range)
        subject_idx : [B]

        Returns
        -------
        dict with ``digit_probs`` [B, K] and ``expected_n`` [B]
        """
        mu = self.attractor.centers()                              # [K]
        tau = self.attractor.subject_params.tau_sym(subject_idx)   # [B]

        n_min = self.attractor.support_cfg.n_min
        digit_idx = (digits - n_min).long()
        mu_d = mu[digit_idx]                                       # [B]

        diff = mu.unsqueeze(0) - mu_d.unsqueeze(1)                # [B, K]
        logits = -0.5 * (diff / tau.unsqueeze(1)).pow(2)          # [B, K]

        probs = F.softmax(logits, dim=-1)
        expected_n = (probs * self.attractor.integers.unsqueeze(0)).sum(-1)

        return {"digit_probs": probs, "expected_n": expected_n}
