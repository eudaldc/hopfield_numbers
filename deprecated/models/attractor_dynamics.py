"""Dynamic attractor cleanup core: noisy evidence → stable integer attractor.

Energy landscape:
    E(x | y₀) = (x - y₀)² / (2 τ_in²)
                - κ · logsumexp_n( -(x - μ_n)² / (2 σ_n²) )

Cleanup via gradient descent on E:
    x_{t+1} = x_t - η · dE/dx

The analytical gradient is:
    dE/dx = (x - y₀) / τ_in²
            + κ · Σ_n[ softmax_n(f) · (x - μ_n) / σ_n² ]

where f_n = -(x - μ_n)² / (2 σ_n²).

Setting cleanup_mode="none" recovers the static readout (x* = y₀).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.subject_params import inverse_softplus


def _boxcox(z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Box-Cox transform, continuous through rho = 0.

    boxcox(z, rho) = (z^rho - 1) / rho   for rho != 0
                   = log(z)               for rho = 0
    """
    log_z = torch.log(z)
    return torch.where(
        rho.abs() > 1e-6,
        torch.expm1(rho * log_z) / rho,
        log_z,
    )


_CONVERGE_CHECK_INTERVAL = 4


def _cleanup_loop(
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma_sq: torch.Tensor,
    inv_tau_in_sq: torch.Tensor,
    kappa: torch.Tensor,
    eta: torch.Tensor,
    max_steps: int,
    tol: float,
) -> torch.Tensor:
    """Gradient-descent loop with lightweight convergence tracking.

    All geometry constants are pre-computed by the caller.  Converged
    samples are frozen via ``torch.where`` (no divergence), and the
    GPU→CPU sync for early exit runs only every few steps.
    """
    converged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for t in range(max_steps):
        diff = x.unsqueeze(-1) - mu                             # [B, K]
        f_n = diff.square() / (-2.0 * sigma_sq)                 # [B, K]
        weights = F.softmax(f_n, dim=-1)                         # [B, K]
        pull = kappa * (weights * diff / sigma_sq).sum(dim=-1)   # [B]
        anchor = (x - y) * inv_tau_in_sq                         # [B]
        x_new = x - eta * (anchor + pull)

        delta = (x_new - x).abs()
        converged = converged | (delta < tol)
        x = torch.where(converged, x, x_new)

        if (t + 1) % _CONVERGE_CHECK_INTERVAL == 0 and converged.all():
            break
    return x


class DynamicAttractorCore(nn.Module):
    """Iterative gradient-descent cleanup of noisy scalar evidence.

    Parameters
    ----------
    support_min, support_max : inclusive integer range
    cleanup_mode             : "gradient" (iterative) | "none" (static)
    max_steps                : maximum cleanup iterations
    tol                      : per-sample convergence tolerance on |dx|
    """

    def __init__(
        self,
        support_min: int = 1,
        support_max: int = 9,
        rho_init: float = 0.1,
        lambda_init: float = 0.5,
        sigma0_init: float = 1.0,
        nu_init: float = 1.0,
        tau_in_init: float = 1.0,
        kappa_init: float = 1.0,
        eta_init: float = 0.2,
        cleanup_mode: str = "gradient",
        max_steps: int = 16,
        tol: float = 1e-3,
    ):
        super().__init__()
        self.support_min = support_min
        self.support_max = support_max
        self.cleanup_mode = cleanup_mode
        self.max_steps = max_steps
        self.tol = tol

        self.register_buffer(
            "integers",
            torch.arange(support_min, support_max + 1, dtype=torch.float32),
        )

        self.rho = nn.Parameter(torch.tensor(rho_init))
        self._nu_raw = nn.Parameter(torch.tensor(inverse_softplus(nu_init)))
        self._sigma0_raw = nn.Parameter(torch.tensor(inverse_softplus(sigma0_init)))
        self.lam = nn.Parameter(torch.tensor(lambda_init))
        self._tau_in_raw = nn.Parameter(torch.tensor(inverse_softplus(tau_in_init)))
        self._kappa_raw = nn.Parameter(torch.tensor(inverse_softplus(kappa_init)))
        self._eta_raw = nn.Parameter(torch.tensor(inverse_softplus(eta_init)))

    # ── constrained properties ─────────────────────────────────────────────

    @property
    def nu(self) -> torch.Tensor:
        return F.softplus(self._nu_raw) + 1e-6

    @property
    def sigma0(self) -> torch.Tensor:
        return F.softplus(self._sigma0_raw) + 1e-6

    @property
    def tau_in(self) -> torch.Tensor:
        return F.softplus(self._tau_in_raw) + 1e-6

    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self._kappa_raw) + 1e-6

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self._eta_raw) + 1e-6

    # ── geometry ───────────────────────────────────────────────────────────

    def compute_mu_sigma(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Well centres and widths.  Both shape [K]."""
        z = self.integers + self.nu
        mu = _boxcox(z, self.rho)
        sigma = self.sigma0 * z.pow(self.lam)
        return mu, sigma

    def _snapshot_params(self) -> tuple:
        """Compute and cache all constrained parameters once per forward."""
        mu, sigma = self.compute_mu_sigma()
        sigma_sq = sigma.square()
        inv_tau_in_sq = self.tau_in.square().reciprocal()
        return mu, sigma_sq, inv_tau_in_sq, self.kappa, self.eta

    # ── energy (used only by diagnostics / slow path) ──────────────────────

    def energy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scalar energy E(x | y).  Shape [...] → [...]."""
        mu, sigma = self.compute_mu_sigma()
        diff = x.unsqueeze(-1) - mu
        log_unnorm = -(diff ** 2) / (2 * sigma ** 2)
        attraction = self.kappa * torch.logsumexp(log_unnorm, dim=-1)
        anchoring = (x - y) ** 2 / (2 * self.tau_in ** 2)
        return anchoring - attraction

    # ── posterior ──────────────────────────────────────────────────────────

    def posterior_from_x(
        self,
        x: torch.Tensor,
        mu: torch.Tensor | None = None,
        sigma_sq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Posterior p(n | x).  Shape [...] → [..., K].

        Accepts pre-computed *mu* and *sigma_sq* to avoid recomputation.
        """
        if mu is None or sigma_sq is None:
            mu_c, sigma_c = self.compute_mu_sigma()
            if mu is None:
                mu = mu_c
            if sigma_sq is None:
                sigma_sq = sigma_c.square()
        diff = x.unsqueeze(-1) - mu
        log_unnorm = -(diff ** 2) / (2.0 * sigma_sq)
        return F.softmax(log_unnorm, dim=-1)

    # ── cleanup ────────────────────────────────────────────────────────────

    def cleanup(
        self,
        y: torch.Tensor,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Iterate gradient descent on E(x | y₀) from x0 = y₀.

        Parameters
        ----------
        y           : [B] scalar sensory evidence (y₀)
        return_traj : if True, include x_traj and energy_trace in output

        Returns
        -------
        dict with keys: y0, x0, x_star, n_steps, converged, probs_clean,
        n_hat, expected_n_clean, basin_margin (+ optional x_traj, energy_trace)
        """
        B = y.shape[0]
        device = y.device

        if self.cleanup_mode == "none":
            mu, sigma_sq, *_ = self._snapshot_params()
            x_star = y
            probs_clean = self.posterior_from_x(x_star, mu, sigma_sq)
            n_hat_idx = probs_clean.argmax(dim=-1)
            top2 = probs_clean.topk(2, dim=-1).values
            out = {
                "y0": y,
                "x0": y,
                "x_star": x_star,
                "n_steps": torch.zeros(B, dtype=torch.long, device=device),
                "converged": torch.ones(B, dtype=torch.bool, device=device),
                "probs_clean": probs_clean,
                "n_hat": self.integers[n_hat_idx],
                "expected_n_clean": (probs_clean * self.integers).sum(dim=-1),
                "basin_margin": top2[:, 0] - top2[:, 1],
            }
            if return_traj:
                out["x_traj"] = y.unsqueeze(-1)
                out["energy_trace"] = self.energy(y, y).unsqueeze(-1)
            return out

        mu, sigma_sq, inv_tau_in_sq, kappa, eta = self._snapshot_params()

        if not return_traj:
            # ── fast path: lightweight convergence, ≤4 GPU→CPU syncs ──
            x = _cleanup_loop(
                y.clone(), y, mu, sigma_sq, inv_tau_in_sq,
                kappa, eta, self.max_steps, self.tol,
            )
            n_steps = torch.full(
                (B,), self.max_steps, dtype=torch.long, device=device,
            )
            converged = torch.ones(B, dtype=torch.bool, device=device)
        else:
            # ── slow path: full tracking for diagnostics ──
            x = y.clone()
            x0 = y.detach().clone()
            n_steps = torch.full(
                (B,), self.max_steps, dtype=torch.long, device=device,
            )
            converged = torch.zeros(B, dtype=torch.bool, device=device)
            traj: list[torch.Tensor] = [x0]
            energy_trace: list[torch.Tensor] = [
                self.energy(x.detach(), y.detach())
            ]

            for t in range(self.max_steps):
                diff = x.unsqueeze(-1) - mu
                f_n = diff.square() / (-2.0 * sigma_sq)
                weights = F.softmax(f_n, dim=-1)
                pull = kappa * (weights * diff / sigma_sq).sum(dim=-1)
                anchor = (x - y) * inv_tau_in_sq
                x_new = x - eta * (anchor + pull)

                with torch.no_grad():
                    delta = (x_new - x).abs()
                    just_converged = (~converged) & (delta < self.tol)
                    n_steps = torch.where(
                        just_converged,
                        torch.tensor(t + 1, dtype=torch.long, device=device),
                        n_steps,
                    )
                    converged = converged | just_converged

                x = x_new
                traj.append(x.detach())
                energy_trace.append(self.energy(x.detach(), y.detach()))

                if converged.all():
                    break

        probs_clean = self.posterior_from_x(x, mu, sigma_sq)
        n_hat_idx = probs_clean.argmax(dim=-1)
        top2 = probs_clean.topk(2, dim=-1).values

        out: dict[str, torch.Tensor] = {
            "y0": y,
            "x0": y.detach() if not return_traj else x0,
            "x_star": x,
            "n_steps": n_steps,
            "converged": converged,
            "probs_clean": probs_clean,
            "n_hat": self.integers[n_hat_idx],
            "expected_n_clean": (probs_clean * self.integers).sum(dim=-1),
            "basin_margin": top2[:, 0] - top2[:, 1],
        }
        if return_traj:
            out["x_traj"] = torch.stack(traj, dim=-1)
            out["energy_trace"] = torch.stack(energy_trace, dim=-1)
        return out

    def forward(
        self,
        y: torch.Tensor,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.cleanup(y, return_traj=return_traj)

    # ── diagnostics ────────────────────────────────────────────────────────

    @torch.no_grad()
    def separation_ratio(self) -> torch.Tensor:
        """Centre gap / average width for consecutive wells.  Shape [K-1]."""
        mu, sigma = self.compute_mu_sigma()
        return (mu[1:] - mu[:-1]) / (0.5 * (sigma[:-1] + sigma[1:]))

    @torch.no_grad()
    def k_star_geometry(self, q: float = 0.75) -> torch.Tensor:
        """Largest integer whose self-posterior at its centre exceeds q.

        Evaluated using the static posterior (posterior_from_x at mu_n).
        Returns integers[0] - 1 when no integer reaches the threshold.
        """
        mu, _ = self.compute_mu_sigma()
        K = mu.shape[0]
        probs = self.posterior_from_x(mu)          # [K, K]
        diag_idx = torch.arange(K, device=mu.device)
        diag_probs = probs[diag_idx, diag_idx]     # [K]
        above = diag_probs >= q
        masked = torch.where(
            above,
            diag_idx,
            torch.tensor(-1, device=mu.device).expand(K),
        )
        best_idx = masked.max()
        if best_idx < 0:
            return self.integers[0] - 1
        return self.integers[best_idx]
