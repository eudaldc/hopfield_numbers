"""Reaction-time proxy from cleanup dynamics (WP-3).

Two modes
---------
"steps"
    rt = t0 + c_steps * n_steps

    Interpretation: RT grows linearly with the number of cleanup iterations.

"steps_plus_margin"
    rt = t0 + c_steps * n_steps + c_margin / (basin_margin + eps)

    Interpretation: RT is additionally slowed when the attractor landscape is
    ambiguous (small probability margin between top two integer wells).

All free parameters (t0, c_steps, c_margin) are constrained positive via
softplus, making the proxy monotone by construction.

Integration rules (v0.2)
------------------------
- Used only for synthetic benchmarks and simulation plots.
- Not yet wired into empirical task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.subject_params import inverse_softplus


class RTProxy(nn.Module):
    """Scalar reaction-time proxy from cleanup convergence statistics.

    Parameters
    ----------
    mode          : "steps" | "steps_plus_margin"
    t0_init       : baseline RT (positive)
    c_steps_init  : per-step cost (positive)
    c_margin_init : ambiguity penalty scale (positive; only for steps_plus_margin)
    eps           : numerical floor for basin_margin denominator
    """

    def __init__(
        self,
        mode: str = "steps",
        t0_init: float = 1.0,
        c_steps_init: float = 1.0,
        c_margin_init: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in {"steps", "steps_plus_margin"}:
            raise ValueError(
                f"Unknown RT proxy mode {mode!r}. "
                "Choose from: 'steps', 'steps_plus_margin'."
            )
        self.mode = mode
        self.eps = eps

        self._t0_raw = nn.Parameter(torch.tensor(inverse_softplus(t0_init)))
        self._c_steps_raw = nn.Parameter(torch.tensor(inverse_softplus(c_steps_init)))

        if mode == "steps_plus_margin":
            self._c_margin_raw = nn.Parameter(
                torch.tensor(inverse_softplus(c_margin_init))
            )
        else:
            self.register_buffer(
                "_c_margin_raw", torch.zeros(1)
            )

    # ── constrained properties ─────────────────────────────────────────────

    @property
    def t0(self) -> torch.Tensor:
        return F.softplus(self._t0_raw) + self.eps

    @property
    def c_steps(self) -> torch.Tensor:
        return F.softplus(self._c_steps_raw) + self.eps

    @property
    def c_margin(self) -> torch.Tensor:
        return F.softplus(self._c_margin_raw) + self.eps

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        n_steps: torch.Tensor,
        basin_margin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the RT proxy.

        Parameters
        ----------
        n_steps      : [B] integer (or float) number of cleanup iterations
        basin_margin : [B] top1 − top2 posterior probability; required for
                       "steps_plus_margin" mode, ignored otherwise

        Returns
        -------
        rt : [B] positive reaction-time proxy
        """
        rt = self.t0 + self.c_steps * n_steps.float()
        if self.mode == "steps_plus_margin":
            if basin_margin is None:
                raise ValueError(
                    "basin_margin is required for mode='steps_plus_margin'."
                )
            rt = rt + self.c_margin / (basin_margin + self.eps)
        return rt
