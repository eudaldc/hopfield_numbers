"""Why does P(exact) wiggle when the landscape has only 4 smooth parameters?

Decomposition. The 4-parameter landscape is smooth: the well width sigma_n and the
sensory sd are smooth, monotone functions of n (top row). The wiggle/shoulder is in
the *observable* — P(r = n) is the soft-max read-out over the fixed integer well
lattice *after* the clean-up snaps evidence into discrete basins. Toggling the
clean-up off (same landscape) removes the bump (bottom row), proving it is the
discrete integer basins, not extra fitted degrees of freedom.

Outputs `pexact_decomposition.png` (columns: 80 ms and 640 ms).

    .venv/bin/python -m scripts.trial_level_pexact_decomposition
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from figures.plot_style import C_ATTR, C_BENCH, C_STATIC, apply_style, panel_label

WIN = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
OUT_FIG = T.OUT_DIR / "pexact_decomposition.png"


def main() -> None:
    apply_style(2.0)   # rendered at \columnwidth, 12 in canvas
    th = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    quad = T.make_quadrature(1601, kind="grid")
    n = T.N_SHOWN.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    letters = [["a", "c"], ["b", "d"]]
    for col, d in enumerate((80, 640)):
        theta = {p: float(th.loc[d, p]) for p in WIN.param_names}
        sigma_well = np.power(theta["sigma_c"] / T.FREQ, theta["beta"])   # per well 1..19
        sd_sens = T.evidence_sd(n, theta["s0"], theta["lambda0"])         # per n
        g_on = T.compute_grid(WIN, theta, quad)
        g_off = T.compute_grid(WIN, theta, quad, force_no_cleanup=True)
        pexact_on = np.array([g_on["p_resp"][i, int(nn) - 1] for i, nn in enumerate(n)]) * 100
        pexact_off = np.array([g_off["p_resp"][i, int(nn) - 1] for i, nn in enumerate(n)]) * 100

        ax = axes[0][col]
        ax.plot(T.WELLS, sigma_well, "o-", color=C_ATTR, ms=4,
                label=r"Well width $\sigma_n$")
        ax.plot(n, sd_sens, "s-", color=C_BENCH, ms=4, label="Sensory sd")
        ax.axhline(1.0, color="0.25", ls=":", lw=1.0)
        ax.set_ylabel("Width")
        ax.legend(frameon=False, loc="upper left", fontsize=20)
        ax.text(0.97, 0.06, f"{d} ms", transform=ax.transAxes, ha="right",
                va="bottom", color="0.4")
        panel_label(ax, letters[0][col])

        ax = axes[1][col]
        ax.plot(n, pexact_on, "o-", color=C_ATTR, ms=5, label="Clean-up on")
        ax.plot(n, pexact_off, "s--", color=C_STATIC, ms=4, label="Clean-up off")
        ax.set_xlabel("Number shown  $n$"); ax.set_ylabel("% exact")
        ax.legend(frameon=False, loc="upper right", fontsize=20)
        ax.text(0.97, 0.55, f"{d} ms", transform=ax.transAxes, ha="right",
                va="bottom", color="0.4")
        panel_label(ax, letters[1][col])

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {OUT_FIG} and .pdf")


if __name__ == "__main__":
    main()
