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

WIN = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
OUT_FIG = T.OUT_DIR / "pexact_decomposition.png"


def main() -> None:
    th = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    quad = T.make_quadrature(1601, kind="grid")
    n = T.N_SHOWN.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, d in enumerate((80, 640)):
        theta = {p: float(th.loc[d, p]) for p in WIN.param_names}
        sigma_well = np.power(theta["sigma_c"] / T.FREQ, theta["beta"])   # per well 1..19
        sd_sens = T.evidence_sd(n, theta["s0"], theta["lambda0"])         # per n
        g_on = T.compute_grid(WIN, theta, quad)
        g_off = T.compute_grid(WIN, theta, quad, force_no_cleanup=True)
        pexact_on = np.array([g_on["p_resp"][i, int(nn) - 1] for i, nn in enumerate(n)]) * 100
        pexact_off = np.array([g_off["p_resp"][i, int(nn) - 1] for i, nn in enumerate(n)]) * 100

        ax = axes[0][col]
        ax.plot(T.WELLS, sigma_well, "o-", color="#2c7fb8", ms=3,
                label=r"well width $\sigma_n$ (smooth)")
        ax.plot(n, sd_sens, "s-", color="#7f8c8d", ms=3, label="sensory sd (smooth)")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_title(f"{d} ms — the 4-param landscape is SMOOTH")
        ax.set_xlabel("n / well"); ax.set_ylabel("width")
        ax.legend(fontsize=8, frameon=False)

        ax = axes[1][col]
        ax.plot(n, pexact_on, "o-", color="#c0392b", ms=4, label="P(exact), cleanup ON")
        ax.plot(n, pexact_off, "s--", color="#e08e0b", ms=3, label="P(exact), cleanup OFF")
        ax.set_title(f"{d} ms — the READOUT on the integer lattice is bumpy")
        ax.set_xlabel("n shown"); ax.set_ylabel("% exact")
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
