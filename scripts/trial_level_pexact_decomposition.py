"""The exact-accuracy shoulder is a clean-up signature.

The predicted proportion of exact responses P(r = n) shows a small shoulder near
the continuous width proxy n_width, visible in the main-text co-fit figure. The landscape itself is
smooth by construction (the width law and the sensory-noise profile are smooth,
monotone functions of n). Toggling clean-up off at identical parameters removes
the shoulder, showing that it comes from the dynamics on the integer lattice
rather than extra fitted degrees of freedom. The alignment with n_width does not
identify a discrete merger.

Outputs `figures/supfigure3_exact_accuracy_decomposition.png`/`.pdf`
(one panel per duration: 80 and 640 ms).

    .venv/bin/python -m scripts.trial_level_pexact_decomposition
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from scripts.trial_level_weber_fraction import n_width
from scripts.plot_style import C_ATTR, C_STATIC, apply_style, panel_label

WIN = T.MODEL_BY_KEY[T.WINNER_KEY]
OUT_FIG = _ROOT / "figures" / "supfigure3_exact_accuracy_decomposition.png"


def main() -> None:
    apply_style(2.0)   # rendered at \columnwidth, 12 in canvas
    th = T.load_shared_theta()
    quad = T.make_quadrature(1601, kind="grid")
    n = T.N_SHOWN.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, letter, d in zip(axes, "ab", (80, 640)):
        theta = {p: float(th.loc[d, p]) for p in WIN.param_names}
        g_on = T.compute_grid(WIN, theta, quad)
        g_off = T.compute_grid(WIN, theta, quad, force_no_cleanup=True)
        pexact_on = np.array([g_on["p_resp"][i, int(nn) - 1] for i, nn in enumerate(n)]) * 100
        pexact_off = np.array([g_off["p_resp"][i, int(nn) - 1] for i, nn in enumerate(n)]) * 100

        ax.plot(n, pexact_on, "o-", color=C_ATTR, ms=5, label="Clean-up on")
        ax.plot(n, pexact_off, "s--", color=C_STATIC, ms=4, label="Clean-up off")
        ax.axvline(n_width(theta), color="0.25", ls=":", lw=1.0)
        ax.set_xlabel("Number shown  $n$")
        ax.text(0.97, 0.55, f"{d} ms", transform=ax.transAxes, ha="right",
                va="bottom", color="0.4")
        panel_label(ax, letter)
    axes[0].set_ylabel("% exact")
    axes[0].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {OUT_FIG} and .pdf")


if __name__ == "__main__":
    main()
