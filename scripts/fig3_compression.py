#!/usr/bin/env python
"""Generate Figure 3: functional compression from Poisson attractor wells."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.special import gammaln

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plots import STYLE

OUTPUT_DIR = _ROOT / "output" / "piantadosi_fit"
FONT_SCALE = 1.4
C = 0.042
SUPPORT = np.arange(1, 19)
DISPLAY_MAX = 15
ALL_NS = list(range(1, DISPLAY_MAX + 1))


def _scaled_style() -> dict:
    """Matplotlib rc equivalent to ``sns.set_theme(font_scale=FONT_SCALE)``."""
    p = dict(STYLE)
    keys = (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    )
    for key in keys:
        base = p.get(key)
        if base is None:
            base = mpl.rcParamsDefault[key]
        if isinstance(base, (int, float)):
            p[key] = base * FONT_SCALE
    return p


def _poisson_well_logits(x_star: float, support: np.ndarray, c: float) -> np.ndarray:
    x_s = x_star / c
    k_s = support / c
    return x_s * np.log(k_s) - k_s - gammaln(x_s + 1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def _draw_compression(ax: plt.Axes) -> None:
    integers_display = np.arange(1, DISPLAY_MAX + 1, dtype=float)
    peak_prob = 1.0 / integers_display
    peak_prob_norm = peak_prob / peak_prob.max()
    cmap_wells = mpl.colormaps["plasma"]
    norm_c = Normalize(vmin=-0.10, vmax=1.15)

    def well_color(i: int) -> tuple:
        return cmap_wells(norm_c(peak_prob_norm[i]))

    colors = [well_color(i) for i in range(len(ALL_NS))]
    mask = SUPPORT <= DISPLAY_MAX

    for i, n_target in enumerate(ALL_NS):
        logits = _poisson_well_logits(n_target, SUPPORT, C)
        probs = _softmax(logits)
        ax.fill_between(SUPPORT[mask], 0, probs[mask], color=colors[i], alpha=0.25)
        ax.plot(
            SUPPORT[mask],
            probs[mask],
            "o-",
            color=colors[i],
            ms=3,
            lw=1.2,
            label=f"$n = {n_target}$",
            markeredgecolor="white",
            markeredgewidth=0.2,
        )

    ax.set_xlabel("Integer $n$")
    ax.set_ylabel("$p(k \\mid x^* = \\mu_n)$")
    ax.set_xticks(range(1, DISPLAY_MAX + 1))
    ax.set_xlim(0.5, DISPLAY_MAX + 0.5)


def _save_fig(fig: plt.Figure, base_name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{base_name}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(_scaled_style())
    print("Drawing fig3_compression ...", flush=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    _draw_compression(ax)
    _save_fig(fig, "fig3_compression")
    plt.close(fig)


if __name__ == "__main__":
    main()
