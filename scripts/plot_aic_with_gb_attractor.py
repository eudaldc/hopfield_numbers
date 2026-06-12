"""Regenerate the AIC-by-duration figure (Fig. 3) with the Gaussian-Benford
*attractor* added as a sixth model.

Colours encode the two-way structure of the model set: each accumulator family
(Poisson, Gaussian-Benford) gets one hue, with the attractor variant in the full
hue and its no-clean-up (static) counterpart in a paler shade of the same hue.
Weber/ANS and the C&P proxy get their own colours.

    .venv/bin/python -m scripts.plot_aic_with_gb_attractor
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.model_comparison import AIC_PLOT_STYLE, OUT_DIR

PER_DURATION = OUT_DIR / "per_duration_fit_metrics.csv"
GB_ATTRACTOR = OUT_DIR / "gaussian_benford_attractor_fit.csv"


def lighten(color: str, amount: float = 0.30) -> tuple[float, float, float]:
    """Raise lightness in HLS space, keeping the hue and saturation.

    Unlike blending toward white, this stays vivid instead of pastel, so the
    pale member of a pair still reads as the same colour family.
    """
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1.0, l + amount), s)


# Base hues sampled from magma (the palette used in the original figure); one
# per accumulator family, plus standalone colours for Weber/ANS and the proxy.
_magma = plt.colormaps["magma"]
C_POISSON = _magma(0.75)  # warm orange
C_BENFORD = _magma(0.55)  # magenta-red
C_WEBER = _magma(0.30)  # deep purple
C_CP = _magma(0.10)  # near-black purple

# (key in data, legend label, colour). Pairs are adjacent so the bars sit
# next to each other; the no-clean-up variant shares its family's hue, paler.
# Ordered Poisson pair, Gaussian-Benford pair, then the standalone models;
# hues are assigned so the bars follow the magma progression left to right
# (warm orange down through magenta and purple to near-black). Within each
# pair the attractor comes first in the full hue, then its no-clean-up
# counterpart in a paler shade.
MODELS: tuple[tuple[str, str, object], ...] = (
    ("hopfield_poisson", "Poisson attractor", C_POISSON),
    ("no_cleanup_poisson", "Poisson, no clean-up", lighten(C_POISSON, 0.18)),
    ("gb_attractor", "Gaussian--Benford attractor", C_BENFORD),
    ("gaussian_benford", "Gaussian--Benford, no clean-up", lighten(C_BENFORD, 0.20)),
    ("weber_ans", "Weber/ANS", C_WEBER),
    ("cp_log_prior_proxy", "C&P log-prior proxy", C_CP),
)


def load_aic_by_model() -> dict[str, pd.Series]:
    per = pd.read_csv(PER_DURATION)
    gb = pd.read_csv(GB_ATTRACTOR)
    durations = sorted(per["duration_ms"].unique())
    out: dict[str, pd.Series] = {}
    for key in ("hopfield_poisson", "no_cleanup_poisson", "gaussian_benford",
                "weber_ans", "cp_log_prior_proxy"):
        sub = per[per["model_key"] == key].set_index("duration_ms")["AIC"]
        out[key] = sub.reindex(durations)
    out["gb_attractor"] = gb.set_index("duration_ms")["AIC"].reindex(durations)
    return out


def main() -> None:
    plt.rcParams.update(AIC_PLOT_STYLE)
    aic = load_aic_by_model()
    durations = sorted(pd.read_csv(PER_DURATION)["duration_ms"].unique())
    group_labels = [f"{d} ms" for d in durations] + ["Global"]
    x = np.arange(len(group_labels))

    n_models = len(MODELS)
    width = min(0.13, 0.82 / n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(15.0, 7.8))
    for idx, (key, label, color) in enumerate(MODELS):
        y_dur = aic[key].to_numpy(float)
        y = np.concatenate([y_dur, [np.nanmean(y_dur)]])
        ax.bar(x + offsets[idx], y, width=width, color=color, label=label.replace("--", "–"))

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("Exposure duration")
    ax.set_ylabel("AIC (summary-curve profile likelihood)")
    ax.set_title("Model comparison across exposure durations", fontweight="bold")
    ax.axvline(len(durations) - 0.5, color="0.7", lw=1.0, ls=":")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"aic_by_duration.{ext}", dpi=300)
    plt.close(fig)

    means = {label: float(np.nanmean(aic[key].to_numpy())) for key, label, _ in MODELS}
    print("Mean AIC by model:")
    for label, v in sorted(means.items(), key=lambda kv: kv[1]):
        print(f"  {label:32s} {v:6.1f}")
    print(f"Saved {OUT_DIR / 'aic_by_duration.pdf'}")


if __name__ == "__main__":
    main()
