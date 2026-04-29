#!/usr/bin/env python
"""Generate Figure 3: (a) posterior compression, (b) Reynvoet / Weber.

Reynvoet model points use :func:`scripts.eval_reynvoet.simulate_reynvoet_2afc` with
parameters from ``output/piantadosi_fit/summary_all_durations.csv``. Input
anchoring follows ``no_anchor`` in ``configs/config.yaml`` (when true,
``tau_in`` is not read and cleanup uses ``inv_tau_sq = 0``, as in
``fit_helpers``).

Panel (b) shows the Reynvoet model at 640 ms only (solid red line, same
visual language as :mod:`figures.fig2_piantadosi_fit`).

Saves PDF and PNG to ``figures/fig3_compression_reynvoet``."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.special import gammaln

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_reynvoet import (
    load_reynvoet_panamath,
    simulate_reynvoet_2afc,
    fit_weber_fraction,
    _load_fitted_params as reynvoet_load_params,
    CONGRUENCY_PREFIXES,
    RATIO_SUFFIXES,
    DATA_PATH,
)
from scripts.plots import (
    STYLE,
    EMP_COLOR,
)

OUTPUT_DIR = Path("figures")
FONT_SCALE = 1.4
REYNVOET_MODEL_DUR_MS = 640
MOD_COLOR = "#cb181d"


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

C = 0.042
SUPPORT = np.arange(1, 19)
DISPLAY_MAX = 15
ALL_NS = list(range(1, DISPLAY_MAX + 1))


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
            SUPPORT[mask], probs[mask], "o-",
            color=colors[i], ms=3, lw=1.2,
            label=f"$n = {n_target}$",
            markeredgecolor="white", markeredgewidth=0.2,
        )

    ax.set_xlabel("Integer $n$")
    ax.set_ylabel("$p(k \\mid x^* = \\mu_n)$")
    ax.set_xticks(range(1, DISPLAY_MAX + 1))
    ax.set_xlim(0.5, DISPLAY_MAX + 0.5)
    ax.set_title("a", fontweight="bold", loc="left", x=0.075)


def _load_per_subject_by_ratio() -> dict[float, np.ndarray]:
    df = pd.read_csv(DATA_PATH, decimal=",", na_values=["#NULL!"])
    out: dict[float, np.ndarray] = {}
    for ratio_str in RATIO_SUFFIXES:
        cols = [
            f"{prefix}{ratio_str}"
            for prefix in CONGRUENCY_PREFIXES.values()
            if f"{prefix}{ratio_str}" in df.columns
        ]
        if cols:
            per_subject = df[cols].mean(axis=1).dropna().values * 100
            out[float(ratio_str)] = per_subject
    return out


def _draw_weber(ax: plt.Axes) -> None:
    empirical = load_reynvoet_panamath()
    dur = REYNVOET_MODEL_DUR_MS
    theta = reynvoet_load_params(dur)
    model_pred = simulate_reynvoet_2afc(theta)

    subj_data = _load_per_subject_by_ratio()
    ratios_sorted = sorted(subj_data.keys())
    bp_data = [subj_data[r] for r in ratios_sorted]

    ax.boxplot(
        bp_data, positions=ratios_sorted, widths=0.04,
        patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=EMP_COLOR, edgecolor=EMP_COLOR, alpha=0.25, lw=0.8),
        medianprops=dict(color=EMP_COLOR, lw=1.5),
        whiskerprops=dict(color=EMP_COLOR, lw=0.8),
        capprops=dict(color=EMP_COLOR, lw=0.8),
        flierprops=dict(marker=".", ms=3, color=EMP_COLOR, alpha=0.5),
    )

    pred_sorted = model_pred.sort_values("ratio")
    ax.plot(
        pred_sorted["ratio"],
        pred_sorted["accuracy"],
        "-",
        color=MOD_COLOR,
        lw=2.5,
        zorder=10,
    )

    avg = empirical[empirical["congruency"] == "average"].sort_values("ratio")
    w_emp = fit_weber_fraction(avg["ratio"].values, avg["mean_acc"].values)
    w_m = fit_weber_fraction(
        model_pred["ratio"].to_numpy(), model_pred["accuracy"].to_numpy()
    )
    note = (
        f"$w_{{\\mathrm{{emp}}}}$ = {w_emp:.3f}\n"
        f"$w_{{\\mathrm{{model}}}}$ = {w_m:.3f}"
    )

    ax.text(
        0.97, 0.48,
        note,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=mpl.rcParams.get("font.size", 10) * 0.92,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
    )

    leg_handles: list = [
        plt.Line2D([], [], color=MOD_COLOR, lw=2.5, ls="-"),
    ]
    leg_labels = [f"Model ({dur} ms)"]
    leg_handles.append(
        Patch(facecolor=EMP_COLOR, edgecolor=EMP_COLOR, alpha=0.25, lw=0.8)
    )
    leg_labels.append("Empirical")
    ax.legend(leg_handles, leg_labels, loc="lower right")
    ax.set_xlabel("Ratio  $n_{large}\\,/\\,n_{small}$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(48, 105)
    ax.set_title("b", fontweight="bold", loc="left", x=0.075)


def _save_fig(fig: plt.Figure, base_name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{base_name}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(_scaled_style())
    print("Drawing fig3_compression_reynvoet ...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
    _draw_compression(axes[0])
    _draw_weber(axes[1])
    _save_fig(fig, "fig3_compression_reynvoet")
    plt.close(fig)


if __name__ == "__main__":
    main()
