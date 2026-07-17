"""Effective Weber fraction predicted by the winning model.

The old (Poisson) draft claimed a sub-Weber prediction w_eff ~ 1/sqrt(n), because
Poisson wells have sigma_n = sqrt(c n) so the coefficient of variation of the readout
falls as 1/sqrt(n). That derivation does not hold for the winner, whose wells widen
super-linearly (sigma_n = (sigma_c/f_n)^beta, beta ~ 2). Here we compute what the
winner actually predicts: the coefficient of variation (CoV) of its fitted response
distribution, per numerosity and exposure, and compare it with the empirical CoV.

The model CoV is taken over the full fitted response distribution,
(1 - lapse) * p_model(r | n) + lapse * Uniform(1..r_max), restricted to the
in-support range r <= N_SUPPORT_MAX and renormalised, so it is the like-for-like
counterpart of the empirical dots, which are also in-support. This is the model
exactly as fitted (the lapse is one of its parameters, present in every model of
the comparison); we do not strip it out. A power-law fit w_eff ~ a * n^gamma over
the estimation range characterises the Weber plateau.

    .venv/bin/python -m scripts.trial_level_weber_fraction
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T
from figures.plot_style import apply_style

WIN = T.MODEL_BY_KEY[T.WINNER_KEY]
OUT_CSV = T.OUT_DIR / "weber_fraction.csv"
OUT_FIG = T.OUT_DIR / "weber_fraction.png"


def cov_over(support: np.ndarray, p_row: np.ndarray) -> float:
    """Coefficient of variation of a discrete distribution p over `support`."""
    p = p_row / p_row.sum()
    mean = float(np.sum(support * p))
    var = float(np.sum((support - mean) ** 2 * p))
    return float(np.sqrt(var) / mean)


def n_width(theta: dict) -> float:
    # power-law wells: sigma_n = sigma_c * n^p = 1  ->  n_width = sigma_c^(-1/p)
    return float(theta["sigma_c"] ** (-1.0 / theta["pow"]))


def empirical_cov(trial: pd.DataFrame, d: int) -> np.ndarray:
    """CoV of the typed estimates per numerosity, in-support (r <= N_SUPPORT_MAX)
    to match the in-support model CoV."""
    sub = trial[(trial["exp_dur"] == d) & (trial["n_guess"] <= T.N_SUPPORT_MAX)]
    out = []
    for nn in T.N_SHOWN:
        g = sub.loc[sub["n_shown"] == nn, "n_guess"].to_numpy(float)
        out.append(g.std() / g.mean() if g.size > 1 else np.nan)
    return np.array(out)


def main() -> None:
    apply_style(1.4)
    quad = T.make_quadrature(1601, kind="grid")
    wt = T.load_shared_theta()
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())          # 99, as in the headline fit
    wells = T.WELLS                              # integer support 1..N_SUPPORT_MAX
    n = T.N_SHOWN.astype(float)

    rows = []
    fig, ax = plt.subplots(figsize=(8, 6))
    dur_cmap = matplotlib.colormaps["magma"]
    durs = list(T.DURATIONS)
    colors = {d: dur_cmap(i / (len(durs) - 1)) for i, d in enumerate(durs)}

    for d in durs:
        theta = {p: float(wt.loc[d, p]) for p in WIN.param_names}
        lapse = float(wt.loc[d, "lapse"])
        grid = T.compute_grid(WIN, theta, quad)
        nc = n_width(theta)
        pf = T.apply_lapse(grid["p_resp"], lapse, r_max)  # (n, r_max) as fitted
        # in-support CoV, renormalised, the like-for-like match to the data dots
        model = np.array([cov_over(wells, pf[i, : wells.size]) for i in range(n.size)])
        emp = empirical_cov(trial, d)

        for nn, mm, ee in zip(n, model, emp):
            rows.append({"duration_ms": d, "n": int(nn), "n_width": round(nc, 2),
                         "model_cov": mm, "empirical_cov": ee})
        # n = 1 has no lower neighbour, so its CoV is an edge artefact; plot n >= 2
        ax.plot(n[1:], model[1:], "-", color=colors[d], lw=2.2, label=f"{d} ms")
        ax.plot(n[1:], emp[1:], "o", color=colors[d], ms=5, alpha=0.9)
        ax.axvline(nc, color=colors[d], ls=":", lw=1, alpha=0.5)

    ax.plot([], [], "-", color="0.35", label="model")
    ax.plot([], [], "o", color="0.35", label="data")
    ax.set_xlabel("Numerosity  $n$")
    ax.set_ylabel(r"Response CoV  $w_{\rm eff}(n)$")
    ax.set_xlim(1.6, 15.4)
    ax.set_xticks(np.arange(2, 16, 2))
    ax.legend(frameon=False, ncol=2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # characterise the Weber plateau: power law w_eff ~ a n^gamma over the
    # Preserve the width-based analysis window (n >= n_width) used for the
    # fitted model CoV and the data; this is not an exact-range boundary.
    print("=== Effective Weber fraction: CoV of the fitted model response ===")
    print(df.pivot(index="n", columns="duration_ms", values="model_cov").round(3).to_string())
    print("\n=== power-law slope gamma of log(CoV) vs log n, window n >= n_width ===")
    gm, ge = [], []
    for d in durs:
        sub = df[df.duration_ms == d]
        nc = sub["n_width"].iloc[0]
        est = sub[sub.n >= np.ceil(nc)]
        gmod = np.polyfit(np.log(est.n), np.log(est.model_cov), 1)[0]
        ed = est.dropna(subset=["empirical_cov"])
        gemp = np.polyfit(np.log(ed.n), np.log(ed.empirical_cov), 1)[0]
        gm.append(gmod)
        ge.append(gemp)
        print(f"  {d:>4} ms (n >= {int(np.ceil(nc)):2d}): model gamma = {gmod:+.2f}   data gamma = {gemp:+.2f}")
    print(f"\n  pooled gamma: model {np.mean(gm):+.2f}   data {np.mean(ge):+.2f}")
    print("  Reference: Weber's law gamma = 0 (constant CoV); a Poisson code "
          "predicts gamma = -0.5 (w_eff ~ 1/sqrt(n)).")
    print(f"\nSaved {OUT_CSV.name}, {OUT_FIG.name}")


if __name__ == "__main__":
    main()
