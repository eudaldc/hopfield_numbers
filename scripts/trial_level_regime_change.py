"""Exact -> approximate regime change: where does the discrete attractor stop
contributing, and does it coincide with sigma_n = 1 (~8-10)?

Hypothesis (reframed n_crit): the d'=1 / sigma_n = 1 crossing is not a
"transition to density/texture" but the point where the discrete integer wells
stop being resolvable, so beyond it responses are driven purely by approximate
magnitude. Operationalisation: at the *fitted winner landscape* (same theta,
same lapse), the clean-up's per-numerosity contribution to the response
likelihood = LL(clean-up on) - LL(clean-up off). If this benefit is positive at
small n and vanishes around sigma_n = 1, the discrete code is active only below
that boundary -- exact-then-approximate, located empirically.

Also compares, per n, the winner against a pure approximate (Weber) read-out:
above the boundary the discrete attractor should lose its advantage.

    .venv/bin/python -m scripts.trial_level_regime_change
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
WEB = T.MODEL_BY_KEY["weber_ans"]
OUT_FIG = T.OUT_DIR / "regime_change.png"
OUT_CSV = T.OUT_DIR / "regime_change.csv"


def per_n_ll(pf: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (counts * np.log(pf)).sum(axis=1)


def sigma_n(theta: dict, n: np.ndarray) -> np.ndarray:
    freq = np.log10(1.0 + 1.0 / n)
    return np.power(theta["sigma_c"] / freq, theta["beta"])


def crossing(theta: dict, level: float = 1.0) -> float:
    ns = np.linspace(1, 40, 40000)
    s = sigma_n(theta, ns)
    idx = np.where(s >= level)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan")
    i = idx[0]
    return float(ns[i - 1] + (level - s[i - 1]) * (ns[i] - ns[i - 1]) / (s[i] - s[i - 1]))


def main() -> None:
    quad = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)
    wt = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    resp = pd.read_csv(T.OUT_DIR / "per_duration_response_likelihood.csv")

    rows = []
    fig, axes = plt.subplots(1, len(T.DURATIONS), figsize=(3.0 * len(T.DURATIONS), 3.6),
                             squeeze=False)
    for ci, d in enumerate(T.DURATIONS):
        theta = {c: float(wt.loc[d, c]) for c in ("sigma_c", "s0", "lambda0", "beta")}
        lapse = float(wt.loc[d, "lapse"])
        counts = data_by_dur[d].counts
        pf_on = T.apply_lapse(T.compute_grid(WIN, theta, quad)["p_resp"], lapse, r_max)
        pf_off = T.apply_lapse(
            T.compute_grid(WIN, theta, quad, force_no_cleanup=True)["p_resp"], lapse, r_max)
        ll_on, ll_off = per_n_ll(pf_on, counts), per_n_ll(pf_off, counts)
        cleanup_benefit = ll_on - ll_off            # per n (LL units)

        wr = resp[(resp.model_key == "weber_ans") & (resp.duration_ms == d)].iloc[0]
        wtheta = {p: float(wr[f"theta_{p}"]) for p in WEB.param_names}
        pf_web = T.apply_lapse(T.compute_grid(WEB, wtheta, quad)["p_resp"],
                               float(wr["lapse"]), r_max)
        discrete_adv = ll_on - per_n_ll(pf_web, counts)  # winner - Weber, per n

        n_star = crossing(theta, 1.0)
        # boundary: largest n where clean-up still adds >=1 LL unit
        active = np.where(cleanup_benefit >= 1.0)[0]
        n_cleanup_active = int(T.N_SHOWN[active[-1]]) if active.size else 0

        for i, n in enumerate(T.N_SHOWN):
            rows.append({"duration_ms": d, "n_shown": int(n),
                         "cleanup_benefit": float(cleanup_benefit[i]),
                         "winner_minus_weber": float(discrete_adv[i]),
                         "sigma_n": float(sigma_n(theta, np.array([n]))[0])})

        ax = axes[0][ci]
        ax.axhline(0, color="0.7", lw=0.8)
        ax.bar(T.N_SHOWN, cleanup_benefit, color="#c0392b", alpha=0.85,
               label="clean-up benefit")
        ax.plot(T.N_SHOWN, discrete_adv, "o-", color="#2c7fb8", ms=3, lw=1,
                label="winner − Weber")
        if np.isfinite(n_star):
            ax.axvline(n_star, color="k", ls="--", lw=1.2)
            ax.text(n_star + 0.2, ax.get_ylim()[1] * 0.85, f"σ=1\n@{n_star:.1f}",
                    fontsize=7)
        ax.set_title(f"{d} ms", fontsize=9)
        ax.set_xlabel("n shown")
        if ci == 0:
            ax.set_ylabel("Δ response LL (per n)")
            ax.legend(fontsize=6.5, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        print(f"{d}ms: sigma_n=1 at n={n_star:.1f}; clean-up adds >=1 LL up to "
              f"n={n_cleanup_active}; total clean-up benefit={cleanup_benefit.sum():.0f} LL",
              flush=True)

    fig.suptitle("Where the discrete attractor stops contributing (exact → approximate)",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=190)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_FIG} and {OUT_CSV}")


if __name__ == "__main__":
    main()
