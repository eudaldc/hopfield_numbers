"""Transition-to-density (SM4) recomputed with the WINNER model.

The earlier SM4 (`density_transition.py`) summarized the old Poisson width law
with n_width ~ 13–29 (sigma_n = 1, i.e. 1/c). This diagnostic computes three
width-profile crossings for the super-linear Gaussian attractor:

  * n_width at sigma_n = 1
  * additional descriptive crossings at sigma_n = 2 and sigma_n = 3

Here we recompute, from the winner's Piantadosi fit (`winner_theta.csv`), the
width crossings analytically **and** the simulation-based criteria (posterior
entropy > log 3, mode probability < 0.5, MAP-discriminability d' < 1) over n = 1..50,
plus the effective Weber fraction w_eff(n) = std(n̂)/n and whether it is Weber-flat
(ANS-like) or 1/√n in the approximate regime (the "Reynvoet/Weber" check). We reuse
the validated trial-level `compute_grid`, monkey-patching the integer-well support up
to 55 (the trial-level default caps at 19). Non-destructive: writes new
`output/density_transition/gaussian_freebeta_*` files.

    .venv/bin/python -m scripts.density_transition_gaussian_freebeta
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

# ── extend the integer-well support to cover the density range ───────────
N_MAX = 50
N_PAD = 5
T.WELLS = np.arange(1, N_MAX + N_PAD + 1, dtype=float)
T.FREQ = np.log10(1.0 + 1.0 / T.WELLS)
T.N_WELLS = int(T.WELLS.size)
T.LOG_WELLS = np.log(T.WELLS)

SPEC = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
OUT_DIR = _ROOT / "output" / "density_transition"
DURATIONS = T.DURATIONS
NVALS = np.arange(1, N_MAX + 1)
# literature density-transition ranges (from density_transition.py)
ANOBILE = (30.0, 60.0)
DURGIN = 20.0


def sigma_well(sigma_c, beta, n):
    freq = np.log10(1.0 + 1.0 / np.asarray(n, float))
    return np.power(sigma_c / freq, beta)


def cross_level(sigma_c, beta, level):
    ns = np.linspace(1.0, 80.0, 80000)
    s = sigma_well(sigma_c, beta, ns)
    idx = np.where(s >= level)[0]
    return float("nan") if idx.size == 0 or idx[0] == 0 else float(
        ns[idx[0] - 1] + (level - s[idx[0]-1]) * (ns[idx[0]]-ns[idx[0]-1]) /
        (s[idx[0]]-s[idx[0]-1]))


def cross_metric(nvals, metric, level, rising):
    """First n where metric crosses level (rising=True: metric>=level)."""
    m = metric
    hit = np.where(m >= level)[0] if rising else np.where(m <= level)[0]
    if hit.size == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(nvals[0])
    n0, n1, m0, m1 = nvals[i-1], nvals[i], m[i-1], m[i]
    return float(n0 + (level - m0) * (n1 - n0) / (m1 - m0)) if m1 != m0 else float(n1)


def weber_shape(nvals, w_eff, lo=10, hi=40):
    """AIC of constant-w vs a/√n fits to w_eff over [lo,hi] (positive → 1/√n wins)."""
    m = (nvals >= lo) & (nvals <= hi)
    n, w = nvals[m].astype(float), w_eff[m]
    k = w.size
    # constant
    c = w.mean(); sse_c = np.sum((w - c) ** 2)
    # a/sqrt(n): a = sum(w/sqrt n)/sum(1/n)
    x = 1.0 / np.sqrt(n)
    a = np.sum(w * x) / np.sum(x * x); sse_s = np.sum((w - a * x) ** 2)
    aic = lambda sse, p: k * np.log(sse / k + 1e-18) + 2 * p
    return {"w_const": c, "a_sqrt": a, "aic_const": aic(sse_c, 1),
            "aic_sqrt": aic(sse_s, 1), "delta_aic": aic(sse_c, 1) - aic(sse_s, 1)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = T.make_quadrature(801, kind="grid")
    wt = pd.read_csv(T.LEGACY_OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    wells = T.WELLS

    crit_rows, metric_frames, weber_rows = [], [], []
    for d in DURATIONS:
        theta = {p: float(wt.loc[d, p]) for p in SPEC.param_names}
        grid = T.compute_grid(SPEC, theta, quad, n_values=NVALS.astype(float))
        p_resp = grid["p_resp"]                    # (n, K_wells) marginal posterior
        probs_nodes = grid["probs_nodes"]          # (n, K_quad, K_wells)
        w = quad.w

        # per-node MAP estimate and posterior mean → estimate distribution over noise
        n_hat = wells[probs_nodes.argmax(axis=-1)]          # (n, K_quad)
        e_nhat = (n_hat * w[None, :]).sum(axis=1)
        e_nhat2 = (n_hat ** 2 * w[None, :]).sum(axis=1)
        std_nhat = np.sqrt(np.maximum(0.0, e_nhat2 - e_nhat ** 2))
        w_eff = std_nhat / NVALS
        # marginal posterior metrics
        entropy = -(p_resp * np.log(p_resp + 1e-12)).sum(axis=1)   # H_mean(n)
        p_max = p_resp.max(axis=1)
        # adjacent MAP-discriminability d'(n) between n and n+1
        pooled = np.sqrt(0.5 * (std_nhat[:-1] ** 2 + std_nhat[1:] ** 2)) + 1e-9
        dprime = (e_nhat[1:] - e_nhat[:-1]) / pooled                # length n-1

        # analytical width-profile crossings
        n_sig1 = cross_level(theta["sigma_c"], theta["beta"], 1.0)
        n_sig2 = cross_level(theta["sigma_c"], theta["beta"], 2.0)
        n_sig3 = cross_level(theta["sigma_c"], theta["beta"], 3.0)
        crit_rows.append({
            "duration_ms": d, "beta": theta["beta"], "sigma_c": theta["sigma_c"],
            "n_width": n_sig1,
            "n_sigma2_crossing": n_sig2, "n_sigma3_crossing": n_sig3,
            "n_threshold_entropy_log3": cross_metric(NVALS, entropy, np.log(3.0), True),
            "n_threshold_pmax_0.5": cross_metric(NVALS, p_max, 0.5, False),
            "n_threshold_dprime_1": cross_metric(NVALS[:-1], dprime, 1.0, False),
        })
        wb = weber_shape(NVALS, w_eff)
        weber_rows.append({"duration_ms": d, **wb})
        for i, n in enumerate(NVALS):
            metric_frames.append({"duration_ms": d, "n": int(n), "entropy": entropy[i],
                                  "p_max": p_max[i], "std_nhat": std_nhat[i],
                                  "w_eff": w_eff[i], "sigma_well": sigma_well(theta["sigma_c"], theta["beta"], n)})

    crit = pd.DataFrame(crit_rows)
    crit.to_csv(OUT_DIR / "gaussian_freebeta_critical_n.csv", index=False)
    pd.DataFrame(metric_frames).to_csv(OUT_DIR / "gaussian_freebeta_metrics.csv", index=False)
    pd.DataFrame(weber_rows).to_csv(OUT_DIR / "gaussian_freebeta_weber.csv", index=False)

    make_fig(crit, pd.DataFrame(metric_frames), pd.DataFrame(weber_rows))

    print("=== Winner model: width crossings and response thresholds per duration ===")
    print(crit.round(2).to_string(index=False))
    print("\n=== Weber shape of w_eff(n) over n=10-40 (delta_aic>0 → 1/√n beats constant) ===")
    print(pd.DataFrame(weber_rows).round(3)[["duration_ms", "w_const", "a_sqrt", "delta_aic"]]
          .to_string(index=False))
    # comparison to the OLD Poisson SM4
    old = OUT_DIR / "critical_n_summary.csv"
    if old.exists():
        o = pd.read_csv(old)
        print("\n=== OLD Poisson SM4 (critical_n_summary.csv) for reference ===")
        print(o[["duration_ms", "n_width_dprime", "n_threshold_entropy",
                 "n_threshold_pmax"]].to_string(index=False))
    print(f"\nSaved gaussian_freebeta_{{critical_n,metrics,weber}}.csv + figure in {OUT_DIR}")


def make_fig(crit, metrics, weber) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # (1) width-profile crossings vs duration
    a = ax[0]
    a.axhspan(*ANOBILE, color="0.9", label="Anobile density (30–60)")
    a.axhline(DURGIN, color="0.6", ls=":", lw=1, label="Durgin ~20")
    a.plot(crit.duration_ms, crit.n_width, "o-", color="#c0392b",
           label="n_width (σ=1)")
    a.plot(crit.duration_ms, crit.n_sigma2_crossing, "s--", color="#e08e0b",
           label="width crossing (σ=2)")
    a.plot(crit.duration_ms, crit.n_sigma3_crossing, "^--", color="#8e44ad",
           label="width crossing (σ=3)")
    a.set_xscale("log"); a.set_xticks(T.DURATIONS)
    a.set_xticklabels([str(x) for x in T.DURATIONS])
    a.set_xlabel("exposure (ms)"); a.set_ylabel("numerosity n")
    a.set_title("Two emergent boundaries (winner model)")
    a.legend(fontsize=7, frameon=False)
    # (2) w_eff(n) for the reference duration
    b = ax[1]
    for d, c in zip(T.DURATIONS, plt.colormaps["viridis"](np.linspace(0.1, 0.9, len(T.DURATIONS)))):
        m = metrics[metrics.duration_ms == d]
        b.plot(m.n, m.w_eff, color=c, lw=1.4, label=f"{d} ms")
    b.axvspan(10, 40, color="0.93", zorder=0)
    b.set_xlabel("numerosity n"); b.set_ylabel("effective Weber fraction std(n̂)/n")
    b.set_title("Weber fraction — approximate regime (n=10–40 shaded)")
    b.legend(fontsize=7, frameon=False)
    fig.suptitle("Transition to density with the super-linear Gaussian attractor", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_freebeta_density.png", dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    main()
