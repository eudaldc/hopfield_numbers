"""Small-range (n<=NCAP) refit + n_crit re-derivation.

Research question (reframed). The attractor may not be a *unified* OTS+ANS model
but a **mechanistic model of small-range (exact/subitising) numerosity** that, as an
emergent property, also predicts the exact->approximate transition (n ~ 4-9). This
script tests that reframing directly by fitting only to n <= NCAP and asking:

  (A) Does the fitted landscape move vs the full-range fit? If the large-n tail was
      dragging the width exponent up (to accommodate the wide approximate regime),
      restricting to n <= NCAP should give a *purer* small-range mechanism.
  (B) Within n <= NCAP, does the attractor + clean-up still beat Weber/ANS and its
      own static (no-clean-up) twin? If the discrete mechanism is doing real work in
      the small range, its advantage should survive -- or grow -- when the tail is
      removed.
  (C) Where does the re-derived n_crit (well width sigma_n = 1  <=>  adjacent-well
      d' = 1) land? Self-consistency for a "small-range" model means it should fall
      *inside* the fitted range (exact -> approximate ~ 8-10), i.e. the model knows
      where it stops being valid.

Only n <= NCAP trials enter the response likelihood (counts for n > NCAP are zero,
so they contribute nothing to sum counts*log p). n_crit and the predicted curves are
still evaluated over the full 1..15 support, to show the *extrapolated* transition
the small-range fit predicts for numerosities it never saw.

To keep the landscape comparison apples-to-apples, the winner is re-fit on BOTH the
capped (n <= NCAP) and full (n <= 15) data with identical optimiser settings here --
we do not reuse an older saved fit.

    .venv/bin/python -m scripts.trial_level_ncap_fit
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

NCAP = 10               # small-range cap (upper edge of the hypothesised transition)
WINNER = "gaussian_freebeta_attractor"
# Models compared *within* the small range (all fit + scored on the same n<=NCAP
# counts, so their AICs are mutually comparable). Covers well shape (Gaussian vs
# Poisson), clean-up (attractor vs static twin), and pure approximate (Weber/ANS).
MODELS = (
    "gaussian_freebeta_attractor",      # winner: symmetric super-linear + clean-up
    "gaussian_freebeta_no_cleanup",     # its static twin (isolates clean-up)
    "hopfield_poisson",                 # the published skewed-Poisson attractor
    "weber_ans",                        # pure approximate read-out
)

# Optimiser budget (matches run_all's real-run settings).
FIT_NODES, EVAL_NODES = 201, 1201
MAXITER, POPSIZE = 35, 10

OUT_FIT = T.OUT_DIR / "ncap10_fit.csv"
OUT_SUMMARY = T.OUT_DIR / "ncap10_summary.csv"
OUT_NCRIT = T.OUT_DIR / "ncap10_ncrit.csv"
OUT_FIG = T.OUT_DIR / "ncap10_comparison.png"


def cap_trials(trial: pd.DataFrame, ncap: int) -> pd.DataFrame:
    return trial[trial["n_shown"] <= ncap].reset_index(drop=True)


def sigma_n(theta: dict, n: np.ndarray) -> np.ndarray:
    """Gaussian-Benford well width sigma_n = (sigma_c / freq_n)^beta."""
    freq = np.log10(1.0 + 1.0 / np.asarray(n, dtype=float))
    return np.power(theta["sigma_c"] / freq, theta["beta"])


def ncrit_crossing(theta: dict, level: float = 1.0) -> float:
    """Smallest n where sigma_n(n) >= level (linear interpolation)."""
    ns = np.linspace(1.0, 40.0, 40000)
    s = sigma_n(theta, ns)
    idx = np.where(s >= level)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan")
    i = idx[0]
    return float(ns[i - 1] + (level - s[i - 1]) * (ns[i] - ns[i - 1]) /
                 (s[i] - s[i - 1]))


def pct_exact_crossing(pct: np.ndarray, level: float = 50.0) -> float:
    """Largest n where predicted exact accuracy is still >= level (emergent
    'where the discrete code stops dominating' criterion, over n = 1..15)."""
    ns = T.N_SHOWN.astype(float)
    below = np.where(pct < level)[0]
    if below.size == 0:
        return float(ns[-1])          # never drops below within range
    j = below[0]
    if j == 0:
        return float("nan")           # already below at n=1
    # linear interpolation between the last >=level and first <level point
    n0, n1 = ns[j - 1], ns[j]
    p0, p1 = pct[j - 1], pct[j]
    return float(n0 + (level - p0) * (n1 - n0) / (p1 - p0))


def fit_model(spec, data, quad_fit, quad_eval, r_max, seed) -> dict:
    return T.fit_response(spec, data, quad_fit, r_max, maxiter=MAXITER,
                          popsize=POPSIZE, seed=seed, quad_eval=quad_eval)


def main() -> None:
    T.OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad_fit = T.make_quadrature(FIT_NODES, kind="grid")
    quad_eval = T.make_quadrature(EVAL_NODES, kind="grid")

    trial_full = T.load_trials()
    r_max = int(trial_full["n_guess"].max())
    trial_cap = cap_trials(trial_full, NCAP)

    data_cap = T.build_duration_data(trial_cap, r_max)   # counts zero for n>NCAP
    data_full = T.build_duration_data(trial_full, r_max)

    print(f"Full: {len(trial_full)} trials (n<=15). "
          f"Capped: {len(trial_cap)} trials (n<={NCAP}). r_max={r_max}", flush=True)

    fit_rows: list[dict] = []
    # (A/B) fit every model on the capped data; also the winner on the full data.
    jobs = [("cap", key, data_cap) for key in MODELS]
    jobs += [("full", WINNER, data_full)]
    for regime, key, data_by_dur in jobs:
        spec = T.MODEL_BY_KEY[key]
        for d in T.DURATIONS:
            data = data_by_dur[d]
            rf = fit_model(spec, data, quad_fit, quad_eval, r_max, seed=T.SEED + d)
            row = {
                "regime": regime, "model_key": key, "model": spec.label,
                "duration_ms": int(d), "cleanup": spec.cleanup,
                "family": spec.family, "n_data": rf["n_data"],
                "logL": rf["logL"], "k_params": rf["k_params"],
                "AIC": rf["AIC"], "BIC": rf["BIC"], "lapse": rf["lapse"],
                **{f"theta_{p}": rf["theta"][p] for p in spec.param_names},
            }
            fit_rows.append(row)
            print(f"  [{regime}:{key} @ {d}ms] logL={rf['logL']:.1f} "
                  f"AIC={rf['AIC']:.1f} lapse={rf['lapse']:.3f}", flush=True)

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(OUT_FIT, index=False)

    # ── (B) ranking within the small range (capped fits only) ───────────────
    cap_df = fit_df[fit_df["regime"] == "cap"]
    summ_rows = []
    for key in MODELS:
        g = cap_df[cap_df["model_key"] == key]
        summ_rows.append({
            "model_key": key, "model": T.MODEL_BY_KEY[key].label,
            "cleanup": bool(T.MODEL_BY_KEY[key].cleanup),
            "AIC_total": float(g["AIC"].sum()),
            "logL_total": float(g["logL"].sum()),
            "k_total": int(g["k_params"].sum()),
        })
    summ = pd.DataFrame(summ_rows)
    summ["dAIC"] = summ["AIC_total"] - summ["AIC_total"].min()
    summ["akaike_weight"] = T.akaike_weights(summ["AIC_total"].to_numpy(float))
    summ = summ.sort_values("AIC_total").reset_index(drop=True)
    summ.to_csv(OUT_SUMMARY, index=False)

    # ── (A/C) landscape shift + n_crit, winner capped vs full ───────────────
    win = T.MODEL_BY_KEY[WINNER]
    ncrit_rows = []
    for d in T.DURATIONS:
        for regime, data_by_dur in (("cap", data_cap), ("full", data_full)):
            r = fit_df[(fit_df.regime == regime) & (fit_df.model_key == WINNER)
                       & (fit_df.duration_ms == d)].iloc[0]
            theta = {p: float(r[f"theta_{p}"]) for p in win.param_names}
            grid = T.compute_grid(win, theta, quad_eval)          # n = 1..15
            pf = T.apply_lapse(grid["p_resp"], float(r["lapse"]), r_max)
            psum = T.predicted_summaries(pf)
            pct_exact = psum["pct_exact"].to_numpy(float)
            ncrit_rows.append({
                "duration_ms": int(d), "regime": regime,
                "sigma_c": theta["sigma_c"], "s0": theta["s0"],
                "lambda0": theta["lambda0"], "beta": theta["beta"],
                "n_crit_sigma1": ncrit_crossing(theta, 1.0),
                "n_crit_sigma2": ncrit_crossing(theta, 2.0),
                "n_pct_exact_50": pct_exact_crossing(pct_exact, 50.0),
                "sigma_at_ncap": float(sigma_n(theta, np.array([NCAP]))[0]),
            })
    ncrit = pd.DataFrame(ncrit_rows)
    ncrit.to_csv(OUT_NCRIT, index=False)

    make_fig(fit_df, summ, ncrit)

    print("\n=== (B) ranking within n<=%d (capped, AIC summed over durations) ===" % NCAP)
    print(summ[["model", "AIC_total", "dAIC", "akaike_weight"]]
          .round({"AIC_total": 1, "dAIC": 1, "akaike_weight": 3}).to_string(index=False))
    print("\n=== (A/C) winner landscape + n_crit: capped vs full ===")
    show = ncrit.pivot_table(index="duration_ms", columns="regime",
                             values=["beta", "n_crit_sigma1", "n_pct_exact_50"])
    print(show.round(2).to_string())
    print(f"\nSaved {OUT_FIT.name}, {OUT_SUMMARY.name}, {OUT_NCRIT.name}, {OUT_FIG.name}")


def make_fig(fit_df: pd.DataFrame, summ: pd.DataFrame, ncrit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (1) n_crit vs duration, capped vs full
    ax = axes[0]
    for regime, mk, col in (("cap", "o-", "#c0392b"), ("full", "s--", "#2c7fb8")):
        g = ncrit[ncrit.regime == regime].sort_values("duration_ms")
        ax.plot(g["duration_ms"], g["n_crit_sigma1"], mk, color=col,
                label=f"{regime} (sigma_n=1)")
        ax.plot(g["duration_ms"], g["n_pct_exact_50"], mk, color=col, alpha=0.4,
                label=f"{regime} (P(exact)=50%)")
    ax.axhspan(4, 9, color="0.85", alpha=0.5, zorder=0, label="hypothesised 4-9")
    ax.axhline(NCAP, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(T.DURATIONS)
    ax.set_xticklabels([str(d) for d in T.DURATIONS])
    ax.set_xlabel("exposure (ms)")
    ax.set_ylabel("n_crit")
    ax.set_title("(C) exact->approximate boundary")
    ax.legend(fontsize=6.5, frameon=False)

    # (2) beta capped vs full
    ax = axes[1]
    piv = ncrit.pivot_table(index="duration_ms", columns="regime", values="beta")
    x = np.arange(len(piv.index))
    ax.bar(x - 0.2, piv["cap"], 0.4, color="#c0392b", label="n<=%d" % NCAP)
    ax.bar(x + 0.2, piv["full"], 0.4, color="#2c7fb8", label="n<=15")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in piv.index])
    ax.set_xlabel("exposure (ms)")
    ax.set_ylabel(r"width exponent $\beta$")
    ax.set_title("(A) did the tail distort the landscape?")
    ax.legend(fontsize=8, frameon=False)

    # (3) model ranking within n<=NCAP
    ax = axes[2]
    order = summ.sort_values("dAIC")
    cols = ["#c0392b" if c else "#7f8c8d" for c in order["cleanup"]]
    lbl = [T.MODEL_BY_KEY[k].label.split(" (")[0] for k in order["model_key"]]
    ax.barh(np.arange(len(order)), order["dAIC"], color=cols)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(lbl, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$AIC (summed over durations)")
    ax.set_title("(B) model ranking within n<=%d" % NCAP)

    fig.suptitle("Small-range (n<=%d) refit: is this a mechanistic small-number model "
                 "that predicts the ANS transition?" % NCAP, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
