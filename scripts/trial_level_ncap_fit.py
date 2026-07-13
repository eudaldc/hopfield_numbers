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

    .venv/bin/python -m scripts.trial_level_ncap_fit       # full refit (slow)
    .venv/bin/python -m scripts.trial_level_ncap_figure    # figure only, from the saved CSVs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

NCAP = 10               # small-range cap (upper edge of the hypothesised transition)
WINNER = T.WINNER_KEY
# Models compared *within* the small range (all fit + scored on the same n<=NCAP
# counts, so their AICs are mutually comparable). Covers clean-up (attractor vs
# static twin) and pure approximate (Weber/ANS).
MODELS = (
    "gaussian_pow_attractor",           # winner: power-law super-linear + clean-up
    "gaussian_pow_readout",             # its static twin (isolates clean-up)
    "weber_ans",                        # pure approximate read-out
)

# Optimiser budget (matches run_all's real-run settings).
FIT_NODES, EVAL_NODES = 201, 1201
MAXITER, POPSIZE = 35, 10

OUT_FIT = T.OUT_DIR / "ncap10_fit.csv"
OUT_SUMMARY = T.OUT_DIR / "ncap10_summary.csv"
OUT_NCRIT = T.OUT_DIR / "ncap10_ncrit.csv"


def cap_trials(trial: pd.DataFrame, ncap: int) -> pd.DataFrame:
    return trial[trial["n_shown"] <= ncap].reset_index(drop=True)


def sigma_n(theta: dict, n: np.ndarray) -> np.ndarray:
    """Power-law well width sigma_n = sigma_c * n^p."""
    return theta["sigma_c"] * np.power(np.asarray(n, dtype=float), theta["pow"])


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
                "lambda0": theta["lambda0"], "pow": theta["pow"],
                "n_crit_sigma1": ncrit_crossing(theta, 1.0),
                "n_crit_sigma2": ncrit_crossing(theta, 2.0),
                "n_pct_exact_50": pct_exact_crossing(pct_exact, 50.0),
                "sigma_at_ncap": float(sigma_n(theta, np.array([NCAP]))[0]),
            })
    ncrit = pd.DataFrame(ncrit_rows)
    ncrit.to_csv(OUT_NCRIT, index=False)

    # the SM figure is owned by the standalone script (replots from the CSVs)
    from scripts.trial_level_ncap_figure import main as make_ncap_figure
    make_ncap_figure()

    print("\n=== (B) ranking within n<=%d (capped, AIC summed over durations) ===" % NCAP)
    print(summ[["model", "AIC_total", "dAIC", "akaike_weight"]]
          .round({"AIC_total": 1, "dAIC": 1, "akaike_weight": 3}).to_string(index=False))
    print("\n=== (A/C) winner landscape + n_crit: capped vs full ===")
    show = ncrit.pivot_table(index="duration_ms", columns="regime",
                             values=["pow", "n_crit_sigma1", "n_pct_exact_50"])
    print(show.round(2).to_string())
    print(f"\nSaved {OUT_FIT.name}, {OUT_SUMMARY.name}, {OUT_NCRIT.name}")


if __name__ == "__main__":
    main()
