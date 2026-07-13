"""Stage-5 sensitivities for the primary driven fit: T_CAP and well support.

    .venv/bin/python driven_attractor/sensitivity.py

1. T_CAP: re-evaluate (and lightly re-polish) M2_tether_g0 at T_CAP = 1 and
   4 s (fit used 2 s). If the model ranking depended on the cap, the gap to
   M0-shared (-114,915) would move by thousands; report the actual shifts.
2. Well support: extend the integer wells from 1..19 to 1..25 and
   re-evaluate; predictions for n = 1..15 must be stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import fitting as F  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

with open(ROOT / "driven_attractor/output/fits/M2_tether_g0.json") as f:
    fit = json.load(f)
theta = fit["theta"]

trial = T.load_trials()
r_max = int(trial["n_guess"].max())
data = T.build_duration_data(trial, r_max)
counts = {d: data[d].counts for d in M.DURATIONS}

print("baseline (T_CAP=2.0, wells 1..19): logL", round(fit["logL"], 1))

# ── 1. T_CAP ──────────────────────────────────────────────────────────────
defs = F.param_defs("M2_tether")
bounds = [d.bounds() for d in defs]
x0 = np.asarray(fit["x_opt"], float)
for cap in (1.0, 4.0):
    M.T_CAP = cap
    ll_eval, _ = M.total_loglik("M2_tether", theta, counts, r_max,
                                m_paths=16384, dt=0.0025)
    args = ("M2_tether", defs, 0.0, 1024, counts, r_max, 0.0025)
    pol = minimize(F._negll, x0, args=args, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 80, "xatol": 1e-4, "fatol": 0.1})
    th2 = F.vec_to_theta("M2_tether", pol.x, defs, 0.0)
    ll_pol, _ = M.total_loglik("M2_tether", th2, counts, r_max,
                               m_paths=16384, dt=0.0025)
    print(f"T_CAP={cap}: eval-at-fitted {ll_eval:.1f}  re-polished {ll_pol:.1f} "
          f"(baseline -118857.5; M0-shared -114915.2)", flush=True)
M.T_CAP = 2.0

# ── 2. well support 1..25 ────────────────────────────────────────────────
M.WELLS = np.arange(1.0, 26.0)
M.X_GRID = np.linspace(-1.0, 28.0, 5801)
ev19 = None
M.WELLS = T.WELLS  # reset for the 19-well reference
M.X_GRID = np.linspace(-1.0, 22.0, 4601)
ev19 = M.evaluate_model("M2_tether", theta, m_paths=8192, dt=0.0025)
ll19 = 0.0
for i, d in enumerate(M.DURATIONS):
    _, l = M.profile_lapse(ev19.p_wells[i], ev19.p_guess[i], counts[d], r_max)
    ll19 += l
M.WELLS = np.arange(1.0, 26.0)
M.X_GRID = np.linspace(-1.0, 28.0, 5801)
ev25 = M.evaluate_model("M2_tether", theta, m_paths=8192, dt=0.0025)
ll25 = 0.0
for i, d in enumerate(M.DURATIONS):
    _, l = M.profile_lapse(ev25.p_wells[i], ev25.p_guess[i], counts[d], r_max)
    ll25 += l
dp = np.abs(ev25.p_wells[:, :, :19] - ev19.p_wells).max()
print(f"well support 19 -> 25: logL {ll19:.1f} -> {ll25:.1f} "
      f"(diff {ll25 - ll19:+.1f}); max |dp(r|n)| over first 19 wells = {dp:.5f}")
