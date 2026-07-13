"""Numerical validation of the driven-model machinery (pre-fitting gate).

Run:  .venv/bin/python driven_attractor/validate_numerics.py

V1. Evidence process: for gamma = 1 the count K(d) must be
    Binomial(n, 1 - exp(-rho d)); for gamma = 0, min(Poisson(rho d), n)-like
    (checked against the exact capped-Poisson pmf).
V2. Relaxation table: basin assignment and settle behaviour vs a fine-step
    RK4 integration of dx/dth = -dV/dx from a dense set of starts.
V3. Tether-only exposure (M3): the integrator must match the closed-form
    solution of tau dx/dt = beta (K - x) for known arrival times.
V4. Likelihood determinism + MC size: logL at M=1024 vs 4096 vs re-seeded.
V5. Timing benchmark of one total_loglik call.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom, poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

rng = np.random.default_rng(0)
PASS = []


def check(name: str, ok: bool, detail: str) -> None:
    PASS.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


print("V1. evidence-process marginals")
E = M.exp_draws(4096)
for gamma, d, n, rho in ((1.0, 0.16, 10, 8.0), (1.0, 0.04, 5, 20.0)):
    arr = M.arrival_times(n, rho, gamma, E)
    K = (arr <= d).sum(axis=1)
    q = 1.0 - np.exp(-rho * d)
    emp = np.bincount(K, minlength=n + 1) / K.size
    the = binom.pmf(np.arange(n + 1), n, q)
    tv = 0.5 * np.abs(emp - the).sum()
    check(f"gamma=1 binomial (n={n}, rho={rho}, d={d})", tv < 0.02,
          f"total-variation {tv:.4f}")
for d, n, rho in ((0.16, 10, 30.0), (0.64, 15, 15.0)):
    lam = rho * d
    the = poisson.pmf(np.arange(n + 1), lam)
    the[n] = 1.0 - poisson.cdf(n - 1, lam)
    tvs = []
    for m in (4096, 16384):
        arr = M.arrival_times(n, rho, 0.0, M.exp_draws(m))
        K = (arr <= d).sum(axis=1)
        emp = np.bincount(K, minlength=n + 1) / K.size
        tvs.append(0.5 * np.abs(emp - the).sum())
    check(f"gamma=0 capped Poisson (n={n}, rho={rho}, d={d})",
          tvs[0] < 0.03 and tvs[1] < tvs[0],
          f"TV {tvs[0]:.4f} (M=4096) -> {tvs[1]:.4f} (M=16384), must shrink")

print("V2. relaxation table vs fine-step RK4 gradient flow (same horizon)")
TH_REF = 8.0     # dimensionless comparison horizon = T_CAP / tau_x at tau_x = 0.25
for sigma_c, p in ((0.0222, 1.797), (0.05, 1.5), (0.31, 0.94)):
    sig = M.sigma_wells(sigma_c, p)
    table = M.build_relax_table(sig)
    starts = np.linspace(0.3, 16.0, 400)
    x = starts.copy()
    h = 0.002

    def F(z, sig=sig):
        A, B = M.force_AB(z, sig)
        return A - B * z

    settled_ref = np.zeros(starts.shape, bool)
    for _ in range(int(TH_REF / h)):
        active = ~settled_ref
        k1 = F(x); k2 = F(x + 0.5 * h * k1)
        k3 = F(x + 0.5 * h * k2); k4 = F(x + h * k3)
        x = np.where(active, x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), x)
        settled_ref |= np.abs(F(x)) < M.F_TOL   # same stopping rule as the table
        x = np.clip(x, M.X_GRID[0], M.X_GRID[-1])
    # table lookup with the cap binding exactly at TH_REF
    x_dec, t_dec = table.decide(starts, tau_x=M.T_CAP / TH_REF, sig=sig)
    err = np.abs(x_dec - x)
    frac_bad = float(np.mean(err > 0.25))
    check(f"trajectory@th={TH_REF} sigma_c={sigma_c}, p={p}", frac_bad < 0.03,
          f"median |dx|={np.median(err):.4f}, frac>|0.25|={frac_bad:.3f}")

print("V3. tether-only integrator vs closed form")
tau_x, beta, d = 0.25, 4.0, 0.32
sig = M.sigma_wells(0.05, 1.5)
off = M.integrate_exposure("static", rho=12.0, gamma=0.0, tau_x=tau_x,
                           beta=beta, sig=sig, m_paths=256)
E256 = M.exp_draws(256)
i_dur = list(M.DURATIONS).index(320)
worst = 0.0
for n in (3, 9, 15):
    arr = M.arrival_times(n, 12.0, 0.0, E256)
    # closed form: between events x relaxes exponentially toward K
    x = np.zeros(256)
    times = np.hstack([np.zeros((256, 1)), np.clip(arr, 0, d),
                       np.full((256, 1), d)])
    for j in range(times.shape[1] - 1):
        dt_seg = times[:, j + 1] - times[:, j]
        Kseg = (arr <= times[:, j][:, None] + 1e-12).sum(axis=1).astype(float)
        x = Kseg + (x - Kseg) * np.exp(-beta * dt_seg / tau_x)
    err = np.abs(off.x[i_dur, n - 1] - x)
    worst = max(worst, float(err.max()))
check("tether closed form (n=3,9,15 @320ms)", worst < 0.03,
      f"max |dx| = {worst:.4f} (5 ms stepping vs exact)")

print("V4. likelihood determinism and MC error")
trial = T.load_trials()
r_max = int(trial["n_guess"].max())
data = T.build_duration_data(trial, r_max)
counts = {d: data[d].counts for d in M.DURATIONS}
theta = {"sigma_c": 0.0222, "p": 1.797, "rho": 25.0, "gamma": 0.0,
         "tau_x": 0.25, "beta": 4.0}
ll_a, _ = M.total_loglik("M2_tether", theta, counts, r_max, m_paths=2048)
ll_b, _ = M.total_loglik("M2_tether", theta, counts, r_max, m_paths=2048)
check("determinism (same seed, twice)", ll_a == ll_b, f"{ll_a:.4f} == {ll_b:.4f}")
ll_16k, _ = M.total_loglik("M2_tether", theta, counts, r_max, m_paths=16384)
ll_64k, _ = M.total_loglik("M2_tether", theta, counts, r_max, m_paths=65536)
ll_seed, _ = M.total_loglik("M2_tether", theta, counts, r_max, m_paths=16384,
                            seed=M.SOBOL_SEED + 1)
print(f"  fit-size (2048) offset vs 64k: {ll_a - ll_64k:+.2f}")
check("final-eval QMC error (16384 vs 65536)", abs(ll_16k - ll_64k) < 10.0,
      f"|dLL| = {abs(ll_16k - ll_64k):.2f}")
check("final-eval re-seed jitter (16384)", abs(ll_16k - ll_seed) < 15.0,
      f"|dLL| = {abs(ll_16k - ll_seed):.2f}")
# exposure-step sensitivity at the final-eval size
ll_dt, _ = M.total_loglik("M2_tether", theta, counts, r_max, m_paths=16384)
import driven_attractor.driven.model as MM
ev_fine = M.evaluate_model("M2_tether", theta, m_paths=16384, dt=0.0025)
tot_fine = 0.0
for i, d in enumerate(M.DURATIONS):
    _, ll = M.profile_lapse(ev_fine.p_wells[i], ev_fine.p_guess[i], counts[d], r_max)
    tot_fine += ll
check("exposure dt sensitivity (5 ms vs 2.5 ms)", abs(ll_16k - tot_fine) < 15.0,
      f"|dLL| = {abs(ll_16k - tot_fine):.2f}")

print("V5. timing")
t0 = time.time()
M.total_loglik("M2_tether", theta, counts, r_max, m_paths=1024)
t1 = time.time()
M.total_loglik("M1_offset", theta, counts, r_max, m_paths=1024)
t2 = time.time()
print(f"  M2_tether eval: {t1 - t0:.2f}s | M1_offset eval: {t2 - t1:.2f}s (M=1024)")

print(f"\n{'ALL PASS' if all(PASS) else 'FAILURES PRESENT'} ({sum(PASS)}/{len(PASS)})")
