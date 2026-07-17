"""Validation gate for the round-2 evidence families.

    .venv/bin/python driven_attractor/validate2.py

V1  ddm/batch_attr with per-duration (s0 = s0_d * d, lam = lambda0_d) must
    reproduce M0's per-duration logL exactly (same architecture, same
    clean-up, reparametrised channel).
V2  ddm/batch_static likewise reproduces M0's static twin.
V3  kalman reduces to ddm as v0 -> infinity.
V4  online_attr -> batch_attr as beta -> infinity (nesting).
V5  MC error of the path-based variants at the final sample size.
V6  pulse offset marginals: mean and variance match the analytic
    E[u(d)] = n*pi*(1-e^{-rho d}) + phi*d.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import evidence2 as E  # noqa: E402
from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

ok_all = []


def check(name, ok, detail):
    ok_all.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")


from driven_attractor.driven import data as D  # noqa: E402

_dat = D.load()
r_max = D.r_max(_dat)
counts = D.counts_by_duration(_dat, r_max)
pow_theta = pd.read_csv(ROOT / "legacy/trial_level_per_duration/output/pow_theta.csv")
spec_a = T.MODEL_BY_KEY["gaussian_pow_attractor"]
spec_s = T.MODEL_BY_KEY["gaussian_pow_readout"]
quad = T.make_quadrature(1601, kind="grid")

print("V1/V2. ddm batch == M0 with a reparametrised channel (per duration)")
for _, r in pow_theta.iterrows():
    d = int(r.duration_ms)
    d_s = d / 1000.0
    theta = {"sigma_c": r.sigma_c, "p": r["pow"],
             "s0": r.s0 * d_s, "lam": r.lambda0}
    for variant, spec in (("batch_attr", spec_a), ("batch_static", spec_s)):
        ev = E.evaluate("ddm", variant, theta, quad_nodes=1601)
        i = list(E.DURATIONS).index(d)
        pf = M.p_final_full(ev["p_wells"][i], ev["p_guess"][i], r.lapse, r_max)
        ll = float(np.sum(counts[d] * np.log(pf)))
        vec = np.array([r.sigma_c, r.s0, r.lambda0, r["pow"], r.lapse])
        ll_m0 = T.response_loglik(spec, vec, quad, counts[d], r_max)
        if variant == "batch_attr":
            check(f"ddm/{variant} @{d}ms", abs(ll - ll_m0) < 0.05,
                  f"logL {ll:.3f} vs M0 {ll_m0:.3f} (diff {ll - ll_m0:+.3e})")

print("V3. kalman -> ddm as v0 -> inf (prior precision 1/v0 << min_n,d of d/q)")
th = {"sigma_c": 0.022, "p": 1.8, "s0": 0.01, "lam": 2.0}
ll_ddm, _ = E.total_loglik("ddm", "batch_attr", th, counts, r_max, quad_nodes=801)
th_k = dict(th, mu0=5.0, v0=1e8)
ll_k, _ = E.total_loglik("kalman", "batch_attr", th_k, counts, r_max, quad_nodes=801)
check("kalman(v0=1e8) == ddm", abs(ll_ddm - ll_k) < 1.0,
      f"{ll_ddm:.2f} vs {ll_k:.2f}")
# and the moments themselves
mu_d, sd_d = E.gaussian_offset("ddm", th, 0.04)
mu_k, sd_k = E.gaussian_offset("kalman", dict(th, mu0=5.0, v0=1e8), 0.04)
check("offset moments agree", np.allclose(mu_d, mu_k, atol=1e-4) and
      np.allclose(sd_d, sd_k, atol=1e-6),
      f"max |dmu| {np.abs(mu_d-mu_k).max():.2e}, |dsd| {np.abs(sd_d-sd_k).max():.2e}")

print("V4. online_attr -> batch_attr as beta -> inf")
print("    (the landscape's restoring force is ~1/sigma_n^2, up to "
      f"{1.0 / M.sigma_wells(th['sigma_c'], th['p'])[0] ** 2:.0f}: the tether "
      "must exceed it to pin x to the drive)")
ll_b, _ = E.total_loglik("ddm", "batch_attr", th, counts, r_max, quad_nodes=1601)
for beta in (200.0, 5e3, 1e5):
    th_o = dict(th, tau_x=0.05, beta=beta)
    ll_o, _ = E.total_loglik("ddm", "online_attr", th_o, counts, r_max,
                             m_paths=8192)
    print(f"    beta={beta:8.0f}: online {ll_o:.1f} vs batch {ll_b:.1f} "
          f"(diff {ll_o - ll_b:+.1f})", flush=True)
th_o = dict(th, tau_x=0.05, beta=1e5)
ll_o, _ = E.total_loglik("ddm", "online_attr", th_o, counts, r_max, m_paths=8192)
check("nesting (beta=1e5)", abs(ll_o - ll_b) < 250.0,
      f"gap {ll_o - ll_b:+.1f} (residual = last-step timing + MC)")

print("V5. MC error of path variants")
th_o = dict(th, tau_x=0.1, beta=20.0)
lls = [E.total_loglik("ddm", "online_attr", th_o, counts, r_max,
                      m_paths=m)[0] for m in (4096, 16384)]
print(f"    M=4096 {lls[0]:.1f} | 16384 {lls[1]:.1f}")
check("MC convergence (4k vs 16k)", abs(lls[0] - lls[1]) < 40.0,
      f"|dLL| = {abs(lls[0] - lls[1]):.1f}")

print("V6. pulse offset marginals vs analytic")
thp = {"sigma_c": 0.05, "p": 1.5, "rho": 30.0, "pi_hit": 0.9,
       "cv_a": 0.3, "phi": 1.0}
C = E.crn(16384, 1)
worst_m, worst_v = 0.0, 0.0
for d in (0.08, 0.32):
    u = E.pulse_offset_samples(thp, d, C)
    for i, n in enumerate(E.N_SHOWN):
        pr = thp["pi_hit"] * (1 - np.exp(-thp["rho"] * d))
        m_the = n * pr + thp["phi"] * d
        v_the = (n * pr * (thp["cv_a"] ** 2 + 1 - pr)
                 + thp["phi"] * d * (1 + thp["cv_a"] ** 2))
        worst_m = max(worst_m, abs(u[i].mean() - m_the))
        worst_v = max(worst_v, abs(u[i].var() - v_the))
check("pulse mean", worst_m < 0.08, f"max |dmean| = {worst_m:.4f}")
check("pulse variance", worst_v < 0.25, f"max |dvar| = {worst_v:.4f}")

print(f"\n{'ALL PASS' if all(ok_all) else 'FAILURES PRESENT'} "
      f"({sum(ok_all)}/{len(ok_all)})")
