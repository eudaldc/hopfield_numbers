"""Confirm the driven == M0 equivalence across ALL five durations.

At a representative fitted coupling (the log-likelihood plateau, beta~3000),
for each duration report, at the winner's per-duration params:
  - mean |x_offset - y0|   (is the driven state = the completed estimate?)
  - fraction of paths whose settled well differs from M0-at-same-y0
  - driven response logL vs M0 response logL (lapse profiled), the honest gap.

If mean|x_off-y0| ~ 0 and frac_diff ~ 0 at every duration, the fitted driven
model is M0 everywhere, and the equivalence is not a 160 ms accident.

    .venv/bin/python driven_attractor/equality_all_durations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import data as D, model as M  # noqa: E402
import driven_attractor.driven_sibling as S  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

BETA = 3000.0
MP = 8192


def driven_one(par_d, d_ms, m_paths):
    d = d_ms / 1000.0
    sig = M.sigma_wells(par_d["sigma_c"], par_d["p"])
    n = S.N_SHOWN.astype(float)
    sigma_B = np.sqrt(np.maximum(1e-12, par_d["s0"] * np.power(n, par_d["lam"]) * d))
    rng = np.random.default_rng(909 + d_ms)
    half = m_paths // 2
    z = rng.standard_normal((int(round(d / S.DT)), half)); z = np.concatenate([z, -z], 1)
    W = np.cumsum(z * np.sqrt(S.DT), axis=0)
    xi = rng.standard_normal(half); xi = np.concatenate([xi, -xi])
    n_steps = W.shape[0]; m = m_paths
    x = np.zeros((S.N_SHOWN.size, m))
    for s in range(1, n_steps + 1):
        t = (s - 0.5) * S.DT
        u = n[:, None] + sigma_B[:, None] * (W[s - 1][None, :] / t) + S.S00 * xi[None, :]
        u = np.clip(u, -5.0, 60.0)
        x = M._guarded_step(x.reshape(-1), u.reshape(-1), sig, BETA, S.DT,
                            attractor=True, gate=np.ones(x.size, bool)
                            ).reshape(S.N_SHOWN.size, m)
    y0 = (n[:, None] + sigma_B[:, None] * (W[-1][None, :] / ((n_steps - 0.5) * S.DT))
          + S.S00 * xi[None, :])
    x_star, _, _ = T._cleanup(x.reshape(-1), family="gaussian", sigma=sig, max_steps=T.MAX_STEPS)
    x_m0, _, _ = T._cleanup(y0.reshape(-1), family="gaussian", sigma=sig, max_steps=T.MAX_STEPS)
    pw = M.readout_probs(x_star, sig).reshape(S.N_SHOWN.size, m, -1).mean(axis=1)
    frac = float(np.mean(np.round(np.clip(x_star, 1, 19)) != np.round(np.clip(x_m0, 1, 19))))
    return pw, float(np.abs(x.reshape(-1) - y0.reshape(-1)).mean()), frac


def main():
    dat = D.load(); r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)
    par = S.m0_params("perdur")
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")
    print(f"equivalence across durations at beta={BETA:.0f} (perdur params, m={MP})")
    print(f"{'dur':>5} {'sigma_c':>8} {'p':>5} {'mean|x_off-y0|':>15} "
          f"{'frac diff well':>15} {'driven logL':>13} {'M0 logL':>11} {'dlogL':>8}")
    tot_d = tot_m = 0.0
    for d in M.DURATIONS:
        pd = par[d]
        pw, dxy, frac = driven_one(pd, d, MP)
        ld, _ = M.profile_lapse(pw, np.zeros(15), counts[d], r_max)
        th = {"sigma_c": pd["sigma_c"], "s0": pd["s0"], "lambda0": pd["lam"], "pow": pd["p"]}
        g = T.compute_grid(spec, th, quad)
        lm, llm = M.profile_lapse(g["p_resp"], np.zeros(15), counts[d], r_max)
        # driven logL at its profiled lapse
        pf = M.p_final_full(pw, np.zeros(15), ld, r_max)
        lld = float(np.sum(counts[d] * np.log(pf)))
        tot_d += lld; tot_m += llm
        print(f"{d:>5} {pd['sigma_c']:>8.4f} {pd['p']:>5.2f} {dxy:>15.4f} "
              f"{frac:>15.3f} {lld:>13.1f} {llm:>11.1f} {lld-llm:>+8.1f}", flush=True)
    print(f"{'TOT':>5} {'':>8} {'':>5} {'':>15} {'':>15} {tot_d:>13.1f} {tot_m:>11.1f} {tot_d-tot_m:>+8.1f}")
    print("DONE")


if __name__ == "__main__":
    main()
