"""Why does the driven sibling equal M0? Investigate the mechanism.

Hypotheses:
  H1  The post-offset clean-up (M0's 120-step operator) re-projects x(offset)
      to a well, erasing the during-exposure history. If x(offset) lands in the
      same basin as y0, the readout is IDENTICAL to M0 -> the driven model is
      vacuously M0 in the fitted regime.
  H2  The equality is specific to using M0's FULL clean-up post-offset. A driven
      model that reads out at offset (or with limited relaxation) preserves the
      online history and genuinely differs.

Diagnostics, per (n, duration), at the winner's per-duration params:
  A. Compare driven p(r|n,d) to M0 p(r|n,d): max |dp|, and the response logL,
     for the sibling (full M0 clean-up post-offset) at several beta.
  B. The same but reading out AT OFFSET (no post-offset clean-up) and with a
     short bounded relaxation -> does the driven model differ then?
  C. Fraction of paths whose FINAL well differs between driven and M0-at-same-y0
     (basin re-projection check).

    .venv/bin/python driven_attractor/investigate_equality.py
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

DUR = 160  # a mid duration, wells fairly sharp
PAR = "perdur"


def m0_pwells(par_d, r_max, quad):
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    th = {"sigma_c": par_d["sigma_c"], "s0": par_d["s0"],
          "lambda0": par_d["lam"], "pow": par_d["p"]}
    return T.compute_grid(spec, th, quad)["p_resp"]     # (15, 19)


def driven_pwells(par_d, beta, d_ms, m_paths, post="cleanup", relax_steps=T.MAX_STEPS):
    """Driven p_wells for one duration, with a choice of post-offset readout.
    post='cleanup' = M0's 120-step clean-up; post='offset' = read at offset;
    post='short'   = a few clean-up steps only."""
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
        x = M._guarded_step(x.reshape(-1), u.reshape(-1), sig, beta, S.DT,
                            attractor=True, gate=np.ones(x.size, bool)
                            ).reshape(S.N_SHOWN.size, m)
    x_off = x.copy()
    u_off = (n[:, None] + sigma_B[:, None] * (W[-1][None, :] / ((n_steps - 0.5) * S.DT))
             + S.S00 * xi[None, :])   # = y0 (offset evidence)
    if post == "cleanup":
        x_fin, _, _ = T._cleanup(x.reshape(-1), family="gaussian", sigma=sig,
                                 max_steps=T.MAX_STEPS)
    elif post == "short":
        x_fin, _, _ = T._cleanup(x.reshape(-1), family="gaussian", sigma=sig,
                                 max_steps=relax_steps)
    else:  # offset: no post-offset clean-up
        x_fin = x.reshape(-1)
    probs = M.readout_probs(x_fin, sig).reshape(S.N_SHOWN.size, m, -1).mean(axis=1)
    # also M0's clean-up applied to the SAME offset evidence u_off (= y0)
    y0 = u_off.reshape(-1)
    x_m0, _, _ = T._cleanup(y0, family="gaussian", sigma=sig,
                            max_steps=T.MAX_STEPS)
    # basin check: final well of driven vs M0-from-same-y0
    well_driven = np.round(np.clip(x_fin, 1, 19)) if post != "cleanup" else \
        np.round(np.clip(x_fin, 1, 19))
    well_m0 = np.round(np.clip(x_m0, 1, 19))
    frac_diff = float(np.mean(well_driven != well_m0)) if post == "cleanup" else np.nan
    return probs, x_off, u_off, frac_diff


def main():
    dat = D.load(); r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)
    par = S.m0_params(PAR)[DUR]
    quad = T.make_quadrature(1601, kind="grid")
    p_m0 = m0_pwells(par, r_max, quad)

    def logl(pw):
        pf = M.p_final_full(pw, np.zeros(15), 0.06, r_max)
        return float(np.sum(counts[DUR] * np.log(pf)))

    print(f"=== duration {DUR} ms, {PAR} params ===")
    print(f"M0 logL = {logl(p_m0):.1f}")
    print("\nA. driven with M0's FULL clean-up post-offset (the manuscript version):")
    for beta in (200.0, 500.0, 2000.0, 10000.0):
        pw, xo, uo, frac = driven_pwells(par, beta, DUR, 8192, post="cleanup")
        print(f"  beta={beta:7.0f}: logL {logl(pw):.1f}  max|dp vs M0| {np.abs(pw-p_m0).max():.4f}"
              f"  mean|dp| {np.abs(pw-p_m0).mean():.5f}  frac paths in diff well vs M0(y0) {frac:.3f}"
              f"  mean|x_off-y0| {np.abs(xo-uo).mean():.3f}")
    print("\nB. driven READING OUT AT OFFSET (no post-offset clean-up):")
    for beta in (200.0, 500.0, 2000.0, 10000.0):
        pw, xo, uo, _ = driven_pwells(par, beta, DUR, 8192, post="offset")
        print(f"  beta={beta:7.0f}: logL {logl(pw):.1f}  max|dp vs M0| {np.abs(pw-p_m0).max():.4f}"
              f"  mean|x_off-y0| {np.abs(xo-uo).mean():.3f}")
    print("\nC. driven with SHORT post-offset relaxation (10 steps):")
    for beta in (200.0, 500.0, 2000.0):
        pw, _, _, _ = driven_pwells(par, beta, DUR, 8192, post="short", relax_steps=10)
        print(f"  beta={beta:7.0f}: logL {logl(pw):.1f}  max|dp vs M0| {np.abs(pw-p_m0).max():.4f}")


if __name__ == "__main__":
    main()
