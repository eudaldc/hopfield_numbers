"""Does the driven model ever SETTLE DURING EXPOSURE, and does it help?

Tests the manuscript's postulate ("at long exposures the attractor settles
during the exposure itself") mechanistically. Per duration, per coupling beta:
  - mean |x_offset - y0|          : 0 = tracks evidence (M0); large = moved
  - frac paths settled in a well  : |x_offset - nearest integer| < 0.15
  - logL, CLEAN-UP readout        : driven, post-offset clean-up (manuscript)
  - logL, STATIC-at-offset readout: read x_offset directly, no clean-up
  - M0 clean-up logL (reference)

Signature of genuine during-exposure settling: at long exposures + the right
beta, the STATIC-at-offset logL approaches the clean-up logL (the state is
already in a well by offset). At brief exposures, static-at-offset should be
much worse (state not settled). If no beta makes static-at-offset good while
also fitting well, the "settles during exposure" story is really about
evidence precision, not dynamics.

    .venv/bin/python driven_attractor/settle_during_exposure.py
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

DURS = (40, 160, 640)
BETAS = (30.0, 100.0, 300.0, 1000.0, 5000.0)
MP = 8192


def run(par_d, d_ms, beta, m_paths):
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
    y0 = (n[:, None] + sigma_B[:, None] * (W[-1][None, :] / ((n_steps - 0.5) * S.DT))
          + S.S00 * xi[None, :])
    x_off = x.reshape(-1)
    settled = float(np.mean(np.abs(x_off - np.round(np.clip(x_off, 1, 19))) < 0.15))
    x_star, _, _ = T._cleanup(x_off, family="gaussian", sigma=sig, max_steps=T.MAX_STEPS)
    pw_clean = M.readout_probs(x_star, sig).reshape(S.N_SHOWN.size, m, -1).mean(1)
    pw_stat = M.readout_probs(x_off, sig).reshape(S.N_SHOWN.size, m, -1).mean(1)
    dxy = float(np.abs(x_off - y0.reshape(-1)).mean())
    return pw_clean, pw_stat, dxy, settled


def main():
    dat = D.load(); r_max = D.r_max(dat)
    counts = D.counts_by_duration(dat, r_max)
    par = S.m0_params("perdur")
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")

    def ll(pw, d):
        lp, v = M.profile_lapse(pw, np.zeros(15), counts[d], r_max)
        return v

    for d in DURS:
        pd = par[d]
        g = T.compute_grid(spec, {"sigma_c": pd["sigma_c"], "s0": pd["s0"],
                                  "lambda0": pd["lam"], "pow": pd["p"]}, quad)
        m0ll = ll(g["p_resp"], d)
        print(f"\n=== {d} ms  (sigma_c={pd['sigma_c']:.4f}, p={pd['p']:.2f}; "
              f"M0 clean-up logL={m0ll:.1f}) ===")
        print(f"{'beta':>7} {'|x_off-y0|':>11} {'frac settled':>13} "
              f"{'clean-up logL':>14} {'static logL':>12} {'clean-M0':>9}")
        for beta in BETAS:
            pwc, pws, dxy, settled = run(pd, d, beta, MP)
            print(f"{beta:>7.0f} {dxy:>11.3f} {settled:>13.3f} "
                  f"{ll(pwc, d):>14.1f} {ll(pws, d):>12.1f} {ll(pwc, d)-m0ll:>+9.1f}",
                  flush=True)
    print("\nDONE")


if __name__ == "__main__":
    main()
