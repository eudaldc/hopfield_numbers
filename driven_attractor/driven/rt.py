"""RT channel for the driven family (Stage 7).

Established data fact (stage0_audit): RT is timed from stimulus OFFSET to
response submission. The driven model's natural RT predictor is therefore
its post-offset settling time in seconds, plus the residual readout
uncertainty -- the same two-term structure as the current model's hybrid
link, but with the dynamical term in physical time rather than iteration
counts.

Likelihood (identical to the registered convention):
    log rt ~ Normal(log(t0 + a * E[T](n,d) + b * E[H](n,d)), sigma_rt),
t0 = 1.30 s fixed, sigma_rt profiled. The landscape/evidence parameters are
held at the response-fit values (two-stage, matching the paper's logic).

Stopping rules compared (prespecified, small menu):
  A  'settle'  : velocity-threshold settling (the primary rule; also caps
                 at T_CAP) -- the baseline analogous to the current model.
  B  'entrate' : diminishing information gain -- decision at the first
                 checkpoint with |dH/dt| < EPS_H (bits/s) after T_MIN.
Link variants: hybrid (a, b free), dynamics-only (b = 0), entropy-only
(a = 0), and 'physical' (a = 1 FIXED: settle time enters in literal
seconds -- one fewer parameter and directly falsifiable).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from . import model as M

T0 = 1.30
EPS_H = 1.0          # bits/s information-gain threshold (rule B)
T_MIN = 0.05         # minimum post-offset deliberation (s) for rule B
RT_SEED = 731


def entropy_grid(sig: np.ndarray, table: M.RelaxTable) -> np.ndarray:
    """H(x_traj) for every checkpoint x grid state: (S, G)."""
    return M.entropy_of_readout(table.x_traj, sig)


def entrate_decision(table: M.RelaxTable, Hgrid: np.ndarray, x0: np.ndarray,
                     tau_x: float) -> np.ndarray:
    """Rule-B decision times (s) per path, from the relaxation checkpoints."""
    gi = np.clip(np.searchsorted(M.X_GRID, x0), 1, M.X_GRID.size - 1)
    left = np.abs(x0 - M.X_GRID[gi - 1]) <= np.abs(M.X_GRID[gi] - x0)
    gi = np.where(left, gi - 1, gi)
    th = table.th                                   # (S,)
    t_phys = tau_x * th
    H = Hgrid[:, gi]                                # (S, N)
    dH = np.abs(np.diff(H, axis=0))
    dt = np.diff(t_phys)[:, None]
    rate = dH / np.maximum(dt, 1e-12)               # (S-1, N)
    ok = (rate < EPS_H) & (t_phys[1:, None] >= T_MIN)
    ok &= (t_phys[1:, None] <= M.T_CAP)
    first = np.argmax(ok, axis=0)
    none = ~ok.any(axis=0)
    t_dec = t_phys[1:][first]
    t_dec[none] = min(M.T_CAP, float(t_phys[-1]))
    return t_dec


def rt_summaries(model: str, theta: dict, m_paths: int = 8192) -> dict:
    """E[T] and E[H] per (duration, n) for both stopping rules."""
    ev = M.evaluate_model(model, theta, m_paths=m_paths, keep_paths=True)
    out = {"settle_T": ev.e_trelax, "H": ev.e_entropy, "ev": ev}
    if model != "M3_static":
        sig = M.sigma_wells(theta["sigma_c"], theta["p"])
        table = M.build_relax_table(sig)
        Hg = entropy_grid(sig, table)
        x_off = ev.paths["x_off"].reshape(-1)
        K_d = ev.paths["K_d"].reshape(-1)
        t_dec = entrate_decision(table, Hg, x_off, float(theta.get("tau_x", 0.25)))
        sh = (len(M.DUR_S), M.N_SHOWN.size, -1)
        guess = K_d == 0
        pg = guess.reshape(sh).mean(axis=2)
        e_t = np.where(guess, 0.0, t_dec).reshape(sh).mean(axis=2)
        out["entrate_T"] = e_t / np.maximum(1e-12, 1.0 - pg)
    return out


def fit_rt_link(e_T: np.ndarray, e_H: np.ndarray, data_by_dur: dict,
                *, variant: str = "hybrid", shared: bool = True,
                seed: int = RT_SEED) -> dict:
    """Fit the log-normal RT link. e_T/e_H: (5, 15). shared=True fits one
    mapping across durations; shared=False fits one per duration and sums."""

    def negll_for(idx_durs, params):
        a, b = params
        tot, n_tot, jac = 0.0, 0, 0.0
        sse_parts = []
        for i in idx_durs:
            d = M.DURATIONS[i]
            dd = data_by_dur[d]
            mu_n = np.log(T0 + a * e_T[i] + b * e_H[i])
            if not np.all(np.isfinite(mu_n)):
                return None
            resid = dd.logrt_trial - mu_n[dd.n_trial - 1]
            sse_parts.append(resid)
            n_tot += resid.size
            jac += float(np.sum(dd.logrt_trial))
        resid = np.concatenate(sse_parts)
        sigma2 = float(np.mean(resid * resid))
        ll = -0.5 * n_tot * (np.log(2 * np.pi * sigma2) + 1.0) - jac
        return ll

    if variant == "hybrid":
        bounds = [(0.0, 20.0), (0.0, 5.0)]
        pack = lambda z: (z[0], z[1])
    elif variant == "dynamics":
        bounds = [(0.0, 20.0)]
        pack = lambda z: (z[0], 0.0)
    elif variant == "entropy":
        bounds = [(0.0, 5.0)]
        pack = lambda z: (0.0, z[0])
    elif variant == "physical":                 # a = 1 fixed (literal seconds)
        bounds = [(0.0, 5.0)]
        pack = lambda z: (1.0, z[0])
    else:
        raise ValueError(variant)

    def run(idx_durs):
        de = differential_evolution(
            lambda z: -(negll_for(idx_durs, pack(z)) or -1e18),
            bounds=bounds, maxiter=150, popsize=14, tol=1e-8, seed=seed,
            polish=True, updating="deferred")
        return -float(de.fun), [float(v) for v in de.x]

    if shared:
        ll, z = run(range(len(M.DURATIONS)))
        k = len(bounds) + 1                     # + profiled sigma
        params = [z]
    else:
        ll, params = 0.0, []
        for i in range(len(M.DURATIONS)):
            lli, zi = run([i])
            ll += lli
            params.append(zi)
        k = (len(bounds) + 1) * len(M.DURATIONS)
    n = sum(data_by_dur[d].logrt_trial.size for d in M.DURATIONS)
    return {"variant": variant, "shared": shared, "logL": ll, "k": k,
            "AIC": 2 * k - 2 * ll, "params": params, "n": n}
