"""Continuously driven attractor models for the Piantadosi duration data.

Models use the registered winner landscape (power-law-width Gaussian wells,
sigma_m = sigma_c * m^p on integer wells 1..19), with an equal-well ablation
that fixes p = 0. They share the flat softmax readout
p(r | x) = softmax_m(g_m(x)), the per-duration uniform lapse over r = 1..99,
and the count-table categorical likelihood -- imported from
``scripts.trial_level_model_comparison`` so the conventions cannot drift.

Evidence process (Stage 1). A latent count K(t) of registered items rises
from 0 to at most n during the exposure [0, d]. Inter-registration gaps are

    gap_i = E_i / (rho * (n - i)**gamma),   i = 0..n-1,

with E_i ~ Exp(1) drawn once from a fixed scrambled Sobol sequence (common
random numbers across every parameter evaluation). gamma = 0 is the serial
capped accumulator; gamma = 1 is parallel independent registration (order
statistics of n iid Exp(rho) latencies).

Dynamics (Stage 2). Dimensionless time th = t / tau_x. During exposure the
soft-tether model follows

    dx/dth = I[K>0] * (A(x) - B(x) x) + beta * (K - x),

where A(x) = sum_m w_m m / sigma_m^2, B(x) = sum_m w_m / sigma_m^2 and
w = softmax(g(x)) -- so A - Bx = -dV/dx exactly. Integration is exponential
Euler with frozen weights (A-stable): over a step h,
x' = xbar + (x - xbar) exp(-(B + beta) h), xbar = (A + beta K)/(B + beta).
A displacement cap of 0.5 evidence units per step prevents cross-well jumps.

After offset the tether is removed and the state relaxes on -dV/dx alone.
Relaxation depends only on x(d) (and the landscape), so it is precomputed on
a dense x-grid in dimensionless time with geometrically growing steps and
looked up per path. Settling: |dx/dth| < F_TOL (dimensionless force, so the
criterion is tau-independent); decision time t_dec = min(tau_x * th_settle,
T_CAP) with T_CAP a fixed physical cap (the state stops moving where it
stands if the cap binds first, mirroring the bounded clean-up of the
published model).

Zero-evidence rule: paths with K(d) = 0 never engage the landscape and
respond as a pure guess, uniform over 1..r_max (stated in the tech note; a
baseline-well variant is a sensitivity analysis).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.trial_level_model_comparison as T  # noqa: E402

WELLS = T.WELLS                      # 1..19
N_SHOWN = T.N_SHOWN                  # 1..15
DURATIONS = T.DURATIONS              # (40, 80, 160, 320, 640) ms
DUR_S = np.asarray(DURATIONS, float) / 1000.0
N_MAX = int(N_SHOWN.max())           # 15

# fixed numerical constants (stated in the tech note, sensitivity-checked)
DT_EXPO = 0.005          # exposure integration step (s); all durations divide it
F_TOL = 1e-3             # dimensionless settling force threshold
T_CAP = 2.0              # physical cap on post-offset relaxation (s)
RELAX_H0 = 0.02          # first relaxation step (dimensionless time)
RELAX_GROW = 1.05        # geometric step growth
RELAX_THMAX = 400.0      # max dimensionless relaxation horizon
STEP_CAP = 0.5           # max |dx| per integration step (evidence units)
X_GRID = np.linspace(-1.0, 22.0, 4601)   # relaxation-table support
SOBOL_SEED = 20260710

MODEL_KEYS = ("M1_offset", "M2_tether", "M2_tether_const",
              "M2_pulse", "M3_static")


# ── landscape ─────────────────────────────────────────────────────────────

def sigma_wells(sigma_c: float, p: float) -> np.ndarray:
    return np.maximum(1e-9, sigma_c * np.power(WELLS, p))


def logwells(x: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """g_m(x) for each well; x any shape, returns shape x.shape + (19,)."""
    diff = x[..., None] - WELLS
    return -np.log(sig) - diff * diff / (2.0 * sig * sig)


def softmax_lastaxis(g: np.ndarray) -> np.ndarray:
    z = g - g.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def force_AB(x: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A(x), B(x) with -dV/dx = A - B x (softmax-weighted precision sums)."""
    w = softmax_lastaxis(logwells(x, sig))
    inv2 = 1.0 / (sig * sig)
    B = (w * inv2).sum(axis=-1)
    A = (w * (WELLS * inv2)).sum(axis=-1)
    return A, B


def readout_probs(x: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return softmax_lastaxis(logwells(x, sig))


def entropy_of_readout(x: np.ndarray, sig: np.ndarray) -> np.ndarray:
    p = readout_probs(x, sig)
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


# ── evidence process (common random numbers) ──────────────────────────────

_EXP_CACHE: dict[tuple[int, int], np.ndarray] = {}


def exp_draws(m_paths: int, seed: int = SOBOL_SEED) -> np.ndarray:
    """(m_paths, N_MAX) unit-exponential draws from scrambled Sobol, cached."""
    key = (m_paths, seed)
    if key not in _EXP_CACHE:
        eng = qmc.Sobol(d=N_MAX, scramble=True, seed=seed)
        u = eng.random(m_paths)
        _EXP_CACHE[key] = -np.log1p(-np.clip(u, 1e-12, 1 - 1e-12))
    return _EXP_CACHE[key]


def arrival_times(n: int, rho: float, gamma: float, E: np.ndarray) -> np.ndarray:
    """(m_paths, N_MAX) arrival times; entries j >= n are +inf."""
    m = E.shape[0]
    out = np.full((m, N_MAX), np.inf)
    if n == 0:
        return out
    rates = rho * np.power(n - np.arange(n, dtype=float), gamma)  # K=0..n-1
    out[:, :n] = np.cumsum(E[:, :n] / rates, axis=1)
    return out


# ── relaxation table (post-offset, dimensionless time) ────────────────────

@dataclass
class RelaxTable:
    th: np.ndarray            # (S,) cumulative dimensionless times (0 first)
    x_traj: np.ndarray        # (S, G) state at each checkpoint per grid start
    th_settle: np.ndarray     # (G,) dimensionless settle time (inf if never)

    def decide(self, x0: np.ndarray, tau_x: float,
               sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map offset states to (x_dec, t_dec seconds) under cap T_CAP."""
        th_cap = T_CAP / tau_x
        s_cap = int(np.searchsorted(self.th, th_cap, side="right") - 1)
        gi = np.clip(np.searchsorted(X_GRID, x0), 1, X_GRID.size - 1)
        left = np.abs(x0 - X_GRID[gi - 1]) <= np.abs(X_GRID[gi] - x0)
        gi = np.where(left, gi - 1, gi)
        ths = self.th_settle[gi]
        capped = ths > th_cap
        s_idx = np.where(capped, s_cap,
                         np.searchsorted(self.th, np.minimum(ths, self.th[-1]),
                                         side="right") - 1)
        x_dec = self.x_traj[s_idx, gi]
        t_dec = tau_x * np.minimum(ths, th_cap)
        return x_dec, t_dec


def build_relax_table(sig: np.ndarray) -> RelaxTable:
    x = X_GRID.copy()
    ths = [0.0]
    traj = [x.copy()]
    th_settle = np.full(x.shape, np.inf)
    h = RELAX_H0
    th = 0.0
    active = np.ones(x.shape, bool)
    zeros = np.zeros(x.shape)
    while th < RELAX_THMAX and active.any():
        A, B = force_AB(x, sig)
        F = A - B * x
        newly = active & (np.abs(F) < F_TOL)
        th_settle[newly] = th
        active &= ~newly
        x_new = _guarded_step(x, zeros, sig, 0.0, h, attractor=True,
                              gate=np.ones(x.shape, bool))
        x = np.where(active, x_new, x)
        th += h
        h *= RELAX_GROW
        ths.append(th)
        traj.append(x.copy())
    return RelaxTable(th=np.asarray(ths), x_traj=np.asarray(traj),
                      th_settle=th_settle)


# ── exposure-phase integration ────────────────────────────────────────────

def _total_force(x: np.ndarray, K: np.ndarray, sig: np.ndarray, beta: float,
                 gate: np.ndarray) -> np.ndarray:
    A, B = force_AB(x, sig)
    F = np.where(gate, A - B * x, 0.0)
    if beta > 0.0:
        F = F + beta * (K - x)
    return F


def _guarded_step(x: np.ndarray, K: np.ndarray, sig: np.ndarray, beta: float,
                  h: float, *, attractor: bool, gate: np.ndarray) -> np.ndarray:
    """Sign-guarded exponential-Euler step (dt-robust, barrier-safe).

    Take the frozen-weight exponential-Euler step without a displacement cap.
    If the *true* total force changes sign between x and the candidate, the
    continuous trajectory would have stopped at the intervening equilibrium:
    bisect the force zero on that subset instead. In the stiff limit this is
    quasi-static equilibrium tracking; in the smooth limit it is plain
    exponential Euler. It cannot jump across an unvisited barrier.
    """
    A, B = force_AB(x, sig)
    gA = np.where(gate, A, 0.0)
    gB = np.where(gate, B, 0.0)
    denom = gB + beta
    F0 = gA - gB * x + beta * (K - x)
    safe = denom > 1e-300
    xbar = np.where(safe, (gA + beta * K) / np.where(safe, denom, 1.0), x)
    x_cand = xbar + (x - xbar) * np.exp(-denom * h)
    F1 = _total_force(x_cand, K, sig, beta, gate)
    crossed = (F0 * F1 < 0.0) & (np.abs(x_cand - x) > 1e-12)
    idx = np.nonzero(crossed)[0]
    if idx.size:
        lo, hi = x[idx].copy(), x_cand[idx].copy()
        s0 = np.sign(F0[idx])
        Ki, gi = K[idx], gate[idx]
        sigi = sig
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            Fm = _total_force(mid, Ki, sigi, beta, gi)
            same = np.sign(Fm) == s0
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        x_cand[idx] = 0.5 * (lo + hi)
    return x_cand


@dataclass
class OffsetStates:
    """Per (duration, n, path): state and count at stimulus offset."""
    x: np.ndarray        # (5, 15, M)
    K: np.ndarray        # (5, 15, M) int
    m_paths: int


def integrate_exposure(kind: str, rho: float, gamma: float, tau_x: float,
                       beta: float, sig: np.ndarray, m_paths: int,
                       dt: float = DT_EXPO, seed: int = SOBOL_SEED,
                       delta: float = 0.0,
                       arrivals: np.ndarray | None = None,
                       record: bool = False) -> OffsetStates:
    """Integrate the driven dynamics to 640 ms, snapshotting at each offset.

    kind: 'tether' (M2 primary), 'pulse' (M2 robustness, x += dK), 'static'
    (M3: tether only, no attractor force). All use the same arrivals.
    delta: global encoding-onset delay (s); arrivals are shifted by +delta,
    i.e. d_effective = max(0, d - delta).
    arrivals: optional externally specified arrival times, shape
    (n_conditions, m_paths, N_MAX) with +inf padding (e.g. auditory pulse
    trains or Stage-9 history manipulations); overrides the sampled process.
    record: also store the full x(t) trajectory (attr .x_t, (steps+1, N)).
    """
    E = exp_draws(m_paths, seed)
    if arrivals is not None:
        arr = np.asarray(arrivals, float)
    else:
        arr = np.stack([arrival_times(n, rho, gamma, E) + delta
                        for n in N_SHOWN])                # (15,M,15)
    arr_flat = arr.reshape(-1, N_MAX)                 # (15*M, 15)
    n_states = arr_flat.shape[0]
    x = np.zeros(n_states)
    n_steps = int(round(DUR_S[-1] / dt))
    snap_steps = {int(round(ds / dt)): i for i, ds in enumerate(DUR_S)}
    xs = np.zeros((len(DUR_S), n_states))
    Ks = np.zeros((len(DUR_S), n_states), dtype=int)
    K = np.zeros(n_states, dtype=int)
    h = dt / tau_x                                    # dimensionless step
    x_t = [x.copy()] if record else None
    K_mid_prev = np.zeros(n_states, dtype=int)
    for s in range(1, n_steps + 1):
        t = s * dt
        # midpoint count for the dynamics (unbiased +-dt/2 timing jitter);
        # exact count at t for the offset snapshots / K(d)=0 rule.
        K = (arr_flat <= (t - 0.5 * dt)).sum(axis=1)
        if kind == "pulse":
            x = _guarded_step(x, K_mid_prev, sig, 0.0, h,
                              attractor=True, gate=K_mid_prev > 0)
            x = x + (K - K_mid_prev)                   # unit pulses (a = 1)
        elif kind == "tether":
            x = _guarded_step(x, K, sig, beta, h,
                              attractor=True, gate=K > 0)
        elif kind == "static":
            # linear ODE: exponential Euler is exact for constant K
            x = K + (x - K.astype(float)) * np.exp(-beta * h)
        else:
            raise ValueError(kind)
        K_mid_prev = K
        if record:
            x_t.append(x.copy())
        if s in snap_steps:
            i = snap_steps[s]
            xs[i] = x
            Ks[i] = (arr_flat <= t).sum(axis=1)
    n_cond = arr.shape[0]
    out = OffsetStates(x=xs.reshape(len(DUR_S), n_cond, m_paths),
                       K=Ks.reshape(len(DUR_S), n_cond, m_paths),
                       m_paths=m_paths)
    if record:
        out.x_t = np.asarray(x_t)
        out.arrivals = arr
    return out


# ── model evaluation: response probabilities per (d, n) ───────────────────

@dataclass
class ModelEval:
    """p_model over the 19 wells + P(guess) per (duration, n), pre-lapse,
    plus RT ingredients."""
    p_wells: np.ndarray      # (5, 15, 19)
    p_guess: np.ndarray      # (5, 15)  probability of the K=0 uniform guess
    e_trelax: np.ndarray     # (5, 15)  E[post-offset settle time] (s)
    e_entropy: np.ndarray    # (5, 15)  E[readout entropy at x_dec]
    x_dec_mean: np.ndarray   # (5, 15)
    frac_capped: float


def evaluate_model(model: str, theta: dict, m_paths: int = 1024,
                   dt: float = DT_EXPO, seed: int = SOBOL_SEED,
                   keep_paths: bool = False) -> ModelEval:
    """theta keys: sigma_c, p, rho, gamma, tau_x, beta (as applicable)."""
    p = 0.0 if model == "M2_tether_const" else theta["p"]
    sig = sigma_wells(theta["sigma_c"], p)
    rho, gamma = float(theta["rho"]), float(theta.get("gamma", 0.0))
    tau_x = float(theta.get("tau_x", 0.25))
    beta = float(theta.get("beta", 1.0))
    delta = float(theta.get("delta", 0.0))

    if model == "M1_offset":
        # count-only: x(d) = K(d); no exposure-phase dynamics
        E = exp_draws(m_paths, seed)
        arr = np.stack([arrival_times(n, rho, gamma, E) + delta
                        for n in N_SHOWN])
        K_d = (arr[None, :, :, :] <= DUR_S[:, None, None, None]).sum(axis=-1)
        x_off = K_d.astype(float)
    else:
        kind = {"M2_tether": "tether", "M2_tether_const": "tether",
                "M2_pulse": "pulse",
                "M3_static": "static"}[model]
        off = integrate_exposure(kind, rho, gamma, tau_x, beta, sig,
                                 m_paths, dt=dt, seed=seed, delta=delta)
        x_off, K_d = off.x, off.K

    flat_x = x_off.reshape(-1)
    flat_K = K_d.reshape(-1)
    if model == "M3_static":
        x_dec = flat_x
        t_dec = np.zeros_like(flat_x)
        frac_capped = 0.0
    else:
        table = build_relax_table(sig)
        x_dec, t_dec = table.decide(flat_x, tau_x, sig)
        frac_capped = float(np.mean(t_dec >= T_CAP - 1e-9))

    probs = readout_probs(x_dec, sig)                       # (N, 19)
    H = -(probs * np.log(probs + 1e-12)).sum(axis=-1)
    guess = flat_K == 0
    probs[guess] = 0.0                                      # replaced by uniform
    sh = (len(DUR_S), N_SHOWN.size, -1)
    p_wells = probs.reshape(len(DUR_S), N_SHOWN.size, -1,
                            probs.shape[-1]).mean(axis=2)
    p_guess = guess.reshape(sh).mean(axis=2)
    denom = np.maximum(1e-12, 1.0 - p_guess)
    with np.errstate(invalid="ignore"):
        x_dec_mean = np.nanmean(np.where(guess, np.nan, x_dec).reshape(sh), axis=2)
    ev = ModelEval(
        p_wells=p_wells,
        p_guess=p_guess,
        e_trelax=np.where(guess, 0.0, t_dec).reshape(sh).mean(axis=2) / denom,
        e_entropy=np.where(guess, 0.0, H).reshape(sh).mean(axis=2) / denom,
        x_dec_mean=x_dec_mean,
        frac_capped=frac_capped,
    )
    if keep_paths:
        ev.paths = {"x_dec": x_dec.reshape(sh), "t_dec": t_dec.reshape(sh),
                    "K_d": K_d, "x_off": x_off, "H": H.reshape(sh)}
    return ev


# ── likelihood with profiled per-duration lapse ───────────────────────────

def p_final_full(p_wells_d: np.ndarray, p_guess_d: np.ndarray, lapse: float,
                 r_max: int) -> np.ndarray:
    """(15, r_max) final response distribution for one duration.

    Guess paths (K(d)=0) respond uniformly over 1..r_max, like the lapse."""
    n_n = p_wells_d.shape[0]
    unif = (lapse + (1.0 - lapse) * p_guess_d)[:, None] / r_max
    pf = np.tile(unif, (1, r_max))
    pf[:, :WELLS.size] += (1.0 - lapse) * p_wells_d
    return pf


def loglik_duration(p_wells_d: np.ndarray, p_guess_d: np.ndarray,
                    counts: np.ndarray, lapse: float, r_max: int) -> float:
    pf = p_final_full(p_wells_d, p_guess_d, lapse, r_max)
    if not np.all(np.isfinite(pf)) or np.any(pf <= 0):
        return -np.inf
    return float(np.sum(counts * np.log(pf)))


def profile_lapse(p_wells_d: np.ndarray, p_guess_d: np.ndarray,
                  counts: np.ndarray, r_max: int) -> tuple[float, float]:
    """Golden-section maximisation of the concave lapse profile."""
    lo, hi = 1e-4, 0.5
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc = loglik_duration(p_wells_d, p_guess_d, counts, c, r_max)
    fd = loglik_duration(p_wells_d, p_guess_d, counts, d, r_max)
    for _ in range(40):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = loglik_duration(p_wells_d, p_guess_d, counts, c, r_max)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = loglik_duration(p_wells_d, p_guess_d, counts, d, r_max)
    lapse = (a + b) / 2.0
    return lapse, loglik_duration(p_wells_d, p_guess_d, counts, lapse, r_max)


def total_loglik(model: str, theta: dict, counts_by_dur: dict[int, np.ndarray],
                 r_max: int, m_paths: int = 1024, dt: float = DT_EXPO,
                 seed: int = SOBOL_SEED) -> tuple[float, dict[int, float]]:
    """Summed logL over durations with the 5 lapses profiled out."""
    ev = evaluate_model(model, theta, m_paths=m_paths, dt=dt, seed=seed)
    tot = 0.0
    lapses: dict[int, float] = {}
    for i, d in enumerate(DURATIONS):
        lapse, ll = profile_lapse(ev.p_wells[i], ev.p_guess[i],
                                  counts_by_dur[d], r_max)
        lapses[d] = lapse
        tot += ll
    return tot, lapses
