"""Round 2: is the failure the CAPPED COUNT K(t), or online accumulation itself?

Three evidence families, each in four matched variants, all sharing the
registered landscape (sigma_m = sigma_c m^p on wells 1..19), the flat softmax
readout, the per-duration uniform lapse over 1..99, the count-table
categorical likelihood, and -- crucially -- the *same clean-up operator* as
the published model (``scripts.trial_level_model_comparison._cleanup``), so
that "attractor on/off" means exactly what it means in the manuscript.

Evidence families (the drive u(t) that the state follows)
--------------------------------------------------------
ddm     Continuous noisy accumulation. Evidence integral S(t) = n t +
        sqrt(q(n)) W(t), q(n) = s0 n^lam; the running estimate is
        u(t) = S(t)/t + s00 * xi, with xi ~ N(0,1) fixed per trial (a static
        encoding noise, so the offset marginal keeps M0's noise floor).
        => u(d) ~ N(n, s00^2 + q(n)/d): the mean is n and the precision grows
        with exposure as 1/d, the ideal-observer law. 2 parameters (s0, lam)
        for ALL five durations, versus M0's ten (s0_d, lambda0_d).

pulse   Imperfect item pulses. Each item is registered with probability
        pi_hit at an exponential latency (rate rho) and contributes a random
        amplitude a ~ Gamma(mean 1, CV cv_a); false positives arrive as a
        Poisson process of rate phi with the same amplitude law. u(t) is the
        cumulative sum. No exact cap at n, no unit amplitudes.
        4 parameters (rho, pi_hit, cv_a, phi).

kalman  Sequential Bayesian updating. Prior N(mu0, v0) on the numerosity,
        observations dz = n dt + sqrt(q(n)) dW with q(n) = s0 n^lam; u(t) is
        the posterior mean, whose precision 1/v0 + t/q(n) grows with exposure
        and which is shrunk toward mu0 (so short exposures underestimate).
        4 parameters (s0, lam, mu0, v0). Nests ddm as v0 -> infinity.

Variants (the 2x2 the question needs)
-------------------------------------
batch_static    x = u(d) at offset, static readout (no clean-up).
batch_attr      x = u(d) at offset, then the M0 clean-up. For ddm this is
                EXACTLY M0 with a 2-parameter sensory channel.
online_static   tau dx/dt = beta (u(t) - x) during exposure, no landscape
                force, readout at x(d). Identified parameter: r = beta/tau.
online_attr     tau dx/dt = -dV/dx + beta (u(t) - x) during exposure, then
                the M0 clean-up after offset. Nests batch_attr as beta -> inf.

So: batch_attr vs batch_static  = the manuscript's clean-up claim, new channel
    online_* vs batch_*         = does ONLINE delivery buy anything
    online_attr vs online_static = does the attractor buy anything when driven

Numerics: Gaussian families' batch variants use the same deterministic grid
quadrature as the registered analysis (no Monte-Carlo noise). Everything with
a path (pulse batch, all online variants) uses fixed common random numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

DURATIONS = M.DURATIONS
DUR_S = M.DUR_S
N_SHOWN = M.N_SHOWN
N_MAX = int(N_SHOWN.max())
S00 = T.S00                     # 0.05 noise floor, M0 convention

DT = 0.0025                     # exposure step (s)
SEED2 = 20260712
U_CLIP = (-5.0, 60.0)           # guard on the running estimate at tiny t

FAMILIES = ("ddm", "pulse", "kalman")
VARIANTS = ("batch_static", "batch_attr", "online_static", "online_attr")

FAMILY_PARAMS = {
    "ddm": ("s0", "lam"),
    "pulse": ("rho", "pi_hit", "cv_a", "phi"),
    "kalman": ("s0", "lam", "mu0", "v0"),
}
VARIANT_PARAMS = {
    "batch_static": (),
    "batch_attr": (),
    "online_static": ("r_tether",),
    "online_attr": ("tau_x", "beta"),
}
LANDSCAPE_PARAMS = ("sigma_c", "p")


def param_names(family: str, variant: str) -> tuple[str, ...]:
    return LANDSCAPE_PARAMS + FAMILY_PARAMS[family] + VARIANT_PARAMS[variant]


def n_params(family: str, variant: str) -> int:
    return len(param_names(family, variant)) + len(DURATIONS)   # + 5 lapses


# ── common random numbers ────────────────────────────────────────────────

_CRN: dict = {}


def crn(m_paths: int, n_steps: int, seed: int = SEED2) -> dict:
    """Fixed random draws reused by every model and every evaluation.

    W       (n_steps, m) Brownian motion at the step midpoints (dt spacing)
    xi      (m,)         static encoding noise
    e_item  (m, N_MAX)   Exp(1) latencies for item registration
    u_item  (m, N_MAX)   uniforms for the hit/miss draw
    g_item  (m, N_MAX)   Gamma(1,1) draws for amplitudes (rescaled by cv)
    fp_t    (m, N_FP)    false-positive arrival times (unit-rate Poisson)
    fp_g    (m, N_FP)    Gamma(1,1) draws for false-positive amplitudes
    """
    key = (m_paths, n_steps, seed)
    if key in _CRN:
        return _CRN[key]
    rng = np.random.default_rng(seed)
    half = m_paths // 2
    z = rng.standard_normal((n_steps, half))
    z = np.concatenate([z, -z], axis=1)                     # antithetic
    W = np.cumsum(z * np.sqrt(DT), axis=0)
    xi = rng.standard_normal(half)
    xi = np.concatenate([xi, -xi])
    n_fp = 40
    out = {
        "W": W,
        "xi": xi,
        "e_item": rng.exponential(size=(m_paths, N_MAX)),
        "u_item": rng.random((m_paths, N_MAX)),
        "g_item": rng.standard_gamma(1.0, size=(m_paths, N_MAX)),
        "fp_e": rng.exponential(size=(m_paths, n_fp)),
        "fp_g": rng.standard_gamma(1.0, size=(m_paths, n_fp)),
        "m": m_paths,
        "n_steps": n_steps,
    }
    _CRN[key] = out
    return out


def _gamma_amp(g: np.ndarray, cv: float) -> np.ndarray:
    """Mean-1 amplitudes with coefficient of variation cv, from Gamma(1,1)
    draws via a shape-matched transform (deterministic in the CRN)."""
    if cv <= 1e-6:
        return np.ones_like(g)
    k = 1.0 / (cv * cv)
    # Wilson-Hilferty: map Gamma(1,1) -> Gamma(k,1) monotonically through the
    # normal approximation, then scale to mean 1. Monotone in g, so the CRN
    # coupling is preserved.
    z = (g ** (1.0 / 3.0) - (1 - 1.0 / 9.0)) / np.sqrt(1.0 / 9.0)
    gk = k * (1 - 1.0 / (9 * k) + z * np.sqrt(1.0 / (9 * k))) ** 3
    return np.maximum(1e-6, gk / k)


# ── offset (batch) distributions ─────────────────────────────────────────

def gaussian_offset(family: str, theta: dict, d: float) -> tuple[np.ndarray, np.ndarray]:
    """(mean, sd) of u(d) per numerosity for the two Gaussian families."""
    n = N_SHOWN.astype(float)
    q = np.maximum(1e-9, theta["s0"] * np.power(n, theta["lam"]))
    if family == "ddm":
        return n, np.sqrt(S00 ** 2 + q / d)
    if family == "kalman":
        v0, mu0 = float(theta["v0"]), float(theta["mu0"])
        prec = 1.0 / v0 + d / q
        mean = (mu0 / v0 + n * d / q) / prec
        var = (d / q) / prec ** 2 + S00 ** 2
        return mean, np.sqrt(var)
    raise ValueError(family)


def pulse_offset_samples(theta: dict, d: float, C: dict) -> np.ndarray:
    """(15, m) samples of u(d) for the imperfect-pulse family."""
    rho, pi_hit = float(theta["rho"]), float(theta["pi_hit"])
    cv, phi = float(theta["cv_a"]), float(theta["phi"])
    m = C["m"]
    lat = C["e_item"] / rho                                  # (m, N_MAX)
    hit = C["u_item"] < pi_hit
    amp = _gamma_amp(C["g_item"], cv)
    idx = np.arange(N_MAX)[None, :]
    reg = (lat <= d) & hit                                   # (m, N_MAX)
    fp_t = np.cumsum(C["fp_e"], axis=1) / max(phi, 1e-9)     # (m, n_fp)
    fp_amp = _gamma_amp(C["fp_g"], cv)
    fp_sum = np.where(fp_t <= d, fp_amp, 0.0).sum(axis=1)    # (m,)
    out = np.empty((N_SHOWN.size, m))
    for i, n in enumerate(N_SHOWN):
        sel = reg & (idx < n)
        out[i] = (np.where(sel, amp, 0.0).sum(axis=1) + fp_sum)
    return out


# ── online drives ────────────────────────────────────────────────────────

class Drive:
    """u(t) evaluated one step at a time (never materialised over all steps).

    ``at(s)`` returns the (15, m) drive at the midpoint of step s (1-based).
    """

    def __init__(self, family: str, theta: dict, C: dict):
        self.family, self.C = family, C
        self.n = N_SHOWN.astype(float)
        if family in ("ddm", "kalman"):
            self.q = np.maximum(1e-9, theta["s0"] *
                                np.power(self.n, theta["lam"]))    # (15,)
            self.sq = np.sqrt(self.q)
            self.static = S00 * C["xi"]                            # (m,)
            if family == "kalman":
                self.v0, self.mu0 = float(theta["v0"]), float(theta["mu0"])
        elif family == "pulse":
            rho, pi_hit = float(theta["rho"]), float(theta["pi_hit"])
            cv, phi = float(theta["cv_a"]), float(theta["phi"])
            self.lat = C["e_item"] / rho                           # (m, N_MAX)
            amp = np.where(C["u_item"] < pi_hit,
                           _gamma_amp(C["g_item"], cv), 0.0)
            self.amp = amp
            self.fp_t = np.cumsum(C["fp_e"], axis=1) / max(phi, 1e-9)
            self.fp_amp = _gamma_amp(C["fp_g"], cv)
        else:
            raise ValueError(family)

    def at(self, s: int) -> np.ndarray:
        t = (s - 0.5) * DT
        C = self.C
        if self.family in ("ddm", "kalman"):
            W = C["W"][s - 1]                                      # (m,)
            z = self.n[:, None] * t + self.sq[:, None] * W[None, :]  # (15, m)
            if self.family == "ddm":
                u = z / t + self.static[None, :]
            else:
                prec = 1.0 / self.v0 + t / self.q[:, None]
                u = (self.mu0 / self.v0 + z / self.q[:, None]) / prec \
                    + self.static[None, :]
        else:
            mask = self.lat <= t                                   # (m, N_MAX)
            csum = np.cumsum(np.where(mask, self.amp, 0.0), axis=1)  # (m, N_MAX)
            fp = np.where(self.fp_t <= t, self.fp_amp, 0.0).sum(axis=1)  # (m,)
            u = csum[:, N_SHOWN - 1].T + fp[None, :]               # (15, m)
        return np.clip(u, *U_CLIP)


# ── evaluation: p(r | n, d) per model ────────────────────────────────────

def _readout(x: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return M.readout_probs(x, sig)


def _cleanup_m0(x: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """The published clean-up operator (clamped iteration, 120 steps)."""
    x_star, _, _ = T._cleanup(x, family="gaussian", sigma=sig,
                              max_steps=T.MAX_STEPS)
    return x_star


CHUNK = 4096          # paths per block: bounds peak memory, no effect on results


def evaluate(family: str, variant: str, theta: dict, *, m_paths: int = 4096,
             quad_nodes: int = 401, seed: int = SEED2,
             keep: bool = False) -> dict:
    """Returns p_wells (5, 15, 19) and p_guess (5, 15) pre-lapse.

    Path-based variants are evaluated in blocks of CHUNK paths (identical
    results, bounded memory)."""
    if m_paths > CHUNK and not variant.startswith("batch") and not keep:
        n_blocks = int(np.ceil(m_paths / CHUNK))
        acc_w = np.zeros((len(DURATIONS), N_SHOWN.size, M.WELLS.size))
        acc_g = np.zeros((len(DURATIONS), N_SHOWN.size))
        for b in range(n_blocks):
            ev = evaluate(family, variant, theta, m_paths=CHUNK,
                          quad_nodes=quad_nodes, seed=seed + 977 * b)
            acc_w += ev["p_wells"]
            acc_g += ev["p_guess"]
        return {"p_wells": acc_w / n_blocks, "p_guess": acc_g / n_blocks}

    sig = M.sigma_wells(theta["sigma_c"], theta["p"])
    n_n = N_SHOWN.size
    p_wells = np.zeros((len(DURATIONS), n_n, M.WELLS.size))
    p_guess = np.zeros((len(DURATIONS), n_n))
    extras: dict = {}

    if variant.startswith("batch"):
        do_clean = variant == "batch_attr"
        if family in ("ddm", "kalman"):
            quad = T.make_quadrature(quad_nodes, kind="grid")
            zz = np.sqrt(2.0) * quad.t                        # standard normal
            w = quad.w
            for di, d in enumerate(DUR_S):
                mu, sd = gaussian_offset(family, theta, float(d))
                x0 = (mu[:, None] + sd[:, None] * zz[None, :]).reshape(-1)
                x = _cleanup_m0(x0, sig) if do_clean else x0
                pr = _readout(x, sig).reshape(n_n, zz.size, -1)
                p_wells[di] = (pr * w[None, :, None]).sum(axis=1)
        else:
            n_blocks = max(1, int(np.ceil(m_paths / CHUNK)))
            for b in range(n_blocks):
                C = crn(min(m_paths, CHUNK), 1, seed + 977 * b)
                for di, d in enumerate(DUR_S):
                    u = pulse_offset_samples(theta, float(d), C)   # (15, m)
                    zero = u <= 0.0
                    x = (_cleanup_m0(u.reshape(-1), sig) if do_clean
                         else u.reshape(-1))
                    pr = _readout(x, sig).reshape(n_n, C["m"], -1)
                    pr[zero] = 0.0
                    p_wells[di] += pr.mean(axis=1) / n_blocks
                    p_guess[di] += zero.mean(axis=1) / n_blocks
        return {"p_wells": p_wells, "p_guess": p_guess, **extras}

    # ── online variants ──────────────────────────────────────────────────
    n_steps = int(round(DUR_S[-1] / DT))
    C = crn(m_paths, n_steps, seed)
    drive = Drive(family, theta, C)
    m = C["m"]
    x = np.zeros((n_n, m))
    snap = {int(round(float(d) / DT)): i for i, d in enumerate(DUR_S)}
    xs = np.zeros((len(DURATIONS), n_n, m))
    u_snap = np.zeros((len(DURATIONS), n_n, m)) if keep else None
    if variant == "online_static":
        r_t = float(theta["r_tether"])
        decay = np.exp(-r_t * DT)
        for s in range(1, n_steps + 1):
            u = drive.at(s)
            x = u + (x - u) * decay                            # exact for linear
            if s in snap:
                xs[snap[s]] = x
                if keep:
                    u_snap[snap[s]] = u
    else:                                                      # online_attr
        tau_x, beta = float(theta["tau_x"]), float(theta["beta"])
        h = DT / tau_x
        gate = np.ones(n_n * m, bool)
        for s in range(1, n_steps + 1):
            u = drive.at(s)
            x = M._guarded_step(x.reshape(-1), u.reshape(-1), sig, beta, h,
                                attractor=True, gate=gate).reshape(n_n, m)
            if s in snap:
                xs[snap[s]] = x
                if keep:
                    u_snap[snap[s]] = u
    if keep:
        extras["x_off"] = xs.copy()
        extras["u_off"] = u_snap
        extras["drive"] = drive
    for di in range(len(DURATIONS)):
        x_off = xs[di].reshape(-1)
        x_fin = _cleanup_m0(x_off, sig) if variant == "online_attr" else x_off
        pr = _readout(x_fin, sig).reshape(n_n, m, -1)
        p_wells[di] = pr.mean(axis=1)
        if keep:
            extras.setdefault("x_dec", []).append(x_fin.reshape(n_n, m))
    return {"p_wells": p_wells, "p_guess": p_guess, **extras}


def total_loglik(family: str, variant: str, theta: dict, counts_by_dur: dict,
                 r_max: int, *, m_paths: int = 4096, quad_nodes: int = 401,
                 seed: int = SEED2) -> tuple[float, dict]:
    ev = evaluate(family, variant, theta, m_paths=m_paths,
                  quad_nodes=quad_nodes, seed=seed)
    tot, lapses = 0.0, {}
    for i, d in enumerate(DURATIONS):
        lapse, ll = M.profile_lapse(ev["p_wells"][i], ev["p_guess"][i],
                                    counts_by_dur[d], r_max)
        lapses[d] = lapse
        tot += ll
    return tot, lapses
