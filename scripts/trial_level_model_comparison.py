"""Trial-level model comparison for the Cheyette & Piantadosi duration data.

This is the reviewer-requested replacement for the summary-curve AIC/BIC
analysis (reviewer major point 1). Instead of fitting three summary curves
(mean guess, percent exact, median RT) with a profile-Gaussian residual
likelihood, every model is scored here against the *discrete, trial-level*
integer responses and the *trial-level* RTs.

For each model and exposure duration ``d`` we define a deterministic response
likelihood

    p_theta(r | n, d) = integral  p_theta(r | C_theta(y0)) * p(y0 | n, d)  dy0

where

  * ``y0 ~ Normal(n, sqrt(0.05^2 + s0 * n^lambda0))`` is the sensory evidence
    (the same ``evidence_noise_std`` convention as ``scripts/fit_helpers``),
  * ``C_theta(y0)`` is the attractor cleanup result ``x*`` for the dynamic
    models, or the identity ``y0`` for the no-cleanup readout models,
  * ``p_theta(r | x*)`` is the softmax readout over the integer wells.

The integral is evaluated with **deterministic quadrature** (no Monte-Carlo
noise), validated against a high-sample Monte-Carlo estimate. Gauss-Hermite is
exact for the smooth no-clean-up readouts but poor for the attractor (clean-up
makes the integrand near-step, ~1.7-4% error even at 241 nodes), so a robust
uniform standard-normal grid is used for every model -- see
``docs/trial_level_model_comparison_notes.md``.

A lapse / contamination component handles typed responses that fall outside the
model's integer-well support (~5 % of trials reach values up to 99):

    p_final(r | n, d) = (1 - lapse) * p_model(r | n, d)
                        + lapse * Uniform(1 .. R_max)

so no observed response ever receives zero probability.

Channels
--------
  * Response-only  : categorical log-likelihood over the (d, n, r) count table
                     -- this is the *main* model comparison.
  * RT-only        : log(rt_s) ~ Normal(log(t0 + metric(n, d)), sigma_rt),
                     with metric = entropy, convergence steps, or hybrid.
  * Joint          : couples response and RT through the shared latent y0
                     (the node-posterior given the observed response), so the
                     attractor's convergence-time distribution feeds RT.

Run from the project root::

    .venv/bin/python -m scripts.trial_level_model_comparison            # full
    .venv/bin/python -m scripts.trial_level_model_comparison --smoke    # fast

Outputs land in ``output/trial_level_model_comparison/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import differential_evolution, minimize
from scipy.special import digamma, gammaln, logsumexp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.yaml"
DATA_PATH = ROOT / "data" / "data_piantadosi.csv"
OUT_DIR = ROOT / "output" / "trial_level_model_comparison"

DURATIONS: tuple[int, ...] = (40, 80, 160, 320, 640)

with open(CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

# Integer-well support: N_TRUE_MIN .. N_SUPPORT_MAX (= n_true_max + padding).
N_TRUE_MIN: int = int(_CFG["n_true_min"])              # 1
N_TRUE_MAX: int = int(_CFG["n_true_max"])              # 15
N_SUPPORT_MAX: int = N_TRUE_MAX + int(_CFG["support_padding"])  # 19

T0: float = 1.30          # task-fixed base RT (s), Piantadosi convention
S00: float = 0.05         # baseline sensory spread (FIXED["s00"] in fit_helpers)
ETA: float = 1.0          # cleanup step size (fixed_eta)
KAPPA_LSE: float = 1.0    # LSE coupling for the Gaussian pull (fixed_kappa)
GAUSS_BETA: float = 0.5   # Benford well-width exponent (fixed_beta)
MAX_STEPS: int = 100
TOL: float = 1e-4

# Integer wells and Benford frequencies.
WELLS: np.ndarray = np.arange(N_TRUE_MIN, N_SUPPORT_MAX + 1, dtype=float)  # 1..19
N_WELLS: int = WELLS.size
FREQ: np.ndarray = np.log10(1.0 + 1.0 / WELLS)
LOG_WELLS: np.ndarray = np.log(WELLS)
N_SHOWN: np.ndarray = np.arange(N_TRUE_MIN, N_TRUE_MAX + 1)  # 1..15 (true numerosities)

LAPSE_BOUNDS: tuple[float, float] = (1e-4, 0.5)
SIGMA_RT_BOUNDS: tuple[float, float] = (0.05, 2.0)

SEED = 73


# ── Gauss-Hermite quadrature ─────────────────────────────────────────────


@dataclass(frozen=True)
class Quadrature:
    """Probabilists' nodes/weights for ``E[f] = integral f(y) N(y; mu, sd) dy``.

    With physicists' Gauss-Hermite nodes ``t`` and weights ``g`` (weight
    function ``exp(-t^2)``), the substitution ``y = mu + sqrt(2) * sd * t``
    gives ``E[f] ~= sum_k w_k f(y_k)`` with ``w_k = g_k / sqrt(pi)`` summing
    to one.
    """

    t: np.ndarray
    w: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.t.size


def make_quadrature(
    n_nodes: int = 61, kind: str = "hermite", zmax: float = 7.0
) -> Quadrature:
    """Deterministic nodes/weights for ``E_{y0~N(mu,sd)}[f(y0)]``.

    ``kind="hermite"`` uses Gauss-Hermite (spectral accuracy for *smooth*
    integrands -- ideal for the no-clean-up readouts). ``kind="grid"`` uses a
    uniform standard-normal grid with Gaussian weights; it is only first-order
    but, unlike Gauss-Hermite, is robust to the near-step integrand produced by
    the attractor clean-up (``y0`` snaps to basins), so it integrates every
    model on the same footing. See ``docs/trial_level_model_comparison_notes``.
    """
    if kind == "hermite":
        t, g = hermgauss(n_nodes)
        return Quadrature(t=t.astype(float), w=(g / np.sqrt(np.pi)).astype(float))
    if kind == "grid":
        z = np.linspace(-zmax, zmax, n_nodes)
        w = np.exp(-0.5 * z * z)
        w = w / w.sum()
        # y = mu + sqrt(2)*sd*t == mu + sd*z  ->  t = z / sqrt(2)
        return Quadrature(t=(z / np.sqrt(2.0)).astype(float), w=w.astype(float))
    raise ValueError(f"unknown quadrature kind {kind!r}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


def evidence_sd(n: np.ndarray | float, s0: float, lambda0: float) -> np.ndarray:
    """``sqrt(0.05^2 + s0 * n^lambda0)`` -- matches ``fh.evidence_noise_std``."""
    n_arr = np.asarray(n, dtype=float)
    return np.sqrt(S00 * S00 + s0 * np.power(n_arr, lambda0))


# ── Well potentials and cleanup dynamics (numpy mirror of fit_helpers) ────


def _poisson_logwells(x: np.ndarray, c: float) -> np.ndarray:
    """Scaled-Poisson log-well values ``f_n(x)`` for each well, shape (B, K)."""
    x_s = np.clip(x[:, None] / c, 0.1, None)
    n_s = np.clip(WELLS[None, :] / c, 0.1, None)
    return x_s * np.log(n_s) - n_s - gammaln(x_s + 1.0)


def _poisson_pull(x: np.ndarray, c: float, weights: np.ndarray) -> np.ndarray:
    x_s = np.clip(x / c, 0.1, None)
    n_s = np.clip(WELLS / c, 0.1, None)
    weighted_log_n = (weights * np.log(n_s)[None, :]).sum(axis=1)
    return (digamma(x_s + 1.0) - weighted_log_n) / c


def _poisson_scale_cn(sigma_c: float, qvar: float) -> np.ndarray:
    """Per-well Poisson scale giving Var(x | well n) = sigma_c * n^qvar.

    The scaled Poisson read-out has variance ``c_n * n`` in x-space, so setting
    ``c_n = sigma_c * n^(qvar-1)`` makes the variance grow as ``n^qvar``. At
    ``qvar = 1`` this is the standard Poisson well (Var ∝ n, σ ∝ √n); freeing
    ``qvar`` lets the *Poisson-shaped* wells reach the same super-linear width
    the free-width Gaussian read-outs prefer (matching their parameter count).
    """
    return np.maximum(1e-6, float(sigma_c) * np.power(WELLS, float(qvar) - 1.0))


def _poisson_logwells_cn(x: np.ndarray, c_n: np.ndarray) -> np.ndarray:
    """Scaled-Poisson log-well values with a per-well scale ``c_n`` (shape K)."""
    x_s = np.clip(x[:, None] / c_n[None, :], 0.1, None)
    n_s = np.clip(WELLS[None, :] / c_n[None, :], 0.1, None)
    return x_s * np.log(n_s) - n_s - gammaln(x_s + 1.0)


def _poisson_pull_cn(x: np.ndarray, c_n: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Descent pull for the per-well-scale Poisson landscape (reduces to the
    scalar ``_poisson_pull`` when ``c_n`` is constant)."""
    x_s = np.clip(x[:, None] / c_n[None, :], 0.1, None)        # (B, K)
    n_s = np.clip(WELLS[None, :] / c_n[None, :], 0.1, None)    # (1, K)
    grad = (digamma(x_s + 1.0) - np.log(n_s)) / c_n[None, :]   # (B, K)
    return (weights * grad).sum(axis=1)


def _gauss_sigma(sigma_c: float, beta: float = GAUSS_BETA) -> np.ndarray:
    return np.power(sigma_c / FREQ, beta)


def _gaussian_well_sigma(family: str, theta: dict) -> np.ndarray:
    """Gaussian readout well widths per well, by family.

    * ``gaussian``          -- Benford widths with fixed beta (= GAUSS_BETA),
                               sigma_n = (sigma_c / freq_n)^0.5  (~ sqrt(n)).
    * ``gaussian_freebeta`` -- same Benford form but beta is a free parameter,
                               so the width exponent can move toward linear.
    * ``gaussian_pow``      -- pure power-law widths sigma_n = sigma_c * n^p,
                               which reaches Weber's linear width at p = 1.
    """
    sigma_c = float(theta["sigma_c"])
    if family == "gaussian_pow":
        return np.maximum(1e-6, sigma_c * np.power(WELLS, float(theta["pow"])))
    beta = GAUSS_BETA if family == "gaussian" else float(theta["beta"])
    return np.power(sigma_c / FREQ, beta)


def _gauss_logwells(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    diff = x[:, None] - WELLS[None, :]
    return -np.log(sigma)[None, :] - diff * diff / (2.0 * (sigma * sigma)[None, :])


def _gauss_pull(x: np.ndarray, sigma: np.ndarray, weights: np.ndarray) -> np.ndarray:
    diff = x[:, None] - WELLS[None, :]
    return KAPPA_LSE * (weights * diff / (sigma * sigma)[None, :]).sum(axis=1)


def _cleanup(
    y0: np.ndarray,
    *,
    family: str,
    c: float | None = None,
    sigma: np.ndarray | None = None,
    max_steps: int = MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient descent ``x -> x - eta * pull`` from ``x = y0`` (no anchor).

    Returns ``(x_star, n_steps, conv_time)``. ``n_steps`` is the raw iteration
    count (sensitive to the ``max_step`` clamp). ``conv_time`` is the continuous
    gradient-flow time ``integral dx / (eta * |pull|)`` accumulated along the
    path: it equals the step count when no step is clamped, discounts clamped
    (steep, fast) steps, and grows in flat regions near basin boundaries
    (critical slowing down) — a more principled "first-passage" RT than the raw
    step count. With ``max_steps == 0`` this is the identity map.
    """
    x = y0.copy()
    n_steps = np.full(x.shape, float(max_steps), dtype=float)
    conv_time = np.zeros_like(x)
    if max_steps == 0:
        return x, np.zeros_like(x), conv_time

    if family == "poisson":
        max_step = 0.5 * float(c)
    elif family == "poisson_pow":
        max_step = 0.5 * float(np.min(c))   # c is the per-well scale array c_n
    else:
        max_step = 0.5 * float(np.min(sigma))

    converged = np.zeros(x.shape, dtype=bool)
    prev_step = np.zeros_like(x)

    for t in range(max_steps):
        if family == "poisson":
            f = _poisson_logwells(x, c)
            weights = _softmax(f)
            pull = _poisson_pull(x, c, weights)
        elif family == "poisson_pow":
            f = _poisson_logwells_cn(x, c)
            weights = _softmax(f)
            pull = _poisson_pull_cn(x, c, weights)
        else:
            f = _gauss_logwells(x, sigma)
            weights = _softmax(f)
            pull = _gauss_pull(x, sigma, weights)

        step = np.clip(ETA * pull, -max_step, max_step)
        oscillating = (~converged) & (t > 0) & (step * prev_step < -1e-12)
        just = oscillating | ((~converged) & (np.abs(step) < TOL))
        n_steps = np.where(just, t + 1, n_steps)
        converged = converged | just
        # continuous flow-time for points that actually move this iteration
        dt = np.abs(step) / (ETA * np.abs(pull) + 1e-12)
        conv_time = conv_time + np.where(converged, 0.0, dt)
        x = np.where(converged, x, x - step)
        prev_step = np.where(converged, prev_step, step)
        if (t + 1) % 4 == 0 and converged.all():
            break

    return x, n_steps, conv_time


# ── Model registry ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str                 # "poisson" | "gaussian" | "weber" | "cp"
    cleanup: bool               # attractor (True) vs static readout (False)
    param_names: tuple[str, ...]    # response-channel params (excl. lapse)
    bounds: tuple[tuple[float, float], ...]
    is_proxy: bool = False

    @property
    def has_dynamics(self) -> bool:
        return self.cleanup and self.family in (
            "poisson", "gaussian", "poisson_pow", "gaussian_freebeta", "gaussian_pow"
        )


_POISSON_BOUNDS = ((0.01, 0.5), (1e-6, 0.1), (0.1, 4.0))
_GAUSS_BOUNDS = ((0.005, 0.5), (1e-6, 0.1), (0.1, 4.0))

MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="hopfield_poisson",
        label="Hopfield-Poisson (attractor)",
        family="poisson",
        cleanup=True,
        param_names=("sigma_c", "s0", "lambda0"),
        bounds=_POISSON_BOUNDS,
    ),
    ModelSpec(
        key="no_cleanup_poisson",
        label="Poisson readout (no clean-up)",
        family="poisson",
        cleanup=False,
        param_names=("sigma_c", "s0", "lambda0"),
        bounds=_POISSON_BOUNDS,
    ),
    ModelSpec(
        key="gaussian_benford_attractor",
        label="Gaussian-Benford (attractor)",
        family="gaussian",
        cleanup=True,
        param_names=("sigma_c", "s0", "lambda0"),
        bounds=_GAUSS_BOUNDS,
    ),
    ModelSpec(
        key="gaussian_benford_no_cleanup",
        label="Gaussian-Benford readout (no clean-up)",
        family="gaussian",
        cleanup=False,
        param_names=("sigma_c", "s0", "lambda0"),
        bounds=_GAUSS_BOUNDS,
    ),
    ModelSpec(
        key="weber_ans",
        label="Weber/ANS baseline",
        family="weber",
        cleanup=False,
        param_names=("sigma0", "exponent", "decision_width"),
        bounds=((0.01, 2.0), (0.0, 1.5), (0.01, 1.0)),
    ),
    ModelSpec(
        key="cp_log_prior_proxy",
        label="C&P log-prior proxy",
        family="cp",
        cleanup=False,
        param_names=("sigma_log", "alpha", "readout_temp"),
        bounds=((0.02, 2.0), (-2.5, 2.5), (0.25, 4.0)),
        is_proxy=True,
    ),
    # ── Winner of the trial-level comparison: symmetric super-linear-width
    #    Gaussian attractor (free Benford exponent beta) + its static twin.
    ModelSpec(
        key="gaussian_freebeta_attractor",
        label="Super-linear Gaussian attractor (free beta)",
        family="gaussian_freebeta",
        cleanup=True,
        param_names=("sigma_c", "s0", "lambda0", "beta"),
        bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
    ),
    ModelSpec(
        key="gaussian_freebeta_no_cleanup",
        label="Super-linear Gaussian readout (free beta, no clean-up)",
        family="gaussian_freebeta",
        cleanup=False,
        param_names=("sigma_c", "s0", "lambda0", "beta"),
        bounds=((0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0)),
    ),
)

MODEL_BY_KEY: dict[str, ModelSpec] = {m.key: m for m in MODEL_SPECS}

# The winning model of the trial-level comparison (response + dynamics RT).
WINNER_KEY = "gaussian_freebeta_attractor"

# The clean 2x2 (well family x clean-up) used for the cleanup/well-family split.
GRID_2X2: tuple[str, ...] = (
    "hopfield_poisson",
    "no_cleanup_poisson",
    "gaussian_benford_attractor",
    "gaussian_benford_no_cleanup",
)


# ── Deterministic response grid ──────────────────────────────────────────


def compute_grid(
    spec: ModelSpec,
    theta: dict,
    quad: Quadrature,
    n_values: np.ndarray = N_SHOWN,
    *,
    force_no_cleanup: bool = False,
) -> dict[str, np.ndarray]:
    """Deterministic per-numerosity readout via Gauss-Hermite quadrature.

    Returns a dict with:
      ``p_resp``      (n, N_WELLS)   integrated p_model(r | n) over the wells,
      ``EH``          (n,)          E_y0[H(p(r | C(y0)))],
      ``ET``          (n,)          E_y0[convergence steps]   (0 if no clean-up),
      ``probs_nodes`` (n, K, N_WELLS) per-node readout (for the joint channel),
      ``H_nodes``     (n, K)        per-node entropy,
      ``steps_nodes`` (n, K)        per-node convergence steps.
    """
    n_values = np.asarray(n_values, dtype=float)
    n_n = n_values.size
    k = quad.n_nodes
    sqrt2 = np.sqrt(2.0)

    if spec.family in ("poisson", "poisson_pow", "gaussian",
                       "gaussian_freebeta", "gaussian_pow"):
        sd = evidence_sd(n_values, theta["s0"], theta["lambda0"])
        y = n_values[:, None] + sqrt2 * sd[:, None] * quad.t[None, :]   # (n, K)
        y_flat = y.reshape(-1)
        do_clean = spec.cleanup and not force_no_cleanup
        if spec.family == "poisson":
            c = float(theta["sigma_c"])
            x_star, steps, ctime = _cleanup(
                y_flat, family="poisson", c=c, max_steps=MAX_STEPS if do_clean else 0
            )
            logf = _poisson_logwells(x_star, c)
        elif spec.family == "poisson_pow":
            c_n = _poisson_scale_cn(theta["sigma_c"], theta["qvar"])
            x_star, steps, ctime = _cleanup(
                y_flat, family="poisson_pow", c=c_n,
                max_steps=MAX_STEPS if do_clean else 0,
            )
            logf = _poisson_logwells_cn(x_star, c_n)
        else:
            sigma = _gaussian_well_sigma(spec.family, theta)
            x_star, steps, ctime = _cleanup(
                y_flat, family="gaussian", sigma=sigma,
                max_steps=MAX_STEPS if do_clean else 0,
            )
            logf = _gauss_logwells(x_star, sigma)
        probs = _softmax(logf)
        steps = steps.reshape(n_n, k)
        flowtime = ctime.reshape(n_n, k)
    elif spec.family == "weber":
        sd = np.maximum(1e-4, theta["sigma0"] * np.power(n_values, theta["exponent"]))
        y = n_values[:, None] + sqrt2 * sd[:, None] * quad.t[None, :]
        y_flat = y.reshape(-1)
        sigma = np.maximum(1e-4, theta["decision_width"] * WELLS)
        diff = y_flat[:, None] - WELLS[None, :]
        logf = -0.5 * (diff / sigma[None, :]) ** 2 - np.log(sigma)[None, :]
        probs = _softmax(logf)
        steps = np.zeros((n_n, k))
        flowtime = np.zeros((n_n, k))
    elif spec.family == "cp":
        slog = float(theta["sigma_log"])
        centers = np.log(n_values)
        y = centers[:, None] + sqrt2 * slog * quad.t[None, :]
        y_flat = y.reshape(-1)
        diff = y_flat[:, None] - LOG_WELLS[None, :]
        logf = (-0.5 * (diff / slog) ** 2 - theta["alpha"] * LOG_WELLS[None, :])
        logf = logf / float(theta["readout_temp"])
        probs = _softmax(logf)
        steps = np.zeros((n_n, k))
        flowtime = np.zeros((n_n, k))
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"unknown family {spec.family!r}")

    probs = probs.reshape(n_n, k, N_WELLS)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=-1)   # (n, K)
    w = quad.w
    p_resp = (probs * w[None, :, None]).sum(axis=1)           # (n, N_WELLS)
    eh = (entropy * w[None, :]).sum(axis=1)
    et = (steps * w[None, :]).sum(axis=1)
    et_flow = (flowtime * w[None, :]).sum(axis=1)
    return {
        "p_resp": p_resp,
        "EH": eh,
        "ET": et,
        "ET_flow": et_flow,
        "probs_nodes": probs,
        "H_nodes": entropy,
        "steps_nodes": steps,
        "flowtime_nodes": flowtime,
    }


def apply_lapse(p_resp: np.ndarray, lapse: float, r_max: int) -> np.ndarray:
    """Mix the integer-well model with a uniform contamination over 1..r_max.

    ``p_resp`` has columns for wells ``1..N_WELLS``; the returned array has
    columns for the full response support ``1..r_max`` (column ``j`` -> ``r =
    j + 1``). Out-of-well responses receive only the lapse mass, so nothing is
    ever zero.
    """
    n_n = p_resp.shape[0]
    pf = np.full((n_n, r_max), lapse / r_max, dtype=float)
    pf[:, :N_WELLS] += (1.0 - lapse) * p_resp
    return pf


# ── Empirical trial-level data ───────────────────────────────────────────


@dataclass(frozen=True)
class DurationData:
    duration_ms: int
    n_trial: np.ndarray      # per-trial true numerosity (int, 1..15)
    r_trial: np.ndarray      # per-trial typed response (int, >= 1)
    logrt_trial: np.ndarray  # per-trial log(rt_seconds)
    counts: np.ndarray       # (15, r_max) response count table
    n_total: int


def load_trials(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Reconstruct one row per trial for the DURATION conditions.

    Mirrors the grouping in ``fit_helpers.extract_empirical`` (group by
    ``uniqueid, trial_id``; keep ``n_shown, n_guess, rt``; rt -> seconds) but
    keeps every individual trial instead of collapsing to per-n summaries.
    """
    df = pd.read_csv(csv_path)
    subset = df[(df["cond"] == "DURATION") & (df["exp_dur"].isin(DURATIONS))]
    trial = (
        subset.groupby(["uniqueid", "trial_id"])
        .agg(
            exp_dur=("exp_dur", "first"),
            n_shown=("n_shown", "first"),
            n_guess=("n_guess", "first"),
            rt=("rt", "first"),
        )
        .reset_index()
    )
    trial = trial[(trial["n_shown"] >= N_TRUE_MIN) & (trial["n_shown"] <= N_TRUE_MAX)]
    trial = trial[trial["rt"] > 0]
    trial["rt_s"] = trial["rt"] / 1000.0
    return trial.reset_index(drop=True)


def response_support_report(trial: pd.DataFrame) -> dict:
    """Empirical response support and the fraction outside the well support."""
    guesses = trial["n_guess"].to_numpy()
    out_mask = guesses > N_SUPPORT_MAX
    rows = []
    for d in DURATIONS:
        g = trial.loc[trial["exp_dur"] == d, "n_guess"].to_numpy()
        rows.append(
            {
                "duration_ms": int(d),
                "n_trials": int(g.size),
                "min_guess": int(g.min()),
                "max_guess": int(g.max()),
                "frac_out_of_support": float(np.mean(g > N_SUPPORT_MAX)),
            }
        )
    return {
        "r_max": int(guesses.max()),
        "min_guess": int(guesses.min()),
        "well_support_max": int(N_SUPPORT_MAX),
        "frac_out_of_support_overall": float(np.mean(out_mask)),
        "per_duration": rows,
    }


def build_duration_data(trial: pd.DataFrame, r_max: int) -> dict[int, DurationData]:
    out: dict[int, DurationData] = {}
    for d in DURATIONS:
        sub = trial[trial["exp_dur"] == d]
        n_t = sub["n_shown"].to_numpy(int)
        r_t = sub["n_guess"].to_numpy(int)
        logrt = np.log(sub["rt_s"].to_numpy(float))
        counts = np.zeros((N_SHOWN.size, r_max), dtype=float)
        np.add.at(counts, (n_t - N_TRUE_MIN, np.clip(r_t, 1, r_max) - 1), 1.0)
        out[int(d)] = DurationData(
            duration_ms=int(d),
            n_trial=n_t,
            r_trial=r_t,
            logrt_trial=logrt,
            counts=counts,
            n_total=int(n_t.size),
        )
    return out


# ── Response-only likelihood and fitting ─────────────────────────────────


def response_loglik(
    spec: ModelSpec,
    theta_vec: np.ndarray,
    quad: Quadrature,
    counts: np.ndarray,
    r_max: int,
) -> float:
    """Categorical log-likelihood ``sum_{n,r} count(n,r) log p_final(r | n)``."""
    theta = dict(zip(spec.param_names, theta_vec[:-1]))
    lapse = float(theta_vec[-1])
    try:
        grid = compute_grid(spec, theta, quad)
    except (FloatingPointError, ValueError):
        return -np.inf
    pf = apply_lapse(grid["p_resp"], lapse, r_max)
    if not np.all(np.isfinite(pf)) or np.any(pf <= 0):
        return -np.inf
    ll = float(np.sum(counts * np.log(pf)))
    return ll if np.isfinite(ll) else -np.inf


def fit_response(
    spec: ModelSpec,
    data: DurationData,
    quad_fit: Quadrature,
    r_max: int,
    *,
    maxiter: int,
    popsize: int,
    seed: int,
    quad_eval: Quadrature | None = None,
) -> dict:
    """Deterministic global fit (differential evolution + Nelder-Mead polish).

    The optimiser uses the cheaper ``quad_fit`` to locate the optimum; the
    reported log-likelihood / AIC / BIC are recomputed at the fitted parameters
    with the high-accuracy ``quad_eval`` so every model is scored on the same,
    near-exact footing.
    """
    bounds = list(spec.bounds) + [LAPSE_BOUNDS]
    quad_eval = quad_eval or quad_fit

    def neg_ll(vec: np.ndarray) -> float:
        return -response_loglik(spec, vec, quad_fit, data.counts, r_max)

    de = differential_evolution(
        neg_ll,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.85,
        seed=seed,
        init="latinhypercube",
        polish=False,
        updating="deferred",
        workers=1,
    )
    polish = minimize(
        neg_ll,
        de.x,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-4},
    )
    vec = polish.x if polish.fun <= de.fun else de.x

    logl_fit = -float(min(polish.fun, de.fun))
    logl = float(response_loglik(spec, vec, quad_eval, data.counts, r_max))
    k = len(spec.param_names) + 1  # + lapse
    n = data.n_total
    theta = dict(zip(spec.param_names, vec[: len(spec.param_names)]))
    return {
        "theta": theta,
        "lapse": float(vec[-1]),
        "logL": float(logl),
        "logL_fit": float(logl_fit),
        "k_params": int(k),
        "n_data": int(n),
        "AIC": float(2 * k - 2 * logl),
        "BIC": float(k * np.log(n) - 2 * logl),
        "theta_vec": vec,
    }


# ── RT-only likelihood and fitting ───────────────────────────────────────

RT_METRICS = ("entropy", "steps", "flowtime", "hybrid", "flow_entropy")
_RT_TWO_TERM = ("hybrid", "flow_entropy")


def _rt_link(metric: str, scale: np.ndarray, eh: np.ndarray, et: np.ndarray,
             et_flow: np.ndarray) -> np.ndarray:
    """Link metric (added to t0): single dynamical/static quantity or a 2-term mix.

    * ``entropy``       -- kappa * E[H(p)]                      (static read-out)
    * ``steps``         -- c * E[raw clean-up steps]            (clamp-sensitive)
    * ``flowtime``      -- c * E[gradient-flow convergence time](critical slowing)
    * ``hybrid``        -- alpha * E[steps]    + beta * E[H]
    * ``flow_entropy``  -- alpha * E[flowtime] + beta * E[H]
    """
    if metric == "entropy":
        return scale[0] * eh
    if metric == "steps":
        return scale[0] * et
    if metric == "flowtime":
        return scale[0] * et_flow
    if metric == "hybrid":
        return scale[0] * et + scale[1] * eh
    if metric == "flow_entropy":
        return scale[0] * et_flow + scale[1] * eh
    raise ValueError(f"unknown RT metric {metric!r}")


def _rt_scale_bounds(metric: str) -> list[tuple[float, float]]:
    if metric in _RT_TWO_TERM:
        return [(0.0, 5.0), (0.0, 5.0)]   # (dynamical coeff, entropy coeff)
    return [(0.0, 5.0)]


def fit_rt(
    spec: ModelSpec,
    data: DurationData,
    resp_fit: dict,
    quad: Quadrature,
    *,
    metric: str = "entropy",
    seed: int = SEED,
) -> dict:
    """Fit ``log rt ~ Normal(log(t0 + metric(n)), sigma_rt)`` (sigma profiled)."""
    grid = compute_grid(spec, resp_fit["theta"], quad)
    eh, et, et_flow = grid["EH"], grid["ET"], grid["ET_flow"]
    idx = data.n_trial - N_TRUE_MIN
    logrt = data.logrt_trial
    n = logrt.size
    jac = float(np.sum(logrt))  # Jacobian d log rt: makes logL a density of rt

    def profiled_negll(scale: np.ndarray) -> float:
        link = _rt_link(metric, scale, eh, et, et_flow)
        mu_by_n = np.log(T0 + link)
        if not np.all(np.isfinite(mu_by_n)):
            return 1e18
        resid = logrt - mu_by_n[idx]
        sigma2 = float(np.mean(resid * resid))
        if not (sigma2 > 0 and np.isfinite(sigma2)):
            return 1e18
        return 0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)

    bounds = _rt_scale_bounds(metric)
    de = differential_evolution(
        profiled_negll, bounds=bounds, maxiter=120, popsize=12, tol=1e-6,
        seed=seed, polish=True, updating="deferred", workers=1,
    )
    scale = de.x
    link = _rt_link(metric, scale, eh, et, et_flow)
    mu_by_n = np.log(T0 + link)
    resid = logrt - mu_by_n[idx]
    sigma2 = float(np.mean(resid * resid))
    sigma_rt = float(np.sqrt(sigma2))

    # logL is a true density of rt (normal in log rt minus the Jacobian sum).
    logl_lognorm = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    logl = logl_lognorm - jac
    k = len(bounds) + 1  # scale params + sigma_rt
    scale_names = ("alpha", "beta") if metric in _RT_TWO_TERM else ("rt_scale",)
    return {
        "metric": metric,
        "scale": {name: float(v) for name, v in zip(scale_names, scale)},
        "sigma_rt": sigma_rt,
        "logL": float(logl),
        "k_params": int(k),
        "n_data": int(n),
        "AIC": float(2 * k - 2 * logl),
        "BIC": float(k * np.log(n) - 2 * logl),
    }


# ── Joint (coupled) likelihood ───────────────────────────────────────────


def fit_joint(
    spec: ModelSpec,
    data: DurationData,
    resp_fit: dict,
    quad: Quadrature,
    r_max: int,
    *,
    metric: str = "entropy",
    seed: int = SEED,
) -> dict:
    """Coupled joint ``p(r, rt | n)`` integrated over the same latent nodes.

    The sensory parameters and lapse are held at the response-only MLE; only
    the RT parameters are fit. The RT distribution is a per-trial mixture of
    log-normals whose component weights are the node-posterior *given the
    observed response*, so the attractor's convergence-time spread feeds RT.
    """
    lapse = resp_fit["lapse"]
    grid = compute_grid(spec, resp_fit["theta"], quad)
    probs_nodes = grid["probs_nodes"]   # (15, K, N_WELLS)
    h_nodes = grid["H_nodes"]           # (15, K)
    steps_nodes = grid["steps_nodes"]   # (15, K)
    w = quad.w                          # (K,)

    n_idx = data.n_trial - N_TRUE_MIN
    r = data.r_trial
    logrt = data.logrt_trial
    n_tr = logrt.size
    jac = float(np.sum(logrt))

    # Per-trial node weights given the observed response.
    in_well = r <= N_WELLS
    well_col = np.clip(r, 1, N_WELLS) - 1
    # node response weight: (1-lapse)*p_node(r) + lapse/r_max (constant in k)
    node_resp = (1.0 - lapse) * probs_nodes[n_idx, :, well_col]  # (n_tr, K)
    node_resp = node_resp + lapse / r_max
    # for out-of-well responses the model contributes no node structure -> use
    # the prior node weights (rt depends on n only).
    node_resp[~in_well, :] = (lapse / r_max)
    qk = node_resp * w[None, :]
    qk = qk / qk.sum(axis=1, keepdims=True)   # (n_tr, K) node-posterior

    h_tr = h_nodes[n_idx, :]        # (n_tr, K)
    t_tr = steps_nodes[n_idx, :]    # (n_tr, K)

    def neg_cond_rt(params: np.ndarray) -> float:
        if metric == "hybrid":
            mu = np.log(T0 + params[0] * t_tr + params[1] * h_tr)
            sigma = params[2]
        else:
            link = params[0] * (t_tr if metric == "steps" else h_tr)
            mu = np.log(T0 + link)
            sigma = params[1]
        if sigma <= 0 or not np.all(np.isfinite(mu)):
            return 1e18
        # log-normal density of rt at each node, mixed over the node-posterior.
        log_comp = (
            -0.5 * np.log(2.0 * np.pi) - np.log(sigma)
            - 0.5 * ((logrt[:, None] - mu) / sigma) ** 2
        )
        log_mix = logsumexp(log_comp + np.log(qk + 1e-300), axis=1)
        val = float(np.sum(log_mix))
        return -val if np.isfinite(val) else 1e18

    scale_bounds = _rt_scale_bounds(metric)
    bounds = scale_bounds + [SIGMA_RT_BOUNDS]
    # few smooth parameters -> a small global budget + polish is plenty.
    de = differential_evolution(
        neg_cond_rt, bounds=bounds, maxiter=40, popsize=8, tol=1e-6,
        seed=seed, polish=True, updating="deferred", workers=1,
    )
    cond_rt_logl = -float(de.fun) - jac   # density of rt | response

    resp_logl = resp_fit["logL"]
    joint_logl = resp_logl + cond_rt_logl
    k_rt = len(bounds)
    k = resp_fit["k_params"] + k_rt
    n = data.n_total
    scale_names = ("alpha", "beta") if metric == "hybrid" else ("rt_scale",)
    return {
        "metric": metric,
        "scale": {name: float(v) for name, v in zip(scale_names, de.x[:-1])},
        "sigma_rt": float(de.x[-1]),
        "resp_logL": float(resp_logl),
        "cond_rt_logL": float(cond_rt_logl),
        "logL": float(joint_logl),
        "k_params": int(k),
        "n_data": int(n),
        "AIC": float(2 * k - 2 * joint_logl),
        "BIC": float(k * np.log(n) - 2 * joint_logl),
    }


# ── Full joint fit: optimise the landscape on response AND RT together ────


def _cond_rt_loglik(grid: dict, w: np.ndarray, lapse: float,
                    rt_params: np.ndarray, metric: str, data: DurationData,
                    r_max: int) -> float:
    """log p(rt | r, n) summed over trials: a per-trial mixture of log-normals
    weighted by the node-posterior given the observed response (density of rt)."""
    probs_nodes = grid["probs_nodes"]
    h_nodes, steps_nodes = grid["H_nodes"], grid["steps_nodes"]
    n_idx, r, logrt = data.n_trial - N_TRUE_MIN, data.r_trial, data.logrt_trial
    jac = float(np.sum(logrt))
    in_well = r <= N_WELLS
    well_col = np.clip(r, 1, N_WELLS) - 1
    node_resp = (1.0 - lapse) * probs_nodes[n_idx, :, well_col] + lapse / r_max
    node_resp[~in_well, :] = lapse / r_max
    qk = node_resp * w[None, :]
    s = qk.sum(axis=1, keepdims=True)
    qk = qk / np.where(s > 0, s, 1.0)
    h_tr, t_tr = h_nodes[n_idx, :], steps_nodes[n_idx, :]
    if metric in ("hybrid", "flow_entropy"):
        mu = np.log(T0 + rt_params[0] * t_tr + rt_params[1] * h_tr)
        sigma = rt_params[2]
    elif metric == "steps":
        mu = np.log(T0 + rt_params[0] * t_tr); sigma = rt_params[1]
    else:  # entropy
        mu = np.log(T0 + rt_params[0] * h_tr); sigma = rt_params[1]
    if sigma <= 0 or not np.all(np.isfinite(mu)):
        return -np.inf
    log_comp = (-0.5 * np.log(2.0 * np.pi) - np.log(sigma)
                - 0.5 * ((logrt[:, None] - mu) / sigma) ** 2)
    log_mix = logsumexp(log_comp + np.log(qk + 1e-300), axis=1)
    val = float(np.sum(log_mix)) - jac
    return val if np.isfinite(val) else -np.inf


def joint_loglik_full(spec: ModelSpec, full_vec: np.ndarray, quad: Quadrature,
                      counts: np.ndarray, data: DurationData, r_max: int,
                      metric: str) -> float:
    """Joint log-likelihood log p(r, rt | n) for a full parameter vector
    [sensory..., lapse, rt_scale(s)..., sigma_rt] -- response and RT together."""
    n_sens = len(spec.param_names)
    sensory = full_vec[:n_sens]
    lapse = float(full_vec[n_sens])
    rt_params = full_vec[n_sens + 1:]
    theta = dict(zip(spec.param_names, sensory))
    try:
        grid = compute_grid(spec, theta, quad)
    except (FloatingPointError, ValueError):
        return -1e18
    pf = apply_lapse(grid["p_resp"], lapse, r_max)
    if not np.all(np.isfinite(pf)) or np.any(pf <= 0):
        return -1e18
    resp_ll = float(np.sum(counts * np.log(pf)))
    cond_rt = _cond_rt_loglik(grid, quad.w, lapse, rt_params, metric, data, r_max)
    tot = resp_ll + cond_rt
    return tot if np.isfinite(tot) else -1e18


def fit_joint_full(
    spec: ModelSpec,
    data: DurationData,
    quad_fit: Quadrature,
    r_max: int,
    *,
    metric: str,
    maxiter: int,
    popsize: int,
    seed: int,
    quad_eval: Quadrature | None = None,
    resp_fit: dict | None = None,
) -> dict:
    """Co-optimise the *whole* model (landscape + lapse + RT params) on the
    joint response+RT likelihood.

    Strategy: warm-start from the two-stage separate fit (response-only landscape
    + lapse, then conditional-RT params) and *co-adjust* with a local joint
    polish. The warm start is itself a feasible joint point, so the joint result
    can only match or beat the separate fit -- a cold high-dim DE under-converges
    here. The polish is where the landscape is allowed to shift to help RT.

    ``resp_fit`` (a dict with ``theta`` and ``lapse``) skips the expensive
    stage-1 response fit by reusing a cached response-only fit."""
    quad_eval = quad_eval or quad_fit
    rt_bounds = _rt_scale_bounds(metric) + [SIGMA_RT_BOUNDS]
    bounds = list(spec.bounds) + [LAPSE_BOUNDS] + rt_bounds

    def neg(vec: np.ndarray) -> float:
        return -joint_loglik_full(spec, vec, quad_fit, data.counts, data, r_max, metric)

    # Stage 1: response-only fit (landscape + lapse) -- or reuse a cached one.
    rf = resp_fit if resp_fit is not None else fit_response(
        spec, data, quad_fit, r_max, maxiter=maxiter,
        popsize=popsize, seed=seed, quad_eval=quad_fit)
    # Stage 2: conditional-RT fit given that landscape.
    rj = fit_joint(spec, data, rf, quad_fit, r_max, metric=metric, seed=seed)
    x0 = np.array(
        [rf["theta"][p] for p in spec.param_names]
        + [rf["lapse"]]
        + list(rj["scale"].values()) + [rj["sigma_rt"]],
        dtype=float,
    )
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    # Stage 3: joint polish (co-adjust response + RT) from the warm start.
    polish = minimize(neg, x0, method="Nelder-Mead", bounds=bounds,
                      options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-5})
    vec = polish.x if polish.fun <= neg(x0) else x0  # never worse than warm start

    n_sens = len(spec.param_names)
    theta = dict(zip(spec.param_names, vec[:n_sens]))
    lapse = float(vec[n_sens])
    rt_params = np.asarray(vec[n_sens + 1:], dtype=float)
    grid = compute_grid(spec, theta, quad_eval)
    pf = apply_lapse(grid["p_resp"], lapse, r_max)
    resp_ll = float(np.sum(data.counts * np.log(pf)))
    cond_rt = _cond_rt_loglik(grid, quad_eval.w, lapse, rt_params, metric, data, r_max)
    logl = resp_ll + cond_rt
    k, n = len(bounds), data.n_total
    return {
        "theta": theta, "lapse": lapse, "rt_params": rt_params.tolist(),
        "metric": metric, "resp_logL": resp_ll, "cond_rt_logL": cond_rt,
        "logL": float(logl), "k_params": int(k), "n_data": int(n),
        "AIC": float(2 * k - 2 * logl), "BIC": float(k * np.log(n) - 2 * logl),
        "theta_vec": vec.tolist(),
    }


# ── Monte-Carlo validation of the quadrature ─────────────────────────────


def simulate_response_mc(
    spec: ModelSpec,
    theta: dict,
    n_values: np.ndarray = N_SHOWN,
    *,
    n_samples: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """High-sample Monte-Carlo estimate of ``p_model(r | n)`` over the wells."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_values.size, N_WELLS))
    for i, n in enumerate(n_values):
        if spec.family in ("poisson", "gaussian"):
            sd = float(evidence_sd(n, theta["s0"], theta["lambda0"]))
            y = n + sd * rng.standard_normal(n_samples)
            if spec.family == "poisson":
                c = float(theta["sigma_c"])
                x_star, _, _ = _cleanup(
                    y, family="poisson", c=c,
                    max_steps=MAX_STEPS if spec.cleanup else 0,
                )
                probs = _softmax(_poisson_logwells(x_star, c))
            else:
                sigma = _gauss_sigma(float(theta["sigma_c"]))
                x_star, _, _ = _cleanup(
                    y, family="gaussian", sigma=sigma,
                    max_steps=MAX_STEPS if spec.cleanup else 0,
                )
                probs = _softmax(_gauss_logwells(x_star, sigma))
        elif spec.family == "weber":
            sd = max(1e-4, theta["sigma0"] * n ** theta["exponent"])
            y = n + sd * rng.standard_normal(n_samples)
            sigma = np.maximum(1e-4, theta["decision_width"] * WELLS)
            diff = y[:, None] - WELLS[None, :]
            probs = _softmax(-0.5 * (diff / sigma[None, :]) ** 2 - np.log(sigma)[None, :])
        else:  # cp
            slog = float(theta["sigma_log"])
            y = np.log(n) + slog * rng.standard_normal(n_samples)
            diff = y[:, None] - LOG_WELLS[None, :]
            logf = (-0.5 * (diff / slog) ** 2 - theta["alpha"] * LOG_WELLS[None, :])
            probs = _softmax(logf / float(theta["readout_temp"]))
        out[i] = probs.mean(axis=0)
    return out


# ── Posterior-predictive summaries ───────────────────────────────────────


def predicted_summaries(pf: np.ndarray) -> pd.DataFrame:
    """Model mean guess / exact accuracy / entropy from the full ``p_final``.

    ``pf`` spans the whole response support ``1..r_max`` (lapse included), so the
    out-of-support contamination contributes to the predicted mean exactly as it
    does in the data.
    """
    r_max = pf.shape[1]
    support = np.arange(1, r_max + 1, dtype=float)
    expected = (pf * support[None, :]).sum(axis=1)
    mode = support[np.argmax(pf, axis=1)]
    exact = np.array([pf[i, int(n - 1)] for i, n in enumerate(N_SHOWN)])
    entropy = -(pf * np.log(pf + 1e-12)).sum(axis=1)
    return pd.DataFrame(
        {
            "n_shown": N_SHOWN,
            "mean_guess": expected,
            "pct_exact": 100.0 * exact,
            "mode_guess": mode,
            "entropy": entropy,
        }
    )


def empirical_summaries(data: DurationData) -> pd.DataFrame:
    rows = []
    for i, n in enumerate(N_SHOWN):
        sel = data.n_trial == n
        guesses = data.r_trial[sel]
        rts = np.exp(data.logrt_trial[sel])
        rows.append(
            {
                "n_shown": int(n),
                "mean_guess": float(np.mean(guesses)) if guesses.size else np.nan,
                "pct_exact": float(100.0 * np.mean(guesses == n)) if guesses.size else np.nan,
                "median_rt_s": float(np.median(rts)) if rts.size else np.nan,
                "count": int(guesses.size),
            }
        )
    return pd.DataFrame(rows)


def akaike_weights(aic: np.ndarray) -> np.ndarray:
    d = aic - np.min(aic)
    w = np.exp(-0.5 * d)
    return w / w.sum()


def predicted_median_rt(spec: ModelSpec, theta: dict, rt_scale: float,
                        quad: Quadrature) -> np.ndarray:
    """Predicted median RT per numerosity for the entropy metric.

    Median of the log-normal RT is ``exp(mu) = t0 + kappa * E[H](n)``.
    """
    grid = compute_grid(spec, theta, quad)
    return T0 + rt_scale * grid["EH"]


def build_predictive_table(
    resp_df: pd.DataFrame,
    rt_df: pd.DataFrame,
    data_by_dur: dict[int, "DurationData"],
    quad: Quadrature,
    r_max: int,
) -> pd.DataFrame:
    """Observed vs predicted per-numerosity summaries from fitted parameters.

    Reports the lapse-inclusive predicted mean (``pred_mean_guess``), a
    lapse-free 'core' mean over the integer wells (``pred_mean_guess_core``),
    exact accuracy, and the entropy-metric predicted median RT, alongside the
    matching empirical quantities (full and in-support empirical means).
    """
    rows: list[dict] = []
    rt_ent = rt_df[rt_df["metric"] == "entropy"]
    for key in resp_df["model_key"].unique():
        spec = MODEL_BY_KEY[key]
        for d in sorted(resp_df["duration_ms"].unique()):
            rr = resp_df[(resp_df.model_key == key) & (resp_df.duration_ms == d)].iloc[0]
            theta = {p: float(rr[f"theta_{p}"]) for p in spec.param_names}
            lapse = float(rr["lapse"])
            grid = compute_grid(spec, theta, quad)
            pf = apply_lapse(grid["p_resp"], lapse, r_max)
            full = predicted_summaries(pf)
            core_mean = (grid["p_resp"] * WELLS[None, :]).sum(axis=1)
            rt_row = rt_ent[(rt_ent.model_key == key) & (rt_ent.duration_ms == d)]
            rt_scale = float(rt_row["scale_rt_scale"].iloc[0]) if len(rt_row) else np.nan
            pred_rt = T0 + rt_scale * grid["EH"]
            data = data_by_dur[d]
            for i, n in enumerate(N_SHOWN):
                sel = data.n_trial == n
                guesses = data.r_trial[sel]
                rts = np.exp(data.logrt_trial[sel])
                in_sup = guesses[guesses <= N_SUPPORT_MAX]
                rows.append({
                    "model": spec.label, "model_key": key, "duration_ms": int(d),
                    "n_shown": int(n),
                    "emp_mean_guess": float(np.mean(guesses)),
                    "emp_mean_guess_insupport": float(np.mean(in_sup)) if in_sup.size else np.nan,
                    "pred_mean_guess": float(full.loc[i, "mean_guess"]),
                    "pred_mean_guess_core": float(core_mean[i]),
                    "emp_pct_exact": float(100.0 * np.mean(guesses == n)),
                    "pred_pct_exact": float(full.loc[i, "pct_exact"]),
                    "emp_median_rt_s": float(np.median(rts)),
                    "pred_median_rt_s": float(pred_rt[i]),
                })
    return pd.DataFrame(rows)


# ── Orchestration ────────────────────────────────────────────────────────


def _add_within_duration_weights(
    df: pd.DataFrame, group_cols: Sequence[str] = ("duration_ms",)
) -> pd.DataFrame:
    """Add delta_AIC and Akaike weight within each group (across models)."""
    df = df.copy()
    df["delta_AIC"] = np.nan
    df["akaike_weight"] = np.nan
    for _, grp in df.groupby(list(group_cols)):
        aic = grp["AIC"].to_numpy(float)
        df.loc[grp.index, "delta_AIC"] = aic - aic.min()
        df.loc[grp.index, "akaike_weight"] = akaike_weights(aic)
    return df


def run_all(args: argparse.Namespace) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Grid quadrature is robust to the attractor's near-step integrand (see the
    # quadrature benchmark / docs). Cheap grid for the optimiser; accurate grid
    # for reported likelihoods; modest grid for the per-trial joint mixture.
    quad_fit = make_quadrature(args.fit_nodes, kind="grid")
    quad_eval = make_quadrature(args.eval_nodes, kind="grid")
    quad_joint = make_quadrature(args.joint_nodes, kind="grid")
    durations = DURATIONS if not args.smoke else (80, 160, 640)
    model_keys = [m.key for m in MODEL_SPECS]
    if args.smoke:
        model_keys = ["hopfield_poisson", "no_cleanup_poisson", "weber_ans"]

    trial = load_trials()
    support = response_support_report(trial)
    r_max = support["r_max"]
    print(f"Loaded {len(trial)} trials. Response support 1..{r_max}; "
          f"{support['frac_out_of_support_overall']:.1%} of guesses exceed the "
          f"well support (>{N_SUPPORT_MAX}).")
    pd.DataFrame(support["per_duration"]).to_csv(
        OUT_DIR / "response_support_report.csv", index=False
    )
    data_by_dur = build_duration_data(trial, r_max)

    resp_rows: list[dict] = []
    rt_rows: list[dict] = []
    joint_rows: list[dict] = []
    confusion_rows: list[dict] = []
    resp_fits: dict[tuple[str, int], dict] = {}

    for key in model_keys:
        spec = MODEL_BY_KEY[key]
        for d in durations:
            data = data_by_dur[d]
            rf = fit_response(
                spec, data, quad_fit, r_max,
                maxiter=args.maxiter, popsize=args.popsize, seed=SEED + d,
                quad_eval=quad_eval,
            )
            resp_fits[(key, d)] = rf
            resp_rows.append({
                "model": spec.label, "model_key": key, "duration_ms": d,
                "cleanup": spec.cleanup, "family": spec.family,
                **{f"theta_{p}": rf["theta"][p] for p in spec.param_names},
                "lapse": rf["lapse"], "logL": rf["logL"],
                "k_params": rf["k_params"], "n_data": rf["n_data"],
                "AIC": rf["AIC"], "BIC": rf["BIC"],
            })

            # confusion table (predictive summaries are built afterwards from
            # the fitted parameters, so RT predictions are available)
            grid = compute_grid(spec, rf["theta"], quad_eval)
            pf = apply_lapse(grid["p_resp"], rf["lapse"], r_max)
            obs_counts = data.counts  # (15, r_max)
            obs_dist = obs_counts / obs_counts.sum(axis=1, keepdims=True)
            for i, n in enumerate(N_SHOWN):
                for r in range(1, N_SUPPORT_MAX + 1):
                    confusion_rows.append({
                        "model": spec.label, "model_key": key, "duration_ms": d,
                        "n_shown": int(n), "response": int(r),
                        "observed_prob": float(obs_dist[i, r - 1]),
                        "predicted_prob": float(pf[i, r - 1]),
                    })
                confusion_rows.append({
                    "model": spec.label, "model_key": key, "duration_ms": d,
                    "n_shown": int(n), "response": -1,  # aggregated out-of-support
                    "observed_prob": float(obs_dist[i, N_SUPPORT_MAX:].sum()),
                    "predicted_prob": float(pf[i, N_SUPPORT_MAX:].sum()),
                })

            # RT-only channel (entropy common to all; steps/hybrid for attractor)
            metrics = ["entropy"]
            if spec.has_dynamics:
                metrics += ["steps", "hybrid"]
            for metric in metrics:
                rt = fit_rt(spec, data, rf, quad_eval, metric=metric, seed=SEED + d)
                rt_rows.append({
                    "model": spec.label, "model_key": key, "duration_ms": d,
                    "metric": metric, **{f"scale_{k_}": v for k_, v in rt["scale"].items()},
                    "sigma_rt": rt["sigma_rt"], "logL": rt["logL"],
                    "k_params": rt["k_params"], "n_data": rt["n_data"],
                    "AIC": rt["AIC"], "BIC": rt["BIC"],
                })
                # The coupled joint headline uses entropy for every model (uniform
                # k); the steps/hybrid RT-only fits above already test whether the
                # clean-up's convergence time carries extra RT information.
                if metric == "entropy":
                    jt = fit_joint(
                        spec, data, rf, quad_joint, r_max, metric=metric, seed=SEED + d
                    )
                    joint_rows.append({
                        "model": spec.label, "model_key": key, "duration_ms": d,
                        "metric": metric, "resp_logL": jt["resp_logL"],
                        "cond_rt_logL": jt["cond_rt_logL"], "logL": jt["logL"],
                        "k_params": jt["k_params"], "n_data": jt["n_data"],
                        "AIC": jt["AIC"], "BIC": jt["BIC"],
                    })
            print(f"  [{key} @ {d}ms] resp logL={rf['logL']:.1f} "
                  f"AIC={rf['AIC']:.1f} lapse={rf['lapse']:.3f}", flush=True)

    resp_df = _add_within_duration_weights(pd.DataFrame(resp_rows))
    rt_df = _add_within_duration_weights(
        pd.DataFrame(rt_rows), group_cols=("duration_ms", "metric")
    )
    joint_df = _add_within_duration_weights(
        pd.DataFrame(joint_rows), group_cols=("duration_ms", "metric")
    )
    joint_entropy = _add_within_duration_weights(
        joint_df[joint_df["metric"] == "entropy"].copy()
    )

    resp_df.to_csv(OUT_DIR / "per_duration_response_likelihood.csv", index=False)
    rt_df.to_csv(OUT_DIR / "per_duration_rt_likelihood.csv", index=False)
    joint_df.to_csv(OUT_DIR / "per_duration_joint_likelihood.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        OUT_DIR / "response_confusion_observed_vs_predicted.csv", index=False
    )
    psum = build_predictive_table(resp_df, rt_df, data_by_dur, quad_eval, r_max)
    psum.to_csv(OUT_DIR / "posterior_predictive_summaries.csv", index=False)

    summary = build_summary(resp_df, joint_entropy, model_keys)
    summary.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)

    try:
        make_plots(resp_df, joint_entropy, summary, confusion_rows,
                   psum, durations, model_keys)
    except Exception as exc:  # plotting must never break the numeric outputs
        print(f"[warn] plotting failed: {exc}")

    print("\n=== Response-only model comparison (summed over durations) ===")
    print(summary[["model", "resp_AIC_total", "resp_dAIC", "resp_weight"]]
          .round({"resp_AIC_total": 1, "resp_dAIC": 2, "resp_weight": 3})
          .to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")
    return {"summary": summary, "resp_df": resp_df, "rt_df": rt_df,
            "joint_df": joint_df, "support": support}


def build_summary(
    resp_df: pd.DataFrame,
    joint_entropy: pd.DataFrame,
    model_keys: Sequence[str],
) -> pd.DataFrame:
    """Across-duration totals for response-only and (entropy) joint channels."""
    rows = []
    for key in model_keys:
        spec = MODEL_BY_KEY[key]
        r = resp_df[resp_df["model_key"] == key]
        j = joint_entropy[joint_entropy["model_key"] == key]
        rows.append({
            "model": spec.label, "model_key": key,
            "cleanup": spec.cleanup, "family": spec.family, "is_proxy": spec.is_proxy,
            "resp_logL_total": float(r["logL"].sum()),
            "resp_k_total": int(r["k_params"].sum()),
            "resp_AIC_total": float(r["AIC"].sum()),
            "resp_BIC_total": float(r["BIC"].sum()),
            "resp_AIC_mean": float(r["AIC"].mean()),
            "joint_logL_total": float(j["logL"].sum()) if len(j) else np.nan,
            "joint_k_total": int(j["k_params"].sum()) if len(j) else 0,
            "joint_AIC_total": float(j["AIC"].sum()) if len(j) else np.nan,
            "joint_BIC_total": float(j["BIC"].sum()) if len(j) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["resp_dAIC"] = out["resp_AIC_total"] - out["resp_AIC_total"].min()
    out["resp_weight"] = akaike_weights(out["resp_AIC_total"].to_numpy(float))
    if out["joint_AIC_total"].notna().any():
        jv = out["joint_AIC_total"].to_numpy(float)
        out["joint_dAIC"] = jv - np.nanmin(jv)
        valid = ~np.isnan(jv)
        w = np.full(len(jv), np.nan)
        w[valid] = akaike_weights(jv[valid])
        out["joint_weight"] = w
    return out


# ── Plots ────────────────────────────────────────────────────────────────

_PLOT_STYLE = {
    "figure.dpi": 130, "savefig.dpi": 250, "savefig.bbox": "tight",
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
}


def make_plots(resp_df, joint_entropy, summary, confusion_rows,
               psum, durations, model_keys) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PLOT_STYLE)

    conf = pd.DataFrame(confusion_rows)
    psum = pd.DataFrame(psum)

    _plot_confusion_heatmaps(conf, durations, plt)
    _plot_predictive_summaries(psum, durations, model_keys, plt)
    _plot_aic(resp_df, "AIC", "Response-only trial-level AIC",
              OUT_DIR / "aic_response_only.png", plt)
    if len(joint_entropy):
        _plot_aic(joint_entropy, "AIC", "Joint (response + entropy-RT) AIC",
                  OUT_DIR / "aic_joint.png", plt)


def _plot_confusion_heatmaps(conf, durations, plt) -> None:
    import numpy as np
    keys = ["hopfield_poisson", "no_cleanup_poisson"]
    keys = [k for k in keys if k in set(conf["model_key"])]
    if not keys:
        keys = [conf["model_key"].iloc[0]]
    ncol = 1 + len(keys)
    fig, axes = plt.subplots(len(durations), ncol,
                             figsize=(3.4 * ncol, 2.8 * len(durations)),
                             squeeze=False)
    resp_vals = list(range(1, N_SUPPORT_MAX + 1))
    for ri, d in enumerate(durations):
        sub = conf[(conf["duration_ms"] == d) & (conf["response"] >= 1)]
        obs = (sub[sub["model_key"] == keys[0]]
               .pivot(index="n_shown", columns="response", values="observed_prob")
               .reindex(columns=resp_vals).to_numpy())
        panels = [("Observed", obs)]
        for k in keys:
            pred = (sub[sub["model_key"] == k]
                    .pivot(index="n_shown", columns="response", values="predicted_prob")
                    .reindex(columns=resp_vals).to_numpy())
            panels.append((MODEL_BY_KEY[k].label.split(" (")[0], pred))
        for ci, (title, mat) in enumerate(panels):
            ax = axes[ri][ci]
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma",
                           extent=[0.5, N_SUPPORT_MAX + 0.5, 0.5, N_TRUE_MAX + 0.5],
                           vmin=0.0, vmax=np.nanmax(obs))
            if ci == 0:
                ax.set_ylabel(f"{d} ms\nn shown")
            if ri == 0:
                ax.set_title(title)
            if ri == len(durations) - 1:
                ax.set_xlabel("response r")
    fig.suptitle("Observed vs predicted response distributions", y=1.005)
    fig.colorbar(im, ax=axes, shrink=0.6, label="P(r | n)")
    fig.savefig(OUT_DIR / "response_heatmaps.png")
    plt.close(fig)


def _plot_predictive_summaries(psum, durations, model_keys, plt) -> None:
    """Three posterior-predictive rows: mean guess, % exact, median RT.

    The mean-guess row uses the lapse-free 'core' model mean against the
    in-support empirical mean, so the crude uniform contamination (which
    inflates both means by spreading a few percent of mass up to r=99) does not
    distort the central-tendency comparison.
    """
    show = [k for k in ("hopfield_poisson", "no_cleanup_poisson",
                        "gaussian_benford_attractor", "weber_ans") if k in model_keys]
    cmap = plt.colormaps["plasma"]
    colors = {k: cmap(v) for k, v in zip(show, np.linspace(0.1, 0.85, len(show)))}
    fig, axes = plt.subplots(3, len(durations),
                             figsize=(3.0 * len(durations), 8.4), squeeze=False)
    for ci, d in enumerate(durations):
        sd = psum[psum["duration_ms"] == d]
        emp = sd[sd["model_key"] == show[0]]
        ax0, ax1, ax2 = axes[0][ci], axes[1][ci], axes[2][ci]
        ax0.plot(emp["n_shown"], emp["emp_mean_guess_insupport"], "k.", ms=8, label="data")
        ax0.plot([1, 15], [1, 15], color="0.8", lw=0.8, zorder=0)
        ax1.plot(emp["n_shown"], emp["emp_pct_exact"], "k.", ms=8, label="data")
        ax2.plot(emp["n_shown"], emp["emp_median_rt_s"], "k.", ms=8, label="data")
        for k in show:
            m = sd[sd["model_key"] == k]
            lbl = MODEL_BY_KEY[k].label.split(" (")[0]
            ax0.plot(m["n_shown"], m["pred_mean_guess_core"], color=colors[k], lw=1.6, label=lbl)
            ax1.plot(m["n_shown"], m["pred_pct_exact"], color=colors[k], lw=1.6)
            ax2.plot(m["n_shown"], m["pred_median_rt_s"], color=colors[k], lw=1.6)
        ax0.set_title(f"{d} ms")
        if ci == 0:
            ax0.set_ylabel("mean guess (core)")
            ax1.set_ylabel("% exact")
            ax2.set_ylabel("median RT (s)")
        ax2.set_xlabel("n shown")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Posterior-predictive summaries (lines) vs data (dots)", y=1.005)
    fig.savefig(OUT_DIR / "predictive_summaries.png")
    plt.close(fig)


def _plot_aic(df, col, title, path, plt) -> None:
    durations = sorted(df["duration_ms"].unique())
    keys = list(dict.fromkeys(df["model_key"]))
    cmap = plt.colormaps["plasma"]
    colors = {k: cmap(v) for k, v in zip(keys, np.linspace(0.1, 0.9, len(keys)))}
    fig, ax = plt.subplots(figsize=(1.7 * len(durations) + 3, 5.0))
    x = np.arange(len(durations))
    width = min(0.13, 0.8 / len(keys))
    offs = (np.arange(len(keys)) - (len(keys) - 1) / 2) * width
    for i, k in enumerate(keys):
        sub = df[df["model_key"] == k].set_index("duration_ms").reindex(durations)
        # plot delta-AIC within each duration for readability
        ax.bar(x + offs[i], sub["delta_AIC"].to_numpy(), width=width,
               color=colors[k], label=MODEL_BY_KEY[k].label, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} ms" for d in durations])
    ax.set_ylabel(r"$\Delta$AIC within duration")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def replot_from_saved(eval_nodes: int = 1601) -> None:
    """Rebuild the predictive table and all plots from the saved fit CSVs.

    Lets the (slow) fitting and the (fast) reporting be decoupled: predictive
    summaries and figures are regenerated from the fitted parameters without
    re-running the optimiser.
    """
    quad_eval = make_quadrature(eval_nodes, kind="grid")
    resp_df = pd.read_csv(OUT_DIR / "per_duration_response_likelihood.csv")
    rt_df = pd.read_csv(OUT_DIR / "per_duration_rt_likelihood.csv")
    joint_df = pd.read_csv(OUT_DIR / "per_duration_joint_likelihood.csv")
    summary = pd.read_csv(OUT_DIR / "model_comparison_summary.csv")
    conf = pd.read_csv(OUT_DIR / "response_confusion_observed_vs_predicted.csv")
    trial = load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = build_duration_data(trial, r_max)
    durations = sorted(resp_df["duration_ms"].unique())
    model_keys = list(dict.fromkeys(resp_df["model_key"]))
    joint_entropy = joint_df[joint_df["metric"] == "entropy"]

    psum = build_predictive_table(resp_df, rt_df, data_by_dur, quad_eval, r_max)
    psum.to_csv(OUT_DIR / "posterior_predictive_summaries.csv", index=False)
    make_plots(resp_df, joint_entropy, summary, conf.to_dict("records"),
               psum, durations, model_keys)
    print(f"Regenerated predictive summaries + plots in {OUT_DIR}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-nodes", type=int, default=201,
                   help="grid nodes for the optimiser objective")
    p.add_argument("--eval-nodes", type=int, default=1601,
                   help="grid nodes for the reported (accurate) likelihoods")
    p.add_argument("--joint-nodes", type=int, default=101,
                   help="grid nodes for the per-trial joint RT mixture")
    p.add_argument("--maxiter", type=int, default=35, help="DE maxiter (response)")
    p.add_argument("--popsize", type=int, default=10, help="DE popsize (response)")
    p.add_argument("--smoke", action="store_true",
                   help="fast subset (3 models, 3 durations, low effort)")
    p.add_argument("--replot", action="store_true",
                   help="regenerate predictive summaries + plots from saved fits")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.replot:
        replot_from_saved(eval_nodes=args.eval_nodes)
        return
    if args.smoke:
        args.maxiter = min(args.maxiter, 12)
        args.popsize = min(args.popsize, 6)
        args.fit_nodes = min(args.fit_nodes, 151)
        args.eval_nodes = min(args.eval_nodes, 401)
        args.joint_nodes = min(args.joint_nodes, 61)
    run_all(args)


if __name__ == "__main__":
    main()
