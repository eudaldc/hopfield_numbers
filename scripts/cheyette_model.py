"""Vendored core of the Cheyette & Piantadosi (2020) bounded-optimal number model.

Source: https://github.com/samcheyette/info_theory_number (``computeQ.py``),
accompanying Cheyette, S. J. & Piantadosi, S. T. (2020), "A unified account of
numerosity perception", *Nature Human Behaviour*.

The functions below are copied verbatim (only reorganised and documented) so the
analysis is self-contained and reproducible without the external clone. The
model treats numerosity perception as a rate-distortion / information-bottleneck
problem:

  * Prior ``P(x) = (1/x^a) / Z`` with ``a = 2`` (Zipfian).
  * Channel/posterior ``Q(k | n; lambda) ∝ P(k) * exp(-(P(n)/lambda) (n-k)^2)``.
  * For a target information budget ``info_bound`` (bits), ``lambda`` is chosen
    per numerosity so that ``KL(Q || P) = info_bound``. Rarer ``n`` (small P(n))
    get a wider posterior at the same bit budget.

We add nothing to the model here; the fitting layer that maps exposure duration
to ``info_bound = min(B, R*t)`` and scores it against the trial-level Piantadosi
responses lives in ``scripts/trial_level_cp_exact.py``.
"""

from __future__ import annotations

import numpy as np


def KL(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL divergence ``KL(Q || P)`` in bits, row-wise (prior p, posterior q)."""
    return np.sum(q * (np.log2(q) - np.log2(p)), axis=1)


def P(x: np.ndarray, a: float = 2.0) -> np.ndarray:
    """Zipfian prior ``1/x^a`` normalised over the support ``x``."""
    p = 1.0 / (x ** a)
    return p / np.sum(p)


def compute_q_nk(ns, ks, p_n, p_k, lam, sm: float = 1e-10):
    """Posterior ``Q(k | n)`` for a per-numerosity inverse temperature ``lam``.

    ``ns`` is a column ``(N, 1)``, ``ks`` a row ``(K,)``, ``p_n`` a column
    ``(N, 1)`` prior at each ``n``, ``p_k`` a row ``(K,)`` prior at each ``k``,
    ``lam`` a per-``n`` vector ``(N,)``. Returns ``(N, K)`` normalised rows.
    """
    lam = lam.reshape((len(lam), 1))
    p_n = p_n.reshape((len(p_n), 1))
    q_nk = -((p_n * (ns - ks) ** 2.0) / lam)
    q_nk = p_k * np.exp(q_nk)
    q_nk = (q_nk + sm) / np.sum(q_nk + sm, axis=1).reshape(len(q_nk), 1)
    return q_nk


def find_q_nk(ns, ks, p_n, p_k, info_bound, n_steps: int = 10000,
              step_size: float = 0.01):
    """Gradient descent on ``log lambda`` to hit ``KL(Q || P) = info_bound``.

    One ``lambda`` per numerosity; the update ``lam <- exp(log lam + step_size
    * (KL - info_bound))`` raises the temperature where the KL overshoots the
    budget and lowers it where it undershoots.
    """
    lams = np.ones_like(ns)
    q_nk = compute_q_nk(ns, ks, p_n, p_k, lams)
    ents = KL(p_k, q_nk)
    for _ in range(n_steps):
        diffs = ents - info_bound
        deltas = diffs * step_size
        lams = np.exp(np.log(lams) + deltas.reshape(len(deltas), 1))
        q_nk = compute_q_nk(ns, ks, p_n, p_k, lams)
        ents = KL(p_k, q_nk)
    return q_nk


def build_q_cache(alpha: float = 2.0, min_n: int = 1, max_n: int = 15,
                  max_k: int = 25, info_step: float = 0.1, max_info: float = 10.0,
                  n_steps: int = 10000, verbose: bool = False) -> dict:
    """Precompute ``Q(k | n)`` for a grid of information budgets.

    Returns a dict with ``q_cache`` of shape ``(n_info_bounds, n_num, n_k)`` and
    the axis metadata needed by :func:`lookup_q` / :func:`lookup_q_interp`.
    """
    ks = np.arange(1, max_k + 1, dtype=float)
    numerosities = np.arange(min_n, max_n + 1)
    ns_col = numerosities.reshape(-1, 1).astype(float)
    p_ks = P(ks, alpha)
    p_ns = P(ns_col, alpha)

    info_bounds = np.round(np.arange(0, max_info + info_step / 2, info_step), 1)
    q_cache = np.zeros((len(info_bounds), len(numerosities), len(ks)))

    for i, ib in enumerate(info_bounds):
        if ib == 0:
            for j in range(len(numerosities)):
                q_cache[i, j, :] = p_ks
        else:
            q_cache[i] = find_q_nk(ns_col, ks, p_ns, p_ks, ib, n_steps=n_steps)
        if verbose and (i + 1) % 20 == 0:
            print(f"  built {i + 1}/{len(info_bounds)} info bounds")

    return {
        "q_cache": q_cache,
        "info_bounds": info_bounds,
        "info_step": info_step,
        "alpha": alpha,
        "ks": ks,
        "ns": numerosities,
        "p_ks": p_ks,
    }


def lookup_q(cache_dict: dict, info_bound: float) -> np.ndarray:
    """Nearest-grid ``Q(k | n)`` at the requested information budget."""
    step = cache_dict["info_step"]
    max_idx = len(cache_dict["info_bounds"]) - 1
    idx = int(round(info_bound / step))
    idx = max(0, min(idx, max_idx))
    return cache_dict["q_cache"][idx]


def lookup_q_interp(cache_dict: dict, info_bound: float) -> np.ndarray:
    """Linearly interpolated ``Q(k | n)`` between the two nearest budgets."""
    step = cache_dict["info_step"]
    max_idx = len(cache_dict["info_bounds"]) - 1
    fidx = max(0.0, min(info_bound / step, float(max_idx)))
    lo = int(fidx)
    hi = min(lo + 1, max_idx)
    frac = fidx - lo
    if frac < 1e-12:
        return cache_dict["q_cache"][lo]
    return (1.0 - frac) * cache_dict["q_cache"][lo] + frac * cache_dict["q_cache"][hi]
