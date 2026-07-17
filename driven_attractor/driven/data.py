"""Low-memory access to the trial-level data (from the cached .npz).

Identical content to ``scripts.trial_level_model_comparison.load_trials`` /
``build_duration_data``, but without re-reading the 45 MB per-dot CSV.
Run ``driven_attractor/build_cache.py`` once to create the cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "driven_attractor" / "output" / "trials.npz"
DURATIONS = (40, 80, 160, 320, 640)
N_SHOWN_MAX = 15


def load() -> dict:
    z = np.load(CACHE, allow_pickle=True)
    return {k: z[k] for k in ("exp_dur", "n_shown", "n_guess", "rt_s", "subject")}


def r_max(d: dict | None = None) -> int:
    d = d or load()
    return int(d["n_guess"].max())


def counts_by_duration(d: dict | None = None, rmax: int | None = None,
                       mask: np.ndarray | None = None) -> dict[int, np.ndarray]:
    d = d or load()
    rmax = rmax or r_max(d)
    sel_all = np.ones(d["n_shown"].shape, bool) if mask is None else mask
    out = {}
    for dur in DURATIONS:
        sel = sel_all & (d["exp_dur"] == dur)
        c = np.zeros((N_SHOWN_MAX, rmax))
        np.add.at(c, (d["n_shown"][sel] - 1,
                      np.clip(d["n_guess"][sel], 1, rmax) - 1), 1.0)
        out[dur] = c
    return out
