"""Exploratory coarse scan of the driven-model likelihood surface.

Purpose (exploratory, pre-fitting): locate the plausible (rho, tau_x, beta)
region for both evidence processes, check for gross multimodality, and get a
first look at whether the driven family lands anywhere near the data.

    .venv/bin/python driven_attractor/scan_explore.py
"""

from __future__ import annotations

import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
_CTX: dict = {}


def _init() -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    _CTX["counts"] = {d: data[d].counts for d in M.DURATIONS}
    _CTX["r_max"] = r_max


def _one(job) -> dict:
    model, sigma_c, p, rho, tau_x, beta, gamma = job
    theta = {"sigma_c": sigma_c, "p": p, "rho": rho, "tau_x": tau_x,
             "beta": beta, "gamma": gamma}
    try:
        ll, lapses = M.total_loglik(model, theta, _CTX["counts"], _CTX["r_max"],
                                    m_paths=1024)
    except Exception:
        ll, lapses = -np.inf, {}
    return {"model": model, "sigma_c": sigma_c, "p": p, "rho": rho,
            "tau_x": tau_x, "beta": beta, "gamma": gamma, "logL": ll,
            "mean_lapse": float(np.mean(list(lapses.values()))) if lapses else np.nan}


def main() -> None:
    lands = [(0.0222, 1.797), (0.05, 1.5), (0.10, 1.2)]
    rhos = [8.0, 20.0, 50.0, 120.0]
    taus = [0.05, 0.15, 0.4, 1.0]
    betas = [1.0, 4.0, 16.0, 64.0]
    jobs = []
    for (sc, p), rho, tau, beta, gam in itertools.product(
            lands, rhos, taus, betas, (0.0, 1.0)):
        jobs.append(("M2_tether", sc, p, rho, tau, beta, gam))
    for (sc, p), rho, gam in itertools.product(lands, rhos, (0.0, 1.0)):
        jobs.append(("M1_offset", sc, p, rho, 0.25, 1.0, gam))

    with Pool(12, initializer=_init) as pool:
        rows = pool.map(_one, jobs)
    df = pd.DataFrame(rows).sort_values("logL", ascending=False)
    df.to_csv(OUT / "scan_explore.csv", index=False)
    print(df.head(25).to_string(index=False))
    print("\nreference: M0 shared-landscape logL = -114915.2 (k=17); "
          "M0 per-duration = -113150.6 (k=25)")


if __name__ == "__main__":
    main()
