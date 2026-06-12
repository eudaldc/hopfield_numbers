"""Compute LONO/LODO for the Gaussian-Benford attractor, using the same
conventions as the Hopfield-Poisson model in ``model_comparison`` (in-sample
normalized loss for LONO; shared c/kappa + log-duration extrapolation of the
sensory parameters for LODO). Must run with a gaussian ``config.yaml``.

    PYTHONPATH=. .venv/bin/python scripts/compute_gb_attractor_cv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.fit_helpers as fh
from scripts.model_comparison import normalized_loss, empirical_for_duration

assert fh.WELL_TYPE == "gaussian", "config.yaml must set well_type: gaussian"

DURATIONS = (40, 80, 160, 320, 640)
FIT_CSV = Path("output/model_comparison/gaussian_benford_attractor_fit.csv")
T0 = 1.30


def theta_for(row: pd.Series) -> dict:
    return fh.complete_piantadosi_theta(
        {
            "sigma_c": float(row["sigma_c"]),
            "s0": float(row["s0"]),
            "lambda0": float(row["lambda0"]),
            "kappa": float(row["kappa"]),
        }
    )


def main() -> None:
    fit = pd.read_csv(FIT_CSV).set_index("duration_ms")

    # LONO: in-sample normalized loss per duration (matches the main-model row).
    lono = []
    for d in DURATIONS:
        emp = empirical_for_duration(d)
        model = fh.simulate(theta_for(fit.loc[d]), n_samples_per_n=5000, seed=73)
        lono.append(normalized_loss(emp, model))
    lono_mean = float(np.mean(lono))

    # LODO: shared sigma_c/kappa over training durations (>=160 ms), sensory
    # parameters extrapolated in log-duration, then predict the held-out one.
    lodo = []
    for hold in DURATIONS:
        train = fit.drop(index=hold)
        stable = train[train.index >= 160]
        stable = stable if len(stable) else train
        log_d = np.log(train.index.to_numpy(float))
        sigma_c = float(stable["sigma_c"].mean())
        kappa = float(stable["kappa"].mean())
        s0 = float(np.exp(np.polyval(np.polyfit(log_d, np.log(train["s0"]), 1), np.log(hold))))
        lambda0 = float(np.polyval(np.polyfit(log_d, train["lambda0"], 1), np.log(hold)))
        theta = fh.complete_piantadosi_theta(
            {"sigma_c": sigma_c, "s0": s0, "lambda0": lambda0, "kappa": kappa}
        )
        model = fh.simulate(theta, n_samples_per_n=5000, seed=73)
        lodo.append(normalized_loss(empirical_for_duration(hold), model))
    lodo_mean = float(np.mean(lodo))

    print(f"GB-attractor LONO (in-sample) = {lono_mean:.4f}")
    print(f"GB-attractor LODO (constrained) = {lodo_mean:.4f}")
    pd.DataFrame([{"model": "gaussian_benford_attractor", "LONO": lono_mean, "LODO": lodo_mean}]).to_csv(
        "output/model_comparison/gaussian_benford_attractor_cv.csv", index=False
    )


if __name__ == "__main__":
    main()
