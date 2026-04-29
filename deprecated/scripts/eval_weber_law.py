#!/usr/bin/env python
"""Evaluate Weber's law in a Panamath-like comparison task.

Simulates a non-symbolic number comparison ("which side has more?")
using two independent continuous evidence channels.  Each numerosity
is processed through Benford-attractor cleanup and the model responds
based on the expected values of the cleaned posteriors.

All attractor-landscape and evidence-channel parameters come from the
Piantadosi 160 ms fit (see fit_piantadosi_durations.py).  The task
introduces zero free parameters: comparison accuracy is a pure
prediction of the fitted model.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.optimize import curve_fit

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    benford_sigma, cleanup, FIXED, BENFORD_PRIOR,
    FIXED_BETA, FIXED_KAPPA, FIXED_ETA,
    evidence_noise_std,
    psychometric_weber,
)
from scripts.plots import plot_weber

# ── Output & simulation settings ────────────────────────────────────────
OUTPUT_DIR = Path("output/numerical_effects")
N_SAMPLES = 5000
SEED = 73

N_MIN = 1
N_SUPPORT_MAX = 30

# Fitted parameters from the Piantadosi 160 ms condition.
# beta, kappa, eta use fixed values from config when set; update the
# remaining fitted values after re-running fit_piantadosi.py.
THETA = {
    "sigma_c": 0.0237,
    "tau_in": 0.9940,
    "s0": 0.0658,
    "lambda0": 1.4326,
}
THETA["beta"] = FIXED_BETA or THETA.get("beta", 1.0)
THETA["kappa"] = FIXED_KAPPA or THETA.get("kappa", 1.0)
THETA["eta"] = FIXED_ETA or THETA.get("eta", 1.0)

REF_NUMEROSITIES = [4, 8, 12, 16]
MAX_TEST_RATIO = 2.5


# ── Helpers ──────────────────────────────────────────────────────────────

def _setup_wells():
    integers = torch.arange(N_MIN, N_SUPPORT_MAX + 1, dtype=torch.float32)
    sigma = benford_sigma(integers, THETA["sigma_c"], THETA["beta"])
    freq = torch.log10(1.0 + 1.0 / integers)
    return {
        "integers": integers,
        "mu": integers,
        "sigma_sq": sigma.square(),
        "log_sigma": sigma.log(),
        "log_freq": freq.log() if BENFORD_PRIOR else torch.zeros_like(integers),
        "inv_tau_sq": 1.0 / THETA["tau_in"] ** 2,
    }


def _sample_evidence(n_true, n_samples, generator):
    noise_std = evidence_noise_std(n_true, THETA)
    return float(n_true) + noise_std * torch.randn(n_samples, generator=generator)


@torch.no_grad()
def _process(n_true, wells, n_samples, generator):
    y0 = _sample_evidence(n_true, n_samples, generator)
    out = cleanup(
        y0, wells["mu"], wells["log_freq"],
        THETA["kappa"], wells["inv_tau_sq"], THETA["eta"],
        FIXED["max_steps"], FIXED["tol"],
        sigma_sq=wells["sigma_sq"], log_sigma=wells["log_sigma"],
    )
    probs = out["probs"]
    expected_n = (probs * wells["integers"]).sum(dim=-1)
    return {"expected_n": expected_n, "probs": probs}


# ═══════════════════════════════════════════════════════════════════════
#  Weber discrimination task
# ═══════════════════════════════════════════════════════════════════════

def evaluate_weber() -> pd.DataFrame:
    wells = _setup_wells()
    g = torch.Generator().manual_seed(SEED)

    records: list[dict] = []
    for n_ref in REF_NUMEROSITIES:
        n_test_max = min(int(n_ref * MAX_TEST_RATIO), N_SUPPORT_MAX - 3)
        for n_test in range(n_ref + 1, n_test_max + 1):
            out_ref = _process(n_ref, wells, N_SAMPLES, g)
            out_test = _process(n_test, wells, N_SAMPLES, g)

            acc = (out_test["expected_n"] > out_ref["expected_n"]
                   ).float().mean().item() * 100

            records.append({
                "n_ref": n_ref,
                "n_test": n_test,
                "ratio": n_test / n_ref,
                "distance": n_test - n_ref,
                "accuracy": acc,
            })

    return pd.DataFrame(records)


# ── Weber fraction fitting ───────────────────────────────────────────────

def fit_weber_fractions(df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict] = []
    for n_ref, sub in df.groupby("n_ref"):
        r = sub["ratio"].values
        a = sub["accuracy"].values
        try:
            popt, _ = curve_fit(psychometric_weber, r, a,
                                p0=[0.20], bounds=(0.01, 3.0))
            w = float(popt[0])
        except RuntimeError:
            import numpy as np
            w = np.nan
        results.append({"n_ref": int(n_ref), "weber_fraction": w})
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Weber's Law — Panamath-like comparison")
    print("=" * 60)

    df = evaluate_weber()
    print(df.to_string(index=False), flush=True)

    wf = fit_weber_fractions(df)
    print("\nWeber fractions:")
    print(wf.to_string(index=False), flush=True)

    plot_weber(df, wf, OUTPUT_DIR, MAX_TEST_RATIO)
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
