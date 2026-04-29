#!/usr/bin/env python
"""Compare model predictions against DeCarli et al. data.

Data source: https://osf.io/gjpb3/overview

Two tasks:
1. Exact enumeration (1-8 dots): per-subject accuracy and RT by numerosity,
   split by DD (developmental dyscalculia) vs controls.
2. Symbolic comparison: per-subject accuracy and RT by distance (1-8),
   split by DD vs controls.

For enumeration, the model predicts:
  - Exact accuracy = P(MAP = n | true n)  (diagonal of confusion matrix)
  - RT proxy = posterior entropy

For symbolic comparison, the model predicts:
  - Accuracy = P(E[n|larger] > E[n|smaller]) averaged over all digit
    pairs at each distance (1-9 excluding 0).
  - RT proxy = mean posterior entropy of the pair.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fit_helpers import (
    benford_sigma,
    cleanup,
    FIXED,
    WELL_TYPE,
    BENFORD_PRIOR,
    FIXED_BETA,
    FIXED_KAPPA,
    FIXED_ETA,
    evidence_noise_std,
)

# ── Settings ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output/decarli_comparison")
ENUM_PATH = Path("data/decarli_exact_enumeration.csv")
COMP_PATH = Path("data/Symbolic_comparison_task.xlsx")
PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations_poisson.csv")

N_SAMPLES = 5000
SEED = 73
N_MIN = 1
N_SUPPORT_MAX = 18

DURATIONS_TO_TEST = [160, 320, 640]


# ── Shared model infrastructure ──────────────────────────────────────────


def _load_fitted_params(duration_ms: int) -> dict:
    df = pd.read_csv(PARAMS_CSV)
    row = df[df["duration_ms"] == duration_ms].iloc[0]
    theta = {
        "sigma_c": row["sigma_c"],
        "tau_in": row["tau_in"],
        "s0": row["s0"],
        "lambda0": row["lambda0"],
    }
    theta["beta"] = FIXED_BETA or 1.0
    theta["kappa"] = FIXED_KAPPA or 1.0
    theta["eta"] = FIXED_ETA or 1.0
    return theta


def _setup_wells(theta: dict) -> dict:
    integers = torch.arange(N_MIN, N_SUPPORT_MAX + 1, dtype=torch.float32)
    freq = torch.log10(1.0 + 1.0 / integers)
    sigma_sq, log_sigma = None, None
    if WELL_TYPE == "gaussian":
        sigma = benford_sigma(integers, theta["sigma_c"], theta["beta"])
        sigma_sq = sigma.square()
        log_sigma = sigma.log()
    return {
        "integers": integers,
        "mu": integers,
        "sigma_sq": sigma_sq,
        "log_sigma": log_sigma,
        "log_freq": freq.log() if BENFORD_PRIOR else torch.zeros_like(integers),
        "inv_tau_sq": 1.0 / theta["tau_in"] ** 2,
    }


def _sample_evidence(n_true, theta, n_samples, generator):
    noise_std = evidence_noise_std(n_true, theta)
    return float(n_true) + noise_std * torch.randn(n_samples, generator=generator)


@torch.no_grad()
def _process(n_true, theta, wells, n_samples, generator):
    y0 = _sample_evidence(n_true, theta, n_samples, generator)
    out = cleanup(
        y0,
        wells["mu"],
        wells["log_freq"],
        theta["kappa"],
        wells["inv_tau_sq"],
        theta["eta"],
        FIXED["max_steps"],
        FIXED["tol"],
        sigma_c=theta["sigma_c"],
        sigma_sq=wells["sigma_sq"],
        log_sigma=wells["log_sigma"],
    )
    probs = out["probs"]
    expected_n = (probs * wells["integers"]).sum(dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    return {
        "n_hat_idx": out["n_hat_idx"],
        "expected_n": expected_n,
        "entropy": entropy,
    }


# ── 1. Exact enumeration ─────────────────────────────────────────────────


def load_enumeration() -> pd.DataFrame:
    """Group means (± SE) by numerosity × group."""
    df = pd.read_csv(ENUM_PATH)
    agg = (
        df.groupby(["Group", "Numerosity"])
        .agg(
            mean_acc=("Accuracy", "mean"),
            se_acc=("Accuracy", "sem"),
            mean_rt=("RTs", "mean"),
            se_rt=("RTs", "sem"),
            n_subjects=("Sub", "nunique"),
        )
        .reset_index()
    )
    agg["mean_acc_pct"] = agg["mean_acc"] * 100
    agg["se_acc_pct"] = agg["se_acc"] * 100
    return agg


def predict_enumeration(theta: dict) -> pd.DataFrame:
    """Exact accuracy and entropy for numerosities 1-8."""
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)
    integers = wells["integers"]

    records: list[dict] = []
    for n in range(1, 9):
        out = _process(n, theta, wells, N_SAMPLES, g)
        map_vals = integers[out["n_hat_idx"]].long()
        exact_acc = (map_vals == n).float().mean().item() * 100
        mean_entropy = out["entropy"].mean().item()
        records.append(
            {
                "Numerosity": n,
                "exact_acc": exact_acc,
                "mean_entropy": mean_entropy,
            }
        )
    return pd.DataFrame(records)


# ── 2. Symbolic comparison ───────────────────────────────────────────────


def load_comparison() -> pd.DataFrame:
    """Group means (± SE) by distance × group."""
    df = pd.read_excel(COMP_PATH)
    agg = (
        df.groupby(["Group", "Distance"])
        .agg(
            mean_acc=("Accuracy", "mean"),
            se_acc=("Accuracy", "sem"),
            mean_rt=("RTs", "mean"),
            se_rt=("RTs", "sem"),
            n_subjects=("Sub", "nunique"),
        )
        .reset_index()
    )
    agg["mean_acc_pct"] = agg["mean_acc"] * 100
    agg["se_acc_pct"] = agg["se_acc"] * 100
    return agg


def predict_comparison_by_distance(theta: dict) -> pd.DataFrame:
    """Model accuracy and entropy averaged over all pairs at each distance.

    Uses all digit pairs (1-9) at each distance, fully crossed.
    """
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)

    cache: dict[int, dict] = {}
    for n in range(1, 10):
        cache[n] = _process(n, theta, wells, N_SAMPLES, g)

    records_by_dist: dict[int, list[dict]] = {}
    for n_s in range(1, 10):
        for n_l in range(n_s + 1, 10):
            d = n_l - n_s
            acc = (
                cache[n_l]["expected_n"] > cache[n_s]["expected_n"]
            ).float().mean().item() * 100
            ent = (
                cache[n_s]["entropy"].mean().item()
                + cache[n_l]["entropy"].mean().item()
            ) / 2
            records_by_dist.setdefault(d, []).append({"acc": acc, "ent": ent})

    rows: list[dict] = []
    for d in sorted(records_by_dist):
        accs = [r["acc"] for r in records_by_dist[d]]
        ents = [r["ent"] for r in records_by_dist[d]]
        rows.append(
            {
                "Distance": d,
                "mean_acc": np.mean(accs),
                "mean_entropy": np.mean(ents),
                "n_pairs": len(accs),
            }
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Empirical data ────────────────────────────────────────────────────
    enum_emp = load_enumeration()
    comp_emp = load_comparison()

    print("=" * 60)
    print("  DeCarli: Exact Enumeration")
    print("=" * 60)
    print(enum_emp.to_string(index=False), flush=True)

    print("\n" + "=" * 60)
    print("  DeCarli: Symbolic Comparison")
    print("=" * 60)
    print(comp_emp.to_string(index=False), flush=True)

    # ── Model predictions ─────────────────────────────────────────────────
    enum_model: dict[int, pd.DataFrame] = {}
    comp_model: dict[int, pd.DataFrame] = {}

    for dur in DURATIONS_TO_TEST:
        print(f"\nPredicting with {dur} ms ...", flush=True)
        theta = _load_fitted_params(dur)
        enum_model[dur] = predict_enumeration(theta)
        comp_model[dur] = predict_comparison_by_distance(theta)

    print("\nModel enumeration predictions (640 ms):")
    print(enum_model[640].to_string(index=False), flush=True)
    print("\nModel comparison predictions (640 ms):")
    print(comp_model[640].to_string(index=False), flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────
    from scripts.plots import plot_decarli_enumeration, plot_decarli_comparison

    plot_decarli_enumeration(enum_emp, enum_model, OUTPUT_DIR)
    plot_decarli_comparison(comp_emp, comp_model, OUTPUT_DIR)

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
