#!/usr/bin/env python
"""Compare model predictions against dot-enumeration verification data.

Children see a dot array (true_n dots, 1-9) and an Arabic digit
(shown_digit, 1-9) and respond "same" or "different".  Unlike
match-to-sample, only ONE channel is noisy (the dot percept); the
Arabic digit is read perfectly.  So:

  same trial  (true_n == shown_digit):  P(correct) = P(MAP = true_n)
  diff trial  (true_n != shown_digit):  P(correct) = 1 - P(MAP = shown_digit | true_n)

Children have unlimited time and most likely count, so the model
(fitted on brief-exposure estimation) will underpredict accuracy for
larger numerosities.  The comparison targets *patterns*: which pairs
are harder, distance effects, and the numerosity-dependent accuracy
gradient.

The model uses Poisson wells (well width proportional to n via sigma_c).
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
    benford_sigma, cleanup, FIXED, WELL_TYPE, BENFORD_PRIOR,
    FIXED_BETA, FIXED_KAPPA, FIXED_ETA,
    evidence_noise_std,
)

# ── Settings ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output/dot_enum_comparison")
DATA_PATH = Path("data/processed/dot_numeration.parquet")
PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations_poisson.csv")

N_SAMPLES = 5000
SEED = 73
N_MIN = 1
N_SUPPORT_MAX = 18
DOT_NUMBERS = list(range(1, 10))

DURATIONS_TO_TEST = [160, 320, 640]


# ── Empirical data ───────────────────────────────────────────────────────


def load_dot_enum_empirical(
    grade: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dot-enumeration data and summarise.

    Returns
    -------
    by_pair : per-(true_n, shown_digit) accuracy and RT
    by_distance : "different" trial accuracy by |true_n - shown_digit|
    by_same_n : "same" trial accuracy by numerosity
    rt_by_n : mean RT by true_n (all trials)
    """
    df = pd.read_parquet(DATA_PATH)
    if grade is not None:
        df = df[df["grade"] == grade]

    # ── per pair ──────────────────────────────────────────────────────────
    by_pair = (
        df.groupby(["true_n", "shown_digit"])
        .agg(
            n_trials=("correct", "count"),
            accuracy=("correct", "mean"),
            se=("correct", "sem"),
            mean_rt=("rt_ms", "mean"),
            se_rt=("rt_ms", "sem"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    by_pair["same"] = by_pair["true_n"] == by_pair["shown_digit"]
    by_pair["distance"] = (by_pair["true_n"] - by_pair["shown_digit"]).abs()
    by_pair["accuracy_pct"] = by_pair["accuracy"] * 100
    by_pair["se_pct"] = by_pair["se"] * 100

    # ── different trials by distance ──────────────────────────────────────
    diff = df[df["true_n"] != df["shown_digit"]].copy()
    diff["distance"] = (diff["true_n"] - diff["shown_digit"]).abs()
    by_distance = (
        diff.groupby("distance")["correct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    by_distance.columns = ["distance", "mean_acc", "se_acc", "n_trials"]
    by_distance["mean_acc"] *= 100
    by_distance["se_acc"] *= 100

    # ── same trials by numerosity ─────────────────────────────────────────
    same = df[df["true_n"] == df["shown_digit"]].copy()
    by_same_n = (
        same.groupby("true_n")["correct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    by_same_n.columns = ["n", "mean_acc", "se_acc", "n_trials"]
    by_same_n["mean_acc"] *= 100
    by_same_n["se_acc"] *= 100

    # ── RT by true_n (all trials) ─────────────────────────────────────────
    rt_by_n = (
        df.groupby("true_n")["rt_ms"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    rt_by_n.columns = ["n", "mean_rt", "se_rt"]

    return by_pair, by_distance, by_same_n, rt_by_n


# ── Model prediction ──────────────────────────────────────────────────────


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
        y0, wells["mu"], wells["log_freq"],
        theta["kappa"], wells["inv_tau_sq"], theta["eta"],
        FIXED["max_steps"], FIXED["tol"],
        sigma_c=theta["sigma_c"],
        sigma_sq=wells["sigma_sq"], log_sigma=wells["log_sigma"],
    )
    probs = out["probs"]
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    return {
        "n_hat_idx": out["n_hat_idx"],
        "probs": probs,
        "n_steps": out["n_steps"],
        "entropy": entropy,
    }


# ── Confusion matrix ─────────────────────────────────────────────────────


@torch.no_grad()
def compute_confusion_matrix(theta: dict) -> pd.DataFrame:
    """P(MAP = k | true n) for n, k in DOT_NUMBERS."""
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)
    integers = wells["integers"]

    rows: list[dict] = []
    for n_true in DOT_NUMBERS:
        out = _process(n_true, theta, wells, N_SAMPLES, g)
        map_vals = integers[out["n_hat_idx"]].long()
        for k in DOT_NUMBERS:
            p = (map_vals == k).float().mean().item()
            rows.append({"true_n": n_true, "map_n": k, "prob": p})

    return pd.DataFrame(rows)


# ── Pair predictions ─────────────────────────────────────────────────────


def predict_dot_enum(
    theta: dict, pairs: list[tuple[int, int, bool]],
) -> pd.DataFrame:
    """Predict accuracy for every (true_n, shown_digit) pair.

    Only the dot array goes through the noisy attractor channel.
    The Arabic digit is perceived perfectly.
    """
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)
    integers = wells["integers"]

    # pre-compute MAP distributions, steps, and entropy for each unique true_n
    map_distributions: dict[int, torch.Tensor] = {}
    step_distributions: dict[int, torch.Tensor] = {}
    entropy_distributions: dict[int, torch.Tensor] = {}
    for n_true in sorted({p[0] for p in pairs}):
        out = _process(n_true, theta, wells, N_SAMPLES, g)
        map_distributions[n_true] = integers[out["n_hat_idx"]].long()
        step_distributions[n_true] = out["n_steps"].float()
        entropy_distributions[n_true] = out["entropy"]

    records: list[dict] = []
    for true_n, shown_digit, is_same in pairs:
        map_vals = map_distributions[true_n]

        if is_same:
            correct = (map_vals == shown_digit).float()
        else:
            correct = (map_vals != shown_digit).float()

        acc = correct.mean().item() * 100
        se = correct.std().item() / np.sqrt(N_SAMPLES) * 100
        mean_steps = step_distributions[true_n].mean().item()
        mean_entropy = entropy_distributions[true_n].mean().item()

        records.append({
            "true_n": true_n,
            "shown_digit": shown_digit,
            "same": is_same,
            "distance": abs(true_n - shown_digit),
            "accuracy": acc,
            "se": se,
            "mean_steps": mean_steps,
            "mean_entropy": mean_entropy,
        })

    return pd.DataFrame(records)


def aggregate_model_by_distance(model_df: pd.DataFrame) -> pd.DataFrame:
    diff = model_df[~model_df["same"]].copy()
    return (
        diff.groupby("distance")
        .agg(mean_acc=("accuracy", "mean"), se_acc=("se", "mean"))
        .reset_index()
    )


def aggregate_model_same(model_df: pd.DataFrame) -> pd.DataFrame:
    same = model_df[model_df["same"]].copy()
    return same[["true_n", "accuracy", "se"]].rename(
        columns={"accuracy": "mean_acc", "true_n": "n", "se": "se_acc"},
    )


def aggregate_model_difficulty_by_n(model_df: pd.DataFrame) -> pd.DataFrame:
    """Mean entropy and steps grouped by true_n (one value per n since
    both only depend on the dot-array channel)."""
    return (
        model_df.drop_duplicates(subset="true_n")
        [["true_n", "mean_steps", "mean_entropy"]]
        .rename(columns={"true_n": "n"})
        .sort_values("n")
    )


# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Dot Enumeration: Model vs. Empirical")
    print("=" * 60)

    # ── 1. Build pair list ────────────────────────────────────────────────
    by_pair_all, _, _, _ = load_dot_enum_empirical(grade=None)
    pairs = [
        (int(r.true_n), int(r.shown_digit), bool(r.same))
        for _, r in by_pair_all.iterrows()
    ]
    print(f"\nUnique pairs to simulate: {len(pairs)}")

    # ── 2. Model predictions ──────────────────────────────────────────────
    model_preds: dict[int, pd.DataFrame] = {}
    model_by_dist: dict[int, pd.DataFrame] = {}
    model_same: dict[int, pd.DataFrame] = {}
    model_difficulty: dict[int, pd.DataFrame] = {}

    for dur in DURATIONS_TO_TEST:
        print(f"\nPredicting with {dur} ms parameters ...", flush=True)
        theta = _load_fitted_params(dur)
        for k, v in theta.items():
            print(f"  {k:>10s} = {v:.4f}")
        pred = predict_dot_enum(theta, pairs)
        model_preds[dur] = pred
        model_by_dist[dur] = aggregate_model_by_distance(pred)
        model_same[dur] = aggregate_model_same(pred)
        model_difficulty[dur] = aggregate_model_difficulty_by_n(pred)
        print(pred.to_string(index=False), flush=True)

    # ── 3. Confusion matrix diagnostic ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MAP confusion matrix (640 ms params)")
    print("=" * 60)

    theta_640 = _load_fitted_params(640)
    cm = compute_confusion_matrix(theta_640)
    pivot = cm.pivot(index="true_n", columns="map_n", values="prob")
    print("\n  P(MAP = k | true n):")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)

    print("\n  Diagonal (same-trial accuracy):")
    for n in DOT_NUMBERS:
        p = cm[(cm["true_n"] == n) & (cm["map_n"] == n)]["prob"].values[0]
        print(f"    n={n}: P(MAP={n}) = {p:.3f}")

    cm.to_csv(OUTPUT_DIR / "confusion_matrix_640ms.csv", index=False)

    # ── 4. Per-grade analysis ─────────────────────────────────────────────
    from scripts.plots import plot_dot_enum_comparison

    for grade_label, grade_val in [("all", None), ("grade1", 1), ("grade2", 2)]:
        print(f"\n{'=' * 60}")
        print(f"  Grade: {grade_label}")
        print(f"{'=' * 60}")

        by_pair, by_distance, by_same_n, rt_by_n = load_dot_enum_empirical(
            grade=grade_val,
        )

        print(f"\n  Empirical accuracy by distance (different trials):")
        print(by_distance.to_string(index=False), flush=True)

        print(f"\n  Empirical 'same' accuracy by numerosity:")
        print(by_same_n.to_string(index=False), flush=True)

        print(f"\n  Empirical mean RT by numerosity:")
        print(rt_by_n.to_string(index=False), flush=True)

        plot_dot_enum_comparison(
            by_pair, by_distance, by_same_n, rt_by_n, cm,
            model_preds, model_by_dist, model_same, model_difficulty,
            OUTPUT_DIR, tag=grade_label,
        )

    # ── 5. Save summary ──────────────────────────────────────────────────
    _, by_dist_all, _, _ = load_dot_enum_empirical(grade=None)
    summary_rows: list[dict] = []
    for _, row in by_dist_all.iterrows():
        d = row["distance"]
        entry = {"distance": int(d), "empirical": row["mean_acc"]}
        for dur in DURATIONS_TO_TEST:
            mbd = model_by_dist[dur]
            match = mbd[mbd["distance"] == d]
            entry[f"model_{dur}ms"] = match["mean_acc"].values[0] if len(match) else np.nan
        summary_rows.append(entry)

    csv_path = OUTPUT_DIR / "dot_enum_comparison.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
