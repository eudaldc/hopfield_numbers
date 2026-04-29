#!/usr/bin/env python
"""Compare model same/different predictions against match-to-sample data.

The model was fitted on Piantadosi estimation data (adults, numbers 1-15).
Here we test whether its same/different discrimination predictions match
observed accuracy from a match-to-sample task with children (grades 1-2,
sample numerosities 4-6, target numerosities 4-8).

Decision rule: process sample and target through independent attractor
cleanup channels, compare MAP estimates (argmax of posterior).  "Same"
if MAP(sample) = MAP(target), "different" otherwise.  Zero free parameters.

All numerosities (4-8) fall within the model's fitted range, so this is
an interpolation test (unlike the Reynvoet extrapolation to 10-40).
The model was fitted to adults; performance of grade 1-2 children will
be lower, but the *pattern* across ratios/pairs is the prediction of
interest.
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

OUTPUT_DIR = Path("output/mts_comparison")
DATA_PATH = Path("data/processed/match_to_sample.parquet")
PARAMS_CSV = Path("output/piantadosi_fit/summary_all_durations_poisson.csv")

N_SAMPLES = 5000
SEED = 73
N_MIN = 1
N_SUPPORT_MAX = 18

MTS_NUMBERS = [4, 5, 6, 7, 8]
DURATIONS_TO_TEST = [160, 320, 640]


# ── Empirical data ───────────────────────────────────────────────────────


def load_mts_empirical(
    grade: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load match-to-sample trial data and summarise.

    Parameters
    ----------
    grade : filter to a single grade (1 or 2), or None for all.

    Returns
    -------
    by_pair : per-(sample, target) accuracy
    by_ratio : "different" trial accuracy averaged across directions
    by_sample : overall accuracy grouped by sample numerosity
    """
    df = pd.read_parquet(DATA_PATH)
    if grade is not None:
        df = df[df["grade"] == grade]

    # ── per-pair ──────────────────────────────────────────────────────────
    by_pair = (
        df.groupby(["sample_true_n", "target_true_n"])
        .agg(
            n_trials=("correct", "count"),
            accuracy=("correct", "mean"),
            se=("correct", "sem"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    by_pair["same"] = by_pair["sample_true_n"] == by_pair["target_true_n"]
    by_pair["ratio"] = (
        by_pair[["sample_true_n", "target_true_n"]].max(axis=1)
        / by_pair[["sample_true_n", "target_true_n"]].min(axis=1)
    )
    by_pair["accuracy_pct"] = by_pair["accuracy"] * 100
    by_pair["se_pct"] = by_pair["se"] * 100

    # ── by ratio (different trials, both directions pooled) ───────────────
    diff = df[df["sample_true_n"] != df["target_true_n"]].copy()
    diff["ratio"] = (
        diff[["sample_true_n", "target_true_n"]].max(axis=1)
        / diff[["sample_true_n", "target_true_n"]].min(axis=1)
    )
    by_ratio = (
        diff.groupby("ratio")["correct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    by_ratio.columns = ["ratio", "mean_acc", "se_acc", "n_trials"]
    by_ratio["mean_acc"] *= 100
    by_ratio["se_acc"] *= 100

    # ── by sample numerosity (overall accuracy) ───────────────────────────
    by_sample = (
        df.groupby("sample_true_n")["correct"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    by_sample.columns = ["sample_n", "mean_acc", "se_acc", "n_trials"]
    by_sample["mean_acc"] *= 100
    by_sample["se_acc"] *= 100

    return by_pair, by_ratio, by_sample


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
    return {"n_hat_idx": out["n_hat_idx"], "probs": out["probs"]}


def predict_mts(theta: dict, pairs: list[tuple[int, int, bool]]) -> pd.DataFrame:
    """Predict accuracy (with SE) for every (sample, target) pair."""
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)

    records: list[dict] = []
    for n_sample, n_target, is_same in pairs:
        out_s = _process(n_sample, theta, wells, N_SAMPLES, g)
        out_t = _process(n_target, theta, wells, N_SAMPLES, g)

        model_same = (out_s["n_hat_idx"] == out_t["n_hat_idx"]).float()

        if is_same:
            correct = model_same
        else:
            correct = 1.0 - model_same

        acc = correct.mean().item() * 100
        se = correct.std().item() / np.sqrt(N_SAMPLES) * 100

        ratio = max(n_sample, n_target) / min(n_sample, n_target)
        records.append({
            "sample_n": n_sample,
            "target_n": n_target,
            "same": is_same,
            "ratio": ratio,
            "accuracy": acc,
            "se": se,
        })

    return pd.DataFrame(records)


def aggregate_model_by_ratio(model_df: pd.DataFrame) -> pd.DataFrame:
    """Average model predictions across pair directions per ratio."""
    diff = model_df[~model_df["same"]].copy()
    agg = diff.groupby("ratio").agg(
        mean_acc=("accuracy", "mean"),
        se_acc=("accuracy", lambda x: np.sqrt(np.mean(x.map(lambda v: (v / 100 * (1 - v / 100)) / N_SAMPLES))) * 100),
    ).reset_index()
    return agg


def predict_sample_effect(
    model_df: pd.DataFrame, by_pair_emp: pd.DataFrame,
) -> pd.DataFrame:
    """Model accuracy grouped by sample numerosity, weighted by trial counts."""
    merged = by_pair_emp[["sample_true_n", "target_true_n", "n_trials"]].merge(
        model_df.rename(columns={"sample_n": "sample_true_n", "target_n": "target_true_n"}),
        on=["sample_true_n", "target_true_n"],
    )
    records: list[dict] = []
    for sn, sub in merged.groupby("sample_true_n"):
        weights = sub["n_trials"].values
        accs = sub["accuracy"].values
        mean_acc = np.average(accs, weights=weights)
        records.append({"sample_n": int(sn), "mean_acc": mean_acc})
    return pd.DataFrame(records)


# ── Diagnostic: MAP confusion matrix ─────────────────────────────────────


@torch.no_grad()
def compute_confusion_matrix(theta: dict) -> pd.DataFrame:
    """For each true n in MTS_NUMBERS, compute P(MAP = k) for all k."""
    wells = _setup_wells(theta)
    g = torch.Generator().manual_seed(SEED)
    integers = wells["integers"]

    rows: list[dict] = []
    for n_true in MTS_NUMBERS:
        out = _process(n_true, theta, wells, N_SAMPLES, g)
        map_vals = integers[out["n_hat_idx"]].long()
        for k in MTS_NUMBERS:
            p = (map_vals == k).float().mean().item()
            rows.append({"true_n": n_true, "map_n": k, "prob": p})

    return pd.DataFrame(rows)


def explain_ratio_pattern(cm: pd.DataFrame) -> pd.DataFrame:
    """Use the confusion matrix to compute P(MAP(n_s)=MAP(n_t)) for each pair.

    This gives the model's P(respond "same"), which directly predicts the
    non-monotonic accuracy pattern.
    """
    records: list[dict] = []
    for n_s in MTS_NUMBERS:
        for n_t in MTS_NUMBERS:
            if n_s == n_t:
                continue
            p_same = 0.0
            for k in MTS_NUMBERS:
                p_s_k = cm[(cm["true_n"] == n_s) & (cm["map_n"] == k)]["prob"].values
                p_t_k = cm[(cm["true_n"] == n_t) & (cm["map_n"] == k)]["prob"].values
                if len(p_s_k) and len(p_t_k):
                    p_same += p_s_k[0] * p_t_k[0]
            ratio = max(n_s, n_t) / min(n_s, n_t)
            records.append({
                "sample": n_s, "target": n_t,
                "ratio": ratio,
                "P_same": p_same,
                "predicted_diff_acc": (1 - p_same) * 100,
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Match-to-Sample: Model vs. Empirical")
    print("=" * 60)

    # ── 1. Build pair list (same for all grades) ─────────────────────────
    by_pair_all, _, _ = load_mts_empirical(grade=None)
    pairs = [
        (int(r.sample_true_n), int(r.target_true_n), bool(r.same))
        for _, r in by_pair_all.iterrows()
    ]

    # ── 2. Model predictions (grade-independent) ─────────────────────────
    model_preds: dict[int, pd.DataFrame] = {}
    model_by_ratio: dict[int, pd.DataFrame] = {}
    model_sample_eff: dict[int, pd.DataFrame] = {}

    for dur in DURATIONS_TO_TEST:
        print(f"\nPredicting with {dur} ms parameters ...", flush=True)
        theta = _load_fitted_params(dur)
        for k, v in theta.items():
            print(f"  {k:>10s} = {v:.4f}")
        pred = predict_mts(theta, pairs)
        model_preds[dur] = pred
        model_by_ratio[dur] = aggregate_model_by_ratio(pred)
        model_sample_eff[dur] = predict_sample_effect(pred, by_pair_all)
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

    print("\n  Accuracy explained via confusion matrix:")
    expl = explain_ratio_pattern(cm)
    expl_sorted = expl.sort_values("ratio")
    print(expl_sorted.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)

    print("\n  Key insight: P(MAP_correct) per number:")
    for n in MTS_NUMBERS:
        p = cm[(cm["true_n"] == n) & (cm["map_n"] == n)]["prob"].values[0]
        print(f"    n={n}: P(MAP={n}) = {p:.3f}")

    # ── 4. Per-grade analysis ─────────────────────────────────────────────
    from scripts.plots import plot_mts_comparison

    for grade_label, grade_val in [("all", None), ("grade1", 1), ("grade2", 2)]:
        print(f"\n{'=' * 60}")
        print(f"  Grade: {grade_label}")
        print(f"{'=' * 60}")

        by_pair, by_ratio, by_sample = load_mts_empirical(grade=grade_val)

        print(f"\n  Empirical accuracy by ratio (different trials):")
        print(by_ratio.to_string(index=False), flush=True)

        print(f"\n  Empirical accuracy by sample numerosity:")
        print(by_sample.to_string(index=False), flush=True)

        # Recompute model sample effect with this grade's trial counts
        grade_sample_eff = {}
        for dur in DURATIONS_TO_TEST:
            grade_sample_eff[dur] = predict_sample_effect(
                model_preds[dur], by_pair,
            )

        plot_mts_comparison(
            by_pair, by_ratio, by_sample,
            model_preds, model_by_ratio, grade_sample_eff,
            OUTPUT_DIR, tag=grade_label,
        )

    # ── 5. Save summary CSV ───────────────────────────────────────────────
    summary_rows: list[dict] = []
    _, by_ratio_all, _ = load_mts_empirical(grade=None)
    for _, row in by_ratio_all.iterrows():
        r = row["ratio"]
        entry = {"ratio": r, "empirical": row["mean_acc"]}
        for dur in DURATIONS_TO_TEST:
            mbr = model_by_ratio[dur]
            match = mbr[mbr["ratio"].round(6) == round(r, 6)]
            entry[f"model_{dur}ms"] = match["mean_acc"].values[0] if len(match) else np.nan
        summary_rows.append(entry)

    csv_path = OUTPUT_DIR / "mts_comparison.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
