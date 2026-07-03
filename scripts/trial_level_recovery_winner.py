"""Trial-level parameter recovery for the winning super-linear Gaussian attractor.

SM3 of the current manuscript establishes identifiability for the OLD summary-curve
Hopfield-Poisson pipeline; the revision centres on the trial-level winner
(``gaussian_freebeta_attractor``), so recovery must be re-established for the new
model and likelihood. Design mirrors SM3: 50 synthetic datasets per exposure
duration, generating parameters jittered around the fitted solution, refit with the
exact headline pipeline (differential evolution maxiter 35 / popsize 10 on the
201-node grid + Nelder-Mead polish).

Synthetic counts are sampled from the accurate 1601-node grid with the same
per-(duration, numerosity) trial counts as the real data, including the lapse
channel. Recovery is reported for the five response parameters AND for the derived
quantities the manuscript actually interprets: n_crit (sigma_n = 1 crossing), the
well width sigma_3, and the sensory sd at n = 5.

    .venv/bin/python -m scripts.trial_level_recovery_winner            # full run (parallel)
    .venv/bin/python -m scripts.trial_level_recovery_winner --smoke    # 1 rep/duration, tiny optimiser
    .venv/bin/python -m scripts.trial_level_recovery_winner --replot   # summary + figure from saved CSV
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

OUT_CSV = T.OUT_DIR / "recovery_winner.csv"
OUT_SUMMARY = T.OUT_DIR / "recovery_winner_summary.csv"
OUT_PNG = T.OUT_DIR / "recovery_winner.png"

N_REPS = 50
N_WORKERS = 20
PARAMS = ("sigma_c", "s0", "lambda0", "beta", "lapse")
# quantities reported on log scale (scale parameters spanning decades)
LOG_SCALE = ("sigma_c", "s0", "lapse", "sigma3")
# multiplicative (log-uniform) jitter half-width per parameter
JITTER = {"sigma_c": 0.35, "s0": 0.50, "lambda0": 0.20, "beta": 0.25, "lapse": 0.40}
DERIVED = ("ncrit", "sigma3", "sensory_sd5")

_W: dict = {}


def _sigma_n(sigma_c: float, beta: float, n: np.ndarray) -> np.ndarray:
    freq = np.log10(1.0 + 1.0 / np.asarray(n, float))
    return np.power(sigma_c / freq, beta)


def ncrit_sigma1(sigma_c: float, beta: float) -> float:
    """Smallest continuous n >= 1 where sigma_n = 1 (adjacent-well d' = 1)."""
    ns = np.linspace(1.0, 60.0, 60000)
    s = _sigma_n(sigma_c, beta, ns)
    above = np.where(s >= 1.0)[0]
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return 1.0
    x0, x1, y0, y1 = ns[i - 1], ns[i], s[i - 1], s[i]
    return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))


def _init_worker(fit_nodes: int, maxiter: int, popsize: int) -> None:
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    _W["r_max"] = r_max
    _W["data"] = T.build_duration_data(trial, r_max)
    _W["quad_fit"] = T.make_quadrature(fit_nodes, kind="grid")
    _W["quad_gen"] = T.make_quadrature(1601, kind="grid")
    _W["theta"] = pd.read_csv(T.OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    _W["bounds"] = dict(zip(spec.param_names, spec.bounds)) | {"lapse": T.LAPSE_BOUNDS}
    _W["maxiter"] = maxiter
    _W["popsize"] = popsize


def _one(task: tuple[int, int]) -> dict:
    d, rep = task
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    th = _W["theta"]
    rng = np.random.default_rng(1000 * d + rep)

    gen = {}
    for p in PARAMS:
        lo, hi = _W["bounds"][p]
        val = float(th.loc[d, p]) * float(np.exp(rng.uniform(-JITTER[p], JITTER[p])))
        gen[p] = float(np.clip(val, lo * 1.02, hi * 0.98))

    grid = T.compute_grid(spec, {p: gen[p] for p in spec.param_names}, _W["quad_gen"])
    pf = T.apply_lapse(grid["p_resp"], gen["lapse"], _W["r_max"])
    real = _W["data"][d]
    counts = np.zeros_like(real.counts)
    for i in range(real.counts.shape[0]):
        pi = pf[i] / pf[i].sum()
        counts[i] = rng.multinomial(int(real.counts[i].sum()), pi)
    synth = replace(real, counts=counts)

    rf = T.fit_response(spec, synth, _W["quad_fit"], _W["r_max"],
                        maxiter=_W["maxiter"], popsize=_W["popsize"],
                        seed=T.SEED + 7919 * rep + d)
    rec = {p: float(rf["theta"][p]) for p in spec.param_names} | {"lapse": float(rf["lapse"])}

    row: dict = {"duration_ms": d, "rep": rep}
    row |= {f"gen_{p}": gen[p] for p in PARAMS}
    row |= {f"rec_{p}": rec[p] for p in PARAMS}
    row["gen_ncrit"] = ncrit_sigma1(gen["sigma_c"], gen["beta"])
    row["rec_ncrit"] = ncrit_sigma1(rec["sigma_c"], rec["beta"])
    row["gen_sigma3"] = float(_sigma_n(gen["sigma_c"], gen["beta"], np.array([3.0]))[0])
    row["rec_sigma3"] = float(_sigma_n(rec["sigma_c"], rec["beta"], np.array([3.0]))[0])
    row["gen_sensory_sd5"] = float(T.evidence_sd(np.array([5.0]), gen["s0"], gen["lambda0"])[0])
    row["rec_sensory_sd5"] = float(T.evidence_sd(np.array([5.0]), rec["s0"], rec["lambda0"])[0])
    row["rec_logL"] = float(rf["logL"])
    return row


def _pair(df: pd.DataFrame, q: str) -> tuple[np.ndarray, np.ndarray]:
    g, r = df[f"gen_{q}"].to_numpy(float), df[f"rec_{q}"].to_numpy(float)
    if q in LOG_SCALE:
        return np.log10(g), np.log10(r)
    return g, r


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for q in PARAMS + DERIVED:
        g, r = _pair(df, q)
        ok = np.isfinite(g) & np.isfinite(r)
        g, r = g[ok], r[ok]
        ss_res = float(np.sum((r - g) ** 2))
        ss_tot = float(np.sum((g - g.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rho = float(spearmanr(g, r).statistic)
        rows.append({"quantity": q, "scale": "log10" if q in LOG_SCALE else "linear",
                     "n": int(ok.sum()), "R2_identity": r2, "spearman_rho": rho})
    return pd.DataFrame(rows)


def error_corr(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    errs = {}
    for q in PARAMS:
        g, r = _pair(df, q)
        errs[q] = r - g
    e = pd.DataFrame(errs)
    mat = np.zeros((len(PARAMS), len(PARAMS)))
    for i, a in enumerate(PARAMS):
        for j, b in enumerate(PARAMS):
            mat[i, j] = spearmanr(e[a], e[b]).statistic
    return pd.DataFrame(mat, index=PARAMS, columns=PARAMS)


def make_figure(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quantities = list(PARAMS) + ["ncrit", "sigma3"]
    labels = {"sigma_c": r"$\sigma_c$ (log$_{10}$)", "s0": r"$s_0$ (log$_{10}$)",
              "lambda0": r"$\lambda_0$", "beta": r"$\beta$",
              "lapse": "lapse (log$_{10}$)", "ncrit": r"$n_{\rm crit}$ ($\sigma_n=1$)",
              "sigma3": r"$\sigma_3$ (log$_{10}$)"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = plt.get_cmap("viridis")
    durs = sorted(df["duration_ms"].unique())
    colors = {d: cmap(i / max(1, len(durs) - 1)) for i, d in enumerate(durs)}
    for ax, q in zip(axes.flat, quantities):
        g, r = _pair(df, q)
        for d in durs:
            m = (df["duration_ms"] == d).to_numpy()
            ax.scatter(g[m], r[m], s=14, alpha=0.7, color=colors[d], label=f"{d} ms")
        lo = np.nanmin([g.min(), r.min()])
        hi = np.nanmax([g.max(), r.max()])
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
        ax.set_xlabel("generating")
        ax.set_ylabel("recovered")
        ax.set_title(labels[q])
    ax = axes.flat[-1]
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(PARAMS)), PARAMS, rotation=45)
    ax.set_yticks(range(len(PARAMS)), PARAMS)
    for i in range(len(PARAMS)):
        for j in range(len(PARAMS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("recovery-error correlations (Spearman)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Trial-level parameter recovery — super-linear Gaussian attractor (winner)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Saved {OUT_PNG}")


def report(df: pd.DataFrame) -> None:
    summary = summarize(df)
    corr = error_corr(df)
    summary.to_csv(OUT_SUMMARY, index=False)
    pd.set_option("display.width", 200)
    print("\n=== Recovery summary (identity R^2 / Spearman rho) ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== Recovery-error correlation matrix (Spearman) ===")
    print(corr.round(2).to_string())
    print(f"\nSaved {OUT_SUMMARY}")
    make_figure(df, corr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 rep/duration, tiny optimiser budget")
    ap.add_argument("--replot", action="store_true", help="summary + figure from saved CSV")
    args = ap.parse_args(argv)

    if args.replot:
        report(pd.read_csv(OUT_CSV))
        return

    n_reps = 1 if args.smoke else N_REPS
    fit_nodes = 101 if args.smoke else 201
    maxiter = 8 if args.smoke else 35
    popsize = 6 if args.smoke else 10
    out_csv = OUT_CSV.with_name("recovery_winner_smoke.csv") if args.smoke else OUT_CSV

    tasks = [(d, rep) for d in T.DURATIONS for rep in range(n_reps)]
    t0 = time.time()
    rows = []
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(fit_nodes, maxiter, popsize)) as pool:
        for i, row in enumerate(pool.imap_unordered(_one, tasks), 1):
            rows.append(row)
            print(f"  [{i}/{len(tasks)}] dur={row['duration_ms']}ms rep={row['rep']} "
                  f"gen_beta={row['gen_beta']:.2f} rec_beta={row['rec_beta']:.2f} "
                  f"gen_ncrit={row['gen_ncrit']:.1f} rec_ncrit={row['rec_ncrit']:.1f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows).sort_values(["duration_ms", "rep"])
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    if not args.smoke:
        report(df)


if __name__ == "__main__":
    main()
