"""Apply the NEW winner model (super-linear Gaussian attractor, free-β + clean-up)
to the De Carli exact-enumeration data (SM5), non-destructively.

The manuscript's SM5 fits the *old* Poisson √n model to each group's accuracy + RT
curves (n = 1..7) and compares DD vs control via derived quantities (n_crit = 1/c,
σ(n)). Here we refit each group with the winning `gaussian_freebeta_attractor`
landscape (reusing the validated `compute_grid` from the trial-level pipeline) and
re-derive the quantities under the new model, where **n_crit is the resolvability
limit σ_n = 1 ⇔ adjacent-well d′ = 1**, not 1/c.

Targets (per group, n = 1..7; n = 8 dropped as a guessing end-effect, as in the
published fit): mean exact accuracy and median RT over the 32 subjects. Fit is a
two-channel profiled-Gaussian (concentrated NLL) on accuracy + RT. Free parameters:
sigma_c, s0, lambda0, beta (landscape), lapse (accuracy floor), t0, rt_scale (RT).

Writes NEW files under output/decarli_fit/ with a `gaussian_freebeta_` prefix; does
NOT touch the config, fit_helpers, or the published Poisson outputs.

    .venv/bin/python -m scripts.decarli_gaussian_freebeta
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

SPEC = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
DATA = _ROOT / "data" / "decarli_exact_enumeration.csv"
OUT_DIR = _ROOT / "output" / "decarli_fit"
NVALS = np.arange(1, 8)                       # n = 1..7 (drop n=8 end-effect)
GROUPS = ("DD", "contr")

# [sigma_c, s0, lambda0, beta, lapse, t0, rt_scale]
BOUNDS = [(0.001, 0.5), (1e-6, 0.1), (0.1, 4.0), (0.1, 6.0),
          (1e-4, 0.5), (0.05, 1.0), (0.0, 5.0)]
PNAMES = ("sigma_c", "s0", "lambda0", "beta", "lapse", "t0", "rt_scale")


def group_targets(df: pd.DataFrame, group: str) -> tuple[np.ndarray, np.ndarray]:
    g = df[df["Group"] == group]
    acc, rt = [], []
    for n in NVALS:
        sub = g[g["Numerosity"] == n]
        acc.append(float(sub["Accuracy"].mean()))          # proportion exact
        rt.append(float(sub["RTs"].median()))              # seconds
    return np.array(acc), np.array(rt)


def predict(vec: np.ndarray, quad: T.Quadrature) -> tuple[np.ndarray, np.ndarray]:
    theta = dict(zip(("sigma_c", "s0", "lambda0", "beta"), vec[:4]))
    lapse, t0, rt_scale = vec[4], vec[5], vec[6]
    grid = T.compute_grid(SPEC, theta, quad, n_values=NVALS.astype(float))
    p_exact = np.array([grid["p_resp"][i, int(n) - 1] for i, n in enumerate(NVALS)])
    p_exact = (1.0 - lapse) * p_exact + lapse / T.N_WELLS       # lapse floor
    rt = t0 + rt_scale * grid["EH"]
    return p_exact, rt


def neg_ll(vec: np.ndarray, acc: np.ndarray, rt: np.ndarray,
           quad: T.Quadrature) -> float:
    try:
        p_exact, pred_rt = predict(vec, quad)
    except (FloatingPointError, ValueError):
        return 1e18
    if not (np.all(np.isfinite(p_exact)) and np.all(np.isfinite(pred_rt))):
        return 1e18
    n = len(acc)
    mse_acc = float(np.mean((p_exact - acc) ** 2))
    mse_rt = float(np.mean((pred_rt - rt) ** 2))
    if mse_acc <= 0 or mse_rt <= 0:
        return 1e18
    # two-channel concentrated (profiled) Gaussian NLL
    return 0.5 * n * (np.log(mse_acc) + np.log(mse_rt))


def fit_group(acc: np.ndarray, rt: np.ndarray, quad: T.Quadrature,
              seed: int) -> dict:
    de = differential_evolution(
        neg_ll, bounds=BOUNDS, args=(acc, rt, quad), maxiter=80, popsize=15,
        tol=1e-7, mutation=(0.5, 1.0), recombination=0.85, seed=seed,
        init="latinhypercube", polish=False, updating="deferred", workers=1)
    pol = minimize(neg_ll, de.x, args=(acc, rt, quad), method="Nelder-Mead",
                   bounds=BOUNDS, options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-6})
    vec = pol.x if pol.fun <= de.fun else de.x
    return dict(zip(PNAMES, vec)) | {"neg_ll": float(min(pol.fun, de.fun)),
                                     "vec": vec}


def sigma_well(theta: dict, n: np.ndarray) -> np.ndarray:
    freq = np.log10(1.0 + 1.0 / np.asarray(n, float))
    return np.power(theta["sigma_c"] / freq, theta["beta"])


def ncrit(theta: dict, level: float = 1.0) -> float:
    ns = np.linspace(1.0, 60.0, 60000)
    s = sigma_well(theta, ns)
    idx = np.where(s >= level)[0]
    if idx.size == 0 or idx[0] == 0:
        return float("nan")
    i = idx[0]
    return float(ns[i - 1] + (level - s[i - 1]) * (ns[i] - ns[i - 1]) /
                 (s[i] - s[i - 1]))


def evidence_sd(theta: dict, n: float) -> float:
    return float(np.sqrt(T.S00 ** 2 + theta["s0"] * n ** theta["lambda0"]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quad = T.make_quadrature(401, kind="grid")
    df = pd.read_csv(DATA)

    fits, rows = {}, []
    for gi, grp in enumerate(GROUPS):
        acc, rt = group_targets(df, grp)
        f = fit_group(acc, rt, quad, seed=17 + gi)
        fits[grp] = (f, acc, rt)
        theta = {k: f[k] for k in ("sigma_c", "s0", "lambda0", "beta")}
        rows.append({
            "group": grp,
            **{k: f[k] for k in PNAMES},
            "n_crit_sigma1": ncrit(theta, 1.0),
            "n_crit_sigma2": ncrit(theta, 2.0),
            "sigma_n1": evidence_sd(theta, 1), "sigma_n4": evidence_sd(theta, 4),
            "sigma_n7": evidence_sd(theta, 7),
            "well_sigma_n1": float(sigma_well(theta, np.array([1]))[0]),
            "well_sigma_n4": float(sigma_well(theta, np.array([4]))[0]),
            "well_sigma_n7": float(sigma_well(theta, np.array([7]))[0]),
            "neg_ll": f["neg_ll"],
        })
        print(f"[{grp}] beta={f['beta']:.2f} sigma_c={f['sigma_c']:.4f} "
              f"s0={f['s0']:.4g} lambda0={f['lambda0']:.2f} t0={f['t0']:.3f} "
              f"lapse={f['lapse']:.3f} -> n_crit(σ=1)={ncrit(theta):.2f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gaussian_freebeta_fitted.csv", index=False)

    # DD vs contr point-difference table for the key derived quantities
    dd, ct = out[out.group == "DD"].iloc[0], out[out.group == "contr"].iloc[0]
    comp = pd.DataFrame([
        {"quantity": q, "DD": dd[q], "contr": ct[q], "diff_DD_minus_contr": dd[q] - ct[q]}
        for q in ("n_crit_sigma1", "n_crit_sigma2", "beta", "sigma_c",
                  "sigma_n1", "sigma_n4", "sigma_n7")
    ])
    comp.to_csv(OUT_DIR / "gaussian_freebeta_group_comparison.csv", index=False)

    # fit figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    colors = {"DD": "#c0392b", "contr": "#2c7fb8"}
    for grp in GROUPS:
        f, acc, rt = fits[grp]
        p_exact, pred_rt = predict(f["vec"], quad)
        axes[0].plot(NVALS, 100 * acc, "o", color=colors[grp], ms=6, label=f"{grp} data")
        axes[0].plot(NVALS, 100 * p_exact, "-", color=colors[grp], lw=1.8)
        axes[1].plot(NVALS, rt, "o", color=colors[grp], ms=6, label=f"{grp} data")
        axes[1].plot(NVALS, pred_rt, "-", color=colors[grp], lw=1.8)
        nc = ncrit({k: f[k] for k in ("sigma_c", "s0", "lambda0", "beta")})
        axes[0].axvline(nc, color=colors[grp], ls=":", lw=1.0)
    axes[0].set_xlabel("numerosity"); axes[0].set_ylabel("% exact")
    axes[0].set_title("De Carli accuracy — winner model"); axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("numerosity"); axes[1].set_ylabel("median RT (s)")
    axes[1].set_title("De Carli RT — winner model"); axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("De Carli DD vs control — super-linear Gaussian attractor (free β)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gaussian_freebeta_fit.png", dpi=180)
    plt.close(fig)

    print("\n=== DD vs control (point estimates) ===")
    print(comp.round(3).to_string(index=False))
    print(f"\nSaved gaussian_freebeta_fitted.csv, _group_comparison.csv, _fit.png "
          f"in {OUT_DIR}")


if __name__ == "__main__":
    main()
