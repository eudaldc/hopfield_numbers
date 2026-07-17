"""Bivariate hierarchical individual-differences model (Fig. 2c reanalysis).

Participant clustering uses ``uniqueid`` (not ``pid``). In the open Cheyette &
Piantadosi (2020) DURATION file, ``uniqueid`` is the session/participant key
used throughout the trial-level pipeline; ``pid`` is a near-duplicate numeric
label (almost 1:1). The manuscript's "N = 100 per condition" follows C&P's
per-experiment analysis after dropping the 10 highest-error participants from
each N=110 experiment. The open CSV pools the three duration-control arms
(NONE / AREA / DENSITY) without that exclusion and is within-subject across
exposures (~366 participants with both 320 and 640 ms).

Model (partial pooling, non-overlapping channels):
  (u_i^n, u_i^w) ~ MVN(0, Σ) with correlation ρ.
  log n_width_{i,d} = α_n[d] + u_i^n
  log w_{i,d}       = α_w[d] + u_i^w
  Small-n trials (n ≤ n_lo): power-law attractor categorical likelihood with
    σ_c = n_width^{-p}, sensory (s0, λ0) and lapse fixed at the group joint
    winner, R_max = 99 (all responses retained).
  Large-n trials (n ≥ n_hi): discretized linear Weber
    r* ~ Normal(γ_d · n, (w · n)^2) on 1..99; γ_d from a pooled MLE.
  Duration fixed effects α_n, α_w; shared RE covariance (τ_n, τ_w, ρ).

Fitting is staged for speed: univariate hierarchical fits for each channel,
then a 1-D search over ρ, then a short joint Nelder–Mead polish.

Cutoffs checked: (n_lo, n_hi) in {(7,8), (8,9), (9,10)}.

    .venv/bin/python -m scripts.trial_level_bivariate_id

Outputs under ``output/trial_level_model_comparison/``:
  bivariate_id_summary.csv, bivariate_id_participants.csv, bivariate_id_audit.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import chi2, norm
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

DURS = (320, 640)
DUR_INDEX = {320: 0, 640: 1}
R_MAX = 99
CUTOFFS = ((7, 8), (8, 9), (9, 10))
PRIMARY = (8, 9)
N_GH = 9
N_WIDTH_GRID = 40
W_GRID = 40
GAMMA_GRID = 21
QUAD_NODES = 151
OUT_DIR = T.OUT_DIR
CACHE = OUT_DIR / "bivariate_id_tables.npz"
OUT_SUMMARY = OUT_DIR / "bivariate_id_summary.csv"
OUT_PART = OUT_DIR / "bivariate_id_participants.csv"
OUT_AUDIT = OUT_DIR / "bivariate_id_audit.txt"


def load_long_trials() -> pd.DataFrame:
    raw = pd.read_csv(T.DATA_PATH)
    sub = raw[(raw["cond"] == "DURATION") & (raw["exp_dur"].isin(DURS))]
    tr = (
        sub.groupby(["uniqueid", "trial_id"], as_index=False)
        .agg(
            exp_dur=("exp_dur", "first"),
            n_shown=("n_shown", "first"),
            n_guess=("n_guess", "first"),
            rt=("rt", "first"),
            pid=("pid", "first"),
            control=("control", "first"),
        )
    )
    tr = tr[(tr["n_shown"] >= 1) & (tr["n_shown"] <= 15) & (tr["rt"] > 0)].copy()
    tr["n_guess"] = np.clip(tr["n_guess"].to_numpy(int), 1, R_MAX)
    both = tr.groupby("uniqueid")["exp_dur"].nunique()
    return tr[tr["uniqueid"].isin(both[both == 2].index)].reset_index(drop=True)


def audit_identity(tr: pd.DataFrame) -> str:
    raw = pd.read_csv(T.DATA_PATH)
    dur = raw[raw["cond"] == "DURATION"]
    lines = [
        "Participant identity and N=100 reconciliation",
        "============================================",
        f"Clustering unit: uniqueid (n = {tr.uniqueid.nunique()})",
        f"pid in subset: n = {tr.pid.nunique()} "
        f"(max uniqueid/pid = {tr.groupby('pid').uniqueid.nunique().max()}, "
        f"max pid/uniqueid = {tr.groupby('uniqueid').pid.nunique().max()})",
        "",
        "Cheyette & Piantadosi (2020) reported four between-experiment arms "
        "(N=110 each, drop 10 highest-error → N=100 per condition). The open "
        "DURATION file pools three control arms (NONE / AREA / DENSITY) without "
        "that exclusion; exposure duration is within participants.",
        "",
        f"Open DURATION uniqueids (all exposures): {dur.uniqueid.nunique()}",
        f"Analysis sample (both 320 and 640 ms): {tr.uniqueid.nunique()} "
        f"uniqueids, {len(tr)} trials",
        "Control arms:",
    ]
    for c, n in tr.groupby("control").uniqueid.nunique().items():
        lines.append(f"  {c}: {n}")
    return "\n".join(lines) + "\n"


def group_winner_params() -> dict:
    th = pd.read_csv(OUT_DIR / "shared_landscape_ladder_theta.csv")
    th = th[th["model_key"] == T.WINNER_KEY].set_index("duration_ms")
    return {
        "pow": float(th.loc[DURS[0], "pow"]),
        "sens": {
            int(d): {
                "s0": float(th.loc[d, "s0"]),
                "lambda0": float(th.loc[d, "lambda0"]),
                "lapse": float(th.loc[d, "lapse"]),
            }
            for d in DURS
        },
    }


def build_or_load_tables(pow_: float, sens: dict, quad: T.Quadrature):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        print(f"Loaded cached tables from {CACHE}", flush=True)
        nw = {320: (z["log_nw"], z["nw320"]), 640: (z["log_nw"], z["nw640"])}
        return nw, (z["log_w"], z["gammas"], z["weber"])

    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    log_nw = np.linspace(np.log(2.5), np.log(22.0), N_WIDTH_GRID)
    nw = {}
    for d in DURS:
        tables = np.empty((N_WIDTH_GRID, 15, R_MAX), dtype=np.float64)
        for i, lnw in enumerate(tqdm(log_nw, desc=f"n_width {d} ms", leave=False)):
            theta = {
                "sigma_c": float(np.exp(lnw)) ** (-pow_),
                "s0": sens[d]["s0"],
                "lambda0": sens[d]["lambda0"],
                "pow": pow_,
            }
            p = np.zeros((15, R_MAX), dtype=np.float64)
            p[:, : T.N_WELLS] = T.compute_grid(spec, theta, quad)["p_resp"]
            tables[i] = np.log(np.clip(p, 1e-300, None))
        nw[d] = (log_nw, tables)

    log_w = np.linspace(np.log(0.03), np.log(0.7), W_GRID)
    gammas = np.linspace(0.40, 1.20, GAMMA_GRID)
    weber = np.empty((W_GRID, GAMMA_GRID, 15, R_MAX), dtype=np.float64)
    edges = np.arange(0.5, R_MAX + 1.5)
    for iw, lw in enumerate(tqdm(log_w, desc="Weber grid", leave=False)):
        w = float(np.exp(lw))
        for ig, g in enumerate(gammas):
            for n in range(1, 16):
                p = np.diff(norm.cdf(edges, loc=g * n, scale=max(w * n, 1e-3)))
                s = p.sum()
                p = np.full(R_MAX, 1.0 / R_MAX) if s <= 0 else p / s
                weber[iw, ig, n - 1] = np.log(np.clip(p, 1e-300, None))

    np.savez(
        CACHE,
        log_nw=log_nw, nw320=nw[320][1], nw640=nw[640][1],
        log_w=log_w, gammas=gammas, weber=weber,
    )
    print(f"Cached tables → {CACHE}", flush=True)
    return nw, (log_w, gammas, weber)


def fit_pooled_gamma(tr, n_hi, sens, weber_pack) -> dict:
    log_w_grid, gammas, table = weber_pack
    out = {}
    for d in DURS:
        sub = tr[(tr.exp_dur == d) & (tr.n_shown >= n_hi)]
        counts = np.zeros((15, R_MAX), dtype=np.float64)
        np.add.at(
            counts,
            (sub.n_shown.to_numpy(int) - 1, sub.n_guess.to_numpy(int) - 1),
            1.0,
        )
        lapse = sens[d]["lapse"]
        best = (np.inf, 0.9)
        for ig, g in enumerate(gammas):
            for iw in range(log_w_grid.size):
                p = (1.0 - lapse) * np.exp(table[iw, ig]) + lapse / R_MAX
                mask = counts > 0
                nll = -float(np.sum(counts[mask] * np.log(np.clip(p[mask], 1e-300, None))))
                if nll < best[0]:
                    best = (nll, float(g))
        out[d] = best[1]
        print(f"  pooled γ({d} ms)={best[1]:.3f}", flush=True)
    return out


def _lin_weights(grid, x):
    x = np.clip(np.asarray(x, dtype=float), grid[0], grid[-1])
    idx = np.searchsorted(grid, x) - 1
    idx = np.clip(idx, 0, grid.size - 2)
    t = (x - grid[idx]) / (grid[idx + 1] - grid[idx])
    return idx.astype(np.int64), t


def ll_nwidth_nodes(counts, log_nw_grid, table, log_nw_k, lapse):
    nz = np.argwhere(counts > 0)
    if nz.size == 0:
        return np.zeros(np.shape(log_nw_k)[0], dtype=np.float64)
    ns, rs = nz[:, 0], nz[:, 1]
    cts = counts[ns, rs]
    Tg = table[:, ns, rs]
    idx, t = _lin_weights(log_nw_grid, log_nw_k)
    lp = (1.0 - t)[:, None] * Tg[idx] + t[:, None] * Tg[idx + 1]
    p = (1.0 - lapse) * np.exp(lp) + lapse / R_MAX
    return (cts[None, :] * np.log(np.clip(p, 1e-300, None))).sum(axis=1)


def ll_weber_nodes(counts, log_w_grid, gammas, table, log_w_k, gamma, lapse):
    nz = np.argwhere(counts > 0)
    if nz.size == 0:
        return np.zeros(np.shape(log_w_k)[0], dtype=np.float64)
    ns, rs = nz[:, 0], nz[:, 1]
    cts = counts[ns, rs]
    Tg = table[:, :, ns, rs]
    iw, tw = _lin_weights(log_w_grid, log_w_k)
    ig, tg = _lin_weights(gammas, np.array([gamma]))
    j, b = int(ig[0]), float(tg[0])
    T0 = (1.0 - b) * Tg[:, j, :] + b * Tg[:, j + 1, :]
    lp = (1.0 - tw)[:, None] * T0[iw] + tw[:, None] * T0[iw + 1]
    p = (1.0 - lapse) * np.exp(lp) + lapse / R_MAX
    return (cts[None, :] * np.log(np.clip(p, 1e-300, None))).sum(axis=1)


def gh_nodes_1d(n):
    t, g = np.polynomial.hermite.hermgauss(n)
    return t * np.sqrt(2.0), g / np.sqrt(np.pi)


def gh_nodes_2d(n):
    t, g = np.polynomial.hermite.hermgauss(n)
    w = g / np.sqrt(np.pi)
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    w1, w2 = np.meshgrid(w, w, indexing="ij")
    nodes = np.stack([t1.ravel() * np.sqrt(2.0), t2.ravel() * np.sqrt(2.0)], axis=1)
    return nodes, (w1 * w2).ravel()


def prepare_participants(tr, n_lo, n_hi):
    parts = []
    for uid, g in tr.groupby("uniqueid"):
        cn = np.zeros((2, 15, R_MAX), dtype=np.float64)
        cw = np.zeros((2, 15, R_MAX), dtype=np.float64)
        for row in g.itertuples(index=False):
            di = DUR_INDEX[int(row.exp_dur)]
            n, r = int(row.n_shown), int(row.n_guess)
            if n <= n_lo:
                cn[di, n - 1, r - 1] += 1.0
            elif n >= n_hi:
                cw[di, n - 1, r - 1] += 1.0
        if cn.sum() > 0 and cw.sum() > 0:
            parts.append({
                "uniqueid": uid, "cn": cn, "cw": cw,
                "n_small": int(cn.sum()), "n_large": int(cw.sum()),
            })
    return parts


def chol_factors(tau_n, tau_w, rho):
    return tau_n, rho * tau_w, np.sqrt(max(tau_w * tau_w * (1.0 - rho * rho), 1e-12))


# ── univariate hierarchical fits (1-D GH) ─────────────────────────────────


def fit_univariate_nwidth(parts, nw_tables, lapse, nodes1, w1):
    log_w1 = np.log(np.clip(w1, 1e-300, None))

    def nll(x):
        a320, a640, log_tau = x
        tau = float(np.exp(log_tau))
        u = tau * nodes1
        total = 0.0
        for p in parts:
            ll = np.zeros_like(nodes1)
            for di, d in enumerate(DURS):
                if p["cn"][di].sum() == 0:
                    continue
                g, tab = nw_tables[d]
                alpha = a320 if di == 0 else a640
                ll += ll_nwidth_nodes(p["cn"][di], g, tab, alpha + u, lapse[d])
            total += float(logsumexp(ll + log_w1))
        return -total

    x0 = np.array([np.log(8.5), np.log(9.5), np.log(0.3)])
    bounds = [(np.log(3), np.log(20)), (np.log(3), np.log(20)), (np.log(0.05), np.log(1.0))]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40, "ftol": 1e-8})
    return {
        "alpha_n": np.array([res.x[0], res.x[1]]),
        "tau_n": float(np.exp(res.x[2])),
        "nll": float(res.fun),
    }


def fit_univariate_weber(parts, weber_pack, lapse, gamma, nodes1, w1):
    log_w1 = np.log(np.clip(w1, 1e-300, None))
    log_w_grid, gammas, table = weber_pack

    def nll(x):
        a320, a640, log_tau = x
        tau = float(np.exp(log_tau))
        u = tau * nodes1
        total = 0.0
        for p in parts:
            ll = np.zeros_like(nodes1)
            for di, d in enumerate(DURS):
                if p["cw"][di].sum() == 0:
                    continue
                alpha = a320 if di == 0 else a640
                ll += ll_weber_nodes(
                    p["cw"][di], log_w_grid, gammas, table,
                    alpha + u, gamma[d], lapse[d],
                )
            total += float(logsumexp(ll + log_w1))
        return -total

    x0 = np.array([np.log(0.15), np.log(0.12), np.log(0.35)])
    bounds = [(np.log(0.04), np.log(0.55)), (np.log(0.04), np.log(0.55)),
              (np.log(0.05), np.log(1.0))]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40, "ftol": 1e-8})
    return {
        "alpha_w": np.array([res.x[0], res.x[1]]),
        "tau_w": float(np.exp(res.x[2])),
        "nll": float(res.fun),
    }


# ── bivariate ─────────────────────────────────────────────────────────────


def participant_loglik(part, hyp, nodes, log_wts, nw_tables, weber_pack,
                       return_posterior=False):
    L00, L10, L11 = chol_factors(hyp["tau_n"], hyp["tau_w"], hyp["rho"])
    log_w_grid, gammas, w_table = weber_pack
    un = L00 * nodes[:, 0]
    uw = L10 * nodes[:, 0] + L11 * nodes[:, 1]
    ll_k = np.zeros(nodes.shape[0], dtype=np.float64)
    for di, d in enumerate(DURS):
        if part["cn"][di].sum() > 0:
            g, tab = nw_tables[d]
            ll_k += ll_nwidth_nodes(
                part["cn"][di], g, tab, hyp["alpha_n"][di] + un, hyp["lapse"][di],
            )
        if part["cw"][di].sum() > 0:
            ll_k += ll_weber_nodes(
                part["cw"][di], log_w_grid, gammas, w_table,
                hyp["alpha_w"][di] + uw, hyp["gamma"][di], hyp["lapse"][di],
            )
    log_terms = ll_k + log_wts
    mll = float(logsumexp(log_terms))
    if not return_posterior:
        return mll
    wpost = np.exp(log_terms - mll)
    return mll, float(np.sum(wpost * un)), float(np.sum(wpost * uw))


def marg_ll(parts, hyp, nodes, log_wts, nw_tables, weber_pack):
    return sum(
        participant_loglik(p, hyp, nodes, log_wts, nw_tables, weber_pack)
        for p in parts
    )


def fit_cutoff(parts, nw_tables, weber_pack, lapse, gamma, label):
    print(f"\n=== Fitting cutoff {label}: N={len(parts)} ===", flush=True)
    t0 = time.time()
    nodes1, w1 = gh_nodes_1d(N_GH)
    nodes2, w2 = gh_nodes_2d(N_GH)
    log_wts = np.log(np.clip(w2, 1e-300, None))

    print("  univariate n_width...", flush=True)
    uni_n = fit_univariate_nwidth(parts, nw_tables, lapse, nodes1, w1)
    print(f"    α_n={np.exp(uni_n['alpha_n'])}  τ_n={uni_n['tau_n']:.3f}", flush=True)
    print("  univariate weber...", flush=True)
    uni_w = fit_univariate_weber(parts, weber_pack, lapse, gamma, nodes1, w1)
    print(f"    α_w={np.exp(uni_w['alpha_w'])}  τ_w={uni_w['tau_w']:.3f}", flush=True)

    base = {
        "alpha_n": uni_n["alpha_n"],
        "alpha_w": uni_w["alpha_w"],
        "tau_n": uni_n["tau_n"],
        "tau_w": uni_w["tau_w"],
        "lapse": np.array([lapse[320], lapse[640]], float),
        "gamma": np.array([gamma[320], gamma[640]], float),
    }

    def nll_rho(atanh_rho):
        hyp = {**base, "rho": float(np.tanh(atanh_rho))}
        return -marg_ll(parts, hyp, nodes2, log_wts, nw_tables, weber_pack)

    print("  searching ρ...", flush=True)
    res_rho = minimize_scalar(nll_rho, bounds=(-2.0, 2.0), method="bounded",
                              options={"xatol": 1e-3})
    rho_hat = float(np.tanh(res_rho.x))

    # short joint polish (derivative-free)
    def nll_joint(x):
        hyp = {
            "alpha_n": x[0:2],
            "alpha_w": x[2:4],
            "tau_n": float(np.exp(x[4])),
            "tau_w": float(np.exp(x[5])),
            "rho": float(np.tanh(x[6])),
            "lapse": base["lapse"],
            "gamma": base["gamma"],
        }
        if abs(hyp["rho"]) > 0.97:
            return 1e12
        return -marg_ll(parts, hyp, nodes2, log_wts, nw_tables, weber_pack)

    x0 = np.concatenate([
        uni_n["alpha_n"], uni_w["alpha_w"],
        [np.log(uni_n["tau_n"]), np.log(uni_w["tau_w"]), res_rho.x],
    ])
    print("  joint polish...", flush=True)
    res = minimize(nll_joint, x0, method="Nelder-Mead",
                   options={"maxiter": 120, "xatol": 1e-3, "fatol": 0.05})
    x = res.x
    hyp = {
        "alpha_n": x[0:2], "alpha_w": x[2:4],
        "tau_n": float(np.exp(x[4])), "tau_w": float(np.exp(x[5])),
        "rho": float(np.tanh(x[6])),
        "lapse": base["lapse"], "gamma": base["gamma"],
    }
    mll = -float(res.fun)
    hyp0 = {**hyp, "rho": 0.0}
    mll0 = marg_ll(parts, hyp0, nodes2, log_wts, nw_tables, weber_pack)
    lr = 2.0 * (mll - mll0)
    p_two = float(chi2.sf(max(lr, 0.0), 1))
    p_neg = p_two / 2.0 if hyp["rho"] < 0 else 1.0 - p_two / 2.0

    rows = []
    for p in parts:
        pll, eu_n, eu_w = participant_loglik(
            p, hyp, nodes2, log_wts, nw_tables, weber_pack, return_posterior=True,
        )
        nw = np.exp(hyp["alpha_n"] + eu_n)
        ww = np.exp(hyp["alpha_w"] + eu_w)
        rows.append({
            "cutoff": label, "uniqueid": p["uniqueid"],
            "n_small": p["n_small"], "n_large": p["n_large"],
            "u_n": eu_n, "u_w": eu_w,
            "n_width_320": nw[0], "n_width_640": nw[1],
            "n_width_mean": float(nw.mean()),
            "weber_320": ww[0], "weber_640": ww[1],
            "weber_mean": float(ww.mean()),
            "loglik": pll,
        })
    part_df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"  ρ̂={hyp['rho']:+.3f}  LR={lr:.1f}  p_neg={p_neg:.2e}  ({elapsed:.0f}s)",
          flush=True)

    summary = {
        "cutoff": label,
        "n_lo": int(label.split("/")[0]),
        "n_hi": int(label.split("/")[1]),
        "n_participants": len(parts),
        "n_trials_small": int(sum(p["n_small"] for p in parts)),
        "n_trials_large": int(sum(p["n_large"] for p in parts)),
        "rho": hyp["rho"],
        "tau_n": hyp["tau_n"],
        "tau_w": hyp["tau_w"],
        "alpha_n_320": float(hyp["alpha_n"][0]),
        "alpha_n_640": float(hyp["alpha_n"][1]),
        "alpha_w_320": float(hyp["alpha_w"][0]),
        "alpha_w_640": float(hyp["alpha_w"][1]),
        "gamma_320": gamma[320],
        "gamma_640": gamma[640],
        "lapse_320": lapse[320],
        "lapse_640": lapse[640],
        "n_width_320": float(np.exp(hyp["alpha_n"][0])),
        "n_width_640": float(np.exp(hyp["alpha_n"][1])),
        "weber_320": float(np.exp(hyp["alpha_w"][0])),
        "weber_640": float(np.exp(hyp["alpha_w"][1])),
        "marg_loglik": mll,
        "marg_loglik_rho0": mll0,
        "lr_rho": lr,
        "p_rho_neg": p_neg,
        "elapsed_s": elapsed,
        "eb_spearman": float(part_df[["u_n", "u_w"]].corr(method="spearman").iloc[0, 1]),
        "eb_pearson": float(part_df[["u_n", "u_w"]].corr(method="pearson").iloc[0, 1]),
    }
    return summary, part_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = load_long_trials()
    audit = audit_identity(tr)
    print(audit, flush=True)

    gp = group_winner_params()
    lapse = {d: gp["sens"][d]["lapse"] for d in DURS}
    print(f"Group winner p={gp['pow']:.4f}; lapse fixed at joint values", flush=True)

    quad = T.make_quadrature(QUAD_NODES, kind="grid")
    t0 = time.time()
    nw_tables, weber_pack = build_or_load_tables(gp["pow"], gp["sens"], quad)
    print(f"Tables ready in {time.time() - t0:.0f}s", flush=True)

    summaries, parts_all = [], []
    for n_lo, n_hi in CUTOFFS:
        label = f"{n_lo}/{n_hi}"
        print(f"\nPooled Weber gain γ (n≥{n_hi}):", flush=True)
        g_cut = fit_pooled_gamma(tr, n_hi, gp["sens"], weber_pack)
        parts = prepare_participants(tr, n_lo, n_hi)
        summary, part_df = fit_cutoff(
            parts, nw_tables, weber_pack, lapse, g_cut, label,
        )
        summaries.append(summary)
        parts_all.append(part_df)

    summary_df = pd.DataFrame(summaries)
    part_df = pd.concat(parts_all, ignore_index=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    part_df.to_csv(OUT_PART, index=False)

    rhos = summary_df["rho"].to_numpy()
    ps = summary_df["p_rho_neg"].to_numpy()
    robust = bool(np.all(rhos < -0.15) and np.all(ps < 0.05))

    audit += "\nHierarchical fit summary\n------------------------\n"
    audit += summary_df.to_string(index=False) + "\n\n"
    audit += (
        "Fixed: width exponent p, sensory (s0, λ0), duration lapses from the "
        "group joint winner; Weber gain γ_d from pooled large-n MLE. "
        "Free: duration FE on log n_width and log w, RE SDs, correlation ρ. "
        "Staged fit: univariate hierarchical MLEs → 1-D ρ search → Nelder–Mead polish.\n"
    )
    audit += f"Robust negative association across cutoffs: {robust}\n"
    OUT_AUDIT.write_text(audit)
    print(audit)
    print(f"Saved {OUT_SUMMARY}\nSaved {OUT_PART}\nSaved {OUT_AUDIT}")
    print(f"ROBUST={robust}")


if __name__ == "__main__":
    main()
