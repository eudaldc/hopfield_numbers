"""Posterior-predictive and mechanistic diagnostics for a fitted driven model
(Stage 8), plus the M0 reference overlays.

    .venv/bin/python driven_attractor/diagnostics.py --fit output/fits/M2_tether_g0.json

Produces (in driven_attractor/figures/):
  <tag>_summaries.png       mean guess / P(exact) / CoV per duration: data,
                            driven model, M0 shared-landscape reference
  <tag>_distributions.png   full p(r | n, d) overlays for n = 2,4,6,8,10,13
  <tag>_mechanism.png       K(d) distributions; x(d) vs x_dec (clean-up
                            before/after offset); example trajectories;
                            fitted-landscape fixed points
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import model as M  # noqa: E402
import scripts.trial_level_model_comparison as T  # noqa: E402

FIG = ROOT / "driven_attractor" / "figures"
FIG.mkdir(exist_ok=True)


def m0_shared_tables(r_max: int) -> dict[int, np.ndarray]:
    import pandas as pd
    spec = T.MODEL_BY_KEY["gaussian_pow_attractor"]
    th = pd.read_csv(ROOT / "output/trial_level_model_comparison/shared_landscape_ladder_theta.csv")
    th = th[th.model_key == "gaussian_pow_attractor"]
    quad = T.make_quadrature(1601, kind="grid")
    out = {}
    for _, r in th.iterrows():
        grid = T.compute_grid(spec, {"sigma_c": r.sigma_c, "s0": r.s0,
                                     "lambda0": r.lambda0, "pow": r["pow"]}, quad)
        out[int(r.duration_ms)] = T.apply_lapse(grid["p_resp"], r.lapse, r_max)
    return out


def summarise(pf: np.ndarray) -> dict:
    """Full-support stats plus the 'core' mean (renormalised to r <= 19,
    the repo's convention for the mean-guess curve: lapse-dominated tails
    excluded on both sides)."""
    r = np.arange(1, pf.shape[1] + 1, dtype=float)
    mean = (pf * r).sum(axis=1)
    exact = np.array([pf[i, i] for i in range(M.N_SHOWN.size)])
    var = (pf * r * r).sum(axis=1) - mean ** 2
    core = pf[:, :19] / pf[:, :19].sum(axis=1, keepdims=True)
    core_mean = (core * r[:19]).sum(axis=1)
    cv = (core * r[:19] ** 2).sum(axis=1) - core_mean ** 2
    return {"core mean": core_mean, "exact": exact,
            "core CoV": np.sqrt(np.maximum(cv, 0)) / core_mean,
            "full mean": mean}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--m-paths", type=int, default=8192)
    args = ap.parse_args()
    with open(ROOT / "driven_attractor" / args.fit if not Path(args.fit).is_absolute()
              else args.fit) as f:
        fit = json.load(f)
    model, theta = fit["model"], fit["theta"]
    tag = Path(args.fit).stem

    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data = T.build_duration_data(trial, r_max)
    counts = {d: data[d].counts for d in M.DURATIONS}

    ev = M.evaluate_model(model, theta, m_paths=args.m_paths, dt=0.0025,
                          keep_paths=True)
    pf_driven = {}
    for i, d in enumerate(M.DURATIONS):
        lapse, _ = M.profile_lapse(ev.p_wells[i], ev.p_guess[i], counts[d], r_max)
        pf_driven[d] = M.p_final_full(ev.p_wells[i], ev.p_guess[i], lapse, r_max)
    pf_m0 = m0_shared_tables(r_max)

    # ── figure 1: summary curves ──────────────────────────────────────────
    keys = ("core mean", "exact", "core CoV", "full mean")
    fig, axes = plt.subplots(len(keys), 5, figsize=(18, 12), sharex=True)
    for j, d in enumerate(M.DURATIONS):
        obs = counts[d] / counts[d].sum(axis=1, keepdims=True)
        s_obs, s_drv, s_m0 = summarise(obs), summarise(pf_driven[d]), summarise(pf_m0[d])
        for i, key in enumerate(keys):
            ax = axes[i, j]
            ax.plot(M.N_SHOWN, s_obs[key], "ko", ms=4, label="data")
            ax.plot(M.N_SHOWN, s_drv[key], "-", color="tab:red", label="driven")
            ax.plot(M.N_SHOWN, s_m0[key], "-", color="tab:blue", alpha=0.7,
                    label="M0 shared")
            if key.endswith("mean"):
                if i == 0:
                    ax.set_title(f"{d} ms")
                ax.plot(M.N_SHOWN, M.N_SHOWN, ":", color="grey", lw=0.8)
            if j == 0:
                ax.set_ylabel(key)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{tag}: posterior predictive summaries (full support, lapse included)")
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_summaries.png", dpi=140)
    plt.close(fig)

    # ── figure 2: full response distributions ─────────────────────────────
    ns = (2, 4, 6, 8, 10, 13)
    fig, axes = plt.subplots(len(ns), 5, figsize=(16, 14), sharex=True)
    rr = np.arange(1, 26)
    for j, d in enumerate(M.DURATIONS):
        obs = counts[d] / counts[d].sum(axis=1, keepdims=True)
        for i, n in enumerate(ns):
            ax = axes[i, j]
            ax.bar(rr, obs[n - 1, :25], color="0.7", width=0.9, label="data")
            ax.plot(rr, pf_driven[d][n - 1, :25], "-", color="tab:red", lw=1.4,
                    label="driven")
            ax.plot(rr, pf_m0[d][n - 1, :25], "-", color="tab:blue", lw=1.1,
                    alpha=0.8, label="M0 shared")
            ax.axvline(n, color="k", lw=0.6, ls=":")
            if j == 0:
                ax.set_ylabel(f"n={n}")
            if i == 0:
                ax.set_title(f"{d} ms")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{tag}: response distributions p(r | n, d)")
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_distributions.png", dpi=140)
    plt.close(fig)

    # ── figure 3: mechanism panels ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # K(d) distributions for n=8
    ax = axes[0, 0]
    for j, d in enumerate(M.DURATIONS):
        K = ev.paths["K_d"][j, 7, :]
        ax.hist(K, bins=np.arange(-0.5, 9.5), histtype="step", density=True,
                label=f"{d} ms")
    ax.set_title("K(d) at n = 8")
    ax.legend(fontsize=7)
    # clean-up before vs after offset
    ax = axes[0, 1]
    for j, c in zip((0, 2, 4), ("tab:purple", "tab:green", "tab:orange")):
        x_off = ev.paths["x_off"][j].reshape(-1)[::7]
        x_dec = ev.paths["x_dec"][j].reshape(-1)[::7]
        ax.scatter(x_off, x_dec - x_off, s=2, alpha=0.3, color=c,
                   label=f"{M.DURATIONS[j]} ms")
    ax.set_xlabel("x at offset"); ax.set_ylabel("post-offset movement")
    ax.set_title("clean-up after offset"); ax.legend(fontsize=7)
    # example trajectories at 640 ms for n = 2, 4, 8, 12
    sig = M.sigma_wells(theta["sigma_c"], theta["p"])
    if model in ("M2_tether", "M2_pulse"):
        kind = "tether" if model == "M2_tether" else "pulse"
        E = M.exp_draws(8)
        for ax_i, n in zip((axes[0, 2], axes[1, 0]), (4, 10)):
            arr = M.arrival_times(n, theta["rho"], theta.get("gamma", 0.0), E)[None]
            off = M.integrate_exposure(kind, theta["rho"], theta.get("gamma", 0),
                                       theta.get("tau_x", 0.25),
                                       theta.get("beta", 1.0), sig, 8,
                                       dt=0.0025, arrivals=arr, record=True)
            tgrid = np.arange(off.x_t.shape[0]) * 0.0025
            for pth in range(6):
                ax_i.plot(tgrid, off.x_t[:, pth], lw=0.9)
                a = arr[0, pth][np.isfinite(arr[0, pth])]
                ax_i.plot(a, np.interp(a, tgrid, off.x_t[:, pth]), "k.", ms=4)
            for w in range(1, 16):
                ax_i.axhline(w, color="0.9", lw=0.5, zorder=0)
            ax_i.set_title(f"trajectories n={n} (dots = item registrations)")
            ax_i.set_xlabel("time from onset (s)")
    # landscape fixed points
    ax = axes[1, 1]
    xg = np.linspace(0.2, 20, 3000)
    A, B = M.force_AB(xg, sig)
    Fg = A - B * xg
    ax.plot(xg, Fg, lw=1)
    ax.axhline(0, color="k", lw=0.6)
    sgn = np.sign(Fg)
    zc = np.where(np.diff(sgn) != 0)[0]
    stab = zc[Fg[zc] > 0]
    ax.plot(xg[stab], np.zeros_like(xg[stab]), "g.", ms=6)
    ax.set_title(f"-V'(x); stable fixed points (n={len(stab)})")
    ax.set_xlabel("x")
    # entropy vs x
    ax = axes[1, 2]
    ax.plot(xg, M.entropy_of_readout(xg, sig), lw=1)
    ax.set_title("readout entropy H(x)")
    ax.set_xlabel("x")
    fig.suptitle(f"{tag}: mechanism diagnostics")
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_mechanism.png", dpi=140)
    plt.close(fig)
    print(f"saved 3 figures for {tag} in {FIG}")
    print(f"frac paths capped at T_CAP: {ev.frac_capped:.4f}")


if __name__ == "__main__":
    main()
