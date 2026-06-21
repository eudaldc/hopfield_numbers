"""Consolidate the trial-level model comparison around the winning model.

Builds, from the cached per-model fits (no re-optimisation), a single master
ladder (response + joint) and the headline figures featuring the winner
(super-linear-width Gaussian attractor, free beta, + dynamics RT):

  outputs/trial_level_model_comparison/
    consolidated_summary.csv        -- response & joint AIC ladder, all models
    consolidated_response_aic.png   -- response ΔAIC bar chart
    consolidated_heatmaps.png       -- data vs winner vs Hopfield-Poisson
    consolidated_rt_metrics.png     -- winner RT AIC by metric (steps vs entropy)

    .venv/bin/python -m scripts.trial_level_consolidate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

D = T.OUT_DIR
LABELS = {
    "gaussian_freebeta_attractor": "Super-linear Gaussian attractor (free β) ★",
    "gaussian_pow_attractor": "Power-law attractor (n^p)",
    "gaussian_benford_free_beta": "Super-linear Gaussian readout (free β, static)",
    "gaussian_pow_readout": "Power-law readout (static)",
    "poisson_pow_attractor": "Poisson free-variance + clean-up",
    "poisson_pow_readout": "Poisson free-variance (static)",
    "weber_ans": "Weber/ANS",
    "hopfield_poisson": "Hopfield-Poisson (√n, published)",
    "no_cleanup_poisson": "Poisson √n (no clean-up)",
    "gaussian_benford_attractor": "Gaussian-Benford √n (attractor)",
    "gaussian_benford_no_cleanup": "Gaussian-Benford √n (no clean-up)",
    "cp_log_prior_proxy": "C&P log-prior proxy",
}


def _sum_aic(path: Path, col="AIC") -> pd.Series:
    df = pd.read_csv(path)
    return df.groupby("model_key")[col].sum()


def build_master() -> pd.DataFrame:
    resp = {}
    # 6 originals
    for k, v in _sum_aic(D / "per_duration_response_likelihood.csv").items():
        resp[k] = v
    # follow-up static free-width
    for k, v in _sum_aic(D / "well_width_followup.csv").items():
        resp[k] = v
    # free-width attractors (incl winner)
    for k, v in _sum_aic(D / "freewidth_attractor.csv").items():
        resp[k] = v
    # poisson free-variance
    for k, v in _sum_aic(D / "poisson_freevar.csv").items():
        resp[k] = v
    out = pd.DataFrame({"model_key": list(resp), "resp_AIC": list(resp.values())})
    out["label"] = out["model_key"].map(lambda k: LABELS.get(k, k))
    out = out.sort_values("resp_AIC").reset_index(drop=True)
    out["resp_dAIC"] = out["resp_AIC"] - out["resp_AIC"].min()
    return out


def plot_response_ladder(master: pd.DataFrame) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    m = master.sort_values("resp_dAIC", ascending=False)
    colors = ["#c0392b" if "★" in lab else ("#2c7fb8" if "√n" in lab or "Weber" in lab
              else "#7fbf7b") for lab in m["label"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(m["label"], m["resp_dAIC"], color=colors)
    ax.set_xlabel("ΔAIC vs best (trial-level response likelihood)")
    ax.set_title("Trial-level response model comparison", fontweight="bold")
    for y, v in enumerate(m["resp_dAIC"]):
        ax.text(v + 40, y, f"{v:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "consolidated_response_aic.png", dpi=200)
    plt.close(fig)


def plot_heatmaps() -> None:
    quad = T.make_quadrature(801, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)
    wt = pd.read_csv(D / "winner_theta.csv").set_index("duration_ms")
    resp = pd.read_csv(D / "per_duration_response_likelihood.csv")
    win = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
    hop = T.MODEL_BY_KEY["hopfield_poisson"]
    resp_vals = list(range(1, T.N_SUPPORT_MAX + 1))
    durs = list(T.DURATIONS)
    fig, axes = plt.subplots(len(durs), 3, figsize=(10.5, 2.7 * len(durs)), squeeze=False)
    for ri, d in enumerate(durs):
        data = data_by_dur[d]
        obs = data.counts / data.counts.sum(axis=1, keepdims=True)
        wtheta = {c: float(wt.loc[d, c]) for c in ("sigma_c", "s0", "lambda0", "beta")}
        pf_w = T.apply_lapse(T.compute_grid(win, wtheta, quad)["p_resp"],
                             float(wt.loc[d, "lapse"]), r_max)
        hr = resp[(resp.model_key == "hopfield_poisson") & (resp.duration_ms == d)].iloc[0]
        htheta = {c: float(hr[f"theta_{c}"]) for c in ("sigma_c", "s0", "lambda0")}
        pf_h = T.apply_lapse(T.compute_grid(hop, htheta, quad)["p_resp"],
                             float(hr["lapse"]), r_max)
        vmax = float(obs[:, :T.N_SUPPORT_MAX].max())
        panels = [("Data", obs[:, :T.N_SUPPORT_MAX]),
                  ("Winner (super-linear)", pf_w[:, :T.N_SUPPORT_MAX]),
                  ("Hopfield-Poisson (√n)", pf_h[:, :T.N_SUPPORT_MAX])]
        for ci, (title, mat) in enumerate(panels):
            ax = axes[ri][ci]
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma",
                           extent=[0.5, T.N_SUPPORT_MAX + 0.5, 0.5, T.N_TRUE_MAX + 0.5],
                           vmin=0, vmax=vmax)
            if ci == 0:
                ax.set_ylabel(f"{d} ms\nn shown")
            if ri == 0:
                ax.set_title(title)
            if ri == len(durs) - 1:
                ax.set_xlabel("response r")
    fig.suptitle("Observed vs predicted response distributions", y=1.005)
    fig.colorbar(im, ax=axes, shrink=0.6, label="P(r | n)")
    fig.savefig(D / "consolidated_heatmaps.png", dpi=170)
    plt.close(fig)


def plot_rt_metrics() -> None:
    rt = pd.read_csv(D / "rt_exploration.csv")
    sub = rt[rt.landscape == "gaussian_freebeta_attractor"].groupby("metric")["AIC"].sum()
    order = ["entropy", "steps", "flowtime", "hybrid", "flow_entropy"]
    ent = sub["entropy"]
    d = np.array([sub[m] - ent for m in order])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#888" if x >= 0 else "#c0392b" for x in d]
    ax.bar(order, d, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ΔAIC vs entropy (RT-only)")
    ax.set_title("Winner RT channel: do clean-up dynamics beat entropy?",
                 fontweight="bold")
    for i, x in enumerate(d):
        ax.text(i, x + (8 if x >= 0 else -8), f"{x:+.0f}", ha="center",
                va="bottom" if x >= 0 else "top", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(D / "consolidated_rt_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    master = build_master()
    # joint ladder
    joint = {}
    for k, v in _sum_aic(D / "per_duration_joint_likelihood.csv").items():
        joint[k] = v
    wj = pd.read_csv(D / "winner_joint.csv")
    for metric, g in wj.groupby("metric"):
        joint[f"winner_{metric}"] = g["AIC"].sum()
    jdf = pd.DataFrame({"model_key": list(joint), "joint_AIC": list(joint.values())})
    jdf = jdf.sort_values("joint_AIC")
    jdf["joint_dAIC"] = jdf["joint_AIC"] - jdf["joint_AIC"].min()

    master.to_csv(D / "consolidated_summary.csv", index=False)
    jdf.to_csv(D / "consolidated_joint_summary.csv", index=False)

    print("=== Response ladder (all models, summed AIC) ===")
    print(master[["label", "resp_AIC", "resp_dAIC"]].round(1).to_string(index=False))
    print("\n=== Joint ladder ===")
    print(jdf.round(1).to_string(index=False))

    plot_response_ladder(master)
    plot_heatmaps()
    plot_rt_metrics()
    print(f"\nSaved consolidated CSVs + figures to {D}")


if __name__ == "__main__":
    main()
