"""Stage 9: distinctive history-dependent predictions of the fitted driven
model, presented as falsifiable experiments (NOT as fit evidence).

    .venv/bin/python driven_attractor/predictions.py --fit output/fits/M2_tether_g0.json

Simulated protocols (same landscape and parameters throughout):
  P1  n = 8 at 320 ms: items regularly spaced vs clustered early vs
      clustered late (same K = n delivered in every case).
  P2  premature termination: n = 10, exposure cut at 80/160/320/640 ms.
  P3  equal accumulated count, different history: condition on K(d) = k in
      the fitted stochastic process; split paths by early-vs-late mean
      arrival time; compare response distribution and settle time.
  P4  online (M2) vs batch (M1/M4: same K(d) delivered at offset).
Outputs a small CSV + figure per protocol in driven_attractor/output/ and
figures/.
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driven_attractor.driven import model as M  # noqa: E402

OUT = ROOT / "driven_attractor" / "output"
FIG = ROOT / "driven_attractor" / "figures"


def run_protocol(theta: dict, arrivals: np.ndarray, kind: str,
                 m_paths: int) -> dict:
    sig = M.sigma_wells(theta["sigma_c"], theta["p"])
    off = M.integrate_exposure(kind, theta["rho"], theta.get("gamma", 0.0),
                               theta["tau_x"], theta.get("beta", 1.0), sig,
                               m_paths, dt=0.0025, arrivals=arrivals)
    table = M.build_relax_table(sig)
    res = {}
    for i, d in enumerate(M.DURATIONS):
        x_off = off.x[i].reshape(-1)
        x_dec, t_dec = table.decide(x_off, theta["tau_x"], sig)
        probs = M.readout_probs(x_dec, sig)
        res[d] = {"x_off": x_off, "x_dec": x_dec, "t_dec": t_dec,
                  "p": probs.reshape(off.x.shape[1], -1, probs.shape[-1]).mean(axis=1),
                  "K": off.K[i]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--m-paths", type=int, default=2048)
    args = ap.parse_args()
    fp = Path(args.fit)
    with open(fp if fp.is_absolute() else ROOT / "driven_attractor" / fp) as f:
        fit = json.load(f)
    theta = fit["theta"]
    kind = {"M2_tether": "tether", "M2_pulse": "pulse"}.get(fit["model"], "tether")
    tag = fp.stem
    mp = args.m_paths
    rows = []

    # ── P1: regular vs early-clustered vs late-clustered (n=8, use 320 ms) ─
    n, d = 8, 0.32
    protos = {
        "regular": np.linspace(d / (n + 1), d * n / (n + 1), n),
        "early": np.linspace(0.01, 0.2 * d, n),
        "late": np.linspace(0.8 * d, d - 0.005, n),
    }
    support = np.arange(1, M.WELLS.size + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, times in protos.items():
        arr = np.full((1, mp, M.N_MAX), np.inf)
        arr[0, :, :n] = times
        res = run_protocol(theta, arr, kind, mp)[320]
        mean_r = float((res["p"][0] * support).sum())
        rows.append({"protocol": "P1", "condition": name, "n": n,
                     "duration_ms": 320, "mean_response": mean_r,
                     "p_exact": float(res["p"][0, n - 1]),
                     "mean_t_relax": float(res["t_dec"].mean()),
                     "mean_x_offset": float(res["x_off"].mean())})
        axes[0].plot(support[:14], res["p"][0, :14], label=name)
        axes[1].bar(name, res["t_dec"].mean())
        axes[2].bar(name, res["x_off"].mean())
    axes[0].set_title(f"p(r) n={n} @320 ms"); axes[0].legend(fontsize=8)
    axes[1].set_title("mean post-offset settle (s)")
    axes[2].set_title("mean x at offset")
    fig.tight_layout(); fig.savefig(FIG / f"{tag}_P1_timing.png", dpi=140)
    plt.close(fig)

    # ── P2: premature termination, n = 10, natural process ────────────────
    E = M.exp_draws(mp)
    arr10 = M.arrival_times(10, theta["rho"], theta.get("gamma", 0.0), E)[None]
    res = run_protocol(theta, arr10, kind, mp)
    for d_ms in M.DURATIONS:
        r = res[d_ms]
        rows.append({"protocol": "P2", "condition": f"cut@{d_ms}", "n": 10,
                     "duration_ms": d_ms,
                     "mean_response": float((r["p"][0] * support).sum()),
                     "p_exact": float(r["p"][0, 9]),
                     "mean_t_relax": float(r["t_dec"].mean()),
                     "mean_x_offset": float(r["x_off"].mean())})

    # ── P3: equal K(d), different temporal history (natural process) ──────
    d_idx, d_ms, n = 1, 80, 10
    r = res[d_ms]
    K = r["K"][0]
    finite = np.where(np.isfinite(arr10[0]), arr10[0], np.nan)
    for k in (3, 4, 5):
        selk = K == k
        if selk.sum() < 100:
            continue
        mean_arr = np.nanmean(np.where(np.arange(M.N_MAX) < k, finite, np.nan),
                              axis=1)
        med = np.nanmedian(mean_arr[selk])
        early = selk & (mean_arr <= med)
        late = selk & (mean_arr > med)
        sig = M.sigma_wells(theta["sigma_c"], theta["p"])
        for lab, sel in (("early", early), ("late", late)):
            probs = M.readout_probs(r["x_dec"][sel], sig).mean(axis=0)
            rows.append({"protocol": "P3", "condition": f"K={k}:{lab}",
                         "n": n, "duration_ms": d_ms,
                         "mean_response": float((probs * support).sum()),
                         "p_exact": float(probs[n - 1]),
                         "mean_t_relax": float(r["t_dec"][sel].mean()),
                         "mean_x_offset": float(r["x_off"][sel].mean())})

    # ── P4: online vs batch delivery of the same K(d) ──────────────────────
    sig = M.sigma_wells(theta["sigma_c"], theta["p"])
    table = M.build_relax_table(sig)
    for d_i, d_ms in enumerate(M.DURATIONS):
        r = res[d_ms]
        K = r["K"][0]
        x_dec_b, t_dec_b = table.decide(K.astype(float), theta["tau_x"], sig)
        pb = M.readout_probs(x_dec_b, sig)
        pb[K == 0] = 0
        po = M.readout_probs(r["x_dec"], sig)
        po[K == 0] = 0
        rows.append({"protocol": "P4", "condition": "online", "n": 10,
                     "duration_ms": d_ms,
                     "mean_response": float((po.mean(axis=0) * support).sum()),
                     "p_exact": float(po.mean(axis=0)[9]),
                     "mean_t_relax": float(r["t_dec"][K > 0].mean())})
        rows.append({"protocol": "P4", "condition": "batch", "n": 10,
                     "duration_ms": d_ms,
                     "mean_response": float((pb.mean(axis=0) * support).sum()),
                     "p_exact": float(pb.mean(axis=0)[9]),
                     "mean_t_relax": float(t_dec_b[K > 0].mean())})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"predictions_{tag}.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
