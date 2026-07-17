"""Fair RT comparison for the winner: predicted median RT (entropy vs dynamics)
vs observed, with median-RT-curve R^2 comparable to the published Fig. 2.

The published Fig. 2 fitted the median-RT curve directly (in-sample R^2 0.89-0.98).
The trial-level pipeline fits RT as a *separate predicted channel* on a
response-optimised landscape, so it is a stricter test. This shows the winner's
predicted median RT for the entropy and the dynamics (hybrid steps+entropy) RT
metrics, and reports the per-duration R^2 against the observed median-RT curve.

    .venv/bin/python -m scripts.trial_level_rt_figure
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

WIN = T.MODEL_BY_KEY["gaussian_freebeta_attractor"]
OUT_FIG = T.LEGACY_OUT_DIR / "winner_rt_fit.png"
OUT_CSV = T.LEGACY_OUT_DIR / "winner_rt_fit.csv"
# published Fig. 2 RT R^2 (40/80/160/320/640 ms) for reference
FIG2_RT_R2 = {40: 0.91, 80: 0.98, 160: 0.98, 320: 0.93, 640: 0.89}


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    ss = np.sum((obs - pred) ** 2)
    return 1.0 - ss / (np.sum((obs - obs.mean()) ** 2) + 1e-12)


def main() -> None:
    quad = T.make_quadrature(1601, kind="grid")
    trial = T.load_trials()
    r_max = int(trial["n_guess"].max())
    data_by_dur = T.build_duration_data(trial, r_max)
    wt = pd.read_csv(T.LEGACY_OUT_DIR / "winner_theta.csv").set_index("duration_ms")
    rtx = pd.read_csv(T.LEGACY_OUT_DIR / "rt_exploration.csv")
    rtx = rtx[rtx.landscape == "gaussian_freebeta_attractor"]

    fig, axes = plt.subplots(1, len(T.DURATIONS), figsize=(3.0 * len(T.DURATIONS), 3.4),
                             squeeze=False)
    rows = []
    for ci, d in enumerate(T.DURATIONS):
        theta = {c: float(wt.loc[d, c]) for c in ("sigma_c", "s0", "lambda0", "beta")}
        grid = T.compute_grid(WIN, theta, quad)
        eh, et = grid["EH"], grid["ET"]
        ent = rtx[(rtx.duration_ms == d) & (rtx.metric == "entropy")].iloc[0]
        hyb = rtx[(rtx.duration_ms == d) & (rtx.metric == "hybrid")].iloc[0]
        pred_ent = T.T0 + float(ent["rt_scale"]) * eh
        pred_hyb = T.T0 + float(hyb["alpha"]) * et + float(hyb["beta"]) * eh
        data = data_by_dur[d]
        # The log-normal RT MLE fits the mean of log RT per n, i.e. the geometric
        # mean of RT (= exp(mu)). Compare against that, not the median: log RT is
        # right-skewed, so the median sits ~0.1-0.2 s *below* the fitted quantity
        # and would make the line look spuriously high (see docs / rt_domain).
        obs_geo = np.array([np.exp(np.mean(data.logrt_trial[data.n_trial == n]))
                            for n in T.N_SHOWN])
        obs_med = np.array([np.median(np.exp(data.logrt_trial[data.n_trial == n]))
                            for n in T.N_SHOWN])   # shown faint for reference
        r2_ent, r2_hyb = r2(obs_geo, pred_ent), r2(obs_geo, pred_hyb)
        for i, n in enumerate(T.N_SHOWN):
            rows.append({"duration_ms": d, "n_shown": int(n),
                         "obs_geomean_rt": obs_geo[i], "obs_median_rt": obs_med[i],
                         "pred_entropy": pred_ent[i], "pred_hybrid": pred_hyb[i]})
        ax = axes[0][ci]
        ax.plot(T.N_SHOWN, obs_med, ".", color="0.72", ms=5, label="data (median)")
        ax.plot(T.N_SHOWN, obs_geo, "k.", ms=8, label="data (geom. mean)")
        ax.plot(T.N_SHOWN, pred_ent, color="#2c7fb8", lw=1.6,
                label=f"entropy (R²={r2_ent:.2f})")
        ax.plot(T.N_SHOWN, pred_hyb, color="#c0392b", lw=1.6,
                label=f"dynamics (R²={r2_hyb:.2f})")
        ax.set_title(f"{d} ms   [Fig2 R²={FIG2_RT_R2[d]:.2f}]", fontsize=9)
        ax.set_xlabel("n shown")
        if ci == 0:
            ax.set_ylabel("geometric-mean RT (s)")
        ax.legend(fontsize=6.5, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        print(f"{d}ms: R2 entropy={r2_ent:.3f}  dynamics={r2_hyb:.3f}  "
              f"(Fig2 published RT R2={FIG2_RT_R2[d]:.2f})", flush=True)
    fig.suptitle("Winner predicted RT (geometric mean): entropy vs dynamics "
                 "(steps+entropy) vs data — median shown faint for reference", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=190)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    # pooled R^2 across all (duration, n), against the geometric mean
    print(f"\nPooled R2: entropy={r2(df.obs_geomean_rt.values, df.pred_entropy.values):.3f}"
          f"  dynamics={r2(df.obs_geomean_rt.values, df.pred_hybrid.values):.3f}")
    print(f"Saved {OUT_FIG} and {OUT_CSV}")


if __name__ == "__main__":
    main()
