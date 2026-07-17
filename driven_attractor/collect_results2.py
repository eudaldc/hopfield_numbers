"""Assemble the round-2 comparison table (evidence families x variants + M0).

    .venv/bin/python driven_attractor/collect_results2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "driven_attractor" / "output"
FITS2 = OUT / "fits2"
FITS1 = OUT / "fits"
TL = ROOT / "output" / "trial_level_model_comparison"
N = 52995

FAM_LABEL = {"ddm": "Continuous accumulation (1/sqrt(t) precision)",
             "kalman": "Sequential Bayesian (Kalman)",
             "pulse": "Imperfect item pulses"}
VAR_LABEL = {"batch_attr": "batch + attractor", "batch_static": "batch, static",
             "online_attr": "online + attractor", "online_static": "online, static"}


def main() -> None:
    rows = []
    ladder = pd.read_csv(TL / "shared_landscape_ladder.csv").set_index("model_key")
    m0 = ladder.loc["gaussian_pow_attractor"]
    rows += [
        {"model": "M0 current, per-duration", "variant": "batch + attractor",
         "k": 25, "logL": float(m0.free_logL), "AIC": float(m0.free_AIC),
         "BIC": float(m0.free_BIC)},
        {"model": "M0 current, shared landscape", "variant": "batch + attractor",
         "k": 17, "logL": float(m0.shared_logL), "AIC": float(m0.shared_AIC),
         "BIC": float(m0.shared_BIC)},
        {"model": "M0 static twin, shared", "variant": "batch, static", "k": 17,
         "logL": float(ladder.loc["gaussian_pow_readout"].shared_logL),
         "AIC": float(ladder.loc["gaussian_pow_readout"].shared_AIC),
         "BIC": float(ladder.loc["gaussian_pow_readout"].shared_BIC)},
    ]
    # round-1 best (capped count K(t))
    best1 = None
    for f in FITS1.glob("*.json"):
        if "smoke" in f.name:
            continue
        d = json.load(open(f))
        if best1 is None or d["logL"] > best1["logL"]:
            best1 = d
    if best1:
        rows.append({"model": "Round 1 best: capped count K(t)",
                     "variant": f"{best1['model']}", "k": best1["k"],
                     "logL": best1["logL"], "AIC": best1["AIC"],
                     "BIC": best1["BIC"]})

    for f in sorted(FITS2.glob("*.json")):
        d = json.load(open(f))
        rows.append({"model": FAM_LABEL[d["family"]],
                     "variant": VAR_LABEL[d["variant"]], "k": d["k"],
                     "logL": d["logL"], "AIC": d["AIC"], "BIC": d["BIC"],
                     "theta": json.dumps({k: round(v, 5)
                                          for k, v in d["theta"].items()})})
    df = pd.DataFrame(rows)
    df["dAIC"] = df["AIC"] - df["AIC"].min()
    df["dBIC"] = df["BIC"] - df["BIC"].min()
    df["mean_log_score"] = df["logL"] / N
    df = df.sort_values("AIC").reset_index(drop=True)
    df.to_csv(OUT / "model_comparison_round2.csv", index=False)
    print(df[["model", "variant", "k", "logL", "AIC", "dAIC", "dBIC"]]
          .round({"logL": 1, "AIC": 1, "dAIC": 1, "dBIC": 1}).to_string(index=False))

    # the three key contrasts, per family
    print("\n=== key contrasts (logL; + favours the first) ===")
    by = {(r["model"], r["variant"]): r["logL"] for _, r in df.iterrows()}
    for fam, lab in FAM_LABEL.items():
        for a, b, name in (("online + attractor", "batch + attractor", "online vs batch (with attractor)"),
                           ("online, static", "batch, static", "online vs batch (static)"),
                           ("online + attractor", "online, static", "attractor vs none (online)"),
                           ("batch + attractor", "batch, static", "attractor vs none (batch)")):
            if (lab, a) in by and (lab, b) in by:
                print(f"  {lab[:28]:<30} {name:<34} {by[(lab, a)] - by[(lab, b)]:+9.1f}")


if __name__ == "__main__":
    main()
