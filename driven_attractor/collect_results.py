"""Assemble the M0-M4 model-comparison table from fits + registered outputs.

    .venv/bin/python driven_attractor/collect_results.py
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
FITS = OUT / "fits"
TL = ROOT / "output" / "trial_level_model_comparison"

LABELS = {
    "M1_offset": "M1/M4 accumulate-at-offset",
    "M2_tether": "M2 driven attractor (soft tether)",
    "M2_pulse": "M2 driven attractor (pulses)",
    "M3_static": "M3 driven static (no attractor)",
}


def main() -> None:
    rows = []
    # M0 references (reproduced in repro_stage0)
    ladder = pd.read_csv(TL / "shared_landscape_ladder.csv").set_index("model_key")
    n_tot = 52995
    m0s = ladder.loc["gaussian_pow_attractor"]
    rows.append({"model": "M0 current, per-duration", "process": "-",
                 "k": 25, "logL": float(m0s.free_logL),
                 "AIC": float(m0s.free_AIC), "BIC": float(m0s.free_BIC)})
    rows.append({"model": "M0 current, shared landscape", "process": "-",
                 "k": 17, "logL": float(m0s.shared_logL),
                 "AIC": float(m0s.shared_AIC), "BIC": float(m0s.shared_BIC)})
    st = ladder.loc["gaussian_pow_readout"]
    rows.append({"model": "M0 static twin, shared landscape", "process": "-",
                 "k": 17, "logL": float(st.shared_logL),
                 "AIC": float(st.shared_AIC), "BIC": float(st.shared_BIC)})
    wb = ladder.loc["weber_ans"]
    rows.append({"model": "Weber/ANS (per-duration)", "process": "-",
                 "k": 20, "logL": float(wb.shared_logL),
                 "AIC": float(wb.shared_AIC), "BIC": float(wb.shared_BIC)})

    for f in sorted(FITS.glob("*.json")):
        if "smoke" in f.name:
            continue
        with open(f) as fh:
            fit = json.load(fh)
        proc = {0.0: "serial", 1.0: "parallel"}.get(fit["gamma"], f"gamma={fit['gamma']}")
        if fit.get("free_gamma"):
            proc = f"gamma free ({fit['theta'].get('gamma', np.nan):.2f})"
        extras = "+delta" if fit.get("free_delta") else ""
        rows.append({"model": LABELS.get(fit["model"], fit["model"]) + extras,
                     "process": proc, "k": fit["k"], "logL": fit["logL"],
                     "AIC": fit["AIC"], "BIC": fit["BIC"],
                     "file": f.name,
                     "theta": json.dumps({k: round(v, 4)
                                          for k, v in fit["theta"].items()})})
    df = pd.DataFrame(rows)
    df["dAIC"] = df["AIC"] - df["AIC"].min()
    df["dBIC"] = df["BIC"] - df["BIC"].min()
    df["mean_log_score"] = df["logL"] / n_tot
    df = df.sort_values("AIC").reset_index(drop=True)
    df.to_csv(OUT / "model_comparison_m0_m4.csv", index=False)
    show = df[["model", "process", "k", "logL", "AIC", "dAIC", "dBIC",
               "mean_log_score"]]
    print(show.round({"logL": 1, "AIC": 1, "dAIC": 1, "dBIC": 1,
                      "mean_log_score": 4}).to_string(index=False))


if __name__ == "__main__":
    main()
