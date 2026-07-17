"""Cache the trial-level data once, so no analysis script has to re-read the
45 MB per-dot CSV (memory-frugal: chunked read, only the needed columns).

    .venv/bin/python driven_attractor/build_cache.py

Writes driven_attractor/output/trials.npz with one row per trial:
exp_dur, n_shown, n_guess, rt_s, subject index (+ the uniqueid table).
Reproduces scripts.trial_level_model_comparison.load_trials exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "driven_attractor" / "output" / "trials.npz"
CSV = ROOT / "data" / "data_piantadosi.csv"
DURATIONS = (40, 80, 160, 320, 640)

cols = ["uniqueid", "trial_id", "cond", "exp_dur", "n_shown", "n_guess", "rt"]
parts = []
for chunk in pd.read_csv(CSV, usecols=cols, chunksize=500_000):
    chunk = chunk[(chunk["cond"] == "DURATION") & chunk["exp_dur"].isin(DURATIONS)]
    parts.append(chunk.drop(columns=["cond"])
                 .groupby(["uniqueid", "trial_id"], as_index=False)
                 .agg(exp_dur=("exp_dur", "first"), n_shown=("n_shown", "first"),
                      n_guess=("n_guess", "first"), rt=("rt", "first")))
df = pd.concat(parts)
# a trial can straddle a chunk boundary: collapse duplicates once more
df = df.groupby(["uniqueid", "trial_id"], as_index=False).agg(
    exp_dur=("exp_dur", "first"), n_shown=("n_shown", "first"),
    n_guess=("n_guess", "first"), rt=("rt", "first"))
df = df[(df["n_shown"] >= 1) & (df["n_shown"] <= 15) & (df["rt"] > 0)]
uids, subj = np.unique(df["uniqueid"].to_numpy(), return_inverse=True)

np.savez_compressed(
    OUT,
    exp_dur=df["exp_dur"].to_numpy(np.int32),
    n_shown=df["n_shown"].to_numpy(np.int32),
    n_guess=df["n_guess"].to_numpy(np.int32),
    rt_s=(df["rt"].to_numpy(float) / 1000.0),
    subject=subj.astype(np.int32),
    uniqueid=uids,
)
print(f"cached {len(df)} trials, {uids.size} participants -> {OUT}")
print(df.groupby("exp_dur").size().to_dict())
