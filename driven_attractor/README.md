# driven_attractor — can a continuously driven attractor beat the current model?

Isolated analysis directory (nothing outside `driven_attractor/` is modified; the registered analysis in `scripts/` and `output/` is only read). Companion documents: `notes/stage0_audit.md` (audit of the current model + data conventions), `notes/technical_note.md` (equations, numerics, results), `notes/decision_memo.md` (the recommendation).

## Reproduce everything

All commands run from the repository root with the project venv. Seeds are fixed in code (Sobol CRN seed 20260710, fit seed 730, CV folds seed 20260711, recovery seed 42); every reported likelihood is deterministic given the seed.

```bash
# Stage 0 — reproduce the current headline results (exact match required)
.venv/bin/python driven_attractor/repro_stage0.py --refit

# numerical validation gate (evidence process, integrator, QMC error)
.venv/bin/python driven_attractor/validate_numerics.py

# exploratory scan (coarse; informs bounds only)
.venv/bin/python driven_attractor/scan_explore.py

# full-data fits of the driven family (M1/M2/M3 x serial/parallel)
bash driven_attractor/run_all_fits.sh

# comparison table (M0 references + all fits)
.venv/bin/python driven_attractor/collect_results.py

# participant-blocked 5-fold cross-validation (M0 + driven family)
.venv/bin/python driven_attractor/run_cv.py

# diagnostics / RT channel / recovery / distinctive predictions for a fit
.venv/bin/python driven_attractor/diagnostics.py   --fit output/fits/<name>.json
.venv/bin/python driven_attractor/fit_rt_driven.py --fit output/fits/<name>.json
.venv/bin/python driven_attractor/recover.py       --fit output/fits/<name>.json
.venv/bin/python driven_attractor/predictions.py   --fit output/fits/<name>.json
```

## Layout

- `driven/model.py` — landscape (identical to registered winner), evidence processes, guarded exponential-Euler integrator, relaxation table, QMC likelihood with profiled lapses
- `driven/fitting.py` — parameter transforms, DE + Nelder-Mead fitting, final high-accuracy scoring
- `driven/cv.py`, `run_cv.py` — participant-blocked CV for M0 and driven models on identical folds
- `driven/rt.py`, `fit_rt_driven.py` — RT channel (settling vs information-gain stopping; physical-seconds link)
- `output/` — CSVs and fit JSONs; `figures/` — diagnostic figures
