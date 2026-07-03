# A Hopfield attractor account of numerosity perception

Research code for the paper **A Hopfield attractor account of numerosity perception**, fitted to behavioural data.  Noisy scalar evidence is refined by gradient descent on an energy landscape until it settles near a discrete numerosity; well spacing and width implement a “useful discreteness” that weakens for larger numbers.

The codebase fits use `scipy` global optimization on empirical summary statistics (accuracy, mean guess, RT where available), with shared simulation logic in `scripts/fit_helpers.py`.  Configuration for well type, noise, RT mode, and optional ablations lives in `configs/config.yaml`.

## Model sketch

```
y₀  →  cleanup (gradient descent on E(x | y₀))  →  x*  →  posterior p(n | x*)  →  loss vs data
```

Energy (Poisson or Gaussian wells; see `fit_helpers` and `docs/`, and the manuscript). Encoding noise and RT linkage are controlled via YAML and the fit scripts.

<p align="center">
  <img src="figures/conceptual_attractor_wells_poisson.png" alt="Conceptual attractor wells illustration" width="600"/><br>
  <em>Illustration of integer attractor wells along the evidence axis.</em>
</p>

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place inputs under `data/` (e.g. Piantadosi estimation trials, De Carli group summaries, Reynvoet Panamath CSV).  Processed outputs and fit artefacts are written under `output/` by the scripts below.

## Project structure

```
scripts/       Fitting, evaluation, density-transition analysis, figure helpers
configs/       YAML defaults for fits (durations, well type, RT mode, priors, …)
data/          Raw / cleaned CSVs and derived tables
output/        Fit summaries, plots, CSVs from analyses
figures/       Publication figures and generators (e.g. `make_fig*.py`)
manuscript/    LaTeX for the paper and supplementary material
docs/          Model notes and session logs
tests/         pytest suite
```

## Main scripts

| Script | Purpose |
|--------|---------|
| `scripts/fit_piantadosi.py` | Single-condition fit to Piantadosi estimation data (wandb run) |
| `scripts/fit_piantadosi_durations.py` | Sweep all exposure durations (40–640 ms); optional bootstrap CIs |
| `scripts/fit_decarli.py` | Separate fits for DD vs controls (accuracy + RT, no trial-level guesses) |
| `scripts/model_comparison.py` | Summary-curve model comparison across durations (profile-Gaussian AIC/BIC) |
| `scripts/trial_level_model_comparison.py` | Trial-level model comparison: deterministic multinomial response likelihood (grid quadrature) + RT and joint channels; registers the winning super-linear-width Gaussian attractor; `--smoke` / `--replot` |
| `scripts/trial_level_well_width_followup.py` | Frees the Gaussian well-width exponent (β / p): shows the data want super-linear (∼n^1.5–2) widths, not √n |
| `scripts/trial_level_freewidth_attractor.py` | Free-width wells + clean-up (the winning combination) |
| `scripts/trial_level_poisson_freevar.py` | Equal-footing control: Poisson wells with a free variance exponent |
| `scripts/trial_level_rt_exploration.py` | RT channel: convergence-time (steps/flow-time) vs entropy across landscapes |
| `scripts/trial_level_winner_joint.py` | Coupled joint (response + dynamics RT) for the winning model |
| `scripts/trial_level_subitizing_diagnostic.py` | Head-to-head of well forms on the subitizing band (n≤4) |
| `scripts/trial_level_consolidate.py` | Master ladder + headline figures consolidated around the winner |
| `scripts/trial_level_ncrit_rederivation.py` | Re-derives n_crit (well width σ_n=1 ⇔ d′=1) for the winning landscape |
| `scripts/trial_level_regime_change.py` | Isolates where the clean-up / discrete code stops contributing (exact→approximate) |
| `scripts/trial_level_ncap_fit.py` | Small-range (n≤10) refit + n_crit re-derivation: is this a mechanistic small-number model that predicts the ANS transition? |
| `scripts/trial_level_cofit_figure.py` | Winner joint co-fit vs data: mean guess / % exact / geometric-mean RT per duration (`winner_cofit_summaries.png`) |
| `scripts/trial_level_rt_figure.py` | Winner predicted RT (geometric mean) — entropy vs dynamics, R² vs published Fig. 2 (`winner_rt_fit.png`) |
| `scripts/trial_level_rt_domain.py` | RT predicted within the model's domain (n < n_crit), extrapolated above (`rt_domain.png`) |
| `scripts/trial_level_pexact_decomposition.py` | Why P(exact) wiggles: smooth landscape vs discrete cleanup-basin read-out (`pexact_decomposition.png`) |
| `scripts/decarli_gaussian_freebeta.py` | Winner model applied to De Carli (SM5): group accuracy+RT fit + derived n_crit/σ(n) |
| `scripts/decarli_gaussian_freebeta_subjects.py` | Per-subject De Carli n_crit distributions, DD vs control |
| `scripts/decarli_locus.py` / `decarli_locus_clean.py` | Is the DD deficit resolution or sensory noise? (mirror fit + clean global test) |
| `scripts/decarli_dd_summary_figure.py` | Consolidated DD-vs-control 4-panel argument (`decarli_dd_summary.png`) |
| `scripts/density_transition.py` | Post-fit metrics and “transition to density” figure from duration fits |
| `scripts/run_parameter_recovery.py` | Orchestrates recovery / profile / bootstrap identifiability analyses |
| `scripts/plot_parameter_recovery.py` | Plots from recovery outputs |
| `scripts/plot_decarli_param_boxfacets.py` | De Carli parameter panel figure |
| `scripts/plots.py` | Shared plotting utilities where used by other entry points |

Run from the repository root (e.g. `python scripts/fit_piantadosi.py …` or `python -m scripts.density_transition` as documented in each file).

## Configuration

`configs/config.yaml` drives the attractor variant (e.g. Poisson vs Gaussian wells), fixed vs free parameters, Benford readout prior, duration, RT proxy mode, and related flags.  Individual scripts may expose CLI overrides; check their docstrings.

## Running the tests

```bash
pytest tests/ -v
```

## License

Research code — not intended for production use.
