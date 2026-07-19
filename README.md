# A Hopfield attractor account of numerosity perception

Research code for the paper **A Hopfield attractor account of numerosity perception**, fitted to behavioural data.  Noisy scalar evidence is refined by gradient descent on an energy landscape until it settles near a discrete numerosity; well spacing and width implement a “useful discreteness” that weakens for larger numbers.

The codebase fits use `scipy` global optimization on empirical summary statistics (accuracy, mean guess, RT where available), with shared simulation logic in `scripts/fit_helpers.py`.  Configuration for well type, noise, RT mode, and optional ablations lives in `configs/config.yaml`.

## Model sketch

```
y₀  →  cleanup (gradient descent on E(x | y₀))  →  x*  →  posterior p(n | x*)  →  loss vs data
```

Energy (Poisson or Gaussian wells; see `fit_helpers` and `docs/`, and the manuscript). Encoding noise and RT linkage are controlled via YAML and the fit scripts.

<p align="center">
  <img src="figures/figure1_conceptual_attractor_wells.png" alt="Conceptual attractor wells illustration" width="600"/><br>
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
output/        Fit summaries, exploratory plots, CSVs from analyses
figures/       Publication figures used by the manuscript
manuscript/    LaTeX for the paper and supplementary material
docs/          Model notes and session logs
tests/         pytest suite
```

## Main scripts

| Script | Purpose |
|--------|---------|
| `scripts/figure1_conceptual_attractor_wells.py` | Main conceptual attractor figure |
| `scripts/figure2_model_comparison.py` | Main model-comparison and individual-differences figure |
| `scripts/trial_level_model_comparison.py` | Shared trial-level likelihood and model definitions |
| `scripts/trial_level_shared_landscape_ladder.py` | Joint response-model comparison with one landscape shared across exposure durations |
| `scripts/trial_level_clean_up_budget.py` | Profile likelihood for the shared give-up budget \(G\) |
| `scripts/trial_level_rt_joint.py` | Shared RT link on the response-fitted joint landscape |
| `scripts/trial_level_cofit_figure.py` | Main fitted-response and RT figure from the joint model |
| `scripts/trial_level_bivariate_id.py` | Hierarchical individual-differences analysis for landscape resolution and Weber noise |
| `scripts/trial_level_pexact_decomposition.py` | Clean-up contribution to exact-response probability |
| `scripts/trial_level_weber_fraction.py` | Effective Weber fraction implied by the joint fit |
| `scripts/decarli_gaussian_freebeta.py` | Winner model applied to De Carli (SM5): group accuracy+RT fit + derived n_width/σ(n) |
| `scripts/decarli_gaussian_freebeta_subjects.py` | Per-subject De Carli n_width distributions, DD vs control |
| `scripts/decarli_locus.py` / `decarli_locus_clean.py` | Is the DD deficit resolution or sensory noise? (mirror fit + clean global test) |
| `scripts/decarli_dd_summary_figure.py` | Consolidated DD-vs-control 4-panel argument (`figures/supfigure4_developmental_dyscalculia.png`) |

The retired pipeline that fitted an independent landscape at each exposure duration is archived under `legacy/trial_level_per_duration/`.

Run from the repository root (e.g. `python scripts/fit_piantadosi.py …` or `python -m scripts.density_transition` as documented in each file).

## Configuration

`configs/config.yaml` drives the attractor variant (e.g. Poisson vs Gaussian wells), fixed vs free parameters, Benford readout prior, duration, RT proxy mode, and related flags.  Individual scripts may expose CLI overrides; check their docstrings.

## Running the tests

```bash
pytest tests/ -v
```

## License

Research code — not intended for production use.
