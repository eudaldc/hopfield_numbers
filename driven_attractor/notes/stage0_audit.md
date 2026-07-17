# Stage 0 audit — current model, code, and data conventions

Audited 2026-07-10 against `scripts/trial_level_model_comparison.py` (the registered trial-level analysis), `scripts/trial_level_shared_landscape_ladder.py`, `manuscript/main.tex`, and `data/data_piantadosi.csv`. This note records the exact conventions the driven-attractor analysis must match, plus every discrepancy found between documentation and code.

## 1. The current model (M0), exactly as implemented

**Sensory channel.** A completed scalar estimate is sampled once per trial: `y0 ~ Normal(n, sd(n))` with `sd(n) = sqrt(0.05^2 + s0 * n^lambda0)`. The constant `S00 = 0.05` is fixed. `s0` therefore has *variance* units (it multiplies `n^lambda0` inside the square root); the well widths `sigma_n` are *standard deviations*. Both conventions are consistent between code and manuscript.

**Landscape (registered winner, `gaussian_pow_attractor`).** Wells at integers m = 1..19 (support = true range 1..15 plus `support_padding = 4`). Log-well values `g_m(x) = -log(sigma_m) - (x-m)^2 / (2 sigma_m^2)` with `sigma_m = sigma_c * m^p`. Energy `V(x) = -log sum_m exp(g_m(x))` (modern-Hopfield log-sum-exp; the readout softmax uses the same `g_m`). Note the `-log sigma_m` term acts as a precision bonus for small wells: it is the model's only mechanism for underestimation of large n (the sensory channel is unbiased and the lapse is uniform).

**Clean-up.** Discrete map `x <- x - clip(eta * pull, +-0.5*min(sigma))` with `eta = 1.0` fixed, `pull = sum_m softmax(g)_m (x-m)/sigma_m^2` (`KAPPA_LSE = 1.0` fixed), `MAX_STEPS = 120`, convergence when `|step| < TOL = 1e-4` **or when the step changes sign** (oscillation criterion). Important consequence: the step clamp is `0.5 * sigma_1 = 0.5 * sigma_c` *globally* (the smallest well sets it). At the fitted `sigma_c ~ 0.012-0.058 (80-640 ms)`, total movement is bounded by `120 * 0.5 * sigma_c ~ 0.7-3.5` evidence units, so above the resolvable range the "clean-up" is a small bounded drift, not full descent; the reported `E[steps] ~ 60-80` are essentially distance-travelled / clamp. This is documented behaviour (the RT "steps" metric depends on it) but worth keeping in mind: the published dynamics are a clamped fixed-point iteration, not a physical-time ODE.

**Readout.** `p(r | x*) = softmax_m(g_m(x*))` over the 19 wells, flat prior (no Benford readout prior; checked and rejected in the registered analysis).

**Lapse.** `p_final(r|n) = (1-lapse) * p_model(r|n) + lapse/99` over the full typed-response support r = 1..99 (`r_max` = max observed guess = 99). Out-of-well responses (r > 19) receive only lapse mass. Lapse fitted per duration, counted in AIC/BIC. ~4.8% of guesses exceed the well support.

**Trial-level likelihood.** Categorical over the (duration, n, r) count table: `p_theta(r|n,d) = sum_k w_k p(r | C(y0_k))` with y0 nodes from a deterministic uniform grid on z in [-7, 7] (Gaussian weights, renormalised); 201 nodes for optimisation, 1601 for reported log-likelihoods. Gauss-Hermite was shown to fail on the attractor's near-step integrand — a caution that applies equally to any driven-model integration scheme.

**Fitting.** Per duration: `differential_evolution` (maxiter 35, popsize 10, seed 73+d) + Nelder-Mead polish; 4 mechanism params (sigma_c, s0, lambda0, p) + lapse per duration (k = 25 over 5 durations). Shared-landscape variant: sigma_c, p shared, (s0, lambda0, lapse) per duration (k = 17), alternating shared/nuisance optimisation with warm starts.

**RT channel.** `log rt ~ Normal(log(t0 + link), sigma_rt)` with `t0 = 1.30 s` fixed; link = `alpha*E[steps] + beta*E[H]` (hybrid, the registered winner), fitted with the landscape held fixed at the response-fit values (two-stage; the coupled likelihood is deliberately out of the paper). sigma_rt profiled. The log-normal MLE targets the geometric-mean RT (documented; the earlier median comparison was an artefact).

## 2. Data facts

- `data/data_piantadosi.csv`, cond = DURATION, one row per **dot**; trials reconstructed by grouping (uniqueid, trial_id). 52,995 trials after filtering n_shown in 1..15 and rt > 0 (none dropped by these filters in practice; per-duration n = 10,600/10,601/10,609/10,594/10,591).
- **379 participants; 360 of them completed all five durations (within-subject, interleaved trial-by-trial); 150 trials per participant.** Durations are NOT between-group. (The manuscript's "N = 100 per condition" follows C&P's wording but does not describe this file; see discrepancy D4.)
- Typed guesses run 1..99. RT quantiles (s): 1% 0.78, median 1.99, 99% 10.6.
- **RT is timestamped from stimulus OFFSET, not onset.** Established empirically from the `datetime` stamps: the gap between consecutive trial stamps equals `exp_dur(next) + rt(next) + 1101 ms` with the residual *constant across exposure durations* (median 1101 ms, IQR 28 ms), whereas the onset hypothesis leaves a residual that grows exactly with exposure (1141 -> 1741 ms). So the exposure period is excluded from rt; rt runs from stimulus offset (mask onset) to response submission. This matters for a driven model in physical time: pre-offset processing time is not part of measured RT, and `t0 = 1.3 s` absorbs mask/typing/motor time.

## 3. Reproduction targets (from the registered outputs)

| quantity | value | source |
|---|---|---|
| winner per-duration response logL (40/80/160/320/640) | -28627.72 / -23426.26 / -21368.59 / -20801.34 / -18926.71 | freewidth_attractor.csv, pow_theta.csv |
| winner per-duration summed AIC (k=25) | 226351.24 | shared_landscape_ladder.csv (free_AIC) |
| winner shared-landscape logL / AIC (k=17) | -114915.218 / 229864.44 | shared_landscape_ladder.csv |
| shared winner theta | sigma_c=0.022219, p=1.79692 | shared_landscape_ladder_theta.csv |
| static twin shared logL / AIC (k=17) | -115235.518 / 230505.04 | shared_landscape_ladder.csv |
| Weber/ANS per-duration AIC (k=20) | 230621.07 | shared_landscape_ladder.csv |
| RT winner (pooled two-stage, per-duration fits) | hybrid AIC 30646.8/30176.3/30374.6/30601.2/30694.2 | pow_winner_rt.csv |
| RT pooled joint mapping | hybrid 152988.0, steps +500.7, entropy +853.8 | rt_joint.csv |

## 4. Discrepancies found (documentation vs code)

- **D1 (stale docs, not a bug).** `STATUS.md` and `docs/trial_level_model_comparison_notes.md` still name `gaussian_freebeta_attractor` as WINNER_KEY; the code and manuscript switched to `gaussian_pow_attractor` (power-law widths) on 2026-07-08/09. All comparisons here use the power-law winner, which is also exactly the landscape family specified for the driven model.
- **D2 (manuscript vs code, cosmetic).** Manuscript Eq. (RT) writes `RT = t0 + tau E[T] + kappa E[H]`; the code fits `log rt ~ N(log(t0 + alpha E[T] + beta E[H]), sigma)` — the equation describes the location parameter of a log-normal, i.e. the geometric mean, not the arithmetic mean. SM presumably states this; flagging for completeness.
- **D3 (mechanistic subtlety, documented above).** "Gradient descent until settling" in the manuscript corresponds in code to a clamped iteration whose maximum total displacement is bounded by `100 * 0.5 * sigma_c`; for merged wells the dynamics are a bounded drift. Any driven model must not be compared against an idealised unclamped M0.
- **D4 (participant structure).** The trial-level likelihood pools trials; there is no participant-level term anywhere in the current analysis, and durations are within-subject. Participant-blocked cross-validation (this project) is therefore a genuinely new, stricter test for *all* models including M0.
- **D5 (minor).** `load_trials` filters `rt > 0` and n_shown in 1..15; no trials are actually lost (52,995 in = 52,995 out), so reported "~53,000" is exact here.

## 5. Consequences for the driven-model design

1. Use the power-law landscape `sigma_m = sigma_c m^p` on wells 1..19, the same flat-prior softmax readout, the same lapse convention (per-duration lapse, uniform over 1..99), and the same count-table categorical likelihood — nothing else changes between M0 and the driven family.
2. RT-from-offset means the driven model's *post-offset settling time* (in seconds) is the natural RT predictor, and exposure-time processing is invisible to RT. This is a clean, falsifiable structural prediction of the driven account.
3. M0's published clean-up numerics (clamped map) stay untouched as the baseline; the driven family uses a continuous-time integrator (exponential Euler with matched displacement clamp), validated against fine-step gradient flow. Rankings must be shown stable to this choice.
4. All likelihoods for the driven family use fixed common random numbers (scrambled Sobol, seed recorded); final model scores recomputed at >= 4x the fitting sample count.
