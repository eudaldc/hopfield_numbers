# Technical note — continuously driven attractor for numerosity estimation

Status: methods complete; results sections are filled in from `driven_attractor/output/` as analyses finish. All equations here are what the code implements (`driven/model.py`, `driven/fitting.py`, `driven/rt.py`, `driven/cv.py`).

## 1. Question

The published model (M0) completes sensory accumulation into a scalar `y0 ~ N(n, sigma(n))` and only then relaxes on the attractor landscape. The candidate account registers items one at a time and drives the attractor online while evidence arrives. Does the driven account predict the trial-level data better, with duration entering only through physical exposure time?

## 2. Shared ingredients (identical to the registered analysis)

- Wells m = 1..19; widths sigma_m = sigma_c * m^p (SD units). g_m(x) = -log sigma_m - (x-m)^2/(2 sigma_m^2); V(x) = -log sum_m exp(g_m).
- Readout p(r | x) = softmax_m g_m(x) over the 19 wells, flat prior.
- Lapse per duration: p_final(r|n,d) = (1-lapse_d) p_model(r|n,d) + lapse_d/99 on r = 1..99.
- Likelihood: categorical over the (d, n, r) count table of all 52,995 trials; lapses profiled (golden section on the concave 1-D profile).
- Data: Cheyette & Piantadosi (2020) DURATION conditions; 379 participants, durations within-subject; RT timed from stimulus offset (established in `notes/stage0_audit.md`).

## 3. Evidence process (Stage 1)

Latent registered count K(t), K(0) = 0, capped at n. Inter-registration gaps

    gap_i = E_i / (rho * (n - i)^gamma),  E_i ~ Exp(1),  i = 0..n-1.

gamma = 0: serial capped accumulator (Poisson arrivals capped at n). gamma = 1: parallel independent registration (order statistics of n iid Exp(rho) latencies; K(d) ~ Binomial(n, 1 - e^{-rho d})). Optional tier-2 parameters: free gamma in [0,1]; global onset delay delta (arrivals shifted by +delta). Units: rho items/s, delta s.

The E_i are drawn once from a scrambled Sobol sequence (seed 20260710) and reused across every parameter evaluation and every model (common random numbers). Validation: the implied K(d) marginals match the exact Binomial / capped-Poisson laws (TV < 0.01 at M = 16384) and converge with M.

## 4. Dynamics (Stage 2)

Dimensionless time th = t / tau_x. During exposure (soft tether, primary M2):

    dx/dth = I[K > 0] (A(x) - B(x) x) + beta (K - x),

with A = sum_m w_m m / sigma_m^2, B = sum_m w_m / sigma_m^2, w = softmax(g(x)); A - Bx = -dV/dx exactly. x(0) = 0; the indicator gates the landscape force until the first registration. After offset (t = d) the tether is removed and dx/dth = -dV/dx.

Pulse variant (M2_pulse): dx = -dV/dx dth + 1.0 * dK (unit pulses, amplitude fixed).

Decision rule (response channel): the relaxation runs until |dx/dth| < 1e-3 (dimensionless force threshold; tau-independent) or until the physical cap T_CAP = 2.0 s. This cap plays the role that the step clamp + 100-step cap play in M0's published clean-up (bounded drift in the merged region); sensitivity is reported. Paths with K(d) = 0 respond as a pure uniform guess on 1..99 (stated zero-evidence rule).

### Controls (Stage 3)

- M1_offset (= M4 for the response channel): x(d) = K(d) from the same arrival process, then the same relaxation. Any M2 advantage over M1 isolates *online interaction during exposure*; any M1 advantage over M0 isolates the *discrete accumulation front end*. (In the pulse form, delivering all K(d) pulses at offset is exactly M1, so the batch-timing control M4 coincides with M1 for responses; they differ only in RT bookkeeping.)
- M3_static: identical K(t) and tether, landscape force removed during exposure and no post-offset relaxation; readout from x(d) with the same g_m. Response depends on (beta, tau_x) only through beta/tau_x, so M3 has one fewer identifiable parameter; noted in the comparison table.

### Parameter counts (response channel, lapses included)

| model | mechanism params | + lapses | k |
|---|---|---|---|
| M0 per-duration (published) | 4 x 5 | 5 | 25 |
| M0 shared landscape | 2 + 2 x 5 | 5 | 17 |
| M1_offset | sigma_c, p, rho, tau_x | 5 | 9 |
| M2_tether | sigma_c, p, rho, tau_x, beta | 5 | 10 |
| M2_pulse | sigma_c, p, rho, tau_x | 5 | 9 |
| M3_static | sigma_c, p, rho, beta/tau_x | 5 | 9 |

Every driven model is fully shared across durations: exposure duration enters only as the physical integration horizon d.

## 5. Numerics (Stage 5)

- Integrator: exponential Euler with frozen softmax weights, plus a sign guard: if the true total force changes sign between x and the candidate step end, the step is replaced by a bisection to the force zero (16 iterations). This is quasi-static equilibrium tracking in the stiff regime, plain exponential Euler in the smooth regime, and cannot jump an unvisited barrier. A naive displacement clamp was tried first and REJECTED: it made the effective maximum speed scale with 1/dt (logL shifted by thousands between dt = 5 and 2.5 ms in the parallel-registration case).
- Exposure step dt = 5 ms during global search, 2.5 ms for polish and all reported likelihoods (2.5 -> 1.25 ms changes total logL by ~3, below the QMC error).
- Post-offset relaxation precomputed on a 4601-point x-grid in dimensionless time with geometrically growing steps (h0 = 0.02, factor 1.05, horizon 400), then looked up per path; validated against fine-step RK4 (median |dx| < 0.008 at matched horizons, 0 crossings > 0.25).
- QMC: M = 1024/2048 paths per (n, d) cell for optimisation, M = 16384 for all reported likelihoods. |logL(16384) - logL(65536)| ~ 4-5; re-seeding at 16384 moves logL by ~4. Model contrasts below ~10 logL units are treated as within numerical uncertainty.
- Optimiser: differential evolution (Sobol init, seeded) in transformed (log) parameter space + Nelder-Mead polish; five lapses profiled analytically inside every evaluation.

## 6. Fitting and evaluation (Stage 6)

- Primary in-sample comparison: total response logL / AIC / BIC / mean log score, M0 (both forms, reproduced to 1e-11 in `repro_stage0.py`) vs M1/M2/M3 by both evidence processes.
- Participant-blocked 5-fold CV (seed 20260711): all models refit per fold (M0 included), scored on held-out participants; reported in total, per fold, and per participant.
- Parameter recovery (Stage 4): 40 synthetic datasets at jittered truths, full refits, identity R^2 / Spearman + error correlations, including derived quantities (n_crit = sigma_c^{-1/p}, sigma(5)).

## 7. RT channel (Stage 7)

RT is timed from stimulus offset, so the driven model's post-offset settling time (seconds) is the structurally correct predictor. Likelihood: log rt ~ N(log(t0 + a E[T] + b E[H]), sigma), t0 = 1.30 fixed, sigma profiled, landscape held at response-fit values. Stopping rules: A settling (primary), B diminishing information gain (|dH/dt| < 1 bit/s after 50 ms). Links: hybrid (a, b), dynamics-only, entropy-only, physical (a = 1 FIXED: settling time enters in literal seconds). M0 references recomputed under the identical fitting code.

## 8. Results

### 8.1 Response channel (in-sample; final scores at M = 16384, dt = 2.5 ms)

| model | process | k | logL | AIC | dAIC vs M0-perdur |
|---|---|---|---|---|---|
| M0 current, per-duration | - | 25 | -113150.6 | 226351.2 | 0 |
| M0 current, shared landscape | - | 17 | -114915.2 | 229864.4 | +3513 |
| M0 static twin, shared | - | 17 | -115235.5 | 230505.0 | +4154 |
| Weber/ANS (per-duration) | - | 20 | -115290.5 | 230621.1 | +4270 |
| M2 tether, free gamma (tier 2) | gamma=0.46 | 11 | -118362.6 | 236747.2 | +10396 |
| M3 driven static | serial | 9 | -118805.7 | 237629.4 | +11278 |
| M2 driven attractor (tether) | serial | 10 | -118857.5 | 237735.1 | +11384 |
| M1/M4 accumulate-at-offset | serial | 9 | -119029.0 | 238075.9 | +11725 |
| M2 driven attractor (pulses) | serial | 9 | -119138.2 | 238294.4 | +11943 |
| M1/M4 accumulate-at-offset | parallel | 9 | -119271.0 | 238560.0 | +12209 |
| M2 driven attractor (tether) | parallel | 10 | -119319.0 | 238658.0 | +12307 |
| M2 driven attractor (pulses) | parallel | 9 | -119785.8 | 239589.7 | +13238 |
| M3 driven static | parallel | 9 | -119929.5 | 239876.9 | +13526 |

(M0 warm refits with this project's optimiser squeeze a further ~12/~2 logL for perdur/shared — the registered baselines are if anything slightly understated; using the better values only widens the gaps above.)

Key contrasts:
- Best driven model vs M0-shared: **-3,892 logL** (M3-serial) / -3,944 (M2-tether-serial), i.e. ~7,700-7,900 AIC — the driven family is not close, despite the comparison giving it every numerical benefit (identical readout/lapse/support, final scoring at 8x the fit sample count).
- **M2 vs M1 (online vs offset delivery): +171 logL** for online — nonzero but ~4% of the gap to M0-shared.
- **M2 vs M3 (attractor force on vs off): -52 logL — the attractor force HURTS** in the driven regime (echoing the saturated-count ceiling analysis: relaxation moves x off the true count).
- Serial beats parallel registration in both M1 (+242) and M3 (+1,124).

### 8.2 Where and why it fails (decomposition)

Per-duration deficits of the driven fits vs M0-shared are spread over ALL durations (serial M1: -497/-479/-1269/-804/-1065 for 40/80/160/320/640 ms), not concentrated at one regime. Two components:

1. **Intrinsic ceiling (small).** The best possible saturated-count model at 640 ms — x = n deterministically, any (sigma_c, p), lapse profiled — reaches logL -19,345 (static) / -19,446 (with relaxation) vs M0-shared's -19,161: a floor deficit of ~200-300 per duration. Note relaxation hurts even at the ceiling.
2. **Cross-duration compromise (large, ~75% of the gap).** The 640 ms-optimal widths (sigma_c = 0.025, p = 1.76) are far sharper than the fitted global compromise (sigma_c = 0.16, p = 1.12). Because K(d) saturates (fitted rho = 189/s implies saturation by ~80-160 ms) while the data keep sharpening through 640 ms, the only duration-dependent quantity in the model (the count) stops carrying the duration effect, and one width law must serve all durations. It cannot: the data need sharp wells at long exposures (P(exact) ~ 0.95 out to n ~ 6) and broad effective noise at 40 ms.

The structural conclusion: **the exposure-duration effect in these data is an evidence-QUALITY effect, not an evidence-QUANTITY effect.** M0 carries it in sigma(n, d) = sqrt(s00^2 + s0_d n^{lambda0_d}); a registration process whose only duration dependence is the number and timing of registered items cannot reproduce it, for any rho, and neither serial nor parallel arrival statistics help. Onset delays (delta) and serial/parallel interpolation (gamma) do not touch this (tier-2 ablations: see table above / fits directory).

### 8.3 The fitted M2 degenerates to a count readout

The fitted tether stiffness beta/tau_x ~ 5,200 /s pins x(t) to K(t) within ~1 ms; trajectories ramp to n and sit there. The fitted landscape retains only ~5 stable fixed points (wells beyond n ~ 6 dissolve at sigma_c = 0.16, p = 1.12). 71% of post-offset relaxation paths hit the 2 s cap. Consequently the fitted 'driven attractor' IS an accumulate-and-read model; the data select that corner of the parameter space from within the driven family.

Stage-9 protocols confirm: regular vs early- vs late-clustered arrival times (equal count) produce IDENTICAL predictions; online vs batch delivery differs only at 40 ms and only in the 3rd decimal of p(exact). The fitted model makes no distinctive history-dependent predictions — the property that motivated the extension is exactly the property the data refuse.

### 8.4 Small-number exactness (the hoped-for improvement)

The driven model makes it WORSE, not better: at 320-640 ms the data hold P(exact) >= 0.9 out to n ~ 6 (a needle at r = n); M0-shared reproduces the needle (sharp sigma_c = 0.022 wells + small late-exposure sensory noise); the driven fit's compromised wide wells put ~0.47 at r = n for n = 4 at 640 ms with a heavy right tail.

### 8.5 RT channel (two-stage, identical convention, t0 = 1.30 s)

Shared mapping across durations (k = 3 hybrid / 2 single-term):

| source | link | logL | AIC |
|---|---|---|---|
| M0 shared landscape | steps + entropy | -76,478.3 | 152,962.6 |
| M0 shared landscape | steps only (k=2) | -76,478.3 | 152,960.6 |
| driven M2 settle | T_relax + entropy | -76,741.2 | 153,488.5 |
| driven M2 settle | entropy only | -76,951.7 | 153,907.4 |
| driven M2 entrate (info-gain stop) | hybrid | -76,951.7 | 153,909.4 |
| driven M2 settle | physical (a = 1, seconds) | -84,735.2 | 169,474.3 |

The driven model loses the RT channel too (-263 vs M0-shared hybrid), despite RT being measured from stimulus offset — the regime where a physical-time settling account should shine. The literal-seconds link (a = 1) fails by ~8,000 because settle times are cap-dominated (71% at T_CAP): in the merged region the landscape never settles, so E[T] is nearly flat in n. The information-gain stopping rule does not rescue it. Sensitivity: freeing tau_x in the RT stage only (grid 0.073-1.2 s, response channel untouched) never improves on the response-fit value — hybrid stays at -76,741 or worse and the physical link degrades monotonically with tau_x — so the RT loss is not an artefact of the two-stage convention.

### 8.6 Participant-blocked cross-validation (5 folds, seed 20260711)

Held-out log predictive density (all models refit per fold, warm-started; lapse estimated on train only):

| model | held-out lpd | mean/trial | folds won vs M0-shared | participants better than M0-shared |
|---|---|---|---|---|
| M0 per-duration | -113,187.3 | -2.136 | 5/5 (vs shared) | 259/379 |
| M0 shared landscape | -114,939.6 | -2.169 | — | — |
| M3 driven static (serial) | -118,832.7 | -2.242 | 0/5 | — |
| M2 driven tether (serial) | -118,923.3 | -2.244 | 0/5 | 65/379 |
| M1 accumulate-at-offset (serial) | -119,112.0 | -2.248 | 0/5 | — |

- The in-sample ranking replicates exactly out of sample; every driven variant loses to M0-shared in every fold, by ~700-900 lpd per fold.
- Only 65/379 held-out participants are better predicted by M2 than by M0-shared (31/379 vs M0-perdur); the median participant loses ~11 lpd.
- M2 beats M1 held-out by +189 (online timing carries a little information at 40 ms), but M3 beats M2 by +91: the attractor force subtracts predictive value inside the driven family, out of sample as well as in.
- Incidental but noteworthy for the manuscript: **M0-per-duration beats M0-shared out of sample in all 5 folds (259/379 participants)** — the per-duration landscape parameters generalise across participants and are not an overfitting artefact.

### 8.7 Robustness variant and tier-2 ablations

- **Pulse variant (M2_pulse, serial, k = 9): logL -119,138** — worse than the tether (-118,858) and worse than the offset control (-119,029). The structurally trap-immune delivery mechanism does not rescue the family.
- **Free gamma (serial-parallel interpolation, k = 11): logL -118,362.6 at gamma = 0.46** — the best driven variant, +495 over the serial tether, still **3,447 behind M0-shared** while spending an extra parameter.
- **Free delta (onset delay, k = 11): delta fits to 0.35 ms — numerically zero.** The data want no encoding-onset delay. (That run's reported logL of -119,064 sits below the nested delta = 0 fit, which is optimiser variance in the 6-dim space, not a real penalty: by nesting the free-delta MLE is >= -118,857.5.)

### 8.8 Numerical sensitivities (the gap is not an artefact)

- **T_CAP** (post-offset relaxation cap; fit used 2 s): re-polished logL at 1 s: -118,948; at 4 s: -119,257 (baseline -118,858). Both caps make the driven model slightly worse; the ~4,000 gap to M0-shared moves by < 10%.
- **Well support 1..19 -> 1..25** (evaluated at fitted parameters, both families): the extension mechanically awards model mass to the 20-25 guesses that the registered convention treats as lapse-only, so BOTH families gain — driven +2,415 (its wide wells spill more mass rightward), M0-shared +1,411. The driven-vs-M0 gap narrows from 3,942 to 2,935 and the ranking is unchanged. Per-cell predictions for r <= 19 shift by at most |dp| = 0.108, concentrated at n = 14-15 (right-tail spillover of the driven fit's wide wells); the small-n predictions are stable.
- QMC and dt: section 5 (final-score uncertainty ~4-5 logL; dt converged at 2.5 ms).

### 8.9 Parameter recovery (M2_tether serial, 32 synthetic datasets, jittered truths, full refits)

| quantity | identity R^2 | Spearman |
|---|---|---|
| sigma_c | 0.960 | 0.971 |
| p | 0.969 | 0.991 |
| rho | 0.992 | 0.996 |
| tau_x | -59.8 | 0.471 |
| beta | -23.5 | 0.351 |
| n_crit = sigma_c^{-1/p} | 0.998 | 0.999 |
| sigma(5) | 0.998 | 1.000 |

The landscape and the evidence rate are well identified (the sigma_c-p error correlation is -0.96, the familiar width trade-off, but the interpreted derived quantities recover essentially perfectly). **The two parameters that carry the "continuous driving" interpretation — tau_x and beta — do not recover at all**: once the fitted tether is stiff (beta/tau_x large), x(t) = K(t) for any values along the ridge, so the dynamical constants are structurally unidentifiable at the optimum the data select. Even if the driven model had been competitive, its driving dynamics would not have been interpretable quantities.

## 9. Limitations

- The response-channel decision rule (force threshold + 2 s cap) is a fixed convention, not fitted; M0's published clean-up embeds an analogous convention (step clamp + 100 iterations). Sensitivity to T_CAP is reported alongside the main table.
- The K(d) = 0 uniform-guess rule is a modelling choice; a baseline-well variant was prespecified as the alternative.
- The QMC likelihood is deterministic given the seed but carries ~4-5 logL units of residual integration error at the final sample size; contrasts of that order are not interpreted.
