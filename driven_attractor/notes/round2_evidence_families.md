# Round 2 — is the failure the capped count K(t), or online accumulation in general?

Follow-up to `decision_memo.md`. Round 1 rejected the driven attractor, but its evidence process was a **capped discrete count** K(t) in {0..n}. This round tests whether a better evidence process rescues online accumulation, and isolates *where* any gain comes from. Three families x four variants, all sharing the registered landscape (sigma_m = sigma_c m^p), the M0 clean-up operator, the flat readout, the per-duration lapse, and the count-table likelihood. Code: `driven/evidence2.py`, `driven/fitting2.py`; fits in `output/fits2/`.

## Families

- **ddm** — continuous noisy accumulation; running estimate u(d) ~ N(n, s00^2 + s0 n^lam / d): mean n, precision growing as 1/d (ideal-observer law). 2 sensory params for ALL durations (vs M0's 10).
- **kalman** — sequential Bayesian updating; posterior mean shrinks toward a prior N(mu0, v0) under uncertainty. 4 sensory params.
- **pulse** — imperfect item pulses: probabilistic registration (pi_hit), Gamma amplitudes (cv_a), Poisson false positives (phi); no exact cap at n. 4 sensory params.

## Variants

batch (x = u(d) at offset) vs online (tether during exposure) x attractor (M0 clean-up) vs static. So: batch_attr vs batch_static = clean-up; online vs batch = online delivery; online_attr vs online_static = attractor when driven.

## Validation (gate passed)

- **ddm/batch_attr reproduces M0 to machine precision** at all five durations when handed M0's per-duration noise (max |diff| 3.6e-12) -> the comparison is exact, not approximate.
- kalman nests ddm as v0 -> inf (checked). online_attr nests batch_attr as beta -> inf (checked). pulse marginals match the analytic mean/variance.

## In-sample results (final scores, 1601-node grid; 52,995 trials)

| model | k | logL | AIC | vs M0-shared (logL) |
|---|---|---|---|---|
| M0 per-duration | 25 | -113,150.6 | 226,351.2 | +1,764.6 |
| **Kalman batch, static (no attractor)** | 11 | **-114,052.0** | 228,126.0 | **+863.2** |
| Kalman batch + attractor | 11 | -114,287.2 | 228,596.4 | +628.0 |
| M0 shared landscape | 17 | -114,915.2 | 229,864.4 | 0 |
| DDM batch + attractor | 9 | -115,172.3 | 230,362.5 | -257.0 |
| DDM batch, static | 9 | -115,734.8 | 231,487.7 | -819.6 |

## Answers to the three key tests

1. **Does online delivery beat batch?** Marginally, and only for the attractor variant. A beta-sweep of ddm/online_attr at the batch landscape peaks at **+54 logL over batch at beta ~ 500** (the tether lets the attractor settle partly *during* exposure, not only after offset), then decays to batch as beta -> inf and collapses to -27,000 at weak coupling (the round-1 well-capture trap). +54 logL = 0.001/trial: real but negligible. Online never beats batch for the static variant. (`output/online_vs_batch_*.log`.)

2. **Does the attractor beat the matched no-attractor accumulator?** Family-dependent, and this is the important nuance. **DDM: yes** (batch_attr -115,172 > batch_static -115,735; clean-up +563). **Kalman: no** (batch_static -114,052 > batch_attr -114,287). At matched landscapes the story is the usual one: clean-up helps on sharp wells, hurts on wide wells; the Kalman-static fit simply prefers a wide-well landscape because its shrinkage channel already produces the underestimation that clean-up would otherwise supply.

3. **Does any model beat M0-shared out of sample?** YES for the Kalman family, robustly; NONE beats M0-per-duration. See CV below.

## The scientific point (in-sample, pending CV)

The round-1 conclusion "online accumulation is hopeless" was **too strong**: it was the *capped count* K(t) specifically. A proper continuous/Bayesian accumulator lands 3,000-3,900 logL closer to M0 than K(t). BUT the only family that beats M0-shared does so through its **prior-shrinkage sensory channel** (mu0 ~ 4, v0 ~ 50: the posterior mean regresses toward ~4 under uncertainty, reproducing large-n underestimation at short exposures), with the **attractor turned off**. So:

- **Online driving of the attractor** buys ~50 logL. Not a rewrite.
- **The attractor / clean-up** is supported on the plain-accumulation (DDM) channel (+563, replicating the manuscript's claim with a new front end), but is redundant with a shrinkage sensory channel.
- **The Kalman shrinkage channel** is the one thing that beats M0-shared in sample -- a claim about *where underestimation lives* (sensory prior vs landscape -log sigma term), not about online driving or attractor dynamics.

Whether the Kalman-static in-sample win over M0-shared survives participant-held-out CV, and how it compares to M0-per-duration (which it loses to in AIC by 1,775), decides the recommendation. Filled below when CV completes.

## CV (participant-blocked, 5 folds, seed 20260711)

Fixed-parameter generalization CV (`cv_fast.py`, `cv2_fast_summary.csv`): each model's full-data fit is evaluated on held-out participants, re-estimating only the per-duration lapse on the training fold (all models treated identically). This avoids per-fold landscape refits, which round 1 showed are stable across participant folds; the landscape is not the thing that varies across people, the lapse is.

| model | k | held-out lpd | vs M0-shared | participants > M0-shared |
|---|---|---|---|---|
| M0 per-duration | 25 | -113,166.3 | +1,768 | 265/379 |
| **Kalman batch, static** (no attractor) | 11 | **-114,071.3** | **+863** | **230/379** |
| Kalman batch + attractor | 11 | -114,307.3 | +627 | 223/379 |
| M0 shared landscape | 17 | -114,934.5 | 0 | -- |
| DDM batch + attractor | 9 | -115,192.4 | -258 | 170/379 |
| DDM batch, static | 9 | -115,754.3 | -820 | 165/379 |

The out-of-sample ranking is identical to in-sample. The Kalman shrinkage channel beats M0-shared robustly (230/379 held-out participants, median +2.85 lpd), with the attractor OFF (kalman-static > kalman-attr held-out as well). Nothing beats M0-per-duration.

## Recovery + posterior predictive (kalman_batch_static, `recover_pp_kalman.py`)

**Recovery (24 synths, full refits):** landscape and noise recover well (sigma_c R2 0.95, p 0.99, s0 0.95, lam 0.97), but **the two shrinkage parameters do not recover individually** (mu0 R2 -27 / rho 0.58; v0 R2 0.43 / rho 0.93). Same failure mode as the driven model's tau_x/beta: the mechanism that produces the headline win (prior shrinkage) is only weakly identified, because different (mu0, v0) pairs give near-identical shrinkage profiles. The fitted mu0 ~ 4 should be read as "a prior centred in the subitising range", not a precise value.

**Posterior predictive (`figures/kalman_batch_static_pp.png`):** the model fits the **mean guess excellently** at all durations (it reproduces the underestimation saturation directly through shrinkage) and matches the CoV, but **systematically under-predicts mid-range exact accuracy** P(r=n) for n ~ 3-8 at 320-640 ms -- the "shoulder" the manuscript attributes to the discrete clean-up basins. So shrinkage and the attractor capture *different* features:

- **Bayesian shrinkage** -> the mean estimate / large-n underestimation (what dominates the trial-level likelihood, hence the win over M0-shared).
- **Attractor clean-up** -> the P(exact) shoulder (what makes M0-per-duration, attractor + per-duration noise, still the best model overall).

Neither strictly dominates. The best-fitting model would plausibly combine a shrinkage sensory channel *with* the attractor clean-up -- but that is a sensory-front-end refinement, not the online/driven-attractor hypothesis this project set out to test, and it is future work, not a rewrite.

## Recommendation

**On the driven / online attractor: still not supported -- abandon for these visual-array data** (unchanged from round 1, now for a sharper reason). Online delivery buys ~50 logL; the attractor is either helpful (DDM) or harmful (Kalman) depending on the sensory channel, and where the best model lives (Kalman) the attractor is off. No online/driven variant is worth a manuscript change.

**But round 2 corrects one round-1 overstatement and surfaces one genuinely useful finding:**

1. *Correction.* "Online accumulation is hopeless" was too strong -- that was the **capped count K(t)** specifically. A continuous or Bayesian accumulator closes ~3,000-3,900 logL of the gap; a Bayesian one even beats the shared-landscape M0. The right negative claim is narrow: *a registration-count process whose only duration signal is the number/timing of counted items cannot carry the duration effect* -- not "online accumulation cannot."

2. *Finding (supplementary / future work, NOT a rewrite).* A **Bayesian-shrinkage sensory channel** (posterior mean regressing toward a prior ~4 under uncertainty) is a strong, parsimonious way to produce large-n underestimation: with a plain static power-law readout and no attractor it beats the shared-landscape attractor model in and out of sample (+863 lpd, 230/379 participants), while losing to the per-duration attractor model. This is a claim about the **sensory front end**, orthogonal to the attractor question, and it aligns with Cheyette & Piantadosi's own power-law-prior bounded-optimal account (already a cited competitor). The attractor's -log(sigma) precision term and a sensory prior are partly-degenerate routes to the same underestimation; the data mildly prefer the sensory-prior route when the landscape is forced to be shared.

**End classification (per the brief): the driven-attractor idea should be abandoned for the present visual-array data.** It might remain plausible only for *explicitly sequential* stimuli (auditory/tactile trains) where evidence genuinely arrives over time and the batch/online distinction is not a modelling artefact -- the `arrivals` hook in `evidence2.integrate_exposure` supports externally-timed pulses for exactly that test, which would need sequential-presentation data this dataset does not contain.
