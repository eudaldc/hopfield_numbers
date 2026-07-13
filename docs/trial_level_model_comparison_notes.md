# Trial-level model comparison — interpretation notes

> Reviewer-response analysis (i-Perception major point 1). Replaces the
> summary-curve profile-Gaussian AIC/BIC with a **trial-level** likelihood on
> the Cheyette & Piantadosi (2020) duration data. Code:
> `scripts/trial_level_model_comparison.py`. Outputs:
> `output/trial_level_model_comparison/`. **Not yet wired into the manuscript.**

> **STATUS / headline (updated 2026-07).** Winning model unchanged — the symmetric **super-linear-width Gaussian attractor** (free Benford exponent β) with a clean-up and a convergence-time (steps + entropy) RT channel; `WINNER_KEY = "gaussian_freebeta_attractor"`. The continuous width index `n_width` is defined by σ_n=1 and grows with exposure. It is distinct from `n_topo`, the last consecutive integer retaining an independent basin. The RT plateau occurs in a similar numerical region to `n_width`, but the proxy does not define that behavioural boundary. Response ladder: winner ΔAIC 0; the whole √n family sits +4400…+5650 behind. Joint ladder: winner + dynamics-RT best.

## Summary — key steps and decisions (most important first)

1. **The Point-1 fix (method).** Replaced the summary-curve profile-Gaussian AIC/BIC (ΔAIC ≈ 1.6, which the reviewer dismissed) with a **trial-level multinomial likelihood** over the ~53k individual integer guesses, plus a separate RT channel and a coupled joint. Deterministic **uniform-grid quadrature** (Gauss–Hermite fails on the near-step attractor integrand). Lapse/contamination = uniform over r = 1..99 (forced: ~4.8 % of guesses exceed the well support), counted in AIC, identical across models. This alone makes the comparison far more discriminating (ΔAIC tens–hundreds, not 1.6).

2. **First result → the liability is the well *shape*, not the clean-up.** Clean-up is supported (the attractor beats its no-clean-up twin in every well family), which rebuts reviewer points 2–3. But the *published* Hopfield-Poisson (right-skewed, √n width) is beaten by a plain Weber/ANS read-out → the **Poisson √n form** is the weak ingredient.

3. **Decisive follow-up → super-linear width + symmetric wells.** Freeing the well-width exponent, the data want **super-linear** width (σ ∝ n^1.5–2.2, rising with exposure), *not* accumulator √n. Three separable effects, all against the published parameterisation: (a) well **shape** symmetric-Gaussian ≫ skewed-Poisson; (b) **width** super-linear ≫ √n; (c) **clean-up**. → **Winner = super-linear-width Gaussian attractor + clean-up** (`gaussian_freebeta_attractor`), beating the whole √n family by +4400–5650 AIC. This turns every reviewer criticism constructive.

4. **Decision — well form = free-β (Benford), chosen by subitising.** Free-β beats the pure power-law on the n = 3 / subitising band, so it is the registered winner. Honest caveat kept: both forms still slightly under-predict subitising accuracy at n = 2–4 (ties to the OTS concession, reviewer point 6).

5. **RT breakthrough (reviewer point 5).** On the super-linear landscape, **convergence-time (steps) RT works** (it fails on √n), and **steps + entropy (hybrid) beats entropy alone** → the attractor's convergence time carries real RT signal beyond static read-out uncertainty.

6. **Full joint co-fit.** Co-optimising landscape + lapse + RT leaves the ranking unchanged and moves the landscape negligibly (the ~53k-trial response channel dominates). Consequence: RT can be treated as a **downstream cross-prediction** channel without contaminating the accuracy/landscape conclusions.

7. **Framing decision (2026-07).** **We fit the full range n=1..15**; an n≤10 refit (`trial_level_ncap_fit.py`) confirms that the fitted width profile does not depend on the tail. `n_width=σ_c^{-1/p}` is the continuous σ_n=1 width proxy. `n_topo` is the topological limit. Neither quantity is a density/texture threshold.

8. **RT correction (2026-07).** The log-normal RT MLE fits the **geometric mean** rather than the median. Re-scored against the geometric mean, dynamics-RT pooled R²≈0.86. The prediction keeps rising where empirical RT plateaus. Segmenting this mismatch around `n_width` is descriptive and does not make the width proxy a behavioural boundary.

9. **Modelling decisions on RT.** Keep the **log-normal** link. Do not truncate the RT fit at `n_width`; fit the latent-to-seconds link on all n. The lower-numerosity sensitivity window is reported separately.

10. **Aside — why P(exact) "wiggles" with only 4 parameters.** The 4-parameter landscape (σ_n, sensory sd) is smooth; the shoulder in P(r = n) is the **discrete cleanup basins on the fixed integer lattice** (it vanishes with clean-up off), i.e. a signature of the mechanism, not extra fitted degrees of freedom (`pexact_decomposition.png`).

**Standing rules / open items.** Report honestly (RT correction, subitising under-prediction, the 40 ms noise regime). Use `n_width` only for the continuous width-law proxy and `n_topo` for the last consecutive independent basin.

## What the analysis does (methods, fixed)

* **Data.** One row per trial, reconstructed exactly as in
  `fit_helpers.extract_empirical` (group by `uniqueid, trial_id`; keep
  `n_shown, n_guess, rt`; rt → seconds), but *without* collapsing to
  per-numerosity summaries. DURATION conditions, `exp_dur ∈ {40,80,160,320,640}`,
  true numerosities `n = 1..15`. ~10,600 trials per duration, ~53,000 total.

* **Response likelihood (main channel).** For each model and duration,
  `p_theta(r | n) = ∫ p_theta(r | C_theta(y0)) p(y0 | n) dy0`, with
  `y0 ~ Normal(n, sqrt(0.05² + s0·n^lambda0))`. `C_theta` is the attractor
  clean-up `x*` (dynamic models) or the identity (no-clean-up readouts);
  `p_theta(r | x*)` is the softmax over integer wells. The integral is
  **deterministic** (no Monte-Carlo noise).

* **Quadrature.** The no-clean-up integrand is smooth, so Gauss-Hermite is
  exact; the **clean-up integrand is near-step** (`y0` snaps to basins), where
  Gauss-Hermite is poor (≈1.7–4 % error even at 241 nodes). We therefore use a
  **uniform standard-normal grid** with Gaussian weights for every model (fair,
  same scheme throughout). Validated against high-sample Monte-Carlo: grid-801
  matches MC to ≤2e-3 (mean 1.5e-4). The optimiser uses a cheap grid
  (`--fit-nodes`, default 201); reported log-likelihoods are recomputed at the
  fitted parameters with an accurate grid (`--eval-nodes`, default 1601).

* **Lapse / contamination.** Typed responses run up to 99, and **~4.8 % of all
  guesses exceed the integer-well support** (`r > 19`). We never drop them:
  `p_final(r | n) = (1 − lapse)·p_model(r | n) + lapse·Uniform(1..R_max)`,
  `R_max = 99`. The lapse parameter is fit per duration, with an identical
  convention across all models, and is counted in AIC/BIC.

* **Response model comparison.** Categorical log-likelihood over the
  `(duration, n_shown, n_guess)` count table, summed over durations. Each model
  has 4 response parameters per duration (3 mechanism + lapse).

* **RT channel (secondary).** `log rt ~ Normal(log(t0 + metric(n)), sigma_rt)`,
  `t0 = 1.30 s`. Metrics: **entropy** (`kappa·E[H]`, available to every model),
  **convergence steps** (`c_steps·E[T_conv]`, attractor only), and **hybrid**
  (`alpha·E[T] + beta·E[H]`, attractor only). `sigma_rt` is profiled and counted
  in AIC/BIC. The entropy metric is the common, like-for-like RT channel; the
  steps / hybrid metrics test whether the *clean-up dynamics* carry RT
  information that a static readout cannot (reviewer point 5).

* **Joint channel (coupled).** `p(r, rt | n)` integrated over the *same* latent
  `y0` nodes: RT is a per-trial mixture of log-normals whose component weights
  are the node-posterior **given the observed response**, so the attractor's
  convergence-time spread feeds RT. Reported as `resp_logL + cond_rt_logL`. The
  joint headline uses the entropy metric for all models (uniform k); the steps /
  hybrid joints are reported for the attractor models as the dynamics test.

* **Models.** Hopfield–Poisson (attractor) · Poisson readout (no clean-up) ·
  Gaussian–Benford (attractor) · Gaussian–Benford readout (no clean-up) ·
  Weber/ANS · C&P log-prior **proxy** (clearly labelled; not a reproduction of
  Cheyette & Piantadosi). The first four form the **2×2** of
  *well family* (Poisson vs Gaussian–Benford) × *clean-up* (attractor vs static).
  To isolate those two factors cleanly, the readout prior is flat for all four
  (no Benford readout prior); the "Benford" label refers to the well-width
  scaling only. This differs from the SM1 Gaussian–Benford attractor, which
  added a Benford readout prior.

* **Fitting.** Deterministic objective; `scipy.differential_evolution` global
  search + Nelder-Mead polish; reproducible across runs (fixed seeds, no MC in
  the likelihood).

## Results

Fits: grid quadrature (fit 201 / eval 1601 nodes), differential evolution
(maxiter 35, popsize 10) + Nelder-Mead polish. ~10,600 trials per duration. Each
model has 4 response parameters per duration (k = 20 summed over 5 durations).
Lapse is stable across models (mean 0.073–0.082). Numbers below are summed over
the five durations unless noted; raw tables in
`output/trial_level_model_comparison/`.

### Headline table — response-only (summed over durations)

| model | resp logL | AIC | ΔAIC | per-duration wins |
|---|---|---|---|---|
| **Weber/ANS baseline** | −115290.5 | **230621.1** | **0** | 3/5 (40,160,320 ms) |
| Hopfield–Poisson (attractor) | −115474.0 | 230988.0 | +366.9 | 2/5 (80,640 ms) |
| Gaussian–Benford (attractor) | −115561.7 | 231163.4 | +542.3 | 0 |
| Gaussian–Benford readout (no clean-up) | −115813.0 | 231666.1 | +1045.0 | 0 |
| Poisson readout (no clean-up) | −115898.1 | 231836.2 | +1215.2 | 0 |
| C&P log-prior proxy | −116906.0 | 233852.0 | +3231.0 | 0 |

**The first thing to note: the trial-level likelihood is enormously more
discriminating than the summary curves.** Per-duration ΔAICs run from ~50 to
~700, and the summed ΔAICs from hundreds to thousands — versus the summary-curve
ΔAIC of 1.6 between Hopfield–Poisson and no-clean-up Poisson that the reviewer
(rightly) dismissed as "essentially no evidence". With ~53,000 discrete
responses the models are cleanly separated.

### Q1. Does Hopfield–Poisson beat no-clean-up Poisson at the trial level?

**Yes, decisively, on the response channel.** Hopfield–Poisson wins at **all five
durations** (ΔAIC per duration: 40 ms −253, 80 ms −172, 160 ms −173, 320 ms
−103, 640 ms −146; **summed ΔAIC ≈ 848** in favour of clean-up). On the joint
channel the clean-up advantage is **ΔAIC ≈ 905**. By any conventional reading
(ΔAIC > 10 ⇒ essentially no support for the worse model) the clean-up dynamics
are strongly favoured over the matched static Poisson readout.

### Q2. Is the advantage response-only, RT-only, or joint?

**Response-only (and it carries into the joint through the response term).** The
clean-up earns its keep in the *shape of the response distribution*, not in RT:

* RT-only (entropy metric), clean-up effect (no-clean-up − attractor AIC):
  Poisson **−7.8** (a wash), Gaussian–Benford **−906** (clean-up actually
  *hurts* RT, because the post-clean-up readout is near-deterministic and leaves
  too little entropy variation to track RT).
* Joint clean-up effect: Poisson **+905** (helps), Gaussian–Benford −113.

So the dynamics sharpen the response distribution toward the integers in a way
that matches the data's subitizing-range sharpness, but they do not improve RT
beyond the static posterior-entropy proxy.

### Q3. Does the clean-up term add anything beyond the Poisson family?

**Yes — the clean-up main effect is positive in both well families** (response
AIC, no-clean-up − attractor): Poisson **+848**, Gaussian–Benford **+503**. The
*well-family* main effect is by contrast **not robust**: among attractors,
Poisson beats Gaussian–Benford by 175; among static readouts, Gaussian–Benford
beats Poisson by 170. The data support "clean-up helps" far more clearly than
"Poisson wells are the right shape".

### Q4. Are the AIC differences large enough to support a strong claim?

Two different claims, two different answers:

* **"The attractor clean-up does real explanatory work"** — *strongly supported*.
  ΔAIC in the hundreds over the matched no-clean-up control, consistent across
  all five durations. This is the opposite of the summary-curve picture and
  directly answers the reviewer's points 2–3.
* **"Hopfield–Poisson is the single best account"** — *not supported*. A plain
  Weber/ANS readout (constant-CoV Gaussian noise, no clean-up) fits the bulk
  response distribution slightly better (ΔAIC ≈ 367 response, ≈ 727 joint; wins
  3/5 durations). The per-duration winner alternates between Weber and
  Hopfield–Poisson. The natural reading is that the *clean-up* is the supported
  ingredient, while the *Poisson well shape* is bettered by a Weber-law (linear-σ)
  decision profile for the large-n spread — an honest concession that connects to
  the reviewer's point 4 (Poisson-as-accumulator is a parametric choice, not a
  forced one) and suggests a Weber-shaped attractor as the natural next model.

### Q5 (reviewer point 5). Does a convergence-time RT differentiate the dynamics?

**On the √n landscapes, no; but on the super-linear landscape the data demand,
YES — and that resolves the puzzle.** Modelling RT from the clean-up's
convergence steps fits *much worse* than entropy on the **Poisson / fixed-Benford
√n wells** (RT-only ΔAIC **+5295** / **+5908** vs entropy) — which is why a
steps-based RT looked hopeless and was abandoned. But the response analysis shows
the data want **super-linear** wells, and on that landscape
(`gaussian_freebeta_attractor`, β≈1.8–3.4) the picture flips
(`scripts/trial_level_rt_exploration.py`):

| RT metric (winner landscape) | k | AIC | ΔAIC vs entropy |
|---|---|---|---|
| entropy `κ·E[H]` | 2 | 153637.7 | 0 |
| **steps `c·E[T_conv]`** | 2 | 153097.4 | **−540.4** |
| flow-time `c·E[T_flow]` | 2 | 159598.5 | +5960.8 |
| **hybrid `α·E[T_conv]+β·E[H]`** | 3 | 152867.9 | **−769.9** |
| flow+entropy | 3 | 153641.5 | +3.8 |

So the raw **convergence step-count beats the entropy proxy at equal k (−540)**,
and **steps+entropy beats entropy by −770** — the clean-up's *time-to-settle*
carries RT information beyond the static readout uncertainty, exactly the
"first-passage RT" the reviewer asked for. The effect is **landscape-specific**:
it appears only once the wells have the data-preferred super-linear width
(on √n wells the steps RT is +5000 worse). The continuous "flow-time" variant I
tried as a fix is *worse* (+5961): it is the raw step count, not a
clamp-corrected time, that tracks RT — the original step metric was right; the
*well shape* was the problem. (Per reviewer point 5, this turns a defensive
concession into a positive: the attractor dynamics drive RT, once the well shape
is correct.) Data: `output/trial_level_model_comparison/rt_exploration.csv`.

### Manuscript sentences (keyed to the result)

Because the result is two-sided, the honest wording reports both halves. Suggested
text (to be reviewed before any `.tex` edit):

> *Response-distribution channel.* "Re-running the comparison at the trial level —
> a categorical likelihood over the ~53,000 individual integer reports rather than
> a Gaussian residual over 15 summary points — separates the models far more
> sharply than the summary curves (per-duration ΔAIC of tens to hundreds, against
> 1.6 for the summary AIC). On this likelihood the attractor clean-up is clearly
> favoured over a matched static read-out: the Hopfield–Poisson model out-scores
> the no-clean-up Poisson read-out at every exposure duration (summed ΔAIC ≈ 850),
> and the clean-up advantage holds for the Gaussian–Benford family as well (ΔAIC ≈
> 500)."

> *Honest caveat (recommended to include).* "The trial-level likelihood does not,
> however, single out the Poisson well shape: a Weber/ANS read-out with
> magnitude-proportional noise fits the bulk response distribution marginally
> better (ΔAIC ≈ 370), so the evidence supports the clean-up dynamics more than
> the specific accumulator-derived well shape."

> *RT (point 5).* "A response-time channel derived from the clean-up's
> convergence time fits substantially worse than one derived from posterior
> entropy (ΔAIC > 5000), and adds nothing when combined with it; we therefore
> retain the entropy-based RT linkage and do not claim that the attractor's
> convergence dynamics are what fit the RT data."

**Bottom line for the rebuttal.** The trial-level reanalysis the reviewer asked
for *strengthens* the central mechanistic claim (clean-up matters; ΔAIC in the
hundreds, not 1.6) while honestly conceding the two weaker claims the reviewer
attacked (Poisson-over-alternatives is not clearly won; a convergence-time RT
does not improve on the entropy proxy).

### Follow-up: it is the *well-width scaling*, not Weber vs Benford per se

`scripts/trial_level_well_width_followup.py` tests the hypothesis that Weber/ANS
only beat the Gaussian–Benford / Poisson static read-outs because their width
exponent is *fixed* at β = 0.5 (giving σₙ ~ √n, the Poisson scaling), whereas
Weber uses σₙ ∝ n. Two static (no-clean-up) read-outs free that exponent:

| static read-out (resp-only, summed) | k | AIC | ΔAIC |
|---|---|---|---|
| Gaussian–Benford, **β free** | 25 | 227441.9 | 0 |
| Gaussian power-law σₙ = c·nᵖ, **p free** | 25 | 227677.9 | +236 |
| Weber/ANS (σₙ ∝ n, fixed) | 20 | 230621.1 | +3179 |
| Hopfield–Poisson (attractor, σ ~ √n) | 20 | 230988.0 | +3546 |
| Gaussian–Benford, β = 0.5 (σ ~ √n) | 20 | 231666.1 | +4224 |
| Poisson, no clean-up (σ ~ √n) | 20 | 231836.2 | +4394 |

Fitted exponents per duration (40/80/160/320/640 ms; widened bounds, all
interior): **β ≈ 2.59, 0.95, 1.29, 1.74, 1.90**; **p ≈ 2.32, 0.87, 1.43, 1.63,
1.69** (vs the fixed 0.5; Weber = 1). Note the exponent *rises with exposure*
(≈1 at 80 ms → ≈1.9 at 640 ms): longer viewing sharpens the small-n wells while
the large-n spread stays wide, demanding a steeper width function.

**On the AIC vs parameter count.** Raw log-likelihood always improves with more
parameters, so it is never compared directly; AIC charges +2 per parameter
precisely to neutralise that. The free-β model carries a +10 AIC handicap
(k = 25 vs 20) and still wins by thousands — and against its *own nested*
fixed-β=0.5 special case it improves AIC by ≈4224 for 5 extra β's, i.e. ≈+423
log-likelihood per freed β (a useless parameter buys ≈0.5). So the width
exponent is doing real work, not overfitting.

**Conclusions.** (1) Hypothesis confirmed: once the width exponent is free, a
Gaussian read-out *overtakes Weber comfortably* (ΔAIC ≈ 3179) — Weber's earlier
win was about the **width scaling**, not anything Weber-specific. (2) The data
want **σₙ growing linearly-to-super-linearly with n (exponent ~1–2), not the √n
of the Poisson / fixed-Benford wells** — estimation variability rises *faster*
than the accumulator (Var ∝ n) prediction (and faster than even Weber's
constant-CoV at most durations). This sharpens reviewer point 4: the Poisson
well shape is not merely "unforced", it is *actively dispreferred* at the trial
level. (3) These are **static** read-outs; the combined model (clean-up + free
width) is tested next. Data:
`output/trial_level_model_comparison/well_width_followup.csv`.

### Combined model: free-width wells + clean-up (the winner)

`scripts/trial_level_freewidth_attractor.py` puts the two supported ingredients
together — the attractor clean-up *and* the data-preferred super-linear well
width. Full response-only ladder (summed over durations):

| model | k | AIC | ΔAIC |
|---|---|---|---|
| **Benford wells, free β + clean-up** | 25 | 226186.4 | 0 |
| Power-law wells σₙ = c·nᵖ + clean-up | 25 | 226351.2 | +164.8 |
| Benford wells, free β (static) | 25 | 227441.9 | +1255.5 |
| Power-law readout, p free (static) | 25 | 227677.9 | +1491.5 |
| Weber/ANS (σ ∝ n) | 20 | 230621.1 | +4434.6 |
| Hopfield–Poisson (σ ~ √n, attractor) | 20 | 230988.0 | +4801.6 |
| Gaussian–Benford √n (no clean-up) | 20 | 231666.1 | +5479.6 |
| Poisson √n (no clean-up) | 20 | 231836.2 | +5649.8 |

**The clean-up effect at *equal parameters*.** The free-width attractor and its
static twin share the identical parameter space (clean-up is deterministic given
θ), so turning the dynamics on is a *zero-parameter* change. It improves AIC by
**+1326.7** (power-law family) and **+1255.5** (Benford family). This is the
cleanest possible demonstration of the clean-up mechanism — same model, same k,
flip the dynamics on → ΔAIC ≈ 1300 — and it decisively answers the reviewer's
"the clean-up contributes very little". (Per duration, clean-up helps strongly at
40–160 ms and is roughly neutral/slightly negative at 320–640 ms.) E[T_conv] ≈
60–81 confirms the dynamics actually run despite the wide large-n wells.

**Fitted width exponent *with* clean-up** (40/80/160/320/640 ms): power-law
p ≈ 0.94, 1.50, 1.80, 1.99, 1.85; Benford β ≈ 3.42, 1.81, 2.02, 2.19, 2.15 —
still firmly super-linear, so the √n Poisson shape is rejected even after
clean-up.

**Overall reading.** The original model had the right *mechanism* (clean-up) but
the wrong *well shape* (Poisson √n). Correcting the shape to super-linear and
keeping the clean-up gives the best model by a wide margin: it beats the
published Hopfield–Poisson by ΔAIC ≈ 4800 and Weber/ANS by ≈ 4400. Two separable
effects: **well shape** (√n → super-linear) buys ~3000–4400 AIC, and **clean-up**
(at the right width, equal k) buys ~1300 more. Data:
`output/trial_level_model_comparison/freewidth_attractor.csv`.

### Equal-footing control: free the *Poisson* variance exponent too

The free-width Gaussian read-outs carry one parameter the Poisson family lacks
(the Poisson well is structurally locked to Var ∝ n). To rule out a
parameter-count artefact, `scripts/trial_level_poisson_freevar.py` gives the
Poisson *shape* a free variance exponent (Var ∝ n^qvar; qvar = 1 is the standard
Poisson, verified to reproduce it to machine precision) — so it too has k = 25.

Full response-only ladder (summed over durations):

| model | k | AIC | ΔAIC |
|---|---|---|---|
| Benford wells, free β + clean-up | 25 | 226186.4 | 0 |
| Power-law wells + clean-up | 25 | 226351.2 | +164.8 |
| Gaussian–Benford, free β (static) | 25 | 227441.9 | +1255.5 |
| Gaussian power-law, p free (static) | 25 | 227677.9 | +1491.5 |
| **Poisson, free variance + clean-up** | 25 | 230492.7 | +4306.3 |
| **Poisson, free variance (static)** | 25 | 230577.8 | +4391.4 |
| Weber/ANS | 20 | 230621.1 | +4434.6 |
| Hopfield–Poisson (√n, attractor) | 20 | 230988.0 | +4801.6 |
| Gaussian–Benford √n (no clean-up) | 20 | 231666.1 | +5479.6 |
| Poisson √n (no clean-up) | 20 | 231836.2 | +5649.8 |

**Conclusion — it is the well *shape*, not the parameter count.** With matched
parameters and a free width exponent, the Gaussian read-out still beats the
Poisson read-out at **every duration** (static ΔAIC ≈ +2900–3140 summed; per
duration +1675/+190/+191/+340/+503). Freeing the variance helps the Poisson a
little (~1258 vs the locked √n) and brings it level with Weber, but it plateaus
~3000–4300 behind the Gaussian and cannot be rescued by clean-up either
(clean-up adds only **+85** to the free-variance Poisson, vs **+1300** to the
Gaussian). With clean-up the Poisson even reverts toward √n (qvar → ~0.7–1.1).

So there are **three** separable, mutually-reinforcing effects, all pointing away
from the published parameterisation:

1. **Well shape**: symmetric Gaussian ≫ skewed Poisson (~3000–4300, equal k).
   The Poisson PMF is right-skewed, but human estimates are if anything
   *under*-dispersed-upward (underestimation at large n), so the accumulator's
   skew is the wrong sign for the data.
2. **Width scaling**: super-linear (σ ∝ n^1.5–2) ≫ the accumulator √n
   (~3000–4400 within the Gaussian family).
3. **Clean-up**: at the right shape+width, +~1300 at equal k (the cleanest
   single demonstration that the attractor dynamics do real work).

This makes the response to reviewer point 4 fully constructive: the
"accumulator/Poisson" well is not merely an unforced label — at the trial level
it is *doubly* dispreferred (wrong shape *and* wrong width), and the data favour
a symmetric, super-linear-width attractor. Data:
`output/trial_level_model_comparison/poisson_freevar.csv`.

### Joint (response + RT) on the winning landscape — the dynamics drive both

Closing the loop with the coupled joint `p(r, rt | n)` for the winning model
(`scripts/trial_level_winner_joint.py`), comparing an entropy RT against the
convergence-dynamics RT (hybrid = steps + entropy). Joint ladder (entropy joints
of the six originals + the winner), summed over durations:

| model | k | joint AIC | ΔAIC |
|---|---|---|---|
| **winner + dynamics RT (hybrid)** | 40 | 378149.3 | 0 |
| winner + entropy RT | 35 | 378932.8 | +783.5 |
| Weber/ANS | 30 | 382062.3 | +3912.9 |
| Hopfield–Poisson | 30 | 382789.3 | +4640.0 |
| Gaussian–Benford no clean-up | 30 | 383617.7 | +5468.4 |
| Poisson no clean-up | 30 | 383694.1 | +5544.8 |
| Gaussian–Benford attractor | 30 | 383730.7 | +5581.4 |
| C&P log-prior proxy | 30 | 385719.7 | +7570.3 |

The free-width attractor wins the joint by ΔAIC ≈ 3900 over Weber and ≈ 4640 over
the published Hopfield–Poisson, and **within the winner the convergence-dynamics
RT beats the entropy RT by ΔAIC 783** (even with the extra RT parameters). So on
the data-preferred landscape the attractor dynamics carry *both* channels:
clean-up sharpens the **response** distribution (≈1300 at equal k) and the
**convergence time** improves **RT** (steps beat entropy −540 RT-only; hybrid
joint beats entropy joint −783). Data:
`output/trial_level_model_comparison/winner_joint.csv` (+ `winner_theta.csv`).

### Where this leaves the model (synthesis)

A single correction — replace the Poisson √n wells with **symmetric,
super-linear-width** wells — turns every reviewer criticism into support:
* clean-up (the core mechanism) demonstrably helps responses (point 2/3);
* the Poisson/accumulator form is shown to be the weak part, not the attractor
  (point 4), and the data point to a concrete better shape;
* a convergence-time RT — the reviewer's own suggestion (point 5), which had
  failed on √n wells — now beats the entropy proxy, so the dynamics drive RT.

The recommended model for the revision is the **free-/super-linear-width Gaussian
attractor with a convergence-time (or steps+entropy) RT channel**.

### Full joint co-fit (landscape + lapse + RT optimised together)

`scripts/trial_level_joint_refit.py` (`fit_joint_full`, warm-started from the
two-stage fit so the joint can only match/beat it) co-optimises everything on the
joint likelihood. Ladder (summed AIC):

| model | k | joint AIC | ΔAIC |
|---|---|---|---|
| Winner (super-linear attractor) + dynamics RT | 40 | 377875.2 | 0 |
| Winner static twin (no clean-up) + entropy RT | 35 | 378881.4 | +1006.1 |
| Weber/ANS + entropy | 30 | 382029.8 | +4154.6 |
| Hopfield-Poisson (√n) + dynamics RT | 35 | 382686.2 | +4811.0 |
| Poisson √n (no clean-up) + entropy | 30 | 383559.5 | +5684.2 |

Co-fitting **does not change the ranking** (winner wins; clean-up helps +1006;
√n + Weber far behind) — it only nudges totals (winner 378149 → 377875).

**RT — corrected (2026-07): most of the apparent weakness was a median-vs-skew
artefact.** We had been quoting the winner's RT R² as 0.61/0.61/0.65/0.70/0.76
(pooled **0.69**), comparing the prediction to the **median** RT curve. But the
log-normal RT MLE fits the *mean of log RT* per n, i.e. the **geometric mean**
(`exp(mu)`), not the median. Log RT is strongly **right-skewed** (skew ≈ +1.5 to
+3), so the empirical median sits ~0.1–0.2 s *below* the geometric mean, and
comparing the geometric-mean-targeting prediction to the median deflated every R².
Evaluated consistently against the **geometric mean** (`winner_rt_fit.{png,csv}`,
dots now geometric mean with the median shown faint):

* dynamics (steps+entropy) RT R² = **0.93 / 0.82 / 0.83 / 0.87 / 0.87** (40–640 ms),
  **pooled 0.86** — close to, and a genuine *cross-prediction* of, the published
  in-sample Fig. 2 (0.89–0.98). Dynamics beats entropy at 4/5 durations, so the
  attractor's convergence time carries real RT signal.

The residual is a chronometric mismatch: the model tracks geometric-mean RT over
lower numerosities, but its monotone deliberation-time prediction keeps rising
where the data plateau. The `rt_domain.png` segmentation around `n_width` visualizes
this mismatch; it does not define an exact-range or model-validity boundary.

Caveats kept honest: (1) log RT is not exactly log-normal (the skew is real), so the
geometric mean is the *consistent* statistic for this model, not a perfect one;
(2) Fig. 2 was *in-sample*; ours is cross-prediction; (3) a mild over-prediction
at larger numerosities remains. **Do not truncate the RT fit at n_width**; at
80 ms n_width≈7.5 cuts just before saturation and the truncated fit overshoots
(R²|domain 0.41); fit the latent→seconds link on all n and *report* within-domain.
**Revised conclusion: RT is a solid cross-prediction (dynamics pooled R² 0.86)
over lower numerosities, but the model misses the empirical plateau.**

### Robustness / caveats

* **Convergence.** Re-fitting at a higher optimiser budget (grid-201,
  maxiter 50, **a different seed**) changes the reported log-likelihoods by
  ≤ 1 (Hopfield-Poisson @160 ms −1.0; @640 ms −8.3, i.e. the saved fit was
  *better*) and by ~0 for the converged no-clean-up and Weber models
  (`output/trial_level_model_comparison/convergence_check.txt`). The fits are
  not under-converged; the attractor edge over its control and the Weber edge
  over the attractor are both stable, not optimisation artefacts. (Note: the
  *attractor* models need more optimiser budget than the smooth static
  read-outs — a low-budget fit understates them — which is why the headline
  fit uses differential evolution + Nelder-Mead polish.)
* **Lapse and the predicted mean.** The uniform contamination (over r = 1..99,
  forced because typed responses reach 99) is correct for the *likelihood*
  (positive mass everywhere; identical across models, so ΔAIC is unaffected)
  but inflates the *predicted mean guess* by ~`lapse × 50`. The
  `predictive_summaries.png` mean-guess row therefore plots the lapse-free
  "core" model mean against the in-support empirical mean; the response-shape
  comparison itself uses the full lapse-inclusive likelihood and the heatmaps.
* **What "Weber wins" does and doesn't mean.** The Weber/ANS read-out is itself
  a *static* read-out, so its win over Hopfield-Poisson is a statement about the
  *well shape* (linear-σ vs Poisson-√), not about clean-up: clean-up still helps
  within every well family. A Weber-shaped attractor (clean-up + linear wells)
  is the obvious untested model these results point to.

## Small-range scope — a mechanistic small-number model that *predicts* the ANS transition (2026-07)

Framing (author's call): the model is **not a unified OTS+ANS account** but a
**mechanistic model of small-range (exact/subitising) numerosity** whose virtue is
that it predicts, as an *emergent* property, the exact→approximate transition
(n ≈ 4–10) — a single graceful degradation, not a separate ANS module bolted on.
This is a scope **argument made post-hoc on the full-range fit** (n = 1..15), *not*
a restriction of the fit: we keep training on all numerosities (no arbitrary
cutoff, all data used) and then show that the exact/discrete behaviour emerges only
in the small range.

Evidence, all from the full-range winner fit (`winner_theta.csv`):

* **The discrete signature is concentrated at small n.** The clean-up's per-n
  contribution to the response likelihood (`regime_change.csv`) is large at
  n ≤ 3–4 and ≈ 0 by n ≥ 5 — the attractor actively snaps evidence to integers
  only in the subitising band.
* **n_width lies inside the fitted range.** The width-law criterion
  σ_n = 1 ⇔ adjacent-well d′ = 1 (`ncrit_rederivation.csv`, benford-free-β row) is
  crossed at ≈ 4.6, 7.5, 8.7, 9.2, 10.0 for 40/80/160/320/640 ms — growing with
  exposure, well within n=1..15. It summarizes the fitted width profile and is not
  an exact-range or topological boundary.
* **One mechanism, graded readout.** The same integer lattice produces progressively
  coarser responses as the fitted components broaden and overlap.
  The width crossing σ_n=1 is a continuous proxy. Basin disappearance is measured
  separately by n_topo.

(`scripts/trial_level_ncap_fit.py` re-fits on n ≤ 10 as a one-off check and returns
the same β and n_width for 80–640 ms — the width profile is not an artefact of the
fitting range — but this robustness check is not part of the main argument.)

## De Carli developmental dyscalculia (SM5) — winner model applied (2026-07)

Context: a reviewer suggested dropping the DD group-comparison (SM5); the author wants to keep it, made as strong and honest as possible. We applied the winning `gaussian_freebeta_attractor` to the De Carli exact-enumeration data (accuracy + RT per subject × numerosity, 32 DD vs 32 controls, **n=1..7**). Scripts: `decarli_gaussian_freebeta.py` (group fit), `decarli_gaussian_freebeta_subjects.py` (per-subject n_width), `decarli_locus.py` + `decarli_locus_clean.py` (width-vs-noise locus), and `decarli_dd_summary_figure.py`.

First, the winner model fits both groups well, and the De Carli data independently favour super-linear width (β≈1.4–1.6). The fitted **n_width≈8** describes this width profile; it is not a boundary on exact enumeration.

The DD argument has three tiers, strongest-anchored first:

* **Tier 1 — model-agnostic deficit (bulletproof).** DD are ~10 % less accurate at n = 4–6 (**Cohen d = −0.62**) and ~0.3 s slower at n = 4–7 (**d = +0.61**), the difference **concentrated at n = 4** (Mann–Whitney p = 0.001) — the subitising edge, not a uniform slowdown. Needs no model.
* **Tier 2 — the deficit is width-shaped, not noise-shaped.** With an exhaustive global grid search and a fixed common lapse, a width-only model fits each subject's accuracy curve better than a noise-only model: **64/64 subjects, mean SSE 0.010 vs 0.079, Wilcoxon p=3.5e-12**.
* **Tier 3 — the group difference localises to the fitted width profile.** Per-subject **n_width: DD 7.9 vs control 8.7**; accuracy-only **p=0.044**, with RT included **p=0.007**. Sensory noise shows only a weak trend and fits worse.

**Interpretation for the paper:** the fitted group difference is in the width profile rather than the sensory-noise parameter. The analysis does not estimate n_topo.

Honest caveats: coarse data (8 trials/cell); exploratory; noise, width, and lapse are partly degenerate; a small noise component may coexist. Much of the power gain is the per-subject design, not the model per se.

