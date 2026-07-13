# Response to Reviewer 1 — background drafts

Draft material for the response letter. For each point: the reviewer's comment first, our response below. Covers point 3 (Hopfield framing), point 6 (unified account / subitizing), and point 8 (per-duration fitting) so far.

---

# Point 3: "The Hopfield framing is largely decorative"

Tone: acknowledge the one fair part (dimensionality), then establish that the framing is mechanistically exact and now empirically load-bearing.

## The reviewer's comment

> *The Hopfield framing is largely decorative given the actual computations. Stripped of the framing, the model is: noisy sensory sample y₀ ~ N(n, s₀n^λ₀); gradient descent in 1D on a log-sum-exp of Poisson log-likelihoods; softmax readout over integers. In one dimension, gradient descent on a log-sum-exp landscape that is itself the log-mixture of unimodal likelihoods is closely related to a noisy MAP estimator on the same mixture. The reference to Ramsauer et al. (2021) is presentationally apt but mechanistically slight: modern Hopfield networks are interesting because they store many high-dimensional patterns with exponential capacity; here we have 15 ordered scalars on a line. What specifically the attractor dynamics buy beyond a Poisson readout is what the no-cleanup baseline is supposed to isolate, and per point 2, the answer appears to be "very little."*

## Our response

We thank the reviewer for pressing on this: it made us state the mechanism precisely, and we think the precise statement supports the framing rather than undermining it. There are three separate claims in the comment, and we address them in turn.

### 1. The clean-up *is* the retrieval rule of a modern Hopfield network — this is an identity, not an analogy

The reviewer writes that, stripped of the framing, the model is "gradient descent on a log-sum-exp landscape … closely related to a noisy MAP estimator." We agree with the description and would add only that this *is* the modern Hopfield computation, not something distinct from it. The continuous / modern Hopfield network of Ramsauer et al. (2021) is defined by exactly this energy and this update:

- Energy: the attraction potential is the log-sum-exp over the stored patterns, `V(x) = − log Σ_n exp(g_n(x))`, which is the modern-Hopfield energy with the integers as the stored patterns.
- Update: gradient descent on `V` gives `∂V/∂x = Σ_n softmax(g_n(x)) · (x − n)/σ_n²`, so each step moves the state toward a **softmax-weighted (precision-weighted) average of all stored patterns**. For equal widths the fixed point is `x* = Σ_n w_n · n` with `w_n = softmax(g_n(x))` — term for term the retrieval rule of Ramsauer et al. (their Eq. for the update `ξ_new = X softmax(β Xᵀξ)`).

So "gradient descent on the log-sum-exp of the well likelihoods," "softmax retrieval in a modern Hopfield network," and "iterated MAP on the mixture" are three names for the same operation. The reviewer's contrast between a MAP estimator and a Hopfield network is therefore not a contrast between two mechanisms; it is one mechanism under two names. The defining feature — that **every stored pattern contributes to every update through the softmax**, rather than the state snapping to a nearest neighbour — is present here in full. This is what the "attractor" language denotes, and it is exact.

We have made this explicit in the Methods (the descent moves the state toward a softmax combination of all wells, so every stored integer contributes to the outcome) and in the Discussion (the clean-up is associative retrieval in the modern Hopfield sense), so a reader sees the identity rather than a loose analogy.

### 2. On dimensionality and capacity — the one place the reviewer is right, and what we actually use

The reviewer is correct that modern Hopfield networks are celebrated in machine learning for storing *many high-dimensional* patterns with near-exponential capacity, and that we do not have that: we have on the order of fifteen scalar patterns on a line. We do **not** claim the capacity result, and we have removed any wording that could imply we do (the word "Hopfield" is no longer in the title).

What we use is a different property of the same formalism: associative retrieval over a pattern set that is **metrically ordered**. Generic modern Hopfield networks store arbitrary, roughly orthogonal patterns; ours store the integers in their natural order, with widths that grow with magnitude. This ordering is not incidental — it is the source of the paper's central result. Because neighbouring patterns overlap, the softmax competition among them is sharp for small, well-separated numbers and broad for large, overlapping ones. We summarize the width law with `n_width` and determine the last consecutive independent basin with `n_topo`. A high-dimensional, orthogonal-pattern Hopfield network has no such structure and makes no such prediction. So the associative retrieval earns its place through the ordered-pattern regime, which is exactly the regime a magnitude line lives in, rather than through the capacity regime the reviewer rightly notes we do not occupy.

### 3. "The dynamics buy very little" — this was a consequence of the summary-curve analysis, which we have replaced

The reviewer's final claim, that the dynamics "buy very little," is drawn from point 2, i.e. from the summary-curve profile-Gaussian AIC (ΔAIC ≈ 1.6 between the attractor and its no-clean-up control). Following the reviewer's point 1 we have replaced that analysis with a trial-level likelihood over the ~53,000 individual integer responses, and the picture is now the opposite:

- Turning the clean-up on is a **zero-parameter change** (the dynamics are deterministic given the landscape), so the attractor and its static twin occupy the identical parameter space. At the trial level this flips the response fit by **ΔAIC ≈ 1256** in favour of the dynamics (and by a similar margin in a second well family). The same result holds under the constrained model the reviewer asks for in point 8: with a single landscape shared across durations the dynamics still improve the fit at equal parameters by ΔAIC ≈ 781 (see point 8). This is the cleanest possible isolation of what the dynamics buy, and it is not "very little."
- The convergence time of the clean-up also predicts response time better than the static entropy proxy that such models usually rely on (Supplementary Material 2 and the RT panel), so the dynamics carry a second, chronometric signature.

The associative retrieval is therefore not decorative in the operational sense either: at equal parameter count it does measurable work in both the response distribution and the response times. (We also verify with BIC, which penalises parameters far more heavily, that the ranking is unchanged; and a magnitude code given the same free width parameter still loses.)

### Summary of the position

- The model is, exactly and not by analogy, a modern Hopfield network: the energy is the log-sum-exp over the stored integers and the clean-up is the softmax-weighted retrieval update.
- We concede the reviewer's dimensionality point and claim none of the high-dimensional capacity results; what we use is associative retrieval over an **ordered** pattern space, whose fitted width law and basin topology are summarized by `n_width` and `n_topo`.
- The claim that the dynamics buy little came from the summary-curve analysis the reviewer asked us to replace; at the trial level the dynamics improve the fit by ΔAIC ≈ 1256 at equal parameters and additionally carry the response times.

We have kept "Hopfield" out of the title and stated the mechanism where it belongs (Methods and Discussion), so the manuscript makes the mechanistic claim precisely without leaning on the capacity analogy.

---

# Point 6: "The unified account is overstated, and the subitizing claim does not survive scrutiny"

Tone: the first half (density) is fully resolved by the revision. On subitizing, concede the literal facts (no spatial component, no objects, no attention — true by construction), then make the level-of-explanation argument: those properties live upstream of the model, in the evidence channel, and the model's claim is not about the process signatures of subitizing but about the format of the representation and where exactness ends.

## The reviewer's comment

> *The "unified account of the three regimes" is overstated, and the subitizing claim does not survive scrutiny. The abstract and introduction frame the contribution as spanning subitizing, estimation, and density. What the paper actually does is fit subitizing-plus-estimation and predict an analytic threshold n_width = 1/c beyond which the model breaks down. The density regime itself is not modelled; only a boundary condition is offered.*
>
> *The subitizing claim is the more serious problem. The model's "account" of subitizing is that wells at n = 1, 2, 3 are narrow and well-separated, so clean-up converges reliably. This is a description rather than a mechanism. Subitizing has well-documented properties — preattentive processing, attentional-load sensitivity, the sharp transition around n ≈ 4, dependence on configurational and spatial properties — that a 1D attractor with magnitude-dependent Gaussian noise cannot capture even in principle, because the model has no spatial component, no object representation, and no attentional mechanism. The author's own Discussion concedes this: "the model does not fully capture the sharpness of the subitizing–estimation transition… the empirical abruptness may reflect an additional parallel mechanism, suggesting that an Object Tracking System may still be needed." Once that concession is made, the unified-mechanism claim collapses to "a magnitude code for estimation, with subitizing handled by something else we are not modelling," which is the standard ANS-plus-OTS picture rebranded. The contribution should be restated honestly: a model of the estimation regime with a falsifiable density-regime prediction, and an open question about subitizing.*

## Our response

We agree with much of this comment, and the revision restates the contribution — although the trial-level reanalysis (point 1) turned out to license a claim that is different from, and in one respect stronger than, the restatement the reviewer proposes. We take the two halves of the comment in turn.

### 1. The "three regimes" half is resolved: the density claim has been removed entirely

The revised manuscript no longer claims a density regime. The density section and the comparison with Anobile et al. (2014) are gone (the reviewer's minor comment on that comparison being partly numerological contributed to this decision). The revision reports two quantities with different meanings. `n_width` is the continuous width-law proxy defined by σ_n=1; it is not a merger or exact-range boundary. `n_topo` is the last consecutive integer retaining an independent basin. The response-time plateau occurs in a similar numerical region to `n_width`, but this alignment is descriptive. The paper no longer claims anything about density.

### 2. What we concede on subitizing

The reviewer is right that the model has no spatial component, no object representation, and no attentional mechanism. This is true by construction: the model's input is a scalar noisy estimate y₀ ~ N(n, σ(n)), so the visual scene — where the items are, what configuration they form, whether attention was engaged — has been summarised into a single number before the model begins. It follows that the model cannot *predict* configurational effects (canonical arrangements extending the subitizing range, connectedness reducing perceived numerosity) or attentional-load effects, because those manipulations are not expressible in its input format. We do not claim otherwise anywhere in the revision.

### 3. But "cannot capture even in principle" reads the model at the wrong level

The model is a model of the **central number representation and its readout**, not of the visual front end or attention allocation. All four properties the reviewer lists live upstream of that stage, in the evidence channel σ(n) that we deliberately keep neutral (see also our response to point 5). Any manipulation that degrades input quality maps onto larger effective sensory noise. The exposure-duration manipulation shows that the fitted width profile and basin topology can change separately: `n_width` rises from 4.6 to 10, while `n_topo` is computed directly from the potential. Coupling the wells to an image-computable front end (Stoianov & Zorzi, 2012; Testolin et al., 2020) is the stated route to making configurational effects expressible.

We would also note, respectfully, that the reviewer's property list is internally in tension: **preattentive processing** and **attentional-load sensitivity** cannot both be well-documented properties of subitizing, since the second contradicts the first. The classical view held subitizing to be preattentive (Trick & Pylyshyn, 1994), but the dual-task and attentional-manipulation literature has since shown that subitizing depends on attention — if anything more strongly than estimation does (Burr, Turatto & Anobile, 2010; Vetter, Butterworth & Bahrami, 2008; Olivers & Watson, 2008). The property that survives this literature, graded attentional dependence, is precisely the one a noise-channel account accommodates naturally, and it fits a resolvable-well mechanism better than it fits a dedicated preattentive individuation module.

### 4. "A description rather than a mechanism" — the trial-level revision answers this

If the model's account of small-number exactness amounted to "the wells at 1–3 are narrow and well separated, so retrieval is reliable," we would agree that this redescribes the data. The revised, trial-level analysis shows that the well structure does more than redescribe:

- It derives `n_width`, a continuous width-based index that grows with exposure, and `n_topo`, the last consecutive integer retaining an independent basin.
- The empirical RT plateau occurs in a similar numerical region to `n_width`, although the proxy does not define that behavioural boundary.
- The clean-up dynamics are a zero-parameter component that improves the trial-level response fit by ΔAIC ≈ 1256 over the identical static landscape, and the convergence time of those dynamics carries the response-time channel (see point 3 response).
- The **same landscape** reproduces the empirical Weber fraction across the estimation range, so the approximate regime is not a separate stipulation.
- Applied to an independent dataset (developmental dyscalculia, Supplementary Material 5), the fitted difference localises to the width profile rather than sensory noise.

These are quantitative consequences of the well structure that could each have come out otherwise.

### 5. The sharpness of the transition — conceded and now quantified

The reviewer is right that the empirical transition is sharper than the model's. In the revision this is stated quantitatively rather than in passing: the model under-predicts exact accuracy in the subitizing band (at n = 3, predicted 0.77 vs observed 0.82; at n = 4, 0.64 vs 0.71) — it captures the trend toward near-perfect small-number performance but not its full abruptness. This is reported in the Results and returned to in the Discussion, where we keep the possibility open that a fast parallel individuation process contributes for the very smallest sets (Feigenson et al., 2002). The distinction we would draw: what the model claims to explain is *why exactness ends and where the boundary sits*; the residual sharpness at the very bottom of the range is flagged as possibly requiring an additional process, and the manuscript says so.

### 6. The restatement — what the revised paper now claims

The reviewer asks that the contribution be restated as "a model of the estimation regime with a falsifiable density-regime prediction, and an open question about subitizing." We have restated the contribution in two ways. First, the density prediction is gone (point 1 above). Second, the model distinguishes the continuous width-law proxy `n_width` from the topological index `n_topo`. Beyond `n_topo`, at least one consecutive integer lacks an independent basin; the readout then becomes progressively coarser. What remains open is the process-level phenomenology of subitizing: attention, configuration, and the residual sharpness at n=2–4.

### Summary of the position

- The density regime is out of the paper; `n_width` summarizes the fitted width law and `n_topo` identifies the last consecutive independent basin.
- We concede the literal point: no spatial, object, or attentional machinery, by construction; configurational and load effects are not expressible in the model's input format.
- Those properties live upstream, in the deliberately neutral evidence channel; the model accommodates them as input-quality effects (and the load literature — graded shrinkage of the exact range — matches that account), with an image-computable front end as the stated extension.
- The "description not mechanism" charge is answered by the trial-level results: derived and mobile boundary, RT plateau coincidence, equal-parameter clean-up gain, Weber-fraction match, and the dyscalculia localisation.
- The under-predicted sharpness at n = 2–4 is conceded, quantified, and attributed in the Discussion to a possible parallel individuation process for the smallest sets.

---

# Point 8: "The per-duration independent fitting strategy under-tests the model's stability"

Tone: we ran exactly the test the reviewer asks for, and it became one of the strongest results in the revision — it removes the parameter-count objection rather than confirming it, and the requested constrained fit is now a primary analysis rather than an ex-post check.

## The reviewer's comment

> *The per-duration independent fitting strategy under-tests the model's stability. Fitting four parameters per duration yields 20 fitted parameters across the five conditions (plus t₀). The LODO analysis partially addresses this, but the primary analysis should arguably be a constrained model in which the well-scale c (claimed to be the "stable property of the representational system") is shared across durations, with only the sensory-channel parameters varying. The author shows that c is approximately stable across 160–640 ms ex post; this can and should be tested directly by fitting a constrained model and comparing to the unconstrained one.*

## Our response

We agree this is the right test, and we ran exactly it: a constrained model in which the landscape ($\sigma_c$ and the width exponent $\beta$) is shared across all five durations, leaving only the sensory channel and lapse free per duration. It is now a primary analysis, reported in the main text (Fig. 2a) and Supplementary Material 2 (Table S3). Three things came out of it, and together they strengthen the case rather than qualify it.

### 1. The attractor is still the best model under one shared landscape — which removes the parameter-count objection

With a single landscape for all five durations the attractor uses 17 parameters (2 shared landscape parameters, plus the sensory channel and lapse free per duration). The fully free per-duration Weber/ANS code uses 20. Even with this handicap, the shared-landscape attractor beats the free Weber/ANS by **ΔAIC = 905 (ΔBIC = 932)** and the normative log-prior proxy by **4136 (BIC 4163)**. This is the most conservative comparison available: we constrain our model to a single landscape while the benchmark keeps full per-duration freedom, and it still wins under both AIC and BIC. The concern that motivates the point — that the per-duration advantage might be a parameter-count artefact — is therefore answered directly: fewer parameters, still ahead.

### 2. The clean-up dynamics still do work at equal parameters under the constraint

Switching the clean-up off under the same shared landscape is a zero-parameter change, and it costs **ΔAIC = 781**. So the dynamics-vs-static result (points 2 and 3) is not a by-product of per-duration flexibility either: it survives the constrained fit essentially intact.

### 3. The landscape does change with exposure — and that change is the paper's central result, not a defect

Forcing the landscape to be strictly duration-independent costs **ΔAIC = 3529** relative to the separate fits (ΔBIC = 3458), concentrated in the well-scale $\sigma_c$ (sharing only $\sigma_c$ costs 3398; sharing only $\beta$ costs 990). So $\sigma_c$ is not literally constant across durations. The fitted width profile sharpens as sensory evidence improves with viewing time ($\sigma_c$ shrinks), carrying `n_width` from ≈4.6 at a 40 ms flash to ≈10 at 640 ms. The reviewer's observation that $c$ is "approximately stable across 160–640 ms" is consistent with this: most of the movement is at the shortest, noise-dominated exposures.

### What we take from this

The constrained comparison is stronger evidence than the per-duration one, not weaker. It removes the parameter-count objection while leaving the model ranking and the dynamics result intact, and it turns the reviewer's requested test into a load-bearing part of the argument. We now present the comparison at both levels: the joint fit selects the mechanism (which model best explains all the data at once), and the per-duration fit reads how that one mechanism's resolution changes with exposure.

### Summary of the position

- We fit the reviewer's constrained model (landscape shared across durations, sensory channel free per duration): Table S3, main-text Fig. 2a.
- The shared-landscape attractor (17 parameters) beats the fully free per-duration Weber/ANS (20 parameters) by ΔAIC = 905 (ΔBIC = 932), and the clean-up dynamics still help at equal parameters (ΔAIC = 781): the ranking is not a parameter-count artefact.
- Forcing a duration-independent landscape costs ΔAIC = 3529, concentrated in $\sigma_c$: the fitted width profile sharpens with exposure, which increases `n_width`.
