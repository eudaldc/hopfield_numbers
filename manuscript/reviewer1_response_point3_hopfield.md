# Response to Reviewer 1, Point 3 (the "Hopfield framing is decorative")

Draft material for the response letter. Reviewer comment first, our response below.
Tone: acknowledge the one fair part (dimensionality), then establish that the
framing is mechanistically exact and now empirically load-bearing.

---

## The reviewer's comment

> *The Hopfield framing is largely decorative given the actual computations.
> Stripped of the framing, the model is: noisy sensory sample y₀ ~ N(n, s₀n^λ₀);
> gradient descent in 1D on a log-sum-exp of Poisson log-likelihoods; softmax
> readout over integers. In one dimension, gradient descent on a log-sum-exp
> landscape that is itself the log-mixture of unimodal likelihoods is closely
> related to a noisy MAP estimator on the same mixture. The reference to Ramsauer
> et al. (2021) is presentationally apt but mechanistically slight: modern
> Hopfield networks are interesting because they store many high-dimensional
> patterns with exponential capacity; here we have 15 ordered scalars on a line.
> What specifically the attractor dynamics buy beyond a Poisson readout is what
> the no-cleanup baseline is supposed to isolate, and per point 2, the answer
> appears to be "very little."*

---

## Our response

We thank the reviewer for pressing on this: it made us state the mechanism
precisely, and we think the precise statement supports the framing rather than
undermining it. There are three separate claims in the comment, and we address
them in turn.

### 1. The clean-up *is* the retrieval rule of a modern Hopfield network — this is an identity, not an analogy

The reviewer writes that, stripped of the framing, the model is "gradient descent
on a log-sum-exp landscape … closely related to a noisy MAP estimator." We agree
with the description and would add only that this *is* the modern Hopfield
computation, not something distinct from it. The continuous / modern Hopfield
network of Ramsauer et al. (2021) is defined by exactly this energy and this
update:

- Energy: the attraction potential is the log-sum-exp over the stored patterns,
  `V(x) = − log Σ_n exp(g_n(x))`, which is the modern-Hopfield energy with the
  integers as the stored patterns.
- Update: gradient descent on `V` gives
  `∂V/∂x = Σ_n softmax(g_n(x)) · (x − n)/σ_n²`,
  so each step moves the state toward a **softmax-weighted (precision-weighted)
  average of all stored patterns**. For equal widths the fixed point is
  `x* = Σ_n w_n · n` with `w_n = softmax(g_n(x))` — term for term the retrieval
  rule of Ramsauer et al. (their Eq. for the update `ξ_new = X softmax(β Xᵀξ)`).

So "gradient descent on the log-sum-exp of the well likelihoods," "softmax
retrieval in a modern Hopfield network," and "iterated MAP on the mixture" are
three names for the same operation. The reviewer's contrast between a MAP
estimator and a Hopfield network is therefore not a contrast between two
mechanisms; it is one mechanism under two names. The defining feature — that
**every stored pattern contributes to every update through the softmax**, rather
than the state snapping to a nearest neighbour — is present here in full. This is
what the "attractor" language denotes, and it is exact.

We have made this explicit in the Methods (the descent moves the state toward a
softmax combination of all wells, so every stored integer contributes to the
outcome) and in the Discussion (the clean-up is associative retrieval in the
modern Hopfield sense), so a reader sees the identity rather than a loose analogy.

### 2. On dimensionality and capacity — the one place the reviewer is right, and what we actually use

The reviewer is correct that modern Hopfield networks are celebrated in machine
learning for storing *many high-dimensional* patterns with near-exponential
capacity, and that we do not have that: we have on the order of fifteen scalar
patterns on a line. We do **not** claim the capacity result, and we have removed
any wording that could imply we do (the word "Hopfield" is no longer in the
title).

What we use is a different property of the same formalism: associative retrieval
over a pattern set that is **metrically ordered**. Generic modern Hopfield
networks store arbitrary, roughly orthogonal patterns; ours store the integers in
their natural order, with widths that grow with magnitude. This ordering is not
incidental — it is the source of the paper's central result. Because neighbouring
patterns overlap, the softmax competition among them is sharp for small,
well-separated numbers and broad for large, overlapping ones, and it is this
graded degradation of retrieval that produces the resolution limit `n_crit` and
the exact→approximate transition. A high-dimensional, orthogonal-pattern Hopfield
network has no such structure and makes no such prediction. So the associative
retrieval earns its place through the ordered-pattern regime, which is exactly the
regime a magnitude line lives in, rather than through the capacity regime the
reviewer rightly notes we do not occupy.

### 3. "The dynamics buy very little" — this was a consequence of the summary-curve analysis, which we have replaced

The reviewer's final claim, that the dynamics "buy very little," is drawn from
point 2, i.e. from the summary-curve profile-Gaussian AIC (ΔAIC ≈ 1.6 between the
attractor and its no-clean-up control). Following the reviewer's point 1 we have
replaced that analysis with a trial-level likelihood over the ~53,000 individual
integer responses, and the picture is now the opposite:

- Turning the clean-up on is a **zero-parameter change** (the dynamics are
  deterministic given the landscape), so the attractor and its static twin occupy
  the identical parameter space. At the trial level this flips the response fit by
  **ΔAIC ≈ 1256** in favour of the dynamics (and by a similar margin in a second
  well family). This is the cleanest possible isolation of what the dynamics buy,
  and it is not "very little."
- The convergence time of the clean-up also predicts response time better than the
  static entropy proxy that such models usually rely on (Supplementary Material 2
  and the RT panel), so the dynamics carry a second, chronometric signature.

The associative retrieval is therefore not decorative in the operational sense
either: at equal parameter count it does measurable work in both the response
distribution and the response times. (We also verify with BIC, which penalises
parameters far more heavily, that the ranking is unchanged; and a magnitude code
given the same free width parameter still loses.)

### Summary of the position

- The model is, exactly and not by analogy, a modern Hopfield network: the energy
  is the log-sum-exp over the stored integers and the clean-up is the
  softmax-weighted retrieval update.
- We concede the reviewer's dimensionality point and claim none of the
  high-dimensional capacity results; what we use is associative retrieval over an
  **ordered** pattern space, and that ordering is what yields `n_crit`.
- The claim that the dynamics buy little came from the summary-curve analysis the
  reviewer asked us to replace; at the trial level the dynamics improve the fit by
  ΔAIC ≈ 1256 at equal parameters and additionally carry the response times.

We have kept "Hopfield" out of the title and stated the mechanism where it belongs
(Methods and Discussion), so the manuscript makes the mechanistic claim precisely
without leaning on the capacity analogy.
