# Response to Reviewer

We thank the reviewer for a careful and constructive reading of the manuscript. The comments helped us position the work more accurately, temper several claims, and add analyses that make the evidence for the attractor mechanism more transparent. Below we address each point in turn. Reviewer comments are in *italics*; our responses follow, with the corresponding changes in the manuscript indicated in **bold**. New text added to the manuscript is quoted where it is most useful.

---

## Point 1 — Related Hopfield-like work on numerosity

> *The Author properly positions the work with respect to relevant literature, but it should be acknowledged that others have recently tried to model numerosity perception using Hopfield-like approaches: Yago Malo, J., Cicchini, G. M., Morrone, M. C., & Chiofalo, M. L. (2023). Quantum spin models for numerosity perception. PLoS One, 18(4), e0284610.*

We thank the reviewer for pointing us to this work, which is indeed highly relevant and which the author was not familiar with: it shares the statistical-physics roots of the associative-memory framework we build on. **We now cite and discuss Yago Malo et al. (2023) in the Introduction**, in the paragraph reviewing single-framework computational accounts:

> "From a different angle, Yago Malo et al. (2023) recently showed that a small quantum spin model can encode numerosity in the spectrum of its collective dynamics and reproduce Weber-law variability, an approach that shares its statistical-physics roots with the associative-memory networks we build on here."

---

## Point 2 — Modest AIC differences; attractor mechanism vs. Poisson family

> *AIC differences are modest: The Hopfield–Poisson advantage over competitors is relatively small, especially in terms of Global fit. Given the parameter trade-offs, this also makes it unclear how robustly the attractor mechanism — rather than the Poisson family — drives the performance improvement.*

We agree that the in-sample AIC differences are modest and that the original text did not adequately separate the contribution of the *attractor mechanism* from that of the *Poisson family*. We have addressed this in two ways.

**(a) A 2×2 model-comparison design.** We added the Gaussian–Benford attractor variant (Supplementary Material 1) to the comparison, fitted under exactly the same conventions as the main Poisson model. The four accumulator-style models now form a 2×2 design crossing the well family (Poisson vs. Gaussian–Benford) with the presence or absence of the clean-up step, so each ingredient can be assessed independently. **The Results section and Figure 3 (now six models) were rewritten accordingly**, and Supplementary Material 2 reports the full numerical summary. The new analysis shows that *both* ingredients contribute: adding the clean-up step lowers the mean AIC for both well families, and the Poisson parameterization outperforms the Gaussian–Benford one both with and without clean-up. We now state plainly that the differences among the accumulator-style models are modest and that the per-duration winner varies, while the clean-up step adds predictive value on top of *either* static readout — i.e., the advantage is not attributable to the noise family alone.

**(b) The mechanism makes a qualitative, not merely quantitative, contribution.** We added a paragraph noting that the AIC comparison understates an asymmetry between the models: in the attractor variants, response times follow from the mechanism itself (the residual posterior entropy after clean-up maps to commitment time through a single scaling parameter), whereas the static readouts and normative alternatives have no internal dynamics from which a response time could arise and require an added linking assumption.

**We have removed language that overstated the in-sample advantage throughout.**

---

## Point 3 — Parameter identifiability and its consequences for the dyscalculia extension

> *Parameter identifiability: The recovery analysis reveals a strong trade-off between c and κ (ρ = −0.87) and between s₀ and λ₀ (ρ = −0.68). While the Author acknowledges this, it weakens causal interpretation of individual parameters, particularly across the developmental dyscalculia extension. This could be better highlighted.*

We agree, and we have taken this further than highlighting. In the main text (Results), **we now state both trade-offs explicitly and note that, because of them, individual fitted parameters in the dyscalculia extension should not be interpreted in isolation**.

**More importantly, we revised the dyscalculia reanalysis (Supplementary Material 5) so that group comparisons are made on identifiable *derived* quantities rather than on individual parameters**: the breakdown point n_crit = 1/c (which summarizes the attractor landscape on a behavioural scale) and the sensory-noise profile σ(n) = √(s₀·n^λ₀) (which is well constrained even where s₀ and λ₀ are not separately identifiable). Both quantities are evaluated on the bootstrap draws, so the joint parameter uncertainty — including the correlations the reviewer notes — propagates into the comparison. The results are reported in a new table (Table S5).

Crucially, this analysis **reversed a conclusion of the previous version**: once the joint uncertainty is propagated, the attractor-landscape quantity n_crit does *not* separate the groups (median 25.2 for DD vs. 28.2 for controls; difference −3.0, 95% interval [−14.3, 10.4]; P(DD > control) = 0.33). The earlier description of a "less defined attractor landscape" in DD therefore does not survive, and we no longer make that claim. The only consistently directional difference is in the sensory-noise profile, which is higher in the DD group at every numerosity (P(DD > control) between 0.69 and 0.85), although the 95% intervals still include zero. We now describe this as a directional tendency rather than as conclusive evidence.

---

## Point 4 — Why fit parameters separately across exposure durations?

> *It is also not totally clear to me why the model parameters should be fitted separately across exposure durations; is the Author suggesting that the specific mechanism for judging the stimuli differs depending on the timing of the stimulation? This should be discussed more explicitly.*

**We added a paragraph to the Results** making explicit that we do not consider different mechanisms for different durations. Fitting each duration independently is motivated by what the recovered parameters reveal once examined as a function of exposure. Read this way, the parameters show that the same attractor dynamics operate at every duration, and that what changes with exposure is the *quality of the sensory evidence* feeding them: the mechanism-indexing parameters c and κ vary only modestly and monotonically across durations, whereas the sensory-channel parameters s₀ and λ₀ change by orders of magnitude.

**In (Supplementary Material 2), we refitted the model with c and κ shared across durations and only the sensory parameters free per duration**. Sharing the attractor parameters fits substantially worse (across all five durations, ΔAIC = +49.2 and ΔBIC = +21.8. We interpret the smooth, monotonic variation as graded adaptation of a single mechanism to increasingly informative input, not as a switch between mechanisms. We also clarified that all six models in the comparison are fitted under the identical per-duration convention, so the comparison places every model on equal footing.

---

## Point 5 — Lack of a perceptual module and continuous magnitudes

> *As also acknowledged by the Author, the proposed model completely lacks a perceptual module, which would allow simulating key phenomena related to numerosity perception, such as the role of continuous magnitudes. This limitation should be discussed more in depth, also considering that alternative modeling frameworks indeed allow to account for such phenomena (Testolin et al., 2020) and have recently been used also to simulate dyscalculic profiles (Dolfi et al., 2024).*

We have expanded this limitation into a dedicated paragraph in the Discussion. Two points are now made explicitly.

First, the absence of a perceptual front end was a **deliberate scope decision**: numerosity can be recovered from visual input by comparatively simple means — a population of coarsely tuned units suffices to encode number (Dehaene & Changeux, 1993), and number-selective detectors emerge spontaneously in deep networks trained on natural images (Stoianov & Zorzi, 2012) — so we set the perceptual stage aside in order to isolate and characterize the properties of the attractor dynamics themselves.

Second, we **acknowledge the resulting limitation and the frameworks that address it**. By construction the model cannot capture the influence of continuous magnitudes (total area, item size, density, convex hull). We now cite Testolin et al. (2020), whose image-computable networks reproduce these effects, and Dolfi et al. (2024), who use the same framework to simulate dyscalculic profiles through a weakened number sense. We frame these as complementary rather than competing, since they model a different stage of the pipeline (extracting a noisy magnitude estimate from the stimulus) than ours (resolving such an estimate into a discrete percept and a response time), and we identify the natural integration — replacing the parametric noise profile σ(n) with the output statistics of an image-computable front end — as the clearest direction for future work.

Furthermore, we also note that our revised dyscalculia analysis (Point 3) converges with Dolfi et al. (2024) from the opposite side of this pipeline: to the extent the model localizes a group difference at all, it points to a noisier sensory channel rather than to a degraded attractor landscape. This connection is made in Supplementary Material 5.

---

We believe these revisions substantially strengthen the manuscript, and we thank the reviewer again for the very constructive feedback.
