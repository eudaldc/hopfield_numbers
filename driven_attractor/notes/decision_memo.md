# Should the manuscript be rewritten around the driven attractor?

2026-07-10. The decision rules below were written down and committed BEFORE any driven-model comparison number was available (the target values quoted in them come from the reproduced Stage-0 baselines); the results section scores the finished analyses against those rules unchanged.

## Pre-registered decision rules

Classify as **A (rewrite-worthy)** only if most of the following hold:

1. The best driven shared-landscape model (k = 9-10) beats the current shared-landscape model M0-shared (k = 17; logL -114915.2, AIC 229864.4) on participant-held-out lpd (5-fold, participant-blocked, seed 20260711), by more than fold-level uncertainty (participant bootstrap / per-fold sign consistency), and a majority of held-out participants are individually better predicted.
2. It is competitive with the current per-duration model M0-perdur (k = 25; logL -113150.6, AIC 226351.2): held-out lpd within noise or better, despite 15-16 fewer parameters.
3. The improvement survives numerical checks (QMC size, dt, T_CAP sensitivity; contrasts > ~10 logL units).
4. The M2 vs M1 (online vs batch/offset) contrast is positive and > ~10 logL units in-sample AND holds out of sample — otherwise continuous interaction per se is doing no work and any gain belongs to the discrete accumulation front end.
5. M2 vs M3 (attractor force on vs off at equal k) is positive — otherwise the landscape is doing no work in the driven regime.
6. rho, tau_x, beta, sigma_c, p (or their interpreted combinations) recover in simulation.
7. The driven model improves at least one standing qualitative failure: exact-accuracy sharpness at n = 2-4, the long-exposure clean-up profile, or the RT plateau at larger numerosities.

Classify as **B (promising but supplementary)** if the driven model is within ~1 mean-log-score noise of M0-shared out of sample with fewer parameters, or if it cleanly wins the RT channel (physical-seconds link) while tying responses — i.e., better interpretation, not better prediction.

Classify as **C (not supported)** if the driven model loses to M0-shared out of sample, or its gains require duration-specific parameters, or M2 ≈ M1 (no online-interaction effect), or key parameters are unidentifiable, or results flip under numerical settings.

Notes fixed in advance: AIC/BIC are secondary to participant-blocked lpd; "visually appealing trajectories" carry zero weight; if M2 collapses onto M1 (beta -> tether-dominated pinning, x(d) = K(d)), the honest summary is "the data select an accumulate-then-clean-up architecture, not online landscape interaction" and the class is at best B.

## Results against the pre-registered rules

Full numbers: `notes/technical_note.md` and `output/model_comparison_m0_m4.csv`, `output/cv_lpd_summary.csv`.

1. **Held-out prediction vs M0-shared: FAILS, decisively.** Participant-blocked 5-fold CV: best driven variant −118,833 lpd vs M0-shared −114,940 — a deficit of ~3,900 lpd (~0.074 per trial), losing in 5/5 folds; only 65/379 held-out participants are better predicted by the primary driven model (median participant −11 lpd).
2. **Competitive with M0 per-duration: FAILS** (−5,700 lpd held-out; 31/379 participants).
3. **Numerically robust: the NEGATIVE result is robust.** The gap moves < 10% under T_CAP 1-4 s, well support 19→25, dt 5→1.25 ms, QMC 2k→64k; final-score uncertainty ~5 logL vs a gap of ~4,000.
4. **Online vs batch (M2 vs M1): weakly positive** (+171 in-sample, +189 held-out) — real but 4% of the gap to M0, and the fitted mechanism carrying it is a stiff tether (β/τ_x ≈ 5,200/s) that pins x to the count, i.e. not landscape-mediated interaction.
5. **Attractor force on vs off (M2 vs M3): FAILS — negative.** The no-attractor control wins by 52 in-sample and 91 held-out despite one fewer parameter. Inside the driven family the landscape force subtracts predictive value; the saturated-count ceiling analysis shows why (relaxation drags x off the true count).
6. **Recoverability: FAILS for the parameters that matter.** 32-synth recovery with full refits: sigma_c/p/rho and the derived n_width, sigma(5) recover excellently (R^2 0.96-0.998), but tau_x and beta — the constants that embody "continuous driving" — do not recover (identity R^2 < 0; Spearman 0.35-0.47): at the tether-pinned optimum the data select, x(t) = K(t) along the whole beta/tau_x ridge, so the driving dynamics are structurally unidentifiable.
7. **Qualitative improvements: FAILS on all three.** Small-number exactness gets *worse* (needle at r=n for n≤6 at 320-640 ms is lost to the width compromise); the long-exposure profile is the model's *worst* regime (K saturates, data keep sharpening); the RT plateau is not explained — 71% of relaxation paths hit the 2 s cap, E[T] is nearly flat in n, the literal-seconds RT link loses by ~8,000, and the hybrid driven RT still loses to M0's iteration-count hybrid by ~263.

Which assumption failed (deliverable 10): **the evidence-arrival process** — specifically the premise that exposure duration affects only the amount and timing of registered items. Decomposition: the best conceivable saturated-count readout at 640 ms comes within ~200-300 logL of M0 at that duration, but no single (sigma_c, p) landscape can serve all durations once the count has saturated, and ~75% of the deficit is that cross-duration compromise. The data's duration effect is an evidence-QUALITY effect (per-item precision improving with viewing time), which M0 carries in its per-duration sensory channel and a registration-count process cannot carry at any rho, for serial, parallel, interpolated (fitted gamma = 0.46, still −3,447), or pulse delivery, with or without an onset delay. Secondary failures: continuous coupling (the fitted tether is effectively infinite — the data reject landscape-mediated online interaction from within the family) and landscape geometry (the sharp small-n wells that make M0 work are traps a driven state cannot traverse, forcing the width compromise).

The Stage-9 protocols make the epistemic point sharply: the *fitted* driven model predicts NO history dependence (identical output for regular/early/late pulse timing; online ≈ batch except trace effects at 40 ms). The extension's motivating signature is exactly what the data refuse.

## Recommendation

**Abandon the driven-attractor rewrite (class C, not supported).** Do not rewrite the manuscript; do not add the driven model as a supplementary analysis of equal standing. If anything is kept, keep it as a falsification note: the "evidence arrives progressively and cues the memory online" intuition, implemented in its natural form (capped registration process + tethered landscape), is rejected by the trial-level data in favour of the published completed-estimate architecture — by ~4,000 held-out lpd with 7 fewer parameters, with the attractor force itself contributing negatively inside the driven family.

Two by-products are worth keeping regardless:
1. **RT is timed from stimulus offset** in this dataset (established from the trial timestamps; constant 1101 ms inter-trial overhead) — worth stating in the manuscript's methods if RT conventions are ever questioned.
2. **The per-duration landscape is validated out of sample**: M0-per-duration beats M0-shared in all 5 participant-blocked folds (259/379 participants), which strengthens the manuscript's existing position on per-duration fitting against the reviewer's parsimony worry — the extra parameters generalise.

A driven account is not unsalvageable in principle, but the needed ingredient is per-item evidence quality that grows with exposure (e.g. item-precision accrual), which collapses back into M0's sigma(n, d) channel: the data are telling us the published model's abstraction — accumulate first, clean up after — is the right one at this level of description.
