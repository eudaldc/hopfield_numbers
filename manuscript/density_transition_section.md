# Predicting the transition to density-based perception

The Poisson wells used for cleanup have variance $c\cdot n$ at well $n$, so
the $d'$ between two adjacent attractors is
$d'(n)=1/\sqrt{c\,n}$.
The continuous width-based index defined by $\sigma_n=1$ is
$n_{\text{width}}=1/c$.
It summarizes the fitted width law and is not a discrete merger point.
With the $640$~ms fit ($c=0.0363$) this yields
$n_{\text{width}}\approx 27.5$, and across the five duration conditions the
analytical prediction falls in the narrow band $21\le 1/c\le 28$.
This is a parameter-free prediction of the model: $c$ was fitted against
Piantadosi's exact-enumeration data ($n\le 15$) and then extrapolated
without refitting.

To verify that the analytical threshold is not an artefact of Gaussian
linearisation, we ran $M=10{,}000$ Monte-Carlo trials of the full cleanup
for every integer $n\in[1,50]$ and evaluated four behavioural criteria on
the resulting cleaned posteriors (Fig.~X, Panel A): (i) the $d'<1$ rule,
(ii) a posterior-entropy rule $H[p(\cdot\mid x^*)]>\log 3$, (iii) a
mode-probability rule $p_{\max}<0.5$, and (iv) a Bhattacharyya rule on the
empirical decision distributions.  At $640$~ms the three informative
rules fire at $n=28,\,15,\,16$ respectively, giving a spread of $\times
1.9$ that brackets the analytical $1/c=27.5$.  The entropy and
mode-probability rules detect the transition slightly earlier than $d'$
because they flag the point at which the posterior first supports
multiple integers, whereas $d'=1$ is the chosen width-law reference.
The Bhattacharyya rule with threshold
$0.5$ fires much earlier ($n\approx 4$) and we report this as a
conservatively low-power indicator: under the Gaussian approximation
$\mathrm{BC}(n,n+1)>0.5$ is equivalent to $d'<\sqrt{8\ln 2}\approx 2.36$,
i.e.\ mere partial overlap, so this threshold is a poor transition
marker.  Panel~C shows that the same pattern holds across the five
exposure conditions, and that the $d'$ and $1/c$ rules sit inside the
behavioural range reported by Anobile, Cicchini \& Burr~(2014, $\sim
30\text{--}60$ for $9^\circ$ stimuli at $0.25$ dots/deg$^2$) and just
above Durgin's~(1995, 2008) density breakpoint at $n\approx 20$.

Two caveats are worth spelling out.  First, the transition is smooth:
Panel~D shows cleaned posteriors at $n=5$ (sharp mode), $n=20$
(broadened but unimodal) and $n=40$ (near-uniform over a $\sim 20$-wide
window), so there is no single sharp threshold but a gradual broadening
consistent with the entropy and Bhattacharyya traces.  Second, the
empirical Weber fraction of the model (Panel~B) does \emph{not} collapse
onto the raw-decoder scaling $1/\sqrt{c\,n}$: the cleanup snaps noisy
evidence to the nearest well, producing an approximately constant
$w_{\mathrm{eff}}\approx 0.16$ over $n\in[5,30]$ that rises only modestly
toward the transition.  A constant-Weber model fits the $10\le n\le 40$
range of Reynvoet et al.~(2021) almost two orders of magnitude better
than the $a/\sqrt{n}$ alternative ($\Delta\text{AIC}=-59$), and this is
precisely why the model remains compatible with evidence for
near-classical Weber behaviour in primates and corvids (Nieder and
collaborators): over the numerosity range typically probed in those
studies ($n\lesssim 30$) the cleanup-induced Weber fraction is nearly
flat, and departures from strict constancy are modest enough that they
would not be detected against trial-to-trial variability.  The model
therefore unifies the two empirical traditions: classical Weber's law
in the numerosity regime, and the density-dominated transition at
$n\approx 1/c\approx 20\text{--}30$ reported by Anobile and colleagues
and anticipated by Testolin \& McClelland (2021).
