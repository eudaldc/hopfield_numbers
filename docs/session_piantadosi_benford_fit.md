# Sessió: fit del model Benford-attractor a dades Piantadosi (160ms)

Data: 2026-03-30

## Objectiu

Encaixar les dades de Cheyette & Piantadosi (2020), condició DURATION 160ms, amb un model d'atractors Hopfield on l'amplada dels pous depèn de la freqüència ambiental dels nombres (Benford).

## Evolució del model

El punt de partida era `fit_piantadosi_640.py` amb 13 paràmetres, Box-Cox per als centres, sigma basada en gaps geomètrics, i kappa per pou. L'hem simplificat progressivament:

### 1. Model base: pous equiespaiats + sigma Benford (9 params)
- Centres: mu_n = n (equiespaiats, sense Box-Cox)
- Amplada: sigma_n = c / freq_n, on freq_n = log10(1 + 1/n)
- Energia: E(x|y0) = (x-y0)^2/(2*tau^2) - kappa_lse * logsumexp(-(x-n)^2/(2*sigma_n^2))
- **Resultat**: loss = 0.108. Funciona, però sigma creix massa ràpid amb n.

### 2. Variança (no std) proporcional a 1/freq (9 params)
- sigma_n = sqrt(c / freq_n) — transició més suau
- **Resultat**: loss = 0.040. Millora 2.7x.

### 3. Gaussianes normalitzades (9 params)
- f_n = -log(sigma_n) - (x-n)^2/(2*sigma_n^2) en lloc de f_n = -(x-n)^2/(2*sigma_n^2)
- Resol el problema de la corba energètica inclinada a la dreta per n grans (acumulació asimètrica de gaussianes amples)
- **Resultat**: paisatge energètic simètric. mean_guess a n=15 baixa de 15.5 a 14.2 (subestimació realista).

### 4. RT basat en entropia de la posterior (8 params)
- RT = t0 + kappa * H(posterior) en lloc de t0 + c_steps*n_steps - c_margin*log(margin)
- Elimina 2 params (c_steps, c_margin), afegeix 1 (kappa)
- **Resultat**: loss = 0.011. Canvi de fase emergent al RT entre subitizing i estimació.

### 5. Exponent beta sobre la freqüència (9 params)
- sigma_n = (c / freq_n)^beta — beta controla on cau la transició subitizing/estimació
- **Resultat**: beta = 0.36. Loss similar (~0.011). La transició encara cau a n=2 (volem n=3-5).

### 6. RT basat en cleanup steps (9 params) — **estat actual**
- RT = t0 + c_steps * n_steps
- **Resultat**: loss = 0.055. El nombre de steps no té prou resolució per donar un bon RT (gairebé constant ~6 per n=3-15).

## Estat actual del codi

El fitxer `scripts/fit_piantadosi_640.py` té un flag `RT_MODE` a dalt:
```python
RT_MODE = "steps"    # "steps", "entropy", o "both"
```

Model actual (9 params):
```
sigma_c, beta, kappa_lse, tau_in, eta, s0, lambda0, t0, c_steps/kappa
```

## Qestions obertes i next steps

### RT: steps vs entropia vs ambdós
- **Steps sols**: teòricament elegant (RT = temps de cleanup) però n_steps no té prou resolució
- **Entropia sola**: funciona molt bé numèricament (loss=0.011) però és menys fidel al mecanisme del model
- **Ambdós (RT_MODE="both")**: encara no provat. RT = t0 + c_steps*n_steps + kappa*H. Seria el compromís: el cleanup és el mecanisme principal, l'entropia captura el "dubte" addicional.
- **Alternativa**: millorar la resolució dels n_steps. Ara el cleanup convergeix en 2-8 steps per quasi tots els n perquè el step clamping i l'oscillation detection paren la dinàmica molt aviat. Potser cal ajustar la tolerància o el clamping per donar més rang.

### Transició subitizing/estimació
- Amb beta=0.36, les sigma de n=1,2,3 són més semblants, però la transició encara cau a n=2
- El canal d'evidència (noise = s00 + s0*n^lambda0) contribueix: a n=3 ja hi ha soroll significatiu
- Possible solució: canal amb onset (noise = s00 + s0*max(0, n-n0)^lambda0) perquè n=1-3 tinguin soroll mínim

### Diagnostic útil
El gràfic `diagnostic_steps_entropy.png` mostra n_steps i entropia vs n. Patró actual:
- n=1-7: ~3 steps, entropia creixent suaument
- n=8-15: 6-26 steps, entropia alta i plana

## Paràmetres del millor fit (RT=entropy, beta=0.36)
```
sigma_c  = 0.0176    # escala base d'amplada
beta     = 0.3613    # exponent freq (< 0.5 = transició més tardana)
kappa_lse = 28.15    # força global dels pous (LSE)
tau_in   = 3.76      # ancoratge (>1 = feble)
eta      = 1.31      # step size cleanup
s0       = 0.0807    # soroll evidència
lambda0  = 1.33      # exponent soroll
t0       = 0.879     # base RT
kappa    = 1.42     # escala RT–entropia
```
Loss = 0.011 (3 corbes: mean_guess, pct_exact, median_rt)

## Fitxers rellevants
- `scripts/fit_piantadosi_640.py` — script principal de fit (standalone, no usa DynamicAttractorCore)
- `scripts/figure1_conceptual_attractor_wells.py` — figura conceptual del model (referència visual)
- `output/piantadosi_fit/fit_160ms_benford.png` — últim fit (3 panells: estimació, accuracy, RT)
- `output/piantadosi_fit/fit_160ms_energy.png` — paisatge energètic + perfil sigma
- `output/piantadosi_fit/diagnostic_steps_entropy.png` — n_steps i entropia vs n
- `docs/piantadosi_attractor_model.md` — equacions de la versió ANTIGA (Box-Cox, per-well kappa) — **obsolet, cal actualitzar**
