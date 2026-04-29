# Equacions del model d’atractor dinàmic (fit Piantadosi)

Aquest document descriu el model implementat a `scripts/fit_piantadosi.py` per encaixar les corbes empíriques (p. ex. condició DURATION 160 ms). És una variant del nucli `DynamicAttractorCore` amb **κ i σ per pou** i una dinàmica de neteja lleugerament diferent del mòdul base quan κ és global.

---

## 1. Suport i notació

- Enters al suport: $n \in n_{\min}, \ldots, n_{\max}$ amb $n_{\min}=1$, $n_{\max}=N_{\text{support}}$ (al fit: fins a 18 per deixar marge al límit 15).
- Per cada $n$ hi ha un centre $\mu_n$ i uns paràmetres de pou $\sigma_n > 0$, $\kappa_n > 0$.
- L’evidència inicial és un escalar $y_0$ (una mostra per assaig). La variable latent es denota $x$; la neteja produeix $x^\star$ després d’iteracions de descens.

---

## 2. Centres dels pous (Box–Cox sobre els enters)

Els centres a l’eix latent coincideixen amb el nucli del repositori: es defineixen sobre $z_n = n + \nu$ amb els mateixos $\rho,\nu$ que usa el canal d’evidència.

$$
\mu_n = \operatorname{BoxCox}(z_n, \rho)
= \begin{cases}
\dfrac{z_n^{\rho} - 1}{\rho} & \text{si } |\rho| > \varepsilon [0.5em]
\log z_n & \text{si } |\rho| \le \varepsilon
\end{cases}
\qquad z_n = n + \nu.
$$

---

## 3. Freqüència “Benford estesa” i profunditat $\kappa_n$

Es defineix una freqüència positiva per cada enter del suport (mateixa forma que la llei de Benford per al primer dígit, estesa a tot $n\ge 1$):

$$
f_n = \log_{10}\left(1 + \frac{1}{n}\right), \qquad f_1 = \log_{10}(2).
$$

La **força relativa** del pou $n$ (amplitud dins del `logsumexp`) és:

$$
\kappa_n = \kappa_{\text{scale}} \left(\frac{f_n}{f_1}\right)^{s},
$$

on $s = \texttt{wellshape}$ és un exponent positiu (encaixat per dades). Nombres més freqüents ($n$ petit) tenen $\kappa_n$ més gran.

---

## 4. Amplada $\sigma_n$ (fracció del mig forat entre veïns)

Primer es calculen les distàncies entre centres consecutius $d_k = \mu_{k+1} - \mu_k$. Per al pou amb índex $i$ (associat a l’enter $n_i$) es defineix el **mig forat** local:

- Extrem esquerre ($i=0$):

$$
g_i = \frac{d_0}{2}.
$$

- Extrem dret ($i=K-1$):

$$
g_i = \frac{d_{K-2}}{2}.
$$

- Interiors:

$$
g_i = \frac{1}{2}\min(d_{i-1}, d_i).
$$

Després es fa créixer linealment una **fracció d’emplenament** al llarg del suport. Amb $n_{\text{lo}} = \min n$, $n_{\text{hi}} = \max n$ al suport i

$$
t_n = \frac{n - n_{\text{lo}}}{n_{\text{hi}} - n_{\text{lo}}},
$$

aleshores

$$
\text{fill}_n = a + t_n(b-a), \qquad
a = \min(\texttt{sigmafilllo}, \texttt{sigmafillhi}),\quad
b = \max(\texttt{sigmafilllo}, \texttt{sigmafillhi}).
$$

L’amplada del pou és:

$$
\sigma_n = \text{fill}_n \cdot g_n.
$$

Això garanteix $\sigma_n < g_n$ (en la pràctica $\text{fill}<1$), de manera que es mantenen **mínims locals** al voltant de cada $\mu_n$, amb pou més estret per $n$ petit i més ample (més “pla” localment) per $n$ gran.

---

## 5. Canal d’evidència neutral (condició inicial)

El punt de partida no és el model sota contrast; només condiciona la dinàmica. Per a un enter mostrat $N$:

$$
y_0 \sim \mathcal{N}\bigl(m_0(N), s_0(N)^2\bigr),
$$

$$
m_0(N) = \operatorname{BoxCox}(N + \nu, \rho), \qquad
s_0(N) = s_{00} + s_0\,(N+\nu)^{\lambda_0}.
$$

Al fit, $s_{00}$ és fix (`FIXED["s00"]`); $\rho,\nu,s_0,\lambda_0$ s’encaixen.

---

## 6. Energia (forma equivalent a la dinàmica del script)

La part d’atracció és una barreja log-normal amb pesos $\kappa_n$. És consistent amb fer el gradient del potencial:

$$
E(x \mid y_0) = \frac{(x - y_0)^2}{2\tau_{\text{in}}^2}
-
\log \sum_{n} \exp\left(
\log \kappa_n - \frac{(x - \mu_n)^2}{2\sigma_n^2}
\right).
$$

El primer terme és l’**ancoratge** a l’evidència inicial; el segon, la **superposició de pous** amb amplades i profunditats per pou.

*(Nota: al codi del repositori `attractor_dynamics.py`, amb κ escalar apareix el factor $-\kappa\mathrm{LSE}(\cdot)$; aquí $\kappa_n$ dins de l’exponencial és la generalització que usa el script de fit.)*

---

## 7. Gradient i descens per passos (implementació)

Definim, per cada pas i cada mostra,

$$
f_n(x) = -\frac{(x - \mu_n)^2}{2\sigma_n^2}, \qquad
w_n(x) = \frac{\exp\bigl(\log \kappa_n + f_n(x)\bigr)}
{\sum_k \exp\bigl(\log \kappa_k + f_k(x)\bigr)}
\quad \text{(softmax)}.
$$

El gradient de l’energia respecte $x$ és:

$$
\frac{\partial E}{\partial x}
= \frac{x - y_0}{\tau_{\text{in}}^2}
+
\sum_n w_n(x)\frac{x - \mu_n}{\sigma_n^2}.
$$

**Actualització** (gradient descendent amb pas $\eta$):

$$
\Delta x = -\eta \left( \frac{\partial E}{\partial x} \right),
\qquad
\Delta x \leftarrow \mathrm{clip}\left(\Delta x,-\tfrac{1}{2}\sigma_{\min},\tfrac{1}{2}\sigma_{\min}\right),
$$

on $\sigma_{\min} = \min_n \sigma_n$. Això limita el pas per estabilitat quan els pous són estrets.

**Parada**: es marca convergència si $|\Delta x| < \texttt{tol}$, o bé si el pas canvia de signe respecte l’anterior (detecció d’oscil·lació sota el clip). Es compten els passos fins a convergència o fins a `max_steps`.

---

## 8. Llegida posterior i resposta

Després de la neteja, amb $x = x^\star$,

$$
\mathbb{P}(n \mid x) = w_n(x^\star)
= \mathrm{softmax}_n\bigl(\log \kappa_n + f_n(x^\star)\bigr).
$$

- **Nombre puntual**: $\hat n = \arg\max_n \mathbb{P}(n \mid x)$.
- **Mitjana (nombre esperat)**: $\mathbb{E}[n \mid x] = \sum_n n\mathbb{P}(n \mid x)$.
- **Marge de conca** (diferència entre les dues probabilitats més altes): s’usa al proxy de temps de resposta.

---

## 9. Proxy de temps de resposta (RT)

Per cada mostra (vectoritzat al codi):

$$
\mathrm{RT} = t_0 + c_{\text{steps}} T_{\text{steps}}
-
c_{\text{margin}}\log\bigl(\maxm,10^{-4}\bigr),
$$

on $T_{\text{steps}}$ és el nombre de passos de neteja fins a parada, $m$ és el marge de conca entre les dues classes més probables, i $t_0,c_{\text{steps}},c_{\text{margin}}$ són paràmetres encaixats.

---

## 10. Pèrdua d’encaix (resum)

Es comparen, per cada $N \in 1,\ldots,15$, mitjana de resposta, precisió exacta i mediana de RT del model amb les mètriques empíriques del CSV; la pèrdua és una combinació d’errors quadràtics normalitzats (vegeu `loss_fn` al script).

---

## Referència de codi


| Component                                   | On                                                         |
| ------------------------------------------- | ---------------------------------------------------------- |
| $\mu_n$, $\tau_{\text{in}}$, $\eta$ (mòdul) | `DynamicAttractorCore` → `compute_mu_sigma`, paràmetres    |
| $\sigma_n$, $\kappa_n$                      | `well_sigmas_and_kappas` a `scripts/fit_piantadosi_640.py` |
| Canal $y_0$                                 | `NeutralEvidenceChannel` a `src/data/synth_benchmarks.py`  |
| Bucle de neteja                             | `_cleanup_tracked` a `scripts/fit_piantadosi_640.py`       |


