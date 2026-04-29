# Equacions del model d’atractor amb pous de Poisson (fit Piantadosi)

Aquest document descriu la variant del nucli dinàmic on els pous tenen **forma de Poisson escalada**, la **variància creix linealment amb $n$**, la **dinàmica no té terme d’ancoratge** a l’evidència (formalment $\tau_{\text{in}} \to \infty$), i la **RT** es deriva de l’**entropia de la lectura posterior** en comptes del nombre de passos de relaxació.

Existeix una **variant amb ancoratge** que afegeix un pou quadràtic cap a la mostra inicial $y_0$; el que es presenta com a formulació base és el cas **sense** aquest terme.

---

## 1. Suport i notació

- Enters al suport: $n \in \{n_{\min}, \ldots, n_{\max}\}$ amb $n_{\min}=1$ i $n_{\max}=N_{\text{support}} = N_{\text{true max}} + \text{padding}$ (p. ex. suport fins a 18 quan el màxim observable és 15, per donar marge al límit).
- Per cada $n$ hi ha un centre $\mu_n$ i un únic paràmetre d’escala global $\sigma_c$ que defineix l’amplada de tots els pous.
- L’evidència inicial és un escalar $y_0$ (una mostra per assaig). La variable latent es denota $x$; la neteja produeix $x^\star$ després d’iteracions de descens.
- **Paràmetres del sistema** (encaixats o fixats segons la variant): $\sigma_c$ (escala dels pous), $\eta$ (pas del descens), $s_{00}, s_0, \lambda_0$ (canal d’evidència), $t_0$ i $\kappa$ (RT per entropia). Amb ancoratge s’hi afegeix $\tau_{\mathrm{in}}$.

---

## 2. Centres dels pous (recta numèrica lineal)

En la variant amb recta lineal sense compressió, els centres són simplement els enters:

$$
\mu_n = n^{\rho}, \qquad \rho = 1.
$$

És a dir, $\mu_n = n$. Una variant alternativa comprimeix la recta amb $\mu_n = n^\rho$ i $0 < \rho \le 1$ com a paràmetre lliure.

---

## 3. Pou de Poisson escalat

La forma del pou ve de la **log-massa de probabilitat de la llei de Poisson reescalada** amb escala **$\sigma_c > 0$**. Es defineix

$$
x_s = \frac{x}{\sigma_c}, \qquad n_s = \frac{n}{\sigma_c},
$$

i la funció de pou és

$$
f_n(x) = x_s \log(n_s)- n_s- \log \Gamma(x_s + 1),
$$

**Relació exacta amb la llei de Poisson.** La log-massa de la Poisson de paràmetre $\lambda$ avaluada a un enter $k$ és

$$
\log \operatorname{Pois}(k \mid \lambda) = k \log \lambda- \lambda- \log(k!),
$$

i per $k$ real positiu s’estén analíticament via $\log(k!) = \log\Gamma(k+1)$. Substituint $k = x_s$ i $\lambda = n_s$ s’obté **exactament** l’expressió de $f_n(x)$:

$$
f_n(x) = \log \operatorname{Pois}\!\bigl(x_s\bigm|n_s\bigr).
$$

**Per què es cancel·la el terme $-\log\Gamma(x_s+1)$ en la lectura sobre $n$?** Aquest terme depèn només de $x_s = x/\sigma_c$, i $\sigma_c$ és una **escala global comuna a tots els pous**; per tant, fixat un $x$, el valor de $-\log\Gamma(x_s+1)$ és **idèntic per a tot $n$**. Si descomposem

$$
f_n(x) = \underbrace{x_s \log n_s - n_s}_{\tilde f_n(x)\text{ (depèn de }n\text{)}}\underbrace{-\log\Gamma(x_s+1)}_{C(x)\text{ (no depèn de }n\text{)}},
$$

llavors en la _softmax_ sobre $n$ el factor $e^{C(x)}$ apareix tant al numerador com al denominador i es cancel·la:

$$
w_n(x) = \frac{\exp\bigl(f_n(x)\bigr)}{\sum_k \exp\bigl(f_k(x)\bigr)}= \frac{e^{C(x)} \exp\bigl(\tilde f_n(x)\bigr)}{e^{C(x)} \sum_k \exp\bigl(\tilde f_k(x)\bigr)}= \frac{\exp\bigl(\tilde f_n(x)\bigr)}{\sum_k \exp\bigl(\tilde f_k(x)\bigr)}.
$$

Conseqüències:

- La lectura posterior $\mathbb{P}(n\mid x)$ i, per extensió, l’entropia que determina la RT (secció 9) es poden calcular igualment amb $\tilde f_n$ o amb $f_n$.
- El **gradient** $\partial f_n/\partial x$ (secció 7) sí que incorpora la derivada de $-\log\Gamma(x_s+1)$ (funció digamma); aquesta part és **la mateixa per a tot $n$** i surt com a factor comú en la força d’atracció — és per això que al _pull_ apareix $\psi(x_s+1)$ fora de la suma ponderada.

**Propietat clau — variància lineal amb $n$.** Com que la llei de Poisson té mitjana i variància iguals al seu paràmetre, en espai $x$ (des-escalant per $\sigma_c$) es té:

$$
\operatorname{Var}(x \mid n) = \sigma_c\, n, \qquad
\operatorname{std}(x \mid n) = \sqrt{\sigma_c\, n}.
$$

Això encarna el principi de **Weber–Fechner “escalar”** (scalar variability): els nombres més grans tenen pous més amples i per tant es confonen més entre veïns.

---

## 4. Lectura i profunditat dels pous

Sense cap modulació addicional de profunditat per $n$, la forma del pou queda completament determinada per la Poisson. La log-densitat condicional (fins a una constant en $n$) és

$$
\log p(n \mid x) = f_n(x) + \text{const.}
$$

---

## 5. Canal d’evidència (condició inicial)

Per a un enter mostrat $N$:

$$
y_0 \sim \mathcal{N}\bigl(m_0(N), s_0(N)^2\bigr),
$$

$$
m_0(N) = N, \qquad s_0(N) = s_{00} + s_0\, N^{\lambda_0}.
$$

$s_{00}$ pot fixar-se a un valor petit i $s_0, \lambda_0$ encaixar-se com a paràmetres. Una variant amb **soroll constant** imposa $s_0 = 0$ i només $s_{00}$.

---

## 6. Energia i variant amb ancoratge

**Formulació base (sense ancoratge).** Els models finals es formulen sense terme que lligui $x$ a $y_0$ durant la relaxació: formalment $\tau_{\text{in}} \to \infty$, és a dir cap penalització quadràtica $(x-y_0)^2/(2\tau_{\text{in}}^2)$. L’energia és doncs purament el paisatge de pous de Poisson:

$$
E(x) = -\log \sum_{n} \exp\bigl(f_n(x)\bigr).
$$

El punt $y_0$ només serveix com a **condició inicial** del descens; un cop engegada la dinàmica, l’evidència no exerceix cap força de retenció cap a $y_0$.

**Variant amb ancoratge.** Si es vol que la mostra inicial continuï exercint atracció durant la neteja, s’afegeix un pou gaussià sobre $x$ centrat a $y_0$ amb escala $\tau_{\text{in}} > 0$:

$$
E_{\text{anc}}(x) = -\log \sum_{n} \exp\bigl(f_n(x)\bigr) + \frac{(x-y_0)^2}{2\tau_{\text{in}}^2}.
$$

El gradient d’aquest segon terme respecte a $x$ és $(y_0 - x)/\tau_{\text{in}}^2$, és a dir una **força de retenció** cap a $y_0$ que coexisteix amb el _pull_ dels pous de Poisson.

---

## 7. Pesos, _pull_ i descens amb els paràmetres del sistema

### 7.1. Pesos com a competició entre pous

Definim la **_softmax_ sobre pous** (distribució condicional intermèdia sobre el suport, donada la posició $x$):

$$
w_n(x) = \frac{\exp\bigl(f_n(x)\bigr)}{\sum_{k=n_{\min}}^{n_{\max}} \exp\bigl(f_k(x)\bigr)}.
$$

Aquests pesos depenen de **tot** el paisatge: un $x$ donat és “més compatible” amb certs enters $n$ (alts $f_n(x)$) i menys amb d’altres. El _pull_ no “tria” un $n$; **reubica** $x$ de manera que, al llarg dels passos, la combinació dels pous (via $w_n(x)$ canviants) tendeix cap a una posició $x^\star$ on l’energia $E(x) = -\log\sum_n \exp f_n(x)$ és (localment) mínima.

### 7.2. Gradient del pou i _pull_ en funció de $\sigma_c$

Amb $x_s = x/\sigma_c$ i $n_s = n/\sigma_c$, el gradient de $f_n$ respecte a $x$ és ( $\psi = \Gamma'/\Gamma$ digamma):

$$
\frac{\partial f_n}{\partial x}(x) = \frac{1}{\sigma_c}\Bigl(\log n_s - \psi(x_s + 1)\Bigr).
$$

L’energia del paisatge (sense ancoratge) és $E(x) = -\log \sum_n \exp f_n(x)$. La seva derivada és

$$
\frac{\mathrm{d}E}{\mathrm{d}x}(x) = -\sum_n w_n(x)\,\frac{\partial f_n}{\partial x}(x) = \frac{1}{\sigma_c}\left(\psi(x_s+1) - \sum_n w_n(x)\,\log n_s\right).
$$

Definim el **pull** com aquest gradient (la direcció de **més energia** si ens movem en $+x$):

$$
\boxed{\;\mathrm{pull}(x;\sigma_c) = \frac{1}{\sigma_c}\left(\psi\!\left(\frac{x}{\sigma_c}+1\right) - \sum_n w_n(x)\,\log\frac{n}{\sigma_c}\right)\;}
$$

(En el cas amb **sòl de variància**, cada pou pot tenir escala $c_n$; llavors $\log n_s$ i $\psi$ es reemplacen per la versió ponderada amb $c_n$ i un $\sigma_{\mathrm{eff}}$ al denominador; el cas base és $c_n = \sigma_c$ per a tot $n$.)

### 7.3. Pas $\eta$ i actualització de $x$

El descens sobre $E$ amb pas **$\eta$** (al sistema sovint fixat, p. ex. $\eta=1$) és, en una iteració,

$$
\boxed{\;x \leftarrow x - \operatorname{clip}\!\Bigl(\eta\Bigl(\mathrm{pull}(x;\sigma_c) + \frac{x - y_0}{\tau_{\mathrm{in}}^2}\Bigr),\,-\tfrac{\sigma_c}{2},\,\tfrac{\sigma_c}{2}\Bigr)\;}
$$

on $(x-y_0)/\tau_{\mathrm{in}}^2$ és el gradient del pou d’**ancoratge**. En el cas **sense ancoratge** ($\tau_{\mathrm{in}}\to\infty$) aquest terme és zero i només queda $\mathrm{pull}$ dins del parèntesi. El **clip** s’aplica al **pas complet** $\eta(\mathrm{pull}+\cdots)$ i el limita a $\pm\sigma_c/2$, de l’ordre de l’amplada d’un pou per $n$ petit.

**Parada**: quan el pas efectiu és petit (tolerància), o s’detecta oscil·lació, o s’arriba a un màxim de passos. El resultat és $x^\star$.

**Paper del soroll $y_0$:** amb **no ancoratge**, $y_0$ fixa només el **punt de sortida** $x_0 = y_0$. Com que $y_0 \sim \mathcal{N}(N, s_0(N)^2)$ amb $s_0(N) = s_{00} + s_0 N^{\lambda_0}$, assaigs diferents amb el mateix $N$ comencen en llocs diferents del paisatge; el mateix _pull_ des de posicions diferents pot convergir a $x^\star$ diferents, i per tant a posteriors $\mathbb{P}(n\mid x^\star)$ diferents. És aquí on el **soroll d’evidència** ($s_{00}, s_0, \lambda_0$) genera variabilitat entre respostes.

---

## 8. Com es determina el nombre final (latent vs lectura)

El **pull** opera en la variable **contínua** $x$, no retorna un enter. El procés és:

1. **Mostra d’evidència:** es dibuixa $y_0$ donat el nombre mostrat $N$ (mitjana $N$, escala $s_0(N)=s_{00}+s_0N^{\lambda_0}$).
2. **Dinàmica:** es posa $x \leftarrow y_0$ i es iteren els passos de descens; a cada pas es recalculen $w_n(x)$ i $\mathrm{pull}(x;\sigma_c)$ fins a obtenir $x^\star$.
3. **Lectura discreta:** el **nombre reportat** al model ve de la **posterior sobre el suport enter**, no de l’arrodoniment de $x^\star$.

Després de la neteja,

$$
\mathbb{P}(n \mid x^\star)= w_n(x^\star) = \operatorname{softmax}_n\!\bigl(f_n(x^\star)\bigr).
$$

Quantitats que usem al encaix:

- **Mitjana de resposta (nombre esperat):** $\mathbb{E}[n \mid x^\star] = \sum_n n\,\mathbb{P}(n \mid x^\star)$ (resumeix el biaix i la confusió entre veïns).
- **Encert exacte (proporció):** es comparen $N$ amb l’índex del màxim a posteriori, $\hat n = \arg\max_n \mathbb{P}(n \mid x^\star)$ (mapejat a l’enter del suport corresponent).
- **Entropia** (entra a la RT):

$$
H(x^\star) = -\sum_n \mathbb{P}(n \mid x^\star)\log \mathbb{P}(n \mid x^\star).
$$

Així, “en quin nombre cau el _pull_” és una manera imprecisa de dir-ho: el _pull_ **atura** $x^\star$ on els pous estan en equilibri (mínim local d’$E$); **després** la distribució $\mathbb{P}(n\mid x^\star)$ (ampla o estreta) determina tant la mitjana de resposta com l’exactitud i $H(x^\star)$.

---

## 9. RT amb els paràmetres $t_0$ i $c_{\mathrm{ent}}$

En el mode **entropia**, la RT (en les mateixes unitats que $t_0$, p. ex. segons) és

$$
\boxed{\;\mathrm{RT} = t_0 + c_{\mathrm{ent}}\, H(x^\star)\;}
$$

on **$t_0$** és un temps base i **$c_{\mathrm{ent}}$** escala quant pesa la incertesa de la lectura. No s’usa el nombre de passos de neteja: el temps de resposta depèn de **quina distribució** sobre $n$ té el subjecte després de la relaxació, no de quants passos ha calgut per arribar-hi. Si $x^\star$ és en una regió on un sol $n$ domina la posterior, $H$ és petita i la RT tendeix a $t_0$; si diversos $n$ tenen probabilitat apreciable (pous que se solapen, o $x^\star$ entre dues conques), $H$ és gran i la RT creix linealment amb $H$ en aquest proxy.

---

## 10. Encaix amb dades (resum)

L’encaix als resums empírics (mitjana de resposta, precisió exacta, mediana de RT per cada numerositat mostrada) es fa minimitzant una pèrdua que pondera errors quadràtics entre predicció i dades, normalitzats per l’escala de variació observada a cada mètrica.

---

## 11. Paràmetres lliures (formulació Poisson, RT per entropia, sense ancoratge)

En la formulació base (recta lineal, escala $\sigma_c$ global al pou, sense ancoratge, RT lineal en entropia), un conjunt típic de paràmetres a encaixar és:

| Paràmetre            | Rol                                                | Rang típic (ordre de grandària) |
| -------------------- | -------------------------------------------------- | ------------------------------- |
| $\sigma_c$           | escala del pou ($\operatorname{Var}(x\mid n)=\sigma_c n$) | positiu, petit a moderat  |
| $s_{00}$             | soroll mínim del canal d’evidència                 | petit positiu (sovint fixat)    |
| $s_0$                | factor del soroll que escala amb $N$               | positiu                         |
| $\lambda_0$          | exponent de $s_0(N)=s_{00}+s_0N^{\lambda_0}$       | $(0, 3]$ típic                  |
| $\eta$               | pas del descens                                    | sovint fixat (p. ex. $1$)       |
| $t_0$                | RT base                                            | positiu                         |
| $c_{\mathrm{ent}}$   | pes de $H(x^\star)$ a la RT                        | positiu                         |

No cal $\tau_{\mathrm{in}}$ en el cas sense ancoratge. La compressió $\rho$ i d’altres hi entren segons la variant. La variant amb ancoratge afegeix $\tau_{\mathrm{in}}$. La variant amb sòl de variància afegeix $v_{\min}$.
