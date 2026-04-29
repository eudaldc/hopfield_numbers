"""Shared helpers for the Piantadosi attractor model.

Config loading, empirical data extraction, Benford well widths,
well potential computations, cleanup dynamics, simulation, loss,
and goodness-of-fit metrics.
"""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import erf
from scipy.stats import norm
import yaml

OUTPUT_DIR = Path("output/piantadosi_fit")
CONFIG_PATH = Path("configs/config.yaml")

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

N_TRUE_MIN = _cfg["n_true_min"]
N_TRUE_MAX = _cfg["n_true_max"]
N_SUPPORT_MAX = _cfg["n_true_max"] + _cfg["support_padding"]
RT_MODE = _cfg["rt_mode"]
DURATION_MS = _cfg["duration_ms"]
COMPRESS = _cfg["compress_numberline"]
WELL_TYPE = _cfg.get("well_type", "gaussian")  # "gaussian" or "poisson"
VARIANCE_FLOOR = _cfg.get("variance_floor", False)  # adds v_min param
BENFORD_PRIOR = _cfg.get("benford_prior", True)
FIXED_BETA = _cfg.get("fixed_beta")  # None → fitted; float → fixed
FIXED_KAPPA = _cfg.get("fixed_kappa")  # None → fitted; float → fixed
FIXED_ETA = _cfg.get("fixed_eta")  # None → fitted; float → fixed
FIXED_T0 = _cfg.get("fixed_t0")  # None → fitted; float → fixed (base RT, s)
FIXED_S0 = _cfg.get("s_0")  # None → fitted; float → fixed
FIXED_LAMBDA0 = _cfg.get("lambda_0")  # None → fitted; float → fixed
CONSTANT_NOISE = _cfg.get("constant_noise", False)
NO_ANCHOR = _cfg.get("no_anchor", False)  # True → tau_in → ∞ (no input anchoring)

_VALID_DURATIONS = (40, 80, 160, 320, 640)
if DURATION_MS not in _VALID_DURATIONS:
    raise ValueError(
        f"duration_ms={DURATION_MS} not valid, must be one of {_VALID_DURATIONS}"
    )

FIXED = dict(max_steps=100, tol=1e-4, s00=0.05)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_BASE_PARAMS = [
    ("sigma_c", (0.01, 0.50)),  # base well width scale
    *([("v_min", (0.001, 0.5))] if VARIANCE_FLOOR else []),  # variance floor
    *([] if FIXED_BETA else [("beta", (0.1, 1.0))]),
    *([] if FIXED_KAPPA else [("kappa_lse", (0.5, 50.0))]),
    *([] if NO_ANCHOR else [("tau_in", (0.5, 20.0))]),  # anchor width (larger = weaker anchor)
    *([] if FIXED_ETA else [("eta", (0.01, 2.0))]),
    *(
        []
        if CONSTANT_NOISE
        else [
            *([] if FIXED_S0 is not None else [("s0", (0.000001, 0.1))]),
            *([] if FIXED_LAMBDA0 is not None else [("lambda0", (0.1, 4.0))]),
        ]
    ),
    *([] if FIXED_T0 is not None else [("t0", (0.3, 3.0))]),  # base RT (s)
]
_COMPRESS_PARAMS = [("rho", (0.3, 1.0))]  # 1=linear, <1=compressed
_RT_PARAMS = {
    "steps": [("c_steps", (0.001, 0.5))],
    "entropy": [("kappa", (0.1, 3.0))],
    "both": [("c_steps", (0.001, 0.5)), ("kappa", (0.1, 3.0))],
}

_all_params = (
    _BASE_PARAMS + (_COMPRESS_PARAMS if COMPRESS else []) + _RT_PARAMS[RT_MODE]
)
PARAM_NAMES = [p[0] for p in _all_params]
BOUNDS = [p[1] for p in _all_params]

_N_RANGE = N_TRUE_MAX - N_TRUE_MIN + 1
_INTEGERS = torch.arange(
    N_TRUE_MIN,
    N_SUPPORT_MAX + 1,
    dtype=torch.float32,
    device=DEVICE,
)
_FREQ = torch.log10(1.0 + 1.0 / _INTEGERS)
_LOG_FREQ = _FREQ.log() if BENFORD_PRIOR else torch.zeros_like(_INTEGERS)
_N_SHOWN = np.arange(N_TRUE_MIN, N_TRUE_MAX + 1)


def encoding_variance(
    n: float | np.ndarray,
    s0: float,
    lambda0: float,
) -> np.floating | np.ndarray:
    """Numerosity-dependent encoding variance ``v(n) = s0 * n**lambda0``.

    Manuscript-style ``y0 ~ Normal(n, v(n))`` uses the second argument as
    *variance*; NumPy ``normal(loc, scale)`` and SciPy ``norm(..., scale=)``
    expect ``scale`` = standard deviation, i.e. ``sqrt(v(n))``.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    if np.ndim(n_arr) == 0:
        if float(n_arr) <= 0:
            raise ValueError("encoding_variance requires n > 0")
    elif np.any(n_arr <= 0):
        raise ValueError("encoding_variance requires all n > 0")
    return float(s0) * np.power(n_arr, float(lambda0))


def encoding_sd(
    n: float | np.ndarray,
    s0: float,
    lambda0: float,
) -> np.floating | np.ndarray:
    """Standard deviation ``sqrt(s0 * n**lambda0)`` for the power-law noise term."""
    return np.sqrt(encoding_variance(n, s0, lambda0))


def evidence_noise_std(n: float | int, theta: dict) -> float:
    """Standard deviation of continuous evidence ``y0`` given true ``n``.

    Baseline spread ``s00`` and the power-law term combine in *variance* space:
    ``Var(y0 | n) = s00**2 + s0 * n**lambda0``, then ``std = sqrt(Var)``.
    With ``constant_noise`` in config, ``std = s00`` (only the baseline term).
    """
    if CONSTANT_NOISE:
        return float(FIXED["s00"])
    s00 = float(FIXED["s00"])
    v_power = encoding_variance(float(n), float(theta["s0"]), float(theta["lambda0"]))
    v_power = float(np.asarray(v_power, dtype=np.float64).item())
    return float(np.sqrt(s00 * s00 + v_power))


def complete_piantadosi_theta(theta: dict) -> dict:
    """Fill s0, lambda0, t0 when they were omitted from differential evolution."""
    out = dict(theta)
    if CONSTANT_NOISE:
        out["s0"] = 0.0
        out.setdefault("lambda0", 0.0)
    else:
        if FIXED_S0 is not None:
            out["s0"] = float(FIXED_S0)
        if FIXED_LAMBDA0 is not None:
            out["lambda0"] = float(FIXED_LAMBDA0)
    if FIXED_T0 is not None:
        out["t0"] = float(FIXED_T0)
    return out


# ── Empirical targets ───────────────────────────────────────────────────


def extract_empirical(
    duration_ms: int = 160,
    csv_path: str = "data/data_piantadosi.csv",
) -> pd.DataFrame:
    """Per-n_shown summary stats for a given DURATION condition."""
    df = pd.read_csv(csv_path)
    subset = df[(df["cond"] == "DURATION") & (df["exp_dur"] == duration_ms)]

    trial = (
        subset.groupby(["uniqueid", "trial_id"])
        .agg(
            n_shown=("n_shown", "first"),
            n_guess=("n_guess", "first"),
            rt=("rt", "first"),
        )
        .reset_index()
    )
    trial["exact"] = (trial["n_guess"] == trial["n_shown"]).astype(int)

    agg = (
        trial.groupby("n_shown")
        .agg(
            mean_guess=("n_guess", "mean"),
            std_guess=("n_guess", "std"),
            pct_exact=("exact", "mean"),
            median_rt=("rt", "median"),
            count=("n_guess", "count"),
        )
        .reset_index()
    )
    agg["pct_exact"] *= 100
    agg["median_rt_s"] = agg["median_rt"] / 1000.0
    agg = agg[(agg["n_shown"] >= N_TRUE_MIN) & (agg["n_shown"] <= N_TRUE_MAX)]
    return agg.reset_index(drop=True)


def extract_empirical_contrast(
    opacity: float,
    csv_path: str = "data/data_piantadosi.csv",
) -> pd.DataFrame:
    """Per-n_shown summary stats for a given CONTRAST (opacity) condition."""
    df = pd.read_csv(csv_path)
    subset = df[(df["cond"] == "CONTRAST") & (df["opacity"] == opacity)]

    trial = (
        subset.groupby(["uniqueid", "trial_id"])
        .agg(
            n_shown=("n_shown", "first"),
            n_guess=("n_guess", "first"),
            rt=("rt", "first"),
        )
        .reset_index()
    )
    trial["exact"] = (trial["n_guess"] == trial["n_shown"]).astype(int)

    agg = (
        trial.groupby("n_shown")
        .agg(
            mean_guess=("n_guess", "mean"),
            std_guess=("n_guess", "std"),
            pct_exact=("exact", "mean"),
            median_rt=("rt", "median"),
            count=("n_guess", "count"),
        )
        .reset_index()
    )
    agg["pct_exact"] *= 100
    agg["median_rt_s"] = agg["median_rt"] / 1000.0
    agg = agg[(agg["n_shown"] >= N_TRUE_MIN) & (agg["n_shown"] <= N_TRUE_MAX)]
    return agg.reset_index(drop=True)


def _extract_empirical_arrays(empirical: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pre-extract sorted numpy arrays for the hot path (avoids DataFrame overhead)."""
    emp = empirical.sort_values("n_shown").reset_index(drop=True)
    return {
        "n_shown": emp["n_shown"].values,
        "mean_guess": emp["mean_guess"].values,
        "pct_exact": emp["pct_exact"].values,
        "median_rt_s": emp["median_rt_s"].values,
        "model_mask": np.isin(_N_SHOWN, emp["n_shown"].values),
    }


# ── Benford well widths ─────────────────────────────────────────────────


def benford_sigma(integers: torch.Tensor, sigma_c: float, beta: float) -> torch.Tensor:
    """sigma_n = (sigma_c / freq_n)^beta,  where freq_n = log10(1 + 1/n).

    beta controls how strongly frequency shapes well width:
      beta=1   -> sigma proportional to 1/freq (steep transition)
      beta=0.5 -> sqrt (moderate)
      beta<0.5 -> gentler (transition at higher n)
    """
    freq = torch.log10(1.0 + 1.0 / integers.float())
    return (sigma_c / freq).pow(beta)


# ── Well computations ────────────────────────────────────────────────────


def _gaussian_f_n(x_col, mu, sigma_sq, log_sigma):
    """Gaussian well: g_n(x) = -log(sigma_n) - (x-n)^2 / (2 sigma_n^2)."""
    diff = x_col - mu
    return -log_sigma - diff**2 / (2.0 * sigma_sq)


def _gaussian_pull(x_col, mu, sigma_sq, weights, kappa):
    """Gradient of the Gaussian attraction: kappa * sum w_n (x-n)/sigma_n^2."""
    diff = x_col - mu
    return kappa * (weights * diff / sigma_sq).sum(dim=-1)


def _poisson_scale(mu, sigma_c, v_min=0.0):
    """Per-well effective scale for Poisson wells.

    Without floor: variance = c * n, so scale = c for all wells.
    With floor:    variance = v_min + c * n, so scale_n = (v_min + c*n) / n.
    This keeps the Poisson shape but makes small-n wells narrower (dominated
    by v_min) while large-n wells grow as usual.
    """
    if v_min > 0:
        return (v_min + sigma_c * mu) / mu.clamp(min=0.1)
    return sigma_c


def _poisson_f_n(x_col, mu, sigma_c, v_min=0.0):
    """Scaled Poisson well: evaluate Poisson(n/c_n) at x/c_n.

    With variance floor: c_n = (v_min + c*n)/n, giving effective
    variance v_min + c*n in x-space.
    """
    c_n = _poisson_scale(mu, sigma_c, v_min)
    x_s = (x_col / c_n).clamp(min=0.1)
    n_s = (mu / c_n).clamp(min=0.1)
    return x_s * torch.log(n_s) - n_s - torch.lgamma(x_s + 1)


def _poisson_pull(x_col, mu, weights, sigma_c, v_min=0.0):
    """Gradient of scaled Poisson attraction."""
    c_n = _poisson_scale(mu, sigma_c, v_min)
    x_s = (x_col / c_n).clamp(min=0.1)
    n_s = (mu / c_n).clamp(min=0.1)
    weighted_log_n = (weights * torch.log(n_s)).sum(dim=-1)
    # c_n is per-well [K], but we need a scalar for the gradient denominator;
    # use the weighted average scale
    c_eff = (weights * c_n).sum(dim=-1)
    return (torch.digamma(x_s[:, 0] + 1) - weighted_log_n) / c_eff


# ── Cleanup dynamics ────────────────────────────────────────────────────


@torch.no_grad()
def cleanup(
    y0: torch.Tensor,
    mu: torch.Tensor,
    log_freq: torch.Tensor,
    kappa: float,
    inv_tau_sq: float,
    eta: float,
    max_steps: int,
    tol: float,
    sigma_c: float = 1.0,
    v_min: float = 0.0,
    # Gaussian-specific (ignored for Poisson)
    sigma_sq: torch.Tensor | None = None,
    log_sigma: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Gradient descent on E(x|y0) starting at x = y0.

    The Benford prior (log_freq) is added only to the posterior readout,
    not to the dynamics.
    """
    is_poisson = WELL_TYPE == "poisson"
    B = y0.shape[0]
    dev = y0.device
    x = y0.clone()

    if is_poisson:
        max_step = 0.5 * sigma_c
    else:
        max_step = 0.5 * sigma_sq.sqrt().min().item()

    converged = torch.zeros(B, dtype=torch.bool, device=dev)
    n_steps = torch.full((B,), max_steps, dtype=torch.long, device=dev)
    prev_step = torch.zeros(B, device=dev)

    for t in range(max_steps):
        x_col = x.unsqueeze(-1)  # [B, 1]
        if is_poisson:
            f_n = _poisson_f_n(x_col, mu, sigma_c, v_min)  # [B, K]
            weights = F.softmax(f_n, dim=-1)
            pull = _poisson_pull(x_col, mu, weights, sigma_c, v_min)
        else:
            f_n = _gaussian_f_n(x_col, mu, sigma_sq, log_sigma)  # [B, K]
            weights = F.softmax(f_n, dim=-1)
            pull = _gaussian_pull(x_col, mu, sigma_sq, weights, kappa)

        anchor = (x - y0) * inv_tau_sq  # [B]
        step = (eta * (anchor + pull)).clamp(-max_step, max_step)

        oscillating = (~converged) & (t > 0) & (step * prev_step < -1e-12)
        just_converged = oscillating | ((~converged) & (step.abs() < tol))
        n_steps = torch.where(just_converged, t + 1, n_steps)
        converged = converged | just_converged

        x_new = x - step
        x = torch.where(converged, x, x_new)
        prev_step = torch.where(converged, prev_step, step)

        if (t + 1) % 4 == 0 and converged.all():
            break

    # Posterior p(n | x*), optionally with Benford prior
    x_col = x.unsqueeze(-1)
    if is_poisson:
        f_n = _poisson_f_n(x_col, mu, sigma_c, v_min)
    else:
        f_n = _gaussian_f_n(x_col, mu, sigma_sq, log_sigma)
    f_n = log_freq + f_n
    probs = F.softmax(f_n, dim=-1)
    n_hat_idx = probs.argmax(dim=-1)
    top2 = probs.topk(2, dim=-1).values

    return {
        "x_star": x,
        "n_steps": n_steps,
        "probs": probs,
        "n_hat_idx": n_hat_idx,
        "basin_margin": top2[:, 0] - top2[:, 1],
    }


# ── Simulation ──────────────────────────────────────────────────────────


@torch.no_grad()
def _simulate_core(
    theta: dict,
    n_samples_per_n: int = 5000,
    seed: int = 73,
    *,
    include_trials: bool = False,
) -> dict[str, np.ndarray]:
    """Batched attractor cleanup: all n-values in a single call.

    If ``include_trials`` is True, ``trial_exact`` and ``trial_rt`` are added
    with shape ``(N_TRUE_MAX - N_TRUE_MIN + 1, n_samples_per_n)``.
    """
    rho = theta.get("rho", 1.0)
    mu = _INTEGERS.pow(rho)
    kappa = theta.get("kappa_lse", FIXED_KAPPA or 1.0)
    inv_tau_sq = 0.0 if NO_ANCHOR else 1.0 / theta["tau_in"] ** 2
    eta = theta.get("eta", FIXED_ETA or 1.0)

    sigma_sq, log_sigma = None, None
    if WELL_TYPE == "gaussian":
        sigma = benford_sigma(
            _INTEGERS,
            theta["sigma_c"],
            theta.get("beta", FIXED_BETA or 1.0),
        )
        sigma_sq = sigma.square()
        log_sigma = sigma.log()

    g = torch.Generator().manual_seed(seed)
    y0_parts = []
    for n in range(N_TRUE_MIN, N_TRUE_MAX + 1):
        idx = n - N_TRUE_MIN
        noise_std = evidence_noise_std(n, theta)
        y0_n = mu[idx].cpu().item() + noise_std * torch.randn(
            n_samples_per_n,
            generator=g,
        )
        y0_parts.append(y0_n)
    y0_all = torch.cat(y0_parts).to(DEVICE)

    out = cleanup(
        y0_all,
        mu,
        _LOG_FREQ,
        kappa,
        inv_tau_sq,
        eta,
        FIXED["max_steps"],
        FIXED["tol"],
        sigma_c=theta["sigma_c"],
        v_min=theta.get("v_min", 0.0),
        sigma_sq=sigma_sq,
        log_sigma=log_sigma,
    )

    probs = out["probs"].view(_N_RANGE, n_samples_per_n, -1)
    n_hat_idx = out["n_hat_idx"].view(_N_RANGE, n_samples_per_n)
    n_steps = out["n_steps"].float().view(_N_RANGE, n_samples_per_n)

    exp_n = (probs * _INTEGERS).sum(dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)

    correct_idx = torch.arange(_N_RANGE, device=DEVICE).unsqueeze(-1)

    rt = torch.full_like(n_steps, theta["t0"])
    if RT_MODE in ("steps", "both"):
        rt = rt + theta["c_steps"] * n_steps
    if RT_MODE in ("entropy", "both"):
        rt = rt + theta["kappa"] * entropy

    exact = (n_hat_idx == correct_idx).float()

    out = {
        "n_shown": _N_SHOWN,
        "mean_guess": exp_n.mean(dim=1).cpu().numpy(),
        "std_guess": exp_n.std(dim=1).cpu().numpy(),
        "pct_exact": (exact.mean(dim=1).cpu().numpy() * 100),
        "std_exact": (exact.std(dim=1).cpu().numpy() * 100),
        "median_rt": rt.median(dim=1).values.cpu().numpy(),
        "std_rt": rt.std(dim=1).cpu().numpy(),
        "mean_n_steps": n_steps.mean(dim=1).cpu().numpy(),
        "mean_entropy": entropy.mean(dim=1).cpu().numpy(),
    }
    if include_trials:
        out["trial_exact"] = exact.cpu().numpy()
        out["trial_rt"] = rt.cpu().numpy()
    return out


@torch.no_grad()
def simulate(
    theta: dict,
    n_samples_per_n: int = 5000,
    seed: int = 73,
) -> pd.DataFrame:
    """Run attractor cleanup for each n in [N_TRUE_MIN, N_TRUE_MAX]."""
    return pd.DataFrame(_simulate_core(theta, n_samples_per_n, seed))


# ── Loss function ────────────────────────────────────────────────────────


def _single_loss(
    theta_vec: np.ndarray,
    emp_arrays: dict[str, np.ndarray],
    n_samples: int,
    seed: int,
) -> float:
    """Compute loss for one parameter vector (hot path, no DataFrames)."""
    theta = complete_piantadosi_theta(dict(zip(PARAM_NAMES, theta_vec)))

    try:
        model = _simulate_core(theta, n_samples_per_n=n_samples, seed=seed)
    except Exception:
        return 1e6

    mask = emp_arrays["model_mask"]
    mg_e, mg_m = emp_arrays["mean_guess"], model["mean_guess"][mask]
    loss_guess = np.mean((mg_e - mg_m) ** 2) / (np.ptp(mg_e) ** 2 + 1e-6)

    ac_e, ac_m = emp_arrays["pct_exact"], model["pct_exact"][mask]
    loss_acc = np.mean((ac_e - ac_m) ** 2) / (np.ptp(ac_e) ** 2 + 1e-6)

    rt_e, rt_m = emp_arrays["median_rt_s"], model["median_rt"][mask]
    loss_rt = np.mean((rt_e - rt_m) ** 2) / (np.ptp(rt_e) ** 2 + 1e-6)

    total = float(loss_guess + loss_acc + loss_rt)
    return 1e6 if (np.isnan(total) or np.isinf(total)) else total


def loss_fn(
    theta_input: np.ndarray,
    emp_arrays: dict[str, np.ndarray],
    sample_state: list[int],
    seed: int = 73,
) -> float | np.ndarray:
    """Loss compatible with both vectorized DE (2-D input) and scalar polish.

    scipy vectorized=True passes shape (n_params, pop_size) and expects (pop_size,).
    """
    n_samples = sample_state[0]
    if theta_input.ndim == 1:
        return _single_loss(theta_input, emp_arrays, n_samples, seed)
    pop_size = theta_input.shape[1]
    losses = np.empty(pop_size)
    for i in range(pop_size):
        losses[i] = _single_loss(theta_input[:, i], emp_arrays, n_samples, seed)
    return losses


# ── Goodness-of-fit metrics ─────────────────────────────────────────────

# (empirical_column, model_column) per supported channel
_CHANNEL_SPEC: dict[str, tuple[str, str]] = {
    "mean_guess": ("mean_guess", "mean_guess"),
    "pct_exact": ("pct_exact", "pct_exact"),
    "rt": ("median_rt_s", "median_rt"),
}
DEFAULT_CHANNELS: tuple[str, ...] = ("mean_guess", "pct_exact", "rt")


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _align_on_n_shown(
    empirical: pd.DataFrame, model: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ns = np.intersect1d(empirical["n_shown"].values, model["n_shown"].values)
    emp = (
        empirical[empirical["n_shown"].isin(ns)]
        .sort_values("n_shown")
        .reset_index(drop=True)
    )
    mod = (
        model[model["n_shown"].isin(ns)]
        .sort_values("n_shown")
        .reset_index(drop=True)
    )
    return emp, mod


def compute_goodness(
    empirical: pd.DataFrame,
    model: pd.DataFrame,
    k: int | None = None,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> dict:
    """Per-channel R^2/RMSE/MAE, plus logL/AIC/BIC when ``k`` is given.

    The likelihood is a profile Gaussian: one nuisance variance per channel
    estimated as sigma^2_c = RSS_c / n_c (the MLE). Channels with fully-NaN
    empirical targets are silently skipped (DeCarli-style inputs lack
    ``mean_guess``). ``k`` is the number of free theta parameters.
    """
    emp, mod = _align_on_n_shown(empirical, model)

    metrics: dict = {}
    nll_total = 0.0
    n_total = 0
    used_channels: list[str] = []

    for ch in channels:
        if ch not in _CHANNEL_SPEC:
            raise ValueError(f"Unknown channel '{ch}'")
        emp_col, mod_col = _CHANNEL_SPEC[ch]
        if emp_col not in emp.columns:
            continue
        y_e = np.asarray(emp[emp_col].values, dtype=float)
        if y_e.size == 0 or np.all(np.isnan(y_e)):
            continue
        y_m = np.asarray(mod[mod_col].values, dtype=float)
        n_c = int(y_e.size)
        rss = float(np.sum((y_e - y_m) ** 2))
        sigma2 = rss / n_c if n_c > 0 else np.nan

        metrics[f"R2_{ch}"] = _r_squared(y_e, y_m)
        metrics[f"RMSE_{ch}"] = float(np.sqrt(rss / n_c)) if n_c > 0 else np.nan
        metrics[f"MAE_{ch}"] = float(np.mean(np.abs(y_e - y_m)))

        if sigma2 > 0 and np.isfinite(sigma2):
            nll_c = 0.5 * n_c * (np.log(2.0 * np.pi) + np.log(sigma2) + 1.0)
            nll_total += nll_c
            n_total += n_c
            metrics[f"sigma2_{ch}"] = float(sigma2)
            used_channels.append(ch)

    if k is not None and n_total > 0 and np.isfinite(nll_total):
        logL = -float(nll_total)
        metrics["logL"] = logL
        metrics["AIC"] = float(2 * k - 2 * logL)
        metrics["BIC"] = float(k * np.log(n_total) - 2 * logL)
        metrics["n_data"] = int(n_total)
        metrics["k_params"] = int(k)
        metrics["channels_used"] = ",".join(used_channels)

    return metrics


def compute_nll(
    theta: dict,
    empirical: pd.DataFrame,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    n_samples: int = 5000,
    seed: int = 73,
) -> float:
    """Profile Gaussian NLL summed across active channels.

    Intended as the objective for Hessian-based CIs. A fixed ``seed`` keeps the
    Monte-Carlo noise deterministic so finite-difference Hessians are not
    dominated by stochasticity.
    """
    try:
        sim = _simulate_core(theta, n_samples_per_n=n_samples, seed=seed)
    except Exception:
        return 1e12
    model = pd.DataFrame(sim)
    emp, mod = _align_on_n_shown(empirical, model)

    nll = 0.0
    for ch in channels:
        emp_col, mod_col = _CHANNEL_SPEC[ch]
        if emp_col not in emp.columns:
            continue
        y_e = np.asarray(emp[emp_col].values, dtype=float)
        if y_e.size == 0 or np.all(np.isnan(y_e)):
            continue
        y_m = np.asarray(mod[mod_col].values, dtype=float)
        n_c = int(y_e.size)
        rss = float(np.sum((y_e - y_m) ** 2))
        sigma2 = rss / n_c if n_c > 0 else np.nan
        if not (sigma2 > 0 and np.isfinite(sigma2)):
            return 1e12
        nll += 0.5 * n_c * (np.log(2.0 * np.pi) + np.log(sigma2) + 1.0)
    return float(nll)


# ── Confidence intervals ────────────────────────────────────────────────


def _num_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    step: np.ndarray,
) -> np.ndarray:
    """Central-difference Hessian. Assumes per-coordinate step sizes in ``step``."""
    n = x.size
    H = np.zeros((n, n))
    f0 = float(f(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        fpp = float(f(x + ei))
        fmm = float(f(x - ei))
        H[i, i] = (fpp - 2.0 * f0 + fmm) / (step[i] ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = step[i]
            ej[j] = step[j]
            f_pp = float(f(x + ei + ej))
            f_pm = float(f(x + ei - ej))
            f_mp = float(f(x - ei + ej))
            f_mm = float(f(x - ei - ej))
            H[i, j] = H[j, i] = (f_pp - f_pm - f_mp + f_mm) / (
                4.0 * step[i] * step[j]
            )
    return H


def hessian_ci(
    nll_fn: Callable[[np.ndarray], float],
    theta_vec: np.ndarray,
    param_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    alpha: float = 0.05,
    step_frac: float = 5e-2,
    min_step: float = 1e-4,
    bound_tol: float = 1e-3,
    log_space: bool = True,
) -> pd.DataFrame:
    """Laplace-approximation confidence intervals at a fitted theta.

    When ``log_space=True`` (default), each parameter whose lower bound is
    strictly positive is differenced on the log-scale and mapped back via the
    delta method (``SE(theta) = theta * SE(log theta)``). This both matches the
    natural scale of multiplicative parameters (``tau_in``, ``sigma_c``, ...)
    and yields asymmetric, strictly-positive CIs via
    ``[theta * exp(-z * SE_log), theta * exp(+z * SE_log)]``.

    Robustness:
      - Step sizes scale with each parameter's feasible range but are reduced
        near a bound to stay inside it.
      - If the Hessian is not positive-definite (common with stochastic MC
        objectives), falls back to diagonal-only SEs (conservative, ignores
        correlations). If even the diagonal curvature is ~0, SE is NaN.
      - ``status`` column: 'ok', 'at_bound', 'non_pd', or 'singular'.
      - ``method`` column: 'hessian_log' or 'hessian'.
    """
    theta_vec = np.asarray(theta_vec, dtype=float)
    n = theta_vec.size
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    use_log = log_space & (lo > 0) & (theta_vec > 0)
    phi = np.where(use_log, np.log(theta_vec), theta_vec)
    phi_lo = np.where(use_log, np.log(np.maximum(lo, 1e-30)), lo)
    phi_hi = np.where(use_log, np.log(np.maximum(hi, 1e-30)), hi)
    phi_width = phi_hi - phi_lo

    step = np.maximum(step_frac * phi_width, min_step)
    room = 0.5 * np.minimum(phi - phi_lo, phi_hi - phi)
    room = np.where(room > 0, room, min_step)
    step = np.minimum(step, room)
    step = np.maximum(step, 1e-7)

    width = hi - lo
    at_bound = (theta_vec - lo < bound_tol * width) | (
        hi - theta_vec < bound_tol * width
    )

    def nll_phi(phi_vec: np.ndarray) -> float:
        theta = np.where(use_log, np.exp(phi_vec), phi_vec)
        return float(nll_fn(theta))

    H = _num_hessian(nll_phi, phi, step)

    cov_status = "ok"
    se_phi = np.full(n, np.nan)
    try:
        eigvals = np.linalg.eigvalsh(H)
        pd_ok = bool(np.all(eigvals > 1e-8))
    except np.linalg.LinAlgError:
        pd_ok = False

    if pd_ok:
        try:
            cov = np.linalg.inv(H)
            diag = np.diag(cov)
            se_phi = np.where(diag > 0, np.sqrt(diag), np.nan)
        except np.linalg.LinAlgError:
            cov_status = "singular"
    else:
        cov_status = "non_pd"
        diag_H = np.diag(H)
        mask = diag_H > 1e-8
        se_phi[mask] = 1.0 / np.sqrt(diag_H[mask])

    z = float(norm.ppf(1.0 - alpha / 2.0))

    se_nat = np.where(use_log, theta_vec * se_phi, se_phi)
    ci_lo_raw = np.where(
        use_log,
        theta_vec * np.exp(-z * se_phi),
        theta_vec - z * se_phi,
    )
    ci_hi_raw = np.where(
        use_log,
        theta_vec * np.exp(+z * se_phi),
        theta_vec + z * se_phi,
    )
    ci_lo = np.clip(ci_lo_raw, lo, hi)
    ci_hi = np.clip(ci_hi_raw, lo, hi)

    statuses = [
        "at_bound" if at_bound[i] else cov_status for i in range(n)
    ]
    methods = ["hessian_log" if use_log[i] else "hessian" for i in range(n)]

    return pd.DataFrame(
        {
            "param": list(param_names),
            "value": theta_vec,
            "se": se_nat,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "status": statuses,
            "method": methods,
        }
    )


def profile_likelihood_ci(
    nll_of_theta: Callable[[dict], float],
    fit_with_fixed: Callable[[dict[str, float], int], dict],
    theta_hat: dict,
    nll_hat: float,
    param: str,
    bounds_param: tuple[float, float],
    alpha: float = 0.05,
    n_grid: int = 11,
    grid_halfwidth_log: float = 1.0,
    log_scale: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[float, float, pd.DataFrame]:
    """Profile-likelihood CI for a single parameter.

    For each grid value ``g``, fixes ``param = g``, refits the rest via
    ``fit_with_fixed({param: g}, seed)``, and evaluates ``nll_of_theta`` at
    the result. The CI is the interval of grid values where

        2 * (NLL(g, theta_{-param}^*(g)) - NLL_hat) <= chi2(1, alpha)

    i.e. the 95% CI uses threshold ~ 1.92.

    ``grid_halfwidth_log`` is the half-width on the log scale (so ``1.0``
    means the grid spans roughly a factor of e below and above the fitted
    value, clipped to bounds).
    """
    from scipy.stats import chi2

    threshold = float(chi2.ppf(1.0 - alpha, df=1) / 2.0)
    lo, hi = bounds_param
    val = float(theta_hat[param])

    if log_scale and lo > 0 and val > 0:
        log_val = np.log(val)
        lo_g = max(np.log(lo), log_val - grid_halfwidth_log)
        hi_g = min(np.log(hi), log_val + grid_halfwidth_log)
        grid = np.exp(np.linspace(lo_g, hi_g, n_grid))
    else:
        hw = grid_halfwidth_log * (hi - lo) * 0.5
        grid = np.linspace(max(lo, val - hw), min(hi, val + hw), n_grid)

    grid = np.unique(np.sort(np.concatenate([grid, [val]])))

    rows = []
    for i, g in enumerate(grid):
        if abs(g - val) < 1e-12:
            nll = nll_hat
        else:
            try:
                theta_g = fit_with_fixed({param: float(g)}, seed + i)
                nll = float(nll_of_theta(theta_g))
            except Exception as exc:
                if verbose:
                    print(f"  profile {param}={g:.4f} failed: {exc}")
                nll = float("inf")
        rows.append({"value": float(g), "nll": nll, "dnll": nll - nll_hat})

    prof = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)

    idx_hat = int(np.argmin(np.abs(prof["value"].values - val)))
    below = prof["dnll"].values <= threshold

    i_lo = idx_hat
    while i_lo > 0 and below[i_lo - 1]:
        i_lo -= 1
    i_hi = idx_hat
    while i_hi < len(prof) - 1 and below[i_hi + 1]:
        i_hi += 1

    def _interp(i_inside: int, i_outside: int) -> float:
        x_in = float(prof["value"].iloc[i_inside])
        x_out = float(prof["value"].iloc[i_outside])
        y_in = float(prof["dnll"].iloc[i_inside])
        y_out = float(prof["dnll"].iloc[i_outside])
        if not np.isfinite(y_out) or y_out == y_in:
            return x_in
        t = (threshold - y_in) / (y_out - y_in)
        t = float(np.clip(t, 0.0, 1.0))
        return x_in + t * (x_out - x_in)

    ci_lo = (
        float(prof["value"].iloc[i_lo])
        if i_lo == 0
        else _interp(i_lo, i_lo - 1)
    )
    ci_hi = (
        float(prof["value"].iloc[i_hi])
        if i_hi == len(prof) - 1
        else _interp(i_hi, i_hi + 1)
    )

    return float(ci_lo), float(ci_hi), prof


def refine_cis_with_profile(
    ci_df: pd.DataFrame,
    nll_of_theta: Callable[[dict], float],
    fit_with_fixed: Callable[[dict[str, float], int], dict],
    theta_hat: dict,
    nll_hat: float,
    bounds: Sequence[tuple[float, float]],
    alpha: float = 0.05,
    n_grid: int = 11,
    grid_halfwidth_log: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Replace Hessian CIs with profile-likelihood ones for problematic rows.

    A row is refined when its ``se`` is NaN or its ``status`` is 'at_bound'
    or 'non_pd'/'singular'. The fitted value is left unchanged; ``ci_lo``,
    ``ci_hi``, ``status``, and ``method`` are overwritten.
    """
    out = ci_df.copy()
    if "method" not in out.columns:
        out["method"] = "hessian"

    bounds_by_param = dict(zip(out["param"].tolist(), bounds))

    iterator = out.iterrows()
    if verbose:
        try:
            from tqdm import tqdm as _tqdm

            iterator = _tqdm(
                list(iterator), desc="profile-CI", unit="param"
            )
        except ImportError:
            pass

    for idx, row in iterator:
        se = row["se"]
        status = row["status"]
        needs_refine = (
            (not np.isfinite(se))
            or status in ("non_pd", "singular", "at_bound")
        )
        if not needs_refine:
            continue

        param = row["param"]
        if param not in bounds_by_param:
            continue
        if verbose:
            print(f"  refining {param} ({status}, SE={se}) via profile likelihood")

        ci_lo, ci_hi, _prof = profile_likelihood_ci(
            nll_of_theta=nll_of_theta,
            fit_with_fixed=fit_with_fixed,
            theta_hat=theta_hat,
            nll_hat=nll_hat,
            param=param,
            bounds_param=bounds_by_param[param],
            alpha=alpha,
            n_grid=n_grid,
            grid_halfwidth_log=grid_halfwidth_log,
            seed=seed + hash(param) % 1000,
            verbose=verbose,
        )

        out.at[idx, "ci_lo"] = ci_lo
        out.at[idx, "ci_hi"] = ci_hi
        hw_nat = max(ci_hi - row["value"], row["value"] - ci_lo)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        out.at[idx, "se"] = hw_nat / z if z > 0 else np.nan
        out.at[idx, "method"] = "profile"
        if status == "at_bound":
            out.at[idx, "status"] = "at_bound_profile"
        else:
            out.at[idx, "status"] = "profile_ok"

    return out


def _bootstrap_percentiles(
    samples: Sequence[dict],
    param_names: Sequence[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-parameter percentile summary of bootstrap samples."""
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    rows = []
    for p in param_names:
        vals = np.array([s[p] for s in samples if p in s], dtype=float)
        rows.append(
            {
                "param": p,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "ci_lo": float(np.quantile(vals, lo_q)),
                "ci_hi": float(np.quantile(vals, hi_q)),
                "n_boot": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    resample_fn: Callable[[int], pd.DataFrame],
    fit_fn: Callable[[pd.DataFrame, int], dict],
    param_names: Sequence[str],
    n_boot: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Serial non-parametric bootstrap CIs via repeated resample+refit.

    - ``resample_fn(seed)`` should return a fresh empirical DataFrame (e.g. by
      resampling trials or subjects with replacement).
    - ``fit_fn(empirical, seed)`` should refit and return a theta dict.
    - For parallel execution see ``parallel_bootstrap_ci`` below.

    Returns DataFrame with columns: param, mean, median, ci_lo, ci_hi, n_boot.
    """
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    iterator = range(n_boot)
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="bootstrap", unit="fit")
        except ImportError:
            pass
    for b in iterator:
        bseed = int(rng.integers(0, 2**31 - 1))
        try:
            emp_b = resample_fn(bseed)
            theta_b = fit_fn(emp_b, bseed)
        except Exception as exc:
            if verbose:
                print(f"  bootstrap iter {b} failed: {exc}")
            continue
        samples.append(theta_b)

    if not samples:
        raise RuntimeError("all bootstrap iterations failed")

    return _bootstrap_percentiles(samples, param_names, alpha)


def parallel_bootstrap_ci(
    worker_fn: Callable[[tuple], dict | None],
    payloads: Sequence[tuple],
    param_names: Sequence[str],
    alpha: float = 0.05,
    n_jobs: int = 1,
    desc: str = "bootstrap",
    verbose: bool = True,
    force_cpu_workers: bool = True,
    return_draws: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Parallel bootstrap via ``ProcessPoolExecutor``.

    ``worker_fn`` must be importable at module top level (pickle-able for
    ``multiprocessing``). It receives one payload tuple and returns either a
    theta dict or ``None`` on failure. ``payloads`` is an iterable of
    per-iteration argument tuples (e.g. ``(duration_ms, seed)``).

    For CPU efficiency, the worker should call ``torch.set_num_threads(1)``
    on entry to avoid oversubscription when torch is used inside the fit.

    ``n_jobs``: 1 = serial, -1 = all cores, else the exact number of workers.

    ``force_cpu_workers`` (default True): when the parent is using CUDA, all
    workers would contend for the same GPU and serialize — defeating
    parallelism. Setting ``CUDA_VISIBLE_DEVICES=""`` in the environment just
    before the pool forks children makes them import torch as CPU-only, so
    N workers become true multi-core parallel. Parent env is restored on
    exit. Set False only if you know GPU parallelism helps your workload.

    If ``return_draws`` is True, returns ``(summary_df, draws_df)`` where
    ``draws_df`` has one row per successful bootstrap replicate and one column
    per name in ``param_names``. Otherwise returns only ``summary_df``.
    """
    import os

    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, int(n_jobs))

    if n_jobs > 1 and force_cpu_workers:
        try:
            _has_cuda = bool(torch.cuda.is_available())
        except Exception:
            _has_cuda = False
        if _has_cuda and verbose:
            print(
                f"[parallel_bootstrap_ci] n_jobs={n_jobs} with CUDA detected: "
                "forcing workers to CPU to avoid GPU contention. Per-iter "
                "CPU cost is ~3-5× GPU, so n_jobs>1 only pays off when each "
                "fit is cheap (e.g. --only-tau 1D refits). If your fit is "
                "full-dim and GPU is fast, use n_jobs=1 instead."
            )

    samples: list[dict] = []
    iterator: Sequence[tuple] = list(payloads)

    pbar = None
    if verbose:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(iterator), desc=desc, unit="fit")
        except ImportError:
            pbar = None

    if n_jobs == 1:
        for payload in iterator:
            try:
                out = worker_fn(payload)
            except Exception as exc:
                if verbose:
                    print(f"  iter {payload} failed: {exc}")
                out = None
            if out is not None:
                samples.append(out)
            if pbar is not None:
                pbar.update(1)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # torch + fork on Linux silently corrupts the inherited thread pool,
        # causing all simulations inside workers to fail. "spawn" starts a
        # fresh Python interpreter per worker, which is slower to start but
        # gives a clean torch state.
        ctx = mp.get_context("spawn")

        cvd_key = "CUDA_VISIBLE_DEVICES"
        orig_cvd = os.environ.get(cvd_key, None)
        if force_cpu_workers:
            # Spawned children inherit the current env: blank CUDA visibility
            # makes torch fall back to CPU inside the worker, so N workers
            # actually run on N cores instead of queueing on one GPU.
            os.environ[cvd_key] = ""

        try:
            with ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=ctx
            ) as ex:
                futures = [ex.submit(worker_fn, p) for p in iterator]
                for fut in as_completed(futures):
                    try:
                        out = fut.result()
                    except Exception as exc:
                        if verbose:
                            print(f"  worker failed: {exc}")
                        out = None
                    if out is not None:
                        samples.append(out)
                    if pbar is not None:
                        pbar.update(1)
        finally:
            if force_cpu_workers:
                if orig_cvd is None:
                    os.environ.pop(cvd_key, None)
                else:
                    os.environ[cvd_key] = orig_cvd

    if pbar is not None:
        pbar.close()

    if not samples:
        raise RuntimeError("all bootstrap iterations failed")

    summary = _bootstrap_percentiles(samples, param_names, alpha)
    if return_draws:
        draws = pd.DataFrame(
            [{p: s.get(p) for p in param_names} for s in samples]
        )
        return summary, draws
    return summary


# ── Psychometric helper ─────────────────────────────────────────────────


def psychometric_weber(ratio: np.ndarray, w: float) -> np.ndarray:
    """2AFC Weber psychometric for ratios ``r = n_large / n_small``.

    If both numerosities have Gaussian Weber noise, the decision variable
    ``x_large - x_small`` has standard deviation ``w * n_small * sqrt(r^2 + 1)``.
    """
    ratio = np.asarray(ratio, dtype=float)
    denom = w * np.sqrt(2.0 * (ratio * ratio + 1.0))
    return 50.0 * (1.0 + erf((ratio - 1.0) / denom))
