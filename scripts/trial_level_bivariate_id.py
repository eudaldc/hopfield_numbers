"""All-duration hierarchical link between landscape resolution and Weber noise.

The analysis uses the five exposure durations and disjoint numerosity channels:
``n <= 8`` informs participant landscape resolution and ``n >= 9`` informs
participant Weber noise. Duration-specific population means absorb exposure
effects. A participant lapse random intercept is shared across durations and
both response channels.

The participant effects ``(u_n, u_w, u_lapse)`` are jointly Gaussian. The
reported target is the partial correlation between ``u_n`` and ``u_w`` given
``u_lapse``. Its likelihood-ratio null fixes only that partial correlation to
zero and reoptimizes every nuisance parameter.

The marginal likelihood uses tensor Gauss--Hermite quadrature. PyTorch provides
exact gradients and uses CUDA when available; SciPy performs deterministic
multistart L-BFGS-B optimization.

    .venv/bin/python -m scripts.trial_level_bivariate_id

Outputs under ``output/trial_level_model_comparison/``:
  bivariate_id_summary.csv
  bivariate_id_participants.csv
  bivariate_id_audit.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.trial_level_model_comparison as T

DURS = tuple(T.DURATIONS)
DUR_INDEX = {duration: index for index, duration in enumerate(DURS)}
N_DURS = len(DURS)
R_MAX = 99
N_LO = 8
N_HI = 9
N_GH = 9
N_WIDTH_GRID = 40
W_GRID = 40
GAMMA_GRID = 21
QUAD_NODES = 151
CORR_LIMIT = 0.97

OUT_DIR = T.OUT_DIR
CACHE = OUT_DIR / "bivariate_id_all_durations_tables.npz"
OUT_SUMMARY = OUT_DIR / "bivariate_id_summary.csv"
OUT_PARTICIPANTS = OUT_DIR / "bivariate_id_participants.csv"
OUT_AUDIT = OUT_DIR / "bivariate_id_audit.txt"

ALPHA_N_BOUNDS = (float(np.log(3.0)), float(np.log(20.0)))
ALPHA_W_BOUNDS = (float(np.log(0.04)), float(np.log(0.55)))
ALPHA_LAPSE_BOUNDS = (float(logit(0.0005)), float(logit(0.5)))
LOG_TAU_N_BOUNDS = (float(np.log(0.02)), float(np.log(1.5)))
LOG_TAU_W_BOUNDS = (float(np.log(0.02)), float(np.log(1.5)))
LOG_TAU_LAPSE_BOUNDS = (float(np.log(0.03)), float(np.log(4.0)))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def load_complete_trials() -> pd.DataFrame:
    """Return participants who completed all five exposure durations."""
    trials = T.load_trials().copy()
    trials["n_guess"] = np.clip(trials["n_guess"].to_numpy(int), 1, R_MAX)
    n_durations = trials.groupby("uniqueid")["exp_dur"].nunique()
    complete_ids = n_durations[n_durations == N_DURS].index
    trials = trials[trials["uniqueid"].isin(complete_ids)].copy()

    has_small = trials[trials["n_shown"] <= N_LO].groupby("uniqueid").size()
    has_large = trials[trials["n_shown"] >= N_HI].groupby("uniqueid").size()
    valid_ids = complete_ids.intersection(has_small.index).intersection(has_large.index)
    return trials[trials["uniqueid"].isin(valid_ids)].reset_index(drop=True)


def group_winner_params() -> dict:
    theta = pd.read_csv(OUT_DIR / "shared_landscape_ladder_theta.csv")
    theta = theta[theta["model_key"] == T.WINNER_KEY].set_index("duration_ms")
    missing = set(DURS).difference(theta.index)
    if missing:
        raise ValueError(f"group winner is missing durations: {sorted(missing)}")
    powers = theta.loc[list(DURS), "pow"].to_numpy(float)
    if not np.allclose(powers, powers[0], atol=1e-10):
        raise ValueError("group winner does not have one shared width exponent")
    return {
        "pow": float(powers[0]),
        "sens": {
            int(duration): {
                "s0": float(theta.loc[duration, "s0"]),
                "lambda0": float(theta.loc[duration, "lambda0"]),
                "lapse": float(theta.loc[duration, "lapse"]),
            }
            for duration in DURS
        },
    }


def build_or_load_tables(
    power: float,
    sensory: dict[int, dict[str, float]],
    *,
    rebuild: bool = False,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE, allow_pickle=False)
        cached_durations = tuple(cached["durations"].astype(int).tolist())
        if cached_durations == DURS and np.isclose(float(cached["power"]), power):
            width = {
                duration: (cached["log_nwidth"], cached[f"nwidth_{duration}"])
                for duration in DURS
            }
            print(f"Loaded cached likelihood tables from {CACHE}", flush=True)
            return width, (cached["log_w"], cached["gammas"], cached["weber"])

    quadrature = T.make_quadrature(QUAD_NODES, kind="grid")
    spec = T.MODEL_BY_KEY[T.WINNER_KEY]
    log_nwidth = np.linspace(np.log(2.5), np.log(22.0), N_WIDTH_GRID)
    width_tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    save_data: dict[str, np.ndarray | float] = {
        "durations": np.asarray(DURS, dtype=int),
        "power": np.asarray(power),
        "log_nwidth": log_nwidth,
    }
    for duration in DURS:
        table = np.empty((N_WIDTH_GRID, 15, R_MAX), dtype=np.float64)
        for index, log_width in enumerate(
            tqdm(log_nwidth, desc=f"n_width {duration} ms", leave=False)
        ):
            theta = {
                "sigma_c": float(np.exp(log_width)) ** (-power),
                "s0": sensory[duration]["s0"],
                "lambda0": sensory[duration]["lambda0"],
                "pow": power,
            }
            probabilities = np.zeros((15, R_MAX), dtype=np.float64)
            probabilities[:, : T.N_WELLS] = T.compute_grid(
                spec, theta, quadrature
            )["p_resp"]
            table[index] = np.log(np.clip(probabilities, 1e-300, None))
        width_tables[duration] = (log_nwidth, table)
        save_data[f"nwidth_{duration}"] = table

    log_w = np.linspace(np.log(0.03), np.log(0.7), W_GRID)
    gammas = np.linspace(0.40, 1.20, GAMMA_GRID)
    weber = np.empty((W_GRID, GAMMA_GRID, 15, R_MAX), dtype=np.float64)
    edges = np.arange(0.5, R_MAX + 1.5)
    for w_index, log_width in enumerate(tqdm(log_w, desc="Weber", leave=False)):
        width = float(np.exp(log_width))
        for gamma_index, gain in enumerate(gammas):
            for numerosity in range(1, 16):
                probabilities = np.diff(
                    norm.cdf(
                        edges,
                        loc=gain * numerosity,
                        scale=max(width * numerosity, 1e-3),
                    )
                )
                total = probabilities.sum()
                if total <= 0:
                    probabilities = np.full(R_MAX, 1.0 / R_MAX)
                else:
                    probabilities /= total
                weber[w_index, gamma_index, numerosity - 1] = np.log(
                    np.clip(probabilities, 1e-300, None)
                )
    save_data.update({"log_w": log_w, "gammas": gammas, "weber": weber})
    np.savez(CACHE, **save_data)
    print(f"Cached likelihood tables at {CACHE}", flush=True)
    return width_tables, (log_w, gammas, weber)


def _mixture_log_probability(log_probability: np.ndarray, lapse: float) -> np.ndarray:
    probability = (1.0 - lapse) * np.exp(log_probability) + lapse / R_MAX
    return np.log(np.clip(probability, 1e-300, None))


def pooled_initial_values(
    trials: pd.DataFrame,
    width_tables: dict[int, tuple[np.ndarray, np.ndarray]],
    weber_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    sensory: dict[int, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Obtain deterministic group-level starts for the hierarchical fit."""
    alpha_n = np.empty(N_DURS)
    alpha_w = np.empty(N_DURS)
    gamma_by_duration: dict[int, float] = {}
    log_w_grid, gammas, weber_table = weber_pack

    for duration_index, duration in enumerate(DURS):
        lapse = sensory[duration]["lapse"]
        small = trials[
            (trials["exp_dur"] == duration) & (trials["n_shown"] <= N_LO)
        ]
        counts_small = np.zeros((15, R_MAX), dtype=float)
        np.add.at(
            counts_small,
            (
                small["n_shown"].to_numpy(int) - 1,
                small["n_guess"].to_numpy(int) - 1,
            ),
            1.0,
        )
        log_nwidth_grid, width_table = width_tables[duration]
        logp = _mixture_log_probability(width_table, lapse)
        nll = -np.einsum("nr,gnr->g", counts_small, logp)
        alpha_n[duration_index] = log_nwidth_grid[int(np.argmin(nll))]

        large = trials[
            (trials["exp_dur"] == duration) & (trials["n_shown"] >= N_HI)
        ]
        counts_large = np.zeros((15, R_MAX), dtype=float)
        np.add.at(
            counts_large,
            (
                large["n_shown"].to_numpy(int) - 1,
                large["n_guess"].to_numpy(int) - 1,
            ),
            1.0,
        )
        logp_weber = _mixture_log_probability(weber_table, lapse)
        nll_weber = -np.einsum("nr,wgnr->wg", counts_large, logp_weber)
        w_index, gamma_index = np.unravel_index(
            int(np.argmin(nll_weber)), nll_weber.shape
        )
        alpha_w[duration_index] = log_w_grid[w_index]
        gamma_by_duration[duration] = float(gammas[gamma_index])
        print(
            f"  start {duration:>3} ms: n_width={np.exp(alpha_n[duration_index]):.2f}, "
            f"w={np.exp(alpha_w[duration_index]):.3f}, "
            f"gamma={gamma_by_duration[duration]:.3f}",
            flush=True,
        )
    return alpha_n, alpha_w, gamma_by_duration


def normal_product_nodes(n_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    locations, weights = np.polynomial.hermite.hermgauss(n_nodes)
    axes = np.meshgrid(*([locations * np.sqrt(2.0)] * 3), indexing="ij")
    weight_axes = np.meshgrid(*([weights / np.sqrt(np.pi)] * 3), indexing="ij")
    nodes = np.stack([axis.ravel() for axis in axes], axis=1)
    joint_weights = np.prod(np.stack(weight_axes, axis=0), axis=0).ravel()
    return (
        torch.as_tensor(nodes, dtype=torch.float64, device=device),
        torch.as_tensor(np.log(joint_weights), dtype=torch.float64, device=device),
    )


def latent_effects(hypothesis: dict[str, torch.Tensor], nodes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Map independent normals to lapse, n_width, and Weber effects."""
    z_lapse, z_n, z_w = nodes.T
    rho_n_lapse = hypothesis["rho_n_lapse"]
    rho_w_lapse = hypothesis["rho_w_lapse"]
    rho_partial = hypothesis["rho"]
    u_lapse = hypothesis["tau_lapse"] * z_lapse
    u_n = hypothesis["tau_n"] * (
        rho_n_lapse * z_lapse
        + torch.sqrt(torch.clamp(1.0 - rho_n_lapse.square(), min=1e-12)) * z_n
    )
    u_w = hypothesis["tau_w"] * (
        rho_w_lapse * z_lapse
        + torch.sqrt(torch.clamp(1.0 - rho_w_lapse.square(), min=1e-12))
        * (
            rho_partial * z_n
            + torch.sqrt(torch.clamp(1.0 - rho_partial.square(), min=1e-12))
            * z_w
        )
    )
    return u_lapse, u_n, u_w


def marginal_n_w_correlation(
    rho_partial: float, rho_n_lapse: float, rho_w_lapse: float
) -> float:
    return float(
        rho_n_lapse * rho_w_lapse
        + np.sqrt(max(1.0 - rho_n_lapse**2, 0.0))
        * np.sqrt(max(1.0 - rho_w_lapse**2, 0.0))
        * rho_partial
    )


def _bounded(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * torch.sigmoid(raw)


def unpack_parameters(raw: torch.Tensor, *, null: bool) -> dict[str, torch.Tensor]:
    position = 0
    alpha_n = _bounded(raw[position : position + N_DURS], ALPHA_N_BOUNDS)
    position += N_DURS
    alpha_w = _bounded(raw[position : position + N_DURS], ALPHA_W_BOUNDS)
    position += N_DURS
    alpha_lapse = _bounded(
        raw[position : position + N_DURS], ALPHA_LAPSE_BOUNDS
    )
    position += N_DURS
    log_tau_n = _bounded(raw[position], LOG_TAU_N_BOUNDS)
    log_tau_w = _bounded(raw[position + 1], LOG_TAU_W_BOUNDS)
    log_tau_lapse = _bounded(raw[position + 2], LOG_TAU_LAPSE_BOUNDS)
    position += 3
    rho_n_lapse = CORR_LIMIT * torch.tanh(raw[position])
    rho_w_lapse = CORR_LIMIT * torch.tanh(raw[position + 1])
    position += 2
    if null:
        rho = torch.zeros((), dtype=raw.dtype, device=raw.device)
    else:
        rho = CORR_LIMIT * torch.tanh(raw[position])
        position += 1
    if position != raw.numel():
        raise ValueError(f"unexpected parameter count: used {position}, got {raw.numel()}")
    return {
        "alpha_n": alpha_n,
        "alpha_w": alpha_w,
        "alpha_lapse": alpha_lapse,
        "tau_n": torch.exp(log_tau_n),
        "tau_w": torch.exp(log_tau_w),
        "tau_lapse": torch.exp(log_tau_lapse),
        "rho_n_lapse": rho_n_lapse,
        "rho_w_lapse": rho_w_lapse,
        "rho": rho,
    }


def _inverse_bounded(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    proportion = np.clip((np.asarray(values) - lower) / (upper - lower), 1e-7, 1 - 1e-7)
    return np.log(proportion / (1.0 - proportion))


def encode_parameters(hypothesis: dict, *, null: bool) -> np.ndarray:
    pieces = [
        _inverse_bounded(np.asarray(hypothesis["alpha_n"]), ALPHA_N_BOUNDS),
        _inverse_bounded(np.asarray(hypothesis["alpha_w"]), ALPHA_W_BOUNDS),
        _inverse_bounded(np.asarray(hypothesis["alpha_lapse"]), ALPHA_LAPSE_BOUNDS),
        _inverse_bounded(np.log([hypothesis["tau_n"]]), LOG_TAU_N_BOUNDS),
        _inverse_bounded(np.log([hypothesis["tau_w"]]), LOG_TAU_W_BOUNDS),
        _inverse_bounded(np.log([hypothesis["tau_lapse"]]), LOG_TAU_LAPSE_BOUNDS),
        np.arctanh(
            np.clip(
                [hypothesis["rho_n_lapse"], hypothesis["rho_w_lapse"]],
                -CORR_LIMIT + 1e-7,
                CORR_LIMIT - 1e-7,
            )
            / CORR_LIMIT
        ),
    ]
    if not null:
        pieces.append(
            np.arctanh(
                np.clip(hypothesis["rho"], -CORR_LIMIT + 1e-7, CORR_LIMIT - 1e-7)
                / CORR_LIMIT
            ).reshape(1)
        )
    return np.concatenate(pieces).astype(np.float64)


def hypothesis_to_numpy(hypothesis: dict[str, torch.Tensor]) -> dict:
    output = {}
    for name, value in hypothesis.items():
        array = value.detach().cpu().numpy()
        output[name] = float(array) if array.ndim == 0 else array.astype(float)
    return output


class HierarchicalLikelihood:
    def __init__(
        self,
        trials: pd.DataFrame,
        width_tables: dict[int, tuple[np.ndarray, np.ndarray]],
        weber_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
        gamma_by_duration: dict[int, float],
        *,
        n_gh: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.participant_ids = sorted(trials["uniqueid"].unique().tolist())
        self.participant_index = {
            participant_id: index
            for index, participant_id in enumerate(self.participant_ids)
        }
        self.n_participants = len(self.participant_ids)
        self.nodes, self.log_weights = normal_product_nodes(n_gh, device)
        self.width_grid = torch.as_tensor(
            width_tables[DURS[0]][0], dtype=torch.float64, device=device
        )
        self.width_tables = [
            torch.as_tensor(width_tables[d][1], dtype=torch.float64, device=device)
            for d in DURS
        ]
        self.width_slopes = [
            torch.as_tensor(
                PchipInterpolator(
                    width_tables[d][0], width_tables[d][1], axis=0
                ).derivative()(width_tables[d][0]),
                dtype=torch.float64,
                device=device,
            )
            for d in DURS
        ]
        log_w_grid, gammas, weber_table = weber_pack
        self.weber_grid = torch.as_tensor(log_w_grid, dtype=torch.float64, device=device)
        self.weber_tables = []
        self.weber_slopes = []
        for duration in DURS:
            gain = gamma_by_duration[duration]
            gamma_index = int(np.clip(np.searchsorted(gammas, gain) - 1, 0, len(gammas) - 2))
            fraction = (gain - gammas[gamma_index]) / (
                gammas[gamma_index + 1] - gammas[gamma_index]
            )
            table = (
                (1.0 - fraction) * weber_table[:, gamma_index]
                + fraction * weber_table[:, gamma_index + 1]
            )
            self.weber_tables.append(
                torch.as_tensor(table, dtype=torch.float64, device=device)
            )
            self.weber_slopes.append(
                torch.as_tensor(
                    PchipInterpolator(log_w_grid, table, axis=0).derivative()(log_w_grid),
                    dtype=torch.float64,
                    device=device,
                )
            )

        self.trial_blocks: list[dict[str, tuple[torch.Tensor, ...]]] = []
        small_counts = np.zeros(self.n_participants, dtype=int)
        large_counts = np.zeros(self.n_participants, dtype=int)
        for duration in DURS:
            duration_trials = trials[trials["exp_dur"] == duration]
            block: dict[str, tuple[torch.Tensor, ...]] = {}
            for channel, mask in (
                ("n", duration_trials["n_shown"] <= N_LO),
                ("w", duration_trials["n_shown"] >= N_HI),
            ):
                selected = duration_trials[mask]
                participant = torch.as_tensor(
                    selected["uniqueid"]
                    .map(self.participant_index)
                    .to_numpy(dtype=int, copy=True),
                    dtype=torch.long,
                    device=device,
                )
                numerosity = torch.as_tensor(
                    selected["n_shown"].to_numpy(dtype=int, copy=True) - 1,
                    dtype=torch.long,
                    device=device,
                )
                response = torch.as_tensor(
                    selected["n_guess"].to_numpy(dtype=int, copy=True) - 1,
                    dtype=torch.long,
                    device=device,
                )
                block[channel] = (participant, numerosity, response)
                increments = np.bincount(
                    selected["uniqueid"].map(self.participant_index).to_numpy(int),
                    minlength=self.n_participants,
                )
                if channel == "n":
                    small_counts += increments
                else:
                    large_counts += increments
            self.trial_blocks.append(block)
        self.small_counts = small_counts
        self.large_counts = large_counts

    @staticmethod
    def _interpolate(
        grid: torch.Tensor,
        table: torch.Tensor,
        slopes: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.clamp(values, min=grid[0], max=grid[-1])
        position = (values - grid[0]) / (grid[1] - grid[0])
        lower = torch.floor(position).to(torch.long)
        lower = torch.clamp(lower, min=0, max=grid.numel() - 2)
        interval = grid[lower + 1] - grid[lower]
        fraction = (values - grid[lower]) / interval
        fraction2 = fraction.square()
        fraction3 = fraction2 * fraction
        h00 = 2.0 * fraction3 - 3.0 * fraction2 + 1.0
        h10 = fraction3 - 2.0 * fraction2 + fraction
        h01 = -2.0 * fraction3 + 3.0 * fraction2
        h11 = fraction3 - fraction2
        return (
            h00[:, None, None] * table[lower]
            + (h10 * interval)[:, None, None] * slopes[lower]
            + h01[:, None, None] * table[lower + 1]
            + (h11 * interval)[:, None, None] * slopes[lower + 1]
        )

    @staticmethod
    def _add_lapse(log_probability: torch.Tensor, lapse: torch.Tensor) -> torch.Tensor:
        log_model = torch.log1p(-lapse)[:, None, None] + log_probability
        log_contamination = torch.log(lapse)[:, None, None] - np.log(R_MAX)
        return torch.logaddexp(log_model, log_contamination)

    def participant_node_loglik(
        self, hypothesis: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        u_lapse, u_n, u_w = latent_effects(hypothesis, self.nodes)
        node_loglik = torch.zeros(
            (self.n_participants, self.nodes.shape[0]),
            dtype=torch.float64,
            device=self.device,
        )
        for duration_index in range(N_DURS):
            lapse = torch.sigmoid(hypothesis["alpha_lapse"][duration_index] + u_lapse)
            log_n = self._interpolate(
                self.width_grid,
                self.width_tables[duration_index],
                self.width_slopes[duration_index],
                hypothesis["alpha_n"][duration_index] + u_n,
            )
            log_n = self._add_lapse(log_n, lapse)
            participant, numerosity, response = self.trial_blocks[duration_index]["n"]
            node_loglik = node_loglik.index_add(
                0, participant, log_n[:, numerosity, response].T
            )

            log_w = self._interpolate(
                self.weber_grid,
                self.weber_tables[duration_index],
                self.weber_slopes[duration_index],
                hypothesis["alpha_w"][duration_index] + u_w,
            )
            log_w = self._add_lapse(log_w, lapse)
            participant, numerosity, response = self.trial_blocks[duration_index]["w"]
            node_loglik = node_loglik.index_add(
                0, participant, log_w[:, numerosity, response].T
            )
        return node_loglik, (u_lapse, u_n, u_w)

    def loglik(self, raw: torch.Tensor, *, null: bool) -> torch.Tensor:
        hypothesis = unpack_parameters(raw, null=null)
        node_loglik, _ = self.participant_node_loglik(hypothesis)
        return torch.logsumexp(node_loglik + self.log_weights[None, :], dim=1).sum()


def fit_one_start(
    likelihood: HierarchicalLikelihood,
    start: np.ndarray,
    *,
    null: bool,
    maxiter: int,
    adam_steps: int,
) -> object:
    def objective(raw_numpy: np.ndarray) -> tuple[float, np.ndarray]:
        raw = torch.tensor(
            raw_numpy, dtype=torch.float64, device=likelihood.device, requires_grad=True
        )
        nll = -likelihood.loglik(raw, null=null)
        nll.backward()
        gradient = raw.grad.detach().cpu().numpy().astype(np.float64)
        return float(nll.detach().cpu()), gradient

    # Linear table interpolation is continuous but kinked at its knots. The
    # pooled starts lie on those knots, where a line-search method can stop on
    # its first iteration. Adam supplies a deterministic subgradient warm-up;
    # L-BFGS-B then polishes the best warm-up point.
    parameter = torch.nn.Parameter(
        torch.as_tensor(start, dtype=torch.float64, device=likelihood.device).clone()
    )
    optimizer = torch.optim.Adam([parameter], lr=0.025)
    best_raw = start.copy()
    best_fun = np.inf
    for step in range(adam_steps):
        if step == adam_steps * 2 // 5:
            optimizer.param_groups[0]["lr"] = 0.008
        elif step == adam_steps * 4 // 5:
            optimizer.param_groups[0]["lr"] = 0.002
        optimizer.zero_grad(set_to_none=True)
        nll = -likelihood.loglik(parameter, null=null)
        if not torch.isfinite(nll):
            raise RuntimeError("non-finite likelihood during Adam warm-up")
        nll.backward()
        value = float(nll.detach().cpu())
        if value < best_fun:
            best_fun = value
            best_raw = parameter.detach().cpu().numpy().copy()
        torch.nn.utils.clip_grad_norm_([parameter], max_norm=5_000.0)
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(-9.0, 9.0)

    polished = minimize(
        objective,
        best_raw,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-9.0, 9.0)] * len(start),
        options={
            "maxiter": maxiter,
            "maxls": 30,
            "maxcor": 20,
            "ftol": 1e-15,
            "gtol": 2e-5,
        },
    )
    adam_fun, adam_jac = objective(best_raw)
    if adam_fun + 1e-8 < float(polished.fun):
        polished.x = best_raw
        polished.fun = adam_fun
        polished.jac = adam_jac
        polished.success = True
        polished.message = "Adam warm-up retained; L-BFGS-B did not improve"
        polished.nit = adam_steps
    return polished


def fit_multistart(
    likelihood: HierarchicalLikelihood,
    hypotheses: list[dict],
    *,
    null: bool,
    maxiter: int,
    adam_steps: int,
) -> tuple[object, list[object]]:
    results = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        started = time.time()
        result = fit_one_start(
            likelihood,
            encode_parameters(hypothesis, null=null),
            null=null,
            maxiter=maxiter,
            adam_steps=adam_steps,
        )
        results.append(result)
        print(
            f"  start {index}: logL={-result.fun:.3f}, success={result.success}, "
            f"nit={result.nit}, |grad|_inf={np.max(np.abs(result.jac)):.2e}, "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
    best = min(results, key=lambda result: float(result.fun))
    if np.max(np.abs(best.jac)) > 1.0:
        started = time.time()
        refinement = fit_one_start(
            likelihood,
            best.x,
            null=null,
            maxiter=maxiter,
            adam_steps=max(200, adam_steps // 2),
        )
        results.append(refinement)
        print(
            f"  refinement: logL={-refinement.fun:.3f}, "
            f"success={refinement.success}, nit={refinement.nit}, "
            f"|grad|_inf={np.max(np.abs(refinement.jac)):.2e}, "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
        best = min(results, key=lambda result: float(result.fun))
    return best, results


def participant_estimates(
    likelihood: HierarchicalLikelihood,
    raw_numpy: np.ndarray,
) -> pd.DataFrame:
    raw = torch.as_tensor(raw_numpy, dtype=torch.float64, device=likelihood.device)
    hypothesis = unpack_parameters(raw, null=False)
    node_loglik, (u_lapse, u_n, u_w) = likelihood.participant_node_loglik(hypothesis)
    posterior = torch.softmax(node_loglik + likelihood.log_weights[None, :], dim=1)
    marginal_loglik = torch.logsumexp(
        node_loglik + likelihood.log_weights[None, :], dim=1
    )

    mean_u_lapse = posterior @ u_lapse
    mean_u_n = posterior @ u_n
    mean_u_w = posterior @ u_w
    nwidth_by_duration = torch.stack(
        [posterior @ torch.exp(alpha + u_n) for alpha in hypothesis["alpha_n"]],
        dim=1,
    )
    weber_by_duration = torch.stack(
        [posterior @ torch.exp(alpha + u_w) for alpha in hypothesis["alpha_w"]],
        dim=1,
    )
    lapse_by_duration = torch.stack(
        [
            posterior @ torch.sigmoid(alpha + u_lapse)
            for alpha in hypothesis["alpha_lapse"]
        ],
        dim=1,
    )

    output = pd.DataFrame(
        {
            "cutoff": f"{N_LO}/{N_HI}",
            "uniqueid": likelihood.participant_ids,
            "n_small": likelihood.small_counts,
            "n_large": likelihood.large_counts,
            "u_n": mean_u_n.detach().cpu().numpy(),
            "u_w": mean_u_w.detach().cpu().numpy(),
            "u_lapse": mean_u_lapse.detach().cpu().numpy(),
            "n_width_mean": nwidth_by_duration.mean(dim=1).detach().cpu().numpy(),
            "weber_mean": weber_by_duration.mean(dim=1).detach().cpu().numpy(),
            "lapse_mean": lapse_by_duration.mean(dim=1).detach().cpu().numpy(),
            "loglik": marginal_loglik.detach().cpu().numpy(),
        }
    )
    for duration_index, duration in enumerate(DURS):
        output[f"n_width_{duration}"] = (
            nwidth_by_duration[:, duration_index].detach().cpu().numpy()
        )
        output[f"weber_{duration}"] = (
            weber_by_duration[:, duration_index].detach().cpu().numpy()
        )
        output[f"lapse_{duration}"] = (
            lapse_by_duration[:, duration_index].detach().cpu().numpy()
        )
    return output


def summary_row(
    likelihood: HierarchicalLikelihood,
    alternative: object,
    null_fit: object,
    *,
    n_gh: int,
    elapsed: float,
    gamma_by_duration: dict[int, float],
    null_results: list[object],
    alternative_results: list[object],
) -> dict:
    raw = torch.as_tensor(
        alternative.x, dtype=torch.float64, device=likelihood.device
    )
    hypothesis = hypothesis_to_numpy(unpack_parameters(raw, null=False))
    loglik = -float(alternative.fun)
    loglik_null = -float(null_fit.fun)
    lr = max(0.0, 2.0 * (loglik - loglik_null))
    p_value = float(chi2.sf(lr, 1))
    row = {
        "analysis": "all_durations_shared_lapse",
        "cutoff": f"{N_LO}/{N_HI}",
        "n_lo": N_LO,
        "n_hi": N_HI,
        "n_durations": N_DURS,
        "n_participants": likelihood.n_participants,
        "n_trials_small": int(likelihood.small_counts.sum()),
        "n_trials_large": int(likelihood.large_counts.sum()),
        "rho": hypothesis["rho"],
        "rho_partial_n_w_given_lapse": hypothesis["rho"],
        "rho_marginal_n_w": marginal_n_w_correlation(
            hypothesis["rho"],
            hypothesis["rho_n_lapse"],
            hypothesis["rho_w_lapse"],
        ),
        "rho_n_lapse": hypothesis["rho_n_lapse"],
        "rho_w_lapse": hypothesis["rho_w_lapse"],
        "tau_n": hypothesis["tau_n"],
        "tau_w": hypothesis["tau_w"],
        "tau_lapse": hypothesis["tau_lapse"],
        "marg_loglik": loglik,
        "marg_loglik_rho0": loglik_null,
        "lr_rho": lr,
        "p_lr": p_value,
        "p_rho_negative": p_value / 2.0 if hypothesis["rho"] < 0 else 1.0 - p_value / 2.0,
        "quadrature_nodes_per_dimension": n_gh,
        "n_quadrature_nodes": n_gh**3,
        "n_starts_null": len(null_results),
        "n_starts_alternative": len(alternative_results),
        "fit_success": bool(null_fit.success and alternative.success),
        "null_gradient_inf": float(np.max(np.abs(null_fit.jac))),
        "alternative_gradient_inf": float(np.max(np.abs(alternative.jac))),
        "elapsed_s": elapsed,
        "fit_message": f"null: {null_fit.message}; alternative: {alternative.message}",
    }
    for duration_index, duration in enumerate(DURS):
        row[f"alpha_n_{duration}"] = hypothesis["alpha_n"][duration_index]
        row[f"alpha_w_{duration}"] = hypothesis["alpha_w"][duration_index]
        row[f"lapse_{duration}_median"] = float(
            expit(hypothesis["alpha_lapse"][duration_index])
        )
        row[f"gamma_{duration}"] = gamma_by_duration[duration]
    return row


def audit_text(summary: pd.DataFrame, trials: pd.DataFrame, device: torch.device) -> str:
    row = summary.iloc[0]
    return "\n".join(
        [
            "All-duration hierarchical individual differences",
            "================================================",
            f"Complete participants: {int(row.n_participants)}",
            f"Exposure durations: {', '.join(map(str, DURS))} ms",
            f"Outcome trials: {len(trials)} total; n <= {N_LO} for n_width and "
            f"n >= {N_HI} for Weber w.",
            "Population means and lapse intercepts are duration-specific.",
            "Participant effects on log n_width, log w, and logit lapse are shared",
            "across durations and jointly Gaussian. The target rho is the partial",
            "n_width--w correlation conditional on participant lapse.",
            "The rho=0 null reoptimizes every nuisance parameter.",
            f"Quadrature: {int(row.quadrature_nodes_per_dimension)} nodes per dimension "
            f"({int(row.n_quadrature_nodes)} joint nodes).",
            f"Optimization device: {device}",
            "",
            "Result",
            "------",
            summary.to_string(index=False),
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gh-nodes", type=int, default=N_GH)
    parser.add_argument("--maxiter", type=int, default=220)
    parser.add_argument("--adam-steps", type=int, default=500)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gh_nodes < 3:
        raise ValueError("--gh-nodes must be at least 3")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Using {device} with torch {torch.__version__}", flush=True)
    started = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = load_complete_trials()
    group = group_winner_params()
    tables, weber_pack = build_or_load_tables(
        group["pow"], group["sens"], rebuild=args.rebuild_cache
    )
    alpha_n_start, alpha_w_start, gamma = pooled_initial_values(
        trials, tables, weber_pack, group["sens"]
    )
    likelihood = HierarchicalLikelihood(
        trials,
        tables,
        weber_pack,
        gamma,
        n_gh=args.gh_nodes,
        device=device,
    )
    print(
        f"Fitting N={likelihood.n_participants}, trials={len(trials)}, "
        f"nodes={args.gh_nodes ** 3}",
        flush=True,
    )

    base = {
        "alpha_n": alpha_n_start,
        "alpha_w": alpha_w_start,
        "alpha_lapse": np.array(
            [logit(group["sens"][duration]["lapse"]) for duration in DURS]
        ),
        "tau_n": 0.35,
        "tau_w": 0.40,
        "tau_lapse": 0.80,
        "rho_n_lapse": -0.35,
        "rho_w_lapse": 0.40,
        "rho": 0.0,
    }
    null_starts = [
        base,
        {
            **base,
            "tau_n": 0.55,
            "tau_w": 0.55,
            "tau_lapse": 1.40,
            "rho_n_lapse": -0.55,
            "rho_w_lapse": 0.55,
        },
    ]
    print("Null fit (partial rho fixed at zero)", flush=True)
    null_fit, null_results = fit_multistart(
        likelihood,
        null_starts,
        null=True,
        maxiter=args.maxiter,
        adam_steps=args.adam_steps,
    )
    null_hypothesis = hypothesis_to_numpy(
        unpack_parameters(
            torch.as_tensor(null_fit.x, dtype=torch.float64, device=device), null=True
        )
    )

    alternative_starts = [
        {**null_hypothesis, "rho": -0.30},
        {**null_hypothesis, "rho": -0.65},
    ]
    print("Alternative fit (partial rho free)", flush=True)
    alternative, alternative_results = fit_multistart(
        likelihood,
        alternative_starts,
        null=False,
        maxiter=args.maxiter,
        adam_steps=args.adam_steps,
    )
    if -float(alternative.fun) + 1e-5 < -float(null_fit.fun):
        raise RuntimeError("optimized alternative has lower likelihood than the null")

    elapsed = time.time() - started
    summary = pd.DataFrame(
        [
            summary_row(
                likelihood,
                alternative,
                null_fit,
                n_gh=args.gh_nodes,
                elapsed=elapsed,
                gamma_by_duration=gamma,
                null_results=null_results,
                alternative_results=alternative_results,
            )
        ]
    )
    participants = participant_estimates(likelihood, alternative.x)
    summary.to_csv(OUT_SUMMARY, index=False)
    participants.to_csv(OUT_PARTICIPANTS, index=False)
    OUT_AUDIT.write_text(audit_text(summary, trials, device))

    row = summary.iloc[0]
    print(
        f"partial rho={row.rho:+.3f}; marginal rho={row.rho_marginal_n_w:+.3f}; "
        f"LR={row.lr_rho:.2f}; p={row.p_lr:.3g}; elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"Saved {OUT_SUMMARY}\nSaved {OUT_PARTICIPANTS}\nSaved {OUT_AUDIT}")


if __name__ == "__main__":
    main()
