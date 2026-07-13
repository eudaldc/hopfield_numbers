"""Topological resolution of the fitted power-law attractor landscape.

For each fitted landscape, this module locates every stationary point of

    V(x) = -log sum_n exp(g_n(x)),

where g_n is the normalized Gaussian log-well. Stationary points are bracketed
on a dense grid and refined with Brent's method. Local minima are assigned to
their nearest integers, and n_topo is the largest consecutive integer from one
whose minimum is separated from the preceding minimum by a positive barrier.

Run from the project root:

    .venv/bin/python -m scripts.topological_resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from scripts import trial_level_model_comparison as T


OUT_DIR = T.OUT_DIR
SUMMARY_PATH = OUT_DIR / "topological_resolution_summary.csv"
STATIONARY_PATH = OUT_DIR / "topological_stationary_points.csv"
BARRIER_PATH = OUT_DIR / "topological_barriers.csv"
GRID_SIZES = (5_000, 20_000, 80_000)


@dataclass(frozen=True)
class StationaryPoint:
    x: float
    potential: float
    curvature: float
    kind: str
    assigned_integer: int | None = None


@dataclass(frozen=True)
class Barrier:
    left_integer: int
    right_integer: int
    x: float
    potential: float
    height: float


@dataclass(frozen=True)
class TopologyResult:
    n_topo: int
    stationary_points: tuple[StationaryPoint, ...]
    barriers: tuple[Barrier, ...]


def _validate_landscape(wells: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wells = np.asarray(wells, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if wells.ndim != 1 or sigma.shape != wells.shape:
        raise ValueError("wells and sigma must be one-dimensional arrays of equal length")
    if wells.size < 2 or np.any(np.diff(wells) <= 0):
        raise ValueError("wells must contain at least two strictly increasing positions")
    if np.any(sigma <= 0):
        raise ValueError("all well widths must be positive")
    return wells, sigma


def potential_derivatives(
    x: np.ndarray | float,
    wells: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return V, V', and V'' for the Gaussian log-sum-exp landscape."""
    wells, sigma = _validate_landscape(wells, sigma)
    x_arr = np.asarray(x, dtype=float)
    x_col = x_arr[..., None]
    sigma_sq = sigma * sigma
    displacement = x_col - wells
    log_wells = -np.log(sigma) - displacement * displacement / (2.0 * sigma_sq)
    weights = np.exp(log_wells - logsumexp(log_wells, axis=-1, keepdims=True))
    slopes = displacement / sigma_sq
    potential = -logsumexp(log_wells, axis=-1)
    gradient = np.sum(weights * slopes, axis=-1)
    mean_precision = np.sum(weights / sigma_sq, axis=-1)
    slope_variance = np.sum(weights * slopes * slopes, axis=-1) - gradient * gradient
    curvature = mean_precision - slope_variance
    return potential, gradient, curvature


def _deduplicate_roots(roots: list[float], tolerance: float) -> list[float]:
    roots = sorted(roots)
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > tolerance:
            unique.append(root)
    return unique


def locate_stationary_points(
    wells: np.ndarray,
    sigma: np.ndarray,
    grid_points: int,
) -> tuple[StationaryPoint, ...]:
    """Bracket all simple roots of V' on the well support and refine them."""
    wells, sigma = _validate_landscape(wells, sigma)
    if grid_points < 3:
        raise ValueError("grid_points must be at least three")

    grid = np.linspace(wells[0], wells[-1], int(grid_points))
    _, gradient, _ = potential_derivatives(grid, wells, sigma)
    roots: list[float] = []

    exact = np.flatnonzero(gradient == 0.0)
    roots.extend(float(grid[i]) for i in exact)

    crossings = np.flatnonzero(np.signbit(gradient[:-1]) != np.signbit(gradient[1:]))
    for i in crossings:
        left, right = float(grid[i]), float(grid[i + 1])
        if gradient[i] == 0.0:
            roots.append(left)
        elif gradient[i + 1] == 0.0:
            roots.append(right)
        else:
            root = brentq(
                lambda value: float(potential_derivatives(value, wells, sigma)[1]),
                left,
                right,
                xtol=1e-13,
                rtol=4.0 * np.finfo(float).eps,
            )
            roots.append(float(root))

    spacing = float(grid[1] - grid[0])
    roots = _deduplicate_roots(roots, tolerance=max(1e-11, spacing * 1e-6))
    stationary: list[StationaryPoint] = []
    for root in roots:
        value, _, curvature = potential_derivatives(root, wells, sigma)
        curvature_float = float(curvature)
        if curvature_float > 0.0:
            kind = "minimum"
        elif curvature_float < 0.0:
            kind = "maximum"
        else:
            kind = "degenerate"
        stationary.append(
            StationaryPoint(
                x=root,
                potential=float(value),
                curvature=curvature_float,
                kind=kind,
            )
        )
    return tuple(stationary)


def assign_minima_to_integers(
    stationary_points: tuple[StationaryPoint, ...],
    integer_min: int,
    integer_max: int,
) -> tuple[StationaryPoint, ...]:
    """Assign each local minimum injectively to its nearest integer."""
    candidates: dict[int, list[StationaryPoint]] = {}
    for point in stationary_points:
        if point.kind != "minimum":
            continue
        integer = int(np.floor(point.x + 0.5))
        if integer_min <= integer <= integer_max:
            candidates.setdefault(integer, []).append(point)

    selected: dict[float, int] = {}
    for integer, points in candidates.items():
        best = min(points, key=lambda point: (abs(point.x - integer), point.potential))
        selected[best.x] = integer

    return tuple(
        StationaryPoint(
            x=point.x,
            potential=point.potential,
            curvature=point.curvature,
            kind=point.kind,
            assigned_integer=selected.get(point.x),
        )
        for point in stationary_points
    )


def compute_topological_resolution(
    wells: np.ndarray,
    sigma: np.ndarray,
    grid_points: int,
    report_integer_max: int,
) -> TopologyResult:
    """Compute stationary points, barriers, and the consecutive n_topo limit."""
    wells, sigma = _validate_landscape(wells, sigma)
    integer_min = int(round(wells[0]))
    stationary = assign_minima_to_integers(
        locate_stationary_points(wells, sigma, grid_points),
        integer_min=integer_min,
        integer_max=report_integer_max,
    )
    minima = {
        point.assigned_integer: point
        for point in stationary
        if point.kind == "minimum" and point.assigned_integer is not None
    }
    maxima = [point for point in stationary if point.kind == "maximum"]

    if integer_min not in minima:
        return TopologyResult(n_topo=integer_min - 1, stationary_points=stationary, barriers=())

    n_topo = integer_min
    barriers: list[Barrier] = []
    for integer in range(integer_min + 1, report_integer_max + 1):
        if integer not in minima:
            break
        left_minimum = minima[integer - 1]
        right_minimum = minima[integer]
        intervening = [
            point for point in maxima if left_minimum.x < point.x < right_minimum.x
        ]
        if not intervening:
            break
        saddle = max(intervening, key=lambda point: point.potential)
        height = saddle.potential - max(left_minimum.potential, right_minimum.potential)
        scale = max(
            1.0,
            abs(saddle.potential),
            abs(left_minimum.potential),
            abs(right_minimum.potential),
        )
        numerical_tolerance = 64.0 * np.finfo(float).eps * scale
        if height <= numerical_tolerance:
            break
        barriers.append(
            Barrier(
                left_integer=integer - 1,
                right_integer=integer,
                x=saddle.x,
                potential=saddle.potential,
                height=height,
            )
        )
        n_topo = integer

    return TopologyResult(
        n_topo=n_topo,
        stationary_points=stationary,
        barriers=tuple(barriers),
    )


def _fit_rows() -> list[dict[str, float | int | str]]:
    theta = pd.read_csv(OUT_DIR / "pow_theta.csv")
    rows: list[dict[str, float | int | str]] = [
        {
            "fit": f"{int(row.duration_ms)} ms",
            "fit_type": "separate",
            "duration_ms": int(row.duration_ms),
            "sigma_c": float(row.sigma_c),
            "pow": float(row["pow"]),
        }
        for _, row in theta.iterrows()
    ]
    ladder = pd.read_csv(OUT_DIR / "shared_landscape_ladder.csv").set_index("model_key")
    joint = ladder.loc["gaussian_pow_attractor"]
    rows.append(
        {
            "fit": "Joint",
            "fit_type": "joint",
            "duration_ms": np.nan,
            "sigma_c": float(joint.shared_sigma_c),
            "pow": float(joint.shared_pow),
        }
    )
    return rows


def analyse_fitted_landscapes(
    grid_sizes: tuple[int, ...] = GRID_SIZES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    stationary_rows: list[dict] = []
    barrier_rows: list[dict] = []

    for fit in _fit_rows():
        sigma = float(fit["sigma_c"]) * np.power(T.WELLS, float(fit["pow"]))
        n_width = float(fit["sigma_c"]) ** (-1.0 / float(fit["pow"]))
        fit_results: dict[int, TopologyResult] = {}
        for grid_points in grid_sizes:
            result = compute_topological_resolution(
                T.WELLS,
                sigma,
                grid_points=grid_points,
                report_integer_max=T.N_TRUE_MAX,
            )
            fit_results[grid_points] = result
            minima = [p for p in result.stationary_points if p.kind == "minimum"]
            maxima = [p for p in result.stationary_points if p.kind == "maximum"]
            summary_rows.append(
                {
                    **fit,
                    "grid_points": grid_points,
                    "n_width": n_width,
                    "n_topo": result.n_topo,
                    "n_minima": len(minima),
                    "n_maxima": len(maxima),
                    "minimum_barrier_within_topo": min(
                        (barrier.height for barrier in result.barriers),
                        default=np.nan,
                    ),
                }
            )
            for order, point in enumerate(result.stationary_points, start=1):
                stationary_rows.append(
                    {
                        **fit,
                        "grid_points": grid_points,
                        "order": order,
                        "x": point.x,
                        "kind": point.kind,
                        "assigned_integer": point.assigned_integer,
                        "potential": point.potential,
                        "curvature": point.curvature,
                    }
                )
            for barrier in result.barriers:
                barrier_rows.append(
                    {
                        **fit,
                        "grid_points": grid_points,
                        "left_integer": barrier.left_integer,
                        "right_integer": barrier.right_integer,
                        "x": barrier.x,
                        "potential": barrier.potential,
                        "height": barrier.height,
                    }
                )

        reference = fit_results[max(grid_sizes)]
        reference_signature = tuple(
            (point.kind, point.assigned_integer) for point in reference.stationary_points
        )
        n_topo_robust = len({result.n_topo for result in fit_results.values()}) == 1
        classification_robust = all(
            tuple((point.kind, point.assigned_integer) for point in result.stationary_points)
            == reference_signature
            for result in fit_results.values()
        )
        max_root_shift = 0.0
        if classification_robust:
            reference_roots = np.array([point.x for point in reference.stationary_points])
            max_root_shift = max(
                float(
                    np.max(
                        np.abs(
                            np.array([point.x for point in result.stationary_points])
                            - reference_roots
                        )
                    )
                )
                for result in fit_results.values()
            )
        for row in summary_rows[-len(grid_sizes):]:
            row["n_topo_grid_robust"] = n_topo_robust
            row["classification_grid_robust"] = classification_robust
            row["max_root_shift"] = max_root_shift

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(stationary_rows),
        pd.DataFrame(barrier_rows),
    )


def main() -> None:
    summary, stationary, barriers = analyse_fitted_landscapes()
    summary.to_csv(SUMMARY_PATH, index=False)
    stationary.to_csv(STATIONARY_PATH, index=False)
    barriers.to_csv(BARRIER_PATH, index=False)
    finest = summary[summary.grid_points == max(GRID_SIZES)]
    print(finest[["fit", "n_width", "n_topo"]].to_string(index=False))
    print(
        "Grid robustness:",
        bool(summary.n_topo_grid_robust.all() and summary.classification_grid_robust.all()),
        f"(maximum refined-root shift {summary.max_root_shift.max():.2e})",
    )
    print(f"Saved {SUMMARY_PATH}, {STATIONARY_PATH}, and {BARRIER_PATH}")


if __name__ == "__main__":
    main()
