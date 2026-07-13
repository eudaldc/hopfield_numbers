"""Tests for the stationary-point definition of topological resolution."""

from __future__ import annotations

import numpy as np

from scripts import topological_resolution as topo


def test_analytic_potential_derivatives_match_finite_differences() -> None:
    wells = np.arange(1.0, 7.0)
    sigma = 0.08 * wells**1.4
    x = 3.37
    step = 1e-5

    value, gradient, curvature = topo.potential_derivatives(x, wells, sigma)
    value_left = topo.potential_derivatives(x - step, wells, sigma)[0]
    value_right = topo.potential_derivatives(x + step, wells, sigma)[0]
    gradient_left = topo.potential_derivatives(x - step, wells, sigma)[1]
    gradient_right = topo.potential_derivatives(x + step, wells, sigma)[1]

    assert np.isfinite([value, gradient, curvature]).all()
    assert np.isclose(gradient, (value_right - value_left) / (2.0 * step), rtol=1e-6)
    assert np.isclose(curvature, (gradient_right - gradient_left) / (2.0 * step), rtol=1e-6)


def test_separated_equal_width_wells_each_retain_a_basin() -> None:
    wells = np.arange(1.0, 6.0)
    result = topo.compute_topological_resolution(
        wells,
        sigma=np.full(wells.shape, 0.12),
        grid_points=10_000,
        report_integer_max=5,
    )

    assert result.n_topo == 5
    assert len([point for point in result.stationary_points if point.kind == "minimum"]) == 5
    assert len(result.barriers) == 4
    assert all(barrier.height > 0.0 for barrier in result.barriers)


def test_merged_equal_width_wells_have_no_consecutive_basin_from_one() -> None:
    wells = np.arange(1.0, 6.0)
    result = topo.compute_topological_resolution(
        wells,
        sigma=np.full(wells.shape, 2.0),
        grid_points=10_000,
        report_integer_max=5,
    )

    assert result.n_topo == 0
    assert not result.barriers


def test_minimum_assignment_is_nearest_integer_and_injective() -> None:
    points = (
        topo.StationaryPoint(1.05, -2.0, 3.0, "minimum"),
        topo.StationaryPoint(2.10, -1.8, 2.0, "minimum"),
        topo.StationaryPoint(2.35, -1.9, 1.0, "minimum"),
        topo.StationaryPoint(2.70, -1.7, 1.0, "maximum"),
    )
    assigned = topo.assign_minima_to_integers(points, integer_min=1, integer_max=4)
    owners = {point.assigned_integer: point.x for point in assigned if point.assigned_integer}

    assert owners == {1: 1.05, 2: 2.10}


def test_fitted_like_topology_is_grid_invariant() -> None:
    wells = np.arange(1.0, 20.0)
    sigma = 0.022219420407995807 * wells**1.7969228904689682
    coarse = topo.compute_topological_resolution(wells, sigma, 5_000, 15)
    fine = topo.compute_topological_resolution(wells, sigma, 20_000, 15)

    coarse_signature = [
        (point.kind, point.assigned_integer) for point in coarse.stationary_points
    ]
    fine_signature = [
        (point.kind, point.assigned_integer) for point in fine.stationary_points
    ]
    assert coarse.n_topo == fine.n_topo == 6
    assert coarse_signature == fine_signature
