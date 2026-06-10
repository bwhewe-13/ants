########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for one-dimensional nearby problems (nearby1d).
#
########################################################################

import numpy as np
import pytest

import ants
from ants import nearby1d
from ants.datatypes import SolverData
from tests import criticality_benchmarks as benchmarks
from tests import problems1d as prob

CELLS_X = 100
ANGLES = 4


def _mms_problem():
    """manufactured_ss_03: exact solution is 0.5 + 0.25*x^2*exp(mu).

    Quintic splines reproduce degree-2 polynomials almost exactly, so
    curve-fit residuals are dominated by the numerical solver's O(h^2)
    truncation error rather than interpolation error.
    """
    mat_data, sources, geometry, quadrature, solver = prob.manufactured_ss_03(
        CELLS_X, ANGLES
    )
    solver.angular = True
    return mat_data, sources, geometry, quadrature, solver


def _edges_and_centers(geometry):
    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    return edges_x, centers_x


@pytest.mark.smoke
@pytest.mark.nearby1d
def test_fixed_source_center_knots():
    """Center-knot path: shapes are correct and residual is small."""
    mat_data, sources, geometry, quadrature, solver = _mms_problem()
    _, centers_x = _edges_and_centers(geometry)

    result = nearby1d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        centers_x,
        save_residual=False,
    )
    assert len(result) == 4
    numerical_flux, curve_fit_flux, curve_fit_boundary_x, residual = result

    assert numerical_flux.shape == (CELLS_X, ANGLES, 1)
    assert curve_fit_flux.shape == (CELLS_X, 1)
    assert curve_fit_boundary_x.shape == (2, ANGLES, 1)
    assert residual.shape == (CELLS_X, ANGLES, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.smoke
@pytest.mark.nearby1d
def test_fixed_source_edge_knots():
    """Edge-knot path with pre-computed edge flux exercises center_knots=False
    branch."""
    from dataclasses import replace

    from ants import fixed1d

    mat_data, sources, geometry, quadrature, solver = _mms_problem()
    edges_x, _ = _edges_and_centers(geometry)

    # Edge-knot interpolation requires flux at cell edges (cells_x+1 points)
    edge_solver = replace(solver, flux_at_edges=1)
    numerical_flux = fixed1d.fixed_source(
        mat_data, sources, geometry, quadrature, edge_solver
    )

    result = nearby1d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        edges_x,
        numerical_flux=numerical_flux,
        save_residual=False,
    )
    assert len(result) == 4
    provided_flux, curve_fit_flux, curve_fit_boundary_x, residual = result

    assert provided_flux.shape == (CELLS_X + 1, ANGLES, 1)
    assert curve_fit_flux.shape == (CELLS_X, 1)
    assert curve_fit_boundary_x.shape == (2, ANGLES, 1)
    assert residual.shape == (CELLS_X, ANGLES, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.nearby1d
def test_fixed_source_scalar_residual():
    """Scalar-residual path produces correct shapes and small residual."""
    mat_data, sources, geometry, quadrature, solver = _mms_problem()
    _, centers_x = _edges_and_centers(geometry)

    result = nearby1d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        centers_x,
        scalar_residual=True,
        save_residual=False,
    )
    assert len(result) == 4
    _, curve_fit_flux, curve_fit_boundary_x, residual = result

    assert curve_fit_flux.shape == (CELLS_X, 1)
    assert curve_fit_boundary_x.shape == (2, 1)
    assert residual.shape == (CELLS_X, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.smoke
@pytest.mark.nearby1d
def test_fixed_source_return_nearby():
    """return_residual=True runs the nearby problem and returns a scalar flux."""
    mat_data, sources, geometry, quadrature, solver = _mms_problem()
    _, centers_x = _edges_and_centers(geometry)

    result = nearby1d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        centers_x,
        return_residual=True,
        save_residual=False,
    )
    assert len(result) == 5
    numerical_flux, curve_fit_flux, curve_fit_boundary_x, residual, nearby_flux = result

    assert nearby_flux.shape == (CELLS_X, 1)
    assert np.all(nearby_flux >= 0.0)
    # Exact scalar flux: phi_s(x) = sum_n w_n * (0.5 + 0.25*x^2*exp(mu_n))
    angle_x = np.asarray(quadrature.angle_x)
    angle_w = np.asarray(quadrature.angle_w)
    exact = (0.5 + 0.25 * centers_x**2 * np.sum(angle_w * np.exp(angle_x)))[:, None]
    assert np.allclose(nearby_flux, exact, atol=1e-3)


@pytest.mark.nearby1d
def test_fixed_source_precomputed_scalar_flux():
    """Pre-computed scalar flux (ndim==2) is converted to angular via known_flux."""
    from dataclasses import replace

    from ants import fixed1d

    mat_data, sources, geometry, quadrature, solver = _mms_problem()
    _, centers_x = _edges_and_centers(geometry)

    scalar_solver = replace(solver, angular=False)
    scalar_flux = fixed1d.fixed_source(
        mat_data, sources, geometry, quadrature, scalar_solver
    )
    assert scalar_flux.ndim == 2

    result = nearby1d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        centers_x,
        numerical_flux=scalar_flux,
        save_residual=False,
    )
    assert len(result) == 4
    numerical_flux, curve_fit_flux, curve_fit_boundary_x, residual = result
    assert numerical_flux.shape == (CELLS_X, ANGLES, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.nearby1d
def test_k_criticality():
    """Nearby keff stays within 1e-3 of numerical keff for a critical Pu slab."""
    cells_x = 50
    angles = 4
    materials, geometry = benchmarks.PUb_1_0(cells_x, bc_x=[0, 0], geometry_type=1)
    quadrature = ants.angular_x(angles, bc_x=[0, 0])
    solver = SolverData()

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    (
        numerical_scalar,
        numerical_keff,
        curve_fit_scalar,
        curve_fit_keff,
        nearby_scalar,
        nearby_keff,
    ) = nearby1d.k_criticality(
        materials,
        geometry,
        quadrature,
        solver,
        centers_x,
        save_residual=False,
    )

    assert numerical_scalar.shape == (cells_x, 1)
    assert curve_fit_scalar.shape == (cells_x, 1)
    assert nearby_scalar.shape == (cells_x, 1)
    assert abs(nearby_keff - numerical_keff) < 1e-3


@pytest.mark.nearby1d
def test_k_criticality_precomputed_scalar_flux():
    """Pre-computed scalar flux + keff triggers known_flux path in k_criticality."""
    from ants import critical1d

    cells_x = 50
    angles = 4
    materials, geometry = benchmarks.PUb_1_0(cells_x, bc_x=[0, 0], geometry_type=1)
    quadrature = ants.angular_x(angles, bc_x=[0, 0])
    solver = SolverData()

    edges_x = np.concatenate(([0.0], np.cumsum(geometry.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    scalar_flux, numerical_keff = critical1d.k_criticality(
        materials, geometry, quadrature, solver
    )
    assert scalar_flux.ndim == 2

    _, _, _, _, nearby_scalar, nearby_keff = nearby1d.k_criticality(
        materials,
        geometry,
        quadrature,
        solver,
        centers_x,
        numerical_flux=scalar_flux,
        numerical_keff=numerical_keff,
        save_residual=False,
    )
    assert nearby_scalar.shape == (cells_x, 1)
    assert abs(nearby_keff - numerical_keff) < 1e-3
