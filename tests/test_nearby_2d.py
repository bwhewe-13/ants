########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for two-dimensional nearby problems (nearby2d).
#
########################################################################

import numpy as np
import pytest

import ants
from ants import nearby2d
from ants.datatypes import GeometryData, SolverData
from tests import criticality_benchmarks as benchmarks
from tests import problems2d as prob

CELLS = 20
ANGLES = 2  # NN = ANGLES * ANGLES = 4 total angular directions


def _mms_problem():
    """manufactured_ss_01: 1x1 square, sigma_t=1, no scatter, uniform external source.

    _check_nearby2d_fixed_source asserts info.angular == True, so set it here.
    """
    mat_data, sources, geometry, quadrature, solver, *_ = prob.manufactured_ss_01(
        CELLS, ANGLES
    )
    solver.angular = True
    return mat_data, sources, geometry, quadrature, solver


@pytest.mark.smoke
@pytest.mark.nearby2d
@pytest.mark.slab2d
def test_fixed_source_angular_residual():
    """Angular-residual path: shapes are correct and residual is small."""
    mat_data, sources, geometry, quadrature, solver = _mms_problem()

    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        save_residual=False,
    )
    assert len(result) == 5
    (
        numerical_flux,
        curve_fit_flux,
        curve_fit_boundary_x,
        curve_fit_boundary_y,
        residual,
    ) = result

    NN = ANGLES * ANGLES
    assert numerical_flux.shape == (CELLS, CELLS, NN, 1)
    assert curve_fit_flux.shape == (CELLS, CELLS, 1)
    assert curve_fit_boundary_x.shape == (2, CELLS, NN, 1)
    assert curve_fit_boundary_y.shape == (2, CELLS, NN, 1)
    assert residual.shape == (CELLS, CELLS, NN, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.nearby2d
@pytest.mark.slab2d
def test_fixed_source_scalar_residual():
    """Scalar-residual path produces correct shapes and small residual."""
    mat_data, sources, geometry, quadrature, solver = _mms_problem()

    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        scalar_residual=True,
        save_residual=False,
    )
    assert len(result) == 5
    (
        numerical_flux,
        curve_fit_flux,
        curve_fit_boundary_x,
        curve_fit_boundary_y,
        residual,
    ) = result

    assert curve_fit_flux.shape == (CELLS, CELLS, 1)
    assert curve_fit_boundary_x.shape == (2, CELLS, 1)
    assert curve_fit_boundary_y.shape == (2, CELLS, 1)
    assert residual.shape == (CELLS, CELLS, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.smoke
@pytest.mark.nearby2d
@pytest.mark.slab2d
def test_fixed_source_return_nearby():
    """return_nearby=True runs the nearby problem and appends a scalar flux."""
    mat_data, sources, geometry, quadrature, solver = _mms_problem()

    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        return_nearby=True,
        save_residual=False,
    )
    assert len(result) == 6
    *_, nearby_flux = result

    assert nearby_flux.shape == (CELLS, CELLS, 1)
    assert np.all(nearby_flux >= 0.0)


@pytest.mark.nearby2d
@pytest.mark.slab2d
def test_fixed_source_precomputed_scalar_flux():
    """Pre-computed scalar flux is converted to angular via known_flux."""
    from dataclasses import replace

    from ants import fixed2d

    mat_data, sources, geometry, quadrature, solver = _mms_problem()

    scalar_solver = replace(solver, angular=False)
    scalar_flux = fixed2d.fixed_source(
        mat_data, sources, geometry, quadrature, scalar_solver
    )
    assert scalar_flux.ndim == 3

    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        numerical_flux=scalar_flux,
        save_residual=False,
    )
    assert len(result) == 5
    (
        numerical_flux,
        curve_fit_flux,
        curve_fit_boundary_x,
        curve_fit_boundary_y,
        residual,
    ) = result

    NN = ANGLES * ANGLES
    assert numerical_flux.shape == (CELLS, CELLS, NN, 1)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.nearby2d
@pytest.mark.slab2d
def test_fixed_source_precomputed_angular_flux():
    """Pre-computed angular flux is used directly without re-solving."""
    from ants import fixed2d

    mat_data, sources, geometry, quadrature, solver = _mms_problem()

    angular_flux = fixed2d.fixed_source(mat_data, sources, geometry, quadrature, solver)
    assert angular_flux.ndim == 4

    result = nearby2d.fixed_source(
        mat_data,
        sources,
        geometry,
        quadrature,
        solver,
        numerical_flux=angular_flux,
        save_residual=False,
    )
    assert len(result) == 5
    (
        provided_flux,
        curve_fit_flux,
        curve_fit_boundary_x,
        curve_fit_boundary_y,
        residual,
    ) = result

    assert np.array_equal(provided_flux, angular_flux)
    assert np.abs(residual).max() < 1e-3


@pytest.mark.nearby2d
@pytest.mark.slab2d
def test_k_criticality():
    """Nearby keff stays within 1e-3 of numerical keff for an infinite-y Pu slab."""
    cells_x = 50
    cells_y = 10
    length_x = 1.853722 * 2  # critical diameter (vacuum both sides)
    length_y = 2000.0  # effectively infinite
    bc_x = [0, 0]
    bc_y = [0, 0]
    angles = 4

    mat_data, _ = benchmarks.PUa_1_0(cells_x, bc_x)
    quadrature = ants.angular_xy(angles=angles, bc_x=bc_x)
    solver = SolverData()
    geometry = GeometryData(
        medium_map=np.zeros((cells_x, cells_y), dtype=np.int32),
        delta_x=np.repeat(length_x / cells_x, cells_x),
        delta_y=np.repeat(length_y / cells_y, cells_y),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )

    (
        numerical_scalar,
        numerical_keff,
        curve_fit_scalar,
        curve_fit_keff,
        nearby_scalar,
        nearby_keff,
        nearby_rate,
    ) = nearby2d.k_criticality(
        mat_data,
        geometry,
        quadrature,
        solver,
        save_residual=False,
    )

    assert numerical_scalar.shape == (cells_x, cells_y, 1)
    assert curve_fit_scalar.shape == (cells_x, cells_y, 1)
    assert nearby_scalar.shape == (cells_x, cells_y, 1)
    assert abs(nearby_keff - numerical_keff) < 1e-3
