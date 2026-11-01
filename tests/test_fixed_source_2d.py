########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test two-dimensional time-independent problems. Includes method of
# manufactured solutions for two-dimensional slabs.
#
########################################################################

import numpy as np
import pytest

import ants
from ants.datatypes import (
    Geometry,
    GeometryData,
    MaterialData,
    MultigroupSolver,
    SolverData,
    SourceData,
)
from ants.fixed2d import fixed_source
from ants.utils import manufactured_2d as mms
from tests import problems2d

ANGULAR = [True, False]
SPATIAL = [1, 2, 3]
PARAMETERS = [(angular, spatial) for angular in ANGULAR for spatial in SPATIAL]


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_01(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_01(200, 2)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_01(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_02(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_02(200, 2)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_02(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_03(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_03(200, 4)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_03(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "spatial"), PARAMETERS)
def test_manufactured_04(angular, spatial):
    mat_data, sources, geometry, quadrature, solver, edges_x, edges_y = (
        problems2d.manufactured_ss_04(200, 4)
    )
    solver.angular = angular
    geometry.space_disc = spatial
    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)

    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
    exact = mms.solution_ss_04(
        centers_x, centers_y, quadrature.angle_x, quadrature.angle_y
    )
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, None, :, None], axis=2)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(
        flux[(..., 0)], exact[(..., 0)], atol=atol
    ).all(), "Incorrect flux"


@pytest.mark.slab2d
@pytest.mark.source_iteration
def test_dynamic_mode_decomposition():
    # Strong group to group scattering in an optically thick domain so
    # the DMD estimate is needed
    groups = 4
    cells = 12
    scatter = 0.99 / 3 * (np.ones((groups, groups)) - np.eye(groups))
    mat_data = MaterialData(
        total=np.ones((1, groups)),
        scatter=scatter[None],
        fission=np.zeros((1, groups, groups)),
    )
    sources = SourceData(
        external=np.ones((cells, cells, 1, groups)),
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=np.repeat(20.0 / cells, cells),
        delta_y=np.repeat(20.0 / cells, cells),
        geometry=Geometry.SLAB2D,
    )
    quadrature = ants.angular_xy(4)
    solver = SolverData(tol_energy=1e-14, max_iter_energy=5000)
    reference = fixed_source(mat_data, sources, geometry, quadrature, solver)

    # Same number of iterations as DMD (dmd_rank + dmd_snapshots)
    solver = SolverData(tol_energy=1e-14, max_iter_energy=22)
    flux_si = fixed_source(mat_data, sources, geometry, quadrature, solver)
    solver = SolverData(tol_energy=1e-14, mg_solver=MultigroupSolver.DMD)
    flux_dmd = fixed_source(mat_data, sources, geometry, quadrature, solver)

    error_si = np.max(np.fabs(flux_si - reference)) / np.max(reference)
    error_dmd = np.max(np.fabs(flux_dmd - reference)) / np.max(reference)
    assert error_dmd < 1e-5
    assert error_dmd < 1e-4 * error_si
