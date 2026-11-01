########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test one-dimensional time-independent problems. Includes tests for
# Method of Manufactured Solutions for One Dimensional Slabs. Tests are
# for the time-independent solutions for scalar and angular flux, the
# diamond difference and step method, and for calculating at cell edges.
#
########################################################################
import os

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
from ants.fixed1d import fixed_source
from ants.utils import manufactured_1d as mms
from tests import problems1d

ANGULAR = [True, False]
SPATIAL = [1, 2, 3]
EDGES = [0, 1]
PARAMETERS = [
    (angular, edges, spatial)
    for angular in ANGULAR
    for edges in EDGES
    for spatial in SPATIAL
]


@pytest.mark.smoke
@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_01(angular, edges, spatial):
    mat_data, sources, geo, quadrature, solver = problems1d.manufactured_ss_01(400, 4)
    solver.angular = angular
    geo.space_disc = spatial
    solver.flux_at_edges = edges
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)

    edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_01(space_x, quadrature.angle_x)
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, :], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_02(angular, edges, spatial):
    mat_data, sources, geo, quadrature, solver = problems1d.manufactured_ss_02(400, 4)
    solver.angular = angular
    geo.space_disc = spatial
    solver.flux_at_edges = edges
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)

    edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_02(space_x, quadrature.angle_x)
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, :], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_03(angular, edges, spatial):
    mat_data, sources, geo, quadrature, solver = problems1d.manufactured_ss_03(400, 4)
    solver.angular = angular
    geo.space_disc = spatial
    solver.flux_at_edges = edges
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)

    edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_03(space_x, quadrature.angle_x)
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, :], axis=1)
    atol = 1e-5 if spatial == 2 else 5e-3
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_04(angular, edges, spatial):
    mat_data, sources, geo, quadrature, solver = problems1d.manufactured_ss_04(400, 4)
    solver.angular = angular
    geo.space_disc = spatial
    solver.flux_at_edges = edges
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)

    edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_04(space_x, quadrature.angle_x)
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, :], axis=1)
    atol = 1e-4 if spatial == 2 else 1e-2
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize(("angular", "edges", "spatial"), PARAMETERS)
def test_manufactured_05(angular, edges, spatial):
    mat_data, sources, geo, quadrature, solver = problems1d.manufactured_ss_05(400, 4)
    solver.angular = angular
    geo.space_disc = spatial
    solver.flux_at_edges = edges
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)

    edges_x = np.concatenate(([0], np.cumsum(geo.delta_x)))
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
    space_x = edges_x.copy() if edges else centers_x.copy()
    exact = mms.solution_ss_05(space_x, quadrature.angle_x)
    if not angular:
        exact = np.sum(exact * quadrature.angle_w[None, :], axis=1)
    atol = 1e-4 if spatial == 2 else 2e-2
    assert np.isclose(flux[(..., 0)], exact, atol=atol).all(), "Incorrect flux"


@pytest.mark.sphere1d
@pytest.mark.source_iteration
@pytest.mark.multigroup1d
def test_sphere_01_source_iteration():
    mat_data, sources, geometry, quadrature, solver, _ = problems1d.sphere_01("fixed")

    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
    path = os.path.join(problems1d.PATH, "uranium_sphere_source_iteration_flux.npy")
    reference = np.load(path)
    assert np.isclose(flux, reference).all()


@pytest.mark.slab1d
@pytest.mark.source_iteration
def test_dynamic_mode_decomposition():
    # Strong group to group scattering so the DMD estimate is needed
    groups = 4
    cells_x = 40
    scatter = 0.99 / 3 * (np.ones((groups, groups)) - np.eye(groups))
    mat_data = MaterialData(
        total=np.ones((1, groups)),
        scatter=scatter[None],
        fission=np.zeros((1, groups, groups)),
    )
    sources = SourceData(
        external=np.ones((cells_x, 1, groups)),
        boundary_x=np.zeros((2, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells_x,), dtype=np.int32),
        delta_x=np.repeat(1.0 / cells_x, cells_x),
        geometry=Geometry.SLAB1D,
    )
    quadrature = ants.angular_x(8)
    solver = SolverData(tol_energy=1e-14, max_iter_energy=5000)
    reference = fixed_source(mat_data, sources, geometry, quadrature, solver)

    # Same number of iterations as DMD (dmd_rank + dmd_snapshots)
    solver = SolverData(tol_energy=1e-14, max_iter_energy=22)
    flux_si = fixed_source(mat_data, sources, geometry, quadrature, solver)
    solver = SolverData(tol_energy=1e-14, mg_solver=MultigroupSolver.DMD)
    flux_dmd = fixed_source(mat_data, sources, geometry, quadrature, solver)

    error_si = np.max(np.fabs(flux_si - reference)) / np.max(reference)
    error_dmd = np.max(np.fabs(flux_dmd - reference)) / np.max(reference)
    assert error_dmd < 1e-10
    assert error_dmd < 1e-3 * error_si
