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
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
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


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize("edges", EDGES)
def test_step_characteristic_void_exact(edges):
    # In a void the angular flux is constant along each direction, so the
    # step characteristic (diamond weights at sigma_t = 0) is exact
    cells_x = 20
    mat_data = MaterialData(
        total=np.array([[0.0]]),
        scatter=np.array([[[0.0]]]),
        fission=np.array([[[0.0]]]),
    )
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[0] = 1.0
    sources = SourceData(external=np.zeros((cells_x, 1, 1)), boundary_x=boundary_x)
    geo = GeometryData(
        medium_map=np.zeros(cells_x, dtype=np.int32),
        delta_x=np.repeat(0.1, cells_x),
        bc_x=[0, 0],
        geometry=1,
        space_disc=3,
    )
    quadrature = ants.angular_x(4)
    solver = SolverData(angular=True, flux_at_edges=edges)
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)

    exact = np.broadcast_to(quadrature.angle_x > 0, flux.shape[:2]).astype(float)
    assert np.allclose(flux[..., 0], exact), "Incorrect flux in a void"


@pytest.mark.slab1d
@pytest.mark.source_iteration
@pytest.mark.parametrize("edges", EDGES)
def test_step_characteristic_reeds(edges):
    mat_data, sources, geo, quadrature, solver = problems1d.reeds([0, 0])
    solver.flux_at_edges = edges
    geo.space_disc = 3
    flux = fixed_source(mat_data, sources, geo, quadrature, solver)
    assert np.isfinite(flux).all(), "Step characteristic flux is not finite"


@pytest.mark.slab1d
@pytest.mark.parametrize("shape", [(2, 2, 1), (2, 4, 3), (1, 1, 1), (2, 1)])
def test_boundary_wrong_shape(shape):
    # Reed's problem has 4 angles and 1 group
    mat_data, sources, geo, quadrature, solver = problems1d.reeds([0, 0])
    sources.boundary_x = np.zeros(shape)
    with pytest.raises(ValueError, match="boundary_x has shape"):
        fixed_source(mat_data, sources, geo, quadrature, solver)


@pytest.mark.sphere1d
@pytest.mark.source_iteration
@pytest.mark.multigroup1d
def test_sphere_01_source_iteration():
    mat_data, sources, geometry, quadrature, solver, _ = problems1d.sphere_01("fixed")

    flux = fixed_source(mat_data, sources, geometry, quadrature, solver)
    path = os.path.join(problems1d.PATH, "uranium_sphere_source_iteration_flux.npy")
    reference = np.load(path)
    assert np.isclose(flux, reference).all()
