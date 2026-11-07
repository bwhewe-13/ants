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
from ants.datatypes import GeometryData, MaterialData, SolverData, SourceData
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
def test_step_characteristic_void():
    # Scattering source square surrounded by a void, solved with the
    # step characteristic and diamond difference methods
    cells = 40
    length = 4.0
    centers = (np.arange(cells) + 0.5) * length / cells
    inside = np.abs(centers - 0.5 * length) < 1.0
    medium_map = np.where(inside[:, None] & inside[None, :], 0, 1).astype(np.int32)
    mat_data = MaterialData(
        total=np.array([[1.0], [0.0]]),
        scatter=np.array([[[0.5]], [[0.0]]]),
        fission=np.zeros((2, 1, 1)),
    )
    external = np.zeros((cells, cells, 1, 1))
    external[medium_map == 0] = 1.0
    sources = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    quadrature = ants.angular_xy(4)

    flux = {}
    for spatial in (2, 3):
        geometry = GeometryData(
            medium_map=medium_map,
            delta_x=np.repeat(length / cells, cells),
            delta_y=np.repeat(length / cells, cells),
            geometry=3,
            space_disc=spatial,
        )
        flux[spatial] = fixed_source(
            mat_data, sources, geometry, quadrature, SolverData()
        )
    assert np.isfinite(flux[3]).all(), "Step characteristic flux is not finite"
    assert np.allclose(flux[3], flux[2], atol=1e-2 * flux[2].max())


@pytest.mark.slab2d
@pytest.mark.parametrize("shape", [(2, 1, 8, 1), (2, 11, 1, 1)])
def test_boundary_wrong_shape(shape):
    # 10 x 10 cells with 4**2 angles and 1 group
    mat_data, sources, geometry, quadrature, solver, _, _ = (
        problems2d.manufactured_ss_01(10, 4)
    )
    sources.boundary_x = np.zeros(shape)
    with pytest.raises(ValueError, match="boundary_x has shape"):
        fixed_source(mat_data, sources, geometry, quadrature, solver)


def _reflected_problem(bc_x, bc_y):
    # Scattering corner source with reflecting faces
    cells = 8
    external = np.zeros((cells, cells, 1, 1))
    external[:3, :3] = 1.0
    mat_data = MaterialData(
        total=np.array([[1.0]]),
        scatter=np.array([[[0.5]]]),
        fission=np.zeros((1, 1, 1)),
    )
    sources = SourceData(
        external=external,
        boundary_x=np.zeros((2, 1, 1, 1)),
        boundary_y=np.zeros((2, 1, 1, 1)),
    )
    geometry = GeometryData(
        medium_map=np.zeros((cells, cells), dtype=np.int32),
        delta_x=np.full(cells, 0.5),
        delta_y=np.full(cells, 0.5),
        bc_x=bc_x,
        bc_y=bc_y,
        geometry=3,
    )
    return mat_data, sources, geometry


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.parametrize("bc_x,bc_y", [([1, 0], [0, 0]), ([0, 1], [1, 0])])
def test_angular_reflected(bc_x, bc_y):
    # The angular flux collapses to the scalar flux with the angles ordered
    # for the reflecting faces
    mat_data, sources, geometry = _reflected_problem(bc_x, bc_y)
    quadrature = ants.angular_xy(4, bc_x=bc_x, bc_y=bc_y)
    flux = fixed_source(mat_data, sources, geometry, quadrature, SolverData())
    angular_flux = fixed_source(
        mat_data, sources, geometry, quadrature, SolverData(angular=True)
    )
    scalar_flux = np.einsum("ijnq,n->ijq", angular_flux, quadrature.angle_w)
    assert np.allclose(scalar_flux, flux, rtol=1e-8)


@pytest.mark.smoke
@pytest.mark.slab2d
@pytest.mark.parametrize(
    "solver",
    [SolverData(angular=True), SolverData(flux_at_edges=1), SolverData(sigma_as=0.1)],
)
def test_quadrature_not_ordered(solver):
    mat_data, sources, geometry = _reflected_problem([1, 0], [0, 0])
    with pytest.raises(ValueError, match="not ordered for bc_x"):
        fixed_source(mat_data, sources, geometry, ants.angular_xy(4), solver)
